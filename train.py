#!/usr/bin/python
import os
import json
from collections import defaultdict
import random
import torch
from torch.utils.data import DataLoader

from scene_generation.args import get_args
from scene_generation.data.coco import CocoSceneGraphDataset, coco_collate_fn
from scene_generation.data.coco_panoptic import CocoPanopticSceneGraphDataset, coco_panoptic_collate_fn
from scene_generation.metrics import jaccard
from scene_generation.trainer import Trainer

from scripts.inception_score import InceptionScore


def build_coco_dsets(args):
    dset_kwargs = {
        'image_dir': args.coco_train_image_dir,
        'instances_json': args.coco_train_instances_json,
        'stuff_json': args.coco_train_stuff_json,
        'image_size': args.image_size,
        'mask_size': args.mask_size,
        'max_samples': args.num_train_samples,
        'min_object_size': args.min_object_size,
        'min_objects_per_image': args.min_objects_per_image,
        'instance_whitelist': args.instance_whitelist,
        'stuff_whitelist': args.stuff_whitelist,
        'include_other': args.coco_include_other,
    }
    if args.is_panoptic:
        dset_kwargs['panoptic'] = args.coco_panoptic_train
        dset_kwargs['panoptic_segmentation'] = args.coco_panoptic_segmentation_train
        train_dset = CocoPanopticSceneGraphDataset(**dset_kwargs)
    else:
        train_dset = CocoSceneGraphDataset(**dset_kwargs)
    num_objs = train_dset.total_objects()
    num_imgs = len(train_dset)
    print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
    print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

    dset_kwargs['image_dir'] = args.coco_val_image_dir
    dset_kwargs['instances_json'] = args.coco_val_instances_json
    dset_kwargs['stuff_json'] = args.coco_val_stuff_json
    dset_kwargs['max_samples'] = args.num_val_samples
    if args.is_panoptic:
        dset_kwargs['panoptic'] = args.coco_panoptic_val
        dset_kwargs['panoptic_segmentation'] = args.coco_panoptic_segmentation_val
        val_dset = CocoPanopticSceneGraphDataset(**dset_kwargs)
    else:
        val_dset = CocoSceneGraphDataset(**dset_kwargs)

    assert train_dset.vocab == val_dset.vocab
    vocab = json.loads(json.dumps(train_dset.vocab))

    return vocab, train_dset, val_dset


def build_loaders(args):
    vocab, train_dset, val_dset = build_coco_dsets(args)
    if args.is_panoptic:
        collate_fn = coco_panoptic_collate_fn
    else:
        collate_fn = coco_collate_fn

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': True,
        'collate_fn': collate_fn,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = args.shuffle_val
    val_loader = DataLoader(val_dset, **loader_kwargs)
    return vocab, train_loader, val_loader


def check_model(args, loader, model, inception_score, use_gt):
    fid = None
    num_samples = 0
    total_iou = 0
    total_boxes = 0
    inception_score.clean()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, attributes = batch

            # Run the model as it has been run during training
            if use_gt:
                model_out = model(imgs, objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks, attributes=attributes,
                                  test_mode=True, use_gt_box=True)
            else:
                attributes = torch.zeros_like(attributes)
                model_out = model(imgs, objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=None, attributes=attributes,
                                  test_mode=True, use_gt_box=False)
            imgs_pred, boxes_pred, masks_pred, _, pred_layout, _ = model_out

            iou, _, _ = jaccard(boxes_pred, boxes)
            total_iou += iou
            total_boxes += boxes_pred.size(0)
            inception_score(imgs_pred)

            num_samples += imgs.size(0)
            if num_samples >= args.num_val_samples:
                break

        inception_mean, inception_std = inception_score.compute_score(splits=5)

        avg_iou = total_iou / total_boxes

    out = [avg_iou, inception_mean, inception_std, fid]

    return tuple(out)


def get_checkpoint(args, vocab):
    if args.restore_from_checkpoint:
        restore_path = '%s_with_model.pt' % args.checkpoint_name
        restore_path = os.path.join(args.output_dir, restore_path)
        assert restore_path is not None
        assert os.path.isfile(restore_path)
        print('Restoring from checkpoint:')
        print(restore_path)
        checkpoint = torch.load(restore_path)
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
    else:
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'vocab': vocab,
            'model_kwargs': {},
            'd_obj_kwargs': {},
            'd_mask_kwargs': {},
            'd_img_kwargs': {},
            'd_global_mask_kwargs': {},
            'losses_ts': [],
            'losses': defaultdict(list),
            'd_losses': defaultdict(list),
            'checkpoint_ts': [],
            'train_inception': [],
            'val_losses': defaultdict(list),
            'val_inception': [],
            'norm_d': [],
            'norm_g': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'model_state': None, 'model_best_state': None,
            'optim_state': None, 'optim_best_state': None,
            'd_obj_state': None, 'd_obj_best_state': None,
            'd_obj_optim_state': None, 'd_obj_optim_best_state': None,
            'd_img_state': None, 'd_img_best_state': None,
            'd_img_optim_state': None, 'd_img_optim_best_state': None,
            'd_mask_state': None, 'd_mask_best_state': None,
            'd_mask_optim_state': None, 'd_mask_optim_best_state': None,
            'best_t': [],
        }
    return t, epoch, checkpoint


def main(args):
    print(args)
    vocab, train_loader, val_loader = build_loaders(args)
    t, epoch, checkpoint = get_checkpoint(args, vocab)
    trainer = Trainer(args, vocab, checkpoint)
    if args.restore_from_checkpoint:
        trainer.restore_checkpoint(checkpoint)
    else:
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as outfile:
            json.dump(vars(args), outfile)

    inception_score = InceptionScore(cuda=True, batch_size=args.batch_size, resize=True)
    train_results = check_model(args, val_loader, trainer.model, inception_score, use_gt=True)
    t_avg_iou, t_inception_mean, t_inception_std, _ = train_results
    index = int(t / args.print_every)
    trainer.writer.add_scalar('checkpoint/{}'.format('train_iou'), t_avg_iou, index)
    trainer.writer.add_scalar('checkpoint/{}'.format('train_inception_mean'), t_inception_mean, index)
    trainer.writer.add_scalar('checkpoint/{}'.format('train_inception_std'), t_inception_std, index)
    print(t_avg_iou, t_inception_mean, t_inception_std)

    while t < args.num_iterations:
        epoch += 1
        print('Starting epoch %d' % epoch)

        for batch in train_loader:
            t += 1
            batch = [tensor.cuda() for tensor in batch]
            imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, attributes = batch

            use_gt = random.randint(0, 1) != 0
            if not use_gt:
                attributes = torch.zeros_like(attributes)
            model_out = trainer.model(imgs, objs, triples, obj_to_img,
                                      boxes_gt=boxes, masks_gt=masks, attributes=attributes)
            imgs_pred, boxes_pred, masks_pred, layout, layout_pred, layout_wrong = model_out

            layout_one_hot = layout[:, :trainer.num_obj, :, :]
            layout_pred_one_hot = layout_pred[:, :trainer.num_obj, :, :]

            trainer.train_generator(imgs, imgs_pred, masks, masks_pred, layout,
                                    objs, boxes, boxes_pred, obj_to_img, use_gt)

            imgs_pred_detach = imgs_pred.detach()
            masks_pred_detach = masks_pred.detach()
            boxes_pred_detach = boxes.detach()
            layout_detach = layout.detach()
            layout_wrong_detach = layout_wrong.detach()
            trainer.train_mask_discriminator(masks, masks_pred_detach, objs)
            trainer.train_obj_discriminator(imgs, imgs_pred_detach, objs, boxes, boxes_pred_detach, obj_to_img)
            trainer.train_image_discriminator(imgs, imgs_pred_detach, layout_detach, layout_wrong_detach)

            if t % args.print_every == 0 or t == 1:
                trainer.write_losses(checkpoint, t)
                trainer.write_images(t, imgs, imgs_pred, layout_one_hot, layout_pred_one_hot)

            if t % args.checkpoint_every == 0:
                print('begin check model train')
                train_results = check_model(args, val_loader, trainer.model, inception_score, use_gt=True)
                print('begin check model val')
                val_results = check_model(args, val_loader, trainer.model, inception_score, use_gt=False)
                trainer.save_checkpoint(checkpoint, t, args, epoch, train_results, val_results)


if __name__ == '__main__':
    args = get_args()
    main(args)
