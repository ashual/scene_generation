import argparse
import os
from random import randint

import numpy as np
import torch
from scipy.misc import imsave
from torch.utils.data import DataLoader

from scene_generation.data import imagenet_deprocess_batch
from scene_generation.data.coco import CocoSceneGraphDataset, coco_collate_fn
from scene_generation.data.coco_panoptic import CocoPanopticSceneGraphDataset, coco_panoptic_collate_fn
from scene_generation.data.utils import split_graph_batch
from scene_generation.metrics import jaccard
from scene_generation.model import Model
from scene_generation.utils import int_tuple, bool_flag
from scene_generation.bilinear import crop_bbox_batch
from scripts.train_accuracy_net import all_pretrained_models


def load_model(model_path):
    model_name = 'resnet101'
    model = all_pretrained_models(172, name=model_name).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--model_mode', default='eval', choices=['train', 'eval'])
parser.add_argument('--accuracy_model_path', default=None)

# Shared dataset options
parser.add_argument('--dataset', default='coco')
parser.add_argument('--image_size', default=(128, 128), type=int_tuple)
parser.add_argument('--batch_size', default=24, type=int)
parser.add_argument('--shuffle', default=False, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--num_samples', default=10000, type=int)
parser.add_argument('--save_gt_imgs', default=False, type=bool_flag)
parser.add_argument('--save_graphs', default=False, type=bool_flag)
parser.add_argument('--use_gt_boxes', default=False, type=bool_flag)
parser.add_argument('--use_gt_masks', default=False, type=bool_flag)
parser.add_argument('--use_gt_attr', default=False, type=bool_flag)
parser.add_argument('--use_gt_textures', default=False, type=bool_flag)
parser.add_argument('--save_layout', default=False, type=bool_flag)
parser.add_argument('--sample_attributes', default=False, type=bool_flag)
parser.add_argument('--object_size', default=64, type=int)
parser.add_argument('--grid_size', default=25, type=int)

parser.add_argument('--output_dir', default='output')

COCO_DIR = os.path.expanduser('../data/coco')
parser.add_argument('--coco_image_dir',
                    default=os.path.join(COCO_DIR, 'images/val2017'))
parser.add_argument('--instances_json',
                    default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
parser.add_argument('--stuff_json',
                    default=os.path.join(COCO_DIR, 'annotations/stuff_val2017.json'))


def build_coco_dset(args, checkpoint):
    checkpoint_args = checkpoint['args']
    print('include other: ', checkpoint_args.get('coco_include_other'))
    # When using GT masks, using the
    mask_size = args.image_size[0] if args.use_gt_masks else checkpoint_args['mask_size']
    dset_kwargs = {
        'image_dir': args.coco_image_dir,
        'instances_json': args.instances_json,
        'stuff_json': args.stuff_json,
        'image_size': args.image_size,
        'mask_size': mask_size,
        'max_samples': args.num_samples,
        'min_object_size': checkpoint_args['min_object_size'],
        'min_objects_per_image': checkpoint_args['min_objects_per_image'],
        'instance_whitelist': checkpoint_args['instance_whitelist'],
        'stuff_whitelist': checkpoint_args['stuff_whitelist'],
        'include_other': checkpoint_args.get('coco_include_other', True),
        'test_part': True,
        'sample_attributes': args.sample_attributes,
        'grid_size': args.grid_size
    }
    dset = CocoSceneGraphDataset(**dset_kwargs)
    return dset


def build_coco_panoptic_dset(args, checkpoint):
    checkpoint_args = checkpoint['args']
    print('include other: ', checkpoint_args.get('coco_include_other'))
    # When using GT masks, using the
    mask_size = args.image_size[0] if args.use_gt_masks else checkpoint_args['mask_size']
    dset_kwargs = {
        'image_dir': args.coco_image_dir,
        'instances_json': args.instances_json,
        'panoptic': checkpoint_args['coco_panoptic_val'],
        'panoptic_segmentation': checkpoint_args['coco_panoptic_segmentation_val'],
        'stuff_json': args.stuff_json,
        'image_size': args.image_size,
        'mask_size': mask_size,
        'max_samples': args.num_samples,
        'min_object_size': checkpoint_args['min_object_size'],
        'min_objects_per_image': checkpoint_args['min_objects_per_image'],
        'instance_whitelist': checkpoint_args['instance_whitelist'],
        'stuff_whitelist': checkpoint_args['stuff_whitelist'],
        'include_other': checkpoint_args.get('coco_include_other', True),
        'test_part': True,
        'sample_attributes': args.sample_attributes,
        'grid_size': args.grid_size
    }
    dset = CocoPanopticSceneGraphDataset(**dset_kwargs)
    return dset


def build_loader(args, checkpoint, is_panoptic):
    if is_panoptic:
        dset = build_coco_panoptic_dset(args, checkpoint)
        collate_fn = coco_panoptic_collate_fn
    else:
        dset = build_coco_dset(args, checkpoint)
        collate_fn = coco_collate_fn

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': args.shuffle,
        'collate_fn': collate_fn,
    }
    loader = DataLoader(dset, **loader_kwargs)
    return loader


def build_model(args, checkpoint):
    kwargs = checkpoint['model_kwargs']
    model = Model(**kwargs)
    model_state = checkpoint['model_state']
    model.load_state_dict(model_state)
    if args.model_mode == 'eval':
        model.eval()
    elif args.model_mode == 'train':
        model.train()
    model.image_size = args.image_size
    model.cuda()
    return model


def makedir(base, name, flag=True):
    dir_name = None
    if flag:
        dir_name = os.path.join(base, name)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
    return dir_name


def one_hot_to_rgb(layout_pred, colors, num_objs):
    one_hot = layout_pred[:, :num_objs, :, :]
    one_hot_3d = torch.einsum('abcd,be->aecd', [one_hot.cpu(), colors])
    one_hot_3d *= (255.0 / one_hot_3d.max())
    return one_hot_3d


def run_model(args, checkpoint, output_dir, loader=None):
    if args.save_graphs:
        from scene_generation.vis import draw_scene_graph
    dirname = os.path.dirname(args.checkpoint)
    features = None
    if not args.use_gt_textures:
        features_path = os.path.join(dirname, 'features_clustered_001.npy')
        print(features_path)
        if os.path.isfile(features_path):
            features = np.load(features_path, allow_pickle=True).item()
        else:
            raise ValueError('No features file')
    with torch.no_grad():
        vocab = checkpoint['model_kwargs']['vocab']
        model = build_model(args, checkpoint)
        if loader is None:
            loader = build_loader(args, checkpoint, vocab['is_panoptic'])
        accuracy_model = None
        if args.accuracy_model_path is not None and os.path.isfile(args.accuracy_model_path):
            accuracy_model = load_model(args.accuracy_model_path)

        img_dir = makedir(output_dir, 'images')
        graph_dir = makedir(output_dir, 'graphs', args.save_graphs)
        gt_img_dir = makedir(output_dir, 'images_gt', args.save_gt_imgs)
        layout_dir = makedir(output_dir, 'layouts', args.save_layout)

        img_idx = 0
        total_iou = 0
        total_boxes = 0
        r_05 = 0
        r_03 = 0
        corrects = 0
        real_objects_count = 0
        num_objs = model.num_objs
        colors = torch.randint(0, 256, [num_objs, 3]).float()
        for batch in loader:
            imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, attributes = [x.cuda() for x in batch]

            imgs_gt = imagenet_deprocess_batch(imgs)

            if args.use_gt_masks:
                masks_gt = masks
            else:
                masks_gt = None
            if args.use_gt_textures:
                all_features = None
            else:
                all_features = []
                for obj_name in objs:
                    obj_feature = features[obj_name.item()]
                    random_index = randint(0, obj_feature.shape[0] - 1)
                    feat = torch.from_numpy(obj_feature[random_index, :]).type(torch.float32).cuda()
                    all_features.append(feat)
            if not args.use_gt_attr:
                attributes = torch.zeros_like(attributes)

            # Run the model with predicted masks
            model_out = model(imgs, objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks_gt, attributes=attributes,
                              test_mode=True, use_gt_box=args.use_gt_boxes, features=all_features)
            imgs_pred, boxes_pred, masks_pred, _, layout, _ = model_out

            if accuracy_model is not None:
                if args.use_gt_boxes:
                    crops = crop_bbox_batch(imgs_pred, boxes, obj_to_img, 224)
                else:
                    crops = crop_bbox_batch(imgs_pred, boxes_pred, obj_to_img, 224)

                outputs = accuracy_model(crops)
                if type(outputs) == tuple:
                    outputs, _ = outputs
                _, preds = torch.max(outputs, 1)

                # statistics
                for pred, label in zip(preds, objs):
                    if label.item() != 0:
                        real_objects_count += 1
                        corrects += 1 if pred.item() == label.item() else 0

            # Remove the __image__ object
            boxes_pred_no_image = []
            boxes_gt_no_image = []
            for o_index in range(len(obj_to_img)):
                if o_index < len(obj_to_img) - 1 and obj_to_img[o_index] == obj_to_img[o_index + 1]:
                    boxes_pred_no_image.append(boxes_pred[o_index])
                    boxes_gt_no_image.append(boxes[o_index])
            boxes_pred_no_image = torch.stack(boxes_pred_no_image)
            boxes_gt_no_image = torch.stack(boxes_gt_no_image)

            iou, bigger_05, bigger_03 = jaccard(boxes_pred_no_image, boxes_gt_no_image)
            total_iou += iou
            r_05 += bigger_05
            r_03 += bigger_03
            total_boxes += boxes_pred_no_image.size(0)
            imgs_pred = imagenet_deprocess_batch(imgs_pred)

            obj_data = [objs, boxes_pred, masks_pred]
            _, obj_data = split_graph_batch(triples, obj_data, obj_to_img, triple_to_img)
            objs, boxes_pred, masks_pred = obj_data

            obj_data_gt = [boxes.data]
            if masks is not None:
                obj_data_gt.append(masks.data)
            triples, obj_data_gt = split_graph_batch(triples, obj_data_gt, obj_to_img, triple_to_img)
            layouts_3d = one_hot_to_rgb(layout, colors, num_objs)
            for i in range(imgs_pred.size(0)):
                img_filename = '%04d.png' % img_idx
                if args.save_gt_imgs:
                    img_gt = imgs_gt[i].numpy().transpose(1, 2, 0)
                    img_gt_path = os.path.join(gt_img_dir, img_filename)
                    imsave(img_gt_path, img_gt)
                if args.save_layout:
                    layout_3d = layouts_3d[i].numpy().transpose(1, 2, 0)
                    layout_path = os.path.join(layout_dir, img_filename)
                    imsave(layout_path, layout_3d)

                img_pred_np = imgs_pred[i].numpy().transpose(1, 2, 0)
                img_path = os.path.join(img_dir, img_filename)
                imsave(img_path, img_pred_np)

                if args.save_graphs:
                    graph_img = draw_scene_graph(objs[i], triples[i], vocab)
                    graph_path = os.path.join(graph_dir, img_filename)
                    imsave(graph_path, graph_img)

                img_idx += 1

            print('Saved %d images' % img_idx)
        avg_iou = total_iou / total_boxes
        print('avg_iou {}'.format(avg_iou.item()))
        print('r0.5 {}'.format(r_05 / total_boxes))
        print('r0.3 {}'.format(r_03 / total_boxes))
        if accuracy_model is not None:
            print('Accuracy {}'.format(corrects / real_objects_count))


if __name__ == '__main__':
    args = parser.parse_args()
    if args.checkpoint is None:
        raise ValueError('Must specify --checkpoint')

    checkpoint = torch.load(args.checkpoint)
    print('Loading model from ', args.checkpoint)
    run_model(args, checkpoint, args.output_dir)
