import argparse
import os
import pickle
from torch.utils.data import DataLoader
from scene_generation.data.coco import CocoSceneGraphDataset, coco_collate_fn
from scene_generation.data.coco_panoptic import CocoPanopticSceneGraphDataset, coco_panoptic_collate_fn
from scene_generation.utils import int_tuple, bool_flag
from scene_generation.utils import str_tuple


parser = argparse.ArgumentParser()

# Shared dataset options
parser.add_argument('--dataset', default='coco')
parser.add_argument('--image_size', default=(128, 128), type=int_tuple)
parser.add_argument('--min_object_size', default=0.02, type=float)
parser.add_argument('--min_objects_per_image', default=3, type=int)
parser.add_argument('--max_objects_per_image', default=8, type=int)
parser.add_argument('--instance_whitelist', default=None, type=str_tuple)
parser.add_argument('--stuff_whitelist', default=None, type=str_tuple)
parser.add_argument('--coco_include_other', default=False, type=bool_flag)
parser.add_argument('--is_panoptic', default=False, type=bool_flag)
parser.add_argument('--batch_size', default=24, type=int)
parser.add_argument('--shuffle', default=False, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--num_samples', default=None, type=int)
parser.add_argument('--object_size', default=64, type=int)
parser.add_argument('--grid_size', default=25, type=int)
parser.add_argument('--size_attribute_len', default=10, type=int)

parser.add_argument('--output_dir', default='models')


COCO_DIR = os.path.expanduser('datasets/coco')
parser.add_argument('--coco_image_dir',
                    default=os.path.join(COCO_DIR, 'images/train2017'))
parser.add_argument('--instances_json',
                    default=os.path.join(COCO_DIR, 'annotations/instances_train2017.json'))
parser.add_argument('--stuff_json',
                    default=os.path.join(COCO_DIR, 'annotations/stuff_train2017.json'))
parser.add_argument('--coco_panoptic_train', default=os.path.join(COCO_DIR, 'annotations/panoptic_train2017.json'))
parser.add_argument('--coco_panoptic_segmentation_train',
                    default=os.path.join(COCO_DIR, 'panoptic/annotations/panoptic_train2017'))



def build_coco_dset(args):
    dset_kwargs = {
        'image_dir': args.coco_image_dir,
        'instances_json': args.instances_json,
        'stuff_json': args.stuff_json,
        'image_size': args.image_size,
        'mask_size': 32,
        'max_samples': args.num_samples,
        'min_object_size': args.min_object_size,
        'min_objects_per_image': args.min_objects_per_image,
        'max_objects_per_image': args.max_objects_per_image,
        'instance_whitelist': args.instance_whitelist,
        'stuff_whitelist': args.stuff_whitelist,
        'include_other': args.coco_include_other,
        'test_part': False,
        'sample_attributes': False,
        'grid_size': args.grid_size
    }
    dset = CocoSceneGraphDataset(**dset_kwargs)
    return dset


def build_coco_panoptic_dset(args):
    dset_kwargs = {
        'image_dir': args.coco_image_dir,
        'instances_json': args.instances_json,
        'panoptic': args.coco_panoptic_train,
        'panoptic_segmentation': args.coco_panoptic_segmentation_train,
        'stuff_json': args.stuff_json,
        'image_size': args.image_size,
        'mask_size': 32,
        'max_samples': args.num_samples,
        'min_object_size': args.min_object_size,
        'min_objects_per_image': args.min_objects_per_image,
        'max_objects_per_image': args.max_objects_per_image,
        'instance_whitelist': args.instance_whitelist,
        'stuff_whitelist': args.stuff_whitelist,
        'include_other': args.coco_include_other,
        'test_part': False,
        'sample_attributes': args.sample_attributes,
        'grid_size': args.grid_size
    }
    dset = CocoPanopticSceneGraphDataset(**dset_kwargs)
    return dset


def build_loader(args, is_panoptic):
    if is_panoptic:
        dset = build_coco_panoptic_dset(args)
        collate_fn = coco_panoptic_collate_fn
    else:
        dset = build_coco_dset(args)
        collate_fn = coco_collate_fn

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': args.shuffle,
        'collate_fn': collate_fn,
    }
    loader = DataLoader(dset, **loader_kwargs)
    return loader


if __name__ == '__main__':
    args = parser.parse_args()

    print('Loading dataset')
    loader = build_loader(args, args.is_panoptic)
    vocab = loader.dataset.vocab
    idx_to_name = vocab['my_idx_to_obj']
    sample_attributes = {'location': {}, 'size': {}}
    for obj_name in idx_to_name:
        sample_attributes['location'][obj_name] = [0] * args.grid_size
        sample_attributes['size'][obj_name] = [0] * args.size_attribute_len

    print('Iterating objects')
    for _, objs, _, _, _, _, _, attributes in loader:
        for obj, attribute in zip(objs, attributes):
            obj = obj.item()
            if obj == 0:
                continue
            obj_name = idx_to_name[obj - 1]
            size_index = attribute.int().tolist()[:args.size_attribute_len].index(1)
            location_index = attribute.int().tolist()[args.size_attribute_len:].index(1)
            sample_attributes['size'][obj_name][size_index] += 1
            sample_attributes['location'][obj_name][location_index] += 1

    attributes_file = './models/attributes_{}_{}.pickle'.format(args.size_attribute_len, args.grid_size)
    print('Saving attributes file to {}'.format(attributes_file))
    pickle.dump(sample_attributes, open(attributes_file, 'wb'))
