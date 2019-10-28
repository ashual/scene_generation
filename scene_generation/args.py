import argparse
import os
import socket
from datetime import datetime

from scene_generation.utils import int_tuple, str_tuple, bool_flag

COCO_DIR = os.path.expanduser('datasets/coco')

parser = argparse.ArgumentParser()

# Optimization hyperparameters
parser.add_argument('--batch_size', default=12, type=int)
parser.add_argument('--num_iterations', default=1000000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--mask_learning_rate', default=1e-5, type=float)

# Dataset options
parser.add_argument('--image_size', default='128,128', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--coco_train_image_dir',
                    default=os.path.join(COCO_DIR, 'images/train2017'))
parser.add_argument('--coco_val_image_dir',
                    default=os.path.join(COCO_DIR, 'images/val2017'))
parser.add_argument('--coco_train_instances_json',
                    default=os.path.join(COCO_DIR, 'annotations/instances_train2017.json'))
parser.add_argument('--coco_train_stuff_json',
                    default=os.path.join(COCO_DIR, 'annotations/stuff_train2017.json'))
parser.add_argument('--coco_val_instances_json',
                    default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
parser.add_argument('--coco_val_stuff_json',
                    default=os.path.join(COCO_DIR, 'annotations/stuff_val2017.json'))
parser.add_argument('--coco_panoptic_train', default=os.path.join(COCO_DIR, 'annotations/panoptic_train2017.json'))
parser.add_argument('--coco_panoptic_val', default=os.path.join(COCO_DIR, 'annotations/panoptic_val2017.json'))
parser.add_argument('--coco_panoptic_segmentation_train', default=os.path.join(COCO_DIR, 'panoptic/annotations/panoptic_train2017'))
parser.add_argument('--coco_panoptic_segmentation_val', default=os.path.join(COCO_DIR, 'panoptic/annotations/panoptic_val2017'))
parser.add_argument('--instance_whitelist', default=None, type=str_tuple)
parser.add_argument('--stuff_whitelist', default=None, type=str_tuple)
parser.add_argument('--coco_include_other', default=False, type=bool_flag)
parser.add_argument('--min_object_size', default=0.02, type=float)
parser.add_argument('--min_objects_per_image', default=3, type=int)
parser.add_argument('--max_objects_per_image', default=8, type=int)
parser.add_argument('--coco_stuff_only', default=True, type=bool_flag)  # Train over images that have at least one stuff
parser.add_argument('--is_panoptic', default=False, type=bool_flag)

# Generator options
parser.add_argument('--mask_size', default=32, type=int)
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--gconv_dim', default=128, type=int)
parser.add_argument('--gconv_hidden_dim', default=512, type=int)
parser.add_argument('--gconv_num_layers', default=5, type=int)
parser.add_argument('--mlp_normalization', default='none', type=str)
parser.add_argument('--activation', default='leakyrelu-0.2')
parser.add_argument('--pool_size', default=100, type=int)
parser.add_argument('--output_nc', default=3, type=int)
parser.add_argument('--n_downsample_global', default=4, type=int)
parser.add_argument('--box_dim', default=128, type=int)
parser.add_argument('--use_attributes', default=True, type=bool_flag)
parser.add_argument('--beta1', default=0.5, type=float)
parser.add_argument('--box_noise_dim', default=64, type=int)
parser.add_argument('--mask_noise_dim', default=64, type=int)

# Appearance Generator options
parser.add_argument('--rep_size', default=32, type=int)
parser.add_argument('--appearance_normalization', default='batch')

# Generator losses
parser.add_argument('--l1_pixel_loss_weight', default=.0, type=float)
parser.add_argument('--bbox_pred_loss_weight', default=10, type=float)
parser.add_argument('--vgg_features_weight', default=10.0, type=float)
parser.add_argument('--d_img_weight', default=1.0, type=float)
parser.add_argument('--d_img_features_weight', default=10.0, type=float)
parser.add_argument('--d_mask_weight', default=1.0, type=float)
parser.add_argument('--d_mask_features_weight', default=10.0, type=float)
parser.add_argument('--d_obj_weight', default=0.1, type=float)
parser.add_argument('--ac_loss_weight', default=0.1, type=float)

# Image discriminator
parser.add_argument('--ndf', default=64, type=int)
parser.add_argument('--num_D', default=2, type=int)
parser.add_argument('--norm_D', default='instance', type=str)
parser.add_argument('--n_layers_D', default=3, type=int)
parser.add_argument('--no_lsgan', default=False, type=bool_flag)  # Default is LSGAN (no_lsgan == False)

# Mask Discriminator
parser.add_argument('--ndf_mask', default=64, type=int)
parser.add_argument('--num_D_mask', default=1, type=int)
parser.add_argument('--norm_D_mask', default='instance', type=str)
parser.add_argument('--n_layers_D_mask', default=2, type=int)

# Object discriminator
parser.add_argument('--gan_loss_type', default='gan')
parser.add_argument('--d_normalization', default='batch')
parser.add_argument('--d_padding', default='valid')
parser.add_argument('--d_activation', default='leakyrelu-0.2')
parser.add_argument('--d_obj_arch', default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--crop_size', default=32, type=int)

# Output options
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join(os.getcwd(), 'output', current_time + '_' + socket.gethostname())
parser.add_argument('--print_every', default=100, type=int)
parser.add_argument('--checkpoint_every', default=10000, type=int)
parser.add_argument('--output_dir', default=log_dir)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--restore_from_checkpoint', default=False, type=bool_flag)


def get_args():
    return parser.parse_args()
