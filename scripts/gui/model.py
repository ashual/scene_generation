import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
import torch
from imageio import imwrite

from scene_generation.data.utils import imagenet_deprocess_batch
from scene_generation.model import Model

# import code.vis as vis

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--first_checkpoint', default=None)  # default='new_new_output/Mar08_17-10-10_pc-wolf-g04/checkpoint_with_model.pt')
parser.add_argument('--scene_graphs_json', default='scene_graphs/figure_5_coco.json')
parser.add_argument('--output_dir', default='outputs')
parser.add_argument('--draw_scene_graphs', type=int, default=0)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])


def get_model():
    args = parser.parse_args()
    print(args.checkpoint)
    if not os.path.isfile(args.checkpoint):
        print('ERROR: Checkpoint file "%s" not found' % args.checkpoint)
        print('Maybe you forgot to download pretraind models? Try running:')
        print('bash scripts/download_models.sh')
        return

    if not os.path.isdir(args.output_dir):
        print('Output directory "%s" does not exist; creating it' % args.output_dir)
        os.makedirs(args.output_dir)

    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'gpu':
        device = torch.device('cuda:0')
        if not torch.cuda.is_available():
            print('WARNING: CUDA not available; falling back to CPU')
            device = torch.device('cpu')

    # Load the model, with a bit of care in case there are no GPUs
    map_location = 'cpu' if device == torch.device('cpu') else None
    checkpoint = torch.load(args.checkpoint, map_location=map_location)
    dirname = os.path.dirname(args.checkpoint)
    features_path = os.path.join(dirname, 'features_clustered_010.npy')
    # print(features_path)
    # features = None
    if os.path.isfile(features_path):
        features = np.load(features_path).item()
    else:
        features = None
    model = Model(**checkpoint['model_kwargs'])
    model_state = checkpoint['model_state']
    if args.first_checkpoint is not None:
        all_new_keys = []
        first_checkpoint = torch.load(args.first_checkpoint)
        print('Loading first model from ', args.first_checkpoint)
        for (k, v) in first_checkpoint['model_best_inception_state'].items():
            # CHANGE: for (k, v) in first_checkpoint['model_best_state'].items():
            if k == 'repr_net.0.weight':
                break
            # print(k)
            model_state[k] = v
            all_new_keys.append(k)
        remove_old_keys = []
        for (k, v) in model_state.items():
            if 'mask' in k and k not in all_new_keys:
                remove_old_keys.append(k)
        for k in remove_old_keys:
            del model_state[k]
    model.load_state_dict(model_state)
    model.features = features
    model.colors = torch.randint(0, 256, [134, 3]).float()
    model.eval()
    model.to(device)
    return model


def json_to_img(scene_graph, model):
    scene_graphs = json_to_scene_graph(scene_graph)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    print(scene_graphs, current_time)
    # Run the model forward
    with torch.no_grad():
        imgs, boxes_pred, masks_pred, layout, layout_pred, _ = model.forward_json(scene_graphs)
    imgs = imagenet_deprocess_batch(imgs)

    # Save the generated images
    for i in range(imgs.shape[0]):
        img_np = imgs[i].numpy().transpose(1, 2, 0)
        img_path = os.path.join('scripts', 'gui', 'images', 'img{}.png'.format(current_time))
        imwrite(img_path, img_np)
        return_img_path = os.path.join('images', 'img{}.png'.format(current_time))

    # Save the generated images
    for i in range(imgs.shape[0]):
        img_layout_np = one_hot_to_rgb(layout_pred[:, :134, :, :], model.colors)[0].numpy().transpose(1, 2, 0)
        img_layout_path = os.path.join('scripts', 'gui', 'images', 'img_layout{}.png'.format(current_time))
        imwrite(img_layout_path, img_layout_np)
        return_img_layout_path = os.path.join('images', 'img_layout{}.png'.format(current_time))
    # Draw the scene graphs
    # for i, sg in enumerate(scene_graphs):
    #     sg_img = vis.draw_scene_graph(sg['objects'], sg['relationships'])
    #     sg_img_path = os.path.join('images', 'sg%06d.png' % i)
    #     imwrite(sg_img_path, sg_img)
    return return_img_path, return_img_layout_path


def one_hot_to_rgb(one_hot, colors):
    one_hot_3d = torch.einsum('abcd,be->aecd', (one_hot.cpu(), colors.cpu()))
    one_hot_3d *= (255.0 / one_hot_3d.max())
    return one_hot_3d


def json_to_scene_graph(json_text):
    # size = 10
    # location = 16
    scene = json.loads(json_text)
    if len(scene) == 0:
        return []
    image_id = scene['image_id']
    scene = scene['objects']
    objects = [i['text'] for i in scene]
    relationships = []
    size = []
    location = []
    features = []
    for i in range(0, len(objects)):
        obj_s = scene[i]
        # Check for inside / surrounding

        sx0 = obj_s['left']
        sy0 = obj_s['top']
        sx1 = obj_s['width'] + sx0
        sy1 = obj_s['height'] + sy0
        mean_x_s = 0.5 * (sx0 + sx1)
        mean_y_s = 0.5 * (sy0 + sy1)
        #
        # size_index = round((10 - 1) * obj_s['width'] * obj_s['height'])
        # location_index = int(round(mean_x_s * ((16 / 4) - 1)) + (16 / 4) * round(mean_y_s * ((16 / 4) - 1)))
        size.append(obj_s['size'])
        location.append(obj_s['location'])

        # feature = 5 if obj_s['feature'] == -1 else obj_s['feature']
        features.append(obj_s['feature'])
        if i == len(objects) - 1:
            continue

        obj_o = scene[i + 1]
        ox0 = obj_o['left']
        oy0 = obj_o['top']
        ox1 = obj_o['width'] + ox0
        oy1 = obj_o['height'] + oy0

        mean_x_o = 0.5 * (ox0 + ox1)
        mean_y_o = 0.5 * (oy0 + oy1)
        d_x = mean_x_s - mean_x_o
        d_y = mean_y_s - mean_y_o
        theta = math.atan2(d_y, d_x)

        if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
            p = 'surrounding'
        elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
            p = 'inside'
        elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
            p = 'left of'
        elif -3 * math.pi / 4 <= theta < -math.pi / 4:
            p = 'above'
        elif -math.pi / 4 <= theta < math.pi / 4:
            p = 'right of'
        elif math.pi / 4 <= theta < 3 * math.pi / 4:
            p = 'below'
        relationships.append([i, p, i + 1])

    return [{'objects': objects, 'relationships': relationships, 'attributes': {'size': size, 'location': location},
             'features': features}]
