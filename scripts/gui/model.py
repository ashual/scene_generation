import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
import torch
from imageio import imwrite

import scene_generation.vis as vis
from scene_generation.data.utils import imagenet_deprocess_batch
from scene_generation.model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output_dir', default='outputs')
parser.add_argument('--draw_scene_graphs', type=int, default=0)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])
args = parser.parse_args()


def get_model():
    if not os.path.isfile(args.checkpoint):
        print('ERROR: Checkpoint file "%s" not found' % args.checkpoint)
        print('Maybe you forgot to download pretraind models? Try running:')
        print('bash scripts/download_models.sh')
        return

    output_dir = os.path.join('scripts', 'gui', 'images', args.output_dir)
    if not os.path.isdir(output_dir):
        print('Output directory "%s" does not exist; creating it' % args.output_dir)
        os.makedirs(output_dir)

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
    features_path = os.path.join(dirname, 'features_clustered_100.npy')
    features_path_one = os.path.join(dirname, 'features_clustered_001.npy')
    features = np.load(features_path, allow_pickle=True).item()
    features_one = np.load(features_path_one, allow_pickle=True).item()
    model = Model(**checkpoint['model_kwargs'])
    model_state = checkpoint['model_state']
    model.load_state_dict(model_state)
    model.features = features
    model.features_one = features_one
    model.colors = torch.randint(0, 256, [172, 3]).float()
    model.colors[0, :] = 256
    model.eval()
    model.to(device)
    return model


def json_to_img(scene_graph, model):
    output_dir = args.output_dir
    scene_graphs = json_to_scene_graph(scene_graph)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    # Run the model forward
    with torch.no_grad():
        (imgs, boxes_pred, masks_pred, layout, layout_pred, _), objs = model.forward_json(scene_graphs)
    imgs = imagenet_deprocess_batch(imgs)

    # Save the generated image
    for i in range(imgs.shape[0]):
        img_np = imgs[i].numpy().transpose(1, 2, 0).astype('uint8')
        img_path = os.path.join('scripts', 'gui', 'images', output_dir, 'img{}.png'.format(current_time))
        imwrite(img_path, img_np)
        return_img_path = os.path.join('images', output_dir, 'img{}.png'.format(current_time))

    # Save the generated layout image
    for i in range(imgs.shape[0]):
        img_layout_np = one_hot_to_rgb(layout_pred[:, :172, :, :], model.colors)[0].numpy().transpose(1, 2, 0).astype(
            'uint8')
        obj_colors = []
        for obj in objs[:-1]:
            new_color = torch.cat([model.colors[obj] / 256, torch.ones(1)])
            obj_colors.append(new_color)

        img_layout_path = os.path.join('scripts', 'gui', 'images', output_dir, 'img_layout{}.png'.format(current_time))
        vis.add_boxes_to_layout(img_layout_np, scene_graphs[i]['objects'], boxes_pred, img_layout_path,
                                colors=obj_colors)
        return_img_layout_path = os.path.join('images', output_dir, 'img_layout{}.png'.format(current_time))

    # Draw and save the scene graph
    if args.draw_scene_graphs:
        for i, sg in enumerate(scene_graphs):
            sg_img = vis.draw_scene_graph(sg['objects'], sg['relationships'])
            sg_img_path = os.path.join('scripts', 'gui', 'images', output_dir, 'sg{}.png'.format(current_time))
            imwrite(sg_img_path, sg_img)
            sg_img_path = os.path.join('images', output_dir, 'sg{}.png'.format(current_time))

    return return_img_path, return_img_layout_path


def one_hot_to_rgb(one_hot, colors):
    one_hot_3d = torch.einsum('abcd,be->aecd', (one_hot.cpu(), colors.cpu()))
    one_hot_3d *= (255.0 / one_hot_3d.max())
    return one_hot_3d


def json_to_scene_graph(json_text):
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

        margin = (obj_s['size'] + 1) / 10 / 2
        mean_x_s = 0.5 * (sx0 + sx1)
        mean_y_s = 0.5 * (sy0 + sy1)

        sx0 = max(0, mean_x_s - margin)
        sx1 = min(1, mean_x_s + margin)
        sy0 = max(0, mean_y_s - margin)
        sy1 = min(1, mean_y_s + margin)

        size.append(obj_s['size'])
        location.append(obj_s['location'])

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

        margin = (obj_o['size'] + 1) / 10 / 2
        ox0 = max(0, mean_x_o - margin)
        ox1 = min(1, mean_x_o + margin)
        oy0 = max(0, mean_y_o - margin)
        oy1 = min(1, mean_y_o + margin)

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
             'features': features, 'image_id': image_id}]
