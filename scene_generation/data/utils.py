import PIL
import torch
import numpy as np
import torchvision.transforms as T

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

INV_MEAN = [-m for m in MEAN]
INV_STD = [1.0 / s for s in STD]


def imagenet_preprocess():
    return T.Normalize(mean=MEAN, std=STD)


def rescale(x):
    lo, hi = x.min(), x.max()
    return x.sub(lo).div(hi - lo)


def imagenet_deprocess(rescale_image=True):
    transforms = [
        T.Normalize(mean=[0, 0, 0], std=INV_STD),
        T.Normalize(mean=INV_MEAN, std=[1.0, 1.0, 1.0]),
    ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)


def imagenet_deprocess_batch(imgs, rescale=True):
    """
    Input:
    - imgs: FloatTensor of shape (N, C, H, W) giving preprocessed images

    Output:
    - imgs_de: ByteTensor of shape (N, C, H, W) giving deprocessed images
      in the range [0, 255]
    """
    if isinstance(imgs, torch.autograd.Variable):
        imgs = imgs.data
    imgs = imgs.cpu().clone()
    deprocess_fn = imagenet_deprocess(rescale_image=rescale)
    imgs_de = []
    for i in range(imgs.size(0)):
        img_de = deprocess_fn(imgs[i])[None]
        img_de = img_de.mul(255).clamp(0, 255)
        imgs_de.append(img_de)
    imgs_de = torch.cat(imgs_de, dim=0)
    return imgs_de


class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            H, W = size
            self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)


def unpack_var(v):
    if isinstance(v, torch.autograd.Variable):
        return v.data
    return v


def split_graph_batch(triples, obj_data, obj_to_img, triple_to_img):
    triples = unpack_var(triples)
    obj_data = [unpack_var(o) for o in obj_data]
    obj_to_img = unpack_var(obj_to_img)
    triple_to_img = unpack_var(triple_to_img)

    triples_out = []
    obj_data_out = [[] for _ in obj_data]
    obj_offset = 0
    N = obj_to_img.max() + 1
    for i in range(N):
        o_idxs = (obj_to_img == i).nonzero().view(-1)
        t_idxs = (triple_to_img == i).nonzero().view(-1)

        cur_triples = triples[t_idxs].clone()
        cur_triples[:, 0] -= obj_offset
        cur_triples[:, 2] -= obj_offset
        triples_out.append(cur_triples)

        for j, o_data in enumerate(obj_data):
            cur_o_data = None
            if o_data is not None:
                cur_o_data = o_data[o_idxs]
            obj_data_out[j].append(cur_o_data)

        obj_offset += o_idxs.size(0)

    return triples_out, obj_data_out


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.uint32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return color[0] + 256 * color[1] + 256 * 256 * color[2]
