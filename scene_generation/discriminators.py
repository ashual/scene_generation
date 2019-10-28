import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scene_generation.bilinear import crop_bbox_batch
from scene_generation.layers import GlobalAvgPool, build_cnn, get_norm_layer


class AcDiscriminator(nn.Module):
    def __init__(self, vocab, arch, normalization='none', activation='relu', padding='same', pooling='avg'):
        super(AcDiscriminator, self).__init__()
        self.vocab = vocab

        cnn_kwargs = {
            'arch': arch,
            'normalization': normalization,
            'activation': activation,
            'pooling': pooling,
            'padding': padding,
        }
        cnn, D = build_cnn(**cnn_kwargs)
        self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(D, 1024))
        num_objects = len(vocab['object_to_idx'])

        self.real_classifier = nn.Linear(1024, 1)
        self.obj_classifier = nn.Linear(1024, num_objects)

    def forward(self, x, y):
        if x.dim() == 3:
            x = x[:, None]
        vecs = self.cnn(x)
        real_scores = self.real_classifier(vecs)
        obj_scores = self.obj_classifier(vecs)
        ac_loss = F.cross_entropy(obj_scores, y)
        return real_scores, ac_loss


class AcCropDiscriminator(nn.Module):
    def __init__(self, vocab, arch, normalization='none', activation='relu',
                 object_size=64, padding='same', pooling='avg'):
        super(AcCropDiscriminator, self).__init__()
        self.vocab = vocab
        self.discriminator = AcDiscriminator(vocab, arch, normalization,
                                             activation, padding, pooling)
        self.object_size = object_size

    def forward(self, imgs, objs, boxes, obj_to_img):
        crops = crop_bbox_batch(imgs, boxes, obj_to_img, self.object_size)
        real_scores, ac_loss = self.discriminator(crops, objs)
        return real_scores, ac_loss, crops


###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D)
    # print(netD)
    assert (torch.cuda.is_available())
    netD.cuda()
    netD.apply(weights_init)
    return netD


def define_mask_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1,
                  num_objects=None):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleMaskDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D,
                                       num_objects)
    assert (torch.cuda.is_available())
    netD.cuda()
    netD.apply(weights_init)
    return netD


class MultiscaleMaskDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, num_objects=None):
        super(MultiscaleMaskDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = NLayerMaskDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, num_objects)
            for j in range(n_layers + 2):
                setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input, cond):
        result = [input]
        for i in range(len(model) - 2):
            # print(result[-1].shape)
            result.append(model[i](result[-1]))

        a, b, c, d = result[-1].shape
        cond = cond.view(a, -1, 1, 1).expand(-1, -1, c, d)
        concat = torch.cat([result[-1], cond], dim=1)
        result.append(model[len(model) - 2](concat))
        result.append(model[len(model) - 1](result[-1]))
        return result[1:]

    def forward(self, input, cond):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                     range(self.n_layers + 2)]
            result.append(self.singleD_forward(model, input_downsampled, cond))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerMaskDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 num_objects=None):
        super(NLayerMaskDiscriminator, self).__init__()
        self.n_layers = n_layers

        kw = 3
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        nf_prev += num_objects
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        res = [input]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            for j in range(n_layers + 2):
                setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        result = [input]
        for i in range(len(model)):
            result.append(model[i](result[-1]))
        return result[1:]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                     range(self.n_layers + 2)]
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        res = [input]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]
