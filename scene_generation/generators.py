import torch
import torch.nn as nn

from scene_generation.layers import GlobalAvgPool, build_cnn, ResnetBlock, get_norm_layer, Interpolate


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def mask_net(dim, mask_size):
    output_dim = 1
    layers, cur_size = [], 1
    while cur_size < mask_size:
        layers.append(Interpolate(scale_factor=2, mode='nearest'))
        layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(dim))
        layers.append(nn.ReLU())
        cur_size *= 2
    if cur_size != mask_size:
        raise ValueError('Mask size must be a power of 2')
    layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
    return nn.Sequential(*layers)


class AppearanceEncoder(nn.Module):
    def __init__(self, vocab, arch, normalization='none', activation='relu',
                 padding='same', vecs_size=1024, pooling='avg'):
        super(AppearanceEncoder, self).__init__()
        self.vocab = vocab

        cnn_kwargs = {
            'arch': arch,
            'normalization': normalization,
            'activation': activation,
            'pooling': pooling,
            'padding': padding,
        }
        cnn, channels = build_cnn(**cnn_kwargs)
        self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(channels, vecs_size))

    def forward(self, crops):
        return self.cnn(crops)


def define_G(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, norm='instance'):
    norm_layer = get_norm_layer(norm_type=norm)
    netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    assert (torch.cuda.is_available())
    netG.cuda()
    netG.apply(weights_init)
    return netG

##############################################################################
# Generator
##############################################################################
class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)