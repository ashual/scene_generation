import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torch.autograd import Variable


def get_gan_losses(gan_type):
    """
    Returns the generator and discriminator loss for a particular GAN type.

    The returned functions have the following API:
    loss_g = g_loss(scores_fake)
    loss_d = d_loss(scores_real, scores_fake)
    """
    if gan_type == 'gan':
        return gan_g_loss, gan_d_loss
    elif gan_type == 'wgan':
        return wgan_g_loss, wgan_d_loss
    elif gan_type == 'lsgan':
        return lsgan_g_loss, lsgan_d_loss
    else:
        raise ValueError('Unrecognized GAN type "%s"' % gan_type)


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def _make_targets(x, y):
    """
    Inputs:
    - x: PyTorch Tensor
    - y: Python scalar

    Outputs:
    - out: PyTorch Variable with same shape and dtype as x, but filled with y
    """
    return torch.full_like(x, y)


def gan_g_loss(scores_fake):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Variable of shape (,) giving GAN generator loss
    """
    if scores_fake.dim() > 1:
        scores_fake = scores_fake.view(-1)
    y_fake = _make_targets(scores_fake, 1)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
    assert scores_real.size() == scores_fake.size()
    if scores_real.dim() > 1:
        scores_real = scores_real.view(-1)
        scores_fake = scores_fake.view(-1)
    y_real = _make_targets(scores_real, 1)
    y_fake = _make_targets(scores_fake, 0)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake


def wgan_g_loss(scores_fake):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving WGAN generator loss
    """
    return -scores_fake.mean()


def wgan_d_loss(scores_real, scores_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving WGAN discriminator loss
    """
    return scores_fake.mean() - scores_real.mean()


def lsgan_g_loss(scores_fake):
    if scores_fake.dim() > 1:
        scores_fake = scores_fake.view(-1)
    y_fake = _make_targets(scores_fake, 1)
    return F.mse_loss(scores_fake.sigmoid(), y_fake)


def lsgan_d_loss(scores_real, scores_fake):
    assert scores_real.size() == scores_fake.size()
    if scores_real.dim() > 1:
        scores_real = scores_real.view(-1)
        scores_fake = scores_fake.view(-1)
    y_real = _make_targets(scores_real, 1)
    y_fake = _make_targets(scores_fake, 0)
    loss_real = F.mse_loss(scores_real.sigmoid(), y_real)
    loss_fake = F.mse_loss(scores_fake.sigmoid(), y_fake)
    return loss_real + loss_fake


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


# VGG Features matching
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss