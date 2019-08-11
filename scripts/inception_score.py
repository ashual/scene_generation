from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from scipy.stats import entropy
from torch import nn
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
from scene_generation.layers import Interpolate


class InceptionScore(nn.Module):
    def __init__(self, cuda=True, batch_size=32, resize=False):
        super(InceptionScore, self).__init__()
        assert batch_size > 0
        self.resize = resize
        self.batch_size = batch_size
        self.cuda = cuda
        # Set up dtype
        self.device = 'cuda' if cuda else 'cpu'
        if not cuda and torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")

        # Load inception model
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model.eval()
        self.up = Interpolate(size=(299, 299), mode='bilinear').to(self.device)
        self.clean()

    def clean(self):
        self.preds = np.zeros((0, 1000))

    def get_pred(self, x):
        if self.resize:
            x = self.up(x)
        x = self.inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    def forward(self, imgs):
        # Get predictions
        preds_imgs = self.get_pred(imgs.to(self.device))
        self.preds = np.append(self.preds, preds_imgs, axis=0)

    def compute_score(self, splits=1):
        # Now compute the mean kl-div
        split_scores = []
        preds = self.preds
        N = self.preds.shape[0]
        for k in range(splits):
            part = preds[k * (N // splits): (k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)


# def inception_score():
#     """Computes the inception score of the generated images imgs
#
#     imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
#     cuda -- whether or not to run on GPU
#     batch_size -- batch size for feeding into Inception v3
#     splits -- number of splits
#     """


if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)


    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', default='', type=str)
    parser.add_argument('--splits', default=1, type=int)
    args = parser.parse_args()
    imagenet_data = dset.ImageFolder(args.dir, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    # data_loader = torch.utils.data.DataLoader(imagenet_data,
    #                                           batch_size=4,
    #                                           shuffle=False,
    #                                           num_workers=4)

    print("Calculating Inception Score...")
    # print(inception_score(imagenet_data, cuda=True, batch_size=32, resize=True, splits=args.splits))
