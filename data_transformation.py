import torch
import torchvision.transforms.functional as F_vision
import numpy as np


class ToTensor(object):
    """Convert samples and labels to Tensors."""
    def __call__(self, sample):
        serie, label = sample['serie'], sample['label']

        return {'serie': torch.from_numpy(serie),
                'label': torch.tensor(label)}
    
class OneHotEncoding(object):
    """Convert labels (tensor) in one hot rappresentation."""
    def __init__(self, numlabels):
        self.numlabels = numlabels

    def __call__(self, sample):
        serie, label = sample['serie'], sample['label']

        return {'serie': serie,
                'label': torch.nn.functional.one_hot(label,self.numlabels)}
      
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        serie, label = sample['serie'], sample['label']
        return {'serie': F_vision.normalize(serie, self.mean, self.std, self.inplace),
                'label': label}


class RandomHorizontalFlip(object):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, sample):
        serie, label = sample['serie'], sample['label']
        if torch.rand(1) < self.p:
          #print(serie)
          serie = F_vision.hflip(serie)
          #print(serie)

        return {'serie': serie,
                'label': label}


