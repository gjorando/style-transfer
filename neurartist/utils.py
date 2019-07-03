"""
[DESCRIPTION]

@author: gjorandon
"""

import torch
import torchvision
from neurartist import _package_manager as _pm


@_pm.export
def input_transforms(
    width,
    model_mean=(0.485, 0.456, 0.406),
    model_std=(0.229, 0.224, 0.225),
    max_value=255
):
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(width),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(model_mean, model_std),
        torchvision.transforms.Lambda(lambda x: x.mul_(max_value))
    ])


@_pm.export
def output_transforms(
    width,
    model_mean=(0.485, 0.456, 0.406),
    model_std=(0.229, 0.224, 0.225),
    max_value=255
):
    return torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda x: x.mul_(1/max_value)),
        torchvision.transforms.Normalize(
            mean=[0]*3,
            std=[1/std for std in model_std]),
        torchvision.transforms.Normalize(
            mean=[-mean for mean in model_mean],
            std=[1]*3
        ),
        torchvision.transforms.Lambda(lambda x: x.clamp(0, 1)),
        torchvision.transforms.ToPILImage()
    ])


@_pm.export
def gram_matrix(array):
    """
    Compute the Gramians for each dimension of each image in a (n_batchs,
    n_dims, size1, size2) tensor.
    """

    n_batchs, n_dims, height, width = array.size()

    array_flattened = array.view(n_batchs, n_dims, -1)
    G = torch.bmm(array_flattened, array_flattened.transpose(1, 2))

    return G.div(height*width)
