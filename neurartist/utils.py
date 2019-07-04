"""
[DESCRIPTION]

@author: gjorando
"""

import torch
import torchvision
from neurartist import _package_manager as _pm
from PIL import Image


@_pm.export
def input_transforms(
    width,
    model_mean=(0.485, 0.456, 0.406),
    model_std=(0.229, 0.224, 0.225),
    max_value=255,
    device="cpu"
):
    """
    Transforms on input images. Default model mean and standard deviations are
    the ones from the ImageNet dataset, which was used to train VGG19. Default
    max_value is 255, as VGG19 expects RGB [0, 255] images.
    """

    return torchvision.transforms.Compose([
        # Resize to target size
        torchvision.transforms.Resize(width),
        # Convert Pillow image to torch tensor
        torchvision.transforms.ToTensor(),
        # transfer tensor to the appropriate device
        torchvision.transforms.Lambda(lambda x: x.to(device)),
        # normalize according to the model mean and standard deviation
        torchvision.transforms.Normalize(model_mean, model_std),
        # convert [0, 1] values to [0, max_value] values
        torchvision.transforms.Lambda(lambda x: x.mul_(max_value))
    ])


@_pm.export
def output_transforms(
    width,
    model_mean=(0.485, 0.456, 0.406),
    model_std=(0.229, 0.224, 0.225),
    max_value=255,
):
    """
    Transforms on output image.
    """

    return torchvision.transforms.Compose([
        # convert [0, max_value] values back to [0, 1] values
        torchvision.transforms.Lambda(lambda x: x.mul_(1/max_value)),
        # Denormalize the tensor
        torchvision.transforms.Normalize(
            mean=[0]*3,
            std=[1/std for std in model_std]),
        torchvision.transforms.Normalize(
            mean=[-mean for mean in model_mean],
            std=[1]*3
        ),
        # Clamp values to [0, 1]
        torchvision.transforms.Lambda(lambda x: x.clamp(0, 1)),
        # Cransfer tensor to the CPU
        torchvision.transforms.Lambda(lambda x: x.cpu()),
        # Convert torch tensor back to a Pillow image
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


@_pm.export
def load_input_images(content_path, style_path, img_size, device="cpu"):
    """
    Load and transform input images.
    """

    images = (Image.open(content_path), Image.open(style_path))
    transformed_images = [
        input_transforms(img_size, device=device)(i).unsqueeze(0)
        for i in images
    ]
    return transformed_images
