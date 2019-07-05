"""
[DESCRIPTION]

@author: gjorando
"""

import json
import torch
import torchvision
from neurartist import _package_manager as _pm
from PIL import Image
from PIL import ImageStat


@_pm.export
def validate_list_parameter(param_value, value_type=int):
    """
    Validate a command line parameter being a literal json list (as a string).
    It returns the parsed value.
    """
    if param_value is None:
        result = None
    else:
        result = json.loads(param_value)
        assert isinstance(result, list), "parameter should be a list"
        for i, v in enumerate(result):
            result[i] = value_type(v)

    return result


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
    n_dims, height, width) tensor.
    """

    n_batchs, n_dims, height, width = array.size()

    array_flattened = array.view(n_batchs, n_dims, -1)
    G = torch.bmm(array_flattened, array_flattened.transpose(1, 2))

    return G.div(height*width)


@_pm.export
def covariance_matrix(array):
    """
    Compute the unbiased covariance matrices for each channel of an image in a
    (1, n_dims, height, width) tensor.
    """
    n_batchs, n_dims, height, width = array.size()

    array_reshaped = array.view(n_dims, -1)

    array_centered = array_reshaped - torch.mean(array_reshaped, 1, True)
    K = torch.mm(array_centered, array_centered.t())

    return K.div((height*width)-1)


def tensor_pow(m, p):
    """
        Raise a 2 dimensional (matrix like) torch tensor to the power of p
        (m^p) and return the value. May not work if m isn't definite positive.

        :param m: PyTorch 2-Tensor
        :param p: Power value
        :return: Result
    """

    U, D, V = torch.svd(m)

    result = torch.mm(torch.mm(U, D.pow(p).diag()), V.t())

    return result


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


@_pm.export
def luminance_only(content_image, output, normalize_luma=False):
    """
    Replace the luma of output with the luma of content_image. The result is
    returned as a copy.
    """

    # luma of the output image
    yuv_output_luma = output.convert("YCbCr").split()[0]
    # content image as YCbCr bands
    yuv_content_image_bands = content_image.convert("YCbCr").split()

    # normalization of the output luma
    if normalize_luma:
        output_stats = ImageStat.Stat(yuv_output_luma)
        content_image_stats = ImageStat.Stat(yuv_content_image_bands[0])

        stddev_ratio = content_image_stats.stddev[0]/output_stats.stddev[0]
        output_mean = output_stats.mean[0]
        content_image_mean = content_image_stats.mean[0]

        yuv_output_luma = yuv_output_luma.point(
            lambda p: stddev_ratio * (p-output_mean) + content_image_mean
        )

    return Image.merge(
        "YCbCr",
        (
            yuv_output_luma,
            yuv_content_image_bands[1],
            yuv_content_image_bands[2]
        )
    ).convert("RGB")


@_pm.export
def color_histogram_matching(content_image, style_image):
    """
    Match the histogram of style_image with the one of content_image.
    style_image is modified inplace.
    """

    style_cov = covariance_matrix(style_image)
    content_cov = covariance_matrix(content_image)
    style_means = torch.mean(
        style_image.reshape(style_image.shape[1], -1),
        1,
        True
    )
    content_means = torch.mean(
        content_image.reshape(content_image.shape[1], -1),
        1,
        True
    )

    # For each pixel of the style image, we get the histogram matched version
    # with p' = chm_A*p+chm_b
    chm_A = torch.mm(tensor_pow(content_cov, 0.5), tensor_pow(style_cov, -0.5))
    chm_b = content_means - torch.mm(chm_A, style_means)

    # TODO find an quicker way to do this (maybe use Pillow?)
    for i in range(style_image.shape[2]):
        for j in range(style_image.shape[3]):
            style_image.squeeze()[:, i, j] = (
                torch.mm(
                    chm_A,
                    style_image.squeeze()[:, i, j].unsqueeze(1)
                ) + chm_b
            ).squeeze()
