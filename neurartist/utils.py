"""
[DESCRIPTION]

@author: gjorando
"""

import os
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

    :param param_value: Raw input string.
    :param value_type: Target type of each element of the list.
    :return: Parsed list.
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
    the ones from the ImageNet dataset, which was used to train VGG19.

    :param width: Target width of the transformed input.
    :param model_mean: Mean of the trained model for normalization (default:
    ImageNet mean).
    :param model_mean: Standard deviation of the trained model for
    normalization (default: ImageNet std).
    :param max_value: Multiplier of the image (default: 255, as VGG19 excepts
    [0,255] images).
    :param device: Device onto which transfer the image.
    :return: A callable which transforms a PIL image into a (1, n_dims, height,
    width) PyTorch tensor.
    """

    return torchvision.transforms.Compose([
        # Resize to target size
        torchvision.transforms.Resize(width),
        # Convert Pillow image to torch tensor
        torchvision.transforms.ToTensor(),
        # Transfer tensor to the appropriate device
        torchvision.transforms.Lambda(lambda x: x.to(device)),
        # Normalize according to the model mean and standard deviation
        torchvision.transforms.Normalize(model_mean, model_std),
        # Convert [0, 1] values to [0, max_value] values
        torchvision.transforms.Lambda(lambda x: x.mul_(max_value)),
        # Add batch dimension
        torchvision.transforms.Lambda(lambda x: x.unsqueeze(0))
    ])


@_pm.export
def output_transforms(
    model_mean=(0.485, 0.456, 0.406),
    model_std=(0.229, 0.224, 0.225),
    max_value=255,
):
    """
    Transforms on output image.

    :param model_mean: Mean of the trained model for denormalization (default:
    ImageNet mean).
    :param model_mean: Standard deviation of the trained model for
    denormalization (default: ImageNet std).
    :param max_value: Same value as max_value that was used for input_tranforms
    (it will bring the image back to [0, 1] values).
    :return: A callable which transforms a (1, n_dims, height, width) PyTorch
    tensor into a PIL image.
    """

    return torchvision.transforms.Compose([
        # Remove batch dimension
        torchvision.transforms.Lambda(lambda x: x.data[0].squeeze()),
        # Convert [0, max_value] values back to [0, 1] values
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
    Compute the Gramians for each dimension of each image in a tensor.

    :param array: A (n_batchs, n_dims, height, width) PyTorch tensor.
    :return: Gramians matrices for each dimension of each image, as a
    (n_batchs, n_dims, n_dims) PyTorch tensor.
    """

    n_batchs, n_dims, height, width = array.size()

    array_flattened = array.view(n_batchs, n_dims, -1)
    G = torch.bmm(array_flattened, array_flattened.transpose(1, 2))

    return G.div(height*width)


@_pm.export
def guided_gram_matrix(array, guidance):
    """
    Compute the Guided Gramians for each dimension of each image in a tensor.

    :param array: A (n_batchs, n_dims, height, width) PyTorch tensor.
    :param guidance: A (n_guidance, n_batchs, 1, height, width) PyTorch tensor.
    :return: Guided Gramians matrices for each dimension of each image, as a
    (n_batchs, n_dims, n_dims) PyTorch tensor.
    """

    n_batchs, n_dims, height, width = array.size()
    n_guidance = guidance.size()[0]

    array_flattened = array.view(n_batchs, n_dims, -1)
    guidance_flattened = guidance.view(n_guidance, n_batchs, 1, -1)

    G = torch.zeros(n_guidance, n_dims, n_dims, device=array.device)
    for c in range(n_guidance):
        array_guided = torch.mul(guidance_flattened[c], array_flattened)
        G[c] = torch.bmm(array_guided, array_guided.transpose(1, 2))

    return G.div(height*width)


@_pm.export
def covariance_matrix(array):
    """
    Compute the unbiased covariance matrices for each channel of an image in a
    tensor.

    :param array: A (n_batchs, n_dims, height, width) PyTorch tensor.
    :return: Unbiased covariance matrices for each dimension of each image, as
    a (n_batches, n_dims, n_dims) PyTorch tensor.
    """

    n_batchs, n_dims, height, width = array.size()

    array_reshaped = array.view(n_batchs, n_dims, -1)

    array_centered = array_reshaped - torch.mean(array_reshaped, 2, True)
    K = torch.bmm(array_centered, array_centered.transpose(1, 2))

    return K.div((height*width)-1)


@_pm.export
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

    :param content_path: Path of the content image.
    :param style_path: Path of the style image.
    :param img_size: Target width of the transformed images.
    :param device: Device onto which transfer the images.
    :return: transformed content and style images.
    """

    images = (Image.open(content_path), Image.open(style_path))
    transformed_images = [
        input_transforms(img_size, device=device)(i)
        for i in images
    ]
    return transformed_images


@_pm.export
def luminance_only(content_image, output, normalize_luma=False):
    """
    Keep the luma of the output, and use the color of the content image.

    :param content_image: Content image.
    :param output: Output image.
    :param normalize_luma: If True, the luma channel is renormalized.
    :return: A new image with the luma of the output and the color of the
    content image.
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

    :param content_image: Content image.
    :param style_image: Style image.
    """

    style_cov = covariance_matrix(style_image).squeeze()
    content_cov = covariance_matrix(content_image).squeeze()
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


@_pm.export
def load_guidance_channels(
    guidance_path,
    img_size,
    model,
    method="simple",
    device="cpu"
):
    """
    Load guidance channels from a folder, containing one image
    file per guidance channel.

    :param guidance_path: Path of the folder in which guidance images are
    stored.
    :param img_size: Target width of the transformed images.
    :param model: Style transfer model, needed to broadcast guidance dimensions
    for each style layer.
    :param method: Propagation method.
    :param device: Device onto which transfer the images.
    :return: Loaded channels.
    """

    assert method in ("simple"), f"{method} is not a valid method"

    guidance_images = [
        Image.open(
            os.path.join(guidance_path, file)
        ).convert("L").convert("RGB")  # Broadcast gray images to RGB channels
        for file in sorted(os.listdir(guidance_path))
    ]

    guidance_channels = [
        input_transforms(
            img_size,
            [0]*3,
            [1]*3,
            1,
            device
        )(img)
        for img in guidance_images
    ]

    layer_sizes = [f.shape[2:4] for f in model(guidance_channels[0])[1]]

    guidance_images = [img.convert("L") for img in guidance_images]
    guidance_channels = []

    for layer_size in layer_sizes:
        if method == "simple":
            guidance_channels.append(torch.stack([
                input_transforms(
                    layer_size,
                    [0],
                    [1],
                    1,
                    device
                )(img)
                for img in guidance_images
            ]))
        elif method == "all":
            raise NotImplementedError("Not implemented")
        elif method == "inside":
            raise NotImplementedError("Not implemented")

    return guidance_channels
