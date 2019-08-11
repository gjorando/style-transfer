"""
Entrypoints.

@author: gjorando
"""

import os
from datetime import datetime
import torch
import click
from PIL import Image
import neurartist


def odd_int(value):
    value = int(value)
    if value % 2 == 0:
        raise ValueError("Odd number required")
    return value

def threshold_or_neg(value):
    value = float(value)
    if value > 1:
        raise ValueError("Value should be between 0 and 1, or negative")
    return value

@click.command()
# General
@click.option(
    "--content", "-c",
    "content_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Content image"
)
@click.option(
    "--style", "-s",
    "style_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Style image"
)
@click.option(
    "--output", "-o",
    "output_path",
    default="./",
    type=click.Path(dir_okay=True, writable=True),
    help="Output path"
)
@click.option(
    "--size", "-S",
    "img_size",
    default=512,
    type=click.INT,
    help="Output size"
)
@click.option(
    "--epochs", "-e",
    "num_epochs",
    default=250,
    type=click.INT,
    help="Maximum number of epochs"
)
@click.option(
    "--trade-off",
    "trade_off",
    default=3,
    type=click.FLOAT,
    help="Trade-off between content (>1) and style (<1) faithfullness"
)
@click.option(
    "--init-random/--init-image",
    "random_init",
    default=False,
    help="Init optimizer either from random noise, or image (default)"
)
@click.option(
    "--init-image-path",
    "random_init_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="If --init-image is set, path to an image (default: content image)"
)
# Layers options
@click.option(
    "--content-layers",
    default=None,
    help="Indexes of content layers (as a string representing a list)"
)
@click.option(
    "--style-layers",
    default=None,
    help="Indexes of style layers (as a string representing a list)"
)
@click.option(
    "--content-weights",
    default=None,
    help="Content weights (as a string representing a list)"
)
@click.option(
    "--style-weights",
    default=None,
    help="Style weights (as a string representing a list)"
)
# Color control
@click.option(
    "--color-control",
    default="none",
    type=click.Choice(["histogram_matching", "luminance_only", "none"]),
    help="Color control method (default: none)"
)
@click.option(
    "--cc-luminance-only-normalize",
    "luminance_only_normalize",
    is_flag=True,
    help="For color control/luminance only method, normalize output luma"
)
# Spatial control
@click.option(
    "--content-guidance",
    "content_guidance_path",
    default=None,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Content guidance channels folder path"
)
@click.option(
    "--style-guidance",
    "style_guidance_path",
    default=None,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Style guidance channels folder path"
)
@click.option(
    "--guidance-propagation-method",
    default="simple",
    type=click.Choice(["simple", "inside", "all"]),
    help="Propagation method for guidance channels"
)
@click.option(
    "--guidance-threshold",
    default=.5,
    type=threshold_or_neg,
    help="Threshold between 0 and 1 for guidance channels thresholding, or any"
         " negative value for non thresholding"
)
@click.option(
    "--guidance-propagation-kernel-size",
    default=None,
    type=odd_int,
    help="Kernel size for propagation of guidance channels (relevant for "
         "inside and all methods)"
)
@click.option(
    "--guidance-propagation-dilation",
    default=None,
    type=click.INT,
    help="Dilation for propagation of guidance channels (relevant for "
         "inside and all methods)"
)
# Meta
@click.option(
    "--device", "-d",
    default=None,
    help="PyTorch device to use (default: cuda if available, otherwise cpu)"
)
@click.option(
    "--verbose/--quiet",
    "verbose",
    default=True,
    help="Verbose flag prints info during computation (default: verbose)"
)
@click.version_option(version=neurartist.__version__)
def main(
    content_path,
    style_path,
    output_path,
    img_size,
    num_epochs,
    trade_off,
    random_init,
    random_init_path,
    content_layers,
    style_layers,
    content_weights,
    style_weights,
    color_control,
    luminance_only_normalize,
    content_guidance_path,
    style_guidance_path,
    guidance_propagation_method,
    guidance_threshold,
    guidance_propagation_kernel_size,
    guidance_propagation_dilation,
    device,
    verbose
):
    """
    Create beautiful art using deep learning.
    """

    # Check that content_guidance_path and style_guidance_path are either both
    # None or both set
    guidance_check = int(content_guidance_path is None)
    guidance_check += int(style_guidance_path is None)
    if guidance_check not in (0, 2):
        raise ValueError(
            "content_guidance and style_guidance must be both set or both None"
        )

    # If a negative value is set, no thresholding is done
    if guidance_threshold < 0:
        guidance_threshold = None

    # If the output path is a directory, we append a generated filename
    if os.path.isdir(output_path):
        output_path = os.path.join(
            output_path,
            "{}.png".format(datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))
        )

    # Load argument lists
    content_layers = neurartist.utils.validate_list_parameter(content_layers)
    content_weights = neurartist.utils.validate_list_parameter(
        content_weights,
        float
    )
    style_layers = neurartist.utils.validate_list_parameter(style_layers)
    style_weights = neurartist.utils.validate_list_parameter(
        style_weights,
        float
    )

    # Automatic detection of optimal device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # RuntimeError if we use a non-valid device
    torch.device(device)

    # Load and transform the input images
    content_image, style_image = neurartist.utils.load_input_images(
        content_path,
        style_path,
        img_size,
        device
    )

    # If color control mode is histogram matching, update style image
    if color_control == "histogram_matching":
        neurartist.utils.color_histogram_matching(content_image, style_image)

    # Instantiate the model
    model = neurartist.models.NeuralStyle(
        content_layers=content_layers,
        style_layers=style_layers,
        content_weights=content_weights,
        style_weights=style_weights,
        trade_off=trade_off,
        device=device
    )

    # Load guidance channels if desired
    if content_guidance_path is None:
        content_guidance = None
        style_guidance = None
    else:
        kernel_params = {}
        if guidance_propagation_kernel_size is not None:
            kernel_params["kernel_size"] = \
                (guidance_propagation_kernel_size,)*2
        if guidance_propagation_dilation is not None:
            kernel_params["dilation"] = \
                (guidance_propagation_dilation,)*2

        content_guidance = neurartist.utils.load_guidance_channels(
            content_guidance_path,
            img_size,
            model,
            method=guidance_propagation_method,
            threshold=guidance_threshold,
            kernel_parameters=kernel_params,
            fallback_channel=True,
            device=device
        )
        style_guidance = neurartist.utils.load_guidance_channels(
            style_guidance_path,
            img_size,
            model,
            method=guidance_propagation_method,
            kernel_parameters=kernel_params,
            fallback_channel=True,
            device=device
        )

    # Initialize the optimizer
    if random_init:
        # despite what's described in the article, initializing the gradient
        # descent with a random input doesn't produce good results at all
        output = torch.randn(content_image.size()).type_as(content_image.data)
    elif random_init_path is None:
        output = content_image.clone()
    else:
        output = neurartist.utils.input_transforms(
            content_image.shape[-2:],  # Use actual content size
            device=device
        )(Image.open(random_init_path))

    # The output image is updated by backward propagation
    output.requires_grad_(True)
    optimizer = torch.optim.LBFGS([output])

    # Fetch the target style and content
    content_targets, style_targets = model.get_images_targets(
        content_image,
        style_image,
        style_guidance
    )

    if verbose:
        print(f"Device={device}")
        print(f"Content={content_path}")
        print(f"Style={style_path}")
        print(f"Output={output_path}")
        print(f"Size={img_size}")
        print(f"Epochs={num_epochs}")
        print(f"Trade-off={trade_off}")
        print(f"Random init={random_init}")
        print(f"Color control={color_control}")
        print(f"Guidance={content_guidance_path is not None}")
        if content_guidance_path is not None:
            print(f"Guidance propagation method={guidance_propagation_method}")
        print(f"Model={model}")
        print()
        print("Ctrl-C to prematurely end computations")
        print("Epoch\tContent loss\tStyle loss\tOverall")

    try:
        for i in range(num_epochs):
            # Run a forward/backward pass
            content_loss, style_loss, overall_loss = model.epoch(
                output,
                content_targets,
                style_targets,
                optimizer,
                content_guidance
            )

            if verbose:
                print("{}/{}\t{:.2f}\t{:.2f}\t{:.2f}".format(
                    str(i+1).zfill(len(str(num_epochs))),
                    num_epochs,
                    content_loss,
                    style_loss,
                    overall_loss
                ))
    except KeyboardInterrupt:  # Handle manual interruption through Ctrl-C
        if verbose:
            print("Manual interruption")

    # Convert the output image
    output_image = neurartist.utils.output_transforms()(
        output
    )

    # Luminance-only
    if color_control == "luminance_only":
        output_image = neurartist.utils.luminance_only(
            neurartist.utils.output_transforms()(
                content_image
            ),
            output_image,
            luminance_only_normalize
        )

    # Finally save the output
    output_image.save(output_path)
