"""
Entrypoints.

@author: gjorando
"""

import os
from datetime import datetime
import torch
import click
import neurartist


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
    "--init-random/--init-content",
    "random_init",
    default=False,
    help="Init optimizer either from random noise or content image (default)"
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
# Efficient high resolution
@click.option(
    "--hr-lowres-size",
    "img_size_lowres",
    type=click.INT,
    default=None,
    help="""
    If set, intermediate size used for efficient high resolution images (this
    value should be set for higher (>~600) values of --size/-S)
    """
)
@click.option(
    "--hr-epochs",
    "num_epochs_hr",
    default=None,
    type=click.INT,
    help="""
        Number of epochs for high resolution second pass (if different from
        --epochs/-e)
    """
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
    content_layers,
    style_layers,
    content_weights,
    style_weights,
    color_control,
    luminance_only_normalize,
    img_size_lowres,
    num_epochs_hr,
    device,
    verbose
):
    """
    Create beautiful art using deep learning.
    """

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

    # High resolution computation
    compute_hr = img_size_lowres is not None
    if compute_hr:
        assert img_size_lowres < img_size, \
            "intermediate lowres size should be smaller than final size"
        if num_epochs_hr is None:
            num_epochs_hr = num_epochs
        # img_size_hr will store the final output size, and img_size the
        # intermediate lowres size
        img_size_hr, img_size = img_size, img_size_lowres
        # Load and transform high resolution input images
        content_image_hr, style_image_hr = neurartist.utils.load_input_images(
            content_path,
            style_path,
            img_size_hr,
            device
        )

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
        # Histogram matching on highres style image as well
        if compute_hr:
            style_image_pil = neurartist.utils.output_transforms()(
                style_image.data[0].squeeze()
            )
            style_image_hr = neurartist.utils.input_transforms(
                # We take the exact actual size, because torchvision transform
                # is not consistent with only one value
                style_image_hr.shape[2:4],
                device=device
            )(
                style_image_pil,
            ).unsqueeze(0)

    # Instantiate the model
    model = neurartist.models.NeuralStyle(
        content_layers=content_layers,
        style_layers=style_layers,
        content_weights=content_weights,
        style_weights=style_weights,
        trade_off=trade_off,
        device=device
    )

    # Initialize the optimizer
    if random_init:
        # despite what's described in the article, initializing the gradient
        # descent with a random input doesn't produce good results at all
        output = torch.randn(content_image.size()).type_as(content_image.data)
    else:
        output = content_image.clone()
    # The output image is updated by backward propagation
    output.requires_grad_(True)
    optimizer = torch.optim.LBFGS([output])

    # Fetch the target style and content
    content_targets, style_targets = model.get_images_targets(
        content_image,
        style_image
    )

    if verbose:
        print(f"Device={device}")
        print(f"Content={content_path}")
        print(f"Style={style_path}")
        print(f"Output={output_path}")
        if compute_hr:
            print(f"Highres size={img_size_hr}")
            print(f"Lowres size={img_size}")
        else:
            print(f"Size={img_size}")
        print(f"Epochs={num_epochs}")
        if compute_hr:
            print(f"Highres epochs={num_epochs_hr}")
        print(f"Trade-off={trade_off}")
        print(f"Random init={random_init}")
        print(f"Color control={color_control}")
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
                optimizer
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
        output.data[0].squeeze()
    )

    # High resolution pass
    if compute_hr:
        # Reinstantiate the model and clear some gpu cache
        if device.startswith("cuda"):
            # FIXME there is still a lot of memory that is not properly fred
            del model
            del output
            del content_image
            del style_image
            torch.cuda.empty_cache()
            model = neurartist.models.NeuralStyle(
                content_layers=content_layers,
                style_layers=style_layers,
                content_weights=content_weights,
                style_weights=style_weights,
                trade_off=trade_off,
                device=device
            )

        # Resize the output to final highres size
        output = neurartist.utils.input_transforms(
            # We take the exact actual size, because torchvision transform
            # is not consistent with only one value
            content_image_hr.shape[2:4],
            device=device
        )(
            output_image
        ).unsqueeze(0)

        # The highres output image is updated by backward propagation
        output.requires_grad_(True)
        optimizer = torch.optim.LBFGS([output])

        # Fetch the highres target style and content
        content_targets, style_targets = model.get_images_targets(
            content_image_hr,
            style_image_hr
        )

        print()
        print("High resolution pass")
        print("Epoch\tContent loss\tStyle loss\tOverall")
        try:
            for i in range(num_epochs_hr):
                # Run a forward/backward pass
                content_loss, style_loss, overall_loss = model.epoch(
                    output,
                    content_targets,
                    style_targets,
                    optimizer
                )

                if verbose:
                    print("{}/{}\t{:.2f}\t{:.2f}\t{:.2f}".format(
                        str(i+1).zfill(len(str(num_epochs_hr))),
                        num_epochs_hr,
                        content_loss,
                        style_loss,
                        overall_loss
                    ))
        except KeyboardInterrupt:  # Handle manual interruption through Ctrl-C
            if verbose:
                print("Manual interruption")

        # Convert the highres output image
        output_image = neurartist.utils.output_transforms()(
            output.data[0].squeeze()
        )

    # Luminance-only
    if color_control == "luminance_only":
        output_image = neurartist.utils.luminance_only(
            neurartist.utils.output_transforms()(
                content_image_hr.data[0].squeeze() if compute_hr else
                content_image.data[0].squeeze()
            ),
            output_image,
            luminance_only_normalize
        )

    # Finally save the output
    output_image.save(output_path)
