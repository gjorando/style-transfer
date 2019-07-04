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
@click.version_option(version=neurartist.__version__)
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
    type=click.INT,
    help="Trade-off between content (>1) and style (<1) faithfullness"
)
@click.option(
    "--device", "-d",
    default=None,
    help="PyTorch device to use (default: cuda if available, otherwise cpu)"
)
@click.option(
    "--verbose/--quiet",
    default=True,
    help="Verbose flag prints info during computation (default: verbose)"
)
def main(
    content_path,
    style_path,
    output_path,
    img_size,
    num_epochs,
    trade_off,
    verbose,
    device
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

    # Automatic detection of optimal device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # RuntimeError if we use a non-valid device
    torch.device(device)

    if verbose:
        print(f"Device={device}")
        print(f"Content={content_path}")
        print(f"Style={style_path}")
        print(f"Output={output_path}")
        print(f"Size={img_size}")
        print(f"Epochs={num_epochs}")
        print(f"Trade-off={trade_off}")

    # FIXME random_init still doesn't work
    random_init = False

    # Load and transform the output images
    content_image, style_image = neurartist.utils.load_input_images(
        content_path,
        style_path,
        img_size,
        device
    )

    # Instantiate the model
    model = neurartist.models.NeuralStyle(
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

    # Save the output image
    output_image = neurartist.utils.output_transforms(img_size)(
        output.clone().data[0].squeeze()
    )
    output_image.save(output_path)
