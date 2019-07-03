"""
Entrypoints.

@author: gjorandon
"""

import torch
import matplotlib.pyplot as plt
from PIL import Image
import neurartist


def main():
    content_path = "./images/fruits.jpg"
    style_path = "./images/cezanne2.jpg"
    width = 512
    num_epochs = 500
    show_every = 20
    trade_off = 3
    random_init = True

    images = (Image.open(content_path), Image.open(style_path))
    transformed_images = [
        neurartist.input_transforms(width)(i) for i in images
    ]
    if torch.cuda.is_available():
        transformed_images = (
            i.unsqueeze(0).cuda() for i in transformed_images
        )
    else:
        transformed_images = (i.unsqueeze(0) for i in transformed_images)
    content_image, style_image = transformed_images

    for img, title in zip(images, ("content", "style")):
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        plt.show()

    model = neurartist.NeuralStyle(trade_off=trade_off)

    if random_init:
        output = torch.randn(content_image.size()).type_as(content_image.data)
    else:
        output = content_image.clone()
    output.requires_grad_(True)
    optimizer = torch.optim.LBFGS([output])

    content_targets = [
        f.detach() for f in model(content_image)[0]
    ]
    style_targets = [
        neurartist.gram_matrix(f).detach() for f in model(style_image)[1]
    ]

    content_losses = []
    style_losses = []
    overall_losses = []
    try:
        for i in range(num_epochs):
            content_loss, style_loss, overall_loss = model.epoch(
                output,
                content_targets,
                style_targets,
                optimizer
            )
            content_losses.append(content_loss)
            style_losses.append(style_loss)
            overall_losses.append(overall_loss)

            print("#{}: content_loss={}, style_loss={}, overall={}".format(
                i,
                content_loss,
                style_loss,
                overall_loss
            ))
            if show_every and i % show_every == 0:
                plt.imshow(neurartist.output_transforms(width)(
                    output.clone().data[0].cpu().squeeze()
                ))
                plt.axis("off")
                plt.show()
    except KeyboardInterrupt:
        print("Manual interruption")

    plt.imshow(neurartist.output_transforms(width)(
        output.clone().data[0].cpu().squeeze()
    ))
    plt.title("End result")
    plt.axis("off")
    plt.show()

    plt.plot(content_losses, label="content")
    plt.plot(style_losses, label="style")
    plt.plot(overall_losses, label="overall")
    plt.title("Losses over time")
    plt.legend(loc=4)
    plt.show
