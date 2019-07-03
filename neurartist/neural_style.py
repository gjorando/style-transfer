"""
Neural style model.

@author: gjorandon
"""

import torch
import torchvision


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


def gram_matrix(array):
    """
    Compute the Gramians for each dimension of each image in a (n_batchs,
    n_dims, size1, size2) tensor.
    """

    n_batchs, n_dims, height, width = array.size()

    array_flattened = array.view(n_batchs, n_dims, -1)
    G = torch.bmm(array_flattened, array_flattened.transpose(1, 2))

    return G.div(height*width)


def content_loss(weights, truth, pred):
    """
    Compute the standard neural style content loss.
    """

    if torch.cuda.is_available():
        return sum([
            weights[i]*torch.nn.MSELoss().cuda()(layer, truth[i])
            for i, layer in enumerate(pred)
        ])
    else:
        return sum([
            weights[i]*torch.nn.MSELoss()(layer, truth[i])
            for i, layer in enumerate(pred)
        ])


def style_loss(weights, truth, pred):
    """
    Compute the standard neural style style loss.
    """

    return content_loss(
        weights,
        truth,
        [gram_matrix(layer) for layer in pred]
    )


class NeuralStyle(torch.nn.Module):
    """
    Base class for a standard neural style transfer, as described in "Image
    Style Transfer Using Convolutional Neural Networks", Gatys et al., 2016.
    """

    def __init__(
        self,
        features=None,
        content_layers=None,
        style_layers=None,
        content_weights=None,
        style_weights=None,
        trade_off=3
    ):
        """
        Not defining parameters with default value to None defines the standard
        neural style transfer used in the original article, with VGG19
        layers to represent both style and content.

        :param features: a Sequential model containing our layers
        :param content_layers: index(es) of content layer(s)
        :param style_layers: index(es) of style layer(s)
        :param content_weights: weights for each content layer
        :param style_weights: weights for each style layer
        :param trade_off: trade-off between a more faithful content
        reconstruction (trade_off>1) or a more faithful style reconstruction
        (trade_off<1 and trade_off>0)
        """

        super().__init__()

        if features is None:
            self.features = torchvision.models.vgg19(pretrained=True).features
            if content_layers is None:
                self.content_layers = [22]
            if style_layers is None:
                self.style_layers = [1, 6, 11, 20, 29]
            if content_weights is None:
                self.content_weights = [1e0]
            if style_weights is None:
                self.style_weights = [
                    1e3/n**2
                    for n in [64, 128, 256, 512, 512]
                ]
                self.style_weights = [
                    w/sum(self.style_weights)
                    for w in self.style_weights
                ]
        else:
            self.features = features
            if content_layers is None:
                raise ValueError("""
                model is not None; content_layers should be defined
                """)
            if style_layers is None:
                raise ValueError("""
                model is not None; style_layers should be defined
                """)
            if content_weights is None:
                raise ValueError("""
                model is not None; content_weights should be defined
                """)
            if style_weights is None:
                raise ValueError("""
                model is not None; style_weights should be defined
                """)

        if content_layers is not None:
            self.content_layers = content_layers
        if style_layers is not None:
            self.style_layers = style_layers
        if content_weights is not None:
            self.content_weights = content_weights
        if style_weights is not None:
            self.style_weights = style_weights

        self.alpha, self.beta = (w/(trade_off+1) for w in (trade_off, 1))
        self.features = self.features.eval()

        for param in self.parameters():
            param.requires_grad_(False)

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, input):
        output_style = []
        output_content = []
        handles = []

        for i in self.style_layers:
            def hook(module, input, output):
                output_style.append(output)
            handles.append(
                self.features[i].register_forward_hook(hook)
            )
        for i in self.content_layers:
            def hook(module, input, output):
                output_content.append(output)
            handles.append(
                self.features[i].register_forward_hook(hook)
            )

        self.features(input)

        [handle.remove() for handle in handles]

        return output_content, output_style

    def epoch(
        self,
        target,
        content_targets,
        style_targets,
        optimizer
    ):
        content_output, style_output = self(target)

        curr_content_loss = content_loss(
            self.content_weights,
            content_targets,
            content_output
        )
        curr_style_loss = style_loss(
            self.style_weights,
            style_targets,
            style_output
        )
        loss = self.alpha*curr_content_loss + self.beta*curr_style_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step(lambda: loss)

        return float(curr_content_loss), float(curr_style_loss), float(loss)
