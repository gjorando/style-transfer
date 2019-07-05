"""
Neural style models.

@author: gjorando
"""

import torch
import torchvision
from neurartist import losses
from neurartist import utils
from neurartist import _package_manager as _pm


@_pm.export
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
        trade_off=3,
        device="cpu"
    ):
        """
        Not defining parameters with default value to None defines the standard
        neural style transfer used in the original article, with VGG19
        layers to represent both style and content.

        :param features: a trained Sequential model containing our layers
        :param content_layers: index(es) of content layer(s)
        :param style_layers: index(es) of style layer(s)
        :param content_weights: weights for each content layer
        :param style_weights: weights for each style layer
        :param trade_off: trade-off between a more faithful content
        reconstruction (trade_off>1) or a more faithful style reconstruction
        (trade_off<1 and trade_off>0)
        """

        super().__init__()
        self.content_layers = None
        self.content_weights = None
        self.style_layers = None
        self.style_weights = None

        if features is None:
            # The default features are the one described in the article
            self.features = torchvision.models.vgg19(pretrained=True).features

            # The next lines set default values for layer indexes and weights
            if content_layers is None:
                # ReLU of conv4_2
                self.content_layers = [22]
                if content_weights is None:
                    self.content_weights = [1e0]
            if style_layers is None:
                # ReLU of conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
                self.style_layers = [1, 6, 11, 20, 29]
                if style_weights is None:
                    # style weights found in another article (they work well)
                    self.style_weights = [
                        1e3/n**2
                        for n in [64, 128, 256, 512, 512]
                    ]
        else:
            # If we use another set of features, we have no default layers
            self.features = features
            if content_layers is None:
                raise ValueError("""
                model is not None; content_layers should be defined
                """)
            if style_layers is None:
                raise ValueError("""
                model is not None; style_layers should be defined
                """)

        # Define layer indexes and weights
        if content_layers is not None:
            for layer in content_layers:  # check that each layer is valid
                assert layer in range(len(self.features)), \
                    "{} is not in features".format(layer)
            self.content_layers = content_layers
        if style_layers is not None:
            for layer in style_layers:  # check that each layer is valid
                assert layer in range(len(self.features)), \
                    "{} is not in features".format(layer)
            self.style_layers = style_layers
        if content_weights is not None:
            # Check that the number of weights matches the number of layers
            assert len(self.content_layers) == len(content_weights), \
                "content_weights size mismatch"
            self.content_weights = content_weights
        elif self.content_weights is None:
            # Default value: same weight for every layer
            self.content_weights = \
                [1/len(self.content_layers)]*len(self.content_layers)
        if style_weights is not None:
            # Check that the number of weights matches the number of layers
            assert len(self.style_layers) == len(style_weights), \
                "style_weights size mismatch"
            self.style_weights = style_weights
        elif self.style_weights is None:
            # Default value: same weight for every layer
            self.style_weights = \
                [1/len(self.style_layers)]*len(self.style_layers)

        # the weights are normalized
        self.content_weights = [
            w/sum(self.content_weights)
            for w in self.content_weights
        ]
        self.style_weights = [
            w/sum(self.style_weights)
            for w in self.style_weights
        ]

        # Trade-off between content and style loss
        normalization_term = (trade_off+1)
        self.alpha, self.beta = (w/normalization_term for w in (trade_off, 1))

        # We use eval mode
        self.features = self.features.eval()

        # The backward propagation isn't performed on the model parameters
        for param in self.parameters():
            param.requires_grad_(False)

        # Set the device of the model
        self.device = device
        self.to(self.device)

    def forward(self, input):
        """
        Forward pass: take a (1, n_dims, height, width) image and return its
        content and style features as a (content_layers, style_layers) tuple.
        """

        output_style = []
        output_content = []

        # Add custom hooks that append the outputs of desired layers to a list
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

        # Perform the forward pass itself
        self.features(input)

        # Remove the custom hooks
        [handle.remove() for handle in handles]

        return output_content, output_style

    def epoch(
        self,
        target,
        content_targets,
        style_targets,
        optimizer
    ):
        """
        Run a single epoch for the model: forward pass, loss computation and
        back propagation towards the target image. The function returns
        content, style and overall losses.
        """

        # Forward pass
        content_output, style_output = self(target)

        # Compute losses
        curr_content_loss = losses.content(
            self.content_weights,
            content_targets,
            content_output,
        )
        curr_style_loss = losses.style(
            self.style_weights,
            style_targets,
            style_output,
        )
        loss = self.alpha*curr_content_loss + self.beta*curr_style_loss

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(lambda: loss)

        return float(curr_content_loss), float(curr_style_loss), float(loss)

    def get_images_targets(self, content_image, style_image):
        """
        Return the content layers of the content image and the style layers of
        the style image.
        """

        content_targets = [
            f.detach() for f in self(content_image)[0]
        ]
        style_targets = [
            utils.gram_matrix(f).detach() for f in self(style_image)[1]
        ]

        return content_targets, style_targets

    def __repr__(self):
        """
        The representation of the class is redefined to be more concise.
        """

        main_str = self._get_name() + "(\n"

        # Content
        main_str += "  Content(\n"
        for i, layer in enumerate(self.content_layers):
            main_str += f"    ({layer}): "
            main_str += f"{self.features[layer]}, "
            main_str += f"weight={self.content_weights[i]:.3g}\n"
        main_str += f"  ), weight={self.alpha}\n"

        # Style
        main_str += "  Style(\n"
        for i, layer in enumerate(self.style_layers):
            main_str += f"    ({layer}): "
            main_str += f"{self.features[layer]}, "
            main_str += f"weight={self.style_weights[i]:.3g}\n"
        main_str += f"  ), weight={self.beta}\n"

        # Extra
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
            main_str += "  " + "\n  ".join(extra_lines) + "\n"

        main_str += ")"
        return main_str
