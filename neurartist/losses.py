"""
Losses for Neural Style transfer.

@author: gjorando
"""

import torch
from neurartist import utils
from neurartist import _package_manager as _pm


@_pm.export
def content(weights, truth, pred):
    """
    Compute the standard neural style content loss.

    :param weights: Array of weights for each layer.
    :param truth: Content layers of the content image.
    :param pred: Content layers from the neural network on the output.
    :return: The content loss being the weighted sum of MSE between the target
    and the output.
    """

    mse_loss = torch.nn.MSELoss().to(truth[0].device)

    # Weighted sum of mse between the output layer and the target layer
    return sum([
        weights[i]*mse_loss(layer, truth[i])
        for i, layer in enumerate(pred)
    ])


@_pm.export
def style(weights, truth, pred, guidance=None):
    """
    Compute the standard neural style style loss.

    :param weights: Array of weights for each layer.
    :param truth: Gram matrices of each style layer of the style image.
    :param pred: Style layers from the neural network on the output.
    :param guidance: Content guidance channels. If set, guided gram matrixes
    are computed instead, and truth should store the guided gram matrices.
    :return: The style loss being the weighted sum of MSE between the gramian
    of the target and the gramian of the output.
    """

    # Weighted sum of mse between the gram matrices of the output and target
    # layers
    if guidance is None:
        return content(
            weights,
            truth,
            [utils.gram_matrix(layer) for layer in pred]
        )
    else:
        return content(
            weights,
            truth,
            [
                utils.guided_gram_matrix(layer, channel)
                for layer, channel in zip(pred, guidance)
            ]
        )
