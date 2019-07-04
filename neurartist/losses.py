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
    """

    mse_loss = torch.nn.MSELoss().to(truth[0].device)

    # Weighted sum of mse between the output layer and the target layer
    return sum([
        weights[i]*mse_loss(layer, truth[i])
        for i, layer in enumerate(pred)
    ])


@_pm.export
def style(weights, truth, pred):
    """
    Compute the standard neural style style loss.
    """

    # Weighted sum of mse between the gram matrices of the output and target
    # layers
    return content(
        weights,
        truth,
        [utils.gram_matrix(layer) for layer in pred]
    )
