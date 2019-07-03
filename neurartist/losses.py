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


@_pm.export
def style(weights, truth, pred):
    """
    Compute the standard neural style style loss.
    """

    return content(
        weights,
        truth,
        [utils.gram_matrix(layer) for layer in pred]
    )
