"""
Module initially used to define a loss for a multi-scenario autoencoders.
Not used in the end.
"""

import torch
import torch.nn as nn
import numpy as np


def ms_loss(latent, output, params, spectra, msencoder, msdecoder):
    # latent is of shape (batch_size,num_scenarios,num_parameters+1) #the +1 being the degeneracy prediction
    # output is of shape (batch_size,num_scenarios,scale+shape)
    # params is of shape (batch_size,num_parameters)
    # spectra is of shape (batch_size,scale+shape)

    scale_index = 2
    num_scenarios = latent.shape[1]
    degeneracy = torch.mean(
        torch.exp(latent[:, :, -1]) / (1 + torch.exp(latent[:, :, -1])), dim=1
    )
    loss_weights = msencoder.loss_weights

    _shape_loss = (
        nn.MSELoss()(
            output[:, :, scale_index:],
            spectra[:, scale_index:].unsqueeze(dim=1).repeat(1, num_scenarios, 1),
        )
        * loss_weights["shape_loss_weight"]
    )
    _scale_loss = (
        nn.MSELoss()(
            output[:, :, scale_index:],
            spectra[:, scale_index:].unsqueeze(dim=1).repeat(1, num_scenarios, 1),
        )
        * loss_weights["scale_loss_weight"]
    )
    _supervised_loss = (
        nn.MSELoss()(latent[:, 0, :-1], params) * loss_weights["latent_loss_weight"]
    )  # we only train the first scenario (index 0) to match the original params

    count = 0
    inv_distance = 0
    min_dist = (
        0.001  # minimum distance between points to consider them  fully degenerate
    )
    for i in range(
        num_scenarios
    ):  # since we expect ~5 scenarios, we loop over all scenarios pairs as it will be ~10 iterations, not too much. Otherwise, we could randomly sample the pairs if we increase the number of scenarios.
        for j in range(i + 1, num_scenarios):
            count += 1  # we count the number of scenarios pairs. Should be equal to (n-1)n/2.
            inv_distance += 1 / (
                min_dist**2
                + torch.mean((latent[:, i, 2:-1] - latent[:, j, 2:-1]) ** 2, dim=1)
                + (latent[:, i, 0] - latent[:, j, 0]) ** 2
            )  # excluding pressure # for each scenarios pair we compute the differentiation loss as the inverse of the MSE loss (distance) between the atmosphere features of the two scenarios
    inv_distance /= count  # we normalise by the count of pairs

    _proximity_loss = (
        torch.mean(
            (degeneracy * inv_distance + (1 - degeneracy) * loss_weights["cst_dgen"])
        )
        * loss_weights["proximity_loss_weight"]
    )  # the proximity loss is a trade off between having a low degeneracy, and thus paying a constant price high, or having a high degenercay and thus paying the distance cost

    loss = _shape_loss + _scale_loss + _supervised_loss + _proximity_loss
    loss_array = np.array(
        [
            _shape_loss.item(),
            _scale_loss.item(),
            _supervised_loss.item(),
            _proximity_loss.item(),
        ]
    )

    return loss, loss_array
