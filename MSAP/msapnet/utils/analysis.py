"""
Calculates the saliency of decoders or autoencoders using backpropagation.
Developped by Ethan Tregidga but not used in this work.
"""

import torch
import numpy as np
from numpy import ndarray
from torch.utils.data import DataLoader

from msapnet.autoencoder.network import Network
from msapnet.utils.utils import open_config, file_names


def autoencoder_saliency(
    loader: DataLoader, device: torch.device, encoder: Network, decoder: Network
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Calculates the importance of each part of the input spectrum on the output spectrum
    by calculating the saliency using backpropagation of the autoencoder

    Parameters
    ----------
    loader : DataLoader
        Autoencoder validation data loader
    device : device
        Which device type PyTorch should use
    encoder : Network
        Encoder half of the network
    decoder : Network
        Decoder half of the network

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        Original spectra, output, and saliency
    """
    # Constants
    spectra = next(iter(loader))[0][:8].to(device)

    # Initialization
    encoder.train()
    spectra.requires_grad_()

    # Calculate saliency through backpropagation
    output = decoder(encoder(spectra))
    loss = torch.nn.MSELoss()(output, spectra)
    loss.backward()
    saliency = spectra.grad.data.abs().cpu()

    return (
        spectra.detach().cpu().numpy(),
        output.detach().cpu().numpy(),
        saliency.numpy(),
    )


def decoder_saliency(loader: DataLoader, device: torch.device, decoder: Network):
    """
    Calculates the importance of each parameter on the output spectrum
    by calculating the saliency using backpropagation of the decoder

    Parameters
    ----------
    loader : DataLoader
        Decoder validation data loader
    device : device
        Which device type PyTorch should use
    decoder : Network
        Decoder half of the network
    """
    # Constants
    d_spectra, d_parameters, d_aux, d_noise, *_ = next(iter(loader))

    # Initialization
    decoder.train()
    d_spectra = d_spectra.to(device)
    if len({}):
        d_parameters = torch.cat([d_aux, d_parameters], dim=1).to(device)
    d_parameters.requires_grad_()

    # Calculate saliency through backpropagation

    d_output = decoder(d_parameters)
    d_loss = torch.nn.MSELoss()(d_output, d_spectra)
    d_loss.backward()
    d_saliency = d_parameters.grad.data.abs().cpu()

    # Measure impact of input parameters on decoder output
    parameter_saliency = torch.mean(d_saliency, dim=0)
    parameter_impact = parameter_saliency / torch.min(parameter_saliency)
    parameter_std = torch.std(d_saliency, dim=0) / torch.min(parameter_saliency)

    print(
        f"\nParameter impact on decoder:\n{parameter_impact.tolist()}"
        f"\nParameter spread:\n{parameter_std.tolist()}\n"
    )


def param_comparison(config: dict) -> tuple[ndarray, ndarray]:
    """
    Gets and transforms data to compare parameter values

    Parameters
    ----------
    config : dictionary | string, default = '../config.yaml'
        Configuration dictionary or file path to the configuration file

    Returns
    -------
    tuple[ndarray, ndarray]
        Target parameters and parameter predictions
    """
    blacklist = ["bkg", ".bg", ".rmf", ".arf"]
    # _, config = open_config(0, config)

    # Load config parameters
    names_path = ""
    target_path = f"{config['path']['local-path']}{config['path']['data-path']}targets_{config['data']['run-id']}.pt"
    prediction_path = config["path"]["local-path"] + config["path"]["outputs-path"]
    log_params = config["model"]["log-parameters"]
    target = torch.load(target_path).detach().cpu().numpy()

    if names_path:
        spectra_names = np.load(names_path)
    else:
        # spectra_names = file_names(config['data']['spectra-directory'], blacklist=blacklist)
        spectra_names = np.arange(target.shape[0], dtype=float).astype(str)

    predictions = np.loadtxt(
        prediction_path
        + f'parameter_predictions_{config["net"]["version"]}.{config["net"]["iteration-save"]}.csv',
        delimiter=",",
        dtype=str,
    )

    # Sort target spectra by name
    sort_idx = np.argsort(spectra_names)
    spectra_names = spectra_names[sort_idx]
    target = target[sort_idx]

    # Filter target parameters using spectra that was predicted and log parameters
    target_idx = np.searchsorted(spectra_names, predictions[:, 0])
    target = target[target_idx]
    predictions = predictions[:, 1:].astype(float)

    return target, predictions
