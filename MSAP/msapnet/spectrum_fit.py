"""
Module developped by Ethan Tregidga for his initial work.
Used for initialization of the autoencoders.
"""

import os
import argparse
import logging as log

import torch
import matplotlib
import numpy as np
from torch.utils.data import DataLoader

from msapnet.utils.utils import open_config
from msapnet.utils.data import data_initialisation
from msapnet.utils.training import training
from msapnet.autoencoder.network import load_network, Network, CombinedDecoder
from msapnet.utils.analysis import (
    param_comparison,
    autoencoder_saliency,
    decoder_saliency,
)
from msapnet.utils.plots import (
    plot_training,
    plot_saliency,
    plot_param_distribution,
    plot_param_comparison,
)


def predict_parameters(config: dict) -> np.ndarray:
    """
    Predicts parameters using the encoder & saves the results to a file

    Parameters
    ----------
    config : dictionary | string, default = '../config.yaml'
        Configuration dictionary or file path to the configuration file

    Returns
    -------
    ndarray
        Spectra names and parameter predictions
    """
    # _, config = open_config(0, config)

    (_, loader), encoder, device = initialization(
        config["net"]["encoder-name"],
        config,
    )[2:]

    output_path = config["path"]["local-path"] + config["path"]["outputs-path"]
    names = []
    params = []
    log_params = loader.dataset.dataset.log_params
    param_transform = loader.dataset.dataset.transform_tg

    encoder.eval()

    # Initialize processes for multiprocessing of encoder loss calculation
    with torch.no_grad():
        for data in loader:
            names.extend(data[-1])
            if len(data[2]):
                param_bat = encoder(
                    torch.cat([data[2].to(device), data[0].to(device)], dim=1)
                ).cpu()
            else:
                param_bat = encoder(data[0].to(device)).cpu()
            # Transform parameters
            param_batch = param_transform.inverse(param_bat)
            params.append(param_batch.clone())

    output = np.hstack((np.expand_dims(names, axis=1), torch.cat(params).numpy()))
    np.savetxt(
        output_path
        + f'parameter_predictions_{config["net"]["version"]}.{config["net"]["iteration-save"]}.csv',
        output,
        delimiter=",",
        fmt="%s",
    )

    return output


def initialization(
    name: str,
    config: dict,
    transform_ft=None,
    transform_tg=None,
    transform_au=None,
    transform_no=None,
) -> tuple[
    int,
    tuple[list, list],
    tuple[DataLoader, DataLoader],
    Network,
    torch.device,
]:
    """
    Trains & validates network, used for progressive learning

    Parameters
    ----------
    name : string
        Name of the network
    config : dictionary | string, default = '../config.yaml'
        Configuration dictionary or file path to the configuration file
    transform : list[list[ndarray]], default = None
        Min and max spectral range and mean & standard deviation of parameters

    Returns
    -------
    tuple[integer, tuple[list, list], tuple[DataLoader, DataLoader], Network, device]
        Initial epoch; train & validation losses; train & validation dataloaders; network; & device
    """

    # Constants
    initial_epoch = 0
    losses = ([], [])

    if "encoder" in name.lower():
        network_type = "encoder"
    elif "decoder" in name.lower():
        network_type = "decoder"
    else:
        raise NameError(
            f"Unknown network type: {name}\n"
            f"Make sure encoder or decoder is included in the name"
        )

    # Load config parameters
    load_name = f"{config['net']['version']}.{config['net']['iteration-load']}"
    num_params = config["model"]["parameters-number"]
    learning_rate = config["training"]["learning-rate"]
    networks_dir = config["path"]["local-path"] + config["path"]["network-config-path"]
    spectra_path = config["path"]["local-path"] + config["path"]["data-path"]
    params_path = config["path"]["local-path"] + config["path"]["data-path"]
    aux_path = config["path"]["local-path"] + config["path"]["data-path"]
    noise_path = config["path"]["local-path"] + config["path"]["data-path"]
    names_path = ""
    states_dir = f"{config['path']['local-path']}{config['path']['states-path']}"
    log_params = config["model"]["log-parameters"]

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if device == "cuda" else {}

    if os.path.exists(f"{states_dir}{name}_{load_name}.pth"):
        try:
            state = torch.load(
                f"{states_dir}{name}_{load_name}.pth", map_location=device
            )
            indices = state["indices"]
            transform_ft = state["transform_ft"]
            transform_tg = state["transform_tg"]
        except FileNotFoundError:
            log.warning(
                f"{states_dir}{name}_{load_name}.pth does not exist\n"
                f"No state will be loaded"
            )
            load_name = 0
            indices = None
    else:
        indices = None

    # See if we have auxiliary variables:
    if transform_au is None:
        aux_path = None
    else:
        aux_path = f"{aux_path}aux_{config['data']['run-id']}.pt"

    if transform_no is None:
        noise_path = None
    else:
        noise_path = f"{noise_path}noises_{config['data']['run-id']}.pt"

    # Initialize datasets
    loaders = data_initialisation(
        f"{spectra_path}features_{config['data']['run-id']}.pt",
        f"{params_path}targets_{config['data']['run-id']}.pt",
        aux_path,
        noise_path,
        log_params,
        kwargs,
        names_path=names_path,
        transform_ft=transform_ft,
        transform_tg=transform_tg,
        transform_au=transform_au,
        transform_no=transform_no,
        indices=indices,
        device=device,
    )

    # See if we have auxiliary variables:
    if transform_au is None:
        aux_size = 0
    else:
        aux_size = loaders[0].dataset[0][2].size(0)

    if "combined" in name.lower():
        decoder_shape = Network(
            loaders[0].dataset[0][0].size(0),
            num_params,
            learning_rate,
            f"Decoder_Shape v{config['net']['version']}",
            networks_dir,
            aux_size,
        )
        decoder_scale = Network(
            loaders[0].dataset[0][0].size(0),
            num_params,
            learning_rate,
            f"Decoder_Scale v{config['net']['version']}",
            networks_dir,
            aux_size,
        )
        network = CombinedDecoder(
            decoder_shape=decoder_shape, decoder_scale=decoder_scale
        )
    else:
        # Initialize network
        network = Network(
            loaders[0].dataset[0][0].size(0),
            num_params,
            learning_rate,
            f"{name} v{config['net']['version']}",
            networks_dir,
            aux_size=aux_size,
        ).to(device)

    # Load states from previous training
    if os.path.exists(f"{states_dir}{name}_{load_name}.pth"):
        print("Loading previously trained network")
        initial_epoch, network, losses = load_network(
            load_name,
            states_dir,
            network,
        )

    return initial_epoch, losses, loaders, network, device
