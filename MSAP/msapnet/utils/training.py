"""
Trains the network and evaluates the performance. It is composed of several parts corresponding to several machine learning models:
    - SEMI SUPERVISED AUTOENCODERS
    - NORMALISING FLOWS based on SBI python package
    - NORMALISING FLOWS based on zuko python package
    - DIRECT CDF 
"""
import os
import subprocess

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from copy import deepcopy
import gc

from sbi.inference import SNPE_C, SNLE
from sbi.inference.snpe.snpe_base import *
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
from time import time

from msapnet.autoencoder.network import Network
from msapnet.utils.transform import Transform
from msapnet.utils.transform import *
from msapnet.ms_loss import ms_loss
from msapnet.inference.direct_cdf import (
    DirectCDFfromNN,
    cdf_sum_of_sigm_and_linear_piecewise,
)


####### TRAINING FOR AUTOENCODER ########


def train_val(
    device: torch.device,
    loader: DataLoader,
    cnn: Network,
    train: bool = True,
    surrogate: Network = None,
    ae_training: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Trains the encoder or decoder using cross entropy or mean squared error

    Parameters
    ----------
    device : device
        Which device type PyTorch should use
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    cnn : Network
        CNN to use for training/validation
    train : bool, default = True
        If network should be trained or validated
    surrogate : Network, defualt = None
        Surrogate network for encoder training

    Returns
    -------
    tuple[float, ndarray, ndarray]
        Average loss value, spectra & reconstructions
    """
    epoch_loss = 0.0

    epoch_shape_loss = 0.0
    epoch_scale_loss = 0.0
    epoch_latent_loss = 0.0
    epoch_sum_to_1_loss = 0.0

    list_spectra = []
    list_outputs = []
    list_aux = []
    list_noise = []
    list_params = []

    if train:
        cnn.train()

        if surrogate:
            surrogate.eval()
            if ae_training:
                surrogate.train()
    else:
        cnn.eval()

        if surrogate:
            surrogate.eval()

    with torch.set_grad_enabled(train):
        for spectra, params, aux, noise, *_ in loader:
            spectra = spectra.to(device)

            # If surrogate is not none, train encoder with surrogate
            if surrogate:
                if aux.nelement():  # if auxiliary variables, we use them
                    latent = cnn(torch.cat([aux.to(device), spectra], dim=1))
                    output = surrogate(torch.cat([aux, latent], dim=1))
                else:
                    latent = cnn(spectra)
                    output = surrogate(latent)
                target = spectra

                _latent_loss = cnn.loss_weights["latent_loss_weight"] * nn.MSELoss()(
                    latent, params
                )
                epoch_latent_loss += _latent_loss.item()
                try:  # if possible, we weight the scale loss (index=0) in the loss
                    _scale_loss = (
                        nn.MSELoss()(
                            output[:, : surrogate.truncate_index],
                            target[:, : surrogate.truncate_index],
                        )
                        * cnn.loss_weights["scale_loss_weight"]
                    )
                    _shape_loss = nn.MSELoss()(
                        output[:, surrogate.truncate_index :],
                        target[:, surrogate.truncate_index :],
                    )
                    epoch_scale_loss += _scale_loss.item()
                    epoch_shape_loss += _shape_loss.item()
                except AttributeError:
                    print("Error on the scale loss")
                    _scale_loss = 0.0
                    _shape_loss = (
                        nn.MSELoss()(output, target)
                        * cnn.loss_weights["shape_loss_weight"]
                    )
                    epoch_shape_loss += _shape_loss.item()

                try:  # if possible, we try to have a sum of abundances equal to 1
                    _sum_to_1_loss = cnn.loss_weights[
                        "sum_to_1_loss_weight"
                    ] * nn.MSELoss()(
                        torch.sum(
                            loader.dataset.dataset.transform_tg.inverse(latent)[:, 2:],
                            dim=1,
                        ),
                        torch.ones(latent.shape[0]),
                    )
                    epoch_sum_to_1_loss += _sum_to_1_loss.item()
                except AttributeError:
                    _sum_to_1_loss = 0.0

                loss = _shape_loss + _scale_loss + _latent_loss + _sum_to_1_loss

            else:
                params = params.to(device)
                # Train encoder with supervision or decoder
                if cnn.encoder:
                    if aux.nelement():  # if auxiliary variables, we use them
                        output = cnn(torch.cat([aux.to(device), spectra], dim=1))
                    else:
                        output = cnn(spectra)
                    target = params
                else:
                    if aux.nelement():  # if auxiliary variables, we use them
                        output = cnn(torch.cat([aux.to(device), params], dim=1))
                    else:
                        output = cnn(params)
                    target = spectra

                loss = nn.MSELoss()(output, target)

            if train:
                # Optimise CNN
                cnn.optimizer.zero_grad()
                if (
                    ae_training
                ):  # if in auto encoder training, optimise also the surrogfate (decoder)
                    surrogate.optimizer.zero_grad()
                loss.backward()
                if ae_training:
                    surrogate.optimizer.step()
                cnn.optimizer.step()
            else:
                list_spectra.append(spectra.clone())
                list_outputs.append(output.clone())
                list_params.append(params.clone())
                list_aux.append(aux.clone())
                list_noise.append(noise.clone())

            epoch_loss += loss.item()

    if train:
        return (
            epoch_loss / len(loader),
            np.array(
                [
                    epoch_shape_loss,
                    epoch_scale_loss,
                    epoch_latent_loss,
                    epoch_sum_to_1_loss,
                ]
            )
            / epoch_loss,
            spectra.cpu().numpy(),
            output.detach().cpu().numpy(),
            params.detach().cpu().numpy(),
            aux.detach().cpu().numpy(),
            noise.detach().cpu().numpy(),
        )
    return (
        epoch_loss / len(loader),
        np.array(
            [epoch_shape_loss, epoch_scale_loss, epoch_latent_loss, epoch_sum_to_1_loss]
        )
        / epoch_loss,
        torch.cat(list_spectra).cpu().numpy(),
        torch.cat(list_outputs).detach().cpu().numpy(),
        torch.cat(list_params).detach().cpu().numpy(),
        torch.cat(list_aux).detach().cpu().numpy(),
        torch.cat(list_noise).detach().cpu().numpy(),
    )


def gtrain_val(
    device: torch.device,
    loader: DataLoader,
    cnn: Network,
    train: bool = True,
    surrogate: Network = None,  # Not Implemented yet
    ae_training: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Trains the encoder or decoder using cross entropy or mean squared error

    Parameters
    ----------
    device : device
        Which device type PyTorch should use
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    cnn : Network
        CNN to use for training/validation
    train : bool, default = True
        If network should be trained or validated
    surrogate : Network, defualt = None
        Surrogate network for encoder training

    Returns
    -------
    tuple[float, ndarray, ndarray]
        Average loss value, spectra & reconstructions
    """
    epoch_loss = 0.0

    list_spectra = []
    list_outputs = []
    list_aux = []
    list_noise = []
    list_params = []

    if train:
        cnn.train()

        if surrogate:
            surrogate.eval()
            if ae_training:
                surrogate.train()
    else:
        cnn.eval()

        if surrogate:
            surrogate.eval()

    with torch.set_grad_enabled(train):
        for spectra, params, aux, noise, *_ in loader:
            spectra = spectra.to(device)
            noise = noise.to(device)

            # TODO: implement the gtrain for the whole autoencoder
            if surrogate:
                raise NotImplementedError
            else:
                params = params.to(device)
                # Train encoder with supervision or decoder
                if cnn.encoder:
                    raise NotImplementedError
                else:
                    if aux.nelement():  # if auxiliary variables, we use them
                        output = cnn(torch.cat([aux.to(device), params], dim=1))
                    else:
                        output = cnn(params)
                    target = spectra

                covariance = output[:, 1] ** 2 + noise**2
                ns = noise + 1e-8 * (noise < 1e-8)
                loss = nn.GaussianNLLLoss(eps=1e-8)(
                    output[:, 0], target, covariance
                ) - torch.mean(torch.log(ns))

            if train:
                # Optimise CNN
                cnn.optimizer.zero_grad()
                # if ae_training: # if in auto encoder training, optimise also the surrogfate (decoder)
                #     surrogate.optimizer.zero_grad()
                loss.backward()
                # if ae_training:
                #     surrogate.optimizer.step()
                cnn.optimizer.step()
            else:
                list_spectra.append(spectra.clone())
                list_outputs.append(output.clone())
                list_params.append(params.clone())
                list_aux.append(aux.clone())
                list_noise.append(noise.clone())

            epoch_loss += loss.item()

    if train:
        return (
            epoch_loss / len(loader),
            np.array([]) / epoch_loss,
            spectra.cpu().numpy(),
            output.detach().cpu().numpy(),
            params.detach().cpu().numpy(),
            aux.detach().cpu().numpy(),
            noise.detach().cpu().numpy(),
        )
    return (
        epoch_loss / len(loader),
        np.array([]) / epoch_loss,
        torch.cat(list_spectra).cpu().numpy(),
        torch.cat(list_outputs).detach().cpu().numpy(),
        torch.cat(list_params).detach().cpu().numpy(),
        torch.cat(list_aux).detach().cpu().numpy(),
        torch.cat(list_noise).detach().cpu().numpy(),
    )


def training(
    epochs: tuple[int, int],
    loaders: tuple[DataLoader, DataLoader],
    cnn: Network,
    device: torch.device,
    save_name: str,
    states_dir: str = None,
    losses: tuple[list, list] = None,
    surrogate: Network = None,
    latent_loss_weight_scheduler=None,
    ae_training: bool = False,
    gtrain: bool = False,
) -> tuple[tuple[list, list], np.ndarray, np.ndarray]:
    """
    Trains & validates the network for each epoch

    Parameters
    ----------
    epochs : tuple[integer, integer]
        Initial epoch & number of epochs to train
    loaders : tuple[DataLoader, DataLoader]
        Train and validation dataloaders
    cnn : Network
        CNN to use for training
    device : device
        Which device type PyTorch should use
    save_name : integer, default = 0
        The file number to save the new state, if 0, nothing will be saved
    states_dir : string, default = None
        Path to the folder where the network state will be saved, not needed if save_name = 0
    losses : tuple[list, list], default = ([], [])
        Train and validation losses for each epoch, can be empty
    surrogate : Network, default = None
        Surrogate network to use for training

    Returns
    -------
    tuple[tuple[list, list], ndarray, ndarray]
        Train & validation losses, spectra & reconstructions
    """
    if not losses:
        losses = ([], [])

    if gtrain:
        t = gtrain_val
    else:
        t = train_val

    losses_compositions_dict = ({}, {})
    losses_compositions = ([], [])

    # Train for each epoch
    epoch_bar = tqdm(range(*epochs), desc=f"Training {cnn.name.split()[0]}_{save_name}")
    epoch_bar.set_postfix({"Tloss": f"{0:.2e}", "Vloss": f"{0:.2e}"})
    for epoch in epoch_bar:
        # t_initial = time()
        epoch += 1

        # Train CNN
        full_loss_train, _loss_comp_train = t(
            device, loaders[0], cnn, surrogate=surrogate, ae_training=ae_training
        )[0:2]
        losses[0].append(full_loss_train)
        losses_compositions[0].append(_loss_comp_train)

        # Validate CNN
        full_loss_val, _loss_comp_val = t(
            device, loaders[1], cnn, train=False, surrogate=surrogate
        )[0:2]
        losses[1].append(full_loss_val)
        losses_compositions[1].append(_loss_comp_val)

        # Check the validation loss consistency
        # losses[2].append(train_val(device, loaders[0], cnn, train=False, surrogate=surrogate)[0])

        cnn.scheduler.step(losses[1][-1])

        # Progressivly reducing the latent loss weight
        if latent_loss_weight_scheduler is not None:
            cnn.latent_loss_weight = latent_loss_weight_scheduler(
                cnn.latent_loss_weight, epoch
            )

        # Save training progress
        if save_name and (epoch % 10 == 0 or epoch == epochs[1]):
            state = {
                "epoch": epoch,
                "transform_ft": loaders[0].dataset.dataset.transform_ft,
                "transform_tg": loaders[0].dataset.dataset.transform_tg,
                "train_loss": losses[0],
                "val_loss": losses[1],
                "indices": loaders[0].dataset.dataset.indices,
                "state_dict": cnn.state_dict(),
                "optimizer": cnn.optimizer.state_dict(),
                "scheduler": cnn.scheduler.state_dict(),
            }

            torch.save(state, f"{states_dir}{cnn.name.split()[0]}_{save_name}.pth")

            if ae_training:
                state["state_dict"] = surrogate.state_dict()
                state["optimizer"] = surrogate.optimizer.state_dict()
                state["scheduler"] = surrogate.scheduler.state_dict()

                torch.save(
                    state, f"{states_dir}{surrogate.name.split()[0]}_{save_name}.pth"
                )

        # print(f'Epoch [{epoch}/{epochs[1]}]\t'
        #       f'Training loss: {losses[0][-1]:.3e}\t'
        #       f'Validation loss: {losses[1][-1]:.3e}\t'
        #       f'Time: {time() - t_initial:.1f}')

        if epoch > 5:
            epoch_bar.set_postfix(
                {
                    "Tloss": f"{losses[0][-1]:.2e}",
                    "Vloss": f"{losses[1][-1]:.2e}",
                    "m5VL": f"{np.mean(losses[1][-5:]):.2e}",
                }
            )
        else:
            epoch_bar.set_postfix(
                {"Tloss": f"{losses[0][-1]:.2e}", "Vloss": f"{losses[1][-1]:.2e}"}
            )

    # Final validation
    loss, _loss_comp, spectra, outputs, params, auxs, noises = t(
        device, loaders[1], cnn, train=False, surrogate=surrogate
    )

    losses[1].append(loss)
    # losses_compositions[1].append(_loss_comp)
    print(f"Final validation loss: {losses[1][-1]:.3e}\n")

    losses_compositions = np.array(losses_compositions)
    if (
        losses_compositions.shape[2] > 0
    ):  # check that we have some loss composition, in the case of gtrain we don't have any
        losses_compositions_dict[0]["Shape loss training"] = losses_compositions[0][
            :, 0
        ]
        losses_compositions_dict[0]["Scale loss training"] = losses_compositions[0][
            :, 1
        ]
        losses_compositions_dict[0]["Sum to 1 loss training"] = losses_compositions[0][
            :, 3
        ]
        losses_compositions_dict[0]["Supervised loss training"] = losses_compositions[
            0
        ][:, 2]

        losses_compositions_dict[1]["Shape loss validation"] = losses_compositions[1][
            :, 0
        ]
        losses_compositions_dict[1]["Scale loss validation"] = losses_compositions[1][
            :, 1
        ]
        losses_compositions_dict[1]["Sum to 1 loss validation"] = losses_compositions[
            1
        ][:, 3]
        losses_compositions_dict[1]["Supervised loss validation"] = losses_compositions[
            1
        ][:, 2]

    return losses, spectra, outputs, losses_compositions_dict, params, auxs, noises


def save_loss(
    cnn: Network,
    device: torch.device,
    config: dict,
    transform_ft,
    transform_tg,
    save_name: str,
    is_decoder: bool = True,
    save: bool = False,
    transform_au=None,
):
    """Save the losses for each spectrum
    ## To be improved to only compute the loss on the validation set

    Args:
        cnn (Network): the model to use
        device (torch.device): the device
        config (dict): the configuration file containing all information
        is_decoder (bool, optional): if the model is a decoder. Defaults to True.
        save_name (int, optional): the id of the loss. Defaults to 0.

    Returns:
        _type_: _description_
    """

    if is_decoder:
        cnn_type = "decoder"
    else:
        cnn_type = "encoder"

    features = torch.load(
        f"{config['path']['local-path']+config['path']['data-path']}features_{config['data']['run-id']}.pt"
    )  # load the spectra
    targets = torch.load(
        f"{config['path']['local-path']+config['path']['data-path']}targets_{config['data']['run-id']}.pt"
    )  # load the parameters
    if transform_au is not None:
        aux = torch.load(
            f"{config['path']['local-path']+config['path']['data-path']}aux_{config['data']['run-id']}.pt"
        )

    if is_decoder:
        with torch.no_grad():
            if transform_au is not None:
                outputs = cnn(
                    torch.cat([transform_au(aux), transform_tg(targets)]).to(device)
                )
            else:
                outputs = cnn(
                    transform_tg(targets.clone().to(device))
                )  # predict the spectra with the decoder
            ft = transform_ft(features.clone())
            losses = torch.mean(
                torch.pow(ft - outputs, 2), dim=1
            )  # compute the loss for each spectrum
            # print(f"{torch.nn.MSELoss()(ft,outputs)} and {losses.mean()} and {torch.mean(torch.pow(ft-outputs,2))}")
        if save:
            torch.save(
                losses,
                f"{config['path']['local-path']}{config['path']['losses-path']}losses_{cnn_type}_{save_name}.pt",
            )  # save the loss in a file

    return losses


def save_loss_ft(
    decoder: Network,
    config: dict,
    transform_ft: Transform,
    transform_tg: Transform,
    q=0.1,
    device: torch.device = torch.device("cpu"),
    save=False,
    transform_au: Transform = None,
):
    """Computes and save the mean loss for each feature
    ## To be improved to only compute the loss on the validation set

    Args:
        decoder (Network): _description_
        config (dict): _description_
        transform_ft (Transform): _description_
        transform_tg (Transform): _description_
        device (torch.device, optional): _description_. Defaults to torch.device('cpu').

    Returns:
        _type_: _description_
    """

    features = torch.load(
        f"{config['path']['local-path']+config['path']['data-path']}features_{config['data']['run-id']}.pt"
    )  # load the spectra
    targets = torch.load(
        f"{config['path']['local-path']+config['path']['data-path']}targets_{config['data']['run-id']}.pt"
    )  # load the parameters
    if transform_au is not None:
        aux = torch.load(
            f"{config['path']['local-path']+config['path']['data-path']}aux_{config['data']['run-id']}.pt"
        )

    with torch.no_grad():
        if transform_au is not None:
            outputs = decoder(
                torch.cat([transform_au(aux), transform_tg(targets)]).to(device)
            )
        else:
            outputs = decoder(transform_tg(targets.clone().to(device)))
        ft = transform_ft(features.clone())
        l = torch.pow(ft - outputs, 2)
        losses = torch.mean(l, dim=0)
        losses_q1 = torch.quantile(l, q=q, dim=0)
        losses_q2 = torch.quantile(l, q=1 - q, dim=0)
    if save:
        torch.save(
            losses,
            f"{config['path']['local-path']}{config['path']['losses-path']}ft_losses_{config['net']['version']}{config['net']['iteration-save']}.pt",
        )  # save the loss in a file
        torch.save(
            losses_q1,
            f"{config['path']['local-path']}{config['path']['losses-path']}ft_losses_q{q}_{config['net']['version']}{config['net']['iteration-save']}.pt",
        )
        torch.save(
            losses_q2,
            f"{config['path']['local-path']}{config['path']['losses-path']}ft_losses_q{1-q}_{config['net']['version']}{config['net']['iteration-save']}.pt",
        )

    return losses, losses_q1, losses_q2


def ms_train_val(
    device: torch.device,
    loader: DataLoader,
    msencoder: Network,
    train: bool = True,
    msdecoder: Network = None,
    ae_training: bool = False,
    loss_array_length: int = 4,
) -> tuple[float, np.ndarray, np.ndarray]:
    """ """
    epoch_loss = 0.0

    epoch_loss_array = np.zeros(loss_array_length)

    list_spectra = []
    list_outputs = []
    list_params = []
    list_aux = []
    list_noise = []
    list_latent = []

    if train:
        msencoder.train()

        msdecoder.eval()
        if ae_training:
            msdecoder.train()
    else:
        msencoder.eval()
        msdecoder.eval()

    with torch.set_grad_enabled(train):
        for spectra, params, aux, noise, *_ in loader:
            spectra = spectra.to(device)

            if aux.nelement():  # if auxiliary variables, we use them
                latent = msencoder(torch.cat([aux.to(device), spectra], dim=1))
                output = msdecoder(torch.cat([aux.to(device), latent], dim=1))
            else:
                latent = msencoder(spectra)
                output = msdecoder(latent)
            target = spectra

            loss, _loss_array = ms_loss(
                latent, output, params, spectra, msencoder, msdecoder
            )

            if train:
                # Optimise CNN
                msencoder.optimizer.zero_grad()
                if (
                    ae_training
                ):  # if in auto encoder training, optimise also the surrogate (decoder)
                    msdecoder.optimizer.zero_grad()
                loss.backward()
                if ae_training:
                    msdecoder.optimizer.step()
                msencoder.optimizer.step()
            else:
                list_spectra.append(spectra.clone())
                list_outputs.append(output.clone())
                list_params.append(params.clone())
                list_latent.append(latent.clone())
                list_aux.append(aux.clone())
                list_noise.append(noise.clone())

            epoch_loss += loss.item()
            epoch_loss_array += _loss_array

    if train:
        return (
            epoch_loss / len(loader),
            epoch_loss_array / epoch_loss,
            spectra.cpu().numpy(),
            output.detach().cpu().numpy(),
            params.detach().cpu().numpy(),
            aux.detach().cpu().numpy(),
            noise.detach().cpu().numpy(),
            latent.detach().cpu().numpy(),
        )
    return (
        epoch_loss / len(loader),
        epoch_loss_array / epoch_loss,
        torch.cat(list_spectra).cpu().numpy(),
        torch.cat(list_outputs).detach().cpu().numpy(),
        torch.cat(list_params).detach().cpu().numpy(),
        torch.cat(list_aux).detach().cpu().numpy(),
        torch.cat(list_noise).detach().cpu().numpy(),
        torch.cat(list_latent).detach().cpu().numpy(),
    )


def mstraining(
    epochs: tuple[int, int],
    loaders: tuple[DataLoader, DataLoader],
    msencoder: Network,
    device: torch.device,
    save_name: str,
    states_dir: str = None,
    losses: tuple[list, list] = None,
    msdecoder: Network = None,
    latent_loss_weight_scheduler=None,
    ae_training: bool = False,
) -> tuple[tuple[list, list], np.ndarray, np.ndarray]:
    """ """
    if not losses:
        losses = ([], [])

    losses_compositions_dict = ({}, {})
    losses_compositions = ([], [])

    # Train for each epoch
    epoch_bar = tqdm(
        range(*epochs), desc=f"MSTraining {msencoder.name.split()[0]}_{save_name}"
    )
    epoch_bar.set_postfix({"Tloss": f"{0:.2e}", "Vloss": f"{0:.2e}"})
    for epoch in epoch_bar:
        # t_initial = time()
        epoch += 1

        # Train CNN
        full_loss_train, _loss_comp_train = ms_train_val(
            device, loaders[0], msencoder, msdecoder=msdecoder, ae_training=ae_training
        )[0:2]
        losses[0].append(full_loss_train)
        losses_compositions[0].append(_loss_comp_train)

        # Validate CNN
        full_loss_val, _loss_comp_val = ms_train_val(
            device, loaders[1], msencoder, train=False, msdecoder=msdecoder
        )[0:2]
        losses[1].append(full_loss_val)
        losses_compositions[1].append(_loss_comp_val)

        # Check the validation loss consistency
        # losses[2].append(train_val(device, loaders[0], cnn, train=False, surrogate=surrogate)[0])

        msencoder.scheduler.step(losses[1][-1])

        # Progressivly reducing the latent loss weight
        if latent_loss_weight_scheduler is not None:
            msencoder.latent_loss_weight = latent_loss_weight_scheduler(
                msencoder.latent_loss_weight, epoch
            )

        # Save training progress
        if save_name and (epoch % 10 == 0 or epoch == epochs[1]):
            state = {
                "epoch": epoch,
                "transform_ft": loaders[0].dataset.dataset.transform_ft,
                "transform_tg": loaders[0].dataset.dataset.transform_tg,
                "train_loss": losses[0],
                "val_loss": losses[1],
                "indices": loaders[0].dataset.dataset.indices,
                "state_dict": msencoder.state_dict(),
                "optimizer": msencoder.optimizer.state_dict(),
                "scheduler": msencoder.scheduler.state_dict(),
            }

            torch.save(
                state, f"{states_dir}{msencoder.name.split()[0]}_{save_name}.pth"
            )

            if ae_training:
                state["state_dict"] = msdecoder.state_dict()
                state["optimizer"] = msdecoder.optimizer.state_dict()
                state["scheduler"] = msdecoder.scheduler.state_dict()

                torch.save(
                    state, f"{states_dir}{msdecoder.name.split()[0]}_{save_name}.pth"
                )

        # print(f'Epoch [{epoch}/{epochs[1]}]\t'
        #       f'Training loss: {losses[0][-1]:.3e}\t'
        #       f'Validation loss: {losses[1][-1]:.3e}\t'
        #       f'Time: {time() - t_initial:.1f}')

        if epoch > 5:
            epoch_bar.set_postfix(
                {
                    "Tloss": f"{losses[0][-1]:.2e}",
                    "Vloss": f"{losses[1][-1]:.2e}",
                    "m5VL": f"{np.mean(losses[1][-5:]):.2e}",
                }
            )
        else:
            epoch_bar.set_postfix(
                {"Tloss": f"{losses[0][-1]:.2e}", "Vloss": f"{losses[1][-1]:.2e}"}
            )

    # Final validation
    loss, _loss_comp, spectra, outputs, params, auxs, noises, latents = ms_train_val(
        device, loaders[1], msencoder, train=False, msdecoder=msdecoder
    )
    losses[1].append(loss)
    # losses_compositions[1].append(_loss_comp)
    print(f"Final validation loss: {losses[1][-1]:.3e}\n")

    losses_compositions = np.array(losses_compositions)
    print(losses_compositions.shape)
    losses_compositions_dict[0]["Shape loss training"] = losses_compositions[0][:, 0]
    losses_compositions_dict[0]["Scale loss training"] = losses_compositions[0][:, 1]
    losses_compositions_dict[0]["Proximity loss training"] = losses_compositions[0][
        :, 3
    ]
    losses_compositions_dict[0][
        "Supervised loss (first scenario only) training"
    ] = losses_compositions[0][:, 2]

    losses_compositions_dict[1]["Shape loss validation"] = losses_compositions[1][:, 0]
    losses_compositions_dict[1]["Scale loss validation"] = losses_compositions[1][:, 1]
    losses_compositions_dict[1]["Proximity loss validation"] = losses_compositions[1][
        :, 3
    ]
    losses_compositions_dict[1][
        "Supervised loss (first scenario only) validation"
    ] = losses_compositions[1][:, 2]

    return (
        losses,
        spectra,
        outputs,
        losses_compositions_dict,
        params,
        auxs,
        noises,
        latents,
    )


####### TRAINING FOR SBI NORMALISING FLOWS ########


def forward_kld(flow, targets, context):
    """
    Defines the KL divergence loss for a flow with context and with a target distribution sampled in targets_samples.
    See https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/core.py#L87
    """

    y, log_det = flow._transform(
        inputs=targets, context=context
    )  # we don't put .inverse because nflow convention is different from normflows.
    return -(log_det + flow._distribution.log_prob(y))


def train_nsf(
    SNPE,
    training_batch_size: int = 500,
    learning_rate: float = 5e-4,
    validation_fraction: float = 0.1,
    stop_after_epochs: int = 20,
    max_num_epochs: int = 400,
    batch_fraction: float = 1.0,
    show_progress_bar: bool = True,
    use_kl_loss: bool = False,
    load_sequentially: bool = False,
    clip_max_norm: Optional[float] = 5.0,
    calibration_kernel: Optional[Callable] = None,
    resume_training: bool = False,
    force_first_round_loss: bool = False,
    discard_prior_samples: bool = False,
    retrain_from_scratch: bool = False,
    show_train_summary: bool = False,
    dataloader_kwargs: Optional[dict] = None,
) -> nn.Module:
    r"""Return density estimator that approximates the distribution $p(\theta|x)$.

    Args:
        training_batch_size: Training batch size.
        learning_rate: Learning rate for Adam optimizer.
        validation_fraction: The fraction of data to use for validation.
        stop_after_epochs: The number of epochs to wait for improvement on the
            validation set before terminating training.
        max_num_epochs: Maximum number of epochs to run. If reached, we stop
            training even when the validation loss is still decreasing. Otherwise,
            we train until validation loss increases (see also `stop_after_epochs`).
        clip_max_norm: Value at which to clip the total gradient norm in order to
            prevent exploding gradients. Use None for no clipping.
        calibration_kernel: A function to calibrate the loss with respect to the
            simulations `x`. See Lueckmann, Gonçalves et al., NeurIPS 2017.
        resume_training: Can be used in case training time is limited, e.g. on a
            cluster. If `True`, the split between train and validation set, the
            optimizer, the number of epochs, and the best validation log-prob will
            be restored from the last time `.train()` was called.
        force_first_round_loss: If `True`, train with maximum likelihood,
            i.e., potentially ignoring the correction for using a proposal
            distribution different from the prior.
        discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
            from the prior. Training may be sped up by ignoring such less targeted
            samples.
        retrain_from_scratch: Whether to retrain the conditional density
            estimator for the posterior from scratch each round.
        show_train_summary: Whether to print the number of epochs and validation
            loss after the training.
        dataloader_kwargs: Additional or updated kwargs to be passed to the training
            and validation dataloaders (like, e.g., a collate_fn)

    Returns:
        Density estimator that approximates the distribution $p(\theta|x)$.
    """
    SNPE._num_atoms = 10
    SNPE.use_non_atomic_loss = False
    SNPE._use_combined_loss = False

    # Load data from most recent round.
    SNPE._round = max(SNPE._data_round_index)

    if SNPE._round == 0 and SNPE._neural_net is not None:
        assert force_first_round_loss, (
            "You have already trained this neural network. After you had trained "
            "the network, you again appended simulations with `append_simulations"
            "(theta, x)`, but you did not provide a proposal. If the new "
            "simulations are sampled from the prior, you can set "
            "`.train(..., force_first_round_loss=True`). However, if the new "
            "simulations were not sampled from the prior, you should pass the "
            "proposal, i.e. `append_simulations(theta, x, proposal)`. If "
            "your samples are not sampled from the prior and you do not pass a "
            "proposal and you set `force_first_round_loss=True`, the result of "
            "SNPE will not be the true posterior. Instead, it will be the proposal "
            "posterior, which (usually) is more narrow than the true posterior."
        )

    # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
    if calibration_kernel is None:
        calibration_kernel = lambda x: torch.ones([len(x)], device=SNPE._device)

    # Starting index for the training set (1 = discard round-0 samples).
    start_idx = int(discard_prior_samples and SNPE._round > 0)

    # For non-atomic loss, we can not reuse samples from previous rounds as of now.
    # SNPE-A can, by construction of the algorithm, only use samples from the last
    # round. SNPE-A is the only algorithm that has an attribute `_ran_final_round`,
    # so this is how we check for whether or not we are using SNPE-A.
    if SNPE.use_non_atomic_loss or hasattr(SNPE, "_ran_final_round"):
        start_idx = SNPE._round

    # Set the proposal to the last proposal that was passed by the user. For
    # atomic SNPE, it does not matter what the proposal is. For non-atomic
    # SNPE, we only use the latest data that was passed, i.e. the one from the
    # last proposal.
    proposal = SNPE._proposal_roundwise[-1]

    if not load_sequentially:
        train_loader, val_loader = SNPE.get_dataloaders(
            start_idx,
            training_batch_size,
            validation_fraction,
            resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )

    else:
        theta, x, prior_masks = SNPE.get_simulations(start_idx)

        dataset = torch.utils.data.TensorDataset(theta, x, prior_masks)

        # Get total number of training examples.
        num_examples = theta.size(0)
        # Select random train and validation splits from (theta, x) pairs.
        num_training_examples = (
            int((1 - validation_fraction) * num_examples / 1000) * 1000
        )
        num_validation_examples = num_examples - num_training_examples

        if not resume_training:
            # Seperate indicies for training and validation
            permuted_indices = torch.randperm(int(num_examples / 1000))
            SNPE.train_indices, SNPE.val_indices = (
                [
                    i + 1000 * j
                    for j in permuted_indices[: int(num_training_examples / 1000)]
                    for i in range(0, 1000)
                ],
                [
                    i + 1000 * j
                    for j in permuted_indices[int(num_training_examples / 1000) :]
                    for i in range(0, 1000)
                ],
            )

        # Create training and validation loaders using a subset sampler.
        # Intentionally use dicts to define the default dataloader args
        # Then, use dataloader_kwargs to override (or add to) any of these defaults
        # https://stackoverflow.com/questions/44784577/in-method-call-args-how-to-override-keyword-argument-of-unpacked-dict
        train_loader_kwargs = {
            "batch_size": min(training_batch_size, num_training_examples),
            "drop_last": True,
            "shuffle": False,
            # "sampler": torch.utils.data.(SNPE.train_indices),
        }
        val_loader_kwargs = {
            "batch_size": min(training_batch_size, num_validation_examples),
            "shuffle": False,
            "drop_last": True,
            # "sampler": SubsetRandomSampler(SNPE.val_indices),
        }
        if dataloader_kwargs is not None:
            train_loader_kwargs = dict(train_loader_kwargs, **dataloader_kwargs)
            val_loader_kwargs = dict(val_loader_kwargs, **dataloader_kwargs)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, SNPE.train_indices), **train_loader_kwargs
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, SNPE.val_indices), **val_loader_kwargs
        )

    # First round or if retraining from scratch:
    # Call the `SNPE._build_neural_net` with the rounds' thetas and xs as
    # arguments, which will build the neural network.
    # This is passed into NeuralPosterior, to create a neural posterior which
    # can `sample()` and `log_prob()`. The network is accessible via `.net`.
    if SNPE._neural_net is None or retrain_from_scratch:
        # Get theta,x to initialize NN
        theta, x, _ = SNPE.get_simulations(starting_round=start_idx)
        # Use only training data for building the neural net (z-scoring transforms)

        SNPE._neural_net = SNPE._build_neural_net(
            theta[SNPE.train_indices].to("cpu"),
            x[SNPE.train_indices].to("cpu"),
        )
        # print(SNPE._neural_net)
        SNPE._x_shape = x_shape_from_simulation(x.to("cpu"))

        test_posterior_net_for_multi_d_x(
            SNPE._neural_net,
            theta.to("cpu"),
            x.to("cpu"),
        )

        del theta, x

    # Move entire net to device for training.
    SNPE._neural_net.to(SNPE._device)

    if not resume_training:
        SNPE.optimizer = torch.optim.Adam(
            list(SNPE._neural_net.parameters()), lr=learning_rate
        )
        SNPE.epoch, SNPE._val_log_prob = 0, float("-Inf")
        SNPE.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            SNPE.optimizer, factor=0.5, verbose=True, patience=8, cooldown=2
        )

    epoch_bar = tqdm(
        range(max_num_epochs),
        desc="Training NS:",
        position=0,
        leave=True,
        disable=(not show_progress_bar),
    )
    epoch_bar.set_postfix(
        {"Tloss": f"{0:.2f}", "Vloss": f"{0:.2f}", "m5VL": f"{0:.2f}"}
    )
    for k in epoch_bar:
        if SNPE._converged(SNPE.epoch, stop_after_epochs):
            print("\n\tStopping training because of loss convergence...")
            break

        # Train for a single epoch.
        SNPE._neural_net.train()
        train_log_probs_sum = 0
        epoch_start_time = time()
        for batch in train_loader:
            p = np.random.choice(
                train_loader.batch_size,
                size=int(train_loader.batch_size * batch_fraction),
            )
            SNPE.optimizer.zero_grad()
            # Get batches on current device.
            theta_batch, x_batch, masks_batch = (
                batch[0][p].to(SNPE._device),
                batch[1][p].to(SNPE._device),
                batch[2][p].to(SNPE._device),
            )

            if use_kl_loss:
                train_losses = forward_kld(SNPE._neural_net, theta_batch, x_batch)
            else:
                train_losses = SNPE._loss(
                    theta_batch,
                    x_batch,
                    masks_batch,
                    proposal,
                    calibration_kernel,
                    force_first_round_loss=force_first_round_loss,
                )
            train_loss = torch.mean(train_losses)
            train_log_probs_sum -= train_losses.sum().item()

            train_loss.backward()
            if clip_max_norm is not None:
                clip_grad_norm_(SNPE._neural_net.parameters(), max_norm=clip_max_norm)
            SNPE.optimizer.step()

        SNPE.epoch += 1

        train_log_prob_average = train_log_probs_sum / (
            len(train_loader) * len(p)  # type: ignore
        )

        SNPE._summary["training_log_probs"].append(train_log_prob_average)

        # Calculate validation performance.
        SNPE._neural_net.eval()
        val_log_prob_sum = 0

        with torch.no_grad():
            for batch in val_loader:
                p = np.random.choice(
                    val_loader.batch_size,
                    size=int(val_loader.batch_size * batch_fraction),
                )
                theta_batch, x_batch, masks_batch = (
                    batch[0][p].to(SNPE._device),
                    batch[1][p].to(SNPE._device),
                    batch[2][p].to(SNPE._device),
                )
                # Take negative loss here to get validation log_prob.
                if use_kl_loss:
                    val_losses = forward_kld(SNPE._neural_net, theta_batch, x_batch)
                else:
                    val_losses = SNPE._loss(
                        theta_batch,
                        x_batch,
                        masks_batch,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=force_first_round_loss,
                    )
                val_log_prob_sum -= val_losses.sum().item()

        # Take mean over all validation samples.
        SNPE._val_log_prob = val_log_prob_sum / (
            len(val_loader) * len(p)  # type: ignore
        )

        SNPE.scheduler.step(-SNPE._val_log_prob)
        # Log validation log prob for every epoch.
        SNPE._summary["validation_log_probs"].append(SNPE._val_log_prob)
        SNPE._summary["epoch_durations_sec"].append(time() - epoch_start_time)
        if k > 5:
            epoch_bar.set_postfix(
                {
                    "Tloss": f"{-train_log_prob_average:.2f}",
                    "Vloss": f"{-SNPE._val_log_prob:.2f}",
                    "m5VL": f"{-np.mean(SNPE._summary['validation_log_probs'][-5:]):.2f}",
                }
            )
        else:
            epoch_bar.set_postfix(
                {
                    "Tloss": f"{-train_log_prob_average:.2f}",
                    "Vloss": f"{-SNPE._val_log_prob:.2f}",
                    "m5VL": f"{-np.mean(SNPE._summary['validation_log_probs']):.2f}",
                }
            )

        SNPE._maybe_show_progress(SNPE._show_progress_bars, SNPE.epoch)

    SNPE._report_convergence_at_end(SNPE.epoch, stop_after_epochs, max_num_epochs)

    # Update summary.
    SNPE._summary["epochs_trained"].append(SNPE.epoch)
    SNPE._summary["best_validation_log_prob"].append(SNPE._best_val_log_prob)

    # Update tensorboard and summary dict.
    SNPE._summarize(round_=SNPE._round)

    # Update description for progress bar.
    if show_train_summary:
        print(SNPE._describe_round(SNPE._round, SNPE._summary))

    # Avoid keeping the gradients in the resulting network, which can
    # cause memory leakage when benchmarking.
    SNPE._neural_net.zero_grad(set_to_none=True)

    return deepcopy(SNPE._neural_net)


def fit_posterior(
    x,
    theta,
    neural_posteriors,
    prior,
    device="cpu",
    proposal=None,
    lr=7.0e-4,
    batch_size=500,
    max_num_epochs=400,
):
    """Train the posteriors estimators.

    Args:
        x (tensor): features
        theta (tensor): targets
        neural_posteriors (list): list of models
        prior (_type_): _description_
        device (str, optional): device to use. Defaults to 'cpu'.

    Returns:
        NeuralPosteriorEnsemble: Ensemble of trained posteriors models
    """
    posteriors, val_loss = [], []
    for n, posterior in enumerate(neural_posteriors):
        print(f"\n Training model {n}")
        model = SNPE_C(prior=prior, density_estimator=posterior, device=device)
        model = model.append_simulations(theta, x, proposal=proposal)
        density_estimator = train_nsf(
            model,
            training_batch_size=batch_size,
            learning_rate=lr,
            max_num_epochs=max_num_epochs,
        )
        posteriors.append(model.build_posterior(density_estimator))
        print(
            f"\n Validation Log prob: {model.summary['best_validation_log_prob'][0]:.2e}"
        )
        val_loss += model.summary["best_validation_log_prob"]
    return NeuralPosteriorEnsemble(
        posteriors=posteriors,
        weights=torch.tensor([float(vl) for vl in val_loss]),
    )


def fit_likelihood(x, theta, neural_density_estimator, prior, device="cpu"):
    """Train the likelihood estimator and returns it with a posterior estimator using Variational Inference

    Args:
        x (tensor): features
        theta (tensor): target
        neural_density_estimator (_type_): normalising flow
        prior (_type_): _description_

    Returns:
        _type_: _description_
    """
    model = SNLE(prior=prior, density_estimator=neural_density_estimator, device=device)
    model = model.append_simulations(theta, x)
    like_estimator = model.train()
    print(f"\n Validation Log prob: {model.summary['best_validation_log_prob'][0]:.2e}")
    # Sample the likelihood with variational inference
    posterior = model.build_posterior(
        like_estimator,
        sample_with="vi",
        vi_method="fKL",
    )
    return like_estimator, posterior, model


####### TRAINING FOR ZUKO NORMALISING FLOWS ########


def train_zuko(
    flow,
    features,
    targets,
    weights,
    training_batch_size: int = 500,
    batch_fraction: float = 1.0,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.1,
    stop_after_epochs: int = 20,
    max_num_epochs: int = 300,
    show_progress_bar: bool = True,
    verbose: bool = True,
):
    
    # variables used to measure the duration of the several parts of the routine
    t_data = t_loss_training= t_loss_backwards= t_optim_step = t_loss_validation = _t = t_training = t_validating = _t2 = t_waiting = 0
    t_init = time()

    dataset = torch.utils.data.TensorDataset(features, targets, weights)
    num_examples = features.shape[0]
    del features, targets, weights
    gc.collect()
    num_training_examples = int((1 - validation_fraction) * num_examples)
    num_validation_examples = num_examples - num_training_examples
    permuted_indices = torch.randperm(num_examples)


    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, permuted_indices[:num_training_examples]),
        batch_size=training_batch_size,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, permuted_indices[num_training_examples:]),
        batch_size=training_batch_size,
    )
    del dataset
    gc.collect()
    
    t_data = time() - t_init

    optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, verbose=verbose, patience=8, cooldown=2
    ) # the scheduler is used to reduce the learning rate when the validation loss doesn't improve

    epoch_bar = tqdm(
        range(max_num_epochs),
        desc="Training ZK:",
        position=0,
        leave=True,
        disable=(not show_progress_bar),
    )
    epoch_bar.set_postfix(
        {"Tloss": f"{0:.2f}", "Vloss": f"{0:.2f}", "m5VL": f"{0:.2f}"}
    ) # m5VL is the mean of the validation loss on the last 5 epochs

    LOSSES = [[], []]
    epochs_since_last_improvement = 0
    for k in epoch_bar:
        if epochs_since_last_improvement > stop_after_epochs:
            if verbose:
                print("\n\tStopping training because of loss convergence...")
            break

        # Training
        flow.train()
        L = 0.0
        _t2 = time()
        _t = time()
        for _features, _targets, _weights in train_loader:
            p = np.random.choice(
                _features.shape[0], size=max(int(_features.shape[0] * batch_fraction),1)
            ) # we permute the ordre of the batch and we take only a fraction of it
            optimizer.zero_grad()
            _t = time()
            loss = (
                -flow(_features[p].unsqueeze(dim=1)).log_prob(_targets[p]) * _weights[p]
            ).sum() / (len(p)*_targets.shape[-1]) # the loss is the -logprob weighted by the weight tensor.
            t_loss_training += time() - _t
            _t = time()
            loss.backward()
            t_loss_backwards += time() - _t
            _t = time()
            optimizer.step()
            t_optim_step += time() - _t
            _t = time()
            L += loss.item()
            t_waiting += time() - _t

        LOSSES[0].append(L / len(train_loader))
        t_training += time() - _t2
        _t2 = time()

        # Validation
        flow.eval()
        L = 0.0
        for _features, _targets, _weights in val_loader:
            p = np.random.choice(
                _features.shape[0], size=max(int(_features.shape[0] * batch_fraction),1)
            )
            _t = time()
            loss = (
                -flow(_features[p].unsqueeze(dim=1)).log_prob(_targets[p]) * _weights[p]
            ).sum() / (len(p)*_targets.shape[-1])
            t_loss_validation += time() - _t
            L += loss.item()
        LOSSES[1].append(L / len(val_loader))
        t_validating += time() - _t2

        scheduler.step(L)

        if k == 0:
            best_loss = L
        elif L < best_loss:
            best_loss = L
            epochs_since_last_improvement = 0
        else:
            epochs_since_last_improvement += 1

        if k > 5:
            epoch_bar.set_postfix(
                {
                    "Tloss": f"{LOSSES[0][-1]:.2f}",
                    "Vloss": f"{LOSSES[1][-1]:.2f}",
                    "m5VL": f"{np.mean(LOSSES[1][-5:]):.2f}",
                }
            )
        else:
            epoch_bar.set_postfix(
                {
                    "Tloss": f"{LOSSES[0][-1]:.2f}",
                    "Vloss": f"{LOSSES[1][-1]:.2f}",
                    "m5VL": f"{np.mean(LOSSES[1]):.2f}",
                }
            )

    t_total = time() - t_init
    if verbose:
        if not epochs_since_last_improvement > stop_after_epochs:
            print("\n\tStopping training because of epoch limit reached...")
        print(
            f"\nEpochs trained: {k}\tTotal duration: {t_total/60:.2f} min\tBest val log prob: {-min(LOSSES[1]):.2f}"
        )
        print("Detailed durations:")
        print(f"\t{'Dataloaders init:':<35} {t_data/60:.2f} min")
        print(f"\t{'Training:':<35} {t_training/60:.2f} min")
        print(f"\t\t{'Loss training computation:':<35} {t_loss_training/60:.2f} min")
        print(f"\t\t{'Loss backwards computation:':<35} {t_loss_backwards/60:.2f} min")
        print(f"\t\t{'Optimiser step:':<35} {t_optim_step/60:.2f} min")
        print(f"\t\t{'Waiting for loss computation:':<35} {t_waiting/60:.2f} min")
        print(f"\t{'Validating:':<35} {t_validating/60:.2f} min")
        print(
            f"\t\t{'Loss validation computation:':<35} {t_loss_validation/60:.2f} min"
        )

    return LOSSES # LOSSES=[Training_losses_list,Validation_losses_list]


def train_zuko_without_weights(
    flow,
    features,
    targets,
    **kwargs
    ):
    targets=targets.unsqueeze(dim=-2)
    weights=torch.ones_like(targets[...,0])
    return train_zuko(flow,
                      features,
                      targets,
                      weights,
                      **kwargs)



def train_independent_zuko(
    list_of_flows,
    features,
    targets,
    weights,
    parameters_names: list = ["R", "T", "H2O", "CO2", "CO", "CH4", "NH3"],
    **kwargs,
):

    L = []
    for k in range(targets.shape[-1]):
        print(f"\t\tTraining model for parameter {parameters_names[k]}:")
        L.append(
            train_zuko(list_of_flows[k], features, targets[..., [k]], weights, **kwargs)
        )
    return L

def train_zuko_cdf(
    flow,
    features,
    targets,
    weights,
    training_batch_size: int = 500,
    batch_fraction: float = 1.0,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.1,
    stop_after_epochs: int = 20,
    max_num_epochs: int = 300,
    show_progress_bar: bool = True,
    verbose: bool = True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    
    # variables used to measure the duration of the several parts of the routine
    t_data = t_loss_training= t_loss_backwards= t_optim_step = t_loss_validation = _t = t_training = t_validating = _t2 = t_waiting = 0
    t_init = time()

    dataset = torch.utils.data.TensorDataset(features, targets, weights)
    num_examples = features.shape[0]
    del features, targets, weights
    gc.collect()
    num_training_examples = int((1 - validation_fraction) * num_examples)
    num_validation_examples = num_examples - num_training_examples
    permuted_indices = torch.randperm(num_examples)


    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, permuted_indices[:num_training_examples]),
        batch_size=training_batch_size,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, permuted_indices[num_training_examples:]),
        batch_size=training_batch_size,
    )
    del dataset
    gc.collect()
    
    t_data = time() - t_init

    optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, verbose=verbose, patience=8, cooldown=2
    ) # the scheduler is used to reduce the learning rate when the validation loss doesn't improve

    epoch_bar = tqdm(
        range(max_num_epochs),
        desc="Training ZK:",
        position=0,
        leave=True,
        disable=(not show_progress_bar),
    )
    epoch_bar.set_postfix(
        {"Tloss": f"{0:.2f}", "Vloss": f"{0:.2f}", "m5VL": f"{0:.2f}"}
    ) # m5VL is the mean of the validation loss on the last 5 epochs

    LOSSES = [[], []]
    epochs_since_last_improvement = 0
    for k in epoch_bar:
        if epochs_since_last_improvement > stop_after_epochs:
            if verbose:
                print("\n\tStopping training because of loss convergence...")
            break

        # Training
        flow.train()
        L = 0.0
        _t2 = time()
        _t = time()
        j=0
        for _features, _targets, _weights in train_loader:
            p = np.random.choice(
                _features.shape[0], size=max(int(_features.shape[0] * batch_fraction),1)
            ) # we permute the ordre of the batch and we take only a fraction of it
            optimizer.zero_grad()
            _features = _features[p]
            _targets = _targets[p]
            _weights = _weights[p]
            logprob_loss= (
                -flow(_features.unsqueeze(dim=1)).log_prob(_targets) * _weights
            ).sum() / len(p) # the loss is the -logprob weighted by the weight tensor.
            _weights=_weights.unsqueeze(dim=-1)*torch.ones(*(_weights.shape),_targets.shape[-1]).to(device)
            _targets, ind = _targets.sort(dim=1)
            _weights = torch.gather(_weights, 1, ind)
            _weights = torch.cumsum(_weights, dim=1)
            _t = time()
            pred_cdf=1/2*(1.+torch.erf(flow(_features.unsqueeze(dim=1)).transform(_targets)/torch.sqrt(torch.Tensor([2.]).to(device))))
            loss=100*torch.nn.MSELoss()(pred_cdf,_weights)+10*torch.mean(torch.max(torch.abs(pred_cdf-_weights),dim=1)[0])
            loss+=1/2*logprob_loss
            t_loss_training += time() - _t
            _t = time()
            loss.backward()
            t_loss_backwards += time() - _t
            _t = time()
            optimizer.step()
            t_optim_step += time() - _t
            _t = time()
            L += loss.item()
            t_waiting += time() - _t
            if k%10==0 and j%50==25:
                print(f"  T: epoch: {k} batch: {j} MSE cdf: {100*torch.nn.MSELoss()(pred_cdf,_weights):.2f}  max cdf: {10*torch.mean(torch.max(torch.abs(pred_cdf-_weights),dim=1)[0]):.2f}  logprob:{(1/2*logprob_loss).item():.2f}")
            j+=1
            
        LOSSES[0].append(L / len(train_loader))
        t_training += time() - _t2
        _t2 = time()

        # Validation
        flow.eval()
        L = 0.0
        j=0
        for _features, _targets, _weights in val_loader:
            p = np.random.choice(
                _features.shape[0], size=max(int(_features.shape[0] * batch_fraction),1)
            )
            _features = _features[p]
            _targets = _targets[p]
            _weights = _weights[p]
            logprob_loss= (
                -flow(_features.unsqueeze(dim=1)).log_prob(_targets) * _weights
            ).sum() / len(p) # the loss is the -logprob weighted by the weight tensor.
            _weights=_weights.unsqueeze(dim=-1)*torch.ones(*(_weights.shape),_targets.shape[-1]).to(device)
            _targets, ind = _targets.sort(dim=1)
            _weights = torch.gather(_weights, 1, ind)
            _weights = torch.cumsum(_weights, dim=1)
            _t = time()
            pred_cdf=1/2*(1.+torch.erf(flow(_features.unsqueeze(dim=1)).transform(_targets)/torch.sqrt(torch.Tensor([2.]).to(device))))
            loss=100*torch.nn.MSELoss()(pred_cdf,_weights)+10*torch.mean(torch.max(torch.abs(pred_cdf-_weights),dim=1)[0])
            loss+=1/2*logprob_loss
            t_loss_validation += time() - _t
            L += loss.item()
            if k%10==0 and j%50==25:
                print(f"  V: epoch: {k} batch: {j} MSE cdf: {100*torch.nn.MSELoss()(pred_cdf,_weights):.2f}  max cdf: {10*torch.mean(torch.max(torch.abs(pred_cdf-_weights),dim=1)[0]):.2f}  logprob:{(1/2*logprob_loss).item():.2f}")
            j+=1
            
        LOSSES[1].append(L / len(val_loader))
        t_validating += time() - _t2

        scheduler.step(L)

        if k == 0:
            best_loss = L
        elif L < best_loss:
            best_loss = L
            epochs_since_last_improvement = 0
        else:
            epochs_since_last_improvement += 1

        if k > 5:
            epoch_bar.set_postfix(
                {
                    "Tloss": f"{LOSSES[0][-1]:.2f}",
                    "Vloss": f"{LOSSES[1][-1]:.2f}",
                    "m5VL": f"{np.mean(LOSSES[1][-5:]):.2f}",
                }
            )
        else:
            epoch_bar.set_postfix(
                {
                    "Tloss": f"{LOSSES[0][-1]:.2f}",
                    "Vloss": f"{LOSSES[1][-1]:.2f}",
                    "m5VL": f"{np.mean(LOSSES[1]):.2f}",
                }
            )

    t_total = time() - t_init
    if verbose:
        if not epochs_since_last_improvement > stop_after_epochs:
            print("\n\tStopping training because of epoch limit reached...")
        print(
            f"\nEpochs trained: {k}\tTotal duration: {t_total/60:.2f} min\tBest val log prob: {-min(LOSSES[1]):.2f}"
        )
        print("Detailed durations:")
        print(f"\t{'Dataloaders init:':<35} {t_data/60:.2f} min")
        print(f"\t{'Training:':<35} {t_training/60:.2f} min")
        print(f"\t\t{'Loss training computation:':<35} {t_loss_training/60:.2f} min")
        print(f"\t\t{'Loss backwards computation:':<35} {t_loss_backwards/60:.2f} min")
        print(f"\t\t{'Optimiser step:':<35} {t_optim_step/60:.2f} min")
        print(f"\t\t{'Waiting for loss computation:':<35} {t_waiting/60:.2f} min")
        print(f"\t{'Validating:':<35} {t_validating/60:.2f} min")
        print(
            f"\t\t{'Loss validation computation:':<35} {t_loss_validation/60:.2f} min"
        )

    return LOSSES # LOSSES=[Training_losses_list,Validation_losses_list]



def train_zuko_with_spectrum_recovering(
    list_of_flows,
    decoder_shape,
    decoder_scale,
    features,
    targets,
    weights,
    alpha_shape=1e-2,
    alpha_scale=2e-1,
    training_batch_size: int = 500,
    batch_fraction: float = 1.0,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.1,
    stop_after_epochs: int = 20,
    max_num_epochs: int = 300,
    show_progress_bar: bool = True,
    verbose: bool = True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    
    # variables used to measure the duration of the several parts of the routine
    t_data = t_loss_cdf=t_loss_recovery= t_loss_backwards= t_optim_step = t_loss_validation = _t = t_training = t_validating = _t2 = t_waiting = 0
    t_init = time()

    dataset = torch.utils.data.TensorDataset(features, targets, weights)
    num_examples = features.shape[0]
    del features, targets, weights
    num_training_examples = int((1 - validation_fraction) * num_examples)
    num_validation_examples = num_examples - num_training_examples
    permuted_indices = torch.randperm(num_examples)


    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, permuted_indices[:num_training_examples]),
        batch_size=training_batch_size,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, permuted_indices[num_training_examples:]),
        batch_size=training_batch_size,
    )
    del dataset

    t_data = time() - t_init
    decoder_scale.eval()
    decoder_shape.eval()
    
    
    optimizers = [torch.optim.Adam(flow.parameters(), lr=learning_rate) for flow in list_of_flows]
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, verbose=verbose, patience=8, cooldown=2
    ) for optimizer in optimizers] # the scheduler is used to reduce the learning rate when the validation loss doesn't improve

    epoch_bar = tqdm(
        range(max_num_epochs),
        desc="Training List of Flows ZK with SR:",
        position=0,
        leave=True,
        disable=(not show_progress_bar),
    )
    epoch_bar.set_postfix(
        {"Tloss": f"{0:.2f}", "Vloss": f"{0:.2f}", "m5VL": f"{0:.2f}"}
    ) # m5VL is the mean of the validation loss on the last 5 epochs

    LOSSES = [[], []]
    epochs_since_last_improvement = [0 for f in list_of_flows]
    stopped=[False for f in list_of_flows]
    for k in epoch_bar:
        stopping=True
        for j,f in enumerate(list_of_flows):
            if epochs_since_last_improvement[j] > stop_after_epochs:
                if verbose:
                    print(f"\n\tStopping training parameter {j} because of loss convergence...")
                stopped[j]=True
            if not stopped[j]:
                stopping=False
        if stopping:
            if verbose:
                    print(f"\n\tStopping the whole training because of loss convergence...")
            break
        # Training
        for flow in list_of_flows:
            flow.train()
        L = torch.zeros(len(list_of_flows))
        _t2 = time()
        _t = time()
        for _features, _targets, _weights in train_loader:
            p = np.random.choice(
                _features.shape[0], size=max(int(_features.shape[0] * batch_fraction),1)
            ) # we permute the ordre of the batch and we take only a fraction of it
            order=np.random.permutation(len(list_of_flows))
            for optimizer in optimizers:
                optimizer.zero_grad()
            _features = _features[p]
            _targets = _targets[p]
            _weights = _weights[p]
            _weights=_weights.unsqueeze(dim=-1)*torch.ones(*(_weights.shape),_targets.shape[-1]).to(device)
            _targets, ind = _targets.sort(dim=1)
            _weights = torch.gather(_weights, 1, ind)
            _weights = torch.cumsum(_weights, dim=1)
            _t = time()
            nsfs=[list_of_flows[j](_features.unsqueeze(dim=1)) for j in order]
            mean_pred=torch.cat([n.transform.inv(torch.zeros(len(p),1,1).to(device)) for n in nsfs],dim=-1)
            recovered_spectrum_shape=decoder_shape(torch.cat([_features[:,:12].unsqueeze(dim=1),mean_pred],dim=-1))
            recovered_spectrum_scale=decoder_scale(torch.cat([_features[:,:12].unsqueeze(dim=1),mean_pred],dim=-1))
            recovering_loss=alpha_shape*torch.mean(torch.clamp((_features[:,-104:-52].unsqueeze(dim=1)-recovered_spectrum_shape[:,0,:])/(torch.abs(recovered_spectrum_shape[:,1,:])+1e-2),0.,10.)**2)
            if k==20:
                print(f"Recovering shape loss: {recovering_loss}")
            recovering_loss+=alpha_scale*torch.nn.MSELoss()(_features[:,-106:-104].unsqueeze(dim=1),recovered_spectrum_scale)
            recovering_loss=torch.clamp(recovering_loss,0.,0.1)
            if k==20:
                print(f"Recovering loss {recovering_loss}")
            # recovering_loss.backward(retain_graph=True)
            t_loss_recovery+=time()-_t
            for j,n in enumerate(nsfs):
                # loss = (
                #     -flow(_features[p].unsqueeze(dim=1)).log_prob(_targets[p]) * _weights[p]
                # ).sum() / len(p) # the loss is the -logprob weighted by the weight tensor.
                _t=time()
                pred_cdf=1/2*(1.+torch.erf(n.transform(_targets[...,[order[j]]])/torch.sqrt(torch.Tensor([2.]).to(device))))
                loss=torch.nn.MSELoss()(pred_cdf,_weights[...,[order[j]]])+1e-1*torch.mean(torch.max(torch.abs(pred_cdf-_weights[...,[order[j]]]),dim=1)[0])+recovering_loss
                t_loss_cdf += time() - _t
                _t = time()
                loss.backward(retain_graph=True)
                t_loss_backwards += time() - _t
                _t = time()
                optimizers[order[j]].step()
                t_optim_step += time() - _t
                _t = time()
                L[order[j]] += loss.item()
                if k==20:
                    print(f"  Total loss param {order[j]} {loss.item()}")
                t_waiting += time() - _t

        LOSSES[0].append(np.array(L) / len(train_loader))
        t_training += time() - _t2
        _t2 = time()

        # Validation
        flow.eval()
        L = torch.zeros(len(list_of_flows))
        for _features, _targets, _weights in val_loader:
            p = np.random.choice(
                _features.shape[0], size=max(int(_features.shape[0] * batch_fraction),1)
            )
            order=np.random.permutation(len(list_of_flows))
            _features = _features[p]
            _targets = _targets[p]
            _weights = _weights[p]
            _weights=_weights.unsqueeze(dim=-1)*torch.ones(*(_weights.shape),_targets.shape[-1]).to(device)
            _targets, ind = _targets.sort(dim=1)
            _weights = torch.gather(_weights, 1, ind)
            _weights = torch.cumsum(_weights, dim=1)
            _t = time()
            nsfs=[list_of_flows[j](_features.unsqueeze(dim=1)) for j in order]
            mean_pred=torch.cat([n.transform.inv(torch.zeros(len(p),1,1).to(device)) for n in nsfs],dim=-1)
            recovered_spectrum_shape=decoder_shape(torch.cat([_features[:,:12].unsqueeze(dim=1),mean_pred],dim=-1))
            recovered_spectrum_scale=decoder_scale(torch.cat([_features[:,:12].unsqueeze(dim=1),mean_pred],dim=-1))
            recovering_loss=alpha_shape*torch.mean(torch.clamp((_features[:,-104:-52].unsqueeze(dim=1)-recovered_spectrum_shape[:,0,:])/(torch.abs(recovered_spectrum_shape[:,1,:])+1e-2),0.,10.)**2)
            recovering_loss+=alpha_scale*torch.nn.MSELoss()(_features[:,-106:-104].unsqueeze(dim=1),recovered_spectrum_scale)
            recovering_loss=torch.clamp(recovering_loss,0.,0.1)
            t_loss_recovery+=time()-_t
            for j,n in enumerate(nsfs):
                # loss = (
                #     -flow(_features[p].unsqueeze(dim=1)).log_prob(_targets[p]) * _weights[p]
                # ).sum() / len(p) # the loss is the -logprob weighted by the weight tensor.
                _t=time()
                pred_cdf=1/2*(1.+torch.erf(n.transform(_targets[...,[order[j]]])/torch.sqrt(torch.Tensor([2.]).to(device))))
                loss=recovering_loss+torch.nn.MSELoss()(pred_cdf,_weights[...,[order[j]]])+1e-1*torch.mean(torch.max(torch.abs(pred_cdf-_weights[...,[order[j]]]),dim=1)[0])
                L[order[j]] += loss.item()
            
            t_loss_validation += time() - _t
            
        LOSSES[1].append(np.array(L)   / len(val_loader))
        t_validating += time() - _t2

        for j,scheduler in enumerate(schedulers):
            scheduler.step(L[j])

        if k == 0:
            best_loss = L
        else:
            for j,l in enumerate(L):
                if l<best_loss[j]:
                    best_loss[j] = l
                    epochs_since_last_improvement[j]=0
                else:
                    epochs_since_last_improvement[j] += 1

        if k > 5:
            epoch_bar.set_postfix(
                {
                    "Tloss": f"{np.mean(LOSSES[0][-1]):.2f}",
                    "Vloss": f"{np.mean(LOSSES[1][-1]):.2f}",
                    "m5VL": f"{np.mean(LOSSES[1][-5:]):.2f}",
                }
            )
        else:
            epoch_bar.set_postfix(
                {
                    "Tloss": f"{np.mean(LOSSES[0][-1]):.2f}",
                    "Vloss": f"{np.mean(LOSSES[1][-1]):.2f}",
                    "m5VL": f"{np.mean(LOSSES[1]):.2f}",
                }
            )

    t_total = time() - t_init
    if verbose:
        if not k==max_num_epochs-1:
            print("\n\tStopping training because of epoch limit reached...")
        print(
            f"\nEpochs trained: {k}\tTotal duration: {t_total/60:.2f} min\tBest val log prob: {-np.min(LOSSES[1]):.2f}"
        )
        print("Detailed durations:")
        print(f"\t{'Dataloaders init:':<35} {t_data/60:.2f} min")
        print(f"\t{'Training:':<35} {t_training/60:.2f} min")
        print(f"\t\t{'Loss cdf computation:':<35} {t_loss_cdf/60:.2f} min")
        print(f"\t\t{'Loss recovering computation:':<35} {t_loss_recovery/60:.2f} min")
        print(f"\t\t{'Loss backwards computation:':<35} {t_loss_backwards/60:.2f} min")
        print(f"\t\t{'Optimiser step:':<35} {t_optim_step/60:.2f} min")
        print(f"\t\t{'Waiting for loss computation:':<35} {t_waiting/60:.2f} min")
        print(f"\t{'Validating:':<35} {t_validating/60:.2f} min")
        print(
            f"\t\t{'Loss validation computation:':<35} {t_loss_validation/60:.2f} min"
        )

    return LOSSES # LOSSES=[Training_losses_list,Validation_losses_list]



################ NEURAL NET FOR DIRECT CDF ##################


def train_cdf(
    directcdf,
    features,
    targets,
    weights,
    training_batch_size: int = 500,
    batch_fraction: float = 1.0,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.1,
    stop_after_epochs: int = 20,
    max_num_epochs: int = 300,
    show_progress_bar: bool = True,
    verbose: bool = True,
):
    t_data = (
        t_loss_training
    ) = (
        t_sorting
    ) = (
        t_loss_validation
    ) = _t = t_training = t_validating = t_sorting_val = _t2 = t_waiting = 0
    t_init = time()

    dataset = torch.utils.data.TensorDataset(features, targets, weights)
    num_examples = features.shape[0]
    del features, targets, weights
    num_training_examples = int((1 - validation_fraction) * num_examples)
    num_validation_examples = num_examples - num_training_examples
    permuted_indices = torch.randperm(num_examples)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, permuted_indices[:num_training_examples]),
        batch_size=training_batch_size,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, permuted_indices[num_training_examples:]),
        batch_size=training_batch_size,
    )

    del dataset

    t_data = time() - t_init

    optimizer = torch.optim.Adam(directcdf.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, verbose=verbose, patience=8, cooldown=2
    )

    epoch_bar = tqdm(
        range(max_num_epochs),
        desc="Training direct CDF:",
        position=0,
        leave=True,
        disable=(not show_progress_bar),
    )
    epoch_bar.set_postfix(
        {"Tloss": f"{0:.2f}", "Vloss": f"{0:.2f}", "m5VL": f"{0:.2f}"}
    )

    LOSSES = [[], []]
    epochs_since_last_improvement = 0
    for k in epoch_bar:
        if epochs_since_last_improvement > stop_after_epochs:
            if verbose:
                print("\n\tStopping training because of loss convergence...")
            break

        # Training
        directcdf.train()
        L = 0.0
        _t2 = time()
        _t = time()
        for _features, _targets, _weights in train_loader:
            _t = time()
            p = np.random.choice(
                _features.shape[0],
                size=max(int(_features.shape[0] * batch_fraction), 1),
            )
            optimizer.zero_grad()
            _features = _features[p]
            _targets = _targets[p]
            _weights = _weights[p]
            # _weights=(_weights.unsqueeze(dim=-1)*torch.ones(*(_weights.shape),_targets.shape[-1]))
            _targets, ind = _targets.sort(dim=1)
            _weights = torch.gather(_weights, 1, ind)
            _weights = torch.cumsum(_weights, dim=1)
            t_sorting = time() - _t
            _t = time()
            distrib = directcdf(_features)
            predictions = distrib.cdf(_targets)
            loss = 100 * torch.nn.MSELoss()(predictions, _weights)
            loss += 10 * (torch.abs(predictions - _weights)).max(dim=1)[0].mean()
            loss.backward()
            optimizer.step()
            t_loss_training += time() - _t
            _t = time()
            L += loss.item()
            t_waiting += time() - _t

        LOSSES[0].append(L / len(train_loader))
        t_training += time() - _t2
        _t2 = time()

        # Validation
        directcdf.eval()
        L = 0.0
        for _features, _targets, _weights in val_loader:
            _t = time()
            p = np.random.choice(
                _features.shape[0],
                size=max(int(_features.shape[0] * batch_fraction), 1),
            )
            optimizer.zero_grad()
            _features = _features[p]
            _targets = _targets[p]
            _weights = _weights[p]
            # _weights=(_weights.unsqueeze(dim=-1)*torch.ones(*(_weights.shape),_targets.shape[-1]))
            _targets, ind = _targets.sort(dim=1)
            _weights = torch.gather(_weights, 1, ind)
            _weights = torch.cumsum(_weights, dim=1)
            t_sorting_val = time() - _t
            _t = time()
            distrib = directcdf(_features)
            predictions = distrib.cdf(_targets)
            loss = 100 * torch.nn.MSELoss()(predictions, _weights)
            loss += 10 * (torch.abs(predictions - _weights)).max(dim=1)[0].mean()
            t_loss_validation += time() - _t
            L += loss.item()
        LOSSES[1].append(L / len(val_loader))
        t_validating += time() - _t2

        scheduler.step(L)

        if k == 0:
            best_loss = L
        elif L < best_loss:
            best_loss = L
            epochs_since_last_improvement = 0
        else:
            epochs_since_last_improvement += 1

        if k > 5:
            epoch_bar.set_postfix(
                {
                    "Tloss": f"{LOSSES[0][-1]:.2f}",
                    "Vloss": f"{LOSSES[1][-1]:.2f}",
                    "m5VL": f"{np.mean(LOSSES[1][-5:]):.2f}",
                }
            )
        else:
            epoch_bar.set_postfix(
                {
                    "Tloss": f"{LOSSES[0][-1]:.2f}",
                    "Vloss": f"{LOSSES[1][-1]:.2f}",
                    "m5VL": f"{np.mean(LOSSES[1]):.2f}",
                }
            )

    t_total = time() - t_init
    if verbose:
        if not epochs_since_last_improvement > stop_after_epochs:
            print("\n\tStopping training because of epoch limit reached...")
        print(
            f"\nEpochs trained: {k}\tTotal duration: {t_total/60:.2f} min\tBest MSE loss: {min(LOSSES[1]):.2f}"
        )
        print("Detailed durations:")
        print(f"\t{'Dataloaders init:':<35} {t_data/60:.2f} min")
        print(f"\t{'Training:':<35} {t_training/60:.2f} min")
        print(f"\t\t{'Target CDF computation:':<35} {t_sorting/60:.2f} min")
        print(f"\t\t{'Loss training computation:':<35} {t_loss_training/60:.2f} min")
        print(f"\t\t{'Waiting for loss computation:':<35} {t_waiting/60:.2f} min")
        print(f"\t{'Validating:':<35} {t_validating/60:.2f} min")
        print(f"\t\t{'Target CDF computation:':<35} {t_sorting_val/60:.2f} min")
        print(
            f"\t\t{'Loss validation computation:':<35} {t_loss_validation/60:.2f} min"
        )

    return LOSSES
