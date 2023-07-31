"""
Defines all the plotting functions used by both the autoencoders and the Normalising Flows.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy import ndarray
from matplotlib.axes import Axes
import torch
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import corner
import h5py

# import scienceplots
# plt.style.use('science')

from msapnet.utils.data import load_x_data

MAJOR = 24
MINOR = 20
color_list = [
    "#726bed",
    "#c457ca",
    "#ec4e9e",
    "#f95c74",
    "#f17752",
]  # adjust to the number of scenarios considered


####### PLOTS FOR AUTOENCODER ########


def _plot_loss(losses, latent_losses=None, yscale="log"):
    """
    Plots training and validation loss as a function of epochs

    Parameters
    ----------
    losses
    """
    plt.figure(figsize=(16, 9), constrained_layout=True)
    plt.plot(losses[0], label="Training Loss")
    plt.plot(losses[1], label="Validation Loss")
    plt.xlabel("Epoch", fontsize=MINOR)
    plt.ylabel("Loss", fontsize=MINOR)
    plt.yscale(yscale)
    plt.text(
        0.8,
        0.75,
        f"Final loss: {losses[1][-1]:.3e}",
        fontsize=MINOR,
        transform=plt.gca().transAxes,
    )
    plt.legend()

    if latent_losses is not None:
        ax2 = plt.twinx()
        ax2.set_ylabel("Ratio latent loss / full loss", fontsize=MINOR)
        ax2.plot(
            np.array(latent_losses[0]) / np.array(losses[0]),
            label="Training latent loss ratio",
            color="#68697b",
            marker="x",
            alpha=0.4,
        )
        ax2.plot(
            np.array(latent_losses[1]) / np.array(losses[1]),
            label="Validation latent loss ratio",
            color="#8c8da5",
            marker="+",
            alpha=0.4,
        )
        ax2.tick_params(axis="y", labelcolor="#45454f")
        ax2.set_yscale("log")

    legend = plt.legend()
    legend.get_frame().set_alpha(None)


def plot_loss_evolution(losses, yscale="log"):
    _plot_loss(losses, yscale=yscale)


def plot_loss_comparison(
    losses,
    latent_losses,
    initial_latent_loss_weight=1e-3,
    latent_loss_weight_scheduler=None,
    prefix="",
    config=None,
):
    """Plot in twin axis the full loss and the latent loss (supervised) to compare their evolutions.

    Args:
        losses (tuple[list,list]): Pair of training and validation losses.
        latent_losses (tuple[list,list]): Pair of training and validation latent losseslosses.
        initial_latent_loss_weight (float, optional): The initial value at epoch=0 of the latent loss weight. Defaults to 1e-3.
        latent_loss_weight_scheduler (function, optional): Scheduler for the latent loss weight. Defaults to None.
        prefix (str, optional): Prefix of the model. Defaults to ''.
        config (dict, optional): Config dictionary. Defaults to None.
    """

    epochs = np.arange(len(losses[0]))
    latent_loss_weights = initial_latent_loss_weight * np.ones_like(epochs)
    if latent_loss_weight_scheduler is not None:
        for i in range(1, len(epochs)):
            latent_loss_weights[i] = latent_loss_weight_scheduler(
                latent_loss_weights[i - 1], epochs[i]
            )
    plt.figure(figsize=(16, 9), constrained_layout=True)
    plt.plot(
        np.array(losses[0]) - np.array(latent_losses[0]),
        label="Training Unsupervised Loss",
        marker="x",
        color="#6f2b78",
    )
    plt.plot(
        np.array(losses[1]) - np.array(latent_losses[1]),
        label="Validation Unsupervised Loss",
        marker="+",
        color="#481c4d",
    )
    plt.ylabel("Unsupervised Loss", fontsize=MINOR)
    plt.xlabel("Epoch", fontsize=MINOR)
    plt.yscale("log")
    plt.legend(loc=1)
    ax2 = plt.twinx()
    ax2.plot(
        np.array(latent_losses[0]) / latent_loss_weights,
        label="Training Supervised Loss",
        marker="x",
        color="#34782b",
    )
    ax2.plot(
        np.array(latent_losses[1][:-1]) / latent_loss_weights,
        label="Validation Supervised Loss",
        marker="+",
        color="#224d1c",
    )
    ax2.legend(loc=3)
    ax2.set_ylabel("Supervised Loss", fontsize=MINOR)
    ax2.set_yscale("log")
    if config is not None:
        plt.title(
            f"Losses over the training of {prefix} v{config['net']['version']}.{config['net']['iteration-save']}"
        )
    else:
        plt.title("Losses over the training")
    plt.savefig(
        f"{config['path']['local-path']}{config['path']['plots-path']}/Training/{prefix}_losses_comparison_{config['net']['version']}.{config['net']['iteration-save']}.png",
        transparent=False,
    )


def plot_losses_composition(losses_composition: tuple[dict, dict], config=None):
    """Plot the composition of the losses as two stackplots, on the left the training and the right the validation loss.
        #### If changing the number of element in the loss composition, change also the colors list.

    Args:
        losses_composition (tuple[dict,dict]): Dictionnaries containing the various components of the loss expressed as the fraction of the total loss.
        config (dict, optional): Config. Defaults to None.
    """
    epochs = np.arange(len(losses_composition[0]["Scale loss training"]))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9), constrained_layout=True)
    axes[0].stackplot(
        epochs,
        losses_composition[0].values(),
        labels=losses_composition[0].keys(),
        colors=["#003f5c", "#58508d", "#bc5090", "#ff6361"],
    )
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Proportion")
    axes[0].set_xlim(0, epochs[-1])
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks(np.arange(0.0, 1.0, 0.1))
    axes[0].legend(loc="lower left")

    axes[1].stackplot(
        epochs,
        losses_composition[1].values(),
        labels=losses_composition[1].keys(),
        colors=["#003f5c", "#58508d", "#bc5090", "#ff6361"],
    )
    axes[1].set_xlabel("Epochs")
    axes[1].set_xlim(0, epochs[-1])
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks(np.arange(0.0, 1.0, 0.1))
    axes[1].legend(loc="lower left")

    fig.suptitle("Compositions of the losses")

    if config is not None:
        plt.savefig(
            f"{config['path']['local-path']}{config['path']['plots-path']}/Training/Losses_composition_{config['net']['version']}.{config['net']['iteration-save']}.png",
            transparent=False,
        )


def _plot_reconstructions(
    x_data: ndarray,
    y_data: ndarray,
    y_recon: ndarray,
    axis: Axes,
    predicted_targets=None,
    targets=None,
    parameters_names=None,
    noise_spectra=None,
    predicted_noise_spectra=None,
) -> Axes:
    """
    Plots reconstructions and residuals

    Parameters
    ----------
    x_data : ndarray
        Energy values
    y_data : ndarray
        Spectrum
    y_recon : ndarray
        Reconstructed Spectrum
    axis : Axes
        Plot axes

    Returns
    -------
    Axes
        Spectrum axis
    """
    if len(y_recon.shape) == 1:  # if no multi scenario then just one residual
        if (
            noise_spectra is not None
        ):  # if we have noise we express the residuals in std units
            axis.scatter(
                x_data,
                (y_recon - y_data) / noise_spectra,
                marker="x",
                label="Residual in std",
            )
        else:  # else we just plot the residuals
            axis.scatter(x_data, y_recon - y_data, marker="x", label="Residual")
    else:  # if multi scenarios, than one residual for each scenarios with a decreasing alpha. Standard scenario color from color_list
        for i in range(y_recon.shape[0]):
            if (
                noise_spectra is not None
            ):  # if we have noise we express the residuals in std units
                axis.scatter(
                    x_data,
                    (y_recon[i] - y_data) / noise_spectra,
                    marker="x",
                    alpha=2 / y_recon.shape[0],
                    c=color_list[i],
                )
            else:
                axis.scatter(
                    x_data,
                    y_recon[i] - y_data,
                    marker="x",
                    alpha=2 / y_recon.shape[0],
                    c=color_list[i],
                )
    axis.locator_params(axis="y", nbins=3)
    axis.tick_params(axis="both", labelsize=MINOR)
    axis.hlines(0, xmin=np.min(x_data), xmax=np.max(x_data), color="k")

    divider = make_axes_locatable(axis)
    axis_2 = divider.append_axes(
        "top", size="150%", pad=0
    )  # We add the reconstructions plot on top of the residuals

    if noise_spectra is None:  # if we don't have noise, just a usual scatter plot
        axis_2.scatter(x_data, y_data, label="Spectrum")
    else:  # if we have noise, plot errorbars
        axis_2.errorbar(x_data, y_data, yerr=noise_spectra, label="Spectrum", fmt="o")

    if len(y_recon.shape) == 1:
        if predicted_noise_spectra is None:
            axis_2.scatter(x_data, y_recon, label="Reconstruction")
        else:
            axis_2.errorbar(
                x_data,
                y_recon,
                yerr=predicted_noise_spectra,
                label="Reconstruction",
                fmt="o",
            )
    else:
        for i in range(y_recon.shape[0]):
            if predicted_noise_spectra is None:
                axis_2.scatter(
                    x_data,
                    y_recon[i],
                    label=f"{i}",
                    alpha=0.2 + 0.8 / y_recon.shape[0],
                    c=color_list[i],
                )
            else:
                axis_2.errorbar(
                    x_data,
                    y_recon[i],
                    yerr=predicted_noise_spectra[i],
                    label=f"{i}",
                    alpha=0.2 + 0.8 / y_recon.shape[0],
                    c=color_list[i],
                )
    axis_2.locator_params(axis="y", nbins=5)
    axis_2.tick_params(axis="y", labelsize=MINOR)
    axis_2.set_xticks([])

    if predicted_targets is not None:
        # divider = make_axes_locatable(axis_2)
        axis_3 = divider.append_axes("bottom", size="100%")
        axis_3.plot(parameters_names, targets, "+", label="Ground truth")
        axis_3.plot(
            parameters_names + ["Degeneracy"],
            np.zeros(targets.shape[0] + 1),
            color="black",
            alpha=0.4,
        )
        axis_3.plot(
            parameters_names + ["Degeneracy"],
            np.ones(targets.shape[0] + 1),
            color="black",
            alpha=0.4,
        )
        if len(predicted_targets.shape) == 1:
            axis_3.plot(parameters_names, predicted_targets, "^", label="Predicted")
        else:
            predicted_targets[:, -1] = np.exp(predicted_targets[:, -1]) / (
                1 + np.exp(predicted_targets[:, -1])
            )
            for i in range(predicted_targets.shape[0]):
                axis_3.plot(
                    parameters_names + ["Degeneracy"],
                    predicted_targets[i],
                    "^",
                    label=f"{i}",
                    color=color_list[i],
                )
        return axis_3

    return axis_2


def _plot_histogram(title: str, data: ndarray, data_twin: ndarray, axis: Axes) -> Axes:
    """
    Plots a histogram subplot with twin data

    Parameters
    ----------
    title : string
        Title of subplot
    data : ndarray
        Primary data to plot
    data_twin : ndarray
        Secondary data to plot
    axis : Axes
        Axis to plot on

    Returns
    -------
    Axes
        Twin axis
    """
    twin_axis = axis.twinx()

    axis.set_title(title, fontsize=MAJOR)
    axis.hist(data, bins=100, alpha=0.5, density=True, label="Target")
    twin_axis.hist(
        data_twin, bins=100, alpha=0.5, density=True, label="Predicted", color="orange"
    )

    return twin_axis


def plot_degeneracy_distribution(predicted_targets: ndarray):
    degeneracies = (
        (
            np.exp(predicted_targets[:, :, -1])
            / (1 + np.exp(predicted_targets[:, :, -1]))
        )
        .mean(axis=1)
        .flatten()
    )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(degeneracies, bins=100, density=True)
    ax.set_xlabel("Degeneracy")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of the predicted degeneracies on the whole dataset")


def plot_saliency(spectra: ndarray, predictions: ndarray, saliencies: ndarray, config):
    """
    Plots saliency map for the autoencoder

    Parameters
    ----------
    plots_dir : string
        Directory to save plots
    spectra : Tensor
        Target spectra
    saliencies : Tensor
        Saliency
    """
    # Constants
    alpha = 0.8
    cmap = plt.cm.hot
    x_data = load_x_data(spectra.shape[1], config)

    x_regions = x_data[::12]
    saliencies = np.mean(saliencies.reshape(saliencies.shape[0], -1, 12), axis=-1)
    # saliencies = data_normalization(saliencies, mean=False, axis=1)[0] * 0.9 + 0.05

    # Initialize e_saliency plots
    _, axes = plt.subplots(
        2, 4, figsize=(24, 12), sharex="col", gridspec_kw={"hspace": 0, "wspace": 0}
    )
    axes = axes.flatten()

    # Plot each saliency map
    for i, (axis, spectrum, prediction, saliency) in enumerate(
        zip(axes, spectra, predictions, saliencies)
    ):
        for j, (x_region, saliency_region) in enumerate(
            zip(x_regions[:-1], saliency[:-1])
        ):
            axis.axvspan(
                x_region, x_regions[j + 1], color=cmap(saliency_region), alpha=alpha
            )

        axis.axvspan(x_regions[-1], x_data[-1], color=cmap(saliency[-1]), alpha=alpha)

        axis.scatter(x_data, spectrum, label="Target")
        axis.scatter(x_data, prediction, color="g", label="Prediction")
        axis.text(0.9, 0.9, i + 1, fontsize=MAJOR, transform=axis.transAxes)
        axis.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False)

    legend = plt.figlegend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.92),
        fontsize=MAJOR,
        columnspacing=10,
    )
    legend.get_frame().set_alpha(None)

    plt.figtext(0.5, 0.02, "Energy (keV)", ha="center", va="center", fontsize=MAJOR)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(
        f"{config['path']['local-path']}{config['path']['plots-path']}/Saliency/Saliency_Plot_{config['net']['version']}.{config['net']['iteration-save']}.png",
        transparent=False,
    )


def plot_param_comparison(
    param_names: list[str], target: ndarray, predictions: ndarray, config: dict
):
    """
    Plots predictions against target for each parameter

    Parameters:
    ----------
    plots_dir : string
        Directory to save plots
    param_names : list[string]
        List of parameter names
    target : ndarray
        Target parameters
    predictions : ndarray
        Parameter predictions
    """
    # _, axes = plt.subplot_mosaic('AABBCC;DDDEEE', constrained_layout=True, figsize=(16, 9))
    fig, axes = plt.subplots(
        nrows=int(np.ceil(len(param_names) / 3)),
        ncols=3,
        figsize=(16, 4 * int(np.ceil(len(param_names) / 3))),
    )

    # Plot each parameter
    for i in range(len(param_names)):
        value_range = [
            min(np.min(target[:, i]), np.min(predictions[:, i])),
            max(np.max(target[:, i]), np.max(predictions[:, i])),
        ]
        axis = axes[i // 3, i % 3]
        axis.scatter(
            target[:, i],
            predictions[:, i],
            alpha=1 / (1 + target.shape[0] / 10000),
            s=1,
        )
        axis.set_xlabel(f"{param_names[i]} ground truth")
        axis.set_ylabel(f"{param_names[i]} predicted")
        axis.plot(value_range, value_range, color="k")
        axis.set_title(param_names[i])

    if config["model"]["log-parameters"]:
        for i in config["model"]["log-parameters"]:
            axes[i // 3, i % 3].set_xscale("log")
            axes[i // 3, i % 3].set_yscale("log")

    fig.suptitle(
        f"Predicted parameters compared to ground truth for Encoder for v{config['net']['version']}.{config['net']['iteration-save']}"
    )
    fig.subplots_adjust(hspace=0.4)

    plt.savefig(
        f"{config['path']['local-path']}{config['path']['plots-path']}/Parameter_comparison/Parameter_comparison_Plot_{config['net']['version']}.{config['net']['iteration-save']}.png",
        transparent=False,
    )
    plt.show()


def plot_training(
    prefix: str,
    config: dict,
    losses: tuple[list, list],
    original_spectra: ndarray = None,
    predicted_spectra: ndarray = None,
    original_scale: ndarray = None,
    predicted_scale: ndarray = None,
    scale_names: list[str] = None,
    log_scales: list[int] = [],
    latent_losses: tuple[list, list] = None,
    losses_composition: tuple[dict, dict] = None,
    initial_latent_loss_weight: float = 1e-3,
    latent_loss_weight_scheduler=None,
    targets: ndarray = None,
    predicted_targets: ndarray = None,
    noise_spectra: ndarray = None,
    predicted_noise_spectra: ndarray = None,
    noise_scale: ndarray = None,
    predicted_noise_scale: ndarray = None,
):
    """
    Initializes & plots reconstruction & loss plots

    Parameters
    ----------
    prefix : string
        Name prefix for plots
    plots_dir : string
        Directory to save plots
    losses : tuple[list, list]
        Training & validation losses
    spectra : ndarray
        Original spectra
    predicted_spectra : ndarray
        Reconstructions
    """

    # Plot spectrum reconstruction
    if original_spectra is not None:
        x_data = load_x_data(original_spectra.shape[-1], config)

        # Plot reconstructions
        if predicted_targets is None:
            # Initialize reconstructions plots
            fig, axes = plt.subplots(
                2,
                2,
                figsize=(24, 12),
                sharex="col",
                gridspec_kw={
                    "top": 0.92,
                    "bottom": 0.07,
                    "left": 0.08,
                    "right": 0.99,
                    "hspace": 0.05,
                    "wspace": 0.15,
                },
            )

            axes = axes.flatten()
            if predicted_noise_spectra is None:
                for spectrum, output, axis in zip(
                    original_spectra, predicted_spectra, axes
                ):
                    main_axis = _plot_reconstructions(x_data, spectrum, output, axis)
            else:
                for spectrum, output, axis, no, predicted_no in zip(
                    original_spectra,
                    predicted_spectra,
                    axes,
                    noise_spectra,
                    predicted_noise_spectra,
                ):
                    main_axis = _plot_reconstructions(
                        x_data,
                        spectrum,
                        output,
                        axis,
                        noise_spectra=no,
                        predicted_noise_spectra=predicted_no,
                    )

        else:
            fig, axes = plt.subplots(
                2,
                2,
                figsize=(24, 24),
                sharex="col",
                gridspec_kw={
                    "top": 0.92,
                    "bottom": 0.07,
                    "left": 0.08,
                    "right": 0.99,
                    "hspace": 0.05,
                    "wspace": 0.15,
                },
            )

            axes = axes.flatten()

            if predicted_noise_spectra is None:
                for spectrum, output, axis, target, predicted_target in zip(
                    original_spectra,
                    predicted_spectra,
                    axes,
                    targets,
                    predicted_targets,
                ):
                    main_axis = _plot_reconstructions(
                        x_data,
                        spectrum,
                        output,
                        axis,
                        targets=target,
                        predicted_targets=predicted_target,
                        parameters_names=config["model"]["parameter-names"],
                    )
            else:
                for (
                    spectrum,
                    output,
                    axis,
                    target,
                    predicted_target,
                    no,
                    predicted_no,
                ) in zip(
                    original_spectra,
                    predicted_spectra,
                    axes,
                    targets,
                    predicted_targets,
                    noise_spectra,
                    predicted_noise_spectra,
                ):
                    main_axis = _plot_reconstructions(
                        x_data,
                        spectrum,
                        output,
                        axis,
                        targets=target,
                        predicted_targets=predicted_target,
                        parameters_names=config["model"]["parameter-names"],
                        noise_spectra=no,
                        predicted_noise_spectra=predicted_no,
                    )

        plt.figtext(
            0.5, 0.02, "Wavelength ($\mu m$)", ha="center", va="center", fontsize=MAJOR
        )
        plt.figtext(
            0.02,
            0.5,
            "Scaled Spectrum (0-1)",
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=MAJOR,
        )

        # labels = np.hstack((main_axis.get_legend_handles_labels(), axes[0].get_legend_handles_labels()))
        # labels= axes[0].get_legend_handles_labels()
        labels = main_axis.get_legend_handles_labels()
        legend = plt.figlegend(
            *labels,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.92),
            fontsize=MAJOR,
            markerscale=2,
            columnspacing=10,
        )
        legend.get_frame().set_alpha(None)

        fig.suptitle(
            f"Reconstructions of 4 random spectra for {prefix} v{config['net']['version']}.{config['net']['iteration-save']}"
        )

        plt.savefig(
            f"{config['path']['local-path']}{config['path']['plots-path']}/Reconstruction/{prefix}_reconstruction_plot_{config['net']['version']}.{config['net']['iteration-save']}.png",
            transparent=False,
        )

    # Plot scale reconstruction
    if original_scale is not None:
        fig2, axes2 = plt.subplots(
            nrows=1, ncols=len(original_scale[0]), figsize=(16, 4)
        )

        for i in range(len(original_scale[0])):
            axis = axes2[i]
            if len(predicted_scale.shape) == 2:
                value_range = [
                    min(np.min(original_scale[:, i]), np.min(predicted_scale[:, i])),
                    max(np.max(original_scale[:, i]), np.max(predicted_scale[:, i])),
                ]
                if (
                    noise_scale is not None
                ):  # if we have uncertainty on the scale, we plot it
                    axis.errorbar(
                        original_scale[:, i],
                        predicted_scale[:, i],
                        xerr=noise_scale[:, i],
                        yerr=predicted_noise_scale[:, i],
                        fmt="o",
                        alpha=1 / (1 + original_scale.shape[0] / 10000),
                        ms=2,
                        elinewidth=0.5,
                        ecolor="#2693de",
                    )
                else:
                    axis.scatter(
                        original_scale[:, i],
                        predicted_scale[:, i],
                        alpha=1 / (1 + original_scale.shape[0] / 10000),
                        s=1,
                    )
            else:
                value_range = [
                    min(np.min(original_scale[:, i]), np.min(predicted_scale[:, :, i])),
                    max(np.max(original_scale[:, i]), np.max(predicted_scale[:, :, i])),
                ]
                for j in range(predicted_scale.shape[1]):
                    if noise_scale is not None:
                        axis.errorbar(
                            original_scale[:, i],
                            predicted_scale[:, j, i],
                            xerr=noise_scale[:, i],
                            yerr=predicted_noise_scale[:, j, i],
                            fmt="o",
                            alpha=1 / predicted_scale.shape[1],
                            ms=2 / (1 + original_scale.shape[0] / 10000),
                            label=f"{j}",
                            c=color_list[j],
                            elinewidth=0.5,
                        )
                    else:
                        axis.scatter(
                            original_scale[:, i],
                            predicted_scale[:, j, i],
                            alpha=1 / predicted_scale.shape[1],
                            s=1 / (1 + original_scale.shape[0] / 10000),
                            label=f"{j}",
                            c=color_list[j],
                        )
                lgnd = axis.legend()
                for h in lgnd.legendHandles:
                    h._sizes = [10]
            axis.set_xlabel(f"{scale_names[i]} ground truth")
            axis.set_ylabel(f"{scale_names[i]} predicted")
            axis.plot(value_range, value_range, color="k")
            axis.set_title(scale_names[i])

        for i in log_scales:
            axes2[i].set_yscale("log")

        fig2.suptitle(
            f"Predicted scales compared to ground truth for {prefix} v{config['net']['version']}.{config['net']['iteration-save']}"
        )
        fig2.subplots_adjust(hspace=0.4)

        plt.savefig(
            f"{config['path']['local-path']}{config['path']['plots-path']}/Reconstruction/Scale_reconstruction_{prefix}_{config['net']['version']}.{config['net']['iteration-save']}.png",
            transparent=False,
        )

    # Plot loss over epochs
    _plot_loss(losses, latent_losses)
    plt.title(
        f"Loss evolution for {prefix} v{config['net']['version']}.{config['net']['iteration-save']}"
    )
    plt.savefig(
        f"{config['path']['local-path']}{config['path']['plots-path']}/Training/{prefix}_training_loss_{config['net']['version']}.{config['net']['iteration-save']}.png",
        transparent=False,
    )

    # Plot losses compositions
    if losses_composition is not None:
        plot_losses_composition(losses_composition, config)

    # Plot losses comparison
    if latent_losses is not None:
        plot_loss_comparison(
            latent_losses,
            initial_latent_loss_weight=initial_latent_loss_weight,
            latent_loss_weight_scheduler=latent_loss_weight_scheduler,
            config=config,
        )

    # Plot degeneracy distribution
    if (
        predicted_targets is not None
        and predicted_targets.shape[-1] == targets.shape[-1] + 1
    ):
        plot_degeneracy_distribution(predicted_targets)
        plt.savefig(
            f"{config['path']['local-path']}{config['path']['plots-path']}/Reconstruction/Degeneracy_Distribution_{config['net']['version']}.{config['net']['iteration-save']}.png",
            transparent=False,
        )
        plt.show()


def plot_param_distribution(
    param_names: list[str], targets: ndarray, predictions: ndarray
):
    """
    Plots histogram of each parameter for both true and predicted

    Parameters
    ----------
    plots_dir : string
        Directory to plots
    param_names : list[string]
        Names for each parameter
    params : Tensor
        Parameter predictions from CNN
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    """
    # param_names = ['nH', r'$\Gamma', 'FracSctr', r'$T_{max}$', 'Norm']

    # _, axes = plt.subplot_mosaic('AABBCC;DDDEEE', figsize=(16, 9))
    fig, axes = plt.subplots(nrows=int(np.ceil(len(param_names) / 3)), ncols=3)

    # Plot subplots
    # for i, (title, axis) in enumerate(zip(param_names, axes.values())):
    #     twin_axis = _plot_histogram(title, params_real[i], params[i], axis)

    # legend = plt.figlegend(
    #     axes['A'].get_legend_handles_labels()[0] + twin_axis.get_legend_handles_labels()[0],
    #     axes['A'].get_legend_handles_labels()[1] + twin_axis.get_legend_handles_labels()[1],
    #     fontsize=MAJOR,
    #     bbox_to_anchor=(0.95, 0.45),
    # )
    # legend.get_frame().set_alpha(None)
    # plt.tight_layout()
    # plt.savefig(f'{config['path']['local-path']}{config['path']['plots-path']}Param_Distribution')


def plot_difference(
    x_data_1: ndarray, y_data_1: ndarray, x_data_2: ndarray, y_data_2: ndarray
):
    """
    Plots the ratio between two data sets (set 1 / set 2)

    Parameters
    ----------
    x_data_1 : ndarray
        x values for first data set
    y_data_1 : ndarray
        y values for first data set
    x_data_2 : ndarray
        x values for second data set
    y_data_2 : ndarray
        y values for second data set
    """
    matching_indices = np.array((), dtype=int)

    for i in x_data_1:
        matching_indices = np.append(matching_indices, np.argmin(np.abs(x_data_2 - i)))

    diff = y_data_1 / y_data_2[matching_indices]

    plt.title("PyXspec compared to fits", fontsize=MAJOR)
    plt.scatter(x_data_1, diff)
    plt.xlabel("Energy (keV)", fontsize=MINOR)
    plt.ylabel("PyXspec / fits data", fontsize=MINOR)
    plt.text(
        0.05,
        0.2,
        f"Average ratio: {round(np.mean(diff), 3)}",
        fontsize=MINOR,
        transform=plt.gca().transAxes,
    )


def plot_spectrum(x_bin: ndarray, y_bin: ndarray, x_px: ndarray, y_px: ndarray):
    """
    Plots the spectrum of PyXspec & fits data

    Parameters
    ----------
    x_bin : ndarray
        Binned x data from fits file
    y_bin : ndarray
        Binned y data from fits file
    x_px : ndarray
        x data from PyXspec
    y_px : ndarray
        y data from PyXspec
    """
    plt.title("Spectrum of PyXspec & fits", fontsize=MAJOR)
    plt.xlabel("Energy (keV)", fontsize=MINOR)
    plt.ylabel(r"Counts $s^{-1}$ $detector^{-1}$ $keV^{-1}$", fontsize=MINOR)
    plt.scatter(x_bin, y_bin, label="Fits data", marker="x")
    plt.scatter(x_px, y_px, label="PyXspec data")
    plt.xlim([0.15, 14.5])
    # plt.xscale('log')
    # plt.yscale('log')
    plt.legend(fontsize=MINOR)


def _plot_element_loss_pairplot(
    ax,
    feature_ind1,
    feature_ind2,
    _features,
    _loss,
    _features_names,
    color_map,
    vmin,
    vmax,
):
    """Plots single pair of features.

    Parameters
    ----------
    ax : Axes
        matplotlib axis to be plotted
    feature_ind1 : int
        index of first feature to be plotted
    feature_ind2 : int
        index of second feature to be plotted
    _X : numpy.ndarray
        Feature dataset of of shape m x n
    _y : numpy.ndarray
        Target list of shape 1 x n
    _features : list of str
        List of n feature titles
    colormap : dict
        Color map of classes existing in target

    Returns
    -------
    None
    """
    # print(f"i={feature_ind1} j={feature_ind2}")

    df = pd.DataFrame(_features, columns=_features_names)
    df["loss"] = _loss
    n_bins = 30

    # Plot distribution histogram if the features are the same (diagonal of the pair-plot).
    if feature_ind1 == feature_ind2:
        a = 0
        ax[feature_ind1, feature_ind2].get_xaxis().set_visible(False)
        ax[feature_ind1, feature_ind2].get_yaxis().set_visible(False)
        _min = df[_features_names[feature_ind1]].min()
        _max = df[_features_names[feature_ind1]].max()
        lsp = np.linspace(start=_min, stop=_max, num=n_bins)
        df[_features_names[feature_ind1] + "_cut"] = pd.cut(
            df[_features_names[feature_ind1]], bins=lsp, labels=lsp[:-1]
        )
        _mean_loss = (
            df.sort_values([_features_names[feature_ind1] + "_cut"])
            .groupby(_features_names[feature_ind1] + "_cut")["loss"]
            .mean()
        )
        # print(_mean_loss)
        for i in range(len(lsp) - 1):
            ax[feature_ind1, feature_ind2].hist(
                x=df[_features_names[feature_ind1] + "_cut"],
                bins=[lsp[i], lsp[i + 1]],
                color=color_map((_mean_loss.iloc[i] - vmin) / (vmax - vmin)),
            )

    else:
        # other wise plot the pair-wise scatter plot
        ax[feature_ind1, feature_ind2].scatter(
            x=_features[:, feature_ind2],
            y=_features[:, feature_ind1],
            s=0.1 / len(df) * 100000,
            c=_loss,
            cmap=color_map,
            vmin=vmin,
            vmax=vmax,
        )
        ax[feature_ind1, feature_ind2].get_xaxis().set_visible(False)
        ax[feature_ind1, feature_ind2].get_yaxis().set_visible(False)
    # Print the feature labels only on the left side of the pair-plot figure
    # and bottom side of the pair-plot figure.
    # Here avoiding printing the labels for inner axis plots.
    if feature_ind1 == len(_features_names) - 1:
        ax[feature_ind1, feature_ind2].set(
            xlabel=_features_names[feature_ind2], ylabel=""
        )
        ax[feature_ind1, feature_ind2].get_xaxis().set_visible(True)
    if feature_ind2 == 0:
        ax[feature_ind1, feature_ind2].get_yaxis().set_visible(True)
        if feature_ind1 == len(_features_names) - 1:
            ax[feature_ind1, feature_ind2].set(
                xlabel=_features_names[feature_ind2],
                ylabel=_features_names[feature_ind1],
            )
        else:
            ax[feature_ind1, feature_ind2].set(
                xlabel="", ylabel=_features_names[feature_ind1]
            )


def plot_loss_pairplot(features, loss, feature_names, vmin, vmax, config):
    """Plots a pair grid of the given features.

    Parameters
    ----------
    X : numpy.ndarray
        Dataset of shape m x n
    y : numpy.ndarray
        Target list of shape 1 x n
    features : list of str
        List of n feature titles

    Returns
    -------
    None
    """
    cm = plt.cm.get_cmap("plasma")

    feature_count = len(feature_names)
    # Create a matplot subplot area with the size of [feature count x feature count]
    fig, axis = plt.subplots(nrows=feature_count, ncols=feature_count)
    # Setting figure size helps to optimize the figure size according to the feature count.
    fig.set_size_inches(feature_count * 4, feature_count * 4)

    # Iterate through features to plot pairwise.
    for i in range(0, feature_count):
        for j in range(0, feature_count):
            _plot_element_loss_pairplot(
                axis, i, j, features, loss, feature_names, cm, vmin, vmax
            )

    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cm
        ),
    )
    # fig.suptitle("Log of the loss of the decoder in the parameter space")
    plt.savefig(
        f"{config['path']['local-path']}{config['path']['plots-path']}/Pairplot_loss/Pairplot_loss_{config['net']['version']}.{config['net']['iteration-save']}.png",
        transparent=False,
    )
    plt.show()


def plot_loss_ft(
    config,
    loss,
    loss_q1,
    loss_q2,
    x_axis=None,
    q=0.1,
    xlabel="Wavelength (in microns)",
    log=True,
):
    plt.figure(figsize=(16, 9))
    if x_axis is None:
        x_axis = load_x_data(64, config)
    plt.bar(x_axis, loss, width=(x_axis[1] - x_axis[0]) * 1.9)
    yerr = np.array([loss - loss_q1, loss_q2 - loss])
    yerr *= yerr > 0
    plt.errorbar(
        x_axis, loss, yerr=yerr, fmt="o", label=f"quartile q1={q}, q2={1-q}", c="r"
    )
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("Mean loss")
    if log:
        plt.yscale("log")
    plt.title(
        f"Mean loss of each feature for v{config['net']['version']}.{config['net']['iteration-save']} and quartiles"
    )
    plt.savefig(
        f"{config['path']['local-path']}{config['path']['plots-path']}/Feature_loss/Feature_loss_{config['net']['version']}.{config['net']['iteration-save']}.png",
        transparent=False,
    )
    plt.show()


####### PLOTS FOR SBI OR ZUKO NORMALISING FLOWS ########


def plot_posteriors_comparison(
    posterior,
    transform_ft,
    transform_tg,
    transform_au,
    transform_no,
    config,
    test_size=4000,
    R_star_transform=True,
    ID: float = -1,
    device=torch.device("cpu"),
):
    path = f"{config['path']['local-path']}FullDataset/TrainingData/"
    infile = h5py.File(
        f"{path}Ground Truth Package/Tracedata.hdf5"
    )  ### loading in tracedata.hdf5
    SpectralData = h5py.File(f"{path}SpectralData.hdf5")  ### load file

    full_df = pd.read_pickle(f"{path}full_df.pt")
    planetlist = [p for p in SpectralData.keys()]
    # ids=[int(p[12:]) for p in planetlist] # used to reorder correctly planetlist so that its the same order as the torch tensors
    planetlist = np.array(planetlist)[-test_size:]  # keeps only the test_set

    if ID == -1.0:
        i = np.random.randint(len(planetlist))
        while not infile[planetlist[i]]["tracedata"].shape:
            i = np.random.randint(len(planetlist))
        p = planetlist[i]
        print(p)
        ID = float(p[12:])
    else:
        p = f"Planet_train{int(ID)}"

    auxparams = [
        "star_distance",
        "star_mass_kg",
        "star_radius_m",
        "star_temperature",
        "planet_mass_kg",
        "planet_orbital_period",
        "planet_distance",
        "planet_surface_gravity",
    ]
    params_names = [
        "planet_radius",
        "planet_temp",
        "log_H2O",
        "log_CO2",
        "log_CO",
        "log_CH4",
        "log_NH3",
    ]
    columns = ["ID"]
    id_noise = 1
    for wl in SpectralData[planetlist[0]]["instrument_wlgrid"]:
        columns.append(f"spectrum_{wl}")
        id_noise += 1
    for wl in SpectralData[planetlist[0]]["instrument_wlgrid"]:
        columns.append(f"noise_{wl}")

    planet_df = full_df[full_df["ID"] == ID]
    pspectrum = transform_ft(
        torch.tensor(planet_df[columns[1:id_noise]].values).float().to(device)
    )
    pnoises = transform_no(
        torch.tensor(planet_df[columns[id_noise:]].values).float().to(device)
    )
    ptarget = torch.tensor(planet_df[params_names].values).float().to(device)
    paux = transform_au(torch.tensor(planet_df[auxparams].values).float().to(device))
    pfeature = torch.cat([paux, pspectrum, pnoises], dim=1)

    if R_star_transform:
        _aux = torch.tensor(planet_df[auxparams].values).float().to(device)
        transform_tg.transforms_list[0].transform.scale = _aux[
            0, 2
        ]  # we have to adjust the transform so that it uses the correct R_star
        transform_tg.transforms_list[0].transform.min = 0.0

    trace = infile[p]["tracedata"][:]  ### accessing Nested Sampling trace data
    weights = infile[p]["weights"][:]

    ground_truth_sample = torch.Tensor(
        trace[np.random.choice(len(weights), size=5000, p=weights)]
    ).cpu()
    predicted_sample = transform_tg.inverse(
        posterior.sample((5000,), x=pfeature.cpu()).to(device)
    ).cpu()

    ground_truth_quartile_1 = torch.quantile(
        ground_truth_sample, q=0.01, dim=0, keepdim=True
    )
    ground_truth_quartile_2 = torch.quantile(
        ground_truth_sample, q=0.99, dim=0, keepdim=True
    )
    predicted_sample_quartile_1 = torch.quantile(
        predicted_sample, q=0.01, dim=0, keepdim=True
    )
    predicted_sample_quartile_2 = torch.quantile(
        predicted_sample, q=0.99, dim=0, keepdim=True
    )
    limits = torch.ones(ground_truth_sample.shape[1], 2)
    limits[:, 0], _ = torch.min(
        torch.cat([ground_truth_quartile_1, predicted_sample_quartile_1], dim=0), dim=0
    )
    limits[:, 1], _ = torch.max(
        torch.cat([ground_truth_quartile_2, predicted_sample_quartile_2], dim=0), dim=0
    )

    fig1 = corner.corner(
        trace,
        labels=params_names,
        weights=weights,
        color="#1e4195",
        range=limits,
        scale_hist=True,
        smooth=1.0,
    )
    #
    fig2 = corner.corner(
        predicted_sample.cpu().numpy(),
        labels=params_names,
        truths=ptarget[0].cpu().numpy(),
        truth_color="r",
        color="#1e955a",
        fig=fig1,
        range=limits,
        weights=np.ones(5000) / 5000,
        scale_hist=True,
        smooth=1.0,
    )
    fig2.suptitle(
        f"Given distribution (blue) and posterior (green) compared to truth (red) for sample {int(ID)}, version {config['net']['version']}.{config['net']['iteration-save']}"
    )
    plt.savefig(
        f"{config['path']['local-path']}{config['path']['plots-path']}Distributions/Posterior&GroundTruth_{p}_{config['net']['decoder-name']}_{config['net']['version']}.{config['net']['iteration-save']}.png"
    )


def plot_coverage(
    ranks,
    config,
    labels=[
        "Planet Radius",
        "Planet Temperature",
        "log H2O",
        "log CO2",
        "log CO",
        "log CH4",
        "log NH3",
    ],
    mean_too=True,
):
    # colors=["#3f7dae","#ae703f",'#117238', '#4e6206', '#6e4d00','#823312','#851433']):

    xr = np.sort(ranks, axis=0)
    xr = xr.astype(dtype=float)
    xr /= xr[-1]
    cdf = np.linspace(0.0, 1.0, len(xr))
    plt.figure(figsize=(4, 4), dpi=300)
    plt.plot(cdf, cdf, color="black", ls="--")
    for i in range(len(xr[0])):
        plt.plot(xr[:, i], cdf, alpha=0.7, label=labels[i])
    plt.ylabel("True coverage")
    plt.xlabel("Predicted coverage")
    plt.grid(which="both", lw=0.5)
    plt.title(
        f"Coverage for each parameter for versions {config['net']['version']}.{config['net']['iteration-save']}"
    )
    plt.legend(loc='upper left')
    plt.savefig(
        f"{config['path']['local-path']}{config['path']['plots-path']}Distributions/Coverage_decomposed_{config['net']['decoder-name']}_{config['net']['version']}.{config['net']['iteration-save']}.png"
    )
    if mean_too:
        xr=xr.mean(axis=1)
        plt.figure(figsize=(4, 4), dpi=300)
        plt.plot(cdf, cdf, color="black", ls="--")
        plt.plot(xr,cdf)
        plt.ylabel("True mean coverage")
        plt.xlabel("Predicted mean coverage")
        plt.grid(which="both", lw=0.5)
        plt.title(
            f"Mean coverage for version {config['net']['version']}.{config['net']['iteration-save']}"
        )
        plt.savefig(
            f"{config['path']['local-path']}{config['path']['plots-path']}Distributions/Coverage_mean_{config['net']['decoder-name']}_{config['net']['version']}.{config['net']['iteration-save']}.png"
        )


def plot_param_comparison_from_samples(samples, targets, config):
    samples = np.array(samples)
    targets = np.array(targets)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 16))
    mean = samples.mean(axis=1)
    targets = np.array(targets)
    params_names = [
        "Planet Radius",
        "Planet Temperature",
        "log H2O",
        "log CO2",
        "log CO",
        "log CH4",
        "log NH3",
    ]

    for i in range(len(params_names)):
        value_range = [
            min(np.min(targets[:, i]), np.min(mean[:, i])),
            max(np.max(targets[:, i]), np.max(mean[:, i])),
        ]
        axis = axes[i // 3, i % 3]
        axis.scatter(targets[:, i], mean[:, i], s=1, alpha=0.5)
        axis.plot(value_range, value_range, ls="--", color="black")
        axis.set_xlabel(f"{params_names[i]} ground truth")
        axis.set_ylabel(f"{params_names[i]} predicted")
    fig.savefig(
        f"{config['path']['local-path']}{config['path']['plots-path']}Parameter_comparison/Parameter_comparison_NS_{config['net']['decoder-name']}_{config['net']['version']}.{config['net']['iteration-save']}.png"
    )


def plot_param_comparison_heatmap(samples, 
                                  targets, 
                                  config,
                                  bins=1000,
                                  q=0.05,
                                  one_parameter=False,
                                  cmap='viridis'):
    params_names = [
        "Planet Radius",
        "Planet Temperature",
        "log H2O",
        "log CO2",
        "log CO",
        "log CH4",
        "log NH3",
    ]
    samples = np.array(samples)
    targets = np.array(targets)
    
    value_range = [
            [np.quantile(targets,q=q,axis=0), np.quantile(samples,q=q,axis=(0,1))],
            [np.quantile(targets,q=1-q,axis=0), np.quantile(samples,q=1-q,axis=(0,1))],
        ]
    targets=np.expand_dims(targets,axis=-2)
    targets=targets*np.ones_like(samples)
    value_range=np.array(value_range)
    H=[]
    
    
    for i in range(len(params_names)):
        number_of_targets=np.expand_dims(np.histogram(targets[:,0,i], bins=bins, range=[value_range[0,0,i],value_range[1,0,i]])[0],axis=0)
        
        Hi, xedges, yedges = np.histogram2d(samples[:,:,i].flatten(),
                                            targets[:,:,i].flatten(),
                                            bins=bins,
                                            range=[[value_range[0,1,i],value_range[1,1,i]],[value_range[0,0,i],value_range[1,0,i]]],
                                            density=True)
        
        Hi*=((number_of_targets!=0)*np.sum(number_of_targets)/(1e-12+number_of_targets))
        H.append(Hi)
    
    if not one_parameter:
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12), dpi=400)
        for i in range(len(params_names)):
            
            axis = axes[i // 3, i % 3]
            pos=axis.imshow(H[i]*(value_range[1,0,i]-value_range[0,0,i])*(value_range[1,1,i]-value_range[0,1,i]),
                            extent=[value_range[0,0,i],value_range[1,0,i],value_range[0,1,i],value_range[1,1,i]],
                            origin='lower',
                            aspect='auto',
                            norm=matplotlib.colors.LogNorm(vmin=1e1,vmax=1e5),
                            cmap=cmap)
            identity=[np.max(value_range[0,:,i]),np.min(value_range[1,:,i])]
            axis.plot(identity, identity, ls="--", color="black")
            axis.set_xlabel(f"{params_names[i]} input parameter")
            axis.set_ylabel(f"{params_names[i]} samples predicted")
        cbar=fig.colorbar(pos,ax=axes[-1,-1])
        fig.suptitle("Density of the predicted samples for each bin of ground truth")
        fig.savefig(
            f"{config['path']['local-path']}{config['path']['plots-path']}Parameter_comparison/Parameter_comparison_heatmap_{config['net']['version']}.{config['net']['iteration-save']}.png"
        )
    else:
        fig=plt.figure(figsize=(4, 3.2), dpi=400)
        i=one_parameter
        pos=plt.imshow(H[i]*(value_range[1,0,i]-value_range[0,0,i])*(value_range[1,1,i]-value_range[0,1,i]),
                            extent=[value_range[0,0,i],value_range[1,0,i],value_range[0,1,i],value_range[1,1,i]],
                            origin='lower',
                            aspect='auto',
                            norm=matplotlib.colors.LogNorm(vmin=1e1,vmax=1e5),
                            cmap=cmap)
        identity=[np.max(value_range[0,:,i]),np.min(value_range[1,:,i])]
        plt.plot(identity, identity, ls="--", color="black")
        plt.xlabel(f"{params_names[i]} input parameter")
        plt.ylabel(f"{params_names[i]} samples predicted")
        cbar=fig.colorbar(pos)
        plt.title("Density of the predicted samples")
        fig.savefig(
            f"{config['path']['local-path']}{config['path']['plots-path']}Parameter_comparison/Parameter_comparison_heatmap_{params_names[i]}_{config['net']['version']}.{config['net']['iteration-save']}.png"
        )





def plot_several_posteriors(
    list_of_samples,
    config,
    ground_truth=None,
    params_names=[
        "planet_radius",
        "planet_temp",
        "log_H2O",
        "log_CO2",
        "log_CO",
        "log_CH4",
        "log_NH3",
    ],
    colors=["#003f5c", "#7a5195", "#ef5675", "#ffa600"],
    labels=["1st Approach", "2nd Approach", "3rd Approach", "4th Approach"],
    title="Comparison of posteriors",
):
    if ground_truth is not None:
        quantiles_1 = [ground_truth.unsqueeze(dim=0)]
        quantiles_2 = [ground_truth.unsqueeze(dim=0)]
    else:
        quantiles_1 = []
        quantiles_2 = []

    for samples in list_of_samples:
        quantiles_1.append(torch.quantile(samples, q=0.1, dim=0, keepdim=True))
        quantiles_2.append(torch.quantile(samples, q=0.9, dim=0, keepdim=True))

    limits = torch.ones(len(params_names), 2)
    limits[:, 0], _ = torch.min(torch.cat(quantiles_1, dim=0), dim=0)
    limits[:, 1], _ = torch.max(torch.cat(quantiles_2, dim=0), dim=0)
    limits[:,0]=limits[:,0]-0.1*torch.abs(limits[:,0])
    limits[:,1]=limits[:,1]+0.1*torch.abs(limits[:,1])

    if ground_truth is not None:
        fig = corner.corner(
            list_of_samples[0].cpu().numpy(),
            labels=params_names,
            truths=ground_truth.cpu().numpy(),
            truth_color="r",
            range=limits,
            color=colors[0],
            scale_hist=True,
            smooth=1.0,
        )
        legend_lines = [
            matplotlib.lines.Line2D(
                [0],
                [0],
                color="r",
                marker="o",
                markersize=10,
                lw=4,
                label="Input parameters",
            ),
            matplotlib.lines.Line2D([0], [0], color=colors[0], lw=6, label=labels[0]),
        ]

    else:
        fig = corner.corner(
            list_of_samples[0].cpu().numpy(),
            labels=params_names,
            range=limits,
            color=colors[0],
            scale_hist=True,
            smooth=1.0,
        )
        legend_lines = [
            matplotlib.lines.Line2D([0], [0], color=colors[0], lw=6, label=labels[0])
        ]

    for i in range(1, len(list_of_samples)):
        fig = corner.corner(
            list_of_samples[i].cpu().numpy(),
            labels=params_names,
            range=limits,
            color=colors[i],
            scale_hist=True,
            smooth=1.0,
            fig=fig,
        )
        legend_lines.append(
            matplotlib.lines.Line2D([0], [0], color=colors[i], lw=6, label=labels[i])
        )

    fig.suptitle(title, fontsize=15)
    plt.figlegend(handles=legend_lines, loc="upper right", fontsize=12, bbox_to_anchor=(0.95,0.95))
    plt.savefig(
        f"{config['path']['local-path']}{config['path']['plots-path']}Distributions/Several_posteriors_comparison_{title}.png"
    )
