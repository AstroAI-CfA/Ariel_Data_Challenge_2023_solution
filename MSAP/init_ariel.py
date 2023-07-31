"""
Initialization for autoencoders on Ariel data
"""

import pathlib

MSAP_path = pathlib.Path(__file__).parent.resolve()

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from msapnet.spectrum_fit import initialization, predict_parameters
from msapnet.utils.training import training
from msapnet.utils.utils import open_config
from msapnet.utils.plots import *
from msapnet.utils.analysis import param_comparison
from msapnet.utils.transform import *
from msapnet.autoencoder.network import CombinedDecoder
from msapnet.utils.scheduler import *

config_path = f"{MSAP_path}/ariel_config.yaml"

# Open configuration file
_, config = open_config(0, config_path)

# Load parameters
epochs = config["training"]["epochs"]
d_save_name = f"{config['net']['version']}.{config['net']['iteration-save']}"
e_save_name = f"{config['net']['version']}.{config['net']['iteration-save']}"
states_dir = f"{config['path']['local-path']}{config['path']['states-path']}"


# Load the data
features = torch.load(
    f"{config['path']['local-path']+config['path']['data-path']}features_{config['data']['run-id']}.pt"
).to(device)
targets = torch.load(
    f"{config['path']['local-path']+config['path']['data-path']}targets_{config['data']['run-id']}.pt"
).to(device)
noises = torch.load(
    f"{config['path']['local-path']+config['path']['data-path']}noises_{config['data']['run-id']}.pt"
).to(device)
metadatas = torch.load(
    f"{config['path']['local-path']+config['path']['data-path']}metadatas_{config['data']['run-id']}.pt"
)
aux = torch.load(
    f"{config['path']['local-path']+config['path']['data-path']}aux_{config['data']['run-id']}.pt"
).to(device)
freq = torch.load(
    f"{config['path']['local-path']+config['path']['data-path']}freq_1.pt"
)


# Define transforms
# transform_ft=normalisation_uniform_transform(features.clone()) # We normalise the spectra between 0 and 1 with the same scale
# transform_no=ScalarTransform(0,transform_ft.scale) # we use the same scale for the noise than for the spectra

transform_tg = normalisation_specific_transform(
    targets.clone()
)  # We normalise every parameter btw 0 and 1

transform_au = normalisation_specific_transform(
    aux.clone()
)  # We normalise every auxiliary variable btw 0 and 1


# temptransform_ft=Mean_each()
# temp_ft=temptransform_ft(features.clone()) # We set the mean of all spectra to zero
# transform_ft=ComposedTransforms([temptransform_ft,IndexTransform(0,normalisation_uniform_transform(temp_ft[:,0])),IndexTransform([k for k in range(1,features.shape[1])],normalisation_uniform_transform(temp_ft[:,1:]))]) # We normalise the spectra between 0 and 1

# transform_no=ComposedTransforms([Mean_each_uncertainty(),IndexTransform(0,ScalarTransform(0,transform_ft.transforms_list[1].transform.scale)),IndexTransform([k for k in range(1,features.shape[1])],ScalarTransform(0,transform_ft.transforms_list[2].transform.scale))])

# transform_ft_scale=ComposedTransforms([transform_ft,UseIndex([0])]) # The decoder_scale will only use the mean (index 0)
# transform_ft_shape=ComposedTransforms([transform_ft,Truncate(1)])
# transform_no_scale=ComposedTransforms([transform_no,UseIndex([0])]) # The decoder_scale will only use the mean (index 0)
# transform_no_shape=ComposedTransforms([transform_no,Truncate(1)])

transform_no = Normalise_each_uncertainty(features, ft="mean_std")

temptransform_ft = ComposedTransforms(
    [Normalise_each(ft="mean_std"), IndexTransform([0, 1], LogTransform())]
)  # We normalise each spectrum btw 0 and 1 and store the min and max as new features in indexes 0 and 1.
tempft = temptransform_ft(
    features.clone()
)  # We apply temporarly the transformation to later normalise btw 0 and 1 the min and max (index=0 and 1)
transform_ft = ComposedTransforms(
    [
        temptransform_ft,
        IndexTransform([0, 1], normalisation_specific_transform(tempft[:, 0:2])),
    ]
)  # We normalise the min and max btw 0 and 1

transform_ft_scale = ComposedTransforms(
    [transform_ft, UseIndex([0, 1])]
)  # The decoder_scale will only use the min and max (indexes 0 and 1)
transform_ft_shape = ComposedTransforms([transform_ft, Truncate(2)])

transform_no_scale = ComposedTransforms(
    [
        transform_no,
        UseIndex([0, 1]),
        LogTransform_uncertainty(Normalise_each(ft="mean_std")(features)[:, 0:2]),
        ScalarTransform(0, normalisation_specific_transform(tempft[:, 0:2]).scale),
    ]
)  # The decoder_scale will only use the mean (index 0)
transform_no_shape = ComposedTransforms([transform_no, Truncate(2)])


# Initialise decoder shape
print(" Loading Decoder_shape...")
d_ini_epoch, d_loss, d_loaders_shape, decoder_shape = initialization(
    config["net"]["decoder_shape-name"],
    config=config,
    transform_ft=transform_ft_shape,
    transform_tg=transform_tg,
    transform_au=transform_au,
    transform_no=transform_no_shape,
)[0:4]


# Initialise decoder_scale
print("\n Loading Decoder_scale...")
d_scale_ini_epoch, d_scale_loss, d_scale_loaders, decoder_scale = initialization(
    config["net"]["decoder_scale-name"],
    config=config,
    transform_ft=transform_ft_scale,
    transform_tg=transform_tg,
    transform_au=transform_au,
    transform_no=transform_no_scale,
)[0:4]
