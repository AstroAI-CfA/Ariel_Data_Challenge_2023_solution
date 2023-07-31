"""
This script is used as a base for all scripts using the ZUKO package (and DirectCDF approach) when noising is required.
It imports the necessary packages and functions, it loads the config file and the data.
Sample a random noise from the noise data and add it to the spectra.
It defines the transforms used for feature engineering, and splits the data in training and test set.
"""


# Import packages
import pathlib

MSAP_path = pathlib.Path(__file__).parent.resolve() # finds the local path

import matplotlib.pyplot as plt
import numpy as np
import torch
torch.cuda.empty_cache()
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_str = "cuda" if torch.cuda.is_available() else "cpu"
# device=torch.device('cpu')
# device_str='cpu'
print(f"Device used : {device}")
from pathlib import Path
from msapnet.utils.utils import open_config # used to load the config
from msapnet.utils.transform import * # contains all the classes of Transforms used. They are not torch.Transforms instances.
from msapnet.inference.inference_zuko import * # contains the functions specifically used by zuko normalising flows.
from msapnet.utils.plots import * # plots for everything
from msapnet.utils.training import * # all the training routines for all machine learning approaches

import zuko

config_path = f"{MSAP_path}/zuko_ariel_config.yaml"

# Open configuration file
_, config = open_config(0, config_path)


# Load the data
features = torch.load(
    f"{config['path']['local-path']+config['path']['data-path']}features_{config['data']['run-id']}.pt"
).to(device) # the features file contains the tensor of spectra. It is of shape (n_examples, n_spectral_features=52)
targets = (
    torch.load(
        f"{config['path']['local-path']+config['path']['data-path']}targets_{config['data']['run-id']}.pt"
    ).to(device)
    + 1e-9
) # the target file contains a tensor of the target parameters. It is of shape (n_examples, n_samples, n_target_parameters). Since we have lots of zeros due to padding, we add 1e-9 to avoid any division by zero later.
try:
    weights = torch.load(
        f"{config['path']['local-path']+config['path']['data-path']}weights_{config['data']['run-id']}.pt"
    ).to(device) # the weights associated to the targets. It is of shape (n_examples,n_samples)
except:
    print("No weights loaded")
    weights=torch.ones_like(features[...,[0]])
noises = torch.load(
    f"{config['path']['local-path']+config['path']['data-path']}noises_{config['data']['run-id']}.pt"
).to(device) # the uncertainty of the spectrum
metadatas = torch.load(
    f"{config['path']['local-path']+config['path']['data-path']}metadatas_{config['data']['run-id']}.pt"
) # containing some information about the dataset
aux = torch.load(
    f"{config['path']['local-path']+config['path']['data-path']}aux_{config['data']['run-id']}.pt"
).to(device)
freq = torch.load(
    f"{config['path']['local-path']+config['path']['data-path']}freq_1.pt"
) # the frequency array for plotting


# Sample the noise and add it
features=torch.normal(mean=features,
                      std=noises)


# Define transforms
R_p_estimated_spectrum = (
    aux[:, [2]] / 69911000 * torch.sqrt(torch.mean(features, dim=1, keepdim=True))
) # we use R*sqrt(mean of the spectrum) as an estimator of the radius coming from the definition of the spectrum units. 69911000 is the radius of Jupiter used as unit for Rp. 
R_p_estimated_gravity = torch.sqrt((aux[:, [4]] / aux[:, [7]] * 6.674e-11)) / 69911000 # we use sqrt(Mp/(gG)) as an estimator of the radius coming from Gauss's law for Gravity. It has a weird vertical line probably due to errors in the dataset generation.
in_vertical_line = (
    1.0
    * (0.71488e8 / 69911000 < R_p_estimated_gravity)
    * (R_p_estimated_gravity < 0.714881e8 / 69911000)
) # we identify the points located in this weird vertical line as we knkow they won't be useful for estimating the radius
R_p_estimated_combo = (
    R_p_estimated_gravity * (1 - in_vertical_line)
    + R_p_estimated_spectrum * in_vertical_line
) # this estimator is based on R_p_estimated_gravity except for the vertical line where we use R_p_estimated_spectrum
aux = torch.cat(
    [
        aux,
        R_p_estimated_spectrum,
        R_p_estimated_gravity,
        in_vertical_line,
        R_p_estimated_combo,
    ],
    dim=1,
)  # We add all of these new features to the auxiliary variables to be used by our models

# transform_tg=IndexTransform([0],ScalarTransform(0.,1.))
transform_tg = IndexTransform(
    [0], ScalarTransform(aux[:, -1], 1.0)
)  # we substract the estimated radius of the planet to the target radius to only predict the offset
transform_tg = ComposedTransforms(
    [transform_tg, normalisation_specific_transform(transform_tg(targets))]
) # we normalise every parameter between 0 and 1

transform_au = normalisation_specific_transform(
    aux
)  # We normalise every auxiliary variable btw 0 and 1

temptransform_ft = ComposedTransforms([Normalise_each(ft="mean_std")]) # we normalise the spectrum by substracting its mean and dividing by its std. We concatenate the mean and std to the features tensor to not lose the information
tempft = temptransform_ft(
    features.clone()
)  # We apply temporarly the transformation to later normalise btw 0 and 1 the min and max (index=0 and 1)
transform_ft = ComposedTransforms(
    [
        temptransform_ft,
        IndexTransform([0, 1], normalisation_specific_transform(tempft[:, 0:2])),
    ]
)  # We normalise the mean and std features between 0 and 1. The spectrum is already normalized.

del tempft
del temptransform_ft
gc.collect()

transform_no = normalisation_specific_transform(
    noises
)  # We normalise the uncertainty btw 0 and 1

# We transform the loaded data
_features = transform_ft(features)
_targets = transform_tg(targets)
_aux = transform_au(aux)
_noises = transform_no(noises)
_weights = weights.clone()

del features, targets, aux, noises, weights
gc.collect()

# we define the input tensor for the model as the concatenation of auxiliary variables, spectrum features and spectrum uncertainty
ft = torch.cat([_aux, _features, _noises], dim=1)


# Split the dataset
# n_test = 4000
n_test = config["data"]["n-test"]

test_features = ft[-n_test:, :]
test_spectra = _features[-n_test:, :]
test_targets = _targets[-n_test:, :]
test_weights = _weights[-n_test:, :]
test_noises = _noises[-n_test:, :]
test_aux = _aux[-n_test:, :]

train_features = ft[:-n_test, :]
train_spectra = _features[:-n_test, :]
train_targets = _targets[:-n_test, :]
train_weights = _weights[:-n_test, :]
train_noises = _noises[:-n_test, :]
train_aux = _aux[:-n_test, :]

del _features, _targets, _weights, _noises, _aux
gc.collect()


# Load the ground truth data:
# The ground truth data is the "input parameters", that is the dataset containing the target parameters used to generate the spectra with the radiative transfer model
# It is used to make some tests and comparisons for our models
GTfeatures = torch.load(
    f"{config['path']['local-path']+config['path']['data-path']}features_1.pt"
).to(device)
GTtargets = torch.load(
    f"{config['path']['local-path']+config['path']['data-path']}targets_1.pt"
).to(device)
GTnoises = torch.load(
    f"{config['path']['local-path']+config['path']['data-path']}noises_1.pt"
).to(device)
GTmetadatas = torch.load(
    f"{config['path']['local-path']+config['path']['data-path']}metadatas_1.pt"
)
GTaux = torch.load(
    f"{config['path']['local-path']+config['path']['data-path']}aux_1.pt"
).to(device)

# Sample the noise and add it
GTfeatures=torch.normal(mean=GTfeatures,
                      std=GTnoises)

GTR_p_estimated_spectrum = (
    GTaux[:, [2]] / 69911000 * torch.sqrt(torch.mean(GTfeatures, dim=1, keepdim=True))
)
GTR_p_estimated_gravity = (
    torch.sqrt((GTaux[:, [4]] / GTaux[:, [7]] * 6.674e-11)) / 69911000
)
GTin_vertical_line = (
    1.0
    * (0.71488e8 / 69911000 < GTR_p_estimated_gravity)
    * (GTR_p_estimated_gravity < 0.714881e8 / 69911000)
)
GTR_p_estimated_combo = (
    GTR_p_estimated_gravity * (1 - GTin_vertical_line)
    + GTR_p_estimated_spectrum * GTin_vertical_line
)
GTaux = torch.cat(
    [
        GTaux,
        GTR_p_estimated_spectrum,
        GTR_p_estimated_gravity,
        GTin_vertical_line,
        GTR_p_estimated_combo,
    ],
    dim=1,
)  # This is a closed form for the Radius of the planet: Rp=Rs*sqrt(mu), 69911000 is the size of Jupiter because Rp is expresseed in Jupiter units


# Define ground truth transforms
GTtransform_tg = IndexTransform(
    [0], ScalarTransform(GTaux[:, -1], 1.0)
)  # we retrieve the estimated radius of the planet
GTtransform_tg = ComposedTransforms(
    [GTtransform_tg, transform_tg.transforms_list[1]]
)  # We normalise every parameter btw 0 and 1
# GTtransform_tg=transform_tg
GTtransform_au = transform_au
GTtransform_ft = transform_ft
GTtransform_no = transform_no


_GTfeatures = GTtransform_ft(GTfeatures)
_GTtargets = GTtransform_tg(GTtargets)
_GTaux = GTtransform_au(GTaux)
_GTnoises = GTtransform_no(GTnoises)

GTft = torch.cat([_GTaux, _GTfeatures, _GTnoises], dim=1)

GTtest_features = GTft[-4000:, :]
GTtest_spectra = _GTfeatures[-4000:, :]
GTtest_targets = _GTtargets[-4000:, :]
GTtest_noises = _GTnoises[-4000:, :]
GTtest_aux = _GTaux[-4000:, :]

GTtrain_features = GTft[:-4000, :]
GTtrain_spectra = _GTfeatures[:-4000, :]
GTtrain_targets = _GTtargets[:-4000, :]
GTtrain_noises = _GTnoises[:-4000, :]
GTtrain_aux = _GTaux[:-4000, :]
