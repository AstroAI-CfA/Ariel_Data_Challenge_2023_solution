"""
Using Zuko package, we try to infer the posterior distribution of the parameters of the atmosphere given the observed spectrum, or
the likelihood of the spectrum given the parameters. We use Neural Spline Normalising Flows.
"""

import zuko
import torch
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# import signal
from msapnet.utils.transform import *
import scipy as sc
from time import time


def save_predictions(full_target_samples, config, prefix="direct_posterior", output_prefix="Planet_public"):
    """
    Save a tensor of samples to a file compatible with the format requirements of the challenge.
    """
    submit_file = f"{config['path']['local-path']}{config['path']['outputs-path']}Predictions_{prefix}_{config['net']['version']}.{config['net']['iteration-save']}.hdf5"

    RT_submission = h5py.File(submit_file, "w")
    full_target_samples = (
        full_target_samples.detach().cpu().numpy().astype(dtype="float64")
    )
    N = full_target_samples.shape[0]
    planet_names = np.array([f"{output_prefix}{n+1}" for n in range(N)])
    planet_names.sort()
    if "public" in output_prefix:
        ID = [np.int64(p[13:]) for p in planet_names]
    elif "test" in output_prefix:
        ID = [np.int64(p[11:]) for p in planet_names]
    for n, target_sample in enumerate(
        tqdm(full_target_samples, desc="Saving the predictions")
    ):
        grp = RT_submission.create_group(planet_names[n])
        pl_id = grp.attrs["ID"] = np.int64(ID[n])
        tracedata = grp.create_dataset("tracedata", data=target_sample)
        weight_adjusted = (
            np.ones(target_sample.shape[0], dtype="float64")
            * 1
            / target_sample.shape[0]
        )  # constant weight

        weights = grp.create_dataset("weights", data=weight_adjusted)
    RT_submission.close()


def predict_sample_challenge(
    flow,
    transform_ft,
    transform_tg,
    transform_au,
    transform_no,
    config,
    device=torch.device("cpu"),
    R_p_estimated=True,
    module_nb=3,
    sampling_size=2500,
    run_id=1
):
    """
    Predicts samples for the challenge. flow can be a flow or a list of flows (for independent normalising flows)
    """
    if isinstance(flow, list):
        list_of_flows = flow
    else:
        list_of_flows = None

    # Create a local version of transform_tg: ## TO CHANGE IF DIFFERENT TRANSFORM TG
    if R_p_estimated:
        challenge_transform_tg = IndexTransform(
            [0], ScalarTransform(0.0, 1.0)
        )  # create a cpoy of transform_tg for allowing to modify the scale according to Rstar

        if 1 <= module_nb < 3:  # depending on the "module version" used
            challenge_transform_tg = ComposedTransforms(
                [challenge_transform_tg, transform_tg.transforms_list[1]]
            )
        else:
            challenge_transform_tg = ComposedTransforms([challenge_transform_tg])
    else:
        challenge_transform_tg = transform_tg

    _challenge_features = torch.load(
        f"{config['path']['local-path']+config['path']['data-path']}test/features_{run_id}.pt"
    ).to(device)
    _challenge_aux = torch.load(
        f"{config['path']['local-path']+config['path']['data-path']}test/aux_{run_id}.pt"
    ).to(device)
    _challenge_no = torch.load(
        f"{config['path']['local-path']+config['path']['data-path']}test/noises_{run_id}.pt"
    ).to(device)

    _R_p_estimated_spectrum = (
        _challenge_aux[:, [2]]
        / 69911000
        * torch.sqrt(torch.mean(_challenge_features, dim=1, keepdim=True))
    )
    _R_p_estimated_gravity = (
        torch.sqrt((_challenge_aux[:, [4]] / _challenge_aux[:, [7]] * 6.674e-11))
        / 69911000
    )
    _in_vertical_line = (
        1.0
        * (0.71488e8 / 69911000 < _R_p_estimated_gravity)
        * (_R_p_estimated_gravity < 0.714881e8 / 69911000)
    )
    _R_p_estimated_combo = (
        _R_p_estimated_gravity * (1 - _in_vertical_line)
        + _R_p_estimated_spectrum * _in_vertical_line
    )
    _challenge_aux = torch.cat(
        [
            _challenge_aux,
            _R_p_estimated_spectrum,
            _R_p_estimated_gravity,
            _in_vertical_line,
            _R_p_estimated_combo,
        ],
        dim=1,
    )  # This is a closed form for the Radius of the planet: Rp=Rs*sqrt(mu), 69911000 is the size of Jupiter because Rp is expresseed in Jupiter units

    challenge_features = transform_ft(_challenge_features)
    challenge_aux = transform_au(_challenge_aux)
    challenge_no = transform_no(_challenge_no)
    challenge_ft = torch.cat([challenge_aux, challenge_features, challenge_no], dim=1)

    challenge_predicted_sample = torch.zeros(
        (challenge_features.shape[0], sampling_size, 7)
    ).to(device)
    for k in tqdm(range(challenge_ft.shape[0]), desc="Sampling the challenge"):
        if R_p_estimated:
            challenge_transform_tg.transforms_list[0].transform.min = _challenge_aux[
                k, -1
            ]

        if list_of_flows is None:
            challenge_predicted_sample[k] = challenge_transform_tg.inverse(
                flow(challenge_ft[k]).sample((sampling_size,))
            )
        else:
            for j in range(7):
                challenge_predicted_sample[k, :, j] = list_of_flows[j](
                    challenge_ft[k]
                ).sample((sampling_size,))[:, 0]
            challenge_predicted_sample[k] = challenge_transform_tg.inverse(
                challenge_predicted_sample[k]
            )

    return challenge_predicted_sample.cpu()


def get_ranks(flow, features, targets, max_tests=4000):
    ranks = []
    p = np.random.choice(len(features), size=min(max_tests, features.shape[0]))

    for k, ft in enumerate(tqdm(features[p], desc="Sampling the ranks")):
        sample = flow(features[p]).sample((4500,))
        ranks.append(
            (sample < targets[p][k].unsqueeze(dim=0)).sum(dim=0).detach().cpu().numpy()
        )

    return np.array(ranks)


def KS_test(
    flow,
    features,
    targets,
    weights,
    aux,
    transform_tg,
    transform_au,
    config,
    R_p_estimated=True,
    device=torch.device("cpu"),
    sampling_nb=2000,
    verbose=True,
):
    """
    Computes the Kolgomorov-Smirnov test between the target samples and samples drawn using the provided flow (or list of flows)
    """
    aux = transform_au.inverse(aux)
    if isinstance(flow, list):
        list_of_flows = flow
    else:
        list_of_flows = None

    KS_tests = []
    pv = 0
    if verbose:
        range_i = tqdm(range(features.shape[0]), desc="Performing KS-test: ")
    else:
        range_i = range(features.shape[0])

    for i in range_i:
        if R_p_estimated:
            transform_tg.transforms_list[0].transform.min = aux[
                i, -1
            ]  # we retrieve the estimated radius

        ground_truth_sample = (
            transform_tg.inverse(
                torch.Tensor(
                    targets[i][
                        np.random.choice(
                            weights[i].shape[0], size=sampling_nb, p=weights[i].cpu()
                        )
                    ]
                )
            )
            .cpu()
            .numpy()
        )

        if list_of_flows is None:
            predicted_sample = (
                transform_tg.inverse(
                    flow(features[i]).sample((sampling_nb,)).to(device)
                )
                .cpu()
                .numpy()
            )
        else:
            predicted_sample = torch.zeros((sampling_nb, targets.shape[-1])).to(device)
            for k in range(targets.shape[-1]):
                predicted_sample[:, k] = list_of_flows[k](features[i]).sample(
                    (sampling_nb,)
                )[:, 0]
            predicted_sample = transform_tg.inverse(predicted_sample).cpu().numpy()

        K = np.zeros(predicted_sample.shape[1])
        for k in range(predicted_sample.shape[1]):
            stat = sc.stats.ks_2samp(ground_truth_sample[:, k], predicted_sample[:, k])
            K[k] = stat.statistic
            pv += stat.pvalue

        KS_tests.append(K)

    KS_tests = np.array(KS_tests)
    if verbose:
        _m = ""
        _s = ""
        for k in range(predicted_sample.shape[1]):
            _m += f" {KS_tests.mean(axis=0)[k]:.4f}"
            _s += f" {KS_tests.std(axis=0)[k]:.3f}"

        print(f"\tMean KS statistic: {KS_tests.mean():.4f}")
        print(f"\tStd of the KS statistics: {KS_tests.std():.3f}")
        print(f"\tMean pvalue: {pv/(predicted_sample.shape[1]*targets.shape[0]):.3e}")
        print(f"\tDetailed KS statistics:" + _m)
        print(f"\tDetailed std of KS statistics:" + _s)

    return KS_tests.mean()

def KS_test_from_samples(samples,targets,verbose=True):
    
    KS_tests = []
    pv = 0
    if verbose:
        range_i = tqdm(range(samples.shape[0]), desc="Performing KS-test: ")
    else:
        range_i = range(samples.shape[0])

    for i in range_i:
        K = np.zeros(targets.shape[-1])
        for k in range(targets.shape[-1]):
            stat = sc.stats.ks_2samp(targets[i,:, k], samples[i,:, k])
            K[k] = stat.statistic
            pv += stat.pvalue

        KS_tests.append(K)

    KS_tests = np.array(KS_tests)
    if verbose:
        _m = ""
        _s = ""
        for k in range(targets.shape[-1]):
            _m += f" {KS_tests.mean(axis=0)[k]:.4f}"
            _s += f" {KS_tests.std(axis=0)[k]:.3f}"

        print(f"\tMean KS statistic: {KS_tests.mean():.4f}")
        print(f"\tStd of the KS statistics: {KS_tests.std():.3f}")
        print(f"\tMean pvalue: {pv/(samples.shape[1]*targets.shape[0]):.3e}")
        print(f"\tDetailed KS statistics:" + _m)
        print(f"\tDetailed std of KS statistics:" + _s)

    return KS_tests.mean()


def get_samples_for_param_comparison(
    flow,
    features,
    targets,
    transform_tg,
    R_p_estimated=None,
    module_nb=1,
    sampling_nb=2000,
    max_tests=4000,
):
    p = np.random.choice(len(features), size=min(max_tests, features.shape[0]))
    samples = []
    tgs = []
    for i, ft in enumerate(tqdm(features[p], desc="Sampling for param comparison")):
        sample = flow(ft).sample((sampling_nb,))

        if R_p_estimated is not None:
            local_transform_tg = IndexTransform(
                [0], ScalarTransform(R_p_estimated[p][i], 1.0)
            )  # create a copy of transform_tg for allowing to modify the scale according to Rstar
            if module_nb < 3:
                local_transform_tg = ComposedTransforms(
                    [local_transform_tg, transform_tg.transforms_list[1]]
                )
            else:
                local_transform_tg = ComposedTransforms([local_transform_tg])
        else:
            local_transform_tg = transform_tg

        samples.append(local_transform_tg.inverse(sample).unsqueeze(dim=0))
        tgs.append(local_transform_tg.inverse(targets[p][i].unsqueeze(dim=0)))
    return torch.cat(samples, dim=0), torch.cat(tgs, dim=0)


def get_samples_from_several_posteriors(
    list_of_flows, feature, transform_tg, R_star=None, module_nb=1, sampling_nb=5000
):
    samples = []
    for i, p in enumerate(tqdm(list_of_flows, desc="Sampling from several flows:")):
        sample = p(feature).sample((sampling_nb,))

        if R_star is not None:
            local_transform_tg = IndexTransform(
                [0], ScalarTransform(0.0, R_star)
            )  # create a copy of transform_tg for allowing to modify the scale according to Rstar
            if module_nb < 3:
                local_transform_tg = ComposedTransforms(
                    [
                        local_transform_tg,
                        IndexTransform([0, 1], LogTransform()),
                        transform_tg.transforms_list[1],
                    ]
                )
            else:
                local_transform_tg = ComposedTransforms([local_transform_tg])
        else:
            local_transform_tg = transform_tg
        samples.append(local_transform_tg(sample))
    return samples


# class IndependentDifferentDistributions(torch.distributions.Distribution):
#     """
#     Implements a way to define the distribution of independent variables that don't follow the same family of distributions.
#     list_of_distributions is the list of distributions that we want to concatenate.
#     A sample from IndependentDifferentDistributions would be the vector X composed of X_i following list_of_distributions[i]
#     """
#     def __init__(self,list_of_distributions):
#         super().__init__(self,event_shape=torch.Size([len(list_of_distributions)]))
#         self.list_of_distributions = list_of_distributions

#     def log_prob(self,value):
#         return sum([dist.log_prob(value) for dist in self.list_of_distributions]) # since the variables are independent, the probability of X is just the product of the probabilities of X_i

#     def sample(self,sample_shape):
#         return torch.cat([dist.sample([s for s  in sample_shape]+[1]) for dist in self.list_of_distributions],dim=-1) # we construct the sample by concatening the variables
