"""
Using SBI package, we try to infer the posterior distribution of the parameters of the atmosphere given the observed spectrum, or
the likelihood of the spectrum given the parameters. We use Neural Spline Normalising Flows.
"""

from sbi.inference import SNPE_C, SNLE
from sbi.utils import BoxUniform, posterior_nn, likelihood_nn
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
from sbi import analysis as analysis
import torch
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import signal
from msapnet.utils.transform import *
import scipy as sc
from time import time


def list_of_posteriors(
    hyperparameters_list=2
    * [
        {
            "hidden_features": 16,
            "num_transforms": 3,
            "num_bins": 34,
            "tail_bound": 4.828,
            "hidden_layers_spline_context": 5,
            "num_blocks": 3,
            "dropout_probability": 0.177,
            "use_batch_norm": True,
        }
    ],
    z_score_theta="independent",
    z_score_x="independent",
):
    """Returns a list of neural normalising flows.

    Args:
        n_ensemble (int, optional): number of models to include (length of the list). Defaults to 2.
        hidden_features (int, optional): hidden features of the network. Defaults to 50.
        num_transforms (int, optional): number of steps in the flow. Defaults to 3.
        embedding_net (torch.nn.Module, optional): embedding network for the features. Defaults to None.
        z_score_theta (str, optional): do we retransform the targets ?. Defaults to 'none'.
        z_score_x (str, optional): do we retransform the features ?. Defaults to 'none'.

    Returns:
        list: list of models
    """

    return [
        posterior_nn(
            model="nsf", z_score_theta=z_score_theta, z_score_x=z_score_x, **kwargs
        )
        for kwargs in hyperparameters_list
    ]


def count_overconfidence(posterior, test_x_data, test_y_data, n_samples, n_tests):
    L = []

    for j in tqdm(
        np.random.choice(test_x_data.shape[0], n_tests), desc="Making statistics"
    ):
        L.append(
            posterior.sample((1, n_samples), x=test_x_data[j], show_progress_bars=False)
        )

    samples = torch.cat(L, dim=0)
    means = samples.mean(dim=1)
    stds = samples.std(dim=1)
    distance = torch.abs(means - test_y_data)
    D = []
    for c in [stds, 2 * stds, 3 * stds]:
        D.append((1.0 * (distance < c)).mean(dim=0, keepdim=True))
    return torch.cat(D, dim=0)


def save_predictions(full_target_samples, config, prefix="direct_posterior"):
    submit_file = f"{config['path']['local-path']}{config['path']['outputs-path']}Predictions_{prefix}_{config['net']['version']}.{config['net']['iteration-save']}.hdf5"

    RT_submission = h5py.File(submit_file, "w")
    full_target_samples = (
        full_target_samples.detach().cpu().numpy().astype(dtype="float64")
    )
    for n, target_sample in enumerate(
        tqdm(full_target_samples, desc="Saving the predictions")
    ):
        grp = RT_submission.create_group(f"Planet_public{n+1}")
        pl_id = grp.attrs["ID"] = np.int64(n)
        tracedata = grp.create_dataset("tracedata", data=target_sample)
        weight_adjusted = (
            np.ones(target_sample.shape[0], dtype="float64")
            * 1
            / target_sample.shape[0]
        )  # constant weight

        weights = grp.create_dataset("weights", data=weight_adjusted)
    RT_submission.close()


def handler(signum, frame):
    raise TimeoutError("Timeout for the allowed time")


def predict_sample_challenge(
    posterior,
    prior,
    transform_ft,
    transform_tg,
    transform_au,
    transform_no,
    config,
    device=torch.device("cpu"),
    R_star_transform=True,
    module_nb=3,
):
    signal.signal(signal.SIGALRM, handler)

    # Create a local version of transform_tg: ## TO CHANGE IF DIFFERENT TRANSFORM TG
    if R_star_transform:
        challenge_transform_tg = IndexTransform(
            [0], ScalarTransform(0.0, 1.0)
        )  # create a cpoy of transform_tg for allowing to modify the scale according to Rstar

        if 1 <= module_nb < 3:  # depending on the "module version" used
            challenge_transform_tg = ComposedTransforms(
                [
                    challenge_transform_tg,
                    IndexTransform([0, 1], LogTransform()),
                    transform_tg.transforms_list[1],
                ]
            )
        else:
            challenge_transform_tg = ComposedTransforms([challenge_transform_tg])
    else:
        challenge_transform_tg = transform_tg

    _challenge_features = torch.load(
        f"{config['path']['local-path']+config['path']['data-path']}test/features_1.pt"
    ).to(device)
    _challenge_aux = torch.load(
        f"{config['path']['local-path']+config['path']['data-path']}test/aux_1.pt"
    ).to(device)
    _challenge_no = torch.load(
        f"{config['path']['local-path']+config['path']['data-path']}test/noises_1.pt"
    ).to(device)

    challenge_features = transform_ft(_challenge_features)
    challenge_aux = transform_au(_challenge_aux)
    challenge_no = transform_no(_challenge_no)
    challenge_ft = torch.cat([challenge_aux, challenge_features, challenge_no], dim=1)

    challenge_predicted_sample = torch.zeros((challenge_features.shape[0], 2500, 7))
    for k in tqdm(range(challenge_ft.shape[0]), desc="Sampling the challenge"):
        signal.alarm(5)
        try:
            if R_star_transform:
                challenge_transform_tg.transforms_list[
                    0
                ].transform.scale = _challenge_aux[
                    k, 2
                ]  # we have to adjust the transform so that it uses the correct R_star
                challenge_transform_tg.transforms_list[
                    0
                ].transform.min = torch.zeros_like(_challenge_aux[k, 2])
            challenge_predicted_sample[k] = challenge_transform_tg.inverse(
                posterior.sample((2500,), x=challenge_ft[k], show_progress_bars=False)
            )
        except TimeoutError:
            print(f"\t Timeout for {k}")
            challenge_predicted_sample[k] = challenge_transform_tg.inverse(
                prior.sample((2500,))
            )
    signal.alarm(0)

    return challenge_predicted_sample


def get_ranks(posterior, prior, features, targets, max_tests=4000):
    ranks = []
    p = np.random.choice(len(features), size=min(max_tests, features.shape[0]))
    signal.signal(signal.SIGALRM, handler)
    for k, ft in enumerate(tqdm(features[p], desc="Sampling the ranks")):
        signal.alarm(5)
        try:
            sample = posterior.sample((4500,), x=ft, show_progress_bars=False)
        except TimeoutError:
            print(f"\t Timeout for {k}")
            sample = prior.sample((4500,))
        signal.alarm(0)
        ranks.append(
            (sample < targets[p][k].unsqueeze(dim=0)).sum(dim=0).detach().cpu().numpy()
        )

    return np.array(ranks)


def KS_test(
    posterior,
    prior,
    transform_ft,
    transform_tg,
    transform_au,
    transform_no,
    config,
    test_size=4000,
    R_star_transform=True,
    device=torch.device("cpu"),
    sampling_nb=2000,
    verbose=True,
):
    path = f"{config['path']['local-path']}FullDataset/TrainingData/"
    infile = h5py.File(
        f"{path}Ground Truth Package/Tracedata.hdf5"
    )  ### loading in tracedata.hdf5
    SpectralData = h5py.File(f"{path}SpectralData.hdf5")  ### load file

    full_df = pd.read_pickle(f"{path}full_df.pt")
    planetlist = [p for p in SpectralData.keys()]
    ids = [
        int(p[12:]) for p in planetlist
    ]  # used to reorder correctly planetlist so that its the same order as the torch tensors
    planetlist = np.array(planetlist)[-test_size:]  # keeps only the test_set

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
    columns = ["ID"]
    id_noise = 1
    for wl in SpectralData[planetlist[0]]["instrument_wlgrid"]:
        columns.append(f"spectrum_{wl}")
        id_noise += 1
    for wl in SpectralData[planetlist[0]]["instrument_wlgrid"]:
        columns.append(f"noise_{wl}")

    nb_eff_test = 0
    KS_tests = []
    pv = 0
    signal.signal(signal.SIGALRM, handler)
    if verbose:
        range_i = tqdm(range(len(planetlist)), desc="Performing KS-test: ")
    else:
        range_i = range(len(planetlist))

    t0 = t1 = t2 = t3 = 0

    for i in range_i:
        if infile[planetlist[i]]["tracedata"].shape:
            nb_eff_test += 1
            p = planetlist[i]
            ID = float(p[12:])
            t0 = time()
            planet_df = full_df[full_df["ID"] == ID]
            pspectrum = transform_ft(
                torch.tensor(planet_df[columns[1:id_noise]].values).float().to(device)
            )
            pnoises = transform_no(
                torch.tensor(planet_df[columns[id_noise:]].values).float().to(device)
            )
            paux = transform_au(
                torch.tensor(planet_df[auxparams].values).float().to(device)
            )
            pfeature = torch.cat([paux, pspectrum, pnoises], dim=1)

            if R_star_transform:
                _aux = torch.tensor(planet_df[auxparams].values).float().to(device)
                transform_tg.transforms_list[0].transform.scale = _aux[
                    0, 2
                ]  # we have to adjust the transform so that it uses the correct R_star
                transform_tg.transforms_list[0].transform.min = 0.0

            trace = infile[p]["tracedata"][:]  ### accessing Nested Sampling trace data
            weights = infile[p]["weights"][:]

            t1 += time() - t0
            t0 = time()

            ground_truth_sample = (
                torch.Tensor(
                    trace[np.random.choice(len(weights), size=sampling_nb, p=weights)]
                )
                .cpu()
                .numpy()
            )
            signal.alarm(5)
            try:
                predicted_sample = (
                    transform_tg.inverse(
                        posterior.sample(
                            (sampling_nb,), x=pfeature.cpu(), show_progress_bars=False
                        ).to(device)
                    )
                    .cpu()
                    .numpy()
                )
            except TimeoutError:
                print(f"\t Timeout for {i}")
                predicted_sample = (
                    transform_tg.inverse(prior.sample((sampling_nb,)).to(device))
                    .cpu()
                    .numpy()
                )
            signal.alarm(0)

            t2 += time() - t0
            t0 = time()

            K = np.zeros(predicted_sample.shape[1])
            for k in range(predicted_sample.shape[1]):
                stat = sc.stats.ks_2samp(
                    ground_truth_sample[:, k], predicted_sample[:, k]
                )
                K[k] = stat.statistic
                pv += stat.pvalue

            t3 += time() - t0

            KS_tests.append(K)

    KS_tests = np.array(KS_tests)
    if verbose:
        _m = ""
        _s = ""
        for k in range(predicted_sample.shape[1]):
            _m += f" {KS_tests.mean(axis=0)[k]:.4f}"
            _s += f" {KS_tests.std(axis=0)[k]:.3f}"

        print(f"KS test finished. Nb of tests effectivly realised: {nb_eff_test}")
        print(f"\tMean KS statistic: {KS_tests.mean():.4f}")
        print(f"\tStd of the KS statistics: {KS_tests.std():.3f}")
        print(f"\tMean pvalue: {pv/(predicted_sample.shape[1]*nb_eff_test):.3e}")
        print(f"\tDetailed KS statistics:" + _m)
        print(f"\tDetailed std of KS statistics:" + _s)
        print(f"\tComputing times: {t1:.2f} {t2:.2f} {t3:.2f}")

    return KS_tests.mean()


def get_samples_for_param_comparison(
    posterior,
    prior,
    features,
    targets,
    transform_tg,
    R_star=None,
    module_nb=1,
    sampling_nb=2000,
    max_tests=4000,
):
    p = np.random.choice(len(features), size=max(max_tests, features.shape[0]))
    signal.signal(signal.SIGALRM, handler)
    samples = []
    tgs = []
    for i, ft in enumerate(tqdm(features[p], desc="Sampling for param comparison")):
        signal.alarm(5)
        try:
            sample = posterior.sample((sampling_nb,), x=ft, show_progress_bars=False)
        except TimeoutError:
            print(f"\t Timeout")
            sample = prior.sample((sampling_nb,))
        signal.alarm(0)

        if R_star is not None:
            local_transform_tg = IndexTransform(
                [0], ScalarTransform(0.0, R_star[p][i])
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

        samples.append(local_transform_tg.inverse(sample).unsqueeze(dim=0))
        tgs.append(local_transform_tg.inverse(targets[p][i].unsqueeze(dim=0)))
    return torch.cat(samples, dim=0), torch.cat(tgs, dim=0)


def get_samples_from_several_posteriors(
    list_of_posteriors,
    prior,
    feature,
    transform_tg,
    R_star=None,
    module_nb=1,
    sampling_nb=5000,
):
    signal.signal(signal.SIGALRM, handler)
    samples = []
    for i, p in enumerate(
        tqdm(list_of_posteriors, desc="Sampling from several posteriors:")
    ):
        signal.alarm(5)
        try:
            sample = p.sample((sampling_nb,), x=feature, show_progress_bars=False)
        except TimeoutError:
            print(f"\t Timeout for posterior {i}")
            sample = prior.sample((sampling_nb,))
        signal.alarm(0)

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
