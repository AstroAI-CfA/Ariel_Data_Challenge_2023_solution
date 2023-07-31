"""
Hyperparameter optimisation using Optuna of the Normalising Flows.
The target data is without weights, so not NS samples, but rather input parameters directly.
The spectra are noised (not the original ideal spectra).
The model is a joined Normalising Flow (no independent Normalising Flows).
"""

import sys
import signal


sys.path.insert(0, "insert_your_path")
from init_zuko_ariel_add_noise import *
import optuna
import numpy as np
from optuna.visualization import *


timeout = float(sys.argv[1]) * 3600

del test_features, test_spectra, test_targets, test_weights, test_noises, test_aux
del train_noises, train_aux, train_spectra


training_parameters = {
    "learning_rate": 1e-3,
    "training_batch_size": 100,
    "max_num_epochs": 300,
    "batch_fraction": 1,
}


print(
    f"\nStarting Hyperparameter Optimisation of version {config['net']['version']}.{config['net']['iteration-save']}"
)
print(
    f"Complete Zuko Flow (7D) trained on true inputs (without weights) with noised spectra (not the ideal ones)"
)
print(f"Training parameters: {training_parameters}")
print(f"WARNING: The objective is the logprob, so it should be maximised not minimised ! The objective isn't the loss, but -loss !")


def objective(trial):
    hidden_features = trial.suggest_int("hidden_features", 1, 500)
    num_transforms = trial.suggest_int("num_transforms", 1, 25)
    num_bins = trial.suggest_int("num_bins", 3, 40)
    # tail_bound=trial.suggest_float("tail_bound",1.,6.)
    hidden_layers_spline_context = trial.suggest_int(
        "hidden_layers_spline_context", 1, 10
    )
    # dropout_probability=trial.suggest_float("dropout_probability",0.,0.2)
    # use_batch_norm=True

    hyperparameters = {
        "hidden_features": [hidden_features] * hidden_layers_spline_context,
        "transforms": num_transforms,
        "bins": num_bins,
    }

    try:
        flow = zuko.flows.NSF(
            features=train_targets.shape[-1], context=train_features.shape[-1], **hyperparameters
        ).to(device)
        t_init = time()
        LOSSES = train_zuko_without_weights(
            flow,
            train_features,
            train_targets,
            show_progress_bar=False,
            verbose=False,
            **training_parameters,
        )
        t_total = time() - t_init
        
        torch.save(
        [flow, hyperparameters, -np.min(LOSSES[1]), t_total],
        f"{config['path']['local-path']}{config['path']['states-path']}Flow_from_study__{trial.number}_{config['net']['version']}.pt",
        )
        print(f"We saved trial {trial.number}")
        
    except torch.cuda.OutOfMemoryError as error:
        print("Out of Memory")
        print(error)
        return -2

    print(
        f"\tEpochs trained: {len(LOSSES[1])} Mean epoch duration: {t_total/len(LOSSES[1]):.2f}"
    )
    return -np.min(LOSSES[1])
    posterior = model.build_posterior(density_estimator)


study = optuna.create_study(direction="maximize")
study.optimize(objective, timeout=timeout, n_jobs=1)

# print("\n")
# print(study.best_params)
# print(study.best_value)

torch.save(
    study,
    f"{config['path']['local-path']}{config['path']['states-path']}Study_ZKIndependentFlow_{config['net']['version']}_{parameter_id}_{timeout}.pt",
)
# study=torch.load(f"{config['path']['local-path']}{config['path']['states-path']}Study_{config['net']['version']}.{cf_it}_{module_nb}_{reduced_size}_{timeout}.pt")

fig1 = plot_contour(study)
fig1.update_layout(height=2000, width=2000)
fig1.write_image(
    file=f"{config['path']['local-path']}{config['path']['plots-path']}Optimisation/Contour_ZKIndependentFlow_{config['net']['version']}_{parameter_id}_{timeout}.png",
    format="png",
)

fig1 = plot_param_importances(study, target=lambda t: t.values[0])
fig1.update_layout(width=800, height=600)
fig1.write_image(
    f"{config['path']['local-path']}{config['path']['plots-path']}Optimisation/ParamImportance_value_ZKIndependentFlow_{config['net']['version']}_{parameter_id}_{timeout}.png",
    format="png",
)
