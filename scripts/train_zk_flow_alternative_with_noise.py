"""
A script to train one single complete Normalising Flow based on the zuko package.
The targets are the input parameters (not the NS samples) and the features the noised spectra (not the original ideal spectra).
"""

import sys
import signal


sys.path.insert(0, "insert_your_path")

try:
    test_this_time = bool(sys.argv[1])
except:
    test_this_time=False

if test_this_time:
    print("Testing the model")
else:
    print("Training the model")

from init_zuko_ariel_add_noise import *


hyperparameters = {
    "hidden_features": 152,
    "transforms": 18,
    "bins": 19,
    "hidden_layers_spline_context": 5,
}
training_parameters = {
    "learning_rate": 1e-3,
    "training_batch_size": 100,
    "max_num_epochs": 300,
    "batch_fraction": 1,
}

print(
    f"\nStarting training an alternative model (Complete Normalising flow trained on input parameters) with the spectra noised of version {config['net']['version']}.{config['net']['iteration-save']}"
)
print(f"Hyperparameters of the model: {hyperparameters}")
print(f"Training parameters: {training_parameters}")



hyperparameters["hidden_features"] = [
    hyperparameters["hidden_features"]
] * hyperparameters["hidden_layers_spline_context"]
del hyperparameters["hidden_layers_spline_context"]
flow = zuko.flows.NSF(
    features=train_targets.shape[-1],
    context=train_features.shape[-1],
    **hyperparameters,
)
flow = flow.to(device)

if not test_this_time:
    LOSSES = train_zuko_without_weights(
        flow, train_features, train_targets, **training_parameters
    )

    plot_loss_evolution(np.array(LOSSES), yscale="linear")
    plt.savefig(
        f"{config['path']['local-path']}{config['path']['plots-path']}/Training/ZK_loss_{config['net']['version']}.{config['net']['iteration-save']}.png"
    )

    torch.save(
        flow,
        f"{config['path']['local-path']}{config['path']['states-path']}ZKFlow_{config['net']['version']}.{config['net']['iteration-save']}.pth",
    )
flow=torch.load(f"{config['path']['local-path']}{config['path']['states-path']}ZKFlow_{config['net']['version']}.{config['net']['iteration-save']}.pth", map_location=device)



# # Plot param comparison
# samples, tgs = get_samples_for_param_comparison(
#     flow, GTtest_features, GTtest_targets, GTtransform_tg, R_p_estimated=GTR_p_estimated_combo, module_nb=1, max_tests=4000
# )

# plot_param_comparison_from_samples(samples.cpu(), tgs.cpu(), config)
# plot_param_comparison_heatmap(samples.cpu(), tgs.cpu(), config)


# # Computes KS test
# plot_transform_tg=IndexTransform([0],ScalarTransform(0.,1.))
# plot_transform_tg=ComposedTransforms([plot_transform_tg,IndexTransform([0,1],LogTransform()),transform_tg.transforms_list[1]])

# KS_test(
#     flow,
#     test_features,
#     test_targets,
#     test_weights,
#     test_aux,
#     transform_tg,
#     transform_au,
#     config,
#     device=device,
#     R_p_estimated=True,
#     sampling_nb=2500,
# )

# Predict challenge values
challenge_predicted_sample = predict_sample_challenge(
    flow,
    transform_ft,
    transform_tg,
    transform_au,
    transform_no,
    config,
    device=device,
    R_p_estimated=True,
    module_nb=1,
    sampling_size=2500,
)
save_predictions(challenge_predicted_sample, config)


