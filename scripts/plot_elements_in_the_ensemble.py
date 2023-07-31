"""
A script to compare models inside an ensemble, notably to search for outliers in the ensemble.
"""

import sys
import signal


sys.path.insert(0, "insert_your_path")

from init_zuko_ariel2 import *

nb_samples=4500
batch_size=100

ens2_name="complete_flow_trained_with_logprob"
Ensemble2=torch.load(f"{config['path']['local-path']}{config['path']['states-path']}Ensemble__{ens2_name}.pth", map_location=device)

# Defining the transform tg 
local_transform_tg = IndexTransform(
                [0], ScalarTransform(GTR_p_estimated_combo[-4000:, :], 1.0)
            )
local_transform_tg1 = ComposedTransforms(
                    [local_transform_tg, transform_tg.transforms_list[1]]
                )
local_transform_tg2 = ComposedTransforms(
                    [local_transform_tg, GTtransform_tg.transforms_list[1]]
                )


print(f"Starting the sampling for Ensemble 2 {ens2_name}...")
couting_samples=0
S=[]
for j in tqdm(range(len(Ensemble2['flows'])-1,-1,-1),desc="Sampling for Ensemble 2"):
    list_of_samples=[]
    for k in range(GTtest_features.shape[0]//batch_size-1):
        list_of_samples.append(torch.swapaxes(Ensemble2['flows'][j](GTtest_features[batch_size*k:batch_size*(k+1)]).sample((nb_samples,)),0,1))
    list_of_samples.append(torch.swapaxes(Ensemble2['flows'][j](GTtest_features[batch_size*(k+1):]).sample((nb_samples,)),0,1))

    _samples=torch.cat(list_of_samples)
    _samples=local_transform_tg2.inverse(_samples).cpu()
    S.append(_samples)
    

tgs2=local_transform_tg2.inverse(GTtest_targets).cpu()
print("Sampling finished")

# plot the coverage
for j in tqdm(range(len(Ensemble2['flows'])-1,-1,-1),desc="Plotting coverages of the models"):
    config['net']['version']=f"model {j} of ensemble 2"
    config['net']['iteration-save']=""
    config['net']['iteration-load']=""
    ranks2=(S[j] < tgs2.unsqueeze(dim=1)).sum(dim=1).detach().cpu().numpy()
    print("Plotting ranks")
    plot_coverage(ranks2,config)