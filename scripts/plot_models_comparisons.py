"""
Main script used to generate figures and results.
Analyse four models:
    1. The winning model
    2. The alternative ideal model
    3. The alternative noised model
    4. The NS samples
Computes for each the K-S test and the logprob on the input parameters.
Plots the heatmaps (density of samples compared to input parameters), the coverage plots, and some
corner plots comparing the models samples.
The script takes as parameters: 
    1. if we are plotting the heatmaps on one parameter or all (0 for all)
    2. if the spectra should be noised (0 if not noised, 1 if noised)
    3. what colormap to be used for the heatmaps (default viridis)
"""

import sys
import signal


sys.path.insert(0, "insert_your_path")

try:
    one_parameter = int(sys.argv[1])
except:
    one_parameter=False
    
try:
    add_noise = bool(int(sys.argv[2]))
except:
    add_noise=False
    
try:
    cmap = str(sys.argv[3])
except:
    cmap='viridis'
    #'YlOrBr'

print(f"One parameter plots: {one_parameter}, Noise added: {add_noise}, Cmap for heatmaps: {cmap}")

if add_noise:
    from init_zuko_ariel2_add_noise import *
    add_noise_str='noised_spectra'
else:
    from init_zuko_ariel2 import *
    add_noise_str='ideal_spectra'

import scienceplots
plt.style.use(['science'])

nb_samples=4500

# Creating the mapping between the test_data (dataset with samples, so called tracedata) and the GTtest_data (dataset of input parameters, no samples)
print("Starting the mapping between the test_data and the GTtest_data...")
Mapping=torch.load(f"{config['path']['local-path']+config['path']['data-path']}mapping.pt")
j=1
while Mapping[0][-j-1]>=GTtrain_noises.shape[0]:
    j+=1
Mapping=np.array([Mapping[0][-j:], Mapping[1][-j:]])
Mapping[0]-=GTtrain_noises.shape[0]
Mapping[1]-=train_noises.shape[0]
j=0
while Mapping[1,j]<0:
    j+=1
Mapping=Mapping[:,j:]
id_tracedata_map=Mapping[1]
id_GT_map=Mapping[0]

untransformed_test_targets = torch.load(f"{config['path']['local-path']+config['path']['data-path']}targets_{config['data']['run-id']}.pt").to(device)

print(f"================ Information on the data ================")
print(f"\tTotal size of the tracedata dataset: {untransformed_test_targets.shape[0]}")
print(f"\tTotal size of the input parameters dataset: {GTtest_noises.shape[0]+GTtrain_noises.shape[0]}")
print(f"\t\tGiving a ratio of {100*untransformed_test_targets.shape[0]/(GTtest_noises.shape[0]+GTtrain_noises.shape[0]):.2f}% of tracedata")
print(f"\tSize of the tracedata training set: {test_noises.shape[0]}")
print(f"\tSize of the input parameters training set: {GTtest_noises.shape[0]}")
print(f"\tNumber of the tracedata training set mapped: {len(id_tracedata_map)}")
print(f"\t\tGiving a ratio of : {100*len(id_tracedata_map)/test_noises.shape[0]:.2f}% of data mapped for the tracedata")
print(f"\t\tAnd of : {100*len(id_tracedata_map)/GTtest_noises.shape[0]:.2f}% for the input parameters data")

untransformed_test_targets=untransformed_test_targets[-n_test:, :]
# untransformed_test_targets=test_targets # no transforms for testing purposes
print("Sampling the tracedata")

_S=[]
for j in tqdm(range(test_noises.shape[0]),desc="Sampling the tracedata:"):
    _S.append(torch.Tensor(untransformed_test_targets[j,np.random.choice(test_weights[j].shape[0], size=nb_samples, p=test_weights[j].cpu())]).cpu().unsqueeze(dim=0))
samples_from_tracedata=torch.cat(_S,dim=0)

ens1_name="Winning model"
ens2_name="Alternative ideal model"
ens3_name="Alternative noised model"
list_of_ensembles1=torch.load(f"{config['path']['local-path']}{config['path']['states-path']}Ensembles__independent_flows.pth", map_location=device)
Ensemble2=torch.load(f"{config['path']['local-path']}{config['path']['states-path']}Ensemble__complete_flow_trained_with_logprob.pth", map_location=device)
Ensemble3=torch.load(f"{config['path']['local-path']}{config['path']['states-path']}Ensemble__noised_complete_flow.pth", map_location=device)

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


# local_transform_tg=ScalarTransform(0.,1.) # no transform for testing purposes

# Get samples for ensemble 1
print(f"Starting the sampling of the list of ensembles 1 {ens1_name}...")
complete_set_of_samples=[]
for k,ensemble in enumerate(tqdm(list_of_ensembles1,desc="Sampling for the list of ensemble 1:")):
    couting_samples=0
    S=[]
    for j in range(len(ensemble['flows'])-1,-1,-1):
        if j!=0:
            n=int(nb_samples*ensemble['scores'][j])
            couting_samples+=n
        else:
            n=nb_samples-couting_samples
        _samples=ensemble['flows'][j](GTtest_features1).sample((n,))
        S.append(torch.swapaxes(_samples,0,1))
    complete_set_of_samples.append(torch.cat(S,dim=1))
samples1=torch.cat(complete_set_of_samples,dim=-1)

samples1=local_transform_tg1.inverse(samples1).cpu()
tgs1=local_transform_tg1.inverse(GTtest_targets1).cpu()

print("Sampling finished.")


# plot_param_comparison_from_samples(samples.cpu(), tgs.cpu(), config)
config['net']['version']=ens1_name
config['net']['iteration-save']=""+add_noise_str
config['net']['iteration-load']=""+add_noise_str
print("Plotting the parameters comparison heatmap")
plot_param_comparison_heatmap(samples1, tgs1, config, one_parameter=one_parameter,cmap=cmap)
plt.close()

# plot the coverage
ranks1=(samples1 < tgs1.unsqueeze(dim=1)).sum(dim=1).detach().cpu().numpy() #ranks define as "fraction of samples whose parameter values are inferior to the targets"
print("Plotting the coverage")
plot_coverage(ranks1[...,1:4],config,labels=[
        "Planet Temperature",
        "log H2O",
        "log CO2",
    ],
    mean_too=False)
plt.close()

# get the KS score
print("Computing the KS score")
ks1=KS_test_from_samples(samples1[id_GT_map],samples_from_tracedata[id_tracedata_map])
print(f"KS score for Ensemble {ens1_name} is :{ks1:.4f}")

# get the logprob:
print("Computing the logprob")
log_probs=[]
for k,ensemble in enumerate(tqdm(list_of_ensembles1,desc="Computing the logprob")):
    _lp=0.
    for j in range(len(ensemble['flows'])-1,-1,-1):
        _logprob=(ensemble['flows'][j](GTtest_features1).log_prob(GTtest_targets1[...,[k]])).mean().detach().cpu()
        _lp+=_logprob*ensemble['scores'][j]
    log_probs.append(_lp)
logprob1=np.mean(log_probs)
print(f"The mean logprob of Ensembles {ens1_name} is: {logprob1:.4f}")


print("Work finished for the first ensembles")
del list_of_ensembles1, ranks1
gc.collect()

# Get samples for ensemble 2
print(f"Starting the sampling for Ensemble 2 {ens2_name}...")
couting_samples=0
S=[]
for j in tqdm(range(len(Ensemble2['flows'])-1,-1,-1),desc="Sampling for Ensemble 2:"):
    if j!=0:
        n=int(nb_samples*Ensemble2['scores'][j])
        couting_samples+=n
    else:
        n=nb_samples-couting_samples
    _samples=Ensemble2['flows'][j](GTtest_features).sample((n,))
    S.append(torch.swapaxes(_samples,0,1))
    
samples2=torch.cat(S,dim=1)
samples2=local_transform_tg2.inverse(samples2).cpu()
tgs2=local_transform_tg2.inverse(GTtest_targets).cpu()
print("Sampling finished")


# plot_param_comparison_from_samples(samples.cpu(), tgs.cpu(), config)
config['net']['version']=ens2_name
config['net']['iteration-save']=""+add_noise_str
config['net']['iteration-load']=""+add_noise_str
print("Plotting heatmap")
plot_param_comparison_heatmap(samples2.cpu(), tgs2.cpu(), config, one_parameter=one_parameter,cmap=cmap)
plt.close()

# plot the coverage
ranks2=(samples2 < tgs2.unsqueeze(dim=1)).sum(dim=1).detach().cpu().numpy()
print("Plotting ranks")
plot_coverage(ranks2[...,1:4],config,labels=[
        "Planet Temperature",
        "log H2O",
        "log CO2",
    ],
    mean_too=False)
plt.close()

# get the KS score
print("Computing KS score")
ks2=KS_test_from_samples(samples2[id_GT_map],samples_from_tracedata[id_tracedata_map])
print(f"KS score for Ensemble {ens2_name} is :{ks2:.4f}")


# get the logprob
print("Computing the logprob")
logprob2=0.
for j in tqdm(range(len(Ensemble2['flows'])-1,-1,-1),desc="Sampling for Ensemble 2:"):
    if j!=0:
        n=int(nb_samples*Ensemble2['scores'][j])
        couting_samples+=n
    else:
        n=nb_samples-couting_samples
    _logprob=(Ensemble2['flows'][j](GTtest_features).log_prob(GTtest_targets)).mean().detach().cpu()/GTtest_targets.shape[-1]
    logprob2+=_logprob*Ensemble2['scores'][j]
print(f"The mean logprob of Ensemble {ens2_name} is: {logprob2:.4f}")

print("Work finished for the second ensemble")
del Ensemble2, ranks2
gc.collect()



# Get samples for ensemble 3
print(f"Starting the sampling for Ensemble 3 {ens3_name}...")
couting_samples=0
S=[]
for j in tqdm(range(len(Ensemble3['flows'])-1,-1,-1),desc="Sampling for Ensemble 3:"):
    if j!=0:
        n=int(nb_samples*Ensemble3['scores'][j])
        couting_samples+=n
    else:
        n=nb_samples-couting_samples
    _samples=Ensemble3['flows'][j](GTtest_features).sample((n,))
    S.append(torch.swapaxes(_samples,0,1))
    
samples3=torch.cat(S,dim=1)
# print("Computing logprobs for the ranks...")
# logprob_samples3=torch.sum(torch.cat([Ensemble3['flows'][j](GTtest_features).log_prob(samples3).unsqueeze(dim=0)*Ensemble3['scores'][j] for j in range(len(Ensemble3['flows']))],dim=0),dim=0)
# logprob_tgs3=torch.sum(torch.cat([Ensemble3['flows'][j](GTtest_features).log_prob(GTtest_targets).unsqueeze(dim=0)*Ensemble3['scores'][j] for j in range(len(Ensemble3['flows']))],dim=0),dim=0)
samples3=local_transform_tg2.inverse(samples3).cpu()
tgs3=local_transform_tg2.inverse(GTtest_targets).cpu()
print("Sampling finished")


# plot_param_comparison_from_samples(samples.cpu(), tgs.cpu(), config)
config['net']['version']=ens3_name
config['net']['iteration-save']=""+add_noise_str
config['net']['iteration-load']=""+add_noise_str
print("Plotting heatmap")
plot_param_comparison_heatmap(samples3.cpu(), tgs3.cpu(), config, one_parameter=one_parameter,cmap=cmap)
plt.close()

# plot the coverage
ranks3=(samples3 < tgs3.unsqueeze(dim=1)).sum(dim=1).detach().cpu().numpy() #ranks define as "fraction of samples inferior parameter value to the target parameters"
print("Plotting ranks")
plot_coverage(ranks3[...,1:4],config,labels=[
        "Planet Temperature",
        "log H2O",
        "log CO2",
    ],
    mean_too=False)
plt.close()
# print("Plotting ranks (seconde version)")
# ranks3=(logprob_samples3 > logprob_tgs3.unsqueeze(dim=1)).mean(dim=1).detach().cpu().numpy() #ranks define as "fraction of samples superior in logprobability to the targets"
# xr=np.sort(ranks3)
# cdf = np.linspace(0.0, 1.0, len(xr))
# plt.figure(figsize=(4, 4), dpi=300)
# plt.plot(cdf, cdf, color="black", ls="--")
# plt.plot(xr,cdf)
# plt.ylabel("Expected coverage")
# plt.xlabel("Credible level")
# plt.grid(which="both", lw=0.5)
# plt.title(
#     f"Coverage2 for version {config['net']['version']}.{config['net']['iteration-save']}"
# )
# plt.savefig(
#     f"{config['path']['local-path']}{config['path']['plots-path']}Distributions/Coverage2__{config['net']['decoder-name']}_{config['net']['version']}.{config['net']['iteration-save']}.png"
# )

# get the KS score
print("Computing KS score")
ks3=KS_test_from_samples(samples3[id_GT_map],samples_from_tracedata[id_tracedata_map])
print(f"KS score for Ensemble {ens3_name} is :{ks3:.4f}")


# get the logprob
print("Computing the logprob")
logprob3=0.
for j in tqdm(range(len(Ensemble3['flows'])-1,-1,-1),desc="Sampling for Ensemble 3:"):
    _logprob=(Ensemble3['flows'][j](GTtest_features).log_prob(GTtest_targets)).mean().detach().cpu()/GTtest_targets.shape[-1]
    logprob3+=_logprob*Ensemble3['scores'][j]
print(f"The mean logprob of Ensemble {ens3_name} is: {logprob3:.4f}")

print("Work finished for the second ensemble")
del Ensemble3, ranks3
gc.collect()



print("Sample the whole set of sample dataset for getting plots on the whole dataset")
full_targets=torch.load(f"{config['path']['local-path']+config['path']['data-path']}targets_{config['data']['run-id']}.pt").to(device)
full_weights=torch.load(f"{config['path']['local-path']+config['path']['data-path']}weights_{config['data']['run-id']}.pt").to(device)
full_GTtargets=torch.load(f"{config['path']['local-path']+config['path']['data-path']}targets_1.pt").to(device)
Mapping=torch.load(f"{config['path']['local-path']+config['path']['data-path']}mapping.pt")

samples_from_full_tracedata=torch.cat([torch.Tensor(full_targets[Mapping[1]][j,np.random.choice(full_weights[Mapping[1]][j].shape[0], size=nb_samples, p=full_weights[Mapping[1]][j].cpu())]).cpu().unsqueeze(dim=0) for j in tqdm(range(len(Mapping[1])),desc="Sampling the full tracedata")]).cpu()




print("Plotting the heatmap for the tracedata")
config['net']['version']="tracedata"
plot_param_comparison_heatmap(samples_from_full_tracedata, full_GTtargets[Mapping[0]].cpu(), config, one_parameter=one_parameter,cmap=cmap)
plt.close()

ranks4=(samples_from_full_tracedata < full_GTtargets[Mapping[0]].unsqueeze(dim=1).cpu()).sum(dim=1).detach().cpu().numpy()
print("Plotting ranks for tracedata")
plot_coverage(ranks4[...,1:4],config,labels=[
        "Planet Temperature",
        "log H2O",
        "log CO2",
    ],
    mean_too=False)
plt.close()

# get the KS score
print("Computing KS score for tracedata")
ks4=KS_test_from_samples(samples_from_tracedata[id_tracedata_map],samples_from_tracedata[id_tracedata_map])
print(f"KS score for tracedata (should be 0) is :{ks4:.4f}")



# # Get samples for a single model of complete flows:
# flow4=torch.load(f"{config['path']['local-path']}{config['path']['states-path']}ZKFlow_ZKA.8.1.1.pth", map_location=device)
# # flow4=torch.load(f"{config['path']['local-path']}{config['path']['states-path']}Flow_from_study__2_ZKA.6.1.pt", map_location=device)[0]
# print("Sampling the single model flow...")
# batch_size=100
# list_of_samples=[]
# for j in tqdm(range(GTtest_features.shape[0]//batch_size-1),desc="Sampling the single flow"):
#     list_of_samples.append(torch.swapaxes(flow4(GTtest_features[batch_size*j:batch_size*(j+1)]).sample((nb_samples,)),0,1))
# list_of_samples.append(torch.swapaxes(flow4(GTtest_features[batch_size*(GTtest_features.shape[0]//batch_size-1):]).sample((nb_samples,)),0,1))
# samples4=torch.cat(list_of_samples)
# samples4=local_transform_tg2.inverse(samples4).cpu()
# tgs4=local_transform_tg2.inverse(GTtest_targets).cpu()
# print("Sampling finished.")


# config['net']['version']="Flow ZKA.8.1"
# config['net']['iteration-save']="1"+add_noise_str
# config['net']['iteration-load']="1"+add_noise_str
# print("Plotting heatmap")
# plot_param_comparison_heatmap(samples4.cpu(), tgs4.cpu(), config,one_parameter=one_parameter,cmap=cmap)


# # plot the coverage
# ranks4=(samples4 < tgs4.unsqueeze(dim=1)).sum(dim=1).detach().cpu().numpy()
# print("Plotting ranks")
# plot_coverage(ranks4,config)

# # get the KS score
# print("Computing KS score")
# ks4=KS_test_from_samples(samples4[id_GT_map],samples_from_tracedata[id_tracedata_map])
# print(f"KS score for model ZKA.8.1.1 is :{ks4:.4f}")


# # get the logprob
# print("Computing the logprob")
# logprob4=(flow4(GTtest_features).log_prob(GTtest_targets)).mean().detach().cpu()/GTtest_targets.shape[-1]
# print(f"The mean logprob of the model ZKA.8.1.1 is :{logprob4:.4f}")


# print("Comparing the tgs:")
# print(f"  Mean of the abs diff: {torch.mean(torch.abs(tgs1-tgs2)):.2e}")



print("Plotting some posterior comparison")
for i in range(5):
    p=np.random.randint(len(id_tracedata_map))
    # plot_several_posteriors([samples_from_tracedata[id_tracedata_map][p,:,1:4],samples1[id_GT_map][p,:,1:4],samples2[id_GT_map][p,:,1:4]],
    #                         config,
    #                         ground_truth=tgs1[id_GT_map][p,1:4],
    #                         labels=['Nested Sampling samples','Winning model (trained on NS samples)','Alternative ideal model (trained on\n input parameters)'],
    #                         title=f"{p}_{add_noise_str}_extract",
    #                         colors=["#377eb8","#ff7f00","#4daf4a",'#f781bf'],
    #                         params_names=[
    #                                 "Planet Temperature",
    #                                 "log H2O",
    #                                 "log CO2",
    #                             ])
    if add_noise:
        plot_several_posteriors([samples1[id_GT_map][p,:,1:4],samples2[id_GT_map][p,:,1:4],samples3[id_GT_map][p,:,1:4]],
                                config,
                                ground_truth=tgs1[id_GT_map][p,1:4],
                                labels=['Winning model (trained on NS samples)','Alternative ideal model (trained on\n input parameters)','Alternative noised model (trained on\n input parameters)'],
                                title=f"{p}_{add_noise_str}_extract",
                                colors=["#ff7f00","#4daf4a",'#f781bf'],
                                params_names=[
                                        "Planet Temperature",
                                        "log H2O",
                                        "log CO2",
                                    ])
        plt.close()
        p=np.random.randint(len(id_tracedata_map))
        plot_several_posteriors([samples1[id_GT_map][p,:,1:4],samples3[id_GT_map][p,:,1:4]],
                                config,
                                ground_truth=tgs1[id_GT_map][p,1:4],
                                labels=['Winning model (trained on NS samples)','Alternative noised model (trained on\n input parameters)'],
                                title=f"{p}_{add_noise_str}_extract2",
                                colors=["#ff7f00",'#f781bf'],
                                params_names=[
                                        "Planet Temperature",
                                        "log H2O",
                                        "log CO2",
                                    ])
        plt.close()
        p=np.random.randint(len(id_tracedata_map))
        plot_several_posteriors([samples_from_tracedata[id_tracedata_map][p,:,:],samples1[id_GT_map][p,:,:],samples3[id_GT_map][p,:,:]],
                                config,
                                ground_truth=tgs1[id_GT_map][p,:],
                                labels=['Nested Sampling samples','Winning model (trained on \n NS samples with ideal spectra)','Alternative model (trained on \ninput parameters with noised spectra)'],
                                title=f"{p}_{add_noise_str}_full",
                                colors=["#377eb8","#ff7f00",'#f781bf'],
                                params_names=[
                                        "Planet Radius",
                                        "Planet Temperature",
                                        "log H2O",
                                        "log CO2",
                                        "log CO",
                                        "log CH4",
                                        "log NH3",
                                    ]
                                )
        plt.close()
    else:
        plot_several_posteriors([samples_from_tracedata[id_tracedata_map][p,:,1:4],samples1[id_GT_map][p,:,1:4],samples2[id_GT_map][p,:,1:4]],
                                config,
                                ground_truth=tgs1[id_GT_map][p,1:4],
                                labels=['Nested Sampling samples','Winning model (trained on NS samples)','Alternative ideal model (trained on\n input parameters)'],
                                title=f"{p}_{add_noise_str}_extract",
                                colors=["#377eb8","#ff7f00","#4daf4a"],
                                params_names=[
                                        "Planet Temperature",
                                        "log H2O",
                                        "log CO2",
                                    ])
        plt.close()
        p=np.random.randint(len(id_tracedata_map))
        plot_several_posteriors([samples_from_tracedata[id_tracedata_map][p,:,1:4],samples1[id_GT_map][p,:,1:4]],
                                config,
                                ground_truth=tgs1[id_GT_map][p,1:4],
                                labels=['Nested Sampling samples','Winning model (trained on NS samples)'],
                                title=f"{p}_{add_noise_str}_extract2",
                                colors=["#377eb8","#ff7f00"],
                                params_names=[
                                        "Planet Temperature",
                                        "log H2O",
                                        "log CO2",
                                    ])
        plt.close()
        p=np.random.randint(len(id_tracedata_map))
        plot_several_posteriors([samples_from_tracedata[id_tracedata_map][p,:,:],samples1[id_GT_map][p,:,:],samples3[id_GT_map][p,:,:]],
                                config,
                                ground_truth=tgs1[id_GT_map][p,:],
                                labels=['Nested Sampling samples','Winning model (trained on \n NS samples with ideal spectra)','Alternative model (trained on \ninput parameters with noised spectra)'],
                                title=f"{p}_{add_noise_str}_full",
                                colors=["#377eb8","#ff7f00",'#f781bf'],
                                params_names=[
                                        "Planet Radius",
                                        "Planet Temperature",
                                        "log H2O",
                                        "log CO2",
                                        "log CO",
                                        "log CH4",
                                        "log NH3",
                                    ]
                                )
        plt.close()