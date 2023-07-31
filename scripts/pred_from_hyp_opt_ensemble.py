"""
Produce the submission for a set of independent ensembles of Normalising Flows.
Requires the zkoptim_{k}_60_57265614.err logs containing the performances of the models obtained during the hyperparameters optimization.
Requires the models states of the hyperparameters optimization to load.
"""

import sys
import signal

sys.path.insert(0, "insert_your_path")
from init_zuko_ariel import *

path='/n/home04/maubin/AstroAI_Exoplanets_Atmospheres/Ariel/logs'
import pandas as pd

total_sampling_nb=4500
num_ensemble=10
number_of_flows=7*num_ensemble
print(f"\nPredicting the challenge samples for Ensemble version {config['net']['version']}.{config['net']['iteration-save']}")
print(f" Loading the hyperparameter optimisation error files to get the best models")

Lpd=[]
for k in range(7):
    with open(path+f"/zkoptim_{k}_60_57265614.err")as f:
        lines=f.readlines()
        d={'iter':[],'score':[],'hidden_features':[],'num_transforms':[],'num_bins':[],'hidden_layers_spline_context':[]}
        for line in lines[1:]:
            if line[:3]=='[I ':
                spl=line.split()[4:]
                d['iter'].append(int(spl[0]))
                d['score'].append(float(spl[4]))
                
                d['hidden_features'].append(int(spl[8][:-1]))
                d['num_transforms'].append(int(spl[10][:-1]))
                d['num_bins'].append(int(spl[12][:-1]))
                d['hidden_layers_spline_context'].append(int(spl[14][:-2]))
           
        dt=pd.DataFrame.from_dict(d)
        dt['target']=k
        dt.sort_values('score',ascending=True,inplace=True)
        Lpd.append(dt)

dt=pd.concat([data.sort_values('score',ascending=True).head(num_ensemble) for data in Lpd])
print("Best values are:")
print(dt)

flows=[]
scores=np.zeros((7,num_ensemble))
print(f" Predicting the challenge test set")
for k in range(number_of_flows):
    parameter_id=k//num_ensemble
    flow_id=k%num_ensemble
    try:
        iteration=int(dt.values[k][0])
        flow, _1, ks, _=torch.load(f"{config['path']['local-path']}{config['path']['states-path']}Flow_from_study_{parameter_id}_{iteration}_{config['net']['version']}.pt")
        scores[parameter_id,flow_id]=ks
        flows.append(flow)
    except Exception as error:
        print(error)
        flows.append(None)
        scores[parameter_id,flow_id]=1.

scores=1-scores
scores=scores/scores.sum(axis=1,keepdims=True)


couting_samples=np.zeros(7)
predictions=[[],[],[],[],[],[],[]]
for k in range(number_of_flows):
    parameter_id=k//num_ensemble
    flow_id=k%num_ensemble
    if flow_id<num_ensemble-1 and (flow_id<num_ensemble-2 or scores[parameter_id,-1]>0.):
        n=int(total_sampling_nb*scores[parameter_id,flow_id])
        couting_samples[parameter_id]+=n
    else:
        n=int(total_sampling_nb-couting_samples[parameter_id])
        couting_samples[parameter_id]+=n
    if n==0:
        print(f"No predictions due to n==0 for parameter {parameter_id} ensemble {flow_id}")
    else:
        print(f"Predicting {n}/{total_sampling_nb} samples for parameter {parameter_id} ensemble {flow_id} of score {scores[parameter_id,flow_id]}. Cumsum of n: {couting_samples[parameter_id]}")
        challenge_predicted_sample=predict_sample_challenge(flows[k],
                                    transform_ft,
                                    transform_tg,
                                    transform_au,
                                    transform_no,
                                    config,
                                    device=device,
                                    R_p_estimated=True,
                                    module_nb=1,
                                    sampling_size=n,
                                    run_id=2)
        predictions[parameter_id].append(challenge_predicted_sample[...,[parameter_id]])

for j in range(7):
    predictions[j]=torch.cat(predictions[j],dim=1)

save_predictions(torch.cat(predictions,dim=2),config,prefix="final",output_prefix="Planet_test")