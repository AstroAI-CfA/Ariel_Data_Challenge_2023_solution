"""
Produce the submission for an ensemble of complete Normalising Flows.
Requires an ensemble state "Ensemble__***.pth"
"""

import sys
import signal

sys.path.insert(0, "insert_your_path")
from init_zuko_ariel import *

path='/n/home04/maubin/AstroAI_Exoplanets_Atmospheres/Ariel/logs'
import pandas as pd

nb_samples=4500
# ens2_name="complete_flow_trained_with_logprob"
ens2_name="noised_complete_flow"
Ensemble2=torch.load(f"{config['path']['local-path']}{config['path']['states-path']}Ensemble__{ens2_name}.pth", map_location=device)

couting_samples=0
predictions=[]
for j in tqdm(range(len(Ensemble2['flows'])-1,-1,-1),desc="Predicting challenge for Complete NF"):
    if j!=0:
        n=max(int(nb_samples*Ensemble2['scores'][j]),1)
        couting_samples+=n
    else:
        n=nb_samples-couting_samples
    print(n)
    challenge_predicted_sample=predict_sample_challenge(Ensemble2['flows'][j],
                                    transform_ft,
                                    transform_tg,
                                    transform_au,
                                    transform_no,
                                    config,
                                    device=device,
                                    R_p_estimated=True,
                                    module_nb=1,
                                    sampling_size=n,
                                    run_id=1)
    predictions.append(challenge_predicted_sample)

save_predictions(torch.cat(predictions,dim=1),config,
                 prefix="Complete_NF",
                #  output_prefix="Planet_test"
                 )


