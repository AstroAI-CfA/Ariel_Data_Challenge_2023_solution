"""
Script used to convert the data provided by the Ariel team into Torch tensors.
Converts test data.
Used to create the features, noises and aux files for the test data.
"""

import h5py
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch 
from tqdm import tqdm

path="insert_your_path/FullDataset/TrainingData/"
test_path="insert_your_path/FullDataset/final_test_set_data/"

run_id=2
test_SpectralData = h5py.File(f'{test_path}SpectralData.hdf5')         ### load file
test_planetlist = [p for p in test_SpectralData.keys()]
print(test_planetlist)
columns=['ID']
id_noise=1
for wl in test_SpectralData[test_planetlist[0]]['instrument_wlgrid']:
    columns.append(f"spectrum_{wl}")
    id_noise+=1
for wl in test_SpectralData[test_planetlist[0]]['instrument_wlgrid']:
    columns.append(f"noise_{wl}")
    
data=np.zeros((len(test_planetlist),len(columns)))
j=0
for key in test_SpectralData.keys():
    data[j,0]=int(key.split("_")[1][4:])
    data[j,1:id_noise]=np.array(test_SpectralData[key]['instrument_spectrum'])
    data[j,id_noise:]=np.array(test_SpectralData[key]['instrument_noise'])
    j+=1
test_df=pd.DataFrame(data,columns=columns)
print(test_df)

test_AuxTable=pd.read_csv(test_path+"AuxillaryTable.csv")
test_AuxTable['ID']=[float(pID[4:]) for pID in test_AuxTable['planet_ID'].values]
test_full_df=test_df.merge(test_AuxTable,on='ID')
print(test_full_df)
params_names=["planet_radius",	"planet_temp",	"log_H2O",	"log_CO2",	"log_CO",	"log_CH4",	"log_NH3"]
auxparams=['star_distance',	'star_mass_kg',	'star_radius_m',	'star_temperature',	'planet_mass_kg',	'planet_orbital_period',	'planet_distance',	'planet_surface_gravity']
spectra=torch.tensor(test_full_df[columns[1:id_noise]].values).float()
noises=torch.tensor(test_full_df[columns[id_noise:]].values).float()
aux=torch.tensor(test_full_df[auxparams].values).float()

torch.save(spectra,f"{path}../../data/test/features_{run_id}.pt")
torch.save(noises,f"{path}../../data/test/noises_{run_id}.pt")
torch.save(aux,f"{path}../../data/test/aux_{run_id}.pt")

metadatas={}
parameters=["R","T","H2O","CO2","CO","CH4","NH3"]
metadatas['parameters']=parameters
metadatas['num_features']=52
metadatas['auxparams']=['Star Distance',	'Star Mass (kg)',	'Star Radius (m)',	'Star T',	'Planet Mass (kg)',	'Orbital period',	'Planet Distance',	'Gravity']
torch.save(metadatas,f"{path}../../data/test/metadatas_{run_id}.pt")