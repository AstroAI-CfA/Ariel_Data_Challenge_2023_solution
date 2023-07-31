"""
Script used to convert the data provided by the Ariel team into Torch tensors.
Converts training data.
The targets here are the input parameters, that is the parameters used to generate the spectra in the first place.
Used to create the features, noises, aux and targets files for the training data.
"""

import h5py
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch 

path="insert_your_path/FullDataset/TrainingData/"
test_path="insert_your_path/FullDataset/TestData/"
infile     = h5py.File(f'{path}Ground Truth Package/Tracedata.hdf5')           ### loading in tracedata.hdf5
planetlist = [p for p in infile.keys()]            ### getting list of planets in file
trace      = infile[planetlist[0]]['tracedata'][:] ### accessing Nested Sampling trace data
weights    = infile[planetlist[0]]['weights'][:]   ### accessing Nested Sampling weight data

run_id="1"


SpectralData = h5py.File(f'{path}SpectralData.hdf5')         ### load file
planetlist = [p for p in SpectralData.keys()]
## access wlgrid, spectrum, noise and wlwidth of a single planet instance
wlgrid = SpectralData[planetlist[0]]['instrument_wlgrid'][:]
spectrum = SpectralData[planetlist[0]]['instrument_spectrum'][:]
noise = SpectralData[planetlist[0]]['instrument_noise'][:]
wlwidth = SpectralData[planetlist[0]]['instrument_width'][:]
parameters=["R","T","H2O","CO2","CO","CH4","NH3"]


fm_targets=pd.read_csv(f'{path}Ground Truth Package/FM_Parameter_Table.csv')
fm_targets=fm_targets.drop(['Unnamed: 0'],axis=1)

AuxTable=pd.read_csv(path+"AuxillaryTable.csv")
AuxTable['ID']=[float(pID[5:]) for pID in AuxTable['planet_ID'].values]
auxparams=['star_distance',	'star_mass_kg',	'star_radius_m',	'star_temperature',	'planet_mass_kg',	'planet_orbital_period',	'planet_distance',	'planet_surface_gravity']


columns=['ID']
id_noise=1
for wl in SpectralData[planetlist[0]]['instrument_wlgrid']:
    columns.append(f"spectrum_{wl}")
    id_noise+=1
for wl in SpectralData[planetlist[0]]['instrument_wlgrid']:
    columns.append(f"noise_{wl}")
    
data=np.zeros((len(planetlist),len(columns)))
j=0
for key in SpectralData.keys():
    data[j,0]=int(key.split("_")[1][5:])
    data[j,1:id_noise]=np.array(SpectralData[key]['instrument_spectrum'])
    data[j,id_noise:]=np.array(SpectralData[key]['instrument_noise'])
    j+=1
df=pd.DataFrame(data,columns=columns)

fm_targets['ID']=[float(pID[5:]) for pID in fm_targets['planet_ID'].values]

full_df=df.merge(fm_targets,on='ID')
full_df=full_df.merge(AuxTable,on='ID')
params_names=["planet_radius",	"planet_temp",	"log_H2O",	"log_CO2",	"log_CO",	"log_CH4",	"log_NH3"]
spectra=torch.tensor(full_df[columns[1:id_noise]].values).float()
noises=torch.tensor(full_df[columns[id_noise:]].values).float()
targets=torch.tensor(full_df[params_names].values).float()
aux=torch.tensor(full_df[auxparams].values).float()

torch.save(spectra,f"{path}../../data/features_{run_id}.pt")
torch.save(noises,f"{path}../../data/noises_{run_id}.pt")
torch.save(targets,f"{path}../../data/targets_{run_id}.pt")
torch.save(aux,f"{path}../../data/aux_{run_id}.pt")

metadatas={}
metadatas['parameters']=parameters
metadatas['num_features']=52
metadatas['auxparams']=['Star Distance',	'Star Mass (kg)',	'Star Radius (m)',	'Star T',	'Planet Mass (kg)',	'Orbital period',	'Planet Distance',	'Gravity']
torch.save(metadatas,f"{path}../../data/metadatas_{run_id}.pt")


test_SpectralData = h5py.File(f'{test_path}SpectralData.hdf5')         ### load file
test_planetlist = [p for p in test_SpectralData.keys()]
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
    data[j,0]=int(key.split("_")[1][6:])
    data[j,1:id_noise]=np.array(test_SpectralData[key]['instrument_spectrum'])
    data[j,id_noise:]=np.array(test_SpectralData[key]['instrument_noise'])
    j+=1
test_df=pd.DataFrame(data,columns=columns)

test_AuxTable=pd.read_csv(test_path+"AuxillaryTable.csv")
test_AuxTable['ID']=[float(pID[6:]) for pID in test_AuxTable['planet_ID'].values]
test_full_df=test_df.merge(test_AuxTable,on='ID')
params_names=["planet_radius",	"planet_temp",	"log_H2O",	"log_CO2",	"log_CO",	"log_CH4",	"log_NH3"]
spectra=torch.tensor(test_full_df[columns[1:id_noise]].values).float()
noises=torch.tensor(test_full_df[columns[id_noise:]].values).float()
aux=torch.tensor(test_full_df[auxparams].values).float()

torch.save(spectra,f"{path}../../data/test/features_{run_id}.pt")
torch.save(noises,f"{path}../../data/test/noises_{run_id}.pt")
torch.save(aux,f"{path}../../data/test/aux_{run_id}.pt")

metadatas={}
metadatas['parameters']=parameters
metadatas['num_features']=52
metadatas['auxparams']=['Star Distance',	'Star Mass (kg)',	'Star Radius (m)',	'Star T',	'Planet Mass (kg)',	'Orbital period',	'Planet Distance',	'Gravity']
torch.save(metadatas,f"{path}../../data/test/metadatas_{run_id}.pt")