"""
Script used to convert the data provided by the Ariel team into Torch tensors.
Converts training data.
The targets here are the tracedata, that is the NS samples (samples and weights).
Used to create the features, noises, aux, targets, weights files for the training data.
"""

import h5py
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch 
from tqdm import tqdm

path="insert_your_path/FullDataset/TrainingData/"
test_path="insert_your_path/FullDataset/TestData/"
infile     = h5py.File(f'{path}Ground Truth Package/Tracedata.hdf5')           ### loading in tracedata.hdf5
planetlist = [p for p in infile.keys()]            ### getting list of planets in file
trace      = infile[planetlist[0]]['tracedata'][:] ### accessing Nested Sampling trace data
weights    = infile[planetlist[0]]['weights'][:]   ### accessing Nested Sampling weight data

count=0
max_weights=1000
for p in planetlist:
    if infile[p]['weights'].shape:
        count+=1
        max_weights=max(infile[p]['weights'].shape[0],max_weights)
print(f"Number of planets with tracedata: {count}")
print(f"Maximum number of weights: {max_weights}")

number_of_points=max_weights
run_id=7

print("Loading datafiles...")

SpectralData = h5py.File(f'{path}SpectralData.hdf5')
## access wlgrid, spectrum, noise and wlwidth of a single planet instance
wlgrid = SpectralData[planetlist[0]]['instrument_wlgrid'][:]
spectrum = SpectralData[planetlist[0]]['instrument_spectrum'][:]
noise = SpectralData[planetlist[0]]['instrument_noise'][:]
wlwidth = SpectralData[planetlist[0]]['instrument_width'][:]
parameters=["R","T","H2O","CO2","CO","CH4","NH3"]
params_names=["planet_radius",	"planet_temp",	"log_H2O",	"log_CO2",	"log_CO",	"log_CH4",	"log_NH3"]

AuxTable=pd.read_csv(path+"AuxillaryTable.csv")
AuxTable['ID']=[int(pID[5:]) for pID in AuxTable['planet_ID'].values]
auxparams=['star_distance',	'star_mass_kg',	'star_radius_m',	'star_temperature',	'planet_mass_kg',	'planet_orbital_period',	'planet_distance',	'planet_surface_gravity']

ids= [int(p[12:]) for p in planetlist]
planetlist=np.array(planetlist)

columns=['ID']
id_noise=1
for wl in SpectralData[planetlist[0]]['instrument_wlgrid']:
    columns.append(f"spectrum_{wl}")
    id_noise+=1
for wl in SpectralData[planetlist[0]]['instrument_wlgrid']:
    columns.append(f"noise_{wl}")
    
data=np.zeros((len(planetlist),len(columns)))

for j, key in enumerate(tqdm(SpectralData.keys(),desc="Constructing spectra df:")):
    data[j,0]=int(key.split("_")[1][5:])
    data[j,1:id_noise]=np.array(SpectralData[key]['instrument_spectrum'])
    data[j,id_noise:]=np.array(SpectralData[key]['instrument_noise'])
df=pd.DataFrame(data,columns=columns)
print(df)

TARGETS=[]
WEIGHTS=[]
SPECTRA=[]
NOISES=[]
AUX=[]
j=0

bar=tqdm(planetlist,desc="Generating dataset:")
for p in bar:
    if len(infile[p]['weights'].shape):
        j+=1
        
        tracedata=np.array(infile[p]['tracedata'])
        weights=np.array(infile[p]['weights'])
        additional_zeros=number_of_points-len(weights)
        
        _targets=torch.Tensor(tracedata).float()
        _weights=torch.Tensor(weights).float()
        _targets=torch.nn.functional.pad(input=_targets, pad=(0,0, 0, additional_zeros), mode='constant', value=0.)
        _weights=torch.nn.functional.pad(input=_weights, pad=(0, additional_zeros), mode='constant', value=0.)
        # print(_targets.shape, _weights.shape)
        
        _targets=_targets.unsqueeze(dim=0)
        _weights=_weights.unsqueeze(dim=0)
        
        # bar.set_postfix({"j": f"{j}", "p": f"{p[12:]}", "sh": f"{infile[p]['weights'].shape}"})
        
        _spectra=torch.tensor(df[df['ID']==int(p[12:])][columns[1:id_noise]].values).float()
        _noises=torch.tensor(df[df['ID']==int(p[12:])][columns[id_noise:]].values).float()
        _aux=torch.tensor(AuxTable[AuxTable['ID']==int(p[12:])][auxparams].values).float()
        if len(_spectra.shape)==1:
            _spectra=_spectra.unsqueeze(dim=0)
            _noises=_noises.unsqueeze(dim=0)
            _aux=_aux.unsqueeze(dim=0)
        elif len(_spectra.shape)==2 and _spectra.shape[0]!=1:
            raise KeyError(f"Too many spectra: {_spectra.shape}")
        # else:
        #     raise KeyError(f"Too many dimensions: {_spectra.shape}")
        
        TARGETS.append(_targets.clone())
        WEIGHTS.append(_weights.clone())
        SPECTRA.append(_spectra.clone())
        NOISES.append(_noises.clone())
        AUX.append(_aux.clone())
    
targets=torch.cat(TARGETS,dim=0)
weights=torch.cat(WEIGHTS,dim=0)
spectra=torch.cat(SPECTRA,dim=0)
noises=torch.cat(NOISES,dim=0)
aux=torch.cat(AUX,dim=0)

print("Saving tensors...")

torch.save(spectra,f"{path}../../data/features_{run_id}.pt")
torch.save(noises,f"{path}../../data/noises_{run_id}.pt")
torch.save(targets,f"{path}../../data/targets_{run_id}.pt")
torch.save(weights,f"{path}../../data/weights_{run_id}.pt")
torch.save(aux,f"{path}../../data/aux_{run_id}.pt")

metadatas={}
metadatas['parameters']=parameters
metadatas['num_features']=52
metadatas['auxparams']=['Star Distance',	'Star Mass (kg)',	'Star Radius (m)',	'Star T',	'Planet Mass (kg)',	'Orbital period',	'Planet Distance',	'Gravity']
metadatas['test_set_excluded']=False
metadatas['test_size']=4000
metadatas['number_of_points']=number_of_points
torch.save(metadatas,f"{path}../../data/metadatas_{run_id}.pt")
print("Done!")