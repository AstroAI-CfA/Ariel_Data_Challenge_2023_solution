"""
Script used to generate a mapping between the datasets of input parameters (created with Ariel_Dataset_to_tensors.py)
and the NS samples (created with Ariel_Dataset_to_tensors4.py)
Create and save the mapping file.
"""

import h5py
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch 
from tqdm import tqdm

path="insert_your_path/FullDataset/TrainingData/"
test_path="insert_your_path/Ariel/FullDataset/TestData/"
infile     = h5py.File(f'{path}Ground Truth Package/Tracedata.hdf5')           ### loading in tracedata.hdf5
planetlist = [p for p in infile.keys()]            ### getting list of planets in file

Mapping=[[],[]]
k=0
for j,p in enumerate(tqdm(planetlist,desc="Saving the mapping")):
    if infile[p]['weights'].shape:
        Mapping[0].append(j)
        Mapping[1].append(k)
        k+=1
print(f"Number of planets with tracedata: {len(Mapping[0])}")
torch.save(Mapping,f"{path}mapping.pt")
