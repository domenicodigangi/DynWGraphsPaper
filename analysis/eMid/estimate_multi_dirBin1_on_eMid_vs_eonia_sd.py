#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday July 7th 2021

"""


#%% import packages
import os
import numpy as np
import torch
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import dynwgraphs
from dynwgraphs.utils.tensortools import tens, splitVec
from dynwgraphs.dirGraphs1_dynNets import  dirBin1_sequence_ss, dirBin1_SD, dirSpW1_SD
import importlib

from torch.functional import split
importlib.reload(dynwgraphs)

#%%


ld_data = np.load( "../../../../data/emid_data/numpyFiles/eMid_numpy.npz",allow_pickle=True)

eMid_w_T, all_dates, eonia_w, nodes = ld_data["arr_0"], ld_data["arr_1"], ld_data["arr_2"], ld_data["arr_3"]


unit_meas = 1e4
Y_T = tens(eMid_w_T / unit_meas) 
N,_, T = Y_T.shape
X_T = tens(np.tile(eonia_w.T, (N, N, 1, 1)))

save_path = Path("../../data/estimates_real_data/eMid/eonia_reg")
save_path.mkdir(parents=True, exist_ok=True)



#%% Score driven binary phi_T

model_bin_0 = dirBin1_SD(Y_T, rescale_SD=False) 

optimizer = model_bin_0.estimate_sd()

model_bin_0.save_model(save_path)

#%% Score driven binary phi_T with const regr
model_bin_1 = dirBin1_SD(Y_T, X_T=X_T,  size_beta_t=1, beta_tv=[False], rescale_SD=False) # 

optimizer = model_bin_1.estimate_sd()

model_bin_1.save_model(save_path)

#%% Score driven binary phi_T with time varying regr
model_bin_2 = dirBin1_SD(Y_T, X_T=X_T,  size_beta_t=1, beta_tv=[True], rescale_SD=False) # 

optimizer = model_bin_2.estimate_sd()

model_bin_2.save_model(save_path)


#%% Score driven weighted phi_T

model_w_0 = dirSpW1_SD(Y_T)  

model_w_0.estimate_sd()

model_w_0.save_model(save_path)

#%% Score driven weighted phi_T with constant beta

model_w_1 = dirSpW1_SD(Y_T, X_T=X_T,  size_beta_t=1, beta_tv=[False], rescale_SD=False) # 

model_w_1.estimate_sd()

model_w_1.save_model(save_path)


#%% Score driven weighted phi_T with time varying beta

model_w_2 = dirSpW1_SD(Y_T, X_T=X_T,  size_beta_t=1, beta_tv=[True], rescale_SD=False) # 

model_w_2.estimate_sd()

model_w_2.save_model(save_path)



# %%
