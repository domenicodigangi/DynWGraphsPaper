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
import matplotlib.pyplot as plt
import dynwgraphs
from dynwgraphs.utils.tensortools import tens, splitVec
from dynwgraphs.dirSpW1_dynNets_new import  dirSpW1_funs, dirSpW1_sequence_ss, dirSpW1_SD
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

#%% Score driven estimates of  phi_T

model = dirSpW1_SD(Y_T, X_T=X_T, beta_tv=[False], ovflw_lm=True, distr = 'gamma', rescale_SD=False) # 'lognormal')

model.opt_options_sd["max_opt_iter"] = 2500
model.opt_options_sd["opt_n"] = "ADAM"

#%%

optimizer = model.estimate_sd()

model.sd_stat_par_un_phi
model.sd_stat_par_un_beta["w"]
model.un2re_A_par(model.sd_stat_par_un_beta["A"])
model.un2re_B_par(model.sd_stat_par_un_beta["B"])

model.un2re_A_par(model.sd_un_phi["A"])
model.un2re_B_par(model.sd_un_phi["B"])

model.plot_phi_T()
model.plot_beta_T(all_dates)

model.loglike_seq_T()
model.beta_T

# %%

plt.plot(all_dates, Y_T.mean(dim=(0,1)))

plt.plot(all_dates, eonia_w)