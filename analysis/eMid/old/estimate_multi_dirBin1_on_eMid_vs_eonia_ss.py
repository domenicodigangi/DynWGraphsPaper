
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday July 13th 2021

"""


# %% import packages
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import dynwgraphs
from dynwgraphs.utils.tensortools import tens, splitVec
from dynwgraphs.dirGraphs1_dynNets import  dirBin1_sequence_ss, dirBin1_SD
import importlib

from torch.functional import split
importlib.reload(dynwgraphs)

# %%


ld_data = np.load( "../../../../data/emid_data/numpyFiles/eMid_numpy.npz",allow_pickle=True)

eMid_w_T, all_dates, eonia_w, nodes = ld_data["arr_0"], ld_data["arr_1"], ld_data["arr_2"], ld_data["arr_3"]


unit_meas = 1e4
Y_T = tens(eMid_w_T / unit_meas) 
N,_, T = Y_T.shape
X_T = tens(np.tile(eonia_w.T, (N, N, 1, 1)))

# %% single snap seq estimates
model_ss_reg = dirBin1_sequence_ss(Y_T, X_T=X_T,  size_beta_t=1, beta_tv=[False]) 
model_ss_reg.opt_options_ss_seq["max_opt_iter"] = 25
model_ss_reg.opt_options_ss_seq["opt_n"] = "ADAM"

model_ss_reg.estimate_ss_seq_joint()
d = model_ss_reg.state_dict()

list(model_ss_reg.named_parameters())

d.keys()

# %%
model_ss_reg.phi_T
model_ss_reg.beta_T
x_T = model_ss_reg.X_T[0, 0, model_ss_reg.reg_cross_unique, :].squeeze()

sum([phi[model_ss_reg.N:] for phi in model_ss_reg.phi_T]).sum()

model_ss_reg.loglike_seq_T()

model_ss_reg.shift_sequence_phi_o_T_beta_const(-model_ss_reg.beta_T[0][0], x_T)
model_ss_reg.identify_sequence()
model_ss_reg.loglike_seq_T()
model_ss_reg.plot_phi_T()


