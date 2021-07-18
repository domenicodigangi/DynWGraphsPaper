#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday July 7th 2021

"""


#%% import packages
import os
import numpy as np
from numpy.core.fromnumeric import size
import torch
from pathlib import Path
import pickle
import copy
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import dynwgraphs
from dynwgraphs.utils.tensortools import tens, splitVec
from dynwgraphs.dirGraphs1_dynNets import  dirBin1_sequence_ss, dirBin1_SD, dirSpW1_SD
import importlib

from torch.functional import split
importlib.reload(dynwgraphs)


ld_data = np.load( "../../../../data/emid_data/numpyFiles/eMid_numpy.npz",allow_pickle=True)

eMid_w_T, all_dates, eonia_w, nodes = ld_data["arr_0"], ld_data["arr_1"], ld_data["arr_2"], ld_data["arr_3"]


unit_meas = 1e4
Y_T = tens(eMid_w_T / unit_meas) 
N,_, T = Y_T.shape
X_T = tens(np.tile(eonia_w.T, (N, N, 1, 1)))

T_train = T*3//4

save_path = Path(f"../../data/estimates_real_data/eMid/eonia_reg/T_train_{T_train}")
save_path.mkdir(parents=True, exist_ok=True)



#%% Score driven binary phi_T
estimate_flag = False
model_bin_0 = dirBin1_sequence_ss(Y_T,  T_train=T_train) 

model_bin_0.opt_options_ss_seq["max_opt_iter"] = 11
model_bin_0.estimate_ss_seq_joint()

model_bin_0.par_l_to_opt[0].requires_grad

#%% Score driven binary phi_T
estimate_flag = False
model_bin_0 = dirBin1_SD(Y_T,  T_train=T_train) 

model_bin_0.run(estimate_flag, save_path)


#%% Score driven binary phi_T with const regr
estimate_flag = False

model_bin_1 = dirBin1_SD(Y_T, T_train=T_train, X_T=X_T,  size_beta_t=1, beta_tv=[False]) # 


model_bin_1.init_par_from_model_without_beta(model_bin_0)

model_bin_1.run(estimate_flag, save_path)


#%% Score driven binary phi_T with time varying regr
estimate_flag=False

model_bin_2 = dirBin1_SD(Y_T, T_train=T_train, X_T=X_T,  size_beta_t=1, beta_tv=[True]) # 

# model_bin_2.init_par_from_model_with_const_par(model_bin_1)
# model_bin_2.run(estimate_flag, save_path)

model_bin_2.sd_stat_par_un_beta["A"].data = model_bin_2.re2un_A_par(torch.ones(1)*0.000001)

model_bin_2.opt_options_sd["lr"] = 0.001
model_bin_2.opt_options_sd["opt_n"] = "LBFGS"


model_bin_2.run(estimate_flag, save_path)

#%% Score driven weighted phi_T
estimate_flag = False
model_w_0 = dirSpW1_SD(Y_T, T_train=T_train)  
model_w_0.run(estimate_flag, save_path)


#%% Score driven weighted phi_T with constant beta
estimate_flag = False

model_w_1 = dirSpW1_SD(Y_T, T_train=T_train, X_T=X_T,  size_beta_t=1, beta_tv=[False]) # 

model_w_1.run(estimate_flag, save_path)


#%% Score driven weighted phi_T with time varying beta
estimate_flag = False

model_w_2 = dirSpW1_SD(Y_T, T_train=T_train, X_T=X_T,  size_beta_t=1, beta_tv=[True]) # 

model_w_2.sd_stat_par_un_beta["A"].data = model_w_2.re2un_A_par(torch.ones(1)*0.000001)

model_w_2.opt_options_sd["lr"] = 0.001
model_w_2.opt_options_sd["opt_n"] = "LBFGS"

# model_w_2.init_par_from_model_with_const_par(model_w_1)

model_w_2.run(estimate_flag, save_path)


# %%

dates = all_dates[:T_train]
_, _, beta_bin = model_bin_2.get_seq_latent_par()
_, _, beta_w = model_w_2.get_seq_latent_par()

fig, ax1 = plt.subplots(figsize = (10, 6))
ax2 = ax1.twinx()

ax1.plot(dates, beta_bin.squeeze(), 'g-')
ax2.plot(dates, beta_w.squeeze(), 'b-')

ax1.set_ylabel(r'$\beta_{bin}$', color='b', size=20)
ax2.set_ylabel(r'$\beta_w$', color='g', size=20)
ax1.tick_params(axis='x', labelrotation=45)
ax1.grid()
plt.show()
# %%
