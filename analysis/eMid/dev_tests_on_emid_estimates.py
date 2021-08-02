#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday July 7th 2021

"""


#%% import packages
import numpy as np
from numpy.core.fromnumeric import size
import torch
from pathlib import Path
import matplotlib.pyplot as plt
# matplotlib.rcParams['text.usetex'] = True
import dynwgraphs
from dynwgraphs.utils.tensortools import tens, splitVec
from dynwgraphs.dirGraphs1_dynNets import  dirBin1_sequence_ss, dirBin1_SD, dirSpW1_SD
import mlflow
import click
import importlib
from torch.functional import split
importlib.reload(dynwgraphs)
from ddg_utils.mlflow import _get_and_set_experiment, uri_to_path, _get_or_run


#%% 
unit_meas = 1e4
train_fract = 3/4
experiment_name = f"application_eMid"

experiment = _get_and_set_experiment(experiment_name)

load_and_log_data_run = _get_or_run("load_and_log_data", {}, None)
load_path = uri_to_path(load_and_log_data_run.info.artifact_uri)

load_file = Path(load_path) / "data" / "eMid_numpy.npz" 

ld_data = np.load(load_file, allow_pickle=True)

eMid_w_T, all_dates, eonia_w, nodes = ld_data["arr_0"], ld_data["arr_1"], ld_data["arr_2"], ld_data["arr_3"]

Y_T = tens(eMid_w_T / unit_meas) 
N, _, T = Y_T.shape
X_T = tens(np.tile(eonia_w.T, (N, N, 1, 1)))
T_train =  int(train_fract * T)

max_opt_iter = 15000
#%% Score driven binary phi_T
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter}

model_bin_0 = dirBin1_SD(Y_T, **filt_kwargs)

# model_bin_0.estimate()
model_bin_0.load_par( uri_to_path("file:///D:/pCloud/Dynamic_Networks/repos/DynWGraphsPaper/analysis/eMid/mlruns/1/2eeeee68399e4ff8bcdfea9f4b8c6f01/artifacts/")) 

model_bin_0.plot_phi_T()
model_bin_0.out_of_sample_eval()

#%% Score driven binary phi_T with const regr
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter, "X_T":X_T, "size_beta_t":1, "beta_tv":False}

model_bin_1 = dirBin1_SD(Y_T, **filt_kwargs)
model_bin_1.init_par_from_model_without_beta(model_bin_0)
model_bin_1.estimate_sd()
model_bin_1.beta_T[0].is_leaf
model_bin_1.phi_T[0].is_leaf

#%% Score driven binary phi_T with time varying regr
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter, "X_T":X_T, "size_beta_t":1, "beta_tv":True}

model_bin_2 = dirBin1_SD(Y_T, **filt_kwargs)
model_bin_2.init_par_from_model_with_const_par(model_bin_1)

model_bin_2.roll_sd_filt_train()

model_bin_2.get_unc_mean(model_bin_2.sd_stat_par_un_beta)
model_bin_2.un2re_A_par(model_bin_2.sd_stat_par_un_beta["A"])
model_bin_2.loglike_seq_T()
model_bin_1.loglike_seq_T()
model_bin_2.plot_beta_T()

#%% Score driven weighted phi_T
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter, "X_T":X_T, "size_beta_t":2*N, "beta_tv":False}

model_bin_3 = dirBin1_SD(Y_T, **filt_kwargs)
model_bin_3.init_par_from_model_without_beta(model_bin_0)
model_bin_3.beta_T

model_bin_3.estimate_sd()
model_bin_3.par_dict_to_save


model_bin_3.loglike_seq_T()
model_bin_3.plot_phi_T()
model_bin_0.loglike_seq_T()
model_bin_0.plot_phi_T()



#%% Score driven weighted phi_T with constant beta
estimate_flag = False

model_w_1 = dirSpW1_SD(Y_T, T_train=T_train, X_T=X_T,  size_beta_t=1, beta_tv=[False]) # 

model_w_1.load_or_est(estimate_flag, save_path)


#%% Score driven weighted phi_T with time varying beta
estimate_flag = False

model_w_2 = dirSpW1_SD(Y_T, T_train=T_train, X_T=X_T,  size_beta_t=1, beta_tv=[True]) # 

model_w_2.sd_stat_par_un_beta["A"].data = model_w_2.re2un_A_par(torch.ones(1)*0.000001)

model_w_2.opt_options_sd["lr"] = 0.001
model_w_2.opt_options_sd["opt_n"] = "LBFGS"

# model_w_2.init_par_from_model_with_const_par(model_w_1)

model_w_2.load_or_est(estimate_flag, save_path)


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

