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

max_opt_iter = 10
#%% Score driven binary phi_T
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter}

model_bin_0 = dirBin1_SD(Y_T, **filt_kwargs)
# model_bin_0.estimate()

run_0 = mlflow.get_run(run_id = "8e9bd87be0fd4e39a0a76b9097e9aa6a")
model_bin_0.load_par( uri_to_path(run_0.info.artifact_uri)) 

model_bin_0.plot_phi_T()
model_bin_0.beta_T
model_bin_0.out_of_sample_eval()
model_bin_0.loglike_seq_T()

#%% Score driven binary phi_T with const regr
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter, "X_T":X_T, "size_beta_t":1, "beta_tv":False}

model_bin_1 = dirBin1_SD(Y_T, **filt_kwargs)
# model_bin_1.init_par_from_model_without_beta(model_bin_0)
# model_bin_1.estimate_sd()

run_1 = mlflow.get_run(run_id = "1e1fb37e2e36472faff090a13a96ee0e")
model_bin_1.load_par( uri_to_path(run_1.info.artifact_uri)) 


model_bin_1.beta_T
model_bin_1.loglike_seq_T()
model_bin_1.out_of_sample_eval()





#%% Score driven binary phi_T with time varying regr
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter, "X_T":X_T, "size_beta_t":1, "beta_tv":True}

model_bin_2 = dirBin1_SD(Y_T, **filt_kwargs)
# model_bin_2.init_par_from_model_with_const_par(model_bin_1)
# model_bin_2.estimate()
run_2 = mlflow.get_run(run_id = "")
model_bin_2.load_par( uri_to_path(run_2.info.artifact_uri)) 

#%% Score driven binary phi_T with const 
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter, "X_T":X_T, "size_beta_t":2*N, "beta_tv":False}

model_bin_3 = dirBin1_SD(Y_T, **filt_kwargs)
run_3 = mlflow.get_run(run_id = "3857fb74659b4717b90cfb1318965d7e")
model_bin_3.load_par( uri_to_path(run_3.info.artifact_uri)) 

# model_bin_3.init_par_from_model_without_beta(model_bin_0)
# model_bin_3.estimate_sd()



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
