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
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import pickle
#%% 
unit_meas = 1e4
train_fract = 3/4
experiment_name = f"eMid_application"

experiment = _get_and_set_experiment(experiment_name)

load_and_log_data_run = _get_or_run("load_and_log_data", {}, None)

load_path = uri_to_path(load_and_log_data_run.info.artifact_uri)

load_file = Path(load_path) / "data" / "eMid_data.pkl" 

ld_data = pickle.load(open(load_file, "rb"))

Y_T = tens(ld_data["Y_T"] / unit_meas) 
Y_tm1_T = tens(ld_data["Y_tm1_T"] / unit_meas) 
N, _, T = Y_T.shape
X_T = tens(np.tile(ld_data["eonia_week"].T, (N, N, 1, 1)))
T_train =  int(train_fract * T)
from types import SimpleNamespace
net_stats = SimpleNamespace()
net_stats.__dict__.update({
    "avg_degs_i":(eMid_w_T >0).sum(axis = (0)).mean(axis=1),
    "avg_degs_o":(eMid_w_T >0).sum(axis = (1)).mean(axis=1),
    "avg_str_i":(eMid_w_T).sum(axis = (0)).mean(axis=1),
    "avg_str_o":(eMid_w_T).sum(axis = (1)).mean(axis=1),
    "eonia_T":eonia_w
})

max_opt_iter = 10
#%% Score driven binary phi_T
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter, "beta_tv": False}

model_bin_0 = dirBin1_SD(Y_T, **filt_kwargs)
# model_bin_0.estimate()

filter_string = f"parameters.bin_or_w = 'bin' and parameters.beta_tv = '{filt_kwargs['beta_tv']}' and parameters.str_size_beta_t = '0' "

runs = MlflowClient().search_runs(experiment_ids=experiment.experiment_id, filter_string= filter_string) 

run_0 = runs[0] if len(runs) == 1 else None

model_bin_0.load_par( uri_to_path(run_0.info.artifact_uri)) 

model_bin_0.plot_phi_T()
model_bin_0.beta_T
model_bin_0.out_of_sample_eval()
model_bin_0.loglike_seq_T()

model_bin_0.plot_phi_T()

#%% Score driven binary phi_T with const regr
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter, "X_T":X_T, "size_beta_t":1, "beta_tv":False}

model_bin_1 = dirBin1_SD(Y_T, **filt_kwargs)
# model_bin_1.init_par_from_model_without_beta(model_bin_0)
# model_bin_1.estimate_sd()

filter_string = f"parameters.bin_or_w = 'bin' and parameters.beta_tv = '{filt_kwargs['beta_tv']}' and parameters.str_size_beta_t = '1' "

runs = MlflowClient().search_runs(experiment_ids=experiment.experiment_id, filter_string= filter_string) 

run_1 = runs[0] if len(runs) == 1 else None

model_bin_1.load_par( uri_to_path(run_1.info.artifact_uri)) 


model_bin_1.beta_T
model_bin_1.loglike_seq_T()
model_bin_1.out_of_sample_eval()


#%% Score driven binary phi_T with time varying regr
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter, "X_T":X_T, "size_beta_t":1, "beta_tv":True}

model_bin_2 = dirBin1_SD(Y_T, **filt_kwargs)
# model_bin_2.init_par_from_model_with_const_par(model_bin_1)
# model_bin_2.estimate()
filter_string = f"parameters.bin_or_w = 'bin' and parameters.beta_tv = '{filt_kwargs['beta_tv']}' and parameters.str_size_beta_t = '1' "

runs = MlflowClient().search_runs(experiment_ids=experiment.experiment_id, filter_string= filter_string) 

run_2 = runs[0] if len(runs) == 1 else None

model_bin_2.load_par( uri_to_path(run_2.info.artifact_uri)) 

#%% Score driven binary phi_T with const 
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter, "X_T":X_T, "size_beta_t":2*N, "beta_tv":False}

model_bin_3 = dirBin1_SD(Y_T, **filt_kwargs)
filter_string = f"parameters.bin_or_w = 'bin' and parameters.beta_tv = '{filt_kwargs['beta_tv']}' and parameters.str_size_beta_t = '2N' "

runs = MlflowClient().search_runs(experiment_ids=experiment.experiment_id, filter_string= filter_string) 

run_3 = runs[0] if len(runs) == 1 else None

model_bin_3.load_par( uri_to_path(run_3.info.artifact_uri)) 

beta_in, beta_out = splitVec(model_bin_3.beta_T[0].detach())

plt.scatter(beta_in, beta_out)

# model_bin_3.init_par_from_model_without_beta(model_bin_0)
# model_bin_3.estimate_sd()

#%% Score driven binary phi_T with tv beta
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter, "X_T":X_T, "size_beta_t":2*N, "beta_tv":True}

model_bin_4 = dirBin1_SD(Y_T, **filt_kwargs)
filter_string = f"parameters.bin_or_w = 'bin' and parameters.beta_tv = '{filt_kwargs['beta_tv']}' and parameters.str_size_beta_t = '2N' "

runs = MlflowClient().search_runs(experiment_ids=experiment.experiment_id, filter_string= filter_string) 

run_4 = runs[0] if len(runs) == 1 else None
model_bin_4.load_par( uri_to_path(run_4.info.artifact_uri)) 

plt.scatter(model_bin_3.beta_T[0].detach(), model_bin_4.beta_T[0].detach())


#%% Score driven weighted phi_T
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter, "beta_tv":False}

model_w_0 = dirSpW1_SD(Y_T, **filt_kwargs)
# model_w_0.estimate()

filter_string = f"parameters.bin_or_w = 'w' and parameters.beta_tv = '{filt_kwargs['beta_tv']}' and parameters.str_size_beta_t = '0' "

runs = MlflowClient().search_runs(experiment_ids=experiment.experiment_id, filter_string= filter_string) 

run_0 = runs[0] if len(runs) == 1 else None
model_w_0.load_par( uri_to_path(run_0.info.artifact_uri)) 

model_w_0.plot_phi_T()
model_w_0.beta_T
model_w_0.out_of_sample_eval()

#%% Score driven weighted phi_T with const regr
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter, "X_T":X_T, "size_beta_t":1, "beta_tv":False}

model_w_1 = dirSpW1_SD(Y_T, **filt_kwargs)
# model_w_1.init_par_from_model_without_beta(model_w_0)
# model_w_1.estimate_sd()

filter_string = f"parameters.bin_or_w = 'w' and parameters.beta_tv = '{filt_kwargs['beta_tv']}' and parameters.str_size_beta_t = '1' "

runs = MlflowClient().search_runs(experiment_ids=experiment.experiment_id, filter_string= filter_string) 
run_1 = runs[0] if len(runs) == 1 else None
model_w_1.load_par( uri_to_path(run_1.info.artifact_uri)) 


model_w_1.beta_T
model_w_1.loglike_seq_T()
model_w_1.out_of_sample_eval()


#%% Score driven weighted phi_T with time varying regr
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter, "X_T":X_T, "size_beta_t":1, "beta_tv":True}

model_w_2 = dirSpW1_SD(Y_T, **filt_kwargs)
# model_w_2.init_par_from_model_with_const_par(model_w_1)
# model_w_2.estimate()
filter_string = f"parameters.bin_or_w = 'w' and parameters.beta_tv = '{filt_kwargs['beta_tv']}' and parameters.str_size_beta_t = '1' "

runs = MlflowClient().search_runs(experiment_ids=experiment.experiment_id, filter_string= filter_string) 

run_2 = runs[0] if len(runs) == 1 else None
model_w_2.load_par( uri_to_path(run_2.info.artifact_uri)) 
model_w_2.out_of_sample_eval()


#%% Score driven weighted phi_T with const 
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter, "X_T":X_T, "size_beta_t":2*N, "beta_tv":False}

model_w_3 = dirSpW1_SD(Y_T, **filt_kwargs)
filter_string = f"parameters.bin_or_w = 'w' and parameters.beta_tv = '{filt_kwargs['beta_tv']}' and parameters.str_size_beta_t = '2N' "

runs = MlflowClient().search_runs(experiment_ids=experiment.experiment_id, filter_string= filter_string) 

run_3 = runs[0] if len(runs) == 1 else None
model_w_3.load_par( uri_to_path(run_3.info.artifact_uri)) 

beta_const = model_w_3.beta_T[0].detach()
beta_in, beta_out = splitVec(beta_const)

#%% Explore results
plt.plot(beta_in, beta_out, ".")
plt.loglog(net_stats.avg_str_i, beta_in, ".")
plt.loglog(net_stats.avg_str_i, -beta_in, ".")
plt.loglog(net_stats.avg_str_o, beta_out, ".")
plt.loglog(net_stats.avg_str_o, -beta_out, ".")

#%%
val, inds = torch.topk(beta_out, 25, dim=0)
k = 3
i = inds[k]
fig, ax = model_w_0.plot_phi_T(i = i)
model_w_3.plot_phi_T(i = i, fig_ax = (fig, ax))

model_w_3.plot_phi_T()
model_w_0.plot_phi_T()

# plt.plot(net_stats.eonia_T)
beta_in[i]
beta_out[i]

# model_w_3.init_par_from_model_without_beta(model_w_0)
# model_w_3.estimate_sd()

#%% Score driven weighted phi_T with tv beta
filt_kwargs = {"T_train": T_train, "max_opt_iter": max_opt_iter, "X_T":X_T, "size_beta_t":2*N, "beta_tv":True}

model_w_4 = dirSpW1_SD(Y_T, **filt_kwargs)
filter_string = f"parameters.bin_or_w = 'w' and parameters.beta_tv = '{filt_kwargs['beta_tv']}' and parameters.str_size_beta_t = '2N' "

runs = MlflowClient().search_runs(experiment_ids=experiment.experiment_id, filter_string= filter_string) 

run_4 = runs[0] if len(runs) == 1 else None
model_w_4.load_par( uri_to_path(run_4.info.artifact_uri)) 

plt.scatter(model_w_3.beta_T[0].detach(), model_w_4.beta_T[0].detach())



# model_w_4.init_par_from_model_without_beta(model_w_0)
# model_w_4.estimate_sd()

# %%
