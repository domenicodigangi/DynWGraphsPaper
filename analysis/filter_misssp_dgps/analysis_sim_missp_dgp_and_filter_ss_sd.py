#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Tuesday August 10th 2021

"""


# %% import packages
from pathlib import Path
import importlib
import dynwgraphs
from dynwgraphs.utils.tensortools import splitVec, strIO_from_tens_T
from dynwgraphs.dirGraphs1_dynNets import dirBin1_sequence_ss, dirBin1_SD, dirSpW1_SD, dirSpW1_sequence_ss
from dynwgraphs.utils.dgps import get_dgp_mod_and_par
import mlflow
import logging
import ddg_utils
from ddg_utils.mlflow import _get_and_set_experiment, check_test_exp, get_df_exp, uri_to_path, dict_from_run
from ddg_utils import drop_keys, pd_filt_on
from mlflow.tracking.client import MlflowClient
import pandas as pd
from run_sim_missp_dgp_and_filter_ss_sd import get_filt_mod
import torch
import numpy as np
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
importlib.reload(dynwgraphs)
importlib.reload(ddg_utils)


# %%
# experiment = _get_and_set_experiment("test")
experiment = _get_and_set_experiment("filter missp dgp")

df_0 = get_df_exp(experiment)

df_0["status"].value_counts()
df_0 = df_0[df_0["status"] == "FINISHED"]
df_0 = df_0[~df_0["bin_sd_actual_n_opt_iter"].isna()]


group_cols = ["beta_dgp_set_bin", "beta_filt_set_bin",  "type_tv_dgp_ext_reg", "type_tv_dgp_phi_bin", "type_tv_dgp_beta_bin"]


print(df_0.groupby(group_cols).count()["run_id"])

ind = 6
ind = 5
ind = 1
row_run = df_0.groupby(group_cols).nth(5).iloc[ind]
run_ind = {k: v for k, v in zip(df_0.groupby(group_cols).first().index.names, df_0.groupby(group_cols).first().index[ind])}
print(run_ind)

#%%
load_path = Path(uri_to_path(row_run["artifact_uri"]))
bin_or_w = "bin"
Y_T, X_T = torch.load(open(load_path / "dgp" / "obs_T_dgp.pt", "rb"))

dgp_dict = drop_keys(eval(row_run["dgp_bin"]), ["n_ext_reg", "type_tv_dgp_phi", "type_tv_dgp_ext_reg", "type_tv_dgp_beta"])

mod_dgp = get_filt_mod(bin_or_w, Y_T, X_T, dgp_dict)["ss"]
mod_dgp.load_par(str(load_path / "dgp"))

filt_dict = eval(row_run["filt_bin"])
mod_filt = get_filt_mod(bin_or_w, Y_T, X_T, filt_dict)["sd"]
mod_filt.load_par(str(load_path))

phi_to_exclude = strIO_from_tens_T(Y_T>0) < 1e-3 
i = torch.where(~splitVec(phi_to_exclude)[0])[0][0]
x_T = X_T[0, 0, 0, :].numpy()
strIO_from_tens_T(Y_T>0)

fig_ax = plt.subplots(2, 1)
[ax.plot(x_T) for ax in fig_ax[1]]
mod_filt.plot_phi_T(i=i, fig_ax=mod_dgp.plot_phi_T(i=i, fig_ax=fig_ax))

fig_ax = plt.subplots(2, 1)
[ax.plot(x_T) for ax in fig_ax[1]]
mod_dgp.plot_phi_T(i=i, fig_ax=fig_ax)


# %%
t0 = 0
i=4
phi_T, _, _ = mod_filt.get_seq_latent_par()
phi_i_T, phi_o_T = splitVec(phi_T)
data_i = phi_i_T[i, t0:].reshape(-1,1)
data_i_2 = phi_i_T[i+1, t0:].reshape(-1,1)
data_o = phi_o_T[i, t0:].reshape(-1,1)
data_o_2 = phi_o_T[i+1, t0:].reshape(-1,1)
x = x_T[t0:]
inds = x > - np.inf 
plt.scatter(x[inds], data_i[inds])
plt.figure()
plt.scatter(data_i, data_i_2)

print(f"{run_ind}")
plt.figure()
corr_i = [np.corrcoef(x_T, phi_i_T[i, :])[0,1] for i in range(int(row_run["n_nodes"]))]
corr_o = [np.corrcoef(x_T, phi_o_T[i, :])[0,1] for i in range(int(row_run["n_nodes"]))]
plt.hist(corr_i, alpha=0.5)
plt.hist(corr_o, alpha=0.5)

plt.figure()
beta_i, beta_o = splitVec(mod_dgp.beta_T[0])
plt.scatter(beta_i, corr_i)
plt.scatter(beta_o, corr_o)

#%%
plt.figure()
plt.scatter(mod_dgp.beta_T[0].detach(), mod_filt.beta_T[0].detach())

mod_dgp.beta_T


#%%
plt.figure()
plt.hist([np.cov(x_T, phi_i_T[i, :])[0,1] for i in range(int(row_run["n_nodes"]))], alpha=0.5)
plt.hist([np.cov(x_T, phi_o_T[i, :])[0,1] for i in range(int(row_run["n_nodes"]))], alpha=0.5)

# %%

mod_dgp.beta_T
mod_filt.beta_T

plt.plot(phi_i_T[0, :])

plt.plot(phi_i_T[0,:])
corr = np.corrcoef(np.concatenate((x_T.T, phi_T)))
corr = np.corrcoef())

tmp = np.concatenate((phi_T[14,:].unsqueeze(dim=1).T, phi_T))

np.corrcoef(tmp)[0,14]

corr[11,0]


# %%
