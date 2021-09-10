#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday September 4th 2021

"""



# %% import packages
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
# matplotlib.rcParams['text.usetex'] = True
import dynwgraphs
from dynwgraphs.utils.tensortools import tens, splitVec
from dynwgraphs.dirGraphs1_dynNets import dirBin1_SD, dirSpW1_SD
from ddg_utils.mlflow import _get_and_set_experiment, uri_to_path, _get_or_run, get_df_exp
from ddg_utils import pd_filt_on
from mlflow.tracking.client import MlflowClient
import importlib
import pickle
import copy
from scipy.stats import zscore, spearmanr
from scipy.stats import gaussian_kde
from eMid_data_utils import get_data_from_data_run, load_all_models_emid, get_model_from_run_dict_emid
import logging
importlib.reload(dynwgraphs)
logger = logging.getLogger(__name__)



#%% load allruns
experiment = _get_and_set_experiment("emid est paper last")
df_all_runs = get_df_exp(experiment, one_df=True)

logger.info(f"Staus of experiment {experiment.name}: \n {df_all_runs['status'].value_counts()}")

df_reg = df_all_runs[(df_all_runs["status"].apply(lambda x: x in ["FINISHED"])) & (~np.isnan(df_all_runs["sd_actual_n_opt_iter"]))]

log_cols = ["regressor_name", "size_phi_t", "phi_tv",  "size_beta_t", "beta_tv"]


#%% get bin models
sel_dic = {"size_phi_t": "2N", "phi_tv": "1", "size_beta_t": "1", "beta_tv": "0", "bin_or_w": "bin", "regressor_name": "eonia", "train_fract": "0.8"}

df_sel = pd_filt_on(df_reg, sel_dic).sort_values("end_time")
if df_sel.shape[0] == 1:
    row_run = df_sel.iloc[0, :]
else:
    row_run = df_sel.iloc[0, :]
    logger.error("more than one run")

Y_T, X_T, regr_list, net_stats = get_data_from_data_run(float(row_run["unit_meas"]), row_run["regressor_name"])

_, mod_sd = load_all_models_emid(Y_T, X_T, row_run)

if mod_sd.size_beta_t < 50:
    cov_beta = mod_sd.get_cov_mat_stat_est("beta")
    logger.info(f"{mod_sd.beta_T} +- {cov_beta.sqrt()*1.96}")

mod_bin_tv_phi_eonia = mod_sd

sel_dic = {"size_phi_t": "2N", "phi_tv": "1", "size_beta_t": "0", "beta_tv": "0", "bin_or_w": "bin", "regressor_name": "eonia", "train_fract": "0.8"}

df_sel = pd_filt_on(df_reg, sel_dic).sort_values("end_time")
if df_sel.shape[0] == 1:
    row_run = df_sel.iloc[0, :]
else:
    row_run = df_sel.iloc[0, :]
    logger.error("more than one run")


_, mod_sd = load_all_models_emid(Y_T, X_T, row_run)

mod_bin_tv_phi = mod_sd
#%% get w models
sel_dic = {"size_phi_t": "2N", "phi_tv": "1", "size_beta_t": "1", "beta_tv": "0", "bin_or_w": "w", "regressor_name": "eonia", "train_fract": "0.8"}

df_sel = pd_filt_on(df_reg, sel_dic).sort_values("end_time")
if df_sel.shape[0] == 1:
    row_run = df_sel.iloc[0, :]
else:
    row_run = df_sel.iloc[0, :]
    logger.error("more than one run")

_, mod_sd = load_all_models_emid(Y_T, X_T, row_run)

if mod_sd.size_beta_t < 50:
    cov_beta = mod_sd.get_cov_mat_stat_est("beta")
    logger.info(f"{mod_sd.beta_T} +- {cov_beta.sqrt()*1.96}")

mod_w_tv_phi_eonia = mod_sd

sel_dic = {"size_phi_t": "2N", "phi_tv": "1", "size_beta_t": "0", "beta_tv": "0", "bin_or_w": "w", "regressor_name": "eonia", "train_fract": "0.8"}

df_sel = pd_filt_on(df_reg, sel_dic).sort_values("end_time")
if df_sel.shape[0] == 1:
    row_run = df_sel.iloc[0, :]
else:
    row_run = df_sel.iloc[0, :]
    logger.error("more than one run")


_, mod_sd = load_all_models_emid(Y_T, X_T, row_run)

mod_w_tv_phi = mod_sd


#%%

def corr_finite(x, y, inds_in=None):
    inds = (np.isfinite(x)) & (np.isfinite(y))
    if inds_in is not None:
        inds = inds & inds_in
    return np.corrcoef(x[inds], y[inds])

def get_all_corr(vec_io_T, x_T, sub=""):
    n_fit = vec_io_T.shape[0]
    if sub == "in":
        vec_io_T = vec_io_T[:n_fit//2, :]
    elif sub == "out":
        vec_io_T = vec_io_T[n_fit//2:, :]

    n_fit = vec_io_T.shape[0]
    all_corr = {"spearman": np.zeros(n_fit), "linear": np.zeros(n_fit),  "linear_exp": np.zeros(n_fit)} 
    for i in range(n_fit):
        all_corr["spearman"][i] = spearmanr(x_T, vec_io_T[i, :])[0]
        # all_corr["linear"][i] = corr_finite(x_T, vec_io_T[i, :])[0,1]
        # all_corr["linear_exp"][i] = corr_finite(x_T, np.exp(vec_io_T[i, :]))[0,1]
    return all_corr

def get_seq_all_fit_sum(mod, ind_links):
    phi_T, _, _ = mod.get_time_series_latent_par(only_train=True)
    n_links = ind_links.sum()
    all_fit_sum = torch.zeros(n_links, phi_T.shape[1])
    for t in range(phi_T.shape[1]):
        phi_t = mod.get_t_or_t0(t, True, mod.phi_T) 
        all_fit_sum[:, t] = mod.get_phi_sum(phi_t)[ind_links].detach()
    return all_fit_sum

 
def plot_dens(data, ax=None):

    data = data[np.isfinite(data)] 
    density = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 200) # Specifying the limits of our data
    density.covariance_factor = lambda : .5 #Smoothing parameter
    density._compute_covariance()
    if ax is None:
        return plt.plot(x_vals,density(x_vals))
    else:
        return ax.plot(x_vals,density(x_vals))


#%%
dates = net_stats.dates[2:]

dates.shape
T = Y_T.shape[2]
T_1 = 157
x_T = X_T[0, 0, 0, :]
L_T = (Y_T>0).sum(dim=(0,1))
# S_T = Y_T.sum(dim=(0,1))
S_T = Y_T.sum(dim=(0,1))/L_T

fig, ax = plt.subplots(3, 2, figsize=(20, 20))
ax[0, 0].plot(dates[:T_1], zscore(L_T)[:T_1])
ax[0, 0].plot(dates[:T_1], zscore(S_T)[:T_1])
ax[0, 0].plot(dates[:T_1], zscore(x_T)[:T_1])
ax[0, 0].legend(["n links", "avg weight", "eonia"], fontsize=18)
ax[1, 0].loglog(x_T[:T_1], L_T[:T_1], ".")
ax[1, 0].set_xlabel("eonia")
ax[1, 0].set_ylabel("n links")
ax[2, 0].loglog(x_T[:T_1], S_T[:T_1], ".")
ax[2, 0].set_xlabel("eonia")
ax[2, 0].set_ylabel("avg weight")
ax[1, 0].legend([f"n links corr = {corr_finite(x_T[:T_1], L_T[:T_1])[0,1]:.2f}, spear = {spearmanr(x_T[:T_1], L_T[:T_1])[0]:.2f}"], fontsize=18)
ax[2, 0].legend([f"avg weight corr = {corr_finite(x_T[:T_1], S_T[:T_1])[0,1]:.2f}, spear = {spearmanr(x_T[:T_1], S_T[:T_1])[0]:.2f}"], fontsize=18)


ax[0, 1].plot(dates[T_1:], zscore(L_T)[T_1:])
ax[0, 1].plot(dates[T_1:], zscore(S_T)[T_1:])
ax[0, 1].plot(dates[T_1:], zscore(x_T)[T_1:])
ax[0, 1].legend(["n links", "avg weight", "eonia"], fontsize=18)
ax[1, 1].loglog(x_T[T_1:], L_T[T_1:], ".")
ax[1, 1].set_ylabel("n links")
ax[1, 1].set_xlabel("eonia")
ax[2, 1].loglog(x_T[T_1:], S_T[T_1:], ".")
ax[2, 1].set_ylabel("avg weight")
ax[2, 1].set_xlabel("eonia")
ax[1, 1].legend([f"n links corr = {corr_finite(x_T[T_1:], L_T[T_1:])[0,1]:.2f}, spear = {spearmanr(x_T[T_1:], L_T[T_1:])[0]:.2f}"], fontsize=18)
ax[2, 1].legend([f"avg weight corr = {corr_finite(x_T[T_1:], S_T[T_1:])[0,1]:.2f}, spear = {spearmanr(x_T[T_1:], S_T[T_1:])[0]:.2f}"], fontsize=18)

#%%
dates.shape
Y_T.shape
# n active banks
fig, ax = plt.subplots(2, figsize=(14,10))
ax[0].plot(dates[1:], ((Y_T.sum(dim=0)>0) & (Y_T.sum(dim=1)>0)).sum(dim=0)[1:])
ax[0].grid()
ax[1].plot(dates[1:], ((Y_T.sum(dim=0)>0) | (Y_T.sum(dim=1)>0)).sum(dim=0)[1:])
ax[1].grid()
ax[0].legend(["lend. & borr."])
ax[1].legend(["lend or borr"])




#%%


all_mods = [mod_bin_tv_phi, mod_bin_tv_phi_eonia, mod_w_tv_phi, mod_w_tv_phi_eonia]

[mod.roll_sd_filt_all() for mod in  all_mods]

type_corr = "spearman"

T_train = mod_bin_tv_phi.T_train

id_type = "set_ind_to_zero"
id_type = "in_sum_eq_out_sum"


fig, ax = plt.subplots(2, 2, figsize=(24,12))

for sub in ["in", "out"]:

    # bin first period
    mod_bin_tv_phi.par_vec_id_type = id_type
    mod_bin_tv_phi.identify_par_seq_T(mod_bin_tv_phi.phi_T)
    phi_T, _, _ = mod_bin_tv_phi.get_time_series_latent_par(only_train=True)
    all_corr = get_all_corr((phi_T[:, :T_1]), x_T[:T_1], sub=sub)
    plot_dens(all_corr[type_corr][2:], ax=ax[0, 0])

    mod_bin_tv_phi_eonia.par_vec_id_type = id_type
    mod_bin_tv_phi_eonia.identify_par_seq_T(mod_bin_tv_phi_eonia.phi_T)
    phi_T, _, _ = mod_bin_tv_phi_eonia.get_time_series_latent_par(only_train=True)
    all_corr = get_all_corr((phi_T[:, :T_1]), x_T[:T_1], sub=sub)
    plot_dens(all_corr[type_corr][2:], ax=ax[0, 0])
    plt.suptitle("spearman: fit vs eonia")
    ax[0, 0].set_title("binary first period")

    # w first period
    mod_w_tv_phi.par_vec_id_type = id_type
    mod_w_tv_phi.identify_par_seq_T(mod_w_tv_phi.phi_T)
    phi_T, _, _ = mod_w_tv_phi.get_time_series_latent_par(only_train=True)
    all_corr = get_all_corr((phi_T[:, :T_1]), x_T[:T_1], sub=sub)
    plot_dens(all_corr[type_corr][2:], ax=ax[1, 0])

    mod_w_tv_phi_eonia.par_vec_id_type = id_type
    mod_w_tv_phi_eonia.identify_par_seq_T(mod_w_tv_phi_eonia.phi_T)
    phi_T, _, _ = mod_w_tv_phi_eonia.get_time_series_latent_par(only_train=True)
    all_corr = get_all_corr((phi_T[:, :T_1]), x_T[:T_1], sub=sub)
    plot_dens(all_corr[type_corr][2:], ax=ax[1, 0])
    ax[1, 0].set_title("weighted first period")

    # bin second period
    mod_bin_tv_phi.par_vec_id_type = id_type
    mod_bin_tv_phi.identify_par_seq_T(mod_bin_tv_phi.phi_T)
    phi_T, _, _ = mod_bin_tv_phi.get_time_series_latent_par(only_train=True)
    all_corr = get_all_corr((phi_T[:, T_1:T_train]), x_T[T_1:T_train], sub=sub)
    plot_dens(all_corr[type_corr][2:], ax=ax[0, 1])

    mod_bin_tv_phi_eonia.par_vec_id_type = id_type
    mod_bin_tv_phi_eonia.identify_par_seq_T(mod_bin_tv_phi_eonia.phi_T)
    phi_T, _, _ = mod_bin_tv_phi_eonia.get_time_series_latent_par(only_train=True)
    all_corr = get_all_corr((phi_T[:, T_1:T_train]), x_T[T_1:T_train], sub=sub)
    plot_dens(all_corr[type_corr][2:], ax=ax[0, 1])
    ax[0, 1].set_title("binary second period")


    # w second period
    mod_w_tv_phi.par_vec_id_type = id_type
    mod_w_tv_phi.identify_par_seq_T(mod_w_tv_phi.phi_T)
    phi_T, _, _ = mod_w_tv_phi.get_time_series_latent_par(only_train=True)
    all_corr = get_all_corr((phi_T[:, T_1:T_train]), x_T[T_1:T_train], sub=sub)
    plot_dens(all_corr[type_corr][2:], ax=ax[1, 1])

    mod_w_tv_phi_eonia.par_vec_id_type = id_type
    mod_w_tv_phi_eonia.identify_par_seq_T(mod_w_tv_phi_eonia.phi_T)
    phi_T, _, _ = mod_w_tv_phi_eonia.get_time_series_latent_par(only_train=True)
    all_corr = get_all_corr((phi_T[:, T_1:T_train]), x_T[T_1:T_train], sub=sub)
    plot_dens(all_corr[type_corr][2:], ax=ax[1, 1])
    ax[1, 1].set_title("weighted second period")

ax[0, 0].legend(["no reg in", "reg = eonia in", "no reg out", "reg = eonia out"])

[a.grid() for a in ax.flatten()]

# ax[0, 0].legend(["no reg", "reg = eonia"])

#%% Correlation of eonia with fit sums

ind_links = Y_T.sum(dim=2)>0
ind_links = (Y_T>0).float().mean(dim=2) > 0.05
ind_links.float().sum()

corr_w = lambda fit_sum, x_T: spearmanr(fit_sum.sum(dim=0), x_T)[0]


fig, ax = plt.subplots(2, 2, figsize=(24, 12))
sub = ""
plt.suptitle(f"for each one of {ind_links.float().sum()} links compute spearman of its fitness sum vs eonia", fontsize=20)

# bin first period
mod_bin_tv_phi.par_vec_id_type = id_type
mod_bin_tv_phi.identify_par_seq_T(mod_bin_tv_phi.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_bin_tv_phi, ind_links)
all_corr = get_all_corr((all_fit_sum[:, :T_1]), x_T[:T_1], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[0, 0])

mod_bin_tv_phi_eonia.par_vec_id_type = id_type
mod_bin_tv_phi_eonia.identify_par_seq_T(mod_bin_tv_phi_eonia.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_bin_tv_phi_eonia, ind_links)
all_corr = get_all_corr((all_fit_sum[:, :T_1]), x_T[:T_1], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[0, 0])
ax[0, 0].set_title("binary first period")

# w first period
mod_w_tv_phi.par_vec_id_type = id_type
mod_w_tv_phi.identify_par_seq_T(mod_w_tv_phi.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_w_tv_phi, ind_links)
corr_w_no_reg = corr_w(all_fit_sum[:, :T_1], x_T[:T_1])
all_corr = get_all_corr((all_fit_sum[:, :T_1]), x_T[:T_1], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[1, 0])


mod_w_tv_phi_eonia.par_vec_id_type = id_type
mod_w_tv_phi_eonia.identify_par_seq_T(mod_w_tv_phi_eonia.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_w_tv_phi_eonia, ind_links)
corr_w_eonia_reg = corr_w(all_fit_sum[:, :T_1], x_T[:T_1])
all_corr = get_all_corr((all_fit_sum[:, :T_1]), x_T[:T_1], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[1, 0])
ax[1, 0].set_title("weighted first period")
# ax[1, 0].legend([f"spearman global no reg = {corr_w_no_reg}", f"spearman global eonia  = {corr_w_eonia_reg}"])




# bin second period
mod_bin_tv_phi.par_vec_id_type = id_type
mod_bin_tv_phi.identify_par_seq_T(mod_bin_tv_phi.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_bin_tv_phi, ind_links)
all_corr = get_all_corr((all_fit_sum[:, T_1:T_train]), x_T[T_1:T_train], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[0, 1])

mod_bin_tv_phi_eonia.par_vec_id_type = id_type
mod_bin_tv_phi_eonia.identify_par_seq_T(mod_bin_tv_phi_eonia.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_bin_tv_phi_eonia, ind_links)
all_corr = get_all_corr((all_fit_sum[:, T_1:T_train]), x_T[T_1:T_train], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[0, 1])
ax[0, 1].set_title("binary second period")


# w second period
mod_w_tv_phi.par_vec_id_type = id_type
mod_w_tv_phi.identify_par_seq_T(mod_w_tv_phi.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_w_tv_phi, ind_links)
corr_w_no_reg = corr_w(all_fit_sum[:, T_1:T_train], x_T[T_1:T_train])
all_corr = get_all_corr((all_fit_sum[:, T_1:T_train]), x_T[T_1:T_train], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[1, 1])

mod_w_tv_phi_eonia.par_vec_id_type = id_type
mod_w_tv_phi_eonia.identify_par_seq_T(mod_w_tv_phi_eonia.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_w_tv_phi_eonia, ind_links)
corr_w_eonia_reg = corr_w(all_fit_sum[:, T_1:T_train], x_T[T_1:T_train])
all_corr = get_all_corr((all_fit_sum[:, T_1:T_train]), x_T[T_1:T_train], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[1, 1])
ax[1, 1].set_title("weighted second period")
# ax[1, 1].legend([f"spearman global no reg = {corr_w_no_reg}", f"spearman global eonia  = {corr_w_eonia_reg}"])

ax[0, 0].legend(["no reg", "reg = eonia"])
[a.grid() for a in ax.flatten()]




# %%
