#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Monday September 20th 2021

"""


# %% import packages
import numpy as np
import torch
import scipy
from pathlib import Path
import matplotlib.pyplot as plt
# matplotlib.rcParams['text.usetex'] = True
import dynwgraphs
from dynwgraphs.utils.tensortools import tens, splitVec, strIO_from_tens_T
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


# To do: 
# fare grafico distribuzioni di correlazioni tra regressori e prob link specific


#%% load allruns
experiment = _get_and_set_experiment("emid est paper last")
experiment = _get_and_set_experiment("emid runs paper dev")

df_all_runs = get_df_exp(experiment, one_df=True)

logger.info(f"Staus of experiment {experiment.name}: \n {df_all_runs['status'].value_counts()}")

df_reg = df_all_runs[(df_all_runs["status"].apply(lambda x: x in ["FINISHED"])) & (~np.isnan(df_all_runs["sd_actual_n_opt_iter"]))]

log_cols = ["regressor_name", "size_phi_t", "phi_tv",  "size_beta_t", "beta_tv"]


train_fract = "0.99"
beta_tv = "0"

#%% get bin models
sel_dic = {"size_phi_t": "2N", "phi_tv": "1", "size_beta_t": "1", "beta_tv": beta_tv, "bin_or_w": "bin", "regressor_name": "Atm1", "train_fract": train_fract}

df_sel = pd_filt_on(df_reg, sel_dic).sort_values("end_time")
if df_sel.shape[0] == 1:
    row_run = df_sel.iloc[0, :]
else:
    row_run = df_sel.iloc[0, :]
    logger.error("more than one run")

Y_T, X_T_bin, regr_list, net_stats = get_data_from_data_run(float(row_run["unit_meas"]), row_run["regressor_name"])

_, mod_sd = load_all_models_emid(Y_T, X_T_bin, row_run)

if mod_sd.size_beta_t < 50:
    cov_beta = mod_sd.get_cov_mat_stat_est("beta")
    logger.info(f"{mod_sd.beta_T} +- {cov_beta.sqrt()*1.96}")

mod_bin_tv_phi_lag_reg = mod_sd

sel_dic = {"size_phi_t": "2N", "phi_tv": "1", "size_beta_t": "0", "beta_tv": beta_tv, "bin_or_w": "bin", "regressor_name": "Atm1", "train_fract": train_fract}

df_sel = pd_filt_on(df_reg, sel_dic).sort_values("end_time")
if df_sel.shape[0] == 1:
    row_run = df_sel.iloc[0, :]
else:
    row_run = df_sel.iloc[0, :]
    logger.error("more than one run")


_, mod_sd = load_all_models_emid(Y_T, X_T_bin, row_run)

mod_bin_tv_phi = mod_sd
#%% get w models
sel_dic = {"size_phi_t": "2N", "phi_tv": "1", "size_beta_t": "1", "beta_tv": beta_tv, "bin_or_w": "w", "regressor_name": "logYtm1", "train_fract": train_fract}

df_sel = pd_filt_on(df_reg, sel_dic).sort_values("end_time")
if df_sel.shape[0] == 1:
    row_run = df_sel.iloc[0, :]
else:
    row_run = df_sel.iloc[0, :]
    logger.error("more than one run")

Y_T, X_T_w, regr_list, net_stats = get_data_from_data_run(float(row_run["unit_meas"]), row_run["regressor_name"])


_, mod_sd = load_all_models_emid(Y_T, X_T_w, row_run)

if mod_sd.size_beta_t < 50:
    cov_beta = mod_sd.get_cov_mat_stat_est("beta")
    logger.info(f"{mod_sd.beta_T} +- {cov_beta.sqrt()*1.96}")

mod_w_tv_phi_lag_reg = mod_sd

sel_dic = {"size_phi_t": "2N", "phi_tv": "1", "size_beta_t": "0", "beta_tv": beta_tv, "bin_or_w": "w", "regressor_name": "logYtm1", "train_fract": train_fract}

df_sel = pd_filt_on(df_reg, sel_dic).sort_values("end_time")
if df_sel.shape[0] == 1:
    row_run = df_sel.iloc[0, :]
else:
    row_run = df_sel.iloc[0, :]
    logger.error("more than one run")


_, mod_sd = load_all_models_emid(Y_T, X_T_w, row_run)

mod_w_tv_phi = mod_sd


#%% define useful functions

def corr_finite(x, y, inds_in=None):
    inds = (np.isfinite(x)) & (np.isfinite(y))
    if inds_in is not None:
        inds = inds & inds_in
    return np.corrcoef(x[inds], y[inds])

def get_all_corr(vec_io_T, x_io_T, sub=""):
    n_fit = vec_io_T.shape[0]
    if sub == "in":
        vec_io_T = vec_io_T[:n_fit//2, :]
        x_io_T = x_io_T[:n_fit//2, :]
    elif sub == "out":
        vec_io_T = vec_io_T[n_fit//2:, :]
        x_io_T = x_io_T[n_fit//2:, :]

    n_fit = vec_io_T.shape[0]
    all_corr = {"spearman": np.zeros(n_fit), "linear": np.zeros(n_fit),  "linear_exp": np.zeros(n_fit)} 
    for i in range(n_fit):
        all_corr["spearman"][i] = spearmanr(x_io_T[i, :], vec_io_T[i, :])[0]
        # all_corr["linear"][i] = corr_finite(x_T, vec_io_T[i, :])[0,1]
        # all_corr["linear_exp"][i] = corr_finite(x_T, np.exp(vec_io_T[i, :]))[0,1]
    return all_corr

 
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



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mod_bin_tv_phi_lag_reg.lm_test_beta()

s_T = mod_bin_tv_phi_lag_reg.get_score_T_train(rescaled=True)
plt.plot(s_T["beta"])
plot_acf(s_T["beta"])


mod_w_tv_phi_lag_reg.lm_test_beta()

#%%

all_mods = [mod_bin_tv_phi, mod_bin_tv_phi_lag_reg, mod_w_tv_phi, mod_w_tv_phi_lag_reg]

[mod.roll_sd_filt_all() for mod in  all_mods]


type_corr = "spearman"

T_train = mod_bin_tv_phi.T_train

id_type = "set_ind_to_zero"
id_type = "in_sum_eq_out_sum"

fig, ax = plt.subplots(2, 1, figsize=(12,12))
sub = ""
for sub in [""]:# ["in", "out"]:

    # bin period
    mod_bin_tv_phi.par_vec_id_type = id_type
    mod_bin_tv_phi.identify_par_seq_T(mod_bin_tv_phi.phi_T)
    phi_T, _, _ = mod_bin_tv_phi.get_time_series_latent_par(only_train=True)
    all_corr_no_reg = get_all_corr((phi_T[:, :]), strIO_from_tens_T(X_T_bin[:, :, :, :T_train]), sub=sub)
    plot_dens(all_corr_no_reg[type_corr][2:], ax=ax[0])

    mod_bin_tv_phi_lag_reg.par_vec_id_type = id_type
    mod_bin_tv_phi_lag_reg.identify_par_seq_T(mod_bin_tv_phi_lag_reg.phi_T)
    phi_T, _, _ = mod_bin_tv_phi_lag_reg.get_time_series_latent_par(only_train=True)
    all_corr_reg = get_all_corr((phi_T[:, :]), strIO_from_tens_T(X_T_bin[:, :, :, :T_train]), sub=sub)
    plot_dens(all_corr_reg[type_corr][2:], ax=ax[0])
    plt.suptitle("spearman: fit vs degrees (top) or strengths (bottom)")

    ks_p_val = scipy.stats.ks_2samp(all_corr_no_reg[type_corr], all_corr_reg[type_corr])[1]
    
    ax[0].set_title(f"binary - ks test p-val {ks_p_val}")


    # w  period
    mod_w_tv_phi.par_vec_id_type = id_type
    mod_w_tv_phi.identify_par_seq_T(mod_w_tv_phi.phi_T)
    phi_T, _, _ = mod_w_tv_phi.get_time_series_latent_par(only_train=True)
    all_corr_no_reg = get_all_corr((phi_T[:, :]), strIO_from_tens_T(X_T_w[:, :, :, :T_train]), sub=sub)
    plot_dens(all_corr_no_reg[type_corr][2:], ax=ax[1])

    mod_w_tv_phi_lag_reg.par_vec_id_type = id_type
    mod_w_tv_phi_lag_reg.identify_par_seq_T(mod_w_tv_phi_lag_reg.phi_T)
    phi_T, _, _ = mod_w_tv_phi_lag_reg.get_time_series_latent_par(only_train=True)
    all_corr_reg = get_all_corr((phi_T[:, :]), strIO_from_tens_T(X_T_w[:, :, :, :T_train]), sub=sub)
    plot_dens(all_corr_reg[type_corr][2:], ax=ax[1])
    ax[1].set_title("weighted ")

    ks_p_val = scipy.stats.ks_2samp(all_corr_no_reg[type_corr], all_corr_reg[type_corr])[1]
    
    ax[1].set_title(f"binary - ks test p-val {ks_p_val}")



ax[0].legend(["no reg", "reg = lagged mats"])

[a.grid() for a in ax.flatten()]

# ax[0, 0].legend(["no reg", "reg = eonia"])

#%% Correlation of eonia with fit sums

ind_links = (Y_T[:, :] > 0).float().mean(dim=2) > 0.05

corr_w = lambda fit_sum, x_T: spearmanr(fit_sum.sum(dim=0), x_T)[0]


fig, ax = plt.subplots(2, 2, figsize=(24, 12))
sub = ""
plt.suptitle(f"for each one of {ind_links.float().sum()} links compute spearman of its fitness sum vs eonia", fontsize=20)

# bin 
mod_bin_tv_phi.par_vec_id_type = id_type
mod_bin_tv_phi.identify_par_seq_T(mod_bin_tv_phi.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_bin_tv_phi, ind_links)
all_corr = get_all_corr((all_fit_sum[:, :T_1]), x_T[:T_1], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[0, 0])

mod_bin_tv_phi_lag_reg.par_vec_id_type = id_type
mod_bin_tv_phi_lag_reg.identify_par_seq_T(mod_bin_tv_phi_lag_reg.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_bin_tv_phi_lag_reg, ind_links)
all_corr = get_all_corr((all_fit_sum[:, :T_1]), x_T[:T_1], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[0, 0])
ax[0, 0].set_title("binary ")

# w 
mod_w_tv_phi.par_vec_id_type = id_type
mod_w_tv_phi.identify_par_seq_T(mod_w_tv_phi.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_w_tv_phi, ind_links)
corr_w_no_reg = corr_w(all_fit_sum[:, :T_1], x_T[:T_1])
all_corr = get_all_corr((all_fit_sum[:, :T_1]), x_T[:T_1], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[1, 0])


mod_w_tv_phi_lag_reg.par_vec_id_type = id_type
mod_w_tv_phi_lag_reg.identify_par_seq_T(mod_w_tv_phi_lag_reg.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_w_tv_phi_lag_reg, ind_links)
corr_w_eonia_reg = corr_w(all_fit_sum[:, :T_1], x_T[:T_1])
all_corr = get_all_corr((all_fit_sum[:, :T_1]), x_T[:T_1], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[1, 0])
ax[1, 0].set_title("weighted ")
# ax[1, 0].legend([f"spearman global no reg = {corr_w_no_reg}", f"spearman global eonia  = {corr_w_eonia_reg}"])




# bin second period
ind_links = (Y_T[:, :, T_1:T_train] > 0).float().mean(dim=2) > 0.05
ind_links.sum()

mod_bin_tv_phi.par_vec_id_type = id_type
mod_bin_tv_phi.identify_par_seq_T(mod_bin_tv_phi.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_bin_tv_phi, ind_links)
all_corr = get_all_corr((all_fit_sum[:, T_1:T_train]), x_T[T_1:T_train], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[0, 1])

mod_bin_tv_phi_lag_reg.par_vec_id_type = id_type
mod_bin_tv_phi_lag_reg.identify_par_seq_T(mod_bin_tv_phi_lag_reg.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_bin_tv_phi_lag_reg, ind_links)
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

mod_w_tv_phi_lag_reg.par_vec_id_type = id_type
mod_w_tv_phi_lag_reg.identify_par_seq_T(mod_w_tv_phi_lag_reg.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_w_tv_phi_lag_reg, ind_links)
corr_w_eonia_reg = corr_w(all_fit_sum[:, T_1:T_train], x_T[T_1:T_train])
all_corr = get_all_corr((all_fit_sum[:, T_1:T_train]), x_T[T_1:T_train], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[1, 1])
ax[1, 1].set_title("weighted second period")
# ax[1, 1].legend([f"spearman global no reg = {corr_w_no_reg}", f"spearman global eonia  = {corr_w_eonia_reg}"])

ax[0, 0].legend(["no reg", "reg = eonia"])
[a.grid() for a in ax.flatten()]


#%% single period
ind_links = Y_T.sum(dim=2)>0
ind_links = (Y_T>0).float().mean(dim=2) > 0.05
ind_links.float().sum()

corr_w = lambda fit_sum, x_T: spearmanr(fit_sum.sum(dim=0), x_T)[0]


fig, ax = plt.subplots(2, 1, figsize=(12, 12))
sub = ""
plt.suptitle(f"for each one of {ind_links.float().sum()} links compute spearman of its fitness sum vs eonia", fontsize=20)
T_train = mod_bin_tv_phi.T_train 
# bin period
mod_bin_tv_phi.par_vec_id_type = id_type
mod_bin_tv_phi.identify_par_seq_T(mod_bin_tv_phi.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_bin_tv_phi, ind_links)
all_corr = get_all_corr((all_fit_sum[:, :T_train]), x_T[:T_train], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[0])

mod_bin_tv_phi_lag_reg.par_vec_id_type = id_type
mod_bin_tv_phi_lag_reg.identify_par_seq_T(mod_bin_tv_phi_lag_reg.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_bin_tv_phi_lag_reg, ind_links)
all_corr = get_all_corr((all_fit_sum[:, :T_train]), x_T[:T_train], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[0])
ax[0].set_title("binary ")

# w period
mod_w_tv_phi.par_vec_id_type = id_type
mod_w_tv_phi.identify_par_seq_T(mod_w_tv_phi.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_w_tv_phi, ind_links)
corr_w_no_reg = corr_w(all_fit_sum[:, :T_train], x_T[:T_train])
all_corr = get_all_corr((all_fit_sum[:, :T_train]), x_T[:T_train], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[1])


mod_w_tv_phi_lag_reg.par_vec_id_type = id_type
mod_w_tv_phi_lag_reg.identify_par_seq_T(mod_w_tv_phi_lag_reg.phi_T)
all_fit_sum = get_seq_all_fit_sum(mod_w_tv_phi_lag_reg, ind_links)
corr_w_eonia_reg = corr_w(all_fit_sum[:, :T_train], x_T[:T_train])
all_corr = get_all_corr((all_fit_sum[:, :T_train]), x_T[:T_train], sub=sub)
plot_dens(all_corr[type_corr][2:], ax=ax[1])
ax[1].set_title("weighted ")
# ax[1].legend([f"spearman global no reg = {corr_w_no_reg}", f"spearman global eonia  = {corr_w_eonia_reg}"])


ax[0].legend(["no reg", "reg = eonia"])
[a.grid() for a in ax.flatten()]




# %%

corr_w = lambda fit_sum, x_T: spearmanr(fit_sum.sum(dim=0), x_T)[0]


spearmanr(all_fit_sum[5, :T_train], x_T[:T_train])



corr_w(all_fit_sum[0, :], x_T)
plt.plot(x_T)
plt.plot(all_fit_sum[0, :])
plt.plot(all_fit_sum[3, :])