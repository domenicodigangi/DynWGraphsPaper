#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday October 13th 2021

"""


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
import scipy
import matplotlib.pyplot as plt
# matplotlib.rcParams['text.usetex'] = True
import dynwgraphs
from dynwgraphs.utils.tensortools import strIO_from_tens_T, strIO_T_from_tens_T, tens, splitVec
from dynwgraphs.dirGraphs1_dynNets import dirBin1_SD, dirSpW1_SD
from ddg_utils.mlflow import _get_and_set_experiment, uri_to_path, _get_or_run, get_df_exp
from ddg_utils import pd_filt_on
from mlflow.tracking.client import MlflowClient
import importlib
import pickle
import copy
from scipy.stats import zscore, spearmanr
from scipy.stats import gaussian_kde
from torch.nn.functional import normalize
from eMid_data_utils import get_data_from_data_run, load_all_models_emid, get_model_from_run_dict_emid
import logging
importlib.reload(dynwgraphs)
logger = logging.getLogger(__name__)









# To Do : run only sd estimates without regressors. re estimate ss and run link prediction



#%% load allruns

experiment = _get_and_set_experiment("emid est paper last")
# experiment = _get_and_set_experiment("emid runs paper dev")
df_all_runs = get_df_exp(experiment, one_df=True)

logger.info(f"Staus of experiment {experiment.name}: \n {df_all_runs['status'].value_counts()}")

df_reg = df_all_runs[(df_all_runs["status"].apply(lambda x: x in ["FINISHED"])) & (~np.isnan(df_all_runs["sd_actual_n_opt_iter"]))]

log_cols = ["regressor_name", "size_phi_t", "phi_tv",  "size_beta_t", "beta_tv"]


train_fract = "0.99"
beta_tv = "0"
size_beta_t = "1"
#%% get bin models
# region
sel_dic = {"size_phi_t": "2N", "phi_tv": "1", "size_beta_t": size_beta_t, "beta_tv": beta_tv, "bin_or_w": "bin", "regressor_name": "eonia", "train_fract": train_fract}

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

sel_dic = {"size_phi_t": "2N", "phi_tv": "1", "size_beta_t": "0", "beta_tv": "0", "bin_or_w": "bin", "regressor_name": "eonia", "train_fract": train_fract}

df_sel = pd_filt_on(df_reg, sel_dic).sort_values("end_time")
if df_sel.shape[0] == 1:
    row_run = df_sel.iloc[0, :]
else:
    row_run = df_sel.iloc[0, :]
    logger.error("more than one run")


_, mod_sd = load_all_models_emid(Y_T, X_T, row_run)

#endregion
mod_bin_tv_phi = mod_sd
#%% get w models
#region
sel_dic = {"size_phi_t": "2N", "phi_tv": "1", "size_beta_t": size_beta_t, "beta_tv": beta_tv, "bin_or_w": "w", "regressor_name": "eonia", "train_fract": train_fract}

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

sel_dic = {"size_phi_t": "2N", "phi_tv": "1", "size_beta_t": "0", "beta_tv": "0", "bin_or_w": "w", "regressor_name": "eonia", "train_fract": train_fract}

df_sel = pd_filt_on(df_reg, sel_dic).sort_values("end_time")
if df_sel.shape[0] == 1:
    row_run = df_sel.iloc[0, :]
else:
    row_run = df_sel.iloc[0, :]
    logger.error("more than one run")


_, mod_sd = load_all_models_emid(Y_T, X_T, row_run)

mod_w_tv_phi = mod_sd

#endregion

T = Y_T.shape[2]
T_1 = 157
x_T = X_T[0, 0, 0, :]
L_T = (Y_T>0).sum(dim=(0,1))
# S_T = Y_T.sum(dim=(0,1))
S_T = Y_T.sum(dim=(0,1))/L_T


#%% define useful functions

#region
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
    density.covariance_factor = lambda : .4 #Smoothing parameter
    density._compute_covariance()
    if ax is None:
        return plt.plot(x_vals,density(x_vals))
    else:
        return ax.plot(x_vals,density(x_vals))
#endregion

#%% eonia and network stats

#region 
dates = net_stats.dates[2:]

dates.shape

plt.plot(dates, L_T)
plt.grid()
plt.ylabel("Number of links")

plt.plot(dates, S_T/100)
plt.grid()
plt.ylabel("Avg. Weight Present Links [Mln]")

plt.plot(dates, x_T)
plt.grid()
plt.ylabel("EONIA")

plt.plot(dates, S_T)
plt.grid()
plt.plot(dates, x_T)
plt.grid()
plt.legend(["n links", "avg weight", "eonia"], fontsize=18)
plt.loglog(x_T, L_T, ".")
plt.set_xlabel("eonia")
plt.set_ylabel("n links")
plt.loglog(x_T, S_T, ".")
plt.set_xlabel("eonia")
plt.set_ylabel("avg weight")
plt.legend([f"n links corr = {corr_finite(x_T, L_T)[0,1]:.2f}, spear = {spearmanr(x_T, L_T)[0]:.2f}"], fontsize=18)
plt.legend([f"avg weight corr = {corr_finite(x_T, S_T)[0,1]:.2f}, spear = {spearmanr(x_T, S_T)[0]:.2f}"], fontsize=18)
#endregion


#%% Correlation between eonia and network stats

#region
dates = net_stats.dates[2:]

dates.shape
T = Y_T.shape[2]
T_1 = 157
x_T = X_T[0, 0, 0, :]
L_T = (Y_T>0).sum(dim=(0,1))
# S_T = Y_T.sum(dim=(0,1))
S_T = Y_T.sum(dim=(0,1))/L_T

fig, ax = plt.subplots(3, 2, figsize=(17, 17))
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

#endregion

#%% Number of banks

#region 
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
plt.suptitle(f"Banks observed at least once = {Y_T.shape[0]}")

in_out_w_sum = Y_T.sum(dim=(0, 2)) + Y_T.sum(dim=(1, 2))
plot_dens(np.log(in_out_w_sum).numpy())

# how many banks left?
Y_T_1 = Y_T[:, :, :T_1]
Y_T_2 = Y_T[:, :, T_1:]
in_out_w_sum_1 = Y_T_1.sum(dim=(0, 2)) + Y_T_1.sum(dim=(1, 2))
in_out_w_sum_2 = Y_T_2.sum(dim=(0, 2)) + Y_T_2.sum(dim=(1, 2))


plt.hist(np.log(in_out_w_sum_1[in_out_w_sum_1>0]).numpy(), alpha=0.5, density=True)
plt.hist(np.log(in_out_w_sum_2[in_out_w_sum_2>0]).numpy(), alpha=0.5, density=True)

(((Y_T_2).sum(dim=(0, 2)) + Y_T_2.sum(dim=(1, 2))) < 10).sum()
#endregion


#%% hist of  correlations between eonia and observed degrees

# region

type_corr = "spearman"

T_train = mod_bin_tv_phi.T_train

fig, ax = plt.subplots(2, 2, figsize=(24,12))


def obs_degs_corr_and_plot(Y_T, x_T, T_in, T_fin, sub, type_corr, ax):
    degsIO_T = strIO_T_from_tens_T(Y_T)
    all_corr = get_all_corr((degsIO_T[:, T_in:T_fin]), x_T[T_in:T_fin], sub=sub)

    # ax.hist(all_corr[type_corr][2:], alpha=0.6, density=False)
    plot_dens(all_corr[type_corr][2:], ax=ax)

A_T = (Y_T > 0).double() 




for sub in ["in", "out"]:

    # bin first period
    obs_degs_corr_and_plot(A_T, x_T, 0, T_1, sub, type_corr, ax[0, 0])

    ax[0, 0].set_title("binary first period")

    # w first period
    obs_degs_corr_and_plot(Y_T, x_T, 0, T_1, sub, type_corr, ax[1, 0])

    ax[1, 0].set_title("weighted first period")


    # bin second period
    obs_degs_corr_and_plot(A_T, x_T, T_1, T_train, sub, type_corr, ax[0, 1])


    ax[0, 1].set_title("binary second period")

    # w second period
    obs_degs_corr_and_plot(Y_T, x_T, T_1, T_train, sub, type_corr, ax[1, 1])

    ax[1, 1].set_title("weighted second period")

plt.suptitle("spearman: observed degrees (str) vs eonia")


ax[0, 0].legend(["in", "out"])

[a.grid() for a in ax.flatten()]

# ax[0, 0].legend(["no reg", "reg = eonia"])

#endregion


#%% Observed vs expected degrees (str) 
# region

phi_T, _, _ = mod_bin_tv_phi.get_time_series_latent_par(only_train=True)
exp_Y_T = mod_bin_tv_phi.exp_Y_T(phi_T, None, None)
exp_degsIO_T = strIO_T_from_tens_T(exp_Y_T).round()
degsIO_T = strIO_T_from_tens_T(A_T)[:, :T_train]

t=28
plt.plot(degsIO_T[:, t-1], exp_degsIO_T[:, t], ".")
i=8
shift = 1
plt.plot(degsIO_T[i, :-shift], exp_degsIO_T[i, shift:], ".")



# all_corr = get_all_corr((degsIO_T[:, T_in:T_fin]), x_T[T_in:T_fin], sub=sub)

# ax.hist(all_corr[type_corr][2:], alpha=0.6, density=False)
plot_dens(all_corr[type_corr][2:], ax=ax)
#endregion

#%% hist of  correlations between eonia and expected degree

# region

# all_mods = [mod_bin_tv_phi, mod_bin_tv_phi_eonia, mod_w_tv_phi, mod_w_tv_phi_eonia]

# [mod.roll_sd_filt_all() for mod in  all_mods]

# type_corr = "spearman"

# T_train = mod_bin_tv_phi.T_train

# id_type = "set_ind_to_zero"
# id_type = "in_sum_eq_out_sum"


# fig, ax = plt.subplots(2, 2, figsize=(24,12))


# def exp_degs_corr_and_plot(mod, x_T, id_type, T_in, T_fin, sub, type_corr, ax):
#     mod.par_vec_id_type = id_type
#     mod.identify_par_seq_T(mod.phi_T)
#     phi_T, _, _ = mod.get_time_series_latent_par(only_train=True)
#     exp_Y_T = mod.exp_Y_T(phi_T, None, None)
#     degsIO_T = strIO_T_from_tens_T(exp_Y_T)
#     all_corr = get_all_corr((degsIO_T[:, T_in:T_fin]), x_T[T_in:T_fin], sub=sub)

#     # ax.hist(all_corr[type_corr][2:], alpha=0.6, density=False)
#     plot_dens(all_corr[type_corr][2:], ax=ax)


# for sub in ["in", "out"]:

#     # bin first period
#     exp_degs_corr_and_plot(mod_bin_tv_phi, x_T, id_type, 0, T_1, sub, type_corr, ax[0, 0])

#     exp_degs_corr_and_plot(mod_bin_tv_phi_eonia, x_T, id_type, 0, T_1, sub, type_corr, ax[0, 0])

#     ax[0, 0].set_title("binary first period")

#     # w first period
#     exp_degs_corr_and_plot(mod_w_tv_phi, x_T, id_type, 0, T_1, sub, type_corr, ax[1, 0])

#     exp_degs_corr_and_plot(mod_w_tv_phi_eonia, x_T, id_type, 0, T_1, sub, type_corr, ax[1, 0])

#     ax[1, 0].set_title("weighted first period")


#     # bin second period
#     exp_degs_corr_and_plot(mod_bin_tv_phi, x_T, id_type, T_1, T_train, sub, type_corr, ax[0, 1])

#     exp_degs_corr_and_plot(mod_bin_tv_phi_eonia, x_T, id_type, T_1, T_train, sub, type_corr, ax[0, 1])

#     ax[0, 1].set_title("binary second period")

#     # w second period
#     exp_degs_corr_and_plot(mod_w_tv_phi, x_T, id_type, T_1, T_train, sub, type_corr, ax[1, 1])

#     exp_degs_corr_and_plot(mod_w_tv_phi_eonia, x_T, id_type, T_1, T_train, sub, type_corr, ax[1, 1])

#     ax[1, 1].set_title("weighted second period")

# plt.suptitle("spearman: expected degrees vs eonia")


# ax[0, 0].legend(["no reg in", "reg = eonia in", "no reg out", "reg = eonia out"])

# [a.grid() for a in ax.flatten()]

# ax[0, 0].legend(["no reg", "reg = eonia"])

#endregion

#%% Correlation of eonia with fit sums in two periods

T_1 = 157

id_type = "in_sum_eq_out_sum"

#region
ind_links = (Y_T[:, :, :T_1] > 0).float().mean(dim=2) > 0.05

corr_w = lambda fit_sum, x_T: spearmanr(fit_sum.sum(dim=0), x_T)[0]

def fit_sum_corr_and_plot(mod, x_T, id_type, T_in, T_fin, ind_links, sub, type_corr, ax):

    mod.par_vec_id_type = id_type
    mod.identify_par_seq_T(mod.phi_T)
    all_fit_sum = get_seq_all_fit_sum(mod, ind_links)
    all_corr = get_all_corr((all_fit_sum[:,  T_in:T_fin]), x_T[ T_in:T_fin], sub=sub)
    
    # ax.hist(all_corr[type_corr][2:], alpha=0.6, density=True)
    plot_dens(all_corr[type_corr][2:], ax=ax)

    return all_corr[type_corr]



fig, ax = plt.subplots(2, 2, figsize=(24, 12))
sub = ""
plt.suptitle(f"for each one of {ind_links.float().sum()} links compute spearman of its fitness sum vs eonia", fontsize=20)

# bin first period
corr_no_reg = fit_sum_corr_and_plot(mod_bin_tv_phi, x_T, id_type, 0, T_1, ind_links, sub, type_corr, ax[0, 0])

corr_reg = fit_sum_corr_and_plot(mod_bin_tv_phi_eonia, x_T, id_type, 0, T_1, ind_links, sub, type_corr, ax[0, 0])

# all_corr = get_all_corr(torch.randn(ind_links.sum(), T_1), x_T[:T_1], sub=sub)
# plot_dens(all_corr[type_corr][2:], ax=ax[0, 0])
ks_p_val = scipy.stats.ks_2samp(corr_no_reg, corr_reg)[1]

ax[0, 0].set_title(f"binary first period - ks test p-val {ks_p_val}")




# w first period
corr_no_reg = fit_sum_corr_and_plot(mod_w_tv_phi, x_T, id_type, 0, T_1, ind_links, sub, type_corr, ax[1, 0])

corr_reg = fit_sum_corr_and_plot(mod_w_tv_phi_eonia, x_T, id_type, 0, T_1, ind_links, sub, type_corr, ax[1, 0])

ks_p_val = scipy.stats.ks_2samp(corr_no_reg, corr_reg)[1]

ax[1, 0].set_title(f"weighted first period - ks test p-val {ks_p_val}")


# all_corr = get_all_corr(torch.randn(ind_links.sum(), T_1), x_T[:T_1], sub=sub)
# plot_dens(all_corr[type_corr][2:], ax=ax[1, 0])



# bin second period
ind_links = (Y_T[:, :, T_1:T_train] > 0).float().mean(dim=2) > 0.01

corr_no_reg = fit_sum_corr_and_plot(mod_bin_tv_phi, x_T, id_type, T_1, T_train, ind_links, sub, type_corr, ax[0, 1])

corr_reg = fit_sum_corr_and_plot(mod_bin_tv_phi_eonia, x_T, id_type, T_1, T_train, ind_links, sub, type_corr, ax[0, 1])


# all_corr = get_all_corr(torch.randn(ind_links.sum(), T_train-T_1), x_T[T_1:T_train], sub=sub)
# plot_dens(all_corr[type_corr][2:], ax=ax[0, 1])
ks_p_val = scipy.stats.ks_2samp(corr_no_reg, corr_reg)[1]
ax[0, 1].set_title(f"binary second period - ks test p-val {ks_p_val}")



# w second period
corr_no_reg = fit_sum_corr_and_plot(mod_w_tv_phi, x_T, id_type, T_1, T_train, ind_links, sub, type_corr, ax[1, 1])

corr_reg = fit_sum_corr_and_plot(mod_w_tv_phi_eonia, x_T, id_type, T_1, T_train, ind_links, sub, type_corr, ax[1, 1])

ks_p_val = scipy.stats.ks_2samp(corr_no_reg, corr_reg)[1]

ax[1, 1].set_title(f"weighted second period - ks test p-val {ks_p_val}")

# ax[1, 1].legend([f"spearman global no reg = {corr_w_no_reg}", f"spearman global eonia  = {corr_w_eonia_reg}"])

ax[0, 0].legend(["no reg", "reg = eonia"])
[a.grid() for a in ax.flatten()]

#endregion


#%% Artificially change the beta coefficient
beta_w = mod_w_tv_phi_eonia.beta_T.copy()
beta_bin = mod_bin_tv_phi_eonia.beta_T.copy()


beta_w[0].shape

mod_w_tv_phi_eonia.beta_T[0] = torch.ones(1,1, requires_grad=True) * (-3)
mod_bin_tv_phi_eonia.beta_T[0] = torch.ones(1,1, requires_grad=True) * (-3)
mod_w_tv_phi_eonia.roll_sd_filt_train()
mod_bin_tv_phi_eonia.roll_sd_filt_train()

#region
ind_links = (Y_T[:, :, :T_1] > 0).float().mean(dim=2) > 0.05

corr_w = lambda fit_sum, x_T: spearmanr(fit_sum.sum(dim=0), x_T)[0]

def fit_sum_corr_and_plot(mod, x_T, id_type, T_in, T_fin, ind_links, sub, type_corr, ax):

    mod.par_vec_id_type = id_type
    mod.identify_par_seq_T(mod.phi_T)
    all_fit_sum = get_seq_all_fit_sum(mod, ind_links)
    all_corr = get_all_corr((all_fit_sum[:,  T_in:T_fin]), x_T[ T_in:T_fin], sub=sub)
    
    # ax.hist(all_corr[type_corr][2:], alpha=0.6, density=True)
    plot_dens(all_corr[type_corr][2:], ax=ax)

    return all_corr[type_corr]



fig, ax = plt.subplots(2, 2, figsize=(24, 12))
sub = ""
plt.suptitle(f"for each one of {ind_links.float().sum()} links compute spearman of its fitness sum vs eonia", fontsize=20)

# bin first period
corr_no_reg = fit_sum_corr_and_plot(mod_bin_tv_phi, x_T, id_type, 0, T_1, ind_links, sub, type_corr, ax[0, 0])

corr_reg = fit_sum_corr_and_plot(mod_bin_tv_phi_eonia, x_T, id_type, 0, T_1, ind_links, sub, type_corr, ax[0, 0])

# all_corr = get_all_corr(torch.randn(ind_links.sum(), T_1), x_T[:T_1], sub=sub)
# plot_dens(all_corr[type_corr][2:], ax=ax[0, 0])
ks_p_val = scipy.stats.ks_2samp(corr_no_reg, corr_reg)[1]

ax[0, 0].set_title(f"binary first period - ks test p-val {ks_p_val}")




# w first period
corr_no_reg = fit_sum_corr_and_plot(mod_w_tv_phi, x_T, id_type, 0, T_1, ind_links, sub, type_corr, ax[1, 0])

corr_reg = fit_sum_corr_and_plot(mod_w_tv_phi_eonia, x_T, id_type, 0, T_1, ind_links, sub, type_corr, ax[1, 0])

ks_p_val = scipy.stats.ks_2samp(corr_no_reg, corr_reg)[1]

ax[1, 0].set_title(f"weighted first period - ks test p-val {ks_p_val}")


# all_corr = get_all_corr(torch.randn(ind_links.sum(), T_1), x_T[:T_1], sub=sub)
# plot_dens(all_corr[type_corr][2:], ax=ax[1, 0])



# bin second period
ind_links = (Y_T[:, :, T_1:T_train] > 0).float().mean(dim=2) > 0.01

corr_no_reg = fit_sum_corr_and_plot(mod_bin_tv_phi, x_T, id_type, T_1, T_train, ind_links, sub, type_corr, ax[0, 1])

corr_reg = fit_sum_corr_and_plot(mod_bin_tv_phi_eonia, x_T, id_type, T_1, T_train, ind_links, sub, type_corr, ax[0, 1])


# all_corr = get_all_corr(torch.randn(ind_links.sum(), T_train-T_1), x_T[T_1:T_train], sub=sub)
# plot_dens(all_corr[type_corr][2:], ax=ax[0, 1])
ks_p_val = scipy.stats.ks_2samp(corr_no_reg, corr_reg)[1]
ax[0, 1].set_title(f"binary second period - ks test p-val {ks_p_val}")



# w second period
corr_no_reg = fit_sum_corr_and_plot(mod_w_tv_phi, x_T, id_type, T_1, T_train, ind_links, sub, type_corr, ax[1, 1])

corr_reg = fit_sum_corr_and_plot(mod_w_tv_phi_eonia, x_T, id_type, T_1, T_train, ind_links, sub, type_corr, ax[1, 1])

ks_p_val = scipy.stats.ks_2samp(corr_no_reg, corr_reg)[1]

ax[1, 1].set_title(f"weighted second period - ks test p-val {ks_p_val}")

# ax[1, 1].legend([f"spearman global no reg = {corr_w_no_reg}", f"spearman global eonia  = {corr_w_eonia_reg}"])

ax[0, 0].legend(["no reg", "reg = eonia"])
[a.grid() for a in ax.flatten()]

#endregion

mod_w_tv_phi_eonia.beta_T[0] = torch.ones(1,1, requires_grad=True) * (3)
mod_bin_tv_phi_eonia.beta_T[0] = torch.ones(1,1, requires_grad=True) * (3)
mod_w_tv_phi_eonia.roll_sd_filt_train()
mod_bin_tv_phi_eonia.roll_sd_filt_train()

#region
ind_links = (Y_T[:, :, :T_1] > 0).float().mean(dim=2) > 0.05

corr_w = lambda fit_sum, x_T: spearmanr(fit_sum.sum(dim=0), x_T)[0]

def fit_sum_corr_and_plot(mod, x_T, id_type, T_in, T_fin, ind_links, sub, type_corr, ax):

    mod.par_vec_id_type = id_type
    mod.identify_par_seq_T(mod.phi_T)
    all_fit_sum = get_seq_all_fit_sum(mod, ind_links)
    all_corr = get_all_corr((all_fit_sum[:,  T_in:T_fin]), x_T[ T_in:T_fin], sub=sub)
    
    # ax.hist(all_corr[type_corr][2:], alpha=0.6, density=True)
    plot_dens(all_corr[type_corr][2:], ax=ax)

    return all_corr[type_corr]



fig, ax = plt.subplots(2, 2, figsize=(24, 12))
sub = ""
plt.suptitle(f"for each one of {ind_links.float().sum()} links compute spearman of its fitness sum vs eonia", fontsize=20)

# bin first period
corr_no_reg = fit_sum_corr_and_plot(mod_bin_tv_phi, x_T, id_type, 0, T_1, ind_links, sub, type_corr, ax[0, 0])

corr_reg = fit_sum_corr_and_plot(mod_bin_tv_phi_eonia, x_T, id_type, 0, T_1, ind_links, sub, type_corr, ax[0, 0])

# all_corr = get_all_corr(torch.randn(ind_links.sum(), T_1), x_T[:T_1], sub=sub)
# plot_dens(all_corr[type_corr][2:], ax=ax[0, 0])
ks_p_val = scipy.stats.ks_2samp(corr_no_reg, corr_reg)[1]

ax[0, 0].set_title(f"binary first period - ks test p-val {ks_p_val}")




# w first period
corr_no_reg = fit_sum_corr_and_plot(mod_w_tv_phi, x_T, id_type, 0, T_1, ind_links, sub, type_corr, ax[1, 0])

corr_reg = fit_sum_corr_and_plot(mod_w_tv_phi_eonia, x_T, id_type, 0, T_1, ind_links, sub, type_corr, ax[1, 0])

ks_p_val = scipy.stats.ks_2samp(corr_no_reg, corr_reg)[1]

ax[1, 0].set_title(f"weighted first period - ks test p-val {ks_p_val}")


# all_corr = get_all_corr(torch.randn(ind_links.sum(), T_1), x_T[:T_1], sub=sub)
# plot_dens(all_corr[type_corr][2:], ax=ax[1, 0])



# bin second period
ind_links = (Y_T[:, :, T_1:T_train] > 0).float().mean(dim=2) > 0.01

corr_no_reg = fit_sum_corr_and_plot(mod_bin_tv_phi, x_T, id_type, T_1, T_train, ind_links, sub, type_corr, ax[0, 1])

corr_reg = fit_sum_corr_and_plot(mod_bin_tv_phi_eonia, x_T, id_type, T_1, T_train, ind_links, sub, type_corr, ax[0, 1])


# all_corr = get_all_corr(torch.randn(ind_links.sum(), T_train-T_1), x_T[T_1:T_train], sub=sub)
# plot_dens(all_corr[type_corr][2:], ax=ax[0, 1])
ks_p_val = scipy.stats.ks_2samp(corr_no_reg, corr_reg)[1]
ax[0, 1].set_title(f"binary second period - ks test p-val {ks_p_val}")



# w second period
corr_no_reg = fit_sum_corr_and_plot(mod_w_tv_phi, x_T, id_type, T_1, T_train, ind_links, sub, type_corr, ax[1, 1])

corr_reg = fit_sum_corr_and_plot(mod_w_tv_phi_eonia, x_T, id_type, T_1, T_train, ind_links, sub, type_corr, ax[1, 1])

ks_p_val = scipy.stats.ks_2samp(corr_no_reg, corr_reg)[1]

ax[1, 1].set_title(f"weighted second period - ks test p-val {ks_p_val}")

# ax[1, 1].legend([f"spearman global no reg = {corr_w_no_reg}", f"spearman global eonia  = {corr_w_eonia_reg}"])

ax[0, 0].legend(["no reg", "reg = eonia"])
[a.grid() for a in ax.flatten()]

#endregion


# %%
# %% plot beta score

s_T_w = mod_w_tv_phi_eonia.get_score_T_train()
s_T_bin = mod_bin_tv_phi_eonia.get_score_T_train()

plt.plot(s_T_w["beta"][0,0,:], ".")
s_T_w["beta"][0,0,:T_1].mean()
s_T_w["beta"][0,0,T_1:].mean()

mod_w_tv_phi_eonia.beta_T

plt.plot(s_T_bin["beta"][0,0,:], ".")
s_T_bin["beta"][0,0,:T_1].mean()
s_T_bin["beta"][0,0,T_1:].mean()

