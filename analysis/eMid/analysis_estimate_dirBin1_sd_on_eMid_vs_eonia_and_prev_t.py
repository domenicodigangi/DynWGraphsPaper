#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday July 7th 2021

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
from eMid_data_utils import get_data_from_data_run, load_all_models_emid, get_model_from_run_dict_emid
import logging
importlib.reload(dynwgraphs)
logger = logging.getLogger(__name__)

# %%

# guardare alla correlazione tra fit bin e fit w 

#  aggiungere test di calvori
#  calcolarlo su parametri mle


#  stimare su dati socio patterns con appertenenza allo stesso reparto come regressore esterno per avere un applicazione con un dataset pubblico:
# dataset quasi pronto, devo aggiungere le comunità come regressori esterni

#  testare intervalli di confidenza mle


experiment = _get_and_set_experiment("emid est paper last")
# experiment = _get_and_set_experiment("emid est paper last 29th august")
df_all_runs = get_df_exp(experiment, one_df=True)

logger.info(f"Staus of experiment {experiment.name}: \n {df_all_runs['status'].value_counts()}")

df_reg = df_all_runs[(df_all_runs["status"].apply(lambda x: x in ["FINISHED"])) & (~np.isnan(df_all_runs["sd_actual_n_opt_iter"]))]


log_cols = ["regressor_name", "size_phi_t", "phi_tv",  "size_beta_t", "beta_tv"]

sel_dic = {"size_phi_t": "2N", "phi_tv": "1", "size_beta_t": "2N", "beta_tv": "0", "bin_or_w": "w", "regressor_name": "logYtm1", "train_fract": "0.8"}


df_reg[df_reg.regressor_name == "logYtm1"].size_beta_t 
df_reg.phi_tv 
df_reg.size_beta_t
df_reg.beta_tv 
# df_reg.size_phi_t[4]

df_sel = pd_filt_on(df_reg, sel_dic).sort_values("end_time")
if df_sel.shape[0] == 1:
    row_run = df_sel.iloc[0, :]
else:
    row_run = df_sel.iloc[0, :]
    logger.error("more than one run")

Y_T, X_T, regr_list, net_stats = get_data_from_data_run(float(row_run["unit_meas"]), row_run["regressor_name"])

_, mod_sd = load_all_models_emid(Y_T, X_T, row_run)


selfo = mod_sd

row_run["sd_avoid_ovflw_fun_flag"]
row_run["init_sd_type"]

if mod_sd.size_beta_t < 50:
    cov_beta = mod_sd.get_cov_mat_stat_est("beta")
    logger.info(f"{mod_sd.beta_T} +- {cov_beta.sqrt()*1.96}")

#%%

# mod_sd.beta_T[0] = torch.tensor(0.34).unsqueeze(0).unsqueeze(1)

mod_sd.loglike_seq_T()
mod_sd.roll_sd_filt_all()

mod_sd.beta_T
cov_beta.sqrt()*1.96


mod_sd.plot_phi_T()
mod_sd.plot_beta_T()

mod_sd.plot_phi_T(i=96)




plt.plot(mod_sd.get_score_T_train()["phi"][96, :].detach())
mod_sd.roll_sd_filt_all()
mod_sd.get_score_T_train()["phi"]
mod_sd.plot_sd_par()

mod_sd.get_unc_mean(mod_sd.sd_stat_par_un_phi)

mod_sd.set_unc_mean(a, mod_sd.sd_stat_par_un_phi)

mod_sd.phi_T[0]
mod_sd.init_sd_type

mod_sd.plot_sd_par()




phi_T, dist_par_un_T, beta_T = selfo.get_time_series_latent_par()


plt.plot(phi_T[1,:selfo.T_train])
import copy

beta_T.shape
mod_sd.mod_for_init.phi_T

mod_sd.un2re_A_par(mod_sd.sd_stat_par_un_phi["A"])[:mod_sd.N][96]
mod_sd.get_unc_mean(mod_sd.sd_stat_par_un_phi)[:mod_sd.N][96]
mod_sd.sd_stat_par_un_beta["w"]
mod_sd.sd_stat_par_un_phi["init_val"]



#%%
def get_fig_ax_and_par(par, ind_par=None):

    n_par = par.shape[0]

    if par.dim() == 1:
        par = par.unsqueeze(1)

    T = par.shape[1]

    if n_par == 2*selfo.N:
        par_i = par[:selfo.N, :]
        par_o = par[selfo.N:, :]
        par_list = [par_i, par_o]

    elif n_par in [1, n_par]:
        if ind_par is not None:
            par_list = [par]

    if ind_par is not None:
        for ind, p in enumerate(par_list):
            par_list[ind] = p[ind_par, :]

    fig_ax = plt.subplots(len(par_list), 1)

    return fig_ax, par_list


def plot_tv(selfo, par_T_in, ind_par=None, x=None, fig_ax=None, train_l_flag=True, T_plot=None):

    par_T = copy.deepcopy(par_T_in)
    
    if T_plot is None:
        T_plot = selfo.T_train 
    
    # time var par
    if par_T.shape[1] == 1:
        par_T = par_T.repeat_interleave(selfo.T_train, dim=1)

    if selfo.inds_never_obs_w is not None:
        par_T[selfo.inds_never_obs_w, :] = float('nan')

    if x is None:
        x = np.array(range(par_T.shape[1]))

    fig_ax, par_list = get_fig_ax_and_par(par_T, ind_par)

    fig, ax = fig_ax
    for i, par in enumerate(par_list):
        ax[i].plot(x, par.detach().numpy().T)

    if train_l_flag:
        for a in ax:
            a.vlines(selfo.T_train, a.get_ylim()[0], a.get_ylim()[1], colors="r", linestyles="dashed")

    return fig, ax

def plot_hist(selfo, par_N_in, x=None, fig_ax=None):
    par_N = par_N_in.detach()
    # time var par
    if par_N.dim() > 1:
        raise

    if selfo.inds_never_obs_w is not None:
        par_N[selfo.inds_never_obs_w] = float('nan')

    fig_ax, par_list = get_fig_ax_and_par(par_N)

    fig, ax = fig_ax
    for i, par in enumerate(par_list):
        if x is None:
            ax[i].hist(par.detach().numpy())
        else:
            ax[i].scatter(x[i], par.detach().T.numpy())
            ax[i].set_xscale('log')
            ax[i].set_yscale('log')

    return fig, ax

def corr_finite(x, y, inds_in=None):
    inds = (np.isfinite(x)) & (np.isfinite(y))
    if inds_in is not None:
        inds = inds & inds_in
    return np.corrcoef(x[inds], y[inds])




#%%
beta_i, beta_o = map(lambda x: torch.nan_to_num(x.detach(), 0), splitVec(selfo.beta_T[0][:, 0]))



plt.loglog(net_stats.avg_degs_i, beta_i, ".")
plt.loglog(net_stats.avg_degs_o, beta_o, ".")
# plt.ylim(0.1, 10)
plt.title(f"Average Node Degree vs beta {row_run['regressor_name']}")

plt.loglog(net_stats.avg_str_i, beta_i, ".")
plt.loglog(net_stats.avg_str_o, beta_o, ".")
# plt.ylim(0.1, 10)
plt.title(f"Average Node Strength vs beta {row_run['regressor_name']}")

# %%
selfo = mod_sd
selfo.plot_phi_T()
phi_T, dist_par_un_T, beta_T = selfo.get_time_series_latent_par()
plot_tv(selfo, selfo.get_time_series_latent_par()[2])

plt.plot(beta_T.squeeze())
selfo.beta_T

selfo.beta_T[0].shape
selfo.get_time_series_latent_par()


#%%




np.corrcoef(np.log(net_stats.avg_str_i), beta_i)

corr_finite(net_stats.avg_degs_i.numpy(), np.exp(beta_i))
corr_finite(np.log(net_stats.avg_str_i.numpy()), np.log(beta_i))
corr_finite(np.log(net_stats.avg_str_o.numpy()), np.log(beta_o))
corr_finite(net_stats.avg_str_o.numpy(), beta_o)
corr_finite(net_stats.avg_degs_o.numpy(), beta_o)
corr_finite(net_stats.avg_degs_i.numpy(), np.exp(beta_i), inds_in=beta_i>0)



beta_mat = mod_sd.get_phi_sum(mod_sd.beta_T[0][:,0]).detach()
beta_mat_obs = beta_mat.clone()
beta_mat_obs[Y_T.sum(dim=2) == 0] = 0

beta_mat_obs[beta_mat_obs!=0]
from dynwgraphs.utils.tensortools import strIO_from_mat

fix, ax = plt.subplots(2, 1)
ax[0].hist(beta_i.unsqueeze(0))
ax[1].hist(beta_o.unsqueeze(0))


plt.loglog(net_stats.avg_str_i, beta_i, ".")
plt.loglog(net_stats.avg_str_i, -beta_i, ".")
plt.loglog(net_stats.avg_str_o, beta_o, ".")
plt.loglog(net_stats.avg_str_o, -beta_o, ".")

plt.semilogx(net_stats.avg_degs_i, beta_i, ".")
plt.semilogx(net_stats.avg_degs_o, beta_o, ".")
plt.ylim(-1, 10)


plt.loglog(net_stats.avg_degs_i, -beta_i, ".")
plt.loglog(net_stats.avg_degs_o, -beta_o, ".")


