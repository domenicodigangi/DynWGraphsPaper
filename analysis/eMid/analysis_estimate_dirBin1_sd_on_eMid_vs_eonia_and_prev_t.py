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
import logging
from eMid_data_utils import get_data_from_data_run, load_all_models_emid
importlib.reload(dynwgraphs)
logger = logging.getLogger(__name__)

# %%
# controllare risultati da unc mean. se non buoni, sistemare codice per stimare solo i beta con init a unc mean
# finire funzioni per plotting
# guardare distribuzioni dei parametri beta 2N stimati e confrontarle con strength media
#  finire intervalli di confidenza mle 
#  calcolare intervalli  di confidenza mle per stime emid
#  aggiungere test di calvori 
#  calcolarlo su parametri mle
#  aggiungere info al paper
#  stimare su dati socio patterns con appertenenza allo stesso reparto come regressore esterno per avere un applicazione con un dataset pubblico
#  testare intervalli di confidenza mle


experiment = _get_and_set_experiment("emid runs paper")
df_all_runs = get_df_exp(experiment, one_df=True)

logger.info(f"Staus of experiment {experiment.name}: \n {df_all_runs['status'].value_counts()}")

df_reg = df_all_runs[(df_all_runs["status"].apply(lambda x: x in ["FINISHED"])) & (~np.isnan(df_all_runs["sd_actual_n_opt_iter"]))]


log_cols = ["regressor_name", "size_phi_t", "phi_tv",  "size_beta_t", "beta_tv"]



sel_dic = {"beta_tv": "1", "size_beta_t": "one", "bin_or_w": "bin", "regressor_name": "eonia"}

row_run = pd_filt_on(df_reg, sel_dic).iloc[0, :]

Y_T, X_T, regr_list, net_stats = get_data_from_data_run(float(row_run["unit_meas"]), row_run["regressor_name"])

mod_ss, mod_sd = load_all_models_emid(Y_T, X_T, row_run)


mod_sd.plot_phi_T()
selfo = mod_sd

phi_T, dist_par_un_T, beta_T = selfo.get_time_series_latent_par()

import copy

beta_T.shape
mod_sd.sd_stat_par_un_beta["init_val"]
mod_sd.sd_stat_par_un_beta["w"]
mod_sd.sd_stat_par_un_phi["init_val"]



plot_tv(selfo, beta_T[:, 0, :])

#%%

def plot_cross(selfo, par_N_in, x=None, fig_ax=None):
    if fig_ax is None:
        fig, ax = plt.subplots(2,1)
    else:
        fig, ax = fig_ax
    
    if x is None:
        #histogram
        
    

def plot_tv(selfo, par_T_in, i=None, x=None, fig_ax=None):
    par_T = copy.deepcopy(par_T_in.detach().numpy(), train_l_flag =True)

    n_par = par_T.shape[0]

    if x is None:
        x = np.array(range(par_T.shape[1]))

    # time var par
    if par_T.shape[1] == 1:
        par_T = par_T.repeat_interleave(x.shape[0], dim=1)

    if selfo.inds_to_exclude_from_id is not None:
        par_T[selfo.inds_to_exclude_from_id, :] = float('nan')

    if n_par == 2*selfo.N:
        if i is not None:
            par_i_T = (par_T[:selfo.N,:])[i,:]
            par_o_T = (par_T[selfo.N:,:])[i,:]
        else:

            par_i_T = par_T[:selfo.N, :]
            par_o_T = par_T[selfo.N:, :]

        if fig_ax is None:
            fig, ax = plt.subplots(2,1)
        else:
            fig, ax = fig_ax

        ax[0].plot(x, par_i_T.T)
        ax[1].plot(x, par_o_T.T)

    elif n_par in [1, selfo.N]:
        if i is not None:
            par_i_T = par_T[i,:]
        else:
            par_i_T = par_T

        if fig_ax is None:
            fig, ax = plt.subplots(1,1)
        else:
            fig, ax = fig_ax

        ax[0].plot(x, par_i_T)

    if train_l_flag:
        for a in ax:
            a.vlines(selfo.T_train, a.get_ylim()[0], a.get_ylim()[1], colors = "r", linestyles="dashed")

    return fig, ax

# %%
