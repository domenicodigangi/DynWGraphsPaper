#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday October 13th 2021

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


# To Do : 
# load SD estimates in roll analysis file
# check old oos eval and eventually update
# one step forecast from sd
# diebold and mariano test

#%% load allruns

experiment = _get_and_set_experiment("emid est rolling")
df_all_runs = get_df_exp(experiment, one_df=True)

logger.info(f"Staus of experiment {experiment.name}: \n {df_all_runs['status'].value_counts()}")

df_reg = df_all_runs[(df_all_runs["status"].apply(lambda x: x in ["FINISHED"])) & (~np.isnan(df_all_runs["actual_n_opt_iter"]))]

log_cols = ["regressor_name", "size_phi_t", "phi_tv",  "size_beta_t", "beta_tv"]

bin_or_w = "w"
t_0 = 0
#%% get bin models
# region
sel_dic = {"size_phi_t": "2N", "phi_tv": "1.0", "size_beta_t": "0", "bin_or_w": bin_or_w, "regressor_name": "eonia", "t_0": str(t_0), "filter_ss_or_sd": "ss"}

df_sel = pd_filt_on(df_reg, sel_dic).sort_values("end_time")
if df_sel.shape[0] == 1:
    row_run = df_sel.iloc[0, :]
else:
    row_run = df_sel.iloc[0, :]
    logger.error("more than one run")

Y_T, X_T, regr_list, net_stats = get_data_from_data_run(float(row_run["unit_meas"]), row_run["regressor_name"], int(row_run.t_0))

mod = load_all_models_emid(Y_T, X_T, row_run)

# %%
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR
import pandas as pd

phi_T, dist_par_un_T, beta_T = mod.get_time_series_latent_par()

i = 3
t_init = 0 
T = 100 
steps = 8

phi_fore = np.zeros((phi_T.shape[0], steps))

for i in range(phi_T.shape[0]):
    phi_i_T = phi_T[i, t_init:t_init+100].numpy()
    armod = AutoReg(phi_i_T, 1, old_names=False)

    # df = pd.DataFrame(phi_T.numpy().T)
    # varmod = VAR(df, 1)

    res = armod.fit()

    # res.params[0] + res.params[1]*armod.endog[-1]
    phi_fore[i, :] = res.predict(start=T, end=T+steps-1)

    # res.predict(start=T-5, end=T+11, dynamic=0)
    # plt.plot(phi_i_T)
    # res.plot_diagnostics()

# %%

phi_fore

mod.