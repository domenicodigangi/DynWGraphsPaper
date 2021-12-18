#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday October 13th 2021

"""


# %% import packages
import numpy as np
import os
import matplotlib.pyplot as plt
# matplotlib.rcParams['text.usetex'] = True
import dynwgraphs
from dynwgraphs.utils.tensortools import strIO_from_tens_T, strIO_T_from_tens_T, tens, splitVec
from dynwgraphs.dirGraphs1_dynNets import dirBin1_SD, dirSpW1_SD
from proj_utils.mlflow import _get_and_set_experiment, uri_to_path, _get_or_run, get_df_exp
from proj_utils import pd_filt_on
from mlflow.tracking.client import MlflowClient
import importlib
from scipy.stats import zscore, spearmanr
from scipy.stats import gaussian_kde
from torch.nn.functional import normalize
import sys
import proj_utils
from proj_utils.eMid_data_utils import get_data_from_data_run, load_all_models_emid, get_model_from_run_dict_emid
importlib.reload(proj_utils)
import logging
importlib.reload(dynwgraphs)
logger = logging.getLogger(__name__)
current_path = os.getcwd()

def get_metric(res, method, metr):
    return np.array([r[metr] for r in res[method]])

# To Do : 
# multi step forecast from sd

#%% load allruns

os.chdir(f"{current_path}/..")
experiment = _get_and_set_experiment("emid est rolling")
df_all_runs = get_df_exp(experiment, one_df=True)

logger.info(f"Staus of experiment {experiment.name}: \n {df_all_runs['status'].value_counts()}")

df_reg = df_all_runs[(df_all_runs["status"].apply(lambda x: x in ["FINISHED"])) & (~np.isnan(df_all_runs["actual_n_opt_iter"]))]

log_cols = ["regressor_name", "size_phi_t", "phi_tv",  "size_beta_t", "beta_tv"]

bin_or_w = "w"
#%% 
sel_dic = {"size_phi_t": "2N", "phi_tv": "1.0", "size_beta_t": "0", "bin_or_w": bin_or_w, "regressor_name": "eonia", "t_0": str(0), "filter_ss_or_sd": "ss"}

df_sel = pd_filt_on(df_reg, sel_dic).sort_values("end_time")
if df_sel.shape[0] == 1:
    row_run = df_sel.iloc[0, :]
else:
    row_run = df_sel.iloc[0, :]
    logger.error("more than one run")

Y_T, X_T, regr_list, net_stats = get_data_from_data_run(float(row_run["unit_meas"]), row_run["regressor_name"], int(row_run.t_0))

mod_ss = load_all_models_emid(Y_T, X_T, row_run)
# mod.plot_phi_T()



# %%
mod_ss.T_train = 100

res = {"ss_flat": [], "ss_AR": [], "sd": []}

obs_pred = {"ss_flat": np.zeros(0), "ss_AR": np.zeros(0), "sd": np.zeros(0), "obs_Y_ss_flat": np.zeros(0), "obs_Y_ss_AR": np.zeros(0), "obs_Y_sd": np.zeros(0), "pred_Y_ss_flat": np.zeros(0), "pred_Y_ss_AR": np.zeros(0), "pred_Y_sd": np.zeros(0)}


t_0 = 2

for t_0 in range(1, 190):
    logger.info(f"eval forecast {t_0}")

    sel_dic = {"size_phi_t": "2N", "phi_tv": "1.0", "size_beta_t": "0", "bin_or_w": bin_or_w, "regressor_name": "eonia", "t_0": str(t_0), "filter_ss_or_sd": "sd"}

    df_sel = pd_filt_on(df_reg, sel_dic).sort_values("end_time")
    if df_sel.shape[0] == 1:
        row_run = df_sel.iloc[0, :]
    else:
        row_run = df_sel.iloc[0, :]
        logger.error("more than one run")

    Y_T, X_T, regr_list, net_stats = get_data_from_data_run(float(row_run["unit_meas"]), row_run["regressor_name"], int(row_run.t_0))

    mod_sd = load_all_models_emid(Y_T, X_T, row_run)

    T_train = mod_sd.T_train

    mod_ss.forecast_type="flat"

    t_start = t_0 + T_train+1
    t_end = t_0 + T_train+1

    res_ss_flat, Y_vec, F_Y_vec = mod_ss.out_of_sample_eval(t_start=t_start, t_end=t_end, steps_ahead=1)
    obs_pred["obs_Y_ss_flat"] = np.append(obs_pred["obs_Y_ss_flat"], Y_vec)
    obs_pred["pred_Y_ss_flat"] = np.append(obs_pred["pred_Y_ss_flat"], F_Y_vec)
    logger.info(f"SS flat : {res_ss_flat}")    
    

    mod_ss.forecast_type="AR_1"
    mod_ss.T_train_AR = 100
    res_ss_AR, Y_vec, F_Y_vec = mod_ss.out_of_sample_eval(t_start=t_start, t_end=t_end, steps_ahead=1)
    obs_pred["obs_Y_ss_AR"] = np.append(obs_pred["obs_Y_ss_AR"], Y_vec)
    obs_pred["pred_Y_ss_AR"] = np.append(obs_pred["pred_Y_ss_AR"], F_Y_vec)
    logger.info(f"SS-AR :  {res_ss_AR}")    
    
    res_sd, Y_vec, F_Y_vec = mod_sd.out_of_sample_eval(t_start=t_start-t_0, t_end=t_end-t_0, steps_ahead=1)
    obs_pred["obs_Y_sd"] = np.append(obs_pred["obs_Y_sd"], Y_vec)
    obs_pred["pred_Y_sd"] = np.append(obs_pred["pred_Y_sd"], F_Y_vec)

    logger.info(f"SD:  {res_sd}")    

    res["sd"].append(res_sd)
    res["ss_flat"].append(res_ss_flat)
    res["ss_AR"].append(res_ss_AR)

# %%
from proj_utils.dm_test import dm_test
from scipy.stats.mstats import winsorize


metr = "n_links"
metr = "mse_log"
np.mean(winsorize(get_metric(res, "ss_flat", metr), limits=[0.05, 0.05]))
np.mean(winsorize(get_metric(res, "ss_AR", metr), limits=[0.05, 0.05]))
np.mean(winsorize(get_metric(res, "sd", metr), limits=[0.05, 0.05]))
np.mean(get_metric(res, "ss_AR", metr))
np.mean(get_metric(res, "sd", metr))

plt.plot(winsorize(get_metric(res, "ss_flat", metr), limits=[0.05, 0.05]))
plt.plot(winsorize(get_metric(res, "ss_AR", metr), limits=[0.05, 0.05]))
plt.plot(winsorize(get_metric(res, "sd", metr), limits=[0.05, 0.05]))




o = obs_pred["obs_Y_sd"]
f1 = winsorize(obs_pred["pred_Y_sd"], limits=[0.01, 0.01])
f2 = winsorize(obs_pred["pred_Y_ss_AR"], limits=[0.01, 0.01])
f3 = winsorize(obs_pred["pred_Y_ss_flat"], limits=[0.01, 0.01])

plt.plot(np.log(o), np.log(f1), ".")
plt.plot(np.log(o), np.log(f2), ".")
plt.plot(np.log(f1), np.log(f2), ".")

#%%
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, r2_score

mean_squared_error(np.log(o), np.log(f1))
mean_squared_error(np.log(o), np.log(f2))
mean_squared_error(np.log(o), np.log(f3))
dm_test(np.log(o), np.log(f1), np.log(f2), crit = "MSE", h=1)

mean_absolute_error(np.log(o), np.log(f1))
mean_absolute_error(np.log(o), np.log(f2))
mean_absolute_error(np.log(o), np.log(f3))
dm_test(np.log(o), np.log(f1), np.log(f2), crit = "MAD", h=1)


r2_score(np.log(o), np.log(f1))
r2_score(np.log(o), np.log(f2))
r2_score(np.log(o), np.log(f3))
