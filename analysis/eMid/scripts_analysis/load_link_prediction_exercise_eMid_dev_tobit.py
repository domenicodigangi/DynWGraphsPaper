#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Tuesday November 2nd 2021

"""


# %% import packages
import itertools
from math import sqrt
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
import os
# matplotlib.rcParams['text.usetex'] = True
from proj_utils.mlflow import _get_and_set_experiment,  get_df_exp
from proj_utils import pd_filt_on
import importlib
import proj_utils
from dynwgraphs.link_specific_models import get_ij_giraitis_reg, predict_kernel_tobit, apply_t, predict_ZA_regression
from proj_utils.eMid_data_utils import get_data_from_data_run
importlib.reload(proj_utils)
import logging
logger = logging.getLogger(__name__)
from proj_utils import get_proj_fold
from run_link_prediction_exercise_eMid_dev_tobit import load_obs
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, r2_score
from matplotlib import pyplot as plt
#%% 
def eval_mod(df):
    ind_pos_obs = df["obs"] > 0
    ind_pos_pred = df["pred"] > 0
    ind_comp_w = ind_pos_obs & ind_pos_pred

    df["log_obs"] = np.log(df["obs"])
    df["log_obs"][~ind_pos_obs] = 0
    
    df["log_pred"] = np.log(df["pred"])
    df["log_pred"][~ind_pos_pred] = 0

    ind_comp_bin = ~ df["bin_pred"].isna()

    w_comp = {
        "ker_type": ker_type,
        "bandwidth": bandwidth,
        "mse_all": mean_squared_error(df["log_obs"], df["log_pred"]),
        "mse_pos": mean_squared_error(df["log_obs"][ind_comp_w], df["log_pred"][ind_comp_w]),
        "mse_pos_obs": mean_squared_error(df["log_obs"][ind_pos_obs], df["log_pred"][ind_pos_obs]),
        "mae_all": mean_absolute_error(df["log_obs"], df["log_pred"]),
        "mae_pos": mean_absolute_error(df["log_obs"][ind_comp_w], df["log_pred"][ind_comp_w])
        }

    bin_comp = {
        "auc_all": roc_auc_score(df["obs"][ind_comp_bin] > 0, df["bin_pred"][ind_comp_bin])
    }

    return {**w_comp, **bin_comp}

loadpath = get_proj_fold() / "analysis" / "eMid" / "scripts_analysis" / "local_pred_data"

Y_T = load_obs()
t_oos = 1
T_all = Y_T.shape[2]
T_train = 100
max_links = None

N = Y_T.shape[0]
t_oos = 1
list = []
for T_train in [100, 200]:
    T_max = T_all - T_train - 3
    for pred_method in ["giraitis", "ZA_regression"]:
        if pred_method == "giraitis":
            ker_type = "exp"
            bandwidth = int(sqrt(T_train))
            filepath = loadpath / f"{pred_method}_pred_T_train_{T_train}_band_{bandwidth}_kern_{ker_type}_max_links_per_t_{max_links}_T_max_{T_max}.parquet"
        elif pred_method == "ZA_regression":
            ker_type = np.nan
            bandwidth = np.nan
            filepath = loadpath / f"{pred_method}_pred_T_train_{T_train}_max_links_per_t_{max_links}_T_max_{T_max}.parquet"

        df = pd.read_parquet(filepath)
        eval_dict = {"pred_meth": pred_method, "T_train": T_train, **eval_mod(df)}
        list.append(eval_dict)


df = pd.DataFrame(list)
df_for_paper = df[["pred_meth", "T_train", "bandwidth", "mse_pos", "mae_pos", "auc_all"]]
df_for_paper.append({"pred_meth": "SD-gen-fit", "T_train": 100, "bandwidth": "NaN", "mse_pos": 0.85, "mae_pos": 0.726, "auc_all": 0.896}, ignore_index=True)

df_for_paper.to_latex()


# %%

def density_eval_mod(df, smooth_wind, y_lims=(0.5, 8)):
    SMALL_SIZE = 18
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 28

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    ind_pos_obs = df["obs"] > 0
    ind_pos_pred = df["pred"] > 0
    ind_comp_w = ind_pos_obs & ind_pos_pred

    df["log_obs"] = np.log(df["obs"])
    df["log_obs"][~ind_pos_obs] = 0
    
    df["log_pred"] = np.log(df["pred"])
    df["log_pred"][~ind_pos_pred] = 0

    ind_comp_bin = ~ df["bin_pred"].isna()

    df["se"] = (df["log_obs"] - df["log_pred"])**2
    df["ae"] = np.abs(df["log_obs"] - df["log_pred"])
    df_pos = df[ind_comp_w].sort_values("fract_nnz_train", ascending=False).reset_index()
    

    x_w = df_pos["fract_nnz_train"].rolling(window=smooth_wind).mean()
    
    fig, axs = plt.subplots(2, 1, figsize=(25, 15))
    axs[0].plot(x_w, df_pos["se"].rolling(window=smooth_wind).mean())
    axs[0].set_ylim(*y_lims)
    axs[0].set_ylabel("Log MSE")
    axs[0].grid()
    
    # axs[1].plot(x_w, df_pos["ae"].rolling(window=smooth_wind).mean())
    # axs[1].set_ylabel("Log MAD")
    # axs[1].grid()


    df["bin_obs"] = df["obs"] > 0
    df = df.dropna().sort_values("fract_nnz_train", ascending=False).reset_index()
    df_wind = [df[["fract_nnz_train", "bin_obs", "bin_pred"]].iloc[i:i+smooth_wind, :].values for i in range(df.shape[0]-smooth_wind)]

    x_bin = pd.DataFrame([d["fract_nnz_train"].mean() for d in df_wind])
    auc_rolling = pd.DataFrame([roc_auc_score(d["bin_obs"], d["bin_pred"]) for d in df_wind])
    axs[1].plot(x_bin.values, auc_rolling.values)
    axs[1].set_ylabel("AUC")
    axs[1].set_xlabel("Density Train")
    axs[1].grid()
    axs[1].set_xlabel("Density Train")

    return fig, axs

smooth_wind = 50000
T_train=100
T_max = T_all - T_train - 3
pred_method = "giraitis"
ker_type = "exp"
bandwidth = int(sqrt(T_train))
filepath = loadpath / f"{pred_method}_pred_T_train_{T_train}_band_{bandwidth}_kern_{ker_type}_max_links_per_t_{max_links}_T_max_{T_max}.parquet"
df_gir = pd.read_parquet(filepath)

fig, axs = density_eval_mod(df_gir, smooth_wind, y_lims=(0.5, 5))
fig.suptitle("Local Tobit")


#%%
pred_method = "ZA_regression"
ker_type = np.nan
bandwidth = np.nan
filepath = loadpath / f"{pred_method}_pred_T_train_{T_train}_max_links_per_t_{max_links}_T_max_{T_max}.parquet"
df_za = pd.read_parquet(filepath)

fig, axs = density_eval_mod(df_za, smooth_wind, y_lims=(0.5, 5))
fig.suptitle("Z.A. Regression")



# %%

plt.plot(df_gir.fract_nnz_train, df_za.fract_nnz_train, ".")

df_gir.dropna()
df_za.dropna()


