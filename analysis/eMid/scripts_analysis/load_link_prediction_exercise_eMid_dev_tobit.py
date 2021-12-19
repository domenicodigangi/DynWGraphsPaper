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


#%% 


T_train = 100
max_links = None
T_max = T_all - T_train - 3
bandwidths_list = [int(b) for b in [10*sqrt(T_train), 2*sqrt(T_train), sqrt(T_train)]]
N = Y_T.shape[0]
t_oos = 1
T_all = Y_T.shape[2]
ker_type = "exp"
bandwidth = bandwidths_list[0]


from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, r2_score


os.listdir(savepath)

list = []
for ker_type in ["gauss", "exp"]:
    for bandwidth in [10, 50, 100]:

        df = pd.read_parquet(savepath / f"giraitis_pred_T_train_{T_train}_band_{bandwidth}_kern_{ker_type}_max_links_per_t_{max_links}_T_max_{T_max}.parquet")

        df["log_obs"] = np.log(df["obs"])

        df["log_pred"] = np.log(df["pred"])
        ind_inf = ~np.isfinite(df["log_pred"])
        df["log_pred"][ind_inf] = 0

        list.append({
            "ker_type": ker_type,
            "bandwidth": bandwidth,
            "mse_all": mean_squared_error(df.log_obs, df.log_pred),
            "mse_pos": mean_squared_error(df.log_obs[~ind_inf], df.log_pred[~ind_inf]), 
            "mae_all": mean_absolute_error(df.log_obs, df.log_pred),
            "mae_pos": mean_absolute_error(df.log_obs[~ind_inf], df.log_pred[~ind_inf])
            })


df = pd.DataFrame(list)

roc_auc_score(df["obs"]>0, df["bin_pred"].fillna(0))

# %%
ker_type = "exp"
bandwidth = 100
df = pd.read_csv(savepath / f"giraitis_pred_T_train_{T_train}_band_{bandwidth}_kern_{ker_type}_max_links_per_t_{max_links}_T_max_{T_max}.csv")

df["log_obs"] = np.log(df["obs"])
df["log_pred"] = np.log(df["pred"])
ind_inf = ~np.isfinite(df["log_pred"])
df["log_pred"][ind_inf] = 0

df["se"] = (df["log_obs"] - df["log_pred"])**2
df["ae"] = np.abs(df["log_obs"] - df["log_pred"])
df_pos = df[df["pred"]>0].sort_values("fract_nnz_train", ascending=False).reset_index()

df_pos["fract_nnz_train"].rolling(window=5000).mean().plot()
df_pos["se"].rolling(window=5000).mean().dropna()
df_pos["se"].rolling(window=5000).mean().plot()
df_pos["ae"].rolling(window=5000).mean().plot()
df_pos["ae"].rolling(window=5000).mean().min()


# %%
