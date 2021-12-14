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
import matplotlib.pyplot as plt
from scipy.stats import norm
# matplotlib.rcParams['text.usetex'] = True
import dynwgraphs
from dynwgraphs.utils.tensortools import strIO_from_tens_T, strIO_T_from_tens_T, tens, splitVec
from dynwgraphs.dirGraphs1_dynNets import dirBin1_SD, dirSpW1_SD
from dynwgraphs.tobit.tobit  import TobitModel
from ddg_utils.mlflow import _get_and_set_experiment, uri_to_path, _get_or_run, get_df_exp
from ddg_utils import pd_filt_on
from mlflow.tracking.client import MlflowClient
import importlib
import ddg_utils

from ddg_utils.eMid_data_utils import get_data_from_data_run, load_all_models_emid, get_model_from_run_dict_emid
importlib.reload(ddg_utils)
import logging
importlib.reload(dynwgraphs)
logger = logging.getLogger(__name__)

def get_metric(res, method, metr):
    return np.array([r[metr] for r in res[method]])

from ddg_utils import get_proj_fold

#%% load allruns

os.chdir(get_proj_fold() / "analysis" / "eMid" ) 
experiment = _get_and_set_experiment("emid est rolling")
df_all_runs = get_df_exp(experiment, one_df=True)

logger.info(f"Staus of experiment {experiment.name}: \n {df_all_runs['status'].value_counts()}")

df_reg = df_all_runs[(df_all_runs["status"].apply(lambda x: x in ["FINISHED"])) & (~np.isnan(df_all_runs["actual_n_opt_iter"]))]

log_cols = ["regressor_name", "size_phi_t", "phi_tv",  "size_beta_t", "beta_tv"]

bin_or_w = "w"

sel_dic = {"size_phi_t": "2N", "phi_tv": "1.0", "size_beta_t": "0", "bin_or_w": bin_or_w, "regressor_name": "eonia", "t_0": str(0), "filter_ss_or_sd": "ss"}

df_sel = pd_filt_on(df_reg, sel_dic).sort_values("end_time")
if df_sel.shape[0] == 1:
    row_run = df_sel.iloc[0, :]
else:
    row_run = df_sel.iloc[0, :]
    logger.error("more than one run")

Y_T, X_T, regr_list, net_stats = get_data_from_data_run(float(row_run["unit_meas"]), row_run["regressor_name"], int(row_run.t_0))

# %% Useful functions to repeat the rolling estimation and forecast
i, j = 1, 13
# get all regressors used by giraitis
def get_ij_giraitis_reg(i, j, Y_T, T_train, t_oos):
    y_ij = Y_T[i, j, 1:T_train+t_oos+1].numpy()
    y_ij_tm1 = Y_T[i, j, :T_train+t_oos].numpy()
    y_ji_tm1 = Y_T[j, i, :T_train+t_oos].numpy()
    y_i_sum_in_tm1 = Y_T[i, :, :T_train+t_oos].sum(0) - y_ij_tm1
    y_i_sum_out_tm1 = Y_T[:, i, :T_train+t_oos].sum(0) - y_ji_tm1
    y_j_sum_in_tm1 = Y_T[j, :, :T_train+t_oos].sum(0) - y_ji_tm1
    y_j_sum_out_tm1 = Y_T[:, j, :T_train+t_oos].sum(0) - y_ij_tm1
    y_sum_not_ij_tm1 = Y_T[:, :, :T_train+t_oos].sum((0,1)) - y_ij_tm1 - y_ji_tm1

    x_T = pd.DataFrame({"y_ij_tm1": y_ij_tm1, "y_i_sum_in_tm1": y_i_sum_in_tm1, "y_j_sum_in_tm1": y_j_sum_in_tm1, "y_i_sum_out_tm1": y_i_sum_out_tm1, "y_j_sum_out_tm1": y_j_sum_out_tm1, "y_sum_not_ij_tm1": y_sum_not_ij_tm1})

    y_T = pd.Series(y_ij)

   
    return x_T, y_T

def predict_kernel_tobit(x_T, y_T, T_train, t_oos, ker_type="gauss", bandwidth=20):

    x_train, y_train = x_T[:T_train], y_T[:T_train]
    tr = TobitModel()
    tr.fit(x_train, y_train, type=ker_type, bandwidth=bandwidth)
    pred = tr.predict(x_T.iloc[T_train:T_train+t_oos, : ].values)
    sigma = tr.sigma_
    pred_prob_zero = norm.cdf(0, loc=pred, scale=sigma)
    bin_pred = 1-pred_prob_zero
    pred[pred<0] = 0
    fract_nnz = (y_train.values > 0).mean()
    
    # assert bin_pred.shape == pred.shape
    return {"obs": y_T.iloc[T_train:T_train+t_oos].values, "pred": pred, "bin_pred": bin_pred, "fract_nnz": fract_nnz}

def get_obs_and_pred_giraitis_whole_mat_nnz(Y_T, T_train, t_oos,ker_type="gauss", bandwidth=20, max_links=None, include_zero_obs=True):
    if t_oos != 1:
        raise "to handle multi step ahead need to fix the check on non zero obs and maybe other parts"
    Y_T_sum = Y_T.sum(axis=2)
    obs_vec = np.zeros(0)
    pred_vec = np.zeros(0)
    bin_pred_vec = np.zeros(0)
    fract_nnz_vec = np.zeros(0)
    counter = 0
    for i, j in itertools.product(range(N), range(N)):
        if i != j:
            if Y_T_sum[i, j] != 0:
                x_T, y_T = get_ij_giraitis_reg(i, j, Y_T, T_train, t_oos)
                if (Y_T[i, j, T_train+t_oos] != 0) | include_zero_obs:
                    counter += 1
                    logger.info(f" running {i,j}")
                    res = predict_kernel_tobit(x_T, y_T, T_train, t_oos, ker_type=ker_type, bandwidth=bandwidth)
                
                    obs_vec = np.append(obs_vec, res["obs"])        
                    pred_vec = np.append(pred_vec, res["pred"]) 
                    bin_pred_vec = np.append(bin_pred_vec, res["bin_pred"]) 
                    fract_nnz_vec = np.append(fract_nnz_vec, res["fract_nnz"]) 

                    # assert bin_pred_vec.shape == pred_vec.shape
                    if max_links is not None:
                        if counter > max_links:
                            return fract_nnz_vec, obs_vec, pred_vec, bin_pred_vec

    return fract_nnz_vec, obs_vec, pred_vec, bin_pred_vec

def apply_t(t_0, Y_T, max_links, T_train, ker_type="gauss", bandwidth=20):
    logger.info(f"eval forecast {t_0}")
    Y_T_train_oos = Y_T[:, :, t_0:t_0+T_train+t_oos+1]
    fract_nnz_vec, obs_vec, pred_vec, bin_pred_vec = get_obs_and_pred_giraitis_whole_mat_nnz(Y_T_train_oos, T_train, t_oos, max_links=max_links, ker_type=ker_type, bandwidth=bandwidth)
    t0_vec = np.full(fract_nnz_vec.shape[0], t_0)
    
    return t0_vec, fract_nnz_vec, obs_vec, pred_vec, bin_pred_vec

#%% 
# t_0 = 2
# i, j = 1, 13
# apply_t(t_0, Y_T)
savepath = Path("D:/pCloud/Dynamic_Networks/repos/DynWGraphsPaper/analysis/eMid/scripts_analysis/local_pred_data")

N = Y_T.shape[0]
t_oos = 1
T_all = Y_T.shape[2]

max_links = None
run_estimates = True

T_train = 100
ker_type = "exp"
if run_estimates:
    for T_train in [100, 150, 200]:
        T_max = T_all - T_train - 3
        bandwidths_list = [int(b) for b in [10*sqrt(T_train), 2*sqrt(T_train), sqrt(T_train)]]
        for bandwidth in bandwidths_list:
            par_res = Parallel(n_jobs=12)(delayed(apply_t)(t_0, Y_T, max_links, T_train, ker_type=ker_type, bandwidth=bandwidth) for t_0 in tqdm(range(1, T_max)))

            t_0_all = np.concatenate(np.array([p[0] for p in par_res]))
            fract_nnz_all = np.concatenate(np.array([p[1] for p in par_res]))
            obs_all = np.concatenate(np.array([p[2] for p in par_res]))
            pred_all = np.concatenate(np.array([p[3] for p in par_res]))
            bin_pred_all = np.concatenate(np.array([p[4] for p in par_res]))
            df_res = pd.DataFrame( np.stack((t_0_all, fract_nnz_all, obs_all, pred_all, bin_pred_all)).T, columns=["t_0_all", "fract_nnz_train", "obs", "pred", "bin_pred"])

            df_res.to_parquet(savepath / f"giraitis_pred_T_train_{T_train}_band_{bandwidth}_kern_{ker_type}_max_links_per_t_{max_links}_T_max_{T_max}.parquet")


#%% 

savepath = Path("D:/pCloud/Dynamic_Networks/repos/DynWGraphsPaper/analysis/eMid/scripts_analysis/local_tobit_pred_data")

T_max = 191
max_links = None


from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, r2_score


ker_type = "exp"
bandwidth = 50
list = []
for ker_type in ["gauss", "exp"]:
    for bandwidth in [10, 50, 100]:

        df = pd.read_csv(savepath / f"giraitis_pred_T_train_{T_train}_band_{bandwidth}_kern_{ker_type}_max_links_per_t_{max_links}_T_max_{T_max}.csv")

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
