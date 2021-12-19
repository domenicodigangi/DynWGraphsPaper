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

#%% load allruns
current_folder = get_proj_fold() / "analysis" / "eMid"
os.chdir(current_folder) 
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

Y_T, X_T, regr_list, net_stats = get_data_from_data_run(float(row_run["unit_meas"]), row_run["regressor_name"], T_0 = int(row_run.t_0), parent_mlruns_folder= Path(current_folder / "mlruns") )

savepath = get_proj_fold() / "analysis" / "eMid" / "scripts_analysis" / "local_pred_data"
n_jobs = 24

#%% RunEstimates ZA regression
def main_ZA_regression(T_train_list):
    N = Y_T.shape[0]
    t_oos = 1
    T_all = Y_T.shape[2]
    T_train = 100

    prediction_method = "ZA_regression"
    pred_fun = predict_ZA_regression
    max_links = None
    run_estimates = True
    if run_estimates:
        for T_train in [100, 200]:
            T_max = T_all - T_train - 3

            par_res = Parallel(n_jobs=n_jobs)(delayed(apply_t)(t_0, Y_T, max_links, T_train, t_oos, pred_fun, ker_type=None, bandwidth=None) for t_0 in tqdm(range(1, T_max)))

            t_0_all = np.concatenate(np.array([p[0] for p in par_res]))
            fract_nnz_all = np.concatenate(np.array([p[1] for p in par_res]))
            obs_all = np.concatenate(np.array([p[2] for p in par_res]))
            pred_all = np.concatenate(np.array([p[3] for p in par_res]))
            bin_pred_all = np.concatenate(np.array([p[4] for p in par_res]))
            df_res = pd.DataFrame( np.stack((t_0_all, fract_nnz_all, obs_all, pred_all, bin_pred_all)).T, columns=["t_0_all", "fract_nnz_train", "obs", "pred", "bin_pred"])

            df_res.to_parquet(savepath / f"{prediction_method}_pred_T_train_{T_train}_max_links_per_t_{max_links}_T_max_{T_max}.parquet")

#%% RunEstimates Giraitis

def main_giraitis(T_train_list):
    N = Y_T.shape[0]
    t_oos = 1
    T_all = Y_T.shape[2]
    prediction_method = "giraitis"
    pred_fun = predict_kernel_tobit
    max_links = None
    run_estimates = True
    T_train = 100

    ker_type = "exp"
    if run_estimates:
        for T_train in T_train_list:
            T_max = T_all - T_train - 3
            bandwidths_list = [int(b) for b in [10*sqrt(T_train), sqrt(T_train)]]
            for bandwidth in bandwidths_list:
                par_res = Parallel(n_jobs=n_jobs)(delayed(apply_t)(t_0, Y_T, max_links, T_train, t_oos, pred_fun, ker_type=ker_type, bandwidth=bandwidth) for t_0 in tqdm(range(1, T_max)))

                t_0_all = np.concatenate(np.array([p[0] for p in par_res]))
                fract_nnz_all = np.concatenate(np.array([p[1] for p in par_res]))
                obs_all = np.concatenate(np.array([p[2] for p in par_res]))
                pred_all = np.concatenate(np.array([p[3] for p in par_res]))
                bin_pred_all = np.concatenate(np.array([p[4] for p in par_res]))
                df_res = pd.DataFrame( np.stack((t_0_all, fract_nnz_all, obs_all, pred_all, bin_pred_all)).T, columns=["t_0_all", "fract_nnz_train", "obs", "pred", "bin_pred"])

                df_res.to_parquet(savepath / f"{prediction_method}_pred_T_train_{T_train}_band_{bandwidth}_kern_{ker_type}_max_links_per_t_{max_links}_T_max_{T_max}.parquet")


if __name__ == "__main__":
    T_train_list = [200]
    main_giraitis(T_train_list)
