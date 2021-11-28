#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Tuesday November 2nd 2021

"""


# %% import packages
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
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
current_path = os.getcwd()

def get_metric(res, method, metr):
    return np.array([r[metr] for r in res[method]])


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

# %%
T_train = 100
N = Y_T.shape[0]
t_oos = 1

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
    pred[pred<0] = 0
    fract_nnz = (y_train.values > 0).mean()
    return {"obs": y_T.iloc[T_train:T_train+t_oos].values, "pred": pred, "fract_nnz": fract_nnz}


def get_obs_and_pred_giraitis_whole_mat_nnz(Y_T, T_train, t_oos,ker_type="gauss", bandwidth=20, max_links = None):
    if t_oos != 1:
        raise "to handle multi step ahead need to fix the check on non zero obs and maybe other parts"
    Y_T_sum = Y_T.sum(axis=2)
    obs_vec = np.zeros(0)
    pred_vec = np.zeros(0)
    fract_nnz_vec = np.zeros(0)
    counter = 0
    for i, j in itertools.product(range(N), range(N)):
        if i != j:
            if Y_T_sum[i, j] != 0:
                x_T, y_T = get_ij_giraitis_reg(i, j, Y_T, T_train, t_oos)
                if Y_T[i, j, T_train+t_oos] != 0:
                    counter += 1
                    logger.info(f" running {i,j}")
                    res = predict_kernel_tobit(x_T, y_T, T_train, t_oos, ker_type=ker_type, bandwidth=bandwidth)
                
                    obs_vec = np.append(obs_vec, res["obs"])        
                    pred_vec = np.append(pred_vec, res["pred"]) 
                    fract_nnz_vec = np.append(fract_nnz_vec, res["fract_nnz"]) 

                    if max_links is not None:
                        if counter > max_links:
                            return fract_nnz_vec, obs_vec, pred_vec
                             

    return obs_vec, pred_vec


def apply_t(t_0, Y_T, max_links, ker_type="gauss", bandwidth=20):
    logger.info(f"eval forecast {t_0}")
    Y_T_train_oos = Y_T[:, :, t_0:t_0+T_train+t_oos+1]
    fract_nnz_vec, obs_vec, pred_vec = get_obs_and_pred_giraitis_whole_mat_nnz(Y_T_train_oos, T_train, t_oos, max_links=max_links, ker_type=ker_type, bandwidth=bandwidth)
    return fract_nnz_vec, obs_vec, pred_vec

#%% 
# t_0 = 2
# i, j = 1, 13
# apply_t(t_0, Y_T)
T_max = 191
max_links = 3
bandwidth = 50
ker_type = "gauss"

if __name__ == "__main__":
    par_res = Parallel(n_jobs=14)(delayed(apply_t)(t_0, Y_T, max_links, ker_type=ker_type, bandwidth=bandwidth) for t_0 in tqdm(range(1, T_max)))

fract_nnz_all = np.concatenate(np.array([p[0] for p in par_res]))
obs_all = np.concatenate(np.array([p[1] for p in par_res]))
pred_all = np.concatenate(np.array([p[2] for p in par_res]))
df_res = pd.DataFrame( np.stack((fract_nnz_all, obs_all, pred_all)).T, columns=["fract_nnz_train", "obs", "pred"])
df_res

#%% 
# import copy
# par_res_store = copy.deepcopy(par_res)


len(par_res)

np.concatenate(obs_all).shape[0]

plt.loglog(par_res[0][0], par_res[0][1], ".")
# %%

# TO Do: 

# Controllare che per un link denso le predizioni abbiano senso per diversi tempi 
# eventualmente controllare allineamento dei tempi e provare con vari kernels
# aggiungere save file dei risultati
# lanciare su tutta la rete