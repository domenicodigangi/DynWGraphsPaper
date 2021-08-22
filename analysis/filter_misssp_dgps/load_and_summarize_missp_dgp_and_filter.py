#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Thursday August 19th 2021

"""

# %% import packages
from pathlib import Path
import importlib
import dynwgraphs
from dynwgraphs.utils.tensortools import splitVec, strIO_from_tens_T
from dynwgraphs.dirGraphs1_dynNets import dirBin1_sequence_ss, dirBin1_SD, dirSpW1_SD, dirSpW1_sequence_ss
from dynwgraphs.utils.dgps import get_dgp_mod_and_par
import mlflow
import logging
import ddg_utils
from ddg_utils.mlflow import _get_and_set_experiment, check_test_exp, get_df_exp, uri_to_path
from ddg_utils import drop_keys, pd_filt_on
from mlflow.tracking.client import MlflowClient
import pandas as pd
from run_sim_missp_dgp_and_filter_ss_sd import get_filt_mod
import torch
import numpy as np
from matplotlib import pyplot as plt
import click
from scipy.stats import trim_mean
logger = logging.getLogger(__name__)
importlib.reload(dynwgraphs)
importlib.reload(ddg_utils)


# %%

# @click.option("--experiment_name", type=str, default="filter missp dgp")

# def load_avg_save(**kwargs):
#     pass

experiment = _get_and_set_experiment("sim missp filter")

dfs = get_df_exp(experiment)

logger.info(f"Staus of experiment {experiment.name}: \n {dfs['info']['status'].value_counts()}")

ind_fin = ( dfs["info"]["status"] == "FINISHED") & (~dfs["metrics"]["filt_bin_sd_actual_n_opt_iter"].isna())
df_i = dfs["info"][ind_fin]
df_p = dfs["par"][ind_fin]
df_m = dfs["metrics"][ind_fin]

group_cols = ["beta_dgp_set_bin", "beta_filt_set_bin",  "type_tv_ext_reg_dgp_set", "phi_dgp_set_type_tv_bin", "beta_dgp_set_type_tv_bin", "phi_dgp_set_bin", "phi_filt_set_bin"]

# check that each group contains elements with the same settings for non grouping columns
df = df_p.drop(columns = ['run_id', 'n_sim', 'n_jobs']).groupby(group_cols).nunique() > 1

df.loc[:,df.any()]
df_p["filt_bin_sd_optimizer"].value_counts()

if not df.values.any():
    pass
else:
    logger.error(df.loc[:,df.any()])

df = df_i.merge(df_p, on="run_id").merge(df_m, on="run_id")

mse_cols = lambda ss_or_sd: [f"bin_mse_phi_{ss_or_sd}", f"w_mse_phi_{ss_or_sd}", f"bin_mse_beta_{ss_or_sd}", f"w_mse_beta_{ss_or_sd}"]

beta_cols = lambda ss_or_sd: [f"bin_avg_beta_{ss_or_sd}", f"w_avg_beta_{ss_or_sd}"]

#%%
if False:
    all_cols = mse_cols("sd") + mse_cols("ss")
    df[all_cols] = df[all_cols].clip(lower=df.quantile(0.05), upper=df.quantile(0.95), axis=1)


df.groupby(group_cols).mean()[mse_cols("sd")]
df.groupby(group_cols).count()[mse_cols("sd")]
[d[["filt_bin_sd_optimizer"]].value_counts() for k, d in df.groupby(group_cols)]
df.groupby(group_cols).mean()[mse_cols("ss")]
df.groupby(group_cols).mean()[mse_cols]

[d[mse_cols("sd")].hist() for k, d in df.groupby(group_cols)]
[d[beta_cols("sd")].hist() for k, d in df.groupby(group_cols)]
[d[beta_cols("sd")].hist() for k, d in df.groupby(group_cols)]
[print((k, d[avg_cols ])) for k, d in df.groupby(group_cols)]


#%% 
row_run = df.loc[df["run_id"] == "7bfd1c1f68bf432697704f7f206cfdd7"].iloc[0]

load_path = Path(uri_to_path(row_run["artifact_uri"]))
bin_or_w = "bin"
Y_T, X_T = torch.load(open(load_path / "dgp" / "obs_T_dgp.pt", "rb"))

filt_or_dgp = "filt"
filt_dict = {"size_phi_t": row_run[f"filt_{bin_or_w}_size_phi_t"], "phi_tv": eval(row_run[f"{filt_or_dgp}_{bin_or_w}_phi_tv"]), "size_beta_t": row_run[f"{filt_or_dgp}_{bin_or_w}_size_beta_t"], "n_ext_reg": eval(row_run[f"{filt_or_dgp}_{bin_or_w}_n_ext_reg"]), "beta_tv": eval(row_run[f"{filt_or_dgp}_{bin_or_w}_beta_tv"])} 

filt_or_dgp = "dgp"
dgp_dict = {"size_phi_t": row_run[f"filt_{bin_or_w}_size_phi_t"], "phi_tv": eval(row_run[f"{filt_or_dgp}_{bin_or_w}_phi_tv"]), "size_beta_t": row_run[f"{filt_or_dgp}_{bin_or_w}_size_beta_t"], "n_ext_reg": eval(row_run[f"{filt_or_dgp}_{bin_or_w}_n_ext_reg"]), "beta_tv": eval(row_run[f"{filt_or_dgp}_{bin_or_w}_beta_tv"])} 

mod_dgp = get_filt_mod(bin_or_w, Y_T, X_T, dgp_dict)["ss"]
mod_dgp.load_par(str(load_path / "dgp"))

mod_filt = get_filt_mod(bin_or_w, Y_T, X_T, filt_dict)["ss"]
mod_filt.load_par(str(load_path))

phi_to_exclude = strIO_from_tens_T(Y_T>0) < 1e-3 
i = torch.where(~splitVec(phi_to_exclude)[0])[0][0]
x_T = X_T[0, 0, 0, :].numpy()
strIO_from_tens_T(Y_T>0)

fig_ax = plt.subplots(2, 1)
[ax.plot(x_T) for ax in fig_ax[1]]
mod_filt.plot_phi_T(i=i, fig_ax=mod_dgp.plot_phi_T(i=i, fig_ax=fig_ax))

fig_ax = plt.subplots(2, 1)
[ax.plot(x_T) for ax in fig_ax[1]]
mod_dgp.plot_phi_T(i=i, fig_ax=fig_ax)


phi_T,  _, beta_T = mod_filt.get_seq_latent_par()


mod_dgp.beta_T
mod_dgp.phi_tv

mod_filt.beta_T
mod_filt.phi_tv

