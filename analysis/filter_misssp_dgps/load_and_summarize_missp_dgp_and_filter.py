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

experiment = _get_and_set_experiment("Table 1")

dfs = get_df_exp(experiment)

logger.info(f"Staus of experiment {experiment.name}: \n {dfs['info']['status'].value_counts()}")

ind_fin = ( dfs["info"]["status"] == "FINISHED") & (~dfs["metrics"]["filt_bin_sd_actual_n_opt_iter"].isna())
df_i = dfs["info"][ind_fin]
df_p = dfs["par"][ind_fin]
df_m = dfs["metrics"][ind_fin]

group_cols = ["beta_set_bin_dgp", "beta_set_bin_filt",  "ext_reg_dgp_set_type_tv", "phi_set_dgp_type_tv_bin", "beta_set_dgp_type_tv_bin", "phi_set_bin_dgp", "phi_set_bin_filt"]

# check that each group contains elements with the same settings for non grouping columns
df = df_p.drop(columns = ['run_id', 'n_sim', 'n_jobs']).groupby(group_cols).nunique() > 1

df.loc[:,df.any()]
df_p["filt_bin_sd_optimizer"].value_counts()

if not df.values.any():
    pass
else:
    logger.error(df.loc[:,df.any()])

df = df_i.merge(df_p, on="run_id").merge(df_m, on="run_id")

eval_cols = lambda ss_or_sd: [f"bin_mse_phi_{ss_or_sd}", f"w_mse_phi_{ss_or_sd}", f"bin_mse_beta_{ss_or_sd}", f"w_mse_beta_{ss_or_sd}", f"filt_bin_{ss_or_sd}_auc_score", f"filt_w_{ss_or_sd}_mse", f"filt_w_{ss_or_sd}_mse_log"]

beta_cols = lambda ss_or_sd: [f"bin_avg_beta_{ss_or_sd}", f"w_avg_beta_{ss_or_sd}"]

# Drop 1% tails
ss_or_sd = "sd"
all_cols = eval_cols(ss_or_sd) 
df[all_cols] = df[all_cols].clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)

df.groupby(group_cols).mean()[eval_cols(ss_or_sd)]
df.groupby(group_cols).count()[eval_cols(ss_or_sd)]
[d[["filt_bin_sd_optimizer"]].value_counts() for k, d in df.groupby(group_cols)]
df.groupby(group_cols).mean()[eval_cols("ss")]
df.groupby(group_cols).mean()[eval_cols]

[d[eval_cols(ss_or_sd)].hist() for k, d in df.groupby(group_cols)]
[d[beta_cols(ss_or_sd)].hist() for k, d in df.groupby(group_cols)]
[d[beta_cols(ss_or_sd)].hist() for k, d in df.groupby(group_cols)]
[print((k, d[avg_cols ])) for k, d in df.groupby(group_cols)]



