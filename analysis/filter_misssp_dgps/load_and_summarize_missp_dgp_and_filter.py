#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Thursday August 19th 2021

"""

# %% import packages
from pathlib import Path
import importlib
import logging
import dynwgraphs
import proj_utils
from proj_utils.mlflow import _get_and_set_experiment, get_df_exp
import pandas as pd
from matplotlib import pyplot as plt
from utils_missp_sim import load_all_models_missp_sim

logger = logging.getLogger(__name__)
importlib.reload(dynwgraphs)
importlib.reload(proj_utils)


# %% Table 1

experiment = _get_and_set_experiment("Default")

dfs = get_df_exp(experiment)

logger.info(
    f"Staus of experiment {experiment.name}: \n {dfs['info']['status'].value_counts()}"
)

ind_fin = (dfs["info"]["status"] == "FINISHED") & (
    ~dfs["metrics"]["filt_bin_sd_actual_n_opt_iter"].isna()
)

ind_fin = ind_fin & (~ dfs["par"]["init_sd_type"].isna())

df_i = dfs["info"][ind_fin]
df_p = dfs["par"][ind_fin]
df_m = dfs["metrics"][ind_fin]

#%%
group_cols = [
    "beta_set_bin_dgp",
    "phi_set_bin_dgp",
    "beta_set_bin_filt",
    "phi_set_bin_filt",
    "phi_set_dgp_type_tv_w",
]
# check that each group contains elements with the same settings for non grouping columns
df = df_p.drop(columns=["run_id", "n_sim", "n_jobs"]).groupby(group_cols).nunique() > 1

df.loc[:, df.any()]
df_p["filt_bin_sd_optimizer"].value_counts()

if not df.values.any():
    pass
else:
    logger.error(df.loc[:, df.any()])

df = df_i.merge(df_p, on="run_id").merge(df_m, on="run_id")

eval_cols = lambda ss_or_sd: [
    f"bin_mse_phi_{ss_or_sd}",
    f"w_mse_phi_{ss_or_sd}",
    f"bin_mse_beta_{ss_or_sd}",
    f"w_mse_beta_{ss_or_sd}",
    f"w_mse_phi_{ss_or_sd}",
]

beta_cols = lambda ss_or_sd: [f"bin_avg_beta_{ss_or_sd}", f"w_avg_beta_{ss_or_sd}"]

ss_or_sd = "sd"

# Drop 1% tails
all_cols = eval_cols(ss_or_sd)
df[all_cols].clip(
    lower=df[all_cols].quantile(0.01), upper=df[all_cols].quantile(0.99), axis=1
)


df_avg = df.groupby(group_cols).mean()[eval_cols(ss_or_sd)]

df.groupby(group_cols).count()[eval_cols(ss_or_sd)]
[d[["filt_bin_sd_optimizer"]].value_counts() for k, d in df.groupby(group_cols)]
df.groupby(group_cols).mean()[eval_cols("sd")]
df.groupby(group_cols).mean()[eval_cols("ss")]

# [d[eval_cols(ss_or_sd)].hist() for k, d in df.groupby(group_cols)]
# [d[beta_cols(ss_or_sd)].hist() for k, d in df.groupby(group_cols)]
# [d[beta_cols(ss_or_sd)].hist() for k, d in df.groupby(group_cols)]
# [print((k, d[avg_cols ])) for k, d in df.groupby(group_cols)]

df_avg
# %% Table 2

experiment = _get_and_set_experiment("dev Table 2 AR col")

dfs = get_df_exp(experiment)

logger.info(
    f"Staus of experiment {experiment.name}: \n {dfs['info']['status'].value_counts()}"
)

ind_fin = (dfs["info"]["status"] == "FINISHED") & (
    ~dfs["metrics"]["filt_bin_sd_actual_n_opt_iter"].isna()
)
ind_fin = ~dfs["metrics"]["filt_bin_sd_actual_n_opt_iter"].isna()
df_i = dfs["info"][ind_fin]
df_p = dfs["par"][ind_fin]
df_m = dfs["metrics"][ind_fin]

group_cols = [
    "beta_set_bin_dgp",
    "phi_set_bin_dgp",
    "beta_set_bin_filt",
    "phi_set_bin_filt",
    "ext_reg_dgp_set_type_tv_bin",
]

# check that each group contains elements with the same settings for non grouping columns
df = df_p.drop(columns=["run_id", "n_sim", "n_jobs"]).groupby(group_cols).nunique() > 1

df.loc[:, df.any()]
df_p["filt_bin_sd_optimizer"].value_counts()

if not df.values.any():
    pass
else:
    logger.error(df.loc[:, df.any()])


def eval_cols_phi(ss_or_sd):
    return [f"bin_mse_phi_{ss_or_sd}", f"w_mse_phi_{ss_or_sd}"]


def eval_cols_beta(ss_or_sd):
    return [f"bin_mse_beta_1_{ss_or_sd}", f"w_mse_beta_1_{ss_or_sd}"]


def beta_cols(ss_or_sd):
    return [
        f"bin_avg_beta_1_{ss_or_sd}",
        f"bin_avg_beta_2_{ss_or_sd}",
        f"w_avg_beta_1_{ss_or_sd}",
        f"w_avg_beta_2_{ss_or_sd}",
    ]


ss_or_sd = "sd"
# Drop 1% tails
df = df_i.merge(df_p, on="run_id").merge(df_m, on="run_id")
all_eval_cols = eval_cols_phi("sd") + eval_cols_phi("ss")
df[all_eval_cols] = df[all_eval_cols].clip(
    lower=df[all_eval_cols].quantile(0.05),
    upper=df[all_eval_cols].quantile(0.95),
    axis=1,
)

df_avg = df.groupby(group_cols).mean()[all_eval_cols]

df.groupby(group_cols).count()[eval_cols(ss_or_sd)]
[d[["filt_bin_sd_optimizer"]].value_counts() for k, d in df.groupby(group_cols)]
df.groupby(group_cols).mean()[eval_cols("sd")]

# [d[eval_cols(ss_or_sd)].hist() for k, d in df.groupby(group_cols)]
# [d[beta_cols(ss_or_sd)].hist() for k, d in df.groupby(group_cols)]
# [d[beta_cols(ss_or_sd)].hist() for k, d in df.groupby(group_cols)]
# [print((k, d[avg_cols ])) for k, d in df.groupby(group_cols)]

df_avg

#%%
inds = df["phi_set_dgp_type_tv_w"] == """('SIN', 'ref_mat', 1.0, 0.15)"""
row_run = df[inds].iloc[0]
(
    mod_filt_sd_bin,
    mod_filt_sd_w,
    mod_filt_ss_bin,
    mod_filt_ss_w,
    mod_dgp_bin,
    mod_dgp_bin,
    mod_dgp_w,
    obs,
    Y_reference,
) = load_all_models_missp_sim(row_run)


phi_T_ss = mod_filt_ss_w.get_ts_phi()
phi_T_sd = mod_filt_sd_w.get_ts_phi()
phi_T_dgp = mod_dgp_w.get_ts_phi()

#%%
i = 1
plt.plot(phi_T_dgp[i, :], "-k")
plt.plot(phi_T_ss[i, :], ".b")
plt.plot(phi_T_sd[i, :], "-r")


# %%
