#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday August 25th 2021

"""
# %% import packages
import mlflow
import torch
import click
import logging
import tempfile
from pathlib import Path
from proj_utils import drop_keys
from proj_utils.mlflow import _get_and_set_experiment, _get_or_run, uri_to_path, get_fold_namespace, check_and_tag_test_run, get_df_exp
import pandas as pd
import numpy as np
from dynwgraphs.utils.tensortools import splitVec, strIO_from_tens_T
from dynwgraphs.dirGraphs1_dynNets import dirBin1_SD, dirSpW1_SD, dirBin1_sequence_ss, dirSpW1_sequence_ss, get_gen_fit_mod
import pickle
from eMid_data_utils import get_data_from_data_run, load_all_models_emid, get_model_from_run_dict_emid
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# %%



@click.command()
@click.option("--experiment_name", type=str, default="emid est paper last")


def evaluate_models_out_of_sample(**kwargs):
    experiment = _get_and_set_experiment(kwargs["experiment_name"])
    df_all_runs = get_df_exp(experiment, one_df=True)

    logger.info(f"Staus of experiment {experiment.name}: \n {df_all_runs['status'].value_counts()}")

    df_reg = df_all_runs[(df_all_runs["status"].apply(lambda x: x in ["FINISHED"])) & (~np.isnan(df_all_runs["sd_actual_n_opt_iter"]))]

    # row_run = df_reg[(df_reg["beta_tv"] == "0") & (df_reg["size_beta_t"]=="0") & (df_reg["bin_or_w"]=="bin")].iloc[0, :]

    log_cols = ["regressor_name", "size_phi_t", "phi_tv",  "size_beta_t", "beta_tv"]
    for i, row_run in df_reg.iterrows():

        with mlflow.start_run(run_id=row_run["run_id"]) as run:

            log_dict = {k: row_run[k] for k in log_cols}
            try:
                Y_T, X_T, regr_list, net_stats = get_data_from_data_run(float(row_run["unit_meas"]), row_run["regressor_name"])

                mod_ss, mod_sd = load_all_models_emid(Y_T, X_T, row_run)

                mod_ss.loglike_seq_T()
                row_run["ss_final_loss"]
                mod_sd.loglike_seq_T()
                mod_sd.init_sd_type
                row_run["sd_final_loss"]

                mod_sd.sd_stat_par_un_phi["init_val"]

                out_sample_fit = { f"sd_out_of_sample_{k}_post": v for k, v in  mod_sd.out_of_sample_eval(exclude_never_obs_train=True).items()}
                if row_run["bin_or_w"] == "bin":
                    out_sample_fit.update({ f"sd_out_of_sample_{k}_all_post": v for k, v in  mod_sd.out_of_sample_eval(exclude_never_obs_train=False).items()})
                mlflow.log_metrics(out_sample_fit)
                logger.info(f"Computed out of sample gof for {log_dict}")
            except:
                logger.error(f"Could not compute out of sample gof for {log_dict}")



# %% Run
if __name__ == "__main__":
    evaluate_models_out_of_sample()

