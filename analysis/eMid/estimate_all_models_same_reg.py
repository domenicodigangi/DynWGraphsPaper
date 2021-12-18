#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Tuesday August 24th 2021

"""


# %% import packages
import mlflow
import torch
import click
import logging
import tempfile
from pathlib import Path
from proj_utils import drop_keys
from joblib import Parallel, delayed
from proj_utils.mlflow import _get_and_set_experiment, _get_or_run, uri_to_path, get_fold_namespace, check_and_tag_test_run
from dynwgraphs.utils.tensortools import splitVec, strIO_from_tens_T
from dynwgraphs.dirGraphs1_dynNets import dirBin1_SD, dirSpW1_SD, dirBin1_sequence_ss, dirSpW1_sequence_ss
import pickle
from eMid_data_utils import get_obs_and_regr_mat_eMid

logger = logging.getLogger(__name__)

# %%


@click.command()
@click.option("--max_opt_iter", default=8000, type=int)
@click.option("--unit_meas", default=10000, type=float)
@click.option("--train_fract", default=8/10, type=float)
@click.option("--T_0", default=0, type=int)
@click.option("--regressor_name", default="eonia", type=str)
@click.option("--bin_or_w", default="bin", type=str)
@click.option("--experiment_name", default="eMid Estimates", type=str)
@click.option("--init_sd_type", default="est_ss_before", type=str)
@click.option("--start_from_prev", default=0, type=float)
@click.option("--n_jobs", default=5, type=int)
@click.option("--estimate_ss", type=float, default=0)
@click.option("--opt_n", default="ADAMHD", type=str)
@click.option("--filter_ss_or_sd", type=str, default="sd")



def estimate_all_models_same_reg(**kwargs):

    kwargs = drop_keys(kwargs, ["experiment_name"])
    start_from_prev = bool(kwargs.pop("start_from_prev"))
    n_jobs = kwargs.pop("n_jobs")
    check_and_tag_test_run(kwargs)

    logger.info(kwargs)
    if start_from_prev:
        run_0_stat = mlflow.run(".", "estimate_single_model", parameters={"size_beta_t": "0", "beta_tv": 0, **kwargs}, use_conda=False)
        run_0_stat = mlflow.tracking.MlflowClient().get_run(run_0_stat.run_id)


        run_1_stat = mlflow.run(".", "estimate_single_model", parameters={"prev_mod_art_uri": run_0_stat.info.artifact_uri, "size_beta_t": "one", "beta_tv": 0, **kwargs}, use_conda=False)
        run_1_stat = mlflow.tracking.MlflowClient().get_run(run_1_stat.run_id)

    
        run_1_tv = mlflow.run(".", "estimate_single_model", parameters={"prev_mod_art_uri": run_1_stat.info.artifact_uri, "size_beta_t": "one", "beta_tv": 1, **kwargs}, use_conda=False)
        run_1_tv = mlflow.tracking.MlflowClient().get_run(run_1_tv.run_id)


        run_2N_stat = mlflow.run(".", "estimate_single_model", parameters={"prev_mod_art_uri": run_0_stat.info.artifact_uri, "size_beta_t": "2N", "beta_tv": 0, **kwargs}, use_conda=False)
        run_2N_stat = mlflow.tracking.MlflowClient().get_run(run_2N_stat.run_id)


        run_2N_tv = mlflow.run(".", "estimate_single_model", parameters={"prev_mod_art_uri": run_2N_stat.info.artifact_uri, "size_beta_t": "2N", "beta_tv": 1, **kwargs}, use_conda=False)
        run_2N_tv = mlflow.tracking.MlflowClient().get_run(run_2N_tv.run_id)
    else:
    
        run_parameters_list = [
            {"size_phi_t": "0", "phi_tv": 0, "size_beta_t": "1", "beta_tv": 0, **kwargs},
            {"size_phi_t": "2N", "phi_tv": 0, "size_beta_t": "1", "beta_tv": 0, **kwargs},
            {"size_phi_t": "2N", "phi_tv": 1, "size_beta_t": "1", "beta_tv": 0, **kwargs},
            {"size_phi_t": "2N", "phi_tv": 1, "size_beta_t": "1", "beta_tv": 1, **kwargs},
            {"size_phi_t": "2N", "phi_tv": 1, "size_beta_t": "0", "beta_tv": 0, **kwargs},
            {"size_phi_t": "2N", "phi_tv": 1, "size_beta_t": "2N", "beta_tv": 0, **kwargs}
        ]


        
        Parallel(n_jobs=n_jobs)(delayed(one_run)(par) for par in run_parameters_list)

def one_run(par):
    mlflow.run(".", "estimate_single_model", parameters=par, use_conda=False)

# %% Run
if __name__ == "__main__":
    estimate_all_models_same_reg()

