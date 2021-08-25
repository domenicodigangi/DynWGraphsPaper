#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Tuesday August 24th 2021

"""


# %% import packages
import mlflow
import click
import logging
from joblib import Parallel, delayed
from ddg_utils import drop_keys
from ddg_utils.mlflow import _get_and_set_experiment, _get_or_run, uri_to_path, get_fold_namespace, check_and_tag_test_run

logger = logging.getLogger(__name__)

# %%


@click.command()
@click.option("--max_opt_iter", default=11000, type=int)
@click.option("--unit_meas", default=10000, type=float)
@click.option("--train_fract", default=3/4, type=float)
@click.option("--test_mod_ind", default=0, type=int)
@click.option("--inds_to_run", default="", type=str)


def estimate_all_models_emid_parallel(**kwargs):

    check_and_tag_test_run(kwargs)
    test_mod_ind = kwargs.pop('test_mod_ind')
    inds_to_run = kwargs.pop("inds_to_run")

    run_parameters_list = [
        {"bin_or_w": "bin", "regressor_name": "eonia", **kwargs},
        {"bin_or_w": "w", "regressor_name": "eonia", **kwargs},
        {"bin_or_w": "bin", "regressor_name": "logYtm1", **kwargs},
        {"bin_or_w": "w", "regressor_name": "logYtm1", **kwargs},
        {"bin_or_w": "bin", "regressor_name": "eonia_logYtm1", **kwargs},
        {"bin_or_w": "w", "regressor_name": "eonia_logYtm1", **kwargs}]


    if test_mod_ind == 0:
        if inds_to_run != "":
            run_parameters_list = eval(f"run_parameters_list[{inds_to_run}]")                
            
            Parallel(n_jobs=6)(delayed(run_one_sequence)(par) for par in run_parameters_list)
    else:
        run_one_sequence(run_parameters_list[test_mod_ind])


def run_one_sequence(parameters):
    run = mlflow.run(".", "estimate_model_sequence_emid", parameters=parameters, use_conda=False)
    run = mlflow.tracking.MlflowClient().get_run(run.run_id)





# %% Run
if __name__ == "__main__":
    estimate_all_models_emid_parallel()

