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
from proj_utils import drop_keys
from proj_utils.mlflow import _get_and_set_experiment, _get_or_run, uri_to_path, get_fold_namespace, check_and_tag_test_run

logger = logging.getLogger(__name__)

# %%


@click.command()
@click.option("--max_opt_iter", default=8000, type=int)
@click.option("--unit_meas", default=10000, type=float)
@click.option("--train_fract", default=8/10, type=float)
@click.option("--inds_to_run", default="", type=str)
@click.option("--init_sd_type", default="est_ss_before", type=str)
@click.option("--n_jobs", default=5, type=int)
@click.option("--estimate_ss", type=float, default=0)
@click.option("--opt_n", default="ADAMHD", type=str)
@click.option("--run_parallel", type=float, default=0)
@click.option("--start_from_prev", default=0, type=float)
@click.option("--filter_ss_or_sd", type=str, default="sd")





def estimate_all_regressors_comb_emid(**kwargs):

    check_and_tag_test_run(kwargs)
    inds_to_run = kwargs.pop("inds_to_run")
    run_parallel = bool(kwargs.pop("run_parallel"))

    run_parameters_list = [
        {"bin_or_w": "bin", "regressor_name": "eonia", **kwargs},
        {"bin_or_w": "w", "regressor_name": "eonia", **kwargs},
        {"bin_or_w": "bin", "regressor_name": "Atm1", **kwargs},
        {"bin_or_w": "w", "regressor_name": "logYtm1", **kwargs}]
        # {"bin_or_w": "w", "regressor_name": "Atm1", **kwargs},
        # {"bin_or_w": "bin", "regressor_name": "logYtm1", **kwargs},
        # {"bin_or_w": "bin", "regressor_name": "eonia_Atm1", **kwargs},
        # {"bin_or_w": "w", "regressor_name": "eonia_Atm1", **kwargs},
        # {"bin_or_w": "bin", "regressor_name": "eonia_logYtm1", **kwargs},
        # {"bin_or_w": "w", "regressor_name": "eonia_logYtm1", **kwargs}


    if inds_to_run != '':
        inds_to_run = eval(", ".join(inds_to_run.split("_")))
        if type(inds_to_run) == int:
            inds_to_run = [inds_to_run]
        run_parameters_list = [run_parameters_list[i] for i in inds_to_run]
        logger.warning(f"Going to run only: {run_parameters_list}")              

    if run_parallel:
        if kwargs["start_from_prev"] == 0:
            raise

        Parallel(n_jobs=kwargs["n_jobs"])(delayed(one_run)(par) for par in run_parameters_list)
    else:
        for p in run_parameters_list:
            one_run(p)               


def one_run(par):
    mlflow.run(".", "estimate_all_models_same_reg", parameters=par, use_conda=False)


# %% Run
if __name__ == "__main__":
    estimate_all_regressors_comb_emid()

