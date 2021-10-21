#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Tuesday October 19th 2021

"""


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
from ddg_utils import drop_keys
from joblib import Parallel, delayed
from ddg_utils.mlflow import _get_and_set_experiment, _get_or_run, uri_to_path, get_fold_namespace, check_and_tag_test_run
from dynwgraphs.utils.tensortools import splitVec, strIO_from_tens_T
from dynwgraphs.dirGraphs1_dynNets import dirBin1_SD, dirSpW1_SD, dirBin1_sequence_ss, dirSpW1_sequence_ss
import pickle
from eMid_data_utils import get_obs_and_regr_mat_eMid

logger = logging.getLogger(__name__)

# %%


@click.command()
@click.option("--max_opt_iter", default=8000, type=int)
@click.option("--unit_meas", default=10000, type=float)
@click.option("--regressor_name", default="eonia", type=str)
@click.option("--bin_or_w", default="bin", type=str)
@click.option("--experiment_name", default="eMid Estimates rolling", type=str)
@click.option("--init_sd_type", default="est_ss_before", type=str)
@click.option("--n_jobs", default=5, type=int)
@click.option("--opt_n", default="ADAMHD", type=str)
@click.option("--filter_ss_or_sd", type=str, default="sd")
@click.option("--t_train", type=int, default=100)
@click.option("--max_T_0", type=int, default=190)
@click.option("--min_T_0", type=int, default=0)
@click.option("--run_in_parallel", type=int, default=1)



def estimate_mod_seq(**kwargs):

    kwargs = drop_keys(kwargs, ["experiment_name"])
    n_jobs = kwargs.pop("n_jobs")
    max_t_0 = kwargs.pop("max_t_0")
    min_t_0 = kwargs.pop("min_t_0")
    run_in_parallel = bool(kwargs.pop("run_in_parallel"))
    check_and_tag_test_run(kwargs)

    logger.info(f"estimate_mod_seq ==== {kwargs}")
    run_par = {"size_phi_t": "2N", "phi_tv": 1.0, "size_beta_t": "0", "beta_tv": 0.0, **kwargs}

    if kwargs["filter_ss_or_sd"] == "sd":
        T_0_list = list(range(min_t_0, max_t_0))
        if run_in_parallel:
            Parallel(n_jobs=n_jobs)(delayed(one_run)(run_par, T_0) for T_0 in T_0_list)
        else:
            for t_0 in T_0_list:
                one_run(run_par, t_0)

    elif kwargs["filter_ss_or_sd"] == "ss": # run single long sequence of SS estimates
        run_par["train_fract"] = 1.0
        run_par.pop("t_train")
        one_run(run_par, 0)


def one_run(par, T_0):
    par["t_0"] = T_0
    mlflow.run(".", "estimate_single_model", parameters=par, use_conda=False)

# %% Run
if __name__ == "__main__":
    estimate_mod_seq()

