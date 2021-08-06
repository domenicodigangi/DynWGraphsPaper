#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Friday August 6th 2021

"""

# %% import packages
from pathlib import Path
import torch
from dynwgraphs.utils.tensortools import splitVec, strIO_from_tens_T
from dynwgraphs.dirGraphs1_dynNets import dirBin1_sequence_ss, dirBin1_SD, dirSpW1_SD, dirSpW1_sequence_ss
from dynwgraphs.utils.dgps import get_mod_and_par
import tempfile
import mlflow
import logging
from ddg_utils.mlflow import _get_and_set_experiment, check_test_exp

from sim_missp_dgp_and_filter_ss_sd import _run_parallel_simulations, filt_err

logger = logging.getLogger(__name__)

# %%

click_args = {}
click_args["n_sim"] = 2
click_args["max_opt_iter"] = 21
click_args["n_nodes"] = 50
click_args["n_time_steps"] = 100
click_args["type_dgp_phi_bin"] = "AR"
click_args["ext_reg_bin_dgp_set"] = (1, "one", False, "AR")
click_args["ext_reg_bin_sd_filt_set"] = (None, None, None)
click_args["exclude_weights"] = False
click_args["type_dgp_phi_w"] = "AR"
click_args["ext_reg_w_dgp_set"] = (0, "one", False, "AR")
click_args["ext_reg_w_sd_filt_set"] = (None, None, None)
click_args["n_jobs"] = 4
click_args["experiment_name"] = "filter missp dgp"


kwargs = click_args

_run_parallel_simulations(**kwargs)

# %%
