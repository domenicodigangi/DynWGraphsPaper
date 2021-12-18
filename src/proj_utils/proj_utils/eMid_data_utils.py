#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday August 4th 2021

"""

import torch
from dynwgraphs.utils.tensortools import tens
from dynwgraphs.dirGraphs1_dynNets import get_gen_fit_mod
import numpy as np
from types import SimpleNamespace
from pathlib import Path
from proj_utils.mlflow import uri_to_path, _get_or_run
import pickle
import logging
logger = logging.getLogger(__name__)


def get_obs_and_regr_mat_eMid(ld_data, unit_meas, regressor_name, T_0):

    assert T_0 >= 0, "T0 must be positive"
    Y_T = ld_data["YeMidWeekly_T"][:, :, T_0 + 2:]/unit_meas #drop the first observation because it's weird and the second because we are using lagged observations as regressors
    # lagged nets to be used as regressor
    Ytm1_T = ld_data["YeMidWeekly_T"][:, :, T_0 + 1:-1].unsqueeze(dim=2)/unit_meas
    logYtm1_T = (torch.log(Ytm1_T)).nan_to_num(posinf=0, neginf=0)
    N, _, T = Y_T.shape
    X_T = torch.zeros(N, N, 1, T)

    regr_list_in = regr_list = regressor_name.replace(" ", "_").split("_")

    for r in regr_list_in:
        assert r in ["eonia", "logYtm1", "Atm1"], f"Invalid regressor name {r}"

    if "eonia" in regr_list_in:
        X_eonia_T = tens(np.tile(ld_data["eonia_T"][:, T_0 + 2:].numpy(), (N, N, 1, 1)))
        X_T = torch.cat((X_T, X_eonia_T), dim=2)
        regr_list.append("eonia")
    if "logYtm1" in regr_list_in:
        X_T = torch.cat((X_T, logYtm1_T), dim=2)
        regr_list.append("logYtm1")
    if "Atm1" in regr_list_in:
        X_T = torch.cat((X_T, Ytm1_T >0), dim=2)
        regr_list.append("Atm1")

    if X_T.shape[2] >1:
        X_T = X_T[:, :, 1:, :]
        regr_list = regr_list[1:]
    else:
        logger.info("No regressor selected")
        X_T = None

    net_stats = SimpleNamespace()
    net_stats.__dict__.update({
        "dates": ld_data["all_dates"],
        "avg_degs_i": (Y_T > 0).sum(axis=(0)).double().mean(axis=1),
        "avg_degs_o": (Y_T > 0).sum(axis=(1)).double().mean(axis=1),
        "avg_str_i": (Y_T).sum(axis=(0)).mean(axis=1),
        "avg_str_o": (Y_T).sum(axis=(1)).mean(axis=1),
    })

    return Y_T, X_T, regr_list, net_stats

def load_all_models_emid(Y_T, X_T, row_run):

    load_path = Path(uri_to_path(row_run["artifact_uri"]))
   
    if "filter_ss_or_sd" in list(row_run.index.values):
        mod = get_model_from_run_dict_emid(Y_T, X_T, row_run, row_run["filter_ss_or_sd"])
        mod.load_par(str(load_path)) 
    
        if row_run["filter_ss_or_sd"] == "sd":
            mod.roll_sd_filt_train()

        return mod 
    else:
        try:
            mod_filt_ss = get_model_from_run_dict_emid(Y_T, X_T, row_run, "ss")
            mod_filt_ss.load_par(str(load_path)) 
        except:
            logger.warning("ss model not found")    
            mod_filt_ss = None
        mod_filt_sd = get_model_from_run_dict_emid(Y_T, X_T, row_run, "sd")
        mod_filt_sd.load_par(str(load_path)) 
        mod_filt_sd.roll_sd_filt_train()


        return mod_filt_ss, mod_filt_sd

def get_model_from_run_dict_emid(Y_T, X_T, run_d, ss_or_sd, use_mod_str=False):

    if run_d["size_beta_t"] in [0, "0", None]:
        X_T=None

    T_all = Y_T.shape[2]

    if "t_train" in run_d.keys():
        T_train = int(float(run_d["t_train"]))
    else:
        T_train = int(float(run_d["train_fract"]) * T_all)

    mod_in_names = ["phi_tv", "beta_tv"]
    if use_mod_str:
        mod_str = f"{ss_or_sd}_"
    else:
        mod_str = f""
    mod_par_dict = {k: run_d[f"{mod_str}{k}"] for k in mod_in_names}
    mod_in_names = ["size_phi_t", "size_beta_t"]

    mod_par_dict.update({k: run_d[f"{k}"] for k in mod_in_names})
    if ss_or_sd == "sd":
        if "init_sd_type" in run_d.keys():
            mod_par_dict["init_sd_type"] = run_d["init_sd_type"]
        else:
            mod_par_dict["init_sd_type"] = run_d["sd_init_sd_type"]

    out_mod =  get_gen_fit_mod(run_d["bin_or_w"], ss_or_sd, Y_T, X_T=X_T, T_train=T_train, **mod_par_dict)

    return out_mod

def get_data_from_data_run(unit_meas, regr_name, parent_mlruns_folder=None, T_0=None):

    if T_0 is None:
        T_0 = 0
    try:
        load_and_log_data_run = _get_or_run("load_and_log_data", None, None)
        load_path_orig = uri_to_path(load_and_log_data_run.info.artifact_uri)
        if parent_mlruns_folder is not None:
            load_path = Path(parent_mlruns_folder) / Path(*Path(load_path_orig).parts[-3:])
        else:
            load_path = load_path_orig

        load_file = Path(load_path) / "data" / "emid_data.pkl" 
        ld_data = pickle.load(open(load_file, "rb"))
    except:
        logger.error("unable to load from data run")
        load_and_log_data_run = _get_or_run("load_and_log_data", None, None, use_cache=False)
        load_path = uri_to_path(load_and_log_data_run.info.artifact_uri)
        load_file = Path(load_path) / "data" / "emid_data.pkl" 
        ld_data = pickle.load(open(load_file, "rb"))


    return get_obs_and_regr_mat_eMid(ld_data, unit_meas, regr_name, T_0)