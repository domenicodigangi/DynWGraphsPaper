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
from ddg_utils.mlflow import uri_to_path, _get_or_run
import pickle

def get_obs_and_regr_mat_eMid(ld_data, unit_meas, regressor_name):
    Y_T = ld_data["YeMidWeekly_T"][:, :, 2:]/unit_meas
    # lagged nets to be used as regressor
    Ytm1_T = ld_data["YeMidWeekly_T"][:, :, 1:-1].unsqueeze(dim=2)/unit_meas
    logYtm1_T = (torch.log(Ytm1_T)).nan_to_num(posinf=0, neginf=0)
    N, _, T = Y_T.shape
    X_T = torch.zeros(N, N, 1, T)

    regr_list_in = regr_list = regressor_name.replace(" ", "_").split("_")

    for r in regr_list_in:
        assert r in ["eonia", "logYtm1", "Atm1"], f"Invalid regressor name {r}"

    if "eonia" in regr_list_in:
        X_eonia_T = tens(np.tile(ld_data["eonia_T"][:, 2:].numpy(), (N, N, 1, 1)))
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
        "avg_degs_i": (Y_T > 0).sum(axis=(0)).double().mean(axis=1),
        "avg_degs_o": (Y_T > 0).sum(axis=(1)).double().mean(axis=1),
        "avg_str_i": (Y_T).sum(axis=(0)).mean(axis=1),
        "avg_str_o": (Y_T).sum(axis=(1)).mean(axis=1),
    })

    return Y_T, X_T, regr_list, net_stats



def load_all_models_emid(Y_T, X_T, row_run):

    load_path = Path(uri_to_path(row_run["artifact_uri"]))
   
    mod_filt_ss = get_model_from_run_dict_emid(Y_T, X_T, row_run, "ss")
    mod_filt_ss.load_par(str(load_path)) 
    
    mod_filt_sd = get_model_from_run_dict_emid(Y_T, X_T, row_run, "sd")
    mod_filt_sd.load_par(str(load_path)) 
    mod_filt_sd.roll_sd_filt_train()


    return mod_filt_ss, mod_filt_sd



def get_model_from_run_dict_emid(Y_T, X_T, run_d, ss_or_sd):

    if run_d["size_beta_t"] in [0, "0", None]:
        X_T=None

    T_all = Y_T.shape[2]

    T_train = int(float(run_d["train_fract"]) * T_all)

    mod_in_names = ["phi_tv", "beta_tv"]
    mod_str = f"{ss_or_sd}"
    mod_par_dict = {k: run_d[f"{mod_str}_{k}"] for k in mod_in_names}
    mod_in_names = ["size_phi_t", "size_beta_t"]
    mod_par_dict.update({k: run_d[f"{k}"] for k in mod_in_names})
    if ss_or_sd == "sd":
        mod_par_dict["init_sd_type"] = run_d["sd_init_sd_type"]

    out_mod =  get_gen_fit_mod(run_d["bin_or_w"], ss_or_sd, Y_T, X_T=X_T, T_train=T_train, **mod_par_dict)

    return out_mod




def get_data_from_data_run(unit_meas, regr_name):
    load_and_log_data_run = _get_or_run("load_and_log_data", None, None)
    load_path = uri_to_path(load_and_log_data_run.info.artifact_uri)

    load_file = Path(load_path) / "data" / "eMid_data.pkl" 

    ld_data = pickle.load(open(load_file, "rb"))


    return get_obs_and_regr_mat_eMid(ld_data, unit_meas, regr_name)



