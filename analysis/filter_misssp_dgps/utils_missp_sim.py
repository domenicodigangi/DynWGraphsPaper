#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday August 4th 2021

"""

import torch
from dynwgraphs.utils.tensortools import tens
import numpy as np
from types import SimpleNamespace
from pathlib import Path
from ddg_utils.mlflow import uri_to_path, _get_or_run
import pickle


def load_all_models_missp_sim(row_run, obs):
    load_path = Path(uri_to_path(row_run["artifact_uri"]))

    Y_T, X_T = obs
    
    mod_filt_ss = get_model_from_run_dict("filt_ss", row_run["bin_or_w"], Y_T, X_T, row_run)
    mod_filt_ss.load_par(str(load_path)) 
    
    mod_filt_sd = get_model_from_run_dict("filt_sd", row_run["bin_or_w"], Y_T, X_T, row_run)
    mod_filt_sd.load_par(str(load_path)) 
    mod_filt_sd.roll_sd_filt_train()


    return mod_filt_ss, mod_filt_sd


def get_model_from_run_dict(dgp_or_filt, bin_or_w, Y_T, X_T, run_d):

    if "T_train" in run_d.keys():
        T_train = int(run_d["T_train"])
    else:
        T_train = None

    if dgp_or_filt == "dgp":
        mod_in_names = ["size_phi_t", "phi_tv", "beta_tv", "size_beta_t"]
        mod_str = f"{dgp_or_filt}_{bin_or_w}"
        mod_par_dict = {k: run_d[f"{mod_str}_{k}"] for k in mod_in_names}
        ss_or_sd = "ss"
        n_ext_reg = X_T.shape[2]
    elif dgp_or_filt[:4] == "filt":
        ss_or_sd = dgp_or_filt[5:]
        mod_in_names = ["phi_tv", "beta_tv"]
        mod_str = f"filt_{bin_or_w}_{ss_or_sd}"
        mod_par_dict = {k: run_d[f"{mod_str}_{k}"] for k in mod_in_names}
        mod_in_names = ["size_phi_t", "size_beta_t"]
        mod_str = f"filt_{bin_or_w}"
        mod_par_dict.update({k: run_d[f"{mod_str}_{k}"] for k in mod_in_names})
        n_ext_reg = int(run_d[f"filt_{bin_or_w}_n_ext_reg"])
    else:
        raise

    out_mod =  get_gen_fit_mod(bin_or_w, ss_or_sd, Y_T, X_T=X_T[:, :, :n_ext_reg, :], T_train=T_train, **mod_par_dict)
    # if w mod init also it's binary submod
    if bin_or_w  == "w":
        bin_mod = get_model_from_run_dict(dgp_or_filt, "bin", Y_T, X_T[:, :, :n_ext_reg, :], run_d)

        out_mod.bin_mod = bin_mod

    return out_mod
