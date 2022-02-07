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
from proj_utils.mlflow import uri_to_path
from dynwgraphs.dirGraphs1_dynNets import get_gen_fit_mod


def load_all_models_missp_sim(row_run):
    load_path = Path(uri_to_path(row_run["artifact_uri"]))

    obs = torch.load(load_path / "dgp" / "obs_T_dgp.pt")
    Y_reference = torch.load(load_path / "dgp" / "Y_reference.pt")

    Y_T, X_T = obs

    mod_filt_sd_bin = get_model_from_run_dict("filt_sd", "bin", Y_T, X_T, row_run)
    mod_filt_sd_bin.load_par(str(load_path))
    mod_filt_sd_bin.roll_sd_filt_train()
    mod_filt_sd_w = get_model_from_run_dict("filt_sd", "w", Y_T, X_T, row_run)
    mod_filt_sd_w.load_par(str(load_path))
    mod_filt_sd_w.roll_sd_filt_train()  
    
    mod_filt_ss_bin = get_model_from_run_dict("filt_ss", "bin", Y_T, X_T, row_run)
    mod_filt_ss_bin.load_par(str(load_path))
    mod_filt_ss_w = get_model_from_run_dict("filt_ss", "w", Y_T, X_T, row_run)
    mod_filt_ss_w.load_par(str(load_path))
 
    mod_dgp_w = get_model_from_run_dict("dgp", "w", Y_T, X_T, row_run)
    mod_dgp_w.load_par(str(load_path / "dgp"))

    mod_dgp_bin = get_model_from_run_dict("dgp", "bin", Y_T, X_T, row_run)
    mod_dgp_bin.load_par(str(load_path / "dgp"))

    return (
        mod_filt_sd_bin,
        mod_filt_sd_w,
        mod_filt_ss_bin,
        mod_filt_ss_w,
        mod_dgp_bin,
        mod_dgp_bin,
        mod_dgp_w,
        obs,
        Y_reference,
    )


def get_model_from_run_dict(dgp_or_filt, bin_or_w, Y_T, X_T, run_d):

    if "T_train" in run_d.keys():
        T_train = int(run_d["T_train"])
    else:
        T_train = None

    if X_T is not None:
        n_ext_reg = X_T.shape[2]
        X_T = X_T[:, :, :n_ext_reg, :]
    else:
        n_ext_reg = 0

    if dgp_or_filt == "dgp":
        mod_in_names = ["size_phi_t", "phi_tv", "beta_tv", "size_beta_t"]
        mod_str = f"{dgp_or_filt}_{bin_or_w}"
        mod_par_dict = {k: run_d[f"{mod_str}_{k}"] for k in mod_in_names}
        ss_or_sd = "ss"
    elif dgp_or_filt[:4] == "filt":
        ss_or_sd = dgp_or_filt[5:]
        mod_in_names = ["phi_tv", "beta_tv"]
        mod_str = f"filt_{bin_or_w}_{ss_or_sd}"
        mod_par_dict = {k: run_d[f"{mod_str}_{k}"] for k in mod_in_names}
        mod_in_names = ["size_phi_t", "size_beta_t"]
        mod_str = f"filt_{bin_or_w}"
        mod_par_dict.update({k: run_d[f"{mod_str}_{k}"] for k in mod_in_names})
        n_ext_reg = int(run_d[f"filt_{bin_or_w}_n_ext_reg"])
        if ss_or_sd == "sd":
            mod_par_dict["init_sd_type"] = run_d["init_sd_type"]
    else:
        raise

    out_mod = get_gen_fit_mod(
        bin_or_w,
        ss_or_sd,
        Y_T,
        X_T=X_T,
        T_train=T_train,
        **mod_par_dict,
    )
    # if w mod init also it's binary submod
    if bin_or_w == "w":
        bin_mod = get_model_from_run_dict(
            dgp_or_filt, "bin", Y_T, X_T, run_d
        )

        out_mod.bin_mod = bin_mod

    return out_mod
