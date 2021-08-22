#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Thursday August 19th 2021


file to run a single instance of simulations in run_sim_missp_dgp_and_filter_ss_sd.py

"""

# %% import packages
from pathlib import Path
import importlib
import torch
import dynwgraphs
from dynwgraphs.utils.tensortools import splitVec, strIO_from_tens_T
from dynwgraphs.utils.dgps import get_dgp_mod_and_par
import tempfile
import mlflow
from joblib import Parallel, delayed
from torch import nn
import logging
import click
from matplotlib import pyplot as plt
from ddg_utils.mlflow import _get_and_set_experiment, check_test_exp
from ddg_utils import drop_keys
from run_sim_missp_dgp_and_filter_ss_sd import get_filt_mod, get_dgp_mod_and_par, get_dgp_and_filt_set_from_cli_options, filt_err
logger = logging.getLogger(__name__)
importlib.reload(dynwgraphs)


# %%
click_args = {}

click_args["max_opt_iter"] = 10000
click_args["n_nodes"] = 50
click_args["n_time_steps"] = 100

click_args["type_tv_dgp_phi"] = ("AR", "ref_mat", 0.98, 0.1)
click_args["type_tv_dgp_phi_bin"] = (None, None, None, None)
click_args["type_tv_dgp_phi_w"] = (None, None, None, None)


click_args["phi_dgp_set"] = ("2N", True)
click_args["phi_dgp_set_bin"] = (None, None)
click_args["phi_dgp_set_w"] = (None, None)


click_args["phi_filt_set"] = ("2N", True)# click_args["phi_dgp_set"] #  #
click_args["phi_filt_set_bin"] = (None, None)
click_args["phi_filt_set_w"] = (None, None)

click_args["beta_dgp_set"] = (1, "one", False)#(0, None, False)
click_args["beta_dgp_set_bin"] = (None, None, None)
click_args["beta_dgp_set_w"] = (None, None, None)


click_args["beta_filt_set"] = click_args["beta_dgp_set"] # (1, "one", False)
click_args["beta_filt_set_bin"] = (None, None, None)
click_args["beta_filt_set_w"] = (None, None, None)


click_args["type_tv_dgp_beta"] = ("AR", 1, 0, 0)
click_args["type_tv_dgp_beta_bin"] = (None, None, None, None)
click_args["type_tv_dgp_beta_w"] = (None, None, None, None)

click_args["type_tv_dgp_ext_reg"] = ("uniform", "AR", 1, 0, 0.1)
click_args["type_tv_dgp_ext_reg_bin"] = (None, None, None, None, None)
click_args["type_tv_dgp_ext_reg_w"] = (None, None, None, None, None)



#%%
kwargs = click_args
T = kwargs["n_time_steps"]
N = kwargs["n_nodes"]

dgp_set_bin, filt_set_bin = get_dgp_and_filt_set_from_cli_options(kwargs, "bin")
    
dgp_set_w, filt_set_w = get_dgp_and_filt_set_from_cli_options(kwargs, "w")
        
# define binary dgp and filter par
logger.info(dgp_set_bin)
mod_dgp_bin, Y_reference_bin = get_dgp_mod_and_par(N=N, T=T, dgp_set_dict=dgp_set_bin)

# get_dgp_mod_and_par(N, T, "dirBin1",  1, "one", "AR", "const_unif_01", False,  Y_reference=None)

mod_dgp_bin.phi_tv
mod_dgp_bin.X_T
mod_dgp_dict = {"bin": mod_dgp_bin}

# define weighted dgp
logger.info(dgp_set_w)
mod_dgp_w, Y_reference_w = get_dgp_mod_and_par(N=N, T=T, dgp_set_dict=dgp_set_w)


mod_dgp_w.bin_mod = mod_dgp_bin
mod_dgp_dict["w"] = mod_dgp_w

run_par_dict = {"N": N, "T": T, "dgp_par_bin": dgp_set_bin, "dgp_par_w": dgp_set_w, "filt_par_bin": filt_set_bin, "filt_par_w": filt_set_w}



tmpdirname = tempfile.TemporaryDirectory()
tmp_path = Path(tmpdirname.name)
dgp_fold = tmp_path / "dgp"
dgp_fold.mkdir(exist_ok=True)
tb_fold = tmp_path / "tb_logs"
tb_fold.mkdir(exist_ok=True)

bin_or_w = "w" 
mod_dgp = mod_dgp_dict[bin_or_w]
logger.info(f" start estimates {bin_or_w}")

# sample obs from dgp and save data
if hasattr(mod_dgp, "bin_mod"):
    if mod_dgp.bin_mod.Y_T.sum() == 0:
        A_T = mod_dgp.bin_mod.sample_Y_T()
    else:
        A_T = mod_dgp.bin_mod.Y_T
    mod_dgp.Y_T = mod_dgp.sample_Y_T(A_T=A_T)
else:
    mod_dgp.Y_T = mod_dgp.sample_Y_T()

#estimate models and log parameters and hpar optimization
# run_par_dict[f"filt_par_{bin_or_w}"].update({"beta_start_val": 1})
# del run_par_dict[f"filt_par_{bin_or_w}"]["size_phi_t"]
# run_par_dict[f"filt_par_{bin_or_w}"]["size_beta_t"] = 1

filt_models = get_filt_mod(bin_or_w, mod_dgp.Y_T, mod_dgp.X_T, run_par_dict[f"filt_par_{bin_or_w}"])

k_filt = "sd" 
mod_filt = filt_models[k_filt]

#%%
_, h_par_opt, stats_opt = mod_filt.estimate_sd(tb_save_fold=tb_fold)


#%%
    
# compute mse for each model and log it 
phi_to_exclude = strIO_from_tens_T(mod_dgp.Y_T) < 1e-3 

mse_dict = filt_err(mod_dgp, mod_filt, phi_to_exclude, suffix=k_filt, prefix=bin_or_w)

# log plots that can be useful for quick visual diagnostic 
i_plot = torch.where(~splitVec(phi_to_exclude)[0])[0][2]

mod_filt.plot_phi_T(i=i_plot, fig_ax= mod_dgp.plot_phi_T(i=i_plot))[1]

mod_filt.identify_sequence_phi_T_beta_const()

if mod_dgp.X_T is not None:
    fig = plt.figure() 
    plt.plot(mod_dgp.X_T[0,0,:,:].T, figure=fig)
    
if mod_dgp.any_beta_tv():
    plot_dgp_fig_ax = mod_dgp.plot_beta_T()
    plot_dgp_fig_ax[0]
if mod_filt.beta_T is not None:
    if mod_filt.any_beta_tv():
        mod_filt.plot_beta_T(fig_ax=plot_dgp_fig_ax)[0]




