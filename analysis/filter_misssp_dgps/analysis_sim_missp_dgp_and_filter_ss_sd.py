#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Tuesday August 10th 2021

"""


# %% import packages
from pathlib import Path
import importlib
import dynwgraphs
from dynwgraphs.utils.tensortools import splitVec, strIO_from_tens_T
from dynwgraphs.dirGraphs1_dynNets import dirBin1_sequence_ss, dirBin1_SD, dirSpW1_SD, dirSpW1_sequence_ss
from dynwgraphs.utils.dgps import get_mod_and_par
import mlflow
from joblib import Parallel
from torch import nn
import logging
import click
from ddg_utils.mlflow import _get_and_set_experiment, check_test_exp, get_df_exp, uri_to_path, dict_from_run
from ddg_utils import drop_keys
from mlflow.tracking.client import MlflowClient
import pandas as pd
from run_sim_missp_dgp_and_filter_ss_sd import get_filt_mod
import torch
logger = logging.getLogger(__name__)
importlib.reload(dynwgraphs)


# %%
experiment = _get_and_set_experiment("filter missp dgp")

df = get_df_exp(experiment)
dgp_set = (1, 'one', False)
filt_set = (0, 'one', False)

df_filt = df[(df["ext_reg_bin_dgp_set"] == f"{dgp_set}") & (df["ext_reg_bin_filt_set"] == f"{filt_set}")]

row_run = df_filt[df_filt.status == "FINISHED" ].iloc[0]

load_path = Path(uri_to_path(row_run["artifact_uri"]))
bin_or_w = "bin"
Y_T, X_T = torch.load(open(load_path / "dgp" / "obs_T_dgp.pt", "rb"))

dgp_dict = drop_keys(eval(row_run["dgp_bin"]), ["n_ext_reg", "type_dgp_phi", "type_dgp_beta"])
mod_dgp = get_filt_mod(bin_or_w, Y_T, X_T, dgp_dict)["ss"]
mod_dgp.load_par(str(load_path / "dgp"))

filt_dict =  eval(row_run["filt_bin"])
mod_filt = get_filt_mod(bin_or_w, Y_T, X_T, filt_dict)["sd"]
mod_filt.load_par(str(load_path))
phi_to_exclude = strIO_from_tens_T(Y_T) < 1e-3 
i = torch.where(~splitVec(phi_to_exclude)[0])[0][0]

i=18
mod_filt.plot_phi_T(i=i, fig_ax=mod_dgp.plot_phi_T(i=i))

mod_filt.un2re_A_par(mod_filt.sd_stat_par_un_phi["A"]).argmax()
mod_filt.plot_phi_T()
# %%

from matplotlib import pyplot as plt
import numpy as np
x_T = X_T[0, 0, :].T.numpy()
plt.plot(x_T)

phi_T, _, _ = mod_filt.get_seq_latent_par()
phi_i_T, phi_o_T = splitVec(phi_T)

tmp = phi_i_T - phi_i_T.mean(0)

plt.plot(tmp.T)

plt.plot(phi_i_T[0,:])
corr = np.corrcoef(np.concatenate((x_T.T, phi_T)))
corr = np.corrcoef())

tmp = np.concatenate((phi_T[14,:].unsqueeze(dim=1).T, phi_T))

np.corrcoef(tmp)[0,14]

corr[11,0]


# %%
