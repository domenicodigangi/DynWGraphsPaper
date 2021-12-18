#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday August 25th 2021

"""

# %% import packages
import mlflow
import torch
import click
import logging
import tempfile
from pathlib import Path
from ddg_utils import drop_keys
from ddg_utils.mlflow import _get_and_set_experiment, _get_or_run, uri_to_path, get_fold_namespace, check_and_tag_test_run
from dynwgraphs.utils.tensortools import splitVec, strIO_from_tens_T
from dynwgraphs.dirGraphs1_dynNets import dirBin1_SD, dirSpW1_SD, dirBin1_sequence_ss, dirSpW1_sequence_ss, get_gen_fit_mod
import pickle
from eMid_data_utils import get_obs_and_regr_mat_eMid, get_data_from_data_run
from urllib.parse import urlparse
import copy

logger = logging.getLogger(__name__)

# %%

kwargs = {}
kwargs["size_phi_t"] = "2N"
kwargs["phi_tv"] = 1.0
kwargs["size_beta_t"] = "one"
kwargs["bin_or_w"] = "bin"
kwargs["beta_tv"] = True
kwargs["max_opt_iter"] = 3000
kwargs["unit_meas"] = 10000
kwargs["train_fract"] = 0.8
kwargs["regressor_name"] = "eonia"
kwargs["prev_mod_art_uri"] = "none://"
kwargs["opt_n"] = "ADAMHD"
kwargs["init_sd_type"] = "unc_mean"
kwargs["avoid_ovflw_fun_flag"] = False

tmp_fns = get_fold_namespace(".dev_test_data", ["tb_logs"])
_get_and_set_experiment("dev test")


Y_T, X_T, regr_list, net_stats = get_data_from_data_run(float(kwargs["unit_meas"]), kwargs["regressor_name"] )



N, _, T = Y_T.shape

T_train = int(kwargs["train_fract"] * T)

filt_kwargs = {"T_train": T_train, "max_opt_iter": kwargs["max_opt_iter"], "opt_n": kwargs["opt_n"], "size_phi_t": kwargs["size_phi_t"], "size_beta_t": kwargs["size_beta_t"], "avoid_ovflw_fun_flag": kwargs["avoid_ovflw_fun_flag"]}

if kwargs["size_beta_t"] not in ["0", 0, None]:
    filt_kwargs["X_T"] =  X_T
    filt_kwargs["beta_tv"] = kwargs["beta_tv"]


#estimate models and log parameters and hpar optimization
mod_sd = get_gen_fit_mod(kwargs["bin_or_w"], "sd", Y_T, init_sd_type=kwargs["init_sd_type"], **filt_kwargs)
mod_ss = get_gen_fit_mod(kwargs["bin_or_w"], "ss", Y_T, **filt_kwargs)
if urlparse(kwargs["prev_mod_art_uri"]).scheme != "none":
    load_path = uri_to_path(kwargs["prev_mod_art_uri"])
    logger.info(f"loading data from previous model path: {load_path}")
    prev_filt_kwargs = pickle.load(open(str(Path(load_path) / "filt_kwargs_dict.pkl"), "rb"))
    prev_mod_sd = get_gen_fit_mod(kwargs["bin_or_w"], "sd", Y_T, **prev_filt_kwargs)
    mod_sd.init_par_from_lower_model(prev_mod_sd) 
else:
    logger.info("Not loading any model as starting point")


mod = mod_sd
#%%

# mod_sd.init_par_from_lower_model(prev_mod_sd) 

mod.mod_for_init.opt_options_ss_seq["max_opt_iter"] = 5
_, h_par_opt, opt_metrics = mod.estimate(tb_save_fold=tmp_fns.tb_logs, log_interval=1, max_opt_iter=20)

s_T_beta = mod.get_score_T_train(rescaled=True)["beta"]
from matplotlib import pyplot as plt
plt.plot(s_T_beta)
mod.beta_T
mod.sd_stat_par_un_beta["B"]
mod.phi_T
# prev_mod_sd=mod_sd



mod_sd.get_score_T_train()["beta"]
#%% 

#%%
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import numpy as np
import matplotlib.pyplot as plt
autocorrelation = np.correlate(s_T, s_T, mode="full")
plot_acf(s_T)

s_T = mod_stat.get_score_T_train()["phi"].detach().numpy()
acorr_1_phi = torch.tensor([acf(s_T[n, :], nlags=1)[1] for n in range(s_T.shape[0])])

A_vec = mod.un2re_A_par(mod.sd_stat_par_un_phi["A"]).detach().numpy()
plt.hist(acorr_1_phi)
plt.hist(A_vec)
plt.scatter(acorr_1_phi, np.log(A_vec))


#%% 

mod.roll_sd_filt_train()
mod.beta_T

mod.mod_stat.opt_options["max_opt_iter"] = 2000

import matplotlib.pyplot as plt
# mod.init_sd_type = "est_ss_before"
mod.init_all_stat_par()

mod.init_static_sd_from_obs()
mod.sd_stat_par_un_phi["init_val"]
phi_T, _, _ = mod.get_time_series_latent_par()
phi_T is None

mod.roll_sd_filt(mod.T_all)




mod.loglike_seq_T()
mod.plot_phi_T()
phi_T, _, _ = mod.get_time_series_latent_par()



mod.opt_options_sd["max_opt_iter"] = 400
mod.plot_phi_T()

mod.un2re_A_par(mod.sd_stat_par_un_phi["A"]).max()
mod.un2re_B_par(mod.sd_stat_par_un_phi["B"]).max()
mod.sd_stat_par_un_phi["init_val"]

mod.par_l_to_opt

t=0
# mod.avoid_ovflw_fun_flag = True
# mod.rescale_SD = True

mod_ss.estimate_ss_t(t, True, False, False)

# mod.phi_T[1] = mod_ss.phi_T[1]
mod.phi_T[t] 
mod_ss.phi_T[t]


mod.phi_T[t] = mod_ss.phi_T[t]
mod.score_t(t)["phi"]


Y_t, X_t = mod.get_obs_t(t)

phi_t, _, beta_t = mod.get_par_t(t)






#%%%%%%%%%%%%%%%%%%%%
# To Do: calvori test










# %%
