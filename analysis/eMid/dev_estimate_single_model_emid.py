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
kwargs["size_beta_t"] = "0"
kwargs["bin_or_w"] = "bin"
kwargs["beta_tv"] = False
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

filt_kwargs = {"T_train": T_train, "max_opt_iter": kwargs["max_opt_iter"], "opt_n": kwargs["opt_n"], "size_beta_t": kwargs["size_beta_t"], "avoid_ovflw_fun_flag": kwargs["avoid_ovflw_fun_flag"]}

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

_, h_par_opt, opt_metrics = mod.estimate(tb_save_fold=tmp_fns.tb_logs, log_interval=100, max_opt_iter=200)
mod_sd.beta_T
# prev_mod_sd=mod_sd

#%% 

mod_stat = mod.mod_for_init


mod.N
mod.plot_phi_T(i=102)
mod.plot_phi_T()
mod.sd_stat_par_un_phi["A"][233]



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




A_t = Y_t 

score_dict = {}
if  mod.any_beta_tv() :
    
    # compute the score with AD using Autograd
    like_t = mod.loglike_t(Y_t, phi_t, beta=beta_t, X_t=X_t)

    if mod.any_beta_tv():
        score_dict["beta"] = grad(like_t, beta_t, create_graph=True)[0]

    if mod.rescale_SD:
        pass
        # raise "Rescaling not ready for beta and dist par"

if mod.phi_tv:

    exp_A = mod.exp_A(phi_t, beta=beta_t, X_t=X_t)

    tmp = A_t - exp_A

N**2-N
exp_A.sum()
phi_t

A_t.sum()

mod.exp_of_fit_plus_reg(phi_t, beta_t, X_t, ret_log=False, ret_all=False)
mod.invPiMat(phi_t, beta_t, X_t, ret_log=False)

phi_i, phi_o = splitVec(phi_t)
phi_i[0] + phi_o[1]

mod.get_phi_sum(phi_t)[1,0]




with open(tmp_fns.main / "filt_kwargs_dict.pkl", 'wb') as f:
    pickle.dump(filt_kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)
    

    # compute mse for each model and log it 
    in_sample_fit[f"{k_filt}_log_like_T"] = mod.loglike_seq_T().item()
    in_sample_fit[f"{k_filt}_BIC"] = mod.get_BIC().item()
    
    for k, v in mod.out_of_sample_eval().items():
        out_sample_fit[f"{k_filt}_out_of_sample_{k}"] = v 

    try:
        # log plots that can be useful for quick visual diagnostic
        mlflow.log_figure(mod.plot_phi_T()[0], f"fig/{kwargs['bin_or_w']}_{k_filt}_filt_all.png")

        phi_to_exclude = strIO_from_tens_T(mod.Y_T) < 1e-3 
        i=torch.where(~splitVec(phi_to_exclude)[0])[0][0]

        mlflow.log_figure(mod.plot_phi_T(i=i)[0], f"fig/{kwargs['bin_or_w']}_{k_filt}_filt_phi_ind_{i}.png")
        
        mlflow.log_figure(mod_ss.plot_phi_T(i=i)[0], f"fig/{kwargs['bin_or_w']}_ss_filt_phi_ind_{i}.png")
        
        if mod.any_beta_tv():
            mlflow.log_figure(mod.plot_beta_T()[0], f"fig/{kwargs['bin_or_w']}_{k_filt}_filt_beta_T.png")
    except:
        logger.error("Error in producing or saving figures")


# log all files and sub-folders in temp fold as artifacts            
mlflow.log_artifacts(tmp_fns.main)




#%%%%%%%%%%%%%%%%%%%%
# To Do: arrive at white sandwhich estimator


rescaled = False
selfo = mod_sd
T = selfo.T_train
list_par = ["phi"]

from torch.autograd import grad
from torch.autograd.functional import hessian


selfo.hess_t(0, True, False, False)

hess_T = selfo.get_hess_T(20, ["beta"])
s_T = selfo.get_score_T(20, ["beta"])

hess_T["beta"]
s_T["beta"]



# %%
