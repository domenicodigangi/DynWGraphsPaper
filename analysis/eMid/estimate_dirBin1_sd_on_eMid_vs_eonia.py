#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday July 31st 2021

"""



#%% import packages
import mlflow
import torch
import click
import logging
logger = logging.getLogger(__name__)
from torch.functional import split
import tempfile
import datetime
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from ddg_utils.mlflow import _get_and_set_experiment, _get_or_run, uri_to_path
from dynwgraphs.utils.tensortools import tens, splitVec, strIO_from_tens_T
from dynwgraphs.dirGraphs1_dynNets import  dirBin1_sequence_ss, dirBin1_SD, dirSpW1_sequence_ss, dirSpW1_SD

import pandas as pd



#%%

@click.command()
@click.option("--experiment_name", type=str, default="application_eMid" )
@click.option("--max_opt_iter", default=10000, type=int)
@click.option("--size_beta_t", default="0", type=str)
@click.option("--unit_meas", default=10000, type=float)
@click.option("--train_fract", default=3/4, type=float)
@click.option("--bin_or_w", default = "bin", type=str)


def estimate_mod(**kwargs):

    experiment = _get_and_set_experiment(f"{kwargs['experiment_name']}")

    with mlflow.start_run() as run:
        with tempfile.TemporaryDirectory() as tmpdirname:

            mlflow.log_params(kwargs)

            # set artifacts folders and subfolders
            tmp_path = Path(tmpdirname)
            tb_fold = tmp_path / "tb_logs"
            tb_fold.mkdir(exist_ok=True)

            load_and_log_data_run = _get_or_run("load_and_log_data", {}, None)
            load_path = uri_to_path(load_and_log_data_run.info.artifact_uri)

            load_file = Path(load_path) / "data" / "eMid_numpy.npz" 

            ld_data = np.load(load_file, allow_pickle=True)

            eMid_w_T, all_dates, eonia_w, nodes = ld_data["arr_0"], ld_data["arr_1"], ld_data["arr_2"], ld_data["arr_3"]

            unit_meas = kwargs["unit_meas"]
            Y_T = tens(eMid_w_T / unit_meas) 
            N, _, T = Y_T.shape
            X_T = tens(np.tile(eonia_w.T, (N, N, 1, 1)))

            T_train =  int(kwargs["train_fract"] * T)
           
            if kwargs["size_beta_t"] == "0":
                filt_kwargs = {}
            else:
                filt_kwargs = {"size_beta_t":kwargs["size_beta_t"], "X_T" : X_T, "beta_tv":kwargs["beta_tv"]}
    
            logger.info(f" start estimates {kwargs['bin_or_w']}")

            #estimate models and log parameters and hpar optimization
            if kwargs["bin_or_w"] == "bin":
                mod_sd = dirBin1_SD(Y_T, **filt_kwargs)
                mod_ss = dirBin1_sequence_ss(Y_T, **filt_kwargs)
            elif kwargs["bin_or_w"] == "w":
                mod_sd = dirSpW1_SD(Y_T, **filt_kwargs)
                mod_ss = dirSpW1_sequence_ss(Y_T, **filt_kwargs)
            else:
                raise
                
            filt_models = {"sd":mod_sd, "ss":mod_ss}

            in_sample_fit = {}
            out_sample_fit = {}

            for k_filt, mod in filt_models.items():
                mod.opt_options["max_opt_iter"] = kwargs["max_opt_iter"]

                _, h_par_opt = mod.estimate(tb_save_fold=tb_fold)

                mlflow.log_params({f"{kwargs['bin_or_w']}_{k_filt}_{key}": val for key, val in h_par_opt.items()})
                mlflow.log_params({f"{kwargs['bin_or_w']}_{k_filt}_{key}": val for key, val in mod.get_info_dict().items() if key not in h_par_opt.keys()})

                mod.save_parameters(save_path=tmp_path)
                

                # compute mse for each model and log it 
                in_sample_fit[f"{k_filt}_log_like_T"] = mod.log_like_T()
                in_sample_fit[f"{k_filt}_BIC"] = mod.log_like_T()
                
                out_sample_fit[f"{k_filt}_out_of_sample"] = mod.out_sample_eval()


                # log plots that can be useful for quick visual diagnostic 
                mlflow.log_figure(mod.plot_phi_T()[0], f"fig/{kwargs['bin_or_w']}_{k_filt}_filt_all.png")

                phi_to_exclude = strIO_from_tens_T(mod.Y_T) < 1e-3 
                i=torch.where(~splitVec(phi_to_exclude)[0])[0][0]

                mlflow.log_figure(mod.plot_phi_T(i=i)[0], f"fig/{kwargs['bin_or_w']}_{k_filt}_filt_phi_ind_{i}.png")
                
                mlflow.log_figure(mod_ss.plot_phi_T(i=i)[0], f"fig/{kwargs['bin_or_w']}_ss_filt_phi_ind_{i}.png")
                
                if mod.any_beta_tv():
                    mlflow.log_figure(mod.plot_beta_T()[0], f"fig/{kwargs['bin_or_w']}_{k_filt}_filt_beta_T.png")

            mlflow.log_metrics(in_sample_fit) 
            mlflow.log_metrics(out_sample_fit) 
            
        
            # log all files and sub-folders in temp fold as artifacts            
            mlflow.log_artifacts(tmp_path)



#%% Run
if __name__ == "__main__":
    estimate_mod()

# %%


# run_i = mlflow.get_run("314f61064d714bb8ae9545bd4fdc1812")

# load_path = Path(run_i.info.artifact_uri[8:] + "/")
# Y_T_dgp = torch.load(load_path / "Y_T_dgp.pkl").float()

# mod_sd = dirBin1_SD(Y_T_dgp)
# mod_sd.load_par(load_path)

# mod_sd.plot_sd_par()
# mod_sd.plot_beta_T()


# mod_ss = dirBin1_sequence_ss(Y_T_dgp)
# mod_ss.load_par(load_path)


# mod_dgp = dirBin1_sequence_ss(Y_T_dgp)
# mod_dgp.load_par(load_path /"dgp")


# mod_sd.plot_phi_T()
# mod_ss.plot_phi_T()
# mod_dgp.plot_phi_T()




# %%



#%% Run
if __name__ == "__main__":
    load_and_log_data()