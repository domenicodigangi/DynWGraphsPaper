#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Tuesday August 24th 2021

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
from urllib.parse import urlparse
import sys
sys.path.insert(0,'..')
from eMid_data_utils import get_data_from_data_run

logger = logging.getLogger(__name__)

# %%
        
@click.command()
@click.option("--bin_or_w", type=str, default="bin")
@click.option("--size_beta_t", type=str, default="0")
@click.option("--beta_tv", type=float, default=0.0)
@click.option("--size_phi_t", type=str, default="2N")
@click.option("--phi_tv", type=float, default=1)
@click.option("--max_opt_iter", default=3000, type=int)
@click.option("--unit_meas", default=10000, type=float)
@click.option("--t_0", default=0, type=int)
@click.option("--t_train", default=0, type=int)
@click.option("--train_fract", default=0.0, type=float)
@click.option("--regressor_name", default="eonia", type=str)
@click.option("--prev_mod_art_uri", default="none://", type=str)
@click.option("--opt_n", default="ADAMHD", type=str)
@click.option("--init_sd_type", default="est_ss_before", type=str)
@click.option("--filter_ss_or_sd", type=str, default="sd")



def estimate_single_model_emid(**kwargs):
    if kwargs["bin_or_w"] == "bin":
        kwargs["avoid_ovflw_fun_flag"] = False
    elif kwargs["bin_or_w"] == "w":
        kwargs["avoid_ovflw_fun_flag"] = False
   
  
    with mlflow.start_run(nested=True) as run:
        check_and_tag_test_run(kwargs)
        mlflow.set_tag("is_estimate_run", "y")
        with tempfile.TemporaryDirectory() as tmpdirname:
            
            # temp fold
            tmp_fns = get_fold_namespace(tmpdirname, ["tb_logs"])
            
            Y_T, X_T, regr_list, net_stats = get_data_from_data_run(float(kwargs["unit_meas"]), kwargs["regressor_name"], kwargs["t_0"])

            N, _, T = Y_T.shape

            if kwargs["train_fract"] > 0:
                t_train = int(kwargs["train_fract"] * T)
                kwargs["t_train"] = t_train
            else:
                if kwargs["t_train"] <= 0:
                    kwargs["t_train"] = T + kwargs["t_train"]

                t_train = kwargs["t_train"] 

            logger.info(f"estimate_single_model_emid ==== {kwargs}")
            mlflow.log_params(kwargs)
           

            filt_kwargs = {"T_train": t_train, "max_opt_iter": kwargs["max_opt_iter"], "opt_n": kwargs["opt_n"], "size_phi_t": kwargs["size_phi_t"], "phi_tv": kwargs["phi_tv"], "size_beta_t": kwargs["size_beta_t"], "avoid_ovflw_fun_flag": kwargs["avoid_ovflw_fun_flag"]}
            
            if kwargs["size_beta_t"] not in ["0", 0, None]:
                filt_kwargs["X_T"] =  X_T
                filt_kwargs["beta_tv"] = kwargs["beta_tv"]

            
            #estimate models and log parameters and hpar optimization
            if kwargs["filter_ss_or_sd"] == "sd":
                mod = get_gen_fit_mod(kwargs["bin_or_w"], kwargs["filter_ss_or_sd"], Y_T, init_sd_type=kwargs["init_sd_type"], **filt_kwargs)
            elif kwargs["filter_ss_or_sd"] == "ss":
                mod = get_gen_fit_mod(kwargs["bin_or_w"], kwargs["filter_ss_or_sd"], Y_T, **filt_kwargs)
            else:
                raise Exception("filter type not recognized")

            
            if urlparse(kwargs["prev_mod_art_uri"]).scheme != "none":
                if kwargs["filter_ss_or_sd"] == "sd":
                    load_path = uri_to_path(kwargs["prev_mod_art_uri"])
                    logger.info(f"loading data from previous model path: {load_path}")
                    prev_filt_kwargs = pickle.load(open(str(Path(load_path) / "filt_kwargs_dict.pkl"), "rb"))
                    prev_mod_sd = get_gen_fit_mod(kwargs["bin_or_w"], "sd", Y_T, **prev_filt_kwargs)
                    mod.init_par_from_lower_model(prev_mod_sd) 
            else:
                logger.info("Not loading any model as starting point")

            
            # filt_models = {kwargs["filter_ss_or_sd"]: mod}

            k_filt = kwargs["filter_ss_or_sd"]

            in_sample_fit = {}
            out_sample_fit = {}

            logger.info(f" Start estimate {k_filt}, {mod.opt_options}")

            try:
                _, h_par_opt, opt_metrics = mod.estimate(tb_save_fold=tmp_fns.tb_logs)
            except:
                if k_filt == "sd":
                    logger.error(f"error in stimate of {k_filt}, par {drop_keys(filt_kwargs, ['X_T'])}, stopping")
                    mlflow.log_artifacts(tmp_fns.main)
                    raise Exception("Error in estimate")
                else:
                    logger.error(f"error in stimate of {k_filt}, par {drop_keys(filt_kwargs, ['X_T'])}, continuing")

            mlflow.log_params({f"opt_info_{key}": val for key, val in h_par_opt.items() if key != "X_T"})
            mlflow.log_params({f"mod_info_{key}": val for key, val in mod.get_info_dict().items() if (key not in h_par_opt.keys()) and ( key != "X_T")})
            mlflow.log_metrics({f"{key}": val for key, val in opt_metrics.items()})

            mod.save_parameters(save_path=tmp_fns.main)
            
            with open(tmp_fns.main / "filt_kwargs_dict.pkl", 'wb') as f:
                pickle.dump(filt_kwargs, f)
            

            # in sample gof measures 
            in_sample_fit[f"log_like_T"] = mod.loglike_seq_T().item()
            in_sample_fit[f"BIC"] = mod.get_BIC().item()
            
            try:
                for k, v in mod.out_of_sample_eval().items():
                    out_sample_fit[f"out_of_sample_{k}"] = v 
            except:
                logger.error("Error in computing out of sample performance")

            try:
                # log plots that can be useful for quick visual diagnostic
                mlflow.log_figure(mod.plot_phi_T()[0], f"fig/{kwargs['bin_or_w']}_{k_filt}_filt_all.png")

                mlflow.log_figure(mod.plot_phi_T(i=i)[0], f"fig/{kwargs['bin_or_w']}_{k_filt}_filt_phi_ind_{i}.png")
                
                if mod.any_beta_tv():
                    mlflow.log_figure(mod.plot_beta_T()[0], f"fig/{kwargs['bin_or_w']}_{k_filt}_filt_beta_T.png")
            except:

                logger.error("Error in producing or saving figures")

            mlflow.log_metrics(in_sample_fit) 
            mlflow.log_metrics(out_sample_fit) 
            
        
            # log all files and sub-folders in temp fold as artifacts            
            mlflow.log_artifacts(tmp_fns.main)

    return run, filt_kwargs

# %% Run
if __name__ == "__main__":
    estimate_single_model_emid()

