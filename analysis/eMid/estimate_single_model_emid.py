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
from eMid_data_utils import get_obs_and_regr_mat_eMid
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# %%
        
@click.command()
@click.option("--size_beta_t", type=str, default="0")
@click.option("--bin_or_w", type=str, default="bin")
@click.option("--beta_tv", type=float, default=0)
@click.option("--max_opt_iter", default=21, type=int)
@click.option("--unit_meas", default=10000, type=float)
@click.option("--train_fract", default=3/4, type=float)
@click.option("--regressor_name", default="eonia", type=str)
@click.option("--prev_mod_art_uri", default="none://", type=str)
@click.option("--opt_n", default="ADAM", type=str)


def estimate_single_model_emid(**kwargs):
    kwargs["beta_tv"] = bool(kwargs["beta_tv"])
    logger.info(kwargs)
    with mlflow.start_run(nested=True) as run:
        check_and_tag_test_run(kwargs)
        with tempfile.TemporaryDirectory() as tmpdirname:
            
            # temp fold
            tmp_fns = get_fold_namespace(tmpdirname, ["tb_logs"])
            
            load_and_log_data_run = _get_or_run("load_and_log_data", None, None)
            load_path = uri_to_path(load_and_log_data_run.info.artifact_uri)

            load_file = Path(load_path) / "data" / "eMid_data.pkl" 

            ld_data = pickle.load(open(load_file, "rb"))

            unit_meas = kwargs["unit_meas"]

            regr_list = kwargs["regressor_name"].replace(" ", "_").split("_")
    
            Y_T, X_T, regr_list, net_stats = get_obs_and_regr_mat_eMid(ld_data, unit_meas, regr_list)

            N, _, T = Y_T.shape

            T_train = int(kwargs["train_fract"] * T)
           
            filt_kwargs = {"T_train": T_train, "max_opt_iter": kwargs["max_opt_iter"], "opt_n": kwargs["opt_n"], "size_beta_t": kwargs["size_beta_t"]}
            
            if kwargs["size_beta_t"] not in ["0", 0, None]:
                filt_kwargs["X_T"] =  X_T
                filt_kwargs["beta_tv"] = kwargs["beta_tv"]

            
            #estimate models and log parameters and hpar optimization
            mod_sd = get_gen_fit_mod(kwargs["bin_or_w"], "sd", Y_T, **filt_kwargs)
            mod_ss = get_gen_fit_mod(kwargs["bin_or_w"], "ss", Y_T, **filt_kwargs)
            if urlparse(kwargs["prev_mod_art_uri"]).scheme != "none":
                load_path = uri_to_path(kwargs["prev_mod_art_uri"])
                logger.info(f"loading data from previous model path: {load_path}")
                prev_filt_kwargs = pickle.load(open(str(Path(load_path) / "filt_kwargs_dict.pkl"), "rb"))
                prev_mod_sd = get_gen_fit_mod(kwargs["bin_or_w"], "sd", Y_T, **prev_filt_kwargs)
                mod_sd.init_par_from_lower_model(prev_mod_sd) 
            else:
                logger.info("Not loading any model as starting point")

            filt_models = {"sd": mod_sd, "ss": mod_ss}

            in_sample_fit = {}
            out_sample_fit = {}

            for k_filt, mod in filt_models.items():
                logger.info(f" Start estimate {k_filt}, {mod.opt_options}")
                _, h_par_opt, opt_metrics = mod.estimate(tb_save_fold=tmp_fns.tb_logs)

                mlflow.log_params({f"{k_filt}_{key}": val for key, val in h_par_opt.items() if key != "X_T"})
                mlflow.log_params({f"{k_filt}_{key}": val for key, val in mod.get_info_dict().items() if (key not in h_par_opt.keys()) and ( key != "X_T")})
                mlflow.log_metrics({f"{k_filt}_{key}": val for key, val in opt_metrics.items()})

                mod.save_parameters(save_path=tmp_fns.main)
                
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

            mlflow.log_metrics(in_sample_fit) 
            mlflow.log_metrics(out_sample_fit) 
            
        
            # log all files and sub-folders in temp fold as artifacts            
            mlflow.log_artifacts(tmp_fns.main)

    return run, filt_kwargs

# %% Run
if __name__ == "__main__":
    estimate_single_model_emid()

