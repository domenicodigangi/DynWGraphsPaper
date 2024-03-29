#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday July 31st 2021

"""



# %% import packages
import mlflow
import torch
import click
import logging
import tempfile
from pathlib import Path
from proj_utils.mlflow import _get_and_set_experiment, _get_or_run, uri_to_path, get_fold_namespace, check_and_tag_test_run
from dynwgraphs.utils.tensortools import splitVec, strIO_from_tens_T
from dynwgraphs.dirGraphs1_dynNets import dirBin1_SD, dirSpW1_SD, dirBin1_sequence_ss, dirSpW1_sequence_ss
import pickle
from eMid_data_utils import get_obs_and_regr_mat_eMid

logger = logging.getLogger(__name__)

# %%

@click.command()
@click.option("--max_opt_iter", default=11000, type=int)
@click.option("--opt_n", default="ADAMHD", type=str)
@click.option("--unit_meas", default=10000, type=float)
@click.option("--train_fract", default=3/4, type=float)
@click.option("--bin_or_w", default="bin", type=str)
@click.option("--regressor_name", default=["eonia"], type=str, multiple=True)

def estimate_multi_models(**kwargs):
    
    check_and_tag_test_run(kwargs["max_opt_iter"])

    #load data
    with mlflow.start_run(nested=True) as run:
        logger.info(kwargs)
        mlflow.log_params(kwargs)

        mod_0_run, filt_kwargs_0 = estimate_mod(str_size_beta_t="0", beta_tv=False, **kwargs)
    
        mod_1_run, filt_kwargs_1 = estimate_mod(**kwargs, str_size_beta_t="1", beta_tv=False, prev_mod={"filt_kwargs": filt_kwargs_0, "load_path": uri_to_path(mod_0_run.info.artifact_uri)})
        
        mod_2_run, filt_kwargs_2 = estimate_mod(**kwargs, str_size_beta_t="1", beta_tv=True, prev_mod={"filt_kwargs": filt_kwargs_1, "load_path": uri_to_path(mod_1_run.info.artifact_uri)})
        
        mod_3_run, filt_kwargs_3 = estimate_mod(**kwargs, str_size_beta_t="2N", beta_tv=False, prev_mod={"filt_kwargs": filt_kwargs_0, "load_path": uri_to_path(mod_0_run.info.artifact_uri)})
        
        mod_4_run, filt_kwargs_4 = estimate_mod(**kwargs, str_size_beta_t="2N", beta_tv=True, prev_mod={"filt_kwargs": filt_kwargs_3, "load_path": uri_to_path(mod_3_run.info.artifact_uri)})
        



def estimate_mod(**kwargs):
    pars_to_log = {k:v for k, v in kwargs.items() if not "prev_mod" in k}
    pars_to_log["regressor_name"] = "-".join(pars_to_log["regressor_name"])
    logger.info(pars_to_log)

    with mlflow.start_run(nested=True) as run:
        with tempfile.TemporaryDirectory() as tmpdirname:
            mlflow.log_params(pars_to_log)

            # temp fold
            tmp_fns = get_fold_namespace(tmpdirname, ["tb_logs"])
            
            load_and_log_data_run = _get_or_run("load_and_log_data", None, None)
            load_path = uri_to_path(load_and_log_data_run.info.artifact_uri)

            load_file = Path(load_path) / "data" / "eMid_data.pkl" 

            ld_data = pickle.load(open(load_file, "rb"))

            unit_meas = kwargs["unit_meas"]

            regr_list = kwargs["regressor_name"]
    
            Y_T, X_T, regr_list, net_stats = get_obs_and_regr_mat_eMid(ld_data, unit_meas, regr_list)

            N, _, T = Y_T.shape

            T_train =  int(kwargs["train_fract"] * T)
           
            if kwargs["str_size_beta_t"] == "0":
                filt_kwargs = {"T_train": T_train, "max_opt_iter": kwargs["max_opt_iter"], "opt_n": kwargs["opt_n"]}
            else:
                filt_kwargs = {"X_T": X_T, "beta_tv": kwargs["beta_tv"], "T_train": T_train}
                if kwargs["str_size_beta_t"] == "1":
                    filt_kwargs["size_beta_t"] = 1
                elif kwargs["str_size_beta_t"] == "N":
                    filt_kwargs["size_beta_t"] = N
                elif kwargs["str_size_beta_t"] == "2N":
                    filt_kwargs["size_beta_t"] = 2*N

            logger.info(f" start estimates {kwargs['bin_or_w']}")

            #estimate models and log parameters and hpar optimization
            if kwargs["bin_or_w"] == "bin":
                mod_ss = dirBin1_sequence_ss(Y_T, **filt_kwargs)
                mod_sd = dirBin1_SD(Y_T, **filt_kwargs)
                if "prev_mod" in kwargs.keys():
                    prev_filt_kwargs = kwargs["prev_mod"]["filt_kwargs"]
                    prev_mod_sd = dirBin1_SD(Y_T, **prev_filt_kwargs)
                    prev_mod_sd.load_par(kwargs["prev_mod"]["load_path"])
                    mod_sd.init_par_from_lower_model(prev_mod_sd)
                    
            elif kwargs["bin_or_w"] == "w":
                mod_ss = dirSpW1_sequence_ss(Y_T, **filt_kwargs)
                mod_sd = dirSpW1_SD(Y_T, **filt_kwargs)
                if "prev_mod" in kwargs.keys():
                    prev_filt_kwargs = kwargs["prev_mod"]["filt_kwargs"]
                    prev_mod_sd = dirSpW1_SD(Y_T, **prev_filt_kwargs)
                    prev_mod_sd.load_par(kwargs["prev_mod"]["load_path"])
                    mod_sd.init_par_from_lower_model(prev_mod_sd)
            else:
                raise
            
            filt_models = {"sd": mod_sd, "ss": mod_ss}

            in_sample_fit = {}
            out_sample_fit = {}

            for k_filt, mod in filt_models.items():
                _, h_par_opt, opt_metrics = mod.estimate(tb_save_fold=tmp_fns.tb_logs)

                mlflow.log_params({f"{k_filt}_{key}": val for key, val in h_par_opt.items() if key != "X_T"})
                mlflow.log_params({f"{k_filt}_{key}": val for key, val in mod.get_info_dict().items() if (key not in h_par_opt.keys()) and ( key != "X_T")})
                mlflow.log_metrics({f"{k_filt}_{key}": val for key, val in opt_metrics.items()})

                mod.save_parameters(save_path=tmp_fns.main)
                

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
    estimate_multi_models()

