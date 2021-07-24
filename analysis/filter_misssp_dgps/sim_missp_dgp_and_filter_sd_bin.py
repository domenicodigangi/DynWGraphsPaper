

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday July 10th 2021

"""

#%% import packages
from pathlib import Path
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import dynwgraphs
from dynwgraphs.utils.dgps import get_dgp_model
from dynwgraphs.dirGraphs1_dynNets import  dirBin1_sequence_ss, dirBin1_SD
import importlib
import tempfile
from pathlib import Path

importlib.reload(dynwgraphs)
import mlflow
from dynwgraphs.utils.dgps import get_test_w_seq
from joblib import Parallel, delayed
from torch import nn


#%%
def filt_err(mod_dgp, mod_filt, suffix=""):
    
    loss_fun = nn.MSELoss()
    phi_T_filt = mod_filt.par_list_to_matrix_T(mod_filt.phi_T)
    phi_T_dgp = mod_dgp.par_list_to_matrix_T(mod_dgp.phi_T)
    mse_phi = loss_fun(phi_T_dgp, phi_T_filt).item()

    if mod_dgp.beta_T is not None:
        beta_T_filt = mod_filt.par_list_to_matrix_T(mod_filt.beta_T)
        beta_T_dgp = mod_dgp.par_list_to_matrix_T(mod_dgp.beta_T)
        mse_beta = loss_fun(beta_T_dgp, beta_T_filt).item()
    else:
        mse_beta = 0
    
    if mod_dgp.dist_par_un_T is not None:
        dist_par_un_T_filt = mod_filt.par_list_to_matrix_T(mod_filt.dist_par_un_T)
        dist_par_un_T_dgp = mod_dgp.par_list_to_matrix_T(mod_dgp.dist_par_un_T)
        mse_dist_par_un = loss_fun(dist_par_un_T_dgp, dist_par_un_T_filt).item()
    else:
        mse_dist_par_un = 0
    
    mse_dict = {f"mse_phi_{suffix}":mse_phi, f"mse_beta_{suffix}":mse_beta, f"mse_dist_par_un_{suffix}":mse_dist_par_un}

    return mse_dict


def sample_estimate_and_log(run_par_dict, run_data_dict, experiment):
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:

        dgp_par = run_par_dict["dgp_par"]
        
        # save files in temp folder, then log them as artifacts in mlflow and delete temp fold

        with tempfile.TemporaryDirectory() as tmpdirname:

            # set artifacts folders and subfolders
            tmp_path = Path(tmpdirname)
            dgp_fold = tmp_path / "dgp"
            dgp_fold.mkdir(exist_ok=True)
            tb_fold = tmp_path / "tb_logs"
            tb_fold.mkdir(exist_ok=True)

            mlflow.log_params({f"{'dgp_'}{key}": val for key, val in dgp_par.items()})

            mod_dgp = get_dgp_model(dgp_par, Y_reference=run_data_dict["Y_reference"])

            filt_kwargs = {"size_beta_t":1, "X_T" : mod_dgp.X_T, "beta_tv":np.array([True])}


            # save dgp data
            torch.save(run_data_dict["Y_reference"], dgp_fold / "Y_reference.pkl")
            torch.save((mod_dgp.get_Y_T_to_save(),mod_dgp.X_T), dgp_fold / "obs_T_dgp.pkl")
            mod_dgp.save_parameters(save_path=dgp_fold)


            #estimate models and log parameters and hpar optimization
          
            mod_sd = dirBin1_SD(mod_dgp.Y_T, **filt_kwargs)
            mod_ss = dirBin1_sequence_ss(mod_dgp.Y_T, **filt_kwargs)

            filt_models = {"sd":mod_sd, "ss":mod_ss}

            for k_mod, mod in filt_models.items():
                mod.opt_options["max_opt_iter"] = run_par_dict["max_opt_iter"]
                _, h_par_opt = mod.estimate(tb_save_fold=tb_fold)

                mlflow.log_params({f"{k_mod}{key}": val for key, val in h_par_opt.items()})
                mlflow.log_params({f"{k_mod}{key}": val for key, val in mod_sd.get_info_dict().items()})

                mod.save_parameters(save_path=tmp_path)
            


            # compute mse for each model and log it 

            mse_dict_sd = filt_err(mod_dgp, mod_sd, suffix="sd")
            mse_dict_ss = filt_err(mod_dgp, mod_ss, suffix="ss")

            mlflow.log_metrics(mse_dict_sd) 
            mlflow.log_metrics(mse_dict_ss) 
          
            # log plots that can be useful for quick visual diagnostic 
            mlflow.log_figure(mod_sd.plot_phi_T()[0], f"fig/sd_filt_all.png")
            mlflow.log_figure(mod_ss.plot_phi_T()[0], f"fig/ss_filt_all.png")
            mlflow.log_figure(mod_dgp.plot_phi_T()[0], f"fig/dgp_all.png")

            i=11
            mlflow.log_figure(mod_sd.plot_phi_T(i=i, fig_ax= mod_dgp.plot_phi_T(i=i))[0], f"fig/sd_filt_phi_ind_{i}.png")
            
            mlflow.log_figure(mod_ss.plot_phi_T(i=i, fig_ax= mod_dgp.plot_phi_T(i=i))[0], f"fig/ss_filt_phi_ind_{i}.png")
            
            
            mlflow.log_figure(mod_sd.plot_beta_T(fig_ax= mod_dgp.plot_beta_T())[0], f"fig/sd_filt_beta_T.png")
            mlflow.log_figure(mod_ss.plot_beta_T(fig_ax= mod_dgp.plot_beta_T())[0], f"fig/ss_filt_beta_T.png")

            
            # log all files and sub-folders in temp fold as artifacts            
            mlflow.log_artifacts(tmp_path)


#%% set experiment
experiment_name = "binary directed sd filter missp dgp"
experiment = mlflow.get_experiment_by_name(experiment_name)
mlflow.set_experiment(experiment_name)

print("Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

#%% Run
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Simulate missp dgp and estimate sd filter."
    )

    parser.add_argument("--n_sim", help="Number of simulations", type=int)
    parser.add_argument("--max_opt_iter", help="max number of opt iter", type=int, default = 3500)

    args = parser.parse_args()


    # define run's parameters

    dgp_phi = {"type":"AR", "B":0.98, "sigma":0.1, "is_tv":True}
    dgp_beta = {"type":"AR", "B":0.98, "sigma":0.1, "is_tv":[True], "X_type":"uniform", "size_beta_t": 1, "beta_0":torch.ones(1,1)}
    dgp_dist_par = None

    dgp_par = {"N" : 50, "T" : 100, "model":"dirBin1", "dgp_phi" : dgp_phi, "dgp_beta":dgp_beta, "dgp_dist_par":dgp_dist_par}

    Y_T_test, _, _ =  get_test_w_seq(avg_weight=1e3)
    run_data_dict = {"Y_reference": (Y_T_test[:dgp_par["N"],:dgp_par["N"],0]>0).float()}

    run_par_dict = {"dgp_par": dgp_par, "max_opt_iter": args.max_opt_iter}


    Parallel(n_jobs=4)(delayed(sample_estimate_and_log)(run_par_dict, run_data_dict, experiment) for _ in range(args.n_sim))


# %%


run_i = mlflow.get_run("314f61064d714bb8ae9545bd4fdc1812")

load_path = Path(run_i.info.artifact_uri[8:] + "/")
Y_T_dgp = torch.load(load_path / "Y_T_dgp.pkl").float()

mod_sd = dirBin1_SD(Y_T_dgp)
mod_sd.load_par(load_path)

mod_sd.plot_sd_par()
mod_sd.plot_beta_T()


mod_ss = dirBin1_sequence_ss(Y_T_dgp)
mod_ss.load_par(load_path)


mod_dgp = dirBin1_sequence_ss(Y_T_dgp)
mod_dgp.load_par(load_path /"dgp")


mod_sd.plot_phi_T()
mod_ss.plot_phi_T()
mod_dgp.plot_phi_T()




# %%
