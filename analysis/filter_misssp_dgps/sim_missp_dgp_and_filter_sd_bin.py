

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
import matplotlib.pyplot as plt
import dynwgraphs
from dynwgraphs.utils.dgps import get_dgp_model
from dynwgraphs.dirGraphs1_dynNets import  dirBin1_sequence_ss, dirBin1_SD
import importlib
import tempfile
from pathlib import Path

from torch.functional import split
importlib.reload(dynwgraphs)
import mlflow
from dynwgraphs.utils.dgps import get_test_w_seq


# %%


#%%
def sample_and_estimate(dgp_par, max_opt_iter, Y_reference):
    with mlflow.start_run() as run:

        with tempfile.TemporaryDirectory() as tmpdirname:

            tmp_path = Path(tmpdirname)
        
            mlflow.log_param("max_opt_iter", max_opt_iter)

            mlflow.log_params(dgp_par)

            mod_dgp = get_dgp_model(dgp_par, Y_reference=Y_reference)

            mod_sd = dirBin1_SD(mod_dgp.Y_T)

            mod_naive = dirBin1_SD(mod_dgp.Y_T)

            mod_naive.init_phi_T_from_obs()

            mod_sd.opt_options_sd["max_opt_iter"] = max_opt_iter
            _, h_par_opt = mod_sd.estimate_sd(tb_save_fold=tmp_path)

            mlflow.log_params(h_par_opt)


            phi_T_sd = mod_sd.par_list_to_matrix_T(mod_sd.phi_T)
            phi_T_naive = mod_naive.par_list_to_matrix_T(mod_naive.phi_T)
            phi_T_dgp = mod_dgp.par_list_to_matrix_T(mod_dgp.phi_T)


            mse_sd = (torch.square(phi_T_dgp - phi_T_sd)).mean().item()
            mse_naive = (torch.square(phi_T_dgp - phi_T_naive)).mean().item()

            metrics = {"mse_sd" : mse_sd, "mse_naive":mse_naive} 

            i=11
            mlflow.log_figure(mod_sd.plot_phi_T(i=i)[0], f"sd_filt_ex_ind_{i}.png")
            mlflow.log_figure(mod_dgp.plot_phi_T(i=i)[0], f"dgp_ex_ind_{i}.png")


            mlflow.log_metric("sd_mse", mse_sd)
            mlflow.log_metric("naive_mse", mse_naive)

            # mod_sd.save_parameters(save_path=tmp_path)
            # mod_dgp.save_parameters(save_path=tmp_path)
            # torch.save(Y_reference, tmp_path / "Y_reference.pkl")
            
            # mlflow.log_artifacts(tmp_path)

#%%

from joblib import Parallel, delayed


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Simulate missp dgp and estimate sd filter."
    )


    parser.add_argument("--n_sim", help="Number of simulations", type=int)
    parser.add_argument("--max_opt_iter", help="max number of opt iter", type=int, default = 3500)

    args = parser.parse_args()

    Y_T_test, _, _ =  get_test_w_seq(avg_weight=1e3)
    Y_reference = (Y_T_test[:,:,0]>0).float()

    # define run's parameters
    dgp_par = {"N" : 50, "T" : 100, "model":"dirBin1", "type" : "AR", "B" : 0.98, "sigma" : 0.1}


    
    experiment_name = "binary directed sd filter missp dgp"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    mlflow.set_experiment(experiment_name)

    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))



    Parallel(n_jobs=4)(delayed(sample_and_estimate)(dgp_par, args.max_opt_iter, Y_reference) for _ in range(args.n_sim))


# %%
#TO DO
# check results
# print every k opt steps
# check tensorboard res