

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

from torch.functional import split
importlib.reload(dynwgraphs)
import mlflow
from dynwgraphs.utils.dgps import get_test_w_seq
from joblib import Parallel, delayed



#%%
def sample_estimate_and_log(dgp_par, max_opt_iter, Y_reference, experiment):
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:

        with tempfile.TemporaryDirectory() as tmpdirname:

            tmp_path = Path(tmpdirname)
        
            mlflow.log_param("max_opt_iter", max_opt_iter)

            mlflow.log_params({f"{'dgp_'}{key}": val for key, val in dgp_par.items()})

            mod_dgp = get_dgp_model(dgp_par, Y_reference=Y_reference)

            mod_naive = dirBin1_SD(mod_dgp.Y_T)
            mod_naive.init_phi_T_from_obs()


            mod_sd = dirBin1_SD(mod_dgp.Y_T)
            mod_sd.opt_options_sd["max_opt_iter"] = max_opt_iter
            _, h_par_opt = mod_sd.estimate_sd(tb_save_fold=tmp_path)
            mlflow.log_params({f"{'sd_'}{key}": val for key, val in h_par_opt.items()})
            mlflow.log_params({f"{'sd_'}{key}": val for key, val in mod_sd.get_info_dict().items()})

            mod_ss = dirBin1_sequence_ss(mod_dgp.Y_T)
            mod_ss.opt_options_ss_seq["max_opt_iter"] = max_opt_iter
            _, h_par_opt = mod_ss.estimate_ss_seq_joint(tb_save_fold=tmp_path)
            mlflow.log_params({f"{'ss_'}{key}": val for key, val in h_par_opt.items()})
            mlflow.log_params({f"{'ss_'}{key}": val for key, val in mod_ss.get_info_dict().items()})



            phi_T_sd = mod_sd.par_list_to_matrix_T(mod_sd.phi_T)
            phi_T_ss = mod_ss.par_list_to_matrix_T(mod_ss.phi_T)
            phi_T_naive = mod_naive.par_list_to_matrix_T(mod_naive.phi_T)
            phi_T_dgp = mod_dgp.par_list_to_matrix_T(mod_dgp.phi_T)


            mse_sd = (torch.square(phi_T_dgp - phi_T_sd)).mean().item()
            mse_ss = (torch.square(phi_T_dgp - phi_T_ss)).mean().item()
            mse_naive = (torch.square(phi_T_dgp - phi_T_naive)).mean().item()

            metrics = {"mse_sd" : mse_sd, "mse_naive":mse_naive, "mse_ss":mse_ss} 

            mlflow.log_metrics(metrics)

            mlflow.log_figure(mod_sd.plot_phi_T()[0], f"sd_filt_all.png")
            mlflow.log_figure(mod_ss.plot_phi_T()[0], f"ss_filt_all.png")
            mlflow.log_figure(mod_dgp.plot_phi_T()[0], f"dgp_all.png")

            i=11
            mlflow.log_figure(mod_sd.plot_phi_T(i=i)[0], f"sd_filt_ex_ind_{i}.png")
            mlflow.log_figure(mod_ss.plot_phi_T(i=i)[0], f"ss_filt_ex_ind_{i}.png")
            mlflow.log_figure(mod_dgp.plot_phi_T(i=i)[0], f"dgp_ex_ind_{i}.png")

            
            mod_sd.save_parameters(save_path=tmp_path)
            mod_ss.save_parameters(save_path=tmp_path)
            mod_dgp.save_parameters(save_path=tmp_path)

            torch.save(Y_reference, tmp_path / "Y_reference.pkl")
                
            torch.save(mod_dgp.get_Y_T_to_save(), tmp_path / "Y_T_dgp.pkl")
            
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

    Y_T_test, _, _ =  get_test_w_seq(avg_weight=1e3)
    Y_reference = (Y_T_test[:,:,0]>0).float()

    # define run's parameters
    dgp_par = {"N" : 50, "T" : 100, "model":"dirBin1", "type" : "AR", "B" : 0.98, "sigma" : 0.1}


    Parallel(n_jobs=4)(delayed(sample_estimate_and_log)(dgp_par, args.max_opt_iter, Y_reference, experiment) for _ in range(args.n_sim))


# # %%

# run_info_list = mlflow.list_run_infos(experiment.experiment_id)
# i = 0
# run_i = mlflow.get_run(run_info_list[i].run_id)
# run_i = mlflow.get_run("80d9eb91ed264b67ab91b933a00008ae")

# from pathlib import Path
# load_path = Path(run_i.info.artifact_uri[8:] + "/")
# Y_T_dgp = torch.load(load_path / "Y_T_dgp.pkl")

# mod_sd = dirBin1_SD(Y_T_dgp)

# file:///D:/pCloud/Dynamic_Networks/repos/DynWGraphsPaper/analysis/filter_misssp_dgps/mlruns/0/80d9eb91ed264b67ab91b933a00008ae/artifacts/sd_init_unc_mean_resc_True_dist_bernoulli_phi_tv_True_size_par_dist_None_tv_None_size_beta_1_tv_tensor([False])_opt_n=ADAMHD_min_opt_iter=50_max_opt_iter=11_lr=0.01__par.pkl


