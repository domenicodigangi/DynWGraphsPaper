#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday July 7th 2021

"""


#%% import packages
import numpy as np
from numpy.core.fromnumeric import size
import torch
from pathlib import Path
import matplotlib.pyplot as plt
# matplotlib.rcParams['text.usetex'] = True
import dynwgraphs
from dynwgraphs.utils.tensortools import tens, splitVec
from dynwgraphs.dirGraphs1_dynNets import  dirBin1_sequence_ss, dirBin1_SD, dirSpW1_SD
import mlflow
import click
import importlib
from torch.functional import split
importlib.reload(dynwgraphs)


ld_data = np.load( "../../../../data/emid_data/numpyFiles/eMid_numpy.npz",allow_pickle=True)

eMid_w_T, all_dates, eonia_w, nodes = ld_data["arr_0"], ld_data["arr_1"], ld_data["arr_2"], ld_data["arr_3"]


unit_meas = 1e4
Y_T = tens(eMid_w_T / unit_meas) 
N,_, T = Y_T.shape
X_T = tens(np.tile(eonia_w.T, (N, N, 1, 1)))

T_train = T*3//4

save_path = Path(f"../../data/estimates_real_data/eMid/eonia_reg/T_train_{T_train}")
save_path.mkdir(parents=True, exist_ok=True)



#%% Score driven binary phi_T

#%% Score driven binary phi_T
estimate_flag = False
model_bin_0 = dirBin1_SD(Y_T,  T_train=T_train) 

model_bin_0.run(estimate_flag, save_path)


#%% Score driven binary phi_T with const regr
estimate_flag = False

model_bin_1 = dirBin1_SD(Y_T, T_train=T_train, X_T=X_T,  size_beta_t=1, beta_tv=[False]) # 


model_bin_1.init_par_from_model_without_beta(model_bin_0)

model_bin_1.run(estimate_flag, save_path)


#%% Score driven binary phi_T with time varying regr
estimate_flag=False

model_bin_2 = dirBin1_SD(Y_T, T_train=T_train, X_T=X_T,  size_beta_t=1, beta_tv=[True]) # 

# model_bin_2.init_par_from_model_with_const_par(model_bin_1)
# model_bin_2.run(estimate_flag, save_path)

model_bin_2.sd_stat_par_un_beta["A"].data = model_bin_2.re2un_A_par(torch.ones(1)*0.000001)

model_bin_2.opt_options_sd["lr"] = 0.001
model_bin_2.opt_options_sd["opt_n"] = "LBFGS"


model_bin_2.run(estimate_flag, save_path)

#%% Score driven weighted phi_T
estimate_flag = False
model_w_0 = dirSpW1_SD(Y_T, T_train=T_train)  
model_w_0.run(estimate_flag, save_path)


#%% Score driven weighted phi_T with constant beta
estimate_flag = False

model_w_1 = dirSpW1_SD(Y_T, T_train=T_train, X_T=X_T,  size_beta_t=1, beta_tv=[False]) # 

model_w_1.run(estimate_flag, save_path)


#%% Score driven weighted phi_T with time varying beta
estimate_flag = False

model_w_2 = dirSpW1_SD(Y_T, T_train=T_train, X_T=X_T,  size_beta_t=1, beta_tv=[True]) # 

model_w_2.sd_stat_par_un_beta["A"].data = model_w_2.re2un_A_par(torch.ones(1)*0.000001)

model_w_2.opt_options_sd["lr"] = 0.001
model_w_2.opt_options_sd["opt_n"] = "LBFGS"

# model_w_2.init_par_from_model_with_const_par(model_w_1)

model_w_2.run(estimate_flag, save_path)


# %%

dates = all_dates[:T_train]
_, _, beta_bin = model_bin_2.get_seq_latent_par()
_, _, beta_w = model_w_2.get_seq_latent_par()

fig, ax1 = plt.subplots(figsize = (10, 6))
ax2 = ax1.twinx()

ax1.plot(dates, beta_bin.squeeze(), 'g-')
ax2.plot(dates, beta_w.squeeze(), 'b-')

ax1.set_ylabel(r'$\beta_{bin}$', color='b', size=20)
ax2.set_ylabel(r'$\beta_w$', color='g', size=20)
ax1.tick_params(axis='x', labelrotation=45)
ax1.grid()
plt.show()
# %%



def sample_estimate_and_log(mod_dgp_dict, run_par_dict, run_data_dict, experiment):
  
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:

        logger.info(run_par_dict)
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


            for k_mod_dgp, mod_dgp in mod_dgp_dict.items():
                logger.info(f" start estimates {k_mod_dgp}")

                filt_kwargs = {"size_beta_t":mod_dgp.size_beta_t, "X_T" : mod_dgp.X_T, "beta_tv":mod_dgp.beta_tv}
                # sample obs from dgp and save data

                torch.save(run_data_dict["Y_reference"], dgp_fold / "Y_reference.pkl")
                torch.save((mod_dgp.get_Y_T_to_save(),mod_dgp.X_T), dgp_fold / "obs_T_dgp.pkl")
                mod_dgp.save_parameters(save_path=dgp_fold)
            

                #estimate models and log parameters and hpar optimization
                if k_mod_dgp == "bin":
                    mod_dgp.Y_T = mod_dgp.sample_Y_T()
                    mod_sd = dirBin1_SD(mod_dgp.Y_T, **filt_kwargs)
                    mod_ss = dirBin1_sequence_ss(mod_dgp.Y_T, **filt_kwargs)
                elif k_mod_dgp == "w":
                    mod_dgp.Y_T = mod_dgp.sample_Y_T(A_T = mod_dgp.bin_mod.Y_T)

                    mod_sd = dirSpW1_SD(mod_dgp.Y_T, **filt_kwargs)
                    mod_ss = dirSpW1_sequence_ss(mod_dgp.Y_T, **filt_kwargs)
                else:
                    raise
                
                filt_models = {"sd":mod_sd, "ss":mod_ss}

                for k_filt, mod in filt_models.items():
                    mod.opt_options["max_opt_iter"] = run_par_dict["max_opt_iter"]

                    _, h_par_opt = mod.estimate(tb_save_fold=tb_fold)

                    mlflow.log_params({f"{k_mod_dgp}_{k_filt}_{key}": val for key, val in h_par_opt.items()})
                    mlflow.log_params({f"{k_mod_dgp}_{k_filt}_{key}": val for key, val in mod.get_info_dict().items() if key not in h_par_opt.keys()})

                    mod.save_parameters(save_path=tmp_path)
                

                # compute mse for each model and log it 
                phi_to_exclude = strIO_from_tens_T(mod_dgp.Y_T) < 1e-3 
                mse_dict_sd = filt_err(mod_dgp, mod_sd, phi_to_exclude, suffix="sd", prefix=k_mod_dgp)
                mse_dict_ss = filt_err(mod_dgp, mod_ss, phi_to_exclude, suffix="ss", prefix=k_mod_dgp)

                mlflow.log_metrics(mse_dict_sd) 
                mlflow.log_metrics(mse_dict_ss) 
            
                # log plots that can be useful for quick visual diagnostic 
                mlflow.log_figure(mod_sd.plot_phi_T()[0], f"fig/{k_mod_dgp}_sd_filt_all.png")
                mlflow.log_figure(mod_ss.plot_phi_T()[0], f"fig/{k_mod_dgp}_ss_filt_all.png")
                mlflow.log_figure(mod_dgp.plot_phi_T()[0], f"fig/{k_mod_dgp}_dgp_all.png")

                i=torch.where(~splitVec(phi_to_exclude)[0])[0][0]

                mlflow.log_figure(mod_sd.plot_phi_T(i=i, fig_ax= mod_dgp.plot_phi_T(i=i))[0], f"fig/{k_mod_dgp}_sd_filt_phi_ind_{i}.png")
                
                mlflow.log_figure(mod_ss.plot_phi_T(i=i, fig_ax= mod_dgp.plot_phi_T(i=i))[0], f"fig/{k_mod_dgp}_ss_filt_phi_ind_{i}.png")
                
                if mod_dgp.any_beta_tv():
                    mlflow.log_figure(mod_sd.plot_beta_T(fig_ax= mod_dgp.plot_beta_T())[0], f"fig/{k_mod_dgp}_sd_filt_beta_T.png")
                    mlflow.log_figure(mod_ss.plot_beta_T(fig_ax= mod_dgp.plot_beta_T())[0], f"fig/{k_mod_dgp}_ss_filt_beta_T.png")

            
            # log all files and sub-folders in temp fold as artifacts            
            mlflow.log_artifacts(tmp_path)


#%%

@click.command()
#"Simulate missp dgp and estimate sd and ss filters"
@click.option("--n_sim", help="Number of simulations", type=int)
@click.option("--max_opt_iter", help="max number of opt iter", type=int, default = 15000)
@click.option("--n_nodes", help="Number of nodes", type=int, default=50)
@click.option("--n_time_steps", help="Number of time steps", type=int, default = 100)
@click.option("--type_dgp_phi_bin", help="what kind of dgp should phi_T follow. AR or const_unif_prob", type=str, default = "AR")
@click.option("--ext_reg_bin_options", help="Options for external regressors and model specification. In order : 1 number of external regressors,  2 size of beta(One for all, N : one per node (both in and out), 2N : two per node, one for in and one for out links), 3 beta_tv (should the regression coefficients, of both dgp and filter, be time varying ?), 4 type_dgp_beta_bin (what kind of dgp should beta_T follow) ", type=(int, str, bool, str), default = (0, "one", False, "AR"))

@click.option("--exclude_weights", help="shall we run the sim only for the binary case? ", type=bool, default = False)
@click.option("--type_dgp_phi_w", help="what kind of dgp should phi_T follow ", type=str, default = "AR")
@click.option("--ext_reg_w_options", help="Options for external regressors and model specification. In order : 1 number of external regressors,  2 size of beta(One for all, N : one per node (both in and out), 2N : two per node, one for in and one for out links), 3 beta_tv (should the regression coefficients, of both dgp and filter, be time varying ?), 4 type_dgp_beta_bin (what kind of dgp should beta_T follow) ", type=(int, str, bool, str), default = (0, "one", False, "AR"))
@click.option("--n_jobs", type = int, default = 4)

def run_parallel_simulations(**kwargs):
   
    beta_set_names = ["n_ext_reg", "size_beta_t", "all_beta_tv", "type_dgp_beta"]
    
    dgp_set_bin = {beta_set_names[i]: o for i, o in enumerate(kwargs["ext_reg_bin_options"]) }
    dgp_set_bin["model"] = "dirBin1"
    dgp_set_bin["type_dgp_phi"] = kwargs["type_dgp_phi_bin"]


    if not kwargs["exclude_weights"]:
        dgp_set_w = {beta_set_names[i]: o for i, o in enumerate(kwargs["ext_reg_w_options"]) } 
        dgp_set_w["model"] = "dirSpW1"
        dgp_set_w["type_dgp_phi"] = kwargs["type_dgp_phi_w"]
    
    # set appropriate experiment
    experiment_name =  f"directed filter missp dgp, bin {str(dgp_set_bin)}"

    if not kwargs["exclude_weights"]:
        experiment_name +=  f" weighted {str(dgp_set_w)}"

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        client = mlflow.tracking.MlflowClient()
        experiment_id = client.create_experiment(experiment_name)
        experiment = client.get_experiment(experiment_id)



    mlflow.set_experiment(experiment_name)
    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))


    T = kwargs["n_time_steps"]
    N = kwargs["n_nodes"]
    dgp_par = {"N": N, "T": T }

    # define binary dgp
    mod_dgp_bin, dgp_par_bin, Y_reference_bin = get_dgp_model(N=N, T=T, **dgp_set_bin)
    dgp_par["bin"] = dgp_par_bin 

    mod_dgp = {"bin": mod_dgp_bin}
    run_data_dict = {"Y_reference":{"Y_reference_bin": Y_reference_bin} }

    if not kwargs["exclude_weights"]:
        # define weighted dgp
        mod_dgp_w, dgp_par_w, Y_reference_w = get_dgp_model(N=N, T=T, **dgp_set_w)
        mod_dgp_w.bin_mod = mod_dgp_bin
        mod_dgp["w"] = mod_dgp_w
        dgp_par["w"] = dgp_par_w 
        run_data_dict["Y_reference"]["Y_reference_w"] = Y_reference_w

   

    run_par_dict = {"dgp_par": dgp_par, "max_opt_iter": kwargs["max_opt_iter"]}


    def try_one_run(mod_dgp, run_par_dict, run_data_dict, experiment):
        try:
            sample_estimate_and_log(mod_dgp, run_par_dict, run_data_dict, experiment)
        except:
            logger.warning("Run failed")

    Parallel(n_jobs=kwargs["n_jobs"])(delayed(try_one_run)(mod_dgp, run_par_dict, run_data_dict, experiment) for _ in range(kwargs["n_sim"]))


#%% Run
if __name__ == "__main__":
    run_parallel_simulations()