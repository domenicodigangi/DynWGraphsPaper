

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday July 10th 2021

Simulate time varying fitness models, binary and weighted, possibly with external regressors, and filter with a SD filter and with a sequence of single snapshots estimates. Log paramters and metrics in mlflow runs.

"""

# %% import packages
from pathlib import Path
import importlib
import torch
import dynwgraphs
from dynwgraphs.utils.tensortools import splitVec, strIO_from_tens_T
from dynwgraphs.dirGraphs1_dynNets import dirBin1_sequence_ss, dirBin1_SD, dirSpW1_SD, dirSpW1_sequence_ss
from dynwgraphs.utils.dgps import get_dgp_mod_and_par
import tempfile
import mlflow
from joblib import Parallel, delayed
from torch import nn
import logging
import click
from matplotlib import pyplot as plt
from ddg_utils.mlflow import _get_and_set_experiment, check_test_exp
from ddg_utils import drop_keys
logger = logging.getLogger(__name__)
importlib.reload(dynwgraphs)


# %%

@click.command()
#"Simulate missp dgp and estimate sd and ss filters"
@click.option("--n_sim", help="Number of simulations", type=int, default=2)
@click.option("--max_opt_iter", help="max number of opt iter", type=int, default=15000)
@click.option("--n_nodes", help="Number of nodes", type=int, default=50)
@click.option("--n_time_steps", help="Number of time steps", type=int, default=100)


@click.option("--type_tv_dgp_phi", help="what kind of dgp should phi_T follow. AR or const_unif", type=(str, str, float, float), default=("AR", "ref_mat", 0.98, 0.01))
@click.option("--type_tv_dgp_phi_bin", help="what kind of dgp should phi_T follow. AR or const_unif", type=(str, str, float, float), default=(None, None, None, None))
@click.option("--type_tv_dgp_phi_w", help="what kind of dgp should phi_T follow. AR or const_unif", type=(str, str, float, float), default=(None, None, None, None))


@click.option("--phi_dgp_set", help="1 - How many fitnesses? one (Erdos-Reny), N (undirected model), 2N (directed).   2 - Should they be Time varying ? True of False  ", type=(str, bool), default=("2N", False))
@click.option("--phi_dgp_set_bin", help="Same as phi_dgp_set for bin parameters ", type=(str, bool), default=(None, None))
@click.option("--phi_dgp_set_w", help="Same as phi_dgp_set for w parameters ", type=(str, bool), default=(None, None))


@click.option("--phi_filt_set", help="1 - How many fitnesses? one (Erdos-Reny), N (undirected model), 2N (directed).   2 - Should they be Time varying ? True of False  ", type=(str, bool), default=("2N", False))
@click.option("--phi_filt_set_bin", help="Same as phi_filt_set for bin parameters ", type=(str, bool), default=(None, None))
@click.option("--phi_filt_set_w", help="Same as phi_filt_set for w parameters ", type=(str, bool), default=(None, None))


@click.option("--beta_dgp_set", help="Options for specification of external regressors' parameters. In order : 1 number of external regressors,  2 size of beta(One for all, N : one per node (both in and out), 2N : two per node, one for in and one for out links), 3 beta_tv (should the regression coefficients, of both dgp and filter, be time varying ?)) ", type=(int, str, bool), default=(1, "one", False))
@click.option("--beta_dgp_set_bin", help="Same as beta_dgp_set for bin parameters ", type=(int, str, bool), default=(None, None, None))
@click.option("--beta_dgp_set_w", help="Same as beta_dgp_set for w parameters ", type=(int, str, bool), default=(None, None, None))


@click.option("--beta_filt_set", help="Same as beta_dgp_set for filter specification ", type=(int, str, bool), default=(1, "one", False))
@click.option("--beta_filt_set_bin", help="Same as beta_filt_set for bin parameters ", type=(int, str, bool), default=(None, None, None))
@click.option("--beta_filt_set_w", help="Same as beta_filt_set for w parameters ", type=(int, str, bool), default=(None, None, None))


@click.option("--type_tv_dgp_beta", help="If time varyng, how should beta par evolve? 1 cross sec setting, 2 time variation type, 3 parameters for time varying dgp (unc_mean, B, sigma) ", type=(str, float, float, float), default=("AR", 1, 0, 0))
@click.option("--type_tv_dgp_beta_bin", help="Same as type_tv_dgp_beta for bin", type=(str, float, float, float), default=(None, None, None, None))
@click.option("--type_tv_dgp_beta_w", help="Same as type_tv_dgp_beta for w", type=(str, float, float, float), default=(None, None, None, None))


@click.option("--type_tv_dgp_ext_reg", help="How should external regressorrs evolve? 1 cross sec setting, 2 time variation type, 3 parameters for time varying dgp (unc_mean, B, sigma) ", type=(str, str, float, float, float), default=("uniform", "AR", 1, 0, 0.1))
@click.option("--type_tv_dgp_ext_reg_bin", help="Same as type_tv_dgp_ext_reg for bin", type=(str, str, float, float, float), default=(None, None, None, None, None))
@click.option("--type_tv_dgp_ext_reg_w", help="Same as type_tv_dgp_ext_reg for w", type=(str, str, float, float, float), default=(None, None, None, None, None))


@click.option("--exclude_weights", help="shall we run the sim only for the binary case? ", type=bool, default=False)


@click.option("--n_jobs", type=int, default=2)
@click.option("--experiment_name", type=str, default="filter missp dgp")


def run_parallel_simulations(**kwargs):
    return _run_parallel_simulations(**kwargs)



def cli_beta_set_to_mod_kwargs_dict(phi_set, reg_set, max_opt_iter):
    """
    convert the settings from the command line to a format that can be given as a model input
    """

    cli_phi_set_input_name = ["size_phi_t", "all_phi_tv"]
    phi_set_dict = {k: v for k, v in zip(cli_phi_set_input_name, phi_set)}

    phi_out_dict = {}
    phi_out_dict["size_phi_t"] = phi_set_dict["size_phi_t"]   
    phi_out_dict["phi_tv"] = phi_set_dict["all_phi_tv"]

    phi_out_dict["max_opt_iter"] = max_opt_iter

    cli_beta_set_input_name = ["n_ext_reg", "size_beta_t", "all_beta_tv"]
    reg_set_dict = {k: v for k, v in zip(cli_beta_set_input_name, reg_set)}

    beta_out_dict = {}
    beta_out_dict["n_ext_reg"] = reg_set_dict["n_ext_reg"]
    beta_out_dict["size_beta_t"] = reg_set_dict["size_beta_t"]
    beta_out_dict["beta_tv"] = [reg_set_dict["all_beta_tv"] for p in range(beta_out_dict["n_ext_reg"])]
    if reg_set_dict["n_ext_reg"] == 0:
        if beta_out_dict["size_beta_t"] not in [None, 0]:
            raise Exception("If n_ext_reg is zero beta should not have a size")
        if beta_out_dict["size_beta_t"] not in [None, False]:
            raise Exception("If n_ext_reg is zero there are no betas to be time varying")
   
    else:
        if reg_set_dict["size_beta_t"] in [None, 0]:
            raise Exception("If n_ext_reg is non zero beta should be non zero")

    beta_out_dict["max_opt_iter"] = max_opt_iter

    out_dict = {**phi_out_dict, **beta_out_dict}

    return out_dict



def get_dgp_and_filt_set_from_cli_options(kwargs, bin_or_w):
    """
    Convert cli inputs to settings for both dgp and filter models.
    """
    N = kwargs["n_nodes"]

    logger.info("using common external regressor settings for binary and weighted parameters")
    
    if kwargs[f"phi_dgp_set_{bin_or_w}"][0] is None:
        kwargs[f"phi_dgp_set_{bin_or_w}"] = kwargs["phi_dgp_set"]

    if kwargs[f"phi_filt_set_{bin_or_w}"][0] is None:
        kwargs[f"phi_filt_set_{bin_or_w}"] = kwargs["phi_filt_set"]    

    if kwargs[f"beta_dgp_set_{bin_or_w}"][0] is None:
        kwargs[f"beta_dgp_set_{bin_or_w}"] = kwargs[f"beta_dgp_set"]

    if kwargs[f"beta_filt_set_{bin_or_w}"][0] is None:
        kwargs[f"beta_filt_set_{bin_or_w}"] = kwargs[f"beta_filt_set"]

    dgp_set = cli_beta_set_to_mod_kwargs_dict(kwargs[f"phi_dgp_set_{bin_or_w}"], kwargs[f"beta_dgp_set_{bin_or_w}"], max_opt_iter=kwargs["max_opt_iter"])

    filt_set = cli_beta_set_to_mod_kwargs_dict(kwargs[f"phi_filt_set_{bin_or_w}"], kwargs[f"beta_filt_set_{bin_or_w}"], max_opt_iter=kwargs["max_opt_iter"])

    if kwargs[f"type_tv_dgp_phi_{bin_or_w}"][0] is None:
         kwargs[f"type_tv_dgp_phi_{bin_or_w}"] = kwargs[f"type_tv_dgp_phi"]

    dgp_set[f"type_tv_dgp_phi"] = kwargs[f"type_tv_dgp_phi_{bin_or_w}"]

    if kwargs[f"type_tv_dgp_beta_{bin_or_w}"][0] is None:
         kwargs[f"type_tv_dgp_beta_{bin_or_w}"] = kwargs[f"type_tv_dgp_beta"]

    dgp_set[f"type_tv_dgp_beta"] = kwargs[f"type_tv_dgp_beta_{bin_or_w}"]

    if kwargs[f"type_tv_dgp_ext_reg_{bin_or_w}"][0] is None:
         kwargs[f"type_tv_dgp_ext_reg_{bin_or_w}"] = kwargs[f"type_tv_dgp_ext_reg"]

    dgp_set[f"type_tv_dgp_ext_reg"] = kwargs[f"type_tv_dgp_ext_reg_{bin_or_w}"]

    dgp_set["bin_or_w"] = bin_or_w
    filt_set["bin_or_w"] = bin_or_w

    return dgp_set, filt_set


def get_filt_mod(bin_or_w, Y_T, X_T_dgp, par_dict):
    par_dict = drop_keys(par_dict, ["bin_or_w"])
    if par_dict["size_beta_t"] in [None, 0]:
        par_dict["X_T"] = None
    elif par_dict["n_ext_reg"] > 0 :
        par_dict["X_T"] = X_T_dgp[:, :, :par_dict["n_ext_reg"]]

    par_dict = drop_keys(par_dict, ["n_ext_reg"])

    if bin_or_w == "bin":
        mod_sd = dirBin1_SD(Y_T, **par_dict)
        mod_ss = dirBin1_sequence_ss(Y_T, **par_dict)
    elif bin_or_w == "w":
        mod_sd = dirSpW1_SD(Y_T, **par_dict)
        mod_ss = dirSpW1_sequence_ss(Y_T, **par_dict)
    else:
        raise

    filt_models = {"sd": mod_sd, "ss": mod_ss}
    return filt_models


def _run_parallel_simulations(**kwargs):

    check_test_exp(kwargs)
    T = kwargs["n_time_steps"]
    N = kwargs["n_nodes"]
    print(kwargs["type_tv_dgp_phi"])
    dgp_set_bin, filt_set_bin = get_dgp_and_filt_set_from_cli_options(kwargs, "bin")
    
    if not kwargs["exclude_weights"]:
        dgp_set_w, filt_set_w = get_dgp_and_filt_set_from_cli_options(kwargs, "w")
        
    experiment = _get_and_set_experiment(kwargs["experiment_name"])

    with mlflow.start_run(experiment_id=experiment.experiment_id) as parent_run:

        parent_runs_par = drop_keys(kwargs, ["type_tv_dgp_phi", "type_tv_dgp_beta", "beta_dgp_set", "beta_filt_set", "phi_dgp_set", "phi_filt_set", "dgp_ext_reg"])
        mlflow.log_params(parent_runs_par)

        mlflow.log_params({f"{key}_dgp_bin": val for key, val in dgp_set_bin.items() if key != "X_T"})
        if not kwargs["exclude_weights"]:
            mlflow.log_params({f"{key}_dgp_w": val for key, val in dgp_set_w.items() if key != "X_T"})

        # define binary dgp and filter par
        logger.info(dgp_set_bin)
        mod_dgp_bin, Y_reference_bin = get_dgp_mod_and_par(N=N, T=T, dgp_set_dict=dgp_set_bin)
        mod_dgp_dict = {"bin": mod_dgp_bin}
        run_data_dict = {"Y_reference":{"Y_reference_bin": Y_reference_bin}}

        if not kwargs["exclude_weights"]:
            # define weighted dgp
            logger.info(dgp_set_w)
            mod_dgp_w, Y_reference_w = get_dgp_mod_and_par(N=N, T=T, dgp_set_dict=dgp_set_w)
            
            mod_dgp_w.bin_mod = mod_dgp_bin
            mod_dgp_dict["w"] = mod_dgp_w
            run_data_dict["Y_reference"]["Y_reference_w"] = Y_reference_w
        


        run_par_dict = {"N": N, "T": T, "dgp_par_bin": dgp_set_bin, "dgp_par_w": dgp_set_w, "filt_par_bin": filt_set_bin, "filt_par_w": filt_set_w}


        def try_one_run(mod_dgp_dict, run_par_dict, run_data_dict, parent_run, parent_runs_par):
            sample_estimate_and_log(mod_dgp_dict, run_par_dict, run_data_dict, parent_run, parent_runs_par)
            try:
                pass
                # sample_estimate_and_log(mod_dgp_dict, run_par_dict, run_data_dict, parent_run, parent_runs_par)
            except Exception as e:
                logger.warning(f"Run failed : \n {e}")

        Parallel(n_jobs=kwargs["n_jobs"])(delayed(try_one_run)(mod_dgp_dict, run_par_dict, run_data_dict, parent_run, parent_runs_par) for _ in range(kwargs["n_sim"]))


def sample_estimate_and_log(mod_dgp_dict, run_par_dict, run_data_dict, parent_run, parent_runs_par):
  
    with mlflow.start_run(run_id=parent_run.info.run_id):
        with mlflow.start_run(experiment_id=parent_run.info.experiment_id, nested=True):

            logger.info(run_par_dict)
            
            # save files in temp folder, then log them as artifacts in mlflow and delete temp fold

            with tempfile.TemporaryDirectory() as tmpdirname:

                # set artifacts folders and subfolders
                tmp_path = Path(tmpdirname)
                dgp_fold = tmp_path / "dgp"
                dgp_fold.mkdir(exist_ok=True)
                tb_fold = tmp_path / "tb_logs"
                tb_fold.mkdir(exist_ok=True)

                mlflow.log_params(parent_runs_par)
                bin_or_w, mod_dgp = list(mod_dgp_dict.items())[0]
                for bin_or_w, mod_dgp in mod_dgp_dict.items():
                    mlflow.log_params({f"dgp_{bin_or_w}_{key}": val for key, val in run_par_dict[f"dgp_par_{bin_or_w}"].items()})
                    mlflow.log_params({f"filt_{bin_or_w}_{key}": val for key, val in run_par_dict[f"filt_par_{bin_or_w}"].items()})
                    logger.info(f" start estimates {bin_or_w}")

                    # sample obs from dgp and save data
                    if hasattr(mod_dgp, "bin_mod"):
                        mod_dgp.Y_T = mod_dgp.sample_Y_T(A_T=mod_dgp.bin_mod.Y_T)
                    else:
                        mod_dgp.Y_T = mod_dgp.sample_Y_T()

                    torch.save(run_data_dict["Y_reference"], dgp_fold / "Y_reference.pt")
                    torch.save((mod_dgp.get_Y_T_to_save(), mod_dgp.X_T), dgp_fold / "obs_T_dgp.pt")
                    mod_dgp.save_parameters(save_path=dgp_fold)
                    mlflow.log_figure(mod_dgp.plot_phi_T()[0], f"fig/{bin_or_w}_dgp_all.png")
                
                    #estimate models and log parameters and hpar optimization
                    filt_models = get_filt_mod(bin_or_w, mod_dgp.Y_T, mod_dgp.X_T, run_par_dict[f"filt_par_{bin_or_w}"])
                    
                    # k_filt, mod = list(filt_models.items())[0]
                    for k_filt, mod_filt in filt_models.items():

                        _, h_par_opt, stats_opt = mod_filt.estimate(tb_save_fold=tb_fold)

                        mlflow.log_params({f"filt_{bin_or_w}_{k_filt}_{key}": val for key, val in h_par_opt.items()})
                        mlflow.log_metrics({f"filt_{bin_or_w}_{k_filt}_{key}": val for key, val in stats_opt.items()})
                        mlflow.log_params({f"filt_{bin_or_w}_{k_filt}_{key}": val for key, val in mod_filt.get_info_dict().items() if key not in h_par_opt.keys()})

                        mod_filt.save_parameters(save_path=tmp_path)
                    
                        # compute mse for each model and log it 
                        phi_to_exclude = strIO_from_tens_T(mod_dgp.Y_T) < 1e-3 

                        mse_dict = filt_err(mod_dgp, mod_filt, phi_to_exclude, suffix=k_filt, prefix=bin_or_w)
                        mlflow.log_metrics(mse_dict) 
                    
                        # log plots that can be useful for quick visual diagnostic 
                        mlflow.log_figure(mod_filt.plot_phi_T()[0], f"fig/{bin_or_w}_{k_filt}_filt_all.png")
                        i_plot = torch.where(~splitVec(phi_to_exclude)[0])[0][0]

                        mlflow.log_figure(mod_filt.plot_phi_T(i=i_plot, fig_ax= mod_dgp.plot_phi_T(i=i_plot))[0], f"fig/{bin_or_w}_{k_filt}_filt_phi_ind_{i_plot}.png")

                        if mod_dgp.X_T is not None:
                            mlflow.log_metric(f"{bin_or_w}_avg_beta_dgp", get_avg_beta(mod_dgp))
                            fig = plt.figure() 
                            plt.plot(mod_dgp.X_T[0,0,:,:].T, figure=fig)
                            mlflow.log_figure(fig, f"fig/{bin_or_w}_X_0_0_T.png")
                            
                        if mod_dgp.any_beta_tv():
                            plot_dgp_fig_ax = mod_dgp.plot_beta_T()
                            mlflow.log_figure(plot_dgp_fig_ax[0], f"fig/{bin_or_w}_sd_filt_beta_T.png")
                        if mod_filt.beta_T is not None:
                            if mod_filt.any_beta_tv():
                                mlflow.log_figure(mod_filt.plot_beta_T(fig_ax=plot_dgp_fig_ax)[0], f"fig/{bin_or_w}_{k_filt}_filt_beta_T.png")
                                
                # log all files and sub-folders in temp fold as artifacts            
                mlflow.log_artifacts(tmp_path)


def filt_err(mod_dgp, mod_filt, phi_to_exclude, suffix="", prefix=""):
    
    loss_fun = nn.MSELoss()
    phi_T_filt = mod_filt.par_list_to_matrix_T(mod_filt.phi_T)
    phi_T_dgp = mod_dgp.par_list_to_matrix_T(mod_dgp.phi_T)
    if mod_filt.phi_tv &(not mod_dgp.phi_tv):
        phi_T_dgp = phi_T_dgp.repeat((1, phi_T_filt.shape[1]))
    elif mod_dgp.phi_tv &(not mod_filt.phi_tv):
        phi_T_filt = phi_T_filt.repeat((1, phi_T_dgp.shape[1]))

    mse_all_phi = loss_fun(phi_T_dgp, phi_T_filt).item()
    mse_phi = loss_fun(phi_T_dgp[~ phi_to_exclude, :], phi_T_filt[~ phi_to_exclude, :]).item()

    if (mod_dgp.beta_T is not None) and (mod_filt.beta_T is not None):
        beta_T_filt = mod_filt.par_list_to_matrix_T(mod_filt.beta_T)
        beta_T_dgp = mod_dgp.par_list_to_matrix_T(mod_dgp.beta_T)
        mse_beta = loss_fun(beta_T_dgp, beta_T_filt).item()
    else:
        mse_beta = 0
    
    if (mod_dgp.dist_par_un_T is not None) and (mod_filt.dist_par_un_T is not None):
        dist_par_un_T_filt = mod_filt.par_list_to_matrix_T(mod_filt.dist_par_un_T)
        dist_par_un_T_dgp = mod_dgp.par_list_to_matrix_T(mod_dgp.dist_par_un_T)
        mse_dist_par_un = loss_fun(dist_par_un_T_dgp, dist_par_un_T_filt).item()
    else:
        mse_dist_par_un = 0

    avg_beta = get_avg_beta(mod_filt)
    
    mse_dict = {f"{prefix}_mse_phi_{suffix}":mse_phi, f"{prefix}_mse_all_phi_{suffix}":mse_all_phi, f"{prefix}_mse_beta_{suffix}":mse_beta, f"{prefix}_mse_dist_par_un_{suffix}":mse_dist_par_un, f"{prefix}_avg_beta_{suffix}":avg_beta}

    return mse_dict

def get_avg_beta(mod):
    if mod.beta_T is not None:
        _,  _, beta_T = mod.get_seq_latent_par()
        avg_beta = beta_T.mean().item()
    else:
        avg_beta = None
    return avg_beta



# %% Run

if __name__ == "__main__":
    run_parallel_simulations()


if False:    
    click_args = {}
    click_args["n_sim"] = 2
    click_args["max_opt_iter"] = 21
    click_args["n_nodes"] = 50
    click_args["n_time_steps"] = 100
    click_args["type_tv_dgp_phi_bin"] = "AR"
    click_args["beta_bin_dgp_set"] = (1, "one", False)
    click_args["type_tv_dgp_beta_bin"] = "AR"
    click_args["beta_filt_set_bin"] = (None, None, None)
    click_args["exclude_weights"] = False
    click_args["type_tv_dgp_phi_w"] = "AR"
    click_args["beta_w_dgp_set"] = (0, "one", False)
    click_args["type_tv_dgp_beta_w"] = "AR"
    click_args["ext_reg_w_par_filt_set"] = (None, None, None)
    click_args["n_jobs"] = 4
    click_args["experiment_name"] = "filter missp dgp"


    kwargs = click_args

    _run_parallel_simulations(**kwargs)



# %%
