

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
from dynwgraphs.utils.dgps import get_mod_and_par
import tempfile
import mlflow
from joblib import Parallel, delayed
from torch import nn
import logging
import click
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
@click.option("--ext_reg_dgp_set", help=" common dgp options for bin and w ", type=(int, str, bool), default=(None, None, None))
@click.option("--ext_reg_filt_set", help="common filter options for bin and w ", type=(int, str, bool), default=(None, None, None))
@click.option("--type_dgp_phi", help="common setting for bin and w", type=str, default=None)
@click.option("--type_dgp_phi_bin", help="what kind of dgp should phi_T follow. AR or const_unif_prob", type=str, default="AR")
@click.option("--ext_reg_bin_dgp_set", help="Options for external regressors and model specification. In order : 1 number of external regressors,  2 size of beta(One for all, N : one per node (both in and out), 2N : two per node, one for in and one for out links), 3 beta_tv (should the regression coefficients, of both dgp and filter, be time varying ?)) ", type=(int, str, bool), default=(1, "one", False))
@click.option("--type_dgp_beta", help="common setting for bin and w", default=None)
@click.option("--type_dgp_beta_bin", help="what kind of dgp should beta_T follow", default="AR")

@click.option("--ext_reg_bin_filt_set", help="Options for external regressors and filter specification. In order : 1 number of external regressors,  2 size of beta(One for all, N : one per node (both in and out), 2N : two per node, one for in and one for out links), 3 beta_tv (should the regression coefficients, of both dgp and filter, be time varying ?)", type=(int, str, bool), default=(0, "one", False))

@click.option("--exclude_weights", help="shall we run the sim only for the binary case? ", type=bool, default=False)
@click.option("--type_dgp_phi_w", help="what kind of dgp should phi_T follow ", type=str, default="AR")
@click.option("--ext_reg_w_dgp_set", help="Options for external regressors and model specification. In order : 1 number of external regressors,  2 size of beta(One for all, N : one per node (both in and out), 2N : two per node, one for in and one for out links), 3 beta_tv (should the regression coefficients, of both dgp and filter, be time varying ?)", type=(int, str, bool), default=(1, "one", False))
@click.option("--type_dgp_beta_w", help="what kind of dgp should beta_T follow", default="AR")

@click.option("--ext_reg_w_filt_set", help="Options for external regressors and filter specification. In order : 1 number of external regressors,  2 size of beta(One for all, N : one per node (both in and out), 2N : two per node, one for in and one for out links), 3 beta_tv (should the regression coefficients, of both dgp and filter, be time varying ?)", type=(int, str, bool), default=(0, "one", False))

@click.option("--n_jobs", type=int, default=4)
@click.option("--experiment_name", type=str, default="filter missp dgp")


def run_parallel_simulations(**kwargs):
    return _run_parallel_simulations(**kwargs)


def cli_reg_set_to_num_mod_par(N: int, reg_set, max_opt_iter):
    """
    convert the regressors settings from the command line to a format that can be given as a model input
    """

    cli_input_name = ["n_ext_reg", "size_beta_t_str", "all_beta_tv"]
    reg_set_dict = {k: v for k, v in zip(cli_input_name, reg_set)}

    out_dict = {}
    out_dict["n_ext_reg"] = reg_set_dict["n_ext_reg"]
    if reg_set_dict["n_ext_reg"] == 0:
        out_dict["size_beta_t"] = None
        out_dict["beta_tv"] = None
    else:
        if reg_set_dict["size_beta_t_str"] == "one":
            out_dict["size_beta_t"] = 1
        elif reg_set_dict["size_beta_t_str"] == "N":
            out_dict["size_beta_t"] = N
        elif reg_set_dict["size_beta_t_str"] == "2N":
            out_dict["size_beta_t"] = 2*N
        else:
            raise
        out_dict["beta_tv"] = [reg_set_dict["all_beta_tv"] for p in range(out_dict["n_ext_reg"])]

    out_dict["max_opt_iter"] = max_opt_iter

    return out_dict


def get_dgp_and_filt_specification(kwargs, bin_or_w):
    N = kwargs["n_nodes"]

    logger.info("using common external regressor settings for binary and weighted parameters")
    
    if kwargs[f"ext_reg_dgp_set"][0] is not None:
        kwargs[f"ext_reg_bin_dgp_set"] = kwargs[f"ext_reg_dgp_set"]
        kwargs[f"ext_reg_w_dgp_set"] = kwargs[f"ext_reg_dgp_set"]
    if kwargs[f"ext_reg_filt_set"][0] is not None:
        kwargs[f"ext_reg_bin_filt_set"] = kwargs[f"ext_reg_filt_set"]
        kwargs[f"ext_reg_w_filt_set"] = kwargs[f"ext_reg_filt_set"]

    dgp_set = cli_reg_set_to_num_mod_par(N, kwargs[f"ext_reg_{bin_or_w}_dgp_set"], max_opt_iter=kwargs["max_opt_iter"])

    if kwargs[f"ext_reg_{bin_or_w}_filt_set"][0] is None:
        filt_set = dgp_set
    else:
        filt_set = cli_reg_set_to_num_mod_par(N, kwargs[f"ext_reg_{bin_or_w}_filt_set"], max_opt_iter=kwargs["max_opt_iter"])

    if kwargs[f"type_dgp_phi"] is not None: 
        dgp_set["type_dgp_phi"] = kwargs[f"type_dgp_phi"]
    else:
        dgp_set["type_dgp_phi"] = kwargs[f"type_dgp_phi_{bin_or_w}"]
    
    if kwargs[f"type_dgp_beta"] is not None: 
        dgp_set["type_dgp_beta"] = kwargs[f"type_dgp_beta"]
    else:
        dgp_set["type_dgp_beta"] = kwargs[f"type_dgp_beta_{bin_or_w}"]

    filt_set = drop_keys(filt_set, ["n_ext_reg"])
    
    return dgp_set, filt_set    


def _run_parallel_simulations(**kwargs):

    check_test_exp(kwargs)
    T = kwargs["n_time_steps"]
    N = kwargs["n_nodes"]

    dgp_set_bin, filt_set_bin = get_dgp_and_filt_specification(kwargs, "bin")
    
    if not kwargs["exclude_weights"]:
        dgp_set_w, filt_set_w = get_dgp_and_filt_specification(kwargs, "w")
        
    experiment = _get_and_set_experiment(kwargs["experiment_name"])

    with mlflow.start_run(experiment_id=experiment.experiment_id) as parent_run:

        parent_runs_par = kwargs
        mlflow.log_params(drop_keys(kwargs,  ["type_dgp_phi_bin", "type_dgp_phi_w", "type_dgp_beta_bin", "type_dgp_beta_w"]))
        mlflow.log_params({f"{key}_bin": val for key, val in dgp_set_bin.items() if key != "X_T"})
        if not kwargs["exclude_weights"]:
            mlflow.log_params({f"{key}_w": val for key, val in dgp_set_w.items() if key != "X_T"})
 
        dgp_par = {"N": N, "T": T}
        filt_par = {"bin": filt_set_bin}

        # define binary dgp and filter par
        mod_dgp_bin, _, Y_reference_bin = get_mod_and_par(N=N, T=T, model="dirBin1", dgp_set_dict=dgp_set_bin)
        mod_dgp_dict = {"bin": mod_dgp_bin}
        dgp_par["bin"] = dgp_set_bin
        run_data_dict = {"Y_reference":{"Y_reference_bin": Y_reference_bin}}

        if not kwargs["exclude_weights"]:
            # define weighted dgp
            mod_dgp_w, _, Y_reference_w = get_mod_and_par(N=N, T=T, model="dirSpW1", dgp_set_dict=dgp_set_w)
            
            mod_dgp_w.bin_mod = mod_dgp_bin
            mod_dgp_dict["w"] = mod_dgp_w
            dgp_par["w"] = dgp_set_w 
            run_data_dict["Y_reference"]["Y_reference_w"] = Y_reference_w

            filt_par["w"] = filt_set_w

        run_par_dict = {"dgp_par": dgp_par, "filt_par": filt_par}


        def try_one_run(mod_dgp_dict, run_par_dict, run_data_dict, parent_run, parent_runs_par):
            # sample_estimate_and_log(mod_dgp_dict, run_par_dict, run_data_dict, parent_run, parent_runs_par)
            try:
                # pass
                sample_estimate_and_log(mod_dgp_dict, run_par_dict, run_data_dict, parent_run, parent_runs_par)
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
                mlflow.log_params({f"{'dgp_'}{key}": val for key, val in run_par_dict["dgp_par"].items()})
                mlflow.log_params({f"{'filt_'}{key}": val for key, val in run_par_dict["filt_par"].items()})

                bin_or_w, mod_dgp = list(mod_dgp_dict.items())[0]
                for bin_or_w, mod_dgp in mod_dgp_dict.items():
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
                    filt_models = get_filt_mod(bin_or_w, mod_dgp.Y_T, mod_dgp.X_T, run_par_dict["filt_par"][bin_or_w])
                    
                    # k_filt, mod = list(filt_models.items())[0]
                    for k_filt, mod_filt in filt_models.items():

                        _, h_par_opt, stats_opt = mod_filt.estimate(tb_save_fold=tb_fold)

                        mlflow.log_params({f"{bin_or_w}_{k_filt}_{key}": val for key, val in h_par_opt.items()})
                        mlflow.log_params({f"{bin_or_w}_{k_filt}_{key}": val for key, val in stats_opt.items()})
                        mlflow.log_params({f"{bin_or_w}_{k_filt}_{key}": val for key, val in mod_filt.get_info_dict().items() if key not in h_par_opt.keys()})

                        mod_filt.save_parameters(save_path=tmp_path)
                    
                        # compute mse for each model and log it 
                        phi_to_exclude = strIO_from_tens_T(mod_dgp.Y_T) < 1e-3 
                        mse_dict = filt_err(mod_dgp, mod_filt, phi_to_exclude, suffix=k_filt, prefix=bin_or_w)
                        mlflow.log_metrics(mse_dict) 
                    
                        # log plots that can be useful for quick visual diagnostic 
                        mlflow.log_figure(mod_filt.plot_phi_T()[0], f"fig/{bin_or_w}_{k_filt}_filt_all.png")

                        i = torch.where(~splitVec(phi_to_exclude)[0])[0][0]

                        mlflow.log_figure(mod_filt.plot_phi_T(i=i, fig_ax= mod_dgp.plot_phi_T(i=i))[0], f"fig/{bin_or_w}_{k_filt}_filt_phi_ind_{i}.png")

                        if mod_dgp.any_beta_tv():
                            plot_dgp_fig_ax = mod_dgp.plot_beta_T()
                            mlflow.log_figure(plot_dgp_fig_ax[0], f"fig/{bin_or_w}_sd_filt_beta_T.png")
                        if mod_filt.any_beta_tv():
                            mlflow.log_figure(mod_filt.plot_beta_T(fig_ax=plot_dgp_fig_ax)[0], f"fig/{bin_or_w}_{k_filt}_filt_beta_T.png")
                            
                # log all files and sub-folders in temp fold as artifacts            
                mlflow.log_artifacts(tmp_path)


def get_filt_mod(bin_or_w, Y_T, X_T, par_dict):
    
    if par_dict["size_beta_t"] is not None:
        if par_dict["size_beta_t"] > 0 :
            par_dict["X_T"] = X_T

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

def filt_err(mod_dgp, mod_filt, phi_to_exclude, suffix="", prefix=""):
    
    loss_fun = nn.MSELoss()
    phi_T_filt = mod_filt.par_list_to_matrix_T(mod_filt.phi_T)
    phi_T_dgp = mod_dgp.par_list_to_matrix_T(mod_dgp.phi_T)
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
    
    mse_dict = {f"{prefix}_mse_phi_{suffix}":mse_phi, f"{prefix}_mse_all_phi_{suffix}":mse_all_phi, f"{prefix}_mse_beta_{suffix}":mse_beta, f"{prefix}_mse_dist_par_un_{suffix}":mse_dist_par_un}

    return mse_dict



# %% Run

if __name__ == "__main__":
    run_parallel_simulations()


if False:    
    click_args = {}
    click_args["n_sim"] = 2
    click_args["max_opt_iter"] = 21
    click_args["n_nodes"] = 50
    click_args["n_time_steps"] = 100
    click_args["type_dgp_phi_bin"] = "AR"
    click_args["ext_reg_bin_dgp_set"] = (1, "one", False)
    click_args["type_dgp_beta_bin"] = "AR"
    click_args["ext_reg_bin_filt_set"] = (None, None, None)
    click_args["exclude_weights"] = False
    click_args["type_dgp_phi_w"] = "AR"
    click_args["ext_reg_w_dgp_set"] = (0, "one", False)
    click_args["type_dgp_beta_w"] = "AR"
    click_args["ext_reg_w_filt_set"] = (None, None, None)
    click_args["n_jobs"] = 4
    click_args["experiment_name"] = "filter missp dgp"


    kwargs = click_args

    _run_parallel_simulations(**kwargs)



# %%
