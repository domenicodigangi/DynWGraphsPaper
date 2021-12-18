#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday August 21st 2021

Run sequence of scripts to get table in filtering missp section of paper

"""
from ddg_utils.mlflow import check_and_tag_test_run
import click
import mlflow
import logging
logger = logging.getLogger(__name__)


@click.command()
#"Simulate missp dgp and estimate sd and ss filters"
@click.option("--n_sim", help="Number of simulations", type=int, default=100)
@click.option("--max_opt_iter", help="max number of opt iter", type=int, default=15000)
@click.option("--stop_on_error", help="shall we stop in case of error in one run? ", type=bool, default=False)
@click.option("--n_jobs", type=int, default=10)
@click.option("--inds_to_run", default="", type=str)
@click.option("--unc_mean_beta_tab_1", default=1)
@click.option("--unc_mean_beta_obs_tab_2", default=1.5)
@click.option("--unc_mean_beta_negl_tab_2", default=2.0)
@click.option("--unc_mean_beta_lag_mat_tab_3", default=0.5)
@click.option("--dgp_ext_reg_dgp_set_type_tv_cross", default="uniform")

def run_sim_seq(**kwargs):
 
    check_and_tag_test_run(kwargs)
    inds_to_run = kwargs.pop("inds_to_run")

    
    logger.info(kwargs)
    mlflow.log_params(kwargs)
    unc_mean_beta_tab_1 = kwargs.pop("unc_mean_beta_tab_1")
    unc_mean_beta_obs_tab_2 = kwargs.pop("unc_mean_beta_obs_tab_2")
    unc_mean_beta_negl_tab_2 = kwargs.pop("unc_mean_beta_negl_tab_2")
    unc_mean_beta_lag_mat_tab_3 = kwargs.pop("unc_mean_beta_lag_mat_tab_3")
    dgp_ext_reg_dgp_set_type_tv_cross = kwargs.pop("dgp_ext_reg_dgp_set_type_tv_cross")

    run_parameters_list = [
        {# dgp: AR fit 1 reg. filt: Const fit 1 reg - Tab 1
            "phi_set_dgp_type_tv_0": "AR",
            "phi_set_dgp_type_tv_1": "ref_mat",
            "phi_set_dgp_type_tv_2": 0.98,
            "phi_set_dgp_type_tv_3": 0.2,
            "phi_set_dgp_0": "2N",
            "phi_set_dgp_1": True,
            "phi_set_filt_0": "2N",
            "phi_set_filt_1": False,
            "beta_set_dgp_type_tv_0": "AR",
            "beta_set_dgp_type_tv_1": unc_mean_beta_tab_1,
            "beta_set_dgp_type_tv_2": 0,
            "beta_set_dgp_type_tv_3": 0,
            "beta_set_dgp_0": 1,
            "beta_set_dgp_1": "one",
            "beta_set_dgp_2": False,
            "beta_set_filt_0": 1,
            "beta_set_filt_1": "one",
            "beta_set_filt_2": False,
            "ext_reg_dgp_set_type_tv_0": dgp_ext_reg_dgp_set_type_tv_cross,
            "ext_reg_dgp_set_type_tv_1": "AR",
            "ext_reg_dgp_set_type_tv_2": 1,
            "ext_reg_dgp_set_type_tv_3": 0.98,
            "ext_reg_dgp_set_type_tv_4": 0.1,
            **kwargs
            },
        {# dgp: AR fit 1 reg. filt: SD fit 1 reg -  Tab 1
            "phi_set_dgp_type_tv_0": "AR",
            "phi_set_dgp_type_tv_1": "ref_mat",
            "phi_set_dgp_type_tv_2": 0.98,
            "phi_set_dgp_type_tv_3": 0.2,
            "phi_set_dgp_0": "2N",
            "phi_set_dgp_1": True,
            "phi_set_filt_0": "2N",
            "phi_set_filt_1": True,
            "beta_set_dgp_type_tv_0": "AR",
            "beta_set_dgp_type_tv_1": unc_mean_beta_tab_1,
            "beta_set_dgp_type_tv_2": 0,
            "beta_set_dgp_type_tv_3": 0,
            "beta_set_dgp_0": 1,
            "beta_set_dgp_1": "one",
            "beta_set_dgp_2": False,
            "beta_set_filt_0": 1,
            "beta_set_filt_1": "one",
            "beta_set_filt_2": False,
            "ext_reg_dgp_set_type_tv_0": dgp_ext_reg_dgp_set_type_tv_cross,
            "ext_reg_dgp_set_type_tv_1": "AR",
            "ext_reg_dgp_set_type_tv_2": 1,
            "ext_reg_dgp_set_type_tv_3": 0.98,
            "ext_reg_dgp_set_type_tv_4": 0.1,
            **kwargs
            },
        {# dgp: AR fit 2 reg. X~WN filt: No fit 1 reg - Tab 2
            "phi_set_dgp_type_tv_0": "AR",
            "phi_set_dgp_type_tv_1": "ref_mat",
            "phi_set_dgp_type_tv_2": 0.98,
            "phi_set_dgp_type_tv_3": 0.2,
            "phi_set_dgp_0": "2N",
            "phi_set_dgp_1": True,
            "phi_set_filt_0": "0",
            "phi_set_filt_1": False,
            "beta_set_dgp_type_tv_0": "AR",
            "beta_set_dgp_type_tv_1": unc_mean_beta_obs_tab_2,
            "beta_set_dgp_type_tv_2": 0,
            "beta_set_dgp_type_tv_3": 0,
            "beta_set_dgp_type_tv_un_mean_2": unc_mean_beta_negl_tab_2,
            "beta_set_dgp_0": 2,
            "beta_set_dgp_1": "one",
            "beta_set_dgp_2": False,
            "beta_set_filt_0": 1,
            "beta_set_filt_1": "one",
            "beta_set_filt_2": False,
            "ext_reg_dgp_set_type_tv_0": dgp_ext_reg_dgp_set_type_tv_cross,
            "ext_reg_dgp_set_type_tv_1": "AR",
            "ext_reg_dgp_set_type_tv_2": 1,
            "ext_reg_dgp_set_type_tv_3": 0,
            "ext_reg_dgp_set_type_tv_4": 0.1,
            **kwargs
            },
        {# dgp: AR fit 2 reg. X~WN filt: const fit 1 reg - Tab 2
            "phi_set_dgp_type_tv_0": "AR",
            "phi_set_dgp_type_tv_1": "ref_mat",
            "phi_set_dgp_type_tv_2": 0.98,
            "phi_set_dgp_type_tv_3": 0.2,
            "phi_set_dgp_0": "2N",
            "phi_set_dgp_1": True,
            "phi_set_filt_0": "2N",
            "phi_set_filt_1": False,
            "beta_set_dgp_type_tv_0": "AR",
            "beta_set_dgp_type_tv_1": unc_mean_beta_obs_tab_2,
            "beta_set_dgp_type_tv_2": 0,
            "beta_set_dgp_type_tv_3": 0,
            "beta_set_dgp_type_tv_un_mean_2": unc_mean_beta_negl_tab_2,
            "beta_set_dgp_0": 2,
            "beta_set_dgp_1": "one",
            "beta_set_dgp_2": False,
            "beta_set_filt_0": 1,
            "beta_set_filt_1": "one",
            "beta_set_filt_2": False,
            "ext_reg_dgp_set_type_tv_0": dgp_ext_reg_dgp_set_type_tv_cross,
            "ext_reg_dgp_set_type_tv_1": "AR",
            "ext_reg_dgp_set_type_tv_2": 1,
            "ext_reg_dgp_set_type_tv_3": 0,
            "ext_reg_dgp_set_type_tv_4": 0.1,
            **kwargs
            },
        {# dgp: AR fit 2 reg. X~WN filt: SD fit 1 reg - Tab 2
            "phi_set_dgp_type_tv_0": "AR",
            "phi_set_dgp_type_tv_1": "ref_mat",
            "phi_set_dgp_type_tv_2": 0.98,
            "phi_set_dgp_type_tv_3": 0.2,
            "phi_set_dgp_0": "2N",
            "phi_set_dgp_1": True,
            "phi_set_filt_0": "2N",
            "phi_set_filt_1": True,
            "beta_set_dgp_type_tv_0": "AR",
            "beta_set_dgp_type_tv_1": unc_mean_beta_obs_tab_2,
            "beta_set_dgp_type_tv_2": 0,
            "beta_set_dgp_type_tv_3": 0,
            "beta_set_dgp_type_tv_un_mean_2": unc_mean_beta_negl_tab_2,
            "beta_set_dgp_0": 2,
            "beta_set_dgp_1": "one",
            "beta_set_dgp_2": False,
            "beta_set_filt_0": 1,
            "beta_set_filt_1": "one",
            "beta_set_filt_2": False,
            "ext_reg_dgp_set_type_tv_0": dgp_ext_reg_dgp_set_type_tv_cross,
            "ext_reg_dgp_set_type_tv_1": "AR",
            "ext_reg_dgp_set_type_tv_2": 1,
            "ext_reg_dgp_set_type_tv_3": 0,
            "ext_reg_dgp_set_type_tv_4": 0.1,
            **kwargs
            },
        {# dgp: AR fit 2 reg. X~AR filt: No fit 1 reg - Tab 2
            "phi_set_dgp_type_tv_0": "AR",
            "phi_set_dgp_type_tv_1": "ref_mat",
            "phi_set_dgp_type_tv_2": 0.98,
            "phi_set_dgp_type_tv_3": 0.2,
            "phi_set_dgp_0": "2N",
            "phi_set_dgp_1": True,
            "phi_set_filt_0": "0",
            "phi_set_filt_1": False,
            "beta_set_dgp_type_tv_0": "AR",
            "beta_set_dgp_type_tv_1": unc_mean_beta_obs_tab_2,
            "beta_set_dgp_type_tv_2": 0,
            "beta_set_dgp_type_tv_3": 0,
            "beta_set_dgp_type_tv_un_mean_2": unc_mean_beta_negl_tab_2,
            "beta_set_dgp_0": 2,
            "beta_set_dgp_1": "one",
            "beta_set_dgp_2": False,
            "beta_set_filt_0": 1,
            "beta_set_filt_1": "one",
            "beta_set_filt_2": False,
            "ext_reg_dgp_set_type_tv_0": dgp_ext_reg_dgp_set_type_tv_cross,
            "ext_reg_dgp_set_type_tv_1": "AR",
            "ext_reg_dgp_set_type_tv_2": 1,
            "ext_reg_dgp_set_type_tv_3": 0.98,
            "ext_reg_dgp_set_type_tv_4": 0.1,
            **kwargs
            },
        {# dgp: AR fit 2 reg. X~AR filt: const fit 1 reg - Tab 2
            "phi_set_dgp_type_tv_0": "AR",
            "phi_set_dgp_type_tv_1": "ref_mat",
            "phi_set_dgp_type_tv_2": 0.98,
            "phi_set_dgp_type_tv_3": 0.2,
            "phi_set_dgp_0": "2N",
            "phi_set_dgp_1": True,
            "phi_set_filt_0": "2N",
            "phi_set_filt_1": False,
            "beta_set_dgp_type_tv_0": "AR",
            "beta_set_dgp_type_tv_1": unc_mean_beta_obs_tab_2,
            "beta_set_dgp_type_tv_2": 0,
            "beta_set_dgp_type_tv_3": 0,
            "beta_set_dgp_type_tv_un_mean_2": unc_mean_beta_negl_tab_2,
            "beta_set_dgp_0": 2,
            "beta_set_dgp_1": "one",
            "beta_set_dgp_2": False,
            "beta_set_filt_0": 1,
            "beta_set_filt_1": "one",
            "beta_set_filt_2": False,
            "ext_reg_dgp_set_type_tv_0": dgp_ext_reg_dgp_set_type_tv_cross,
            "ext_reg_dgp_set_type_tv_1": "AR",
            "ext_reg_dgp_set_type_tv_2": 1,
            "ext_reg_dgp_set_type_tv_3": 0.98,
            "ext_reg_dgp_set_type_tv_4": 0.1,
            **kwargs
            },
        {# dgp: AR fit 2 reg. X~AR filt: SD fit 1 reg - Tab 2
            "phi_set_dgp_type_tv_0": "AR",
            "phi_set_dgp_type_tv_1": "ref_mat",
            "phi_set_dgp_type_tv_2": 0.98,
            "phi_set_dgp_type_tv_3": 0.2,
            "phi_set_dgp_0": "2N",
            "phi_set_dgp_1": True,
            "phi_set_filt_0": "2N",
            "phi_set_filt_1": True,
            "beta_set_dgp_type_tv_0": "AR",
            "beta_set_dgp_type_tv_1": unc_mean_beta_obs_tab_2,
            "beta_set_dgp_type_tv_2": 0,
            "beta_set_dgp_type_tv_3": 0,
            "beta_set_dgp_type_tv_un_mean_2": unc_mean_beta_negl_tab_2,
            "beta_set_dgp_0": 2,
            "beta_set_dgp_1": "one",
            "beta_set_dgp_2": False,
            "beta_set_filt_0": 1,
            "beta_set_filt_1": "one",
            "beta_set_filt_2": False,
            "ext_reg_dgp_set_type_tv_0": dgp_ext_reg_dgp_set_type_tv_cross,
            "ext_reg_dgp_set_type_tv_1": "AR",
            "ext_reg_dgp_set_type_tv_2": 1,
            "ext_reg_dgp_set_type_tv_3": 0.98,
            "ext_reg_dgp_set_type_tv_4": 0.1,
            **kwargs
            },
        {# dgp: AR fit 1 reg lagged matric.: SD fit 1 reg - Tab 3
            "phi_set_dgp_type_tv_0": "AR",
            "phi_set_dgp_type_tv_1": "ref_mat",
            "phi_set_dgp_type_tv_2": 0.98,
            "phi_set_dgp_type_tv_3": 0.2,
            "phi_set_dgp_0": "2N",
            "phi_set_dgp_1": True,
            "phi_set_filt_0": "2N",
            "phi_set_filt_1": True,
            "beta_set_dgp_type_tv_0": "AR",
            "beta_set_dgp_type_tv_1": unc_mean_beta_lag_mat_tab_3,
            "beta_set_dgp_type_tv_2": 0,
            "beta_set_dgp_type_tv_3": 0,
            "beta_set_dgp_type_tv_un_mean_2": 0.0,
            "beta_set_dgp_0": 1,
            "beta_set_dgp_1": "one",
            "beta_set_dgp_2": False,
            "beta_set_filt_0": 1,
            "beta_set_filt_1": "one",
            "beta_set_filt_2": False,
            "ext_reg_dgp_set_type_tv_0": "uniform",
            "ext_reg_dgp_set_type_tv_1": "AR",
            "ext_reg_dgp_set_type_tv_2": 1,
            "ext_reg_dgp_set_type_tv_3": 0,
            "ext_reg_dgp_set_type_tv_4": 0,
            "use_lag_mat_as_reg": True,
            **kwargs
            }
        ]

    
    if inds_to_run != '':
        inds_to_run = eval(", ".join(inds_to_run.split("_")))
        if type(inds_to_run) == int:
            inds_to_run = [inds_to_run]
        run_parameters_list = [run_parameters_list[i] for i in inds_to_run]
        logger.warning(f"Going to run only: {run_parameters_list}")   
    
    for p in run_parameters_list:
        one_run(p)


def one_run(par):
    mlflow.run(".", "run_sim_missp_dgp_and_filter_ss_sd", parameters=par, use_conda=False)



if __name__ == "__main__":
    run_sim_seq()