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
@click.option("--n_jobs", type=int, default=8)
def run_sim_seq(**kwargs):
 
    check_and_tag_test_run(kwargs)

    logger.info(kwargs)
    mlflow.log_params(kwargs)
    
    run_parameters_list = [
        {
            "phi_set_dgp_type_tv_0": "AR",
            "phi_set_dgp_type_tv_1": "ref_mat",
            "phi_set_dgp_type_tv_2": 0.98,
            "phi_set_dgp_type_tv_3": 0.2,
            "phi_set_dgp_0": "2N",
            "phi_set_dgp_1": True,
            "phi_set_filt_0": "2N",
            "phi_set_filt_1": True,
            "beta_set_dgp_type_tv_0": "AR",
            "beta_set_dgp_type_tv_1": 1,
            "beta_set_dgp_type_tv_2": 0.98,
            "beta_set_dgp_type_tv_3": 0.2,
            "beta_set_dgp_type_tv_un_mean_2": -0.5,
            "beta_set_dgp_0": 1,
            "beta_set_dgp_1": "one",
            "beta_set_dgp_2": False,
            "beta_set_filt_0": 1,
            "beta_set_filt_1": "one",
            "beta_set_filt_2": False,
            "ext_reg_dgp_set_type_tv_0": "link_specific",
            "ext_reg_dgp_set_type_tv_1": "AR",
            "ext_reg_dgp_set_type_tv_2": 1,
            "ext_reg_dgp_set_type_tv_3": 0,
            "ext_reg_dgp_set_type_tv_4": 0.1,
            **kwargs
            }
        ]

    for p in run_parameters_list:
        one_run(p)


def one_run(par):
    mlflow.run(".", "run_sim_missp_dgp_and_filter_ss_sd", parameters=par, use_conda=False)



if __name__ == "__main__":
    run_sim_seq()




##
# DEVO RI RUNNARE LE SIMULAZIONI QUI SOTTO CON PARAMETRI BETA DI SEGNO DIVERSO PER VEDERE SE IGNORARE UN BETA PUà FAR CAMBIARE SEGNO ALL'ALTRO 


#####################################
# commands to get all the simulations for table 1
# python run_sim_for_table.py --n_jobs 12 --n_sim 50 --combo tphi_1reg_fphi_1reg
# python run_sim_for_table.py --n_jobs 12 --n_sim 50 --combo tphi_1reg_tphi_1reg

#####################################
# commands to get all the simulations for table 2

# python run_sim_for_table.py --n_jobs 6 --combo no_fit 2 no_fit 1  --ext_reg_dgp_persistency 0  --experiment_name "Table 2" --n_sim 50 

# python run_sim_for_table.py --n_jobs 6 --combo no_fit 2 fit_stat 1  --ext_reg_dgp_persistency 0  --experiment_name "Table 2" --n_sim 50 

# python run_sim_for_table.py --n_jobs 6 --combo no_fit 2 fit_tv 1  --ext_reg_dgp_persistency 0  --experiment_name "Table 2" --n_sim 50 

# python run_sim_for_table.py --n_jobs 6 --combo no_fit 2 no_fit 1  --ext_reg_dgp_persistency 0.98  --experiment_name "Table 2" --n_sim 50 

# python run_sim_for_table.py --n_jobs 6 --combo no_fit 2 fit_stat 1  --ext_reg_dgp_persistency 0.98  --experiment_name "Table 2" --n_sim 50 

# python run_sim_for_table.py --n_jobs 6 --combo no_fit 2 fit_tv 1  --ext_reg_dgp_persistency 0 .98 --experiment_name "Table 2" --n_sim 50 


