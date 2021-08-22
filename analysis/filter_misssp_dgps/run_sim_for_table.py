#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday August 21st 2021

Run sequence of scripts to get table in filtering missp section of paper

"""

from run_sim_missp_dgp_and_filter_ss_sd import run_parallel_simulations
import click
import subprocess

@click.command()
#"Simulate missp dgp and estimate sd and ss filters"
@click.option("--n_sim", help="Number of simulations", type=int, default=80)
@click.option("--max_opt_iter", help="max number of opt iter", type=int, default=15000)
@click.option("--n_nodes", help="Number of nodes", type=int, default=50)

@click.option("--n_time_steps", help="Number of time steps", type=int, default=100)

@click.option("--size_phi_t", type=str, default="2N")

@click.option("--stop_on_error", help="shall we stop in case of error in one run? ", type=bool, default=False)

@click.option("--n_jobs", type=int, default=8)

@click.option("--experiment_name", type=str, default="sim missp filter")

@click.option("--combo", type=str, default="tt")

@click.option("--sigma_AR_phi", type=float, default=0.2)


def run_sim_seq(**kwargs):

    type_tv_dgp_phi = ("AR", "ref_mat", 0.98, kwargs["sigma_AR_phi"])

    #DGP tv Filt tv
    if kwargs["combo"] == "tt":
        subprocess.call(["python", "run_sim_missp_dgp_and_filter_ss_sd.py", "--phi_dgp_set", "2N", "True", "--phi_filt_set", "2N", "True", "--n_sim",  str(kwargs['n_sim']), "--n_jobs", str(kwargs['n_jobs']), "--max_opt_iter", str(kwargs["max_opt_iter"]), "type_tv_dgp_phi", str(type_tv_dgp_phi)])

    elif kwargs["combo"] == "ft":
    
        # #DGP static Filt tv
        subprocess.call(["python", "run_sim_missp_dgp_and_filter_ss_sd.py", "--phi_dgp_set", "2N", "False", "--phi_filt_set", "2N", "True", "--n_sim",  str(kwargs['n_sim']), "--n_jobs", str(kwargs['n_jobs']), "--max_opt_iter", str(kwargs["max_opt_iter"]), "type_tv_dgp_phi", str(type_tv_dgp_phi])
    
 
    elif kwargs["combo"] == "tf":
        # #DGP tv Filt static
        subprocess.call(["python", "run_sim_missp_dgp_and_filter_ss_sd.py", "--phi_dgp_set", "2N", "True", "--phi_filt_set", "2N", "False", "--n_sim",  str(kwargs['n_sim']), "--n_jobs", str(kwargs['n_jobs']), "--max_opt_iter", str(kwargs["max_opt_iter"]), "type_tv_dgp_phi", str(type_tv_dgp_phi])
  

    elif kwargs["combo"] == "ff":

        # #DGP static Filt static
        subprocess.call(["python", "run_sim_missp_dgp_and_filter_ss_sd.py", "--phi_dgp_set", "2N", "False", "--phi_filt_set", "2N", "False", "--n_sim",  str(kwargs['n_sim']), "--n_jobs", str(kwargs['n_jobs']), "--max_opt_iter", str(kwargs["max_opt_iter"]), "type_tv_dgp_phi", str(type_tv_dgp_phi])
    else:
        raise


if __name__ == "__main__":
    run_sim_seq()


