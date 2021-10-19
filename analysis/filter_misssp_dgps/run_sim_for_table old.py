#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Thursday September 2nd 2021

"""


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
@click.option("--n_sim", help="Number of simulations", type=int, default=100)
@click.option("--max_opt_iter", help="max number of opt iter", type=int, default=15000)

@click.option("--size_phi_t", type=str, default="2N")

@click.option("--stop_on_error", help="shall we stop in case of error in one run? ", type=bool, default=False)

@click.option("--n_jobs", type=int, default=8)

@click.option("--experiment_name", type=str, default="Table 1")

@click.option("--combo", type=(str, str, str, str), default=("fit_tv", "1", "fit_tv", "1"))

@click.option("--sigma_ar_phi", type=float, default=0.2)

@click.option("--ext_reg_dgp_cross_type", type=str, default="link_specific")

@click.option("--ext_reg_dgp_persistency", type=float, default=0.0)


def run_sim_seq(**kwargs):
    phi_set_dgp_type_tv = ("AR", "ref_mat", 0.98, kwargs["sigma_ar_phi"])

    kwargs["experiment_name"] = kwargs["experiment_name"] + " " + kwargs["ext_reg_dgp_cross_type"]

    combo = kwargs["combo"]

    if combo[0] == "fit_tv":
        size_phi_t_dgp = "2N"
        phi_dgp_tv = True
    elif combo[0] == "fit_stat":
        size_phi_t_dgp = "2N"
        phi_dgp_tv = False
    elif combo[0] in ["None", "no_fit"]:
        phi_dgp_tv = False
        size_phi_t_dgp = "0"

    n_reg_dgp = int(combo[1])
    if int(n_reg_dgp) == 0:
        size_beta_t_dgp = "0"
    else:
        size_beta_t_dgp = "one"

    if combo[2] == "fit_tv":
        size_phi_t_filt = "2N"
        phi_filt_tv = True
    elif combo[2] == "fit_stat":
        size_phi_t_filt = "2N"
        phi_filt_tv = False
    elif combo[2] in ["None", "no_fit"]:
        phi_filt_tv = False
        size_phi_t_filt = "0"

    n_reg_filt = int(combo[3])
    if int(n_reg_filt) == 0:
        size_beta_t_filt = "0"
    else:
        size_beta_t_filt = "one"

    subprocess.call(["python", "run_sim_missp_dgp_and_filter_ss_sd.py", "--stop_on_error", str(kwargs["stop_on_error"]), "--experiment_name", kwargs["experiment_name"],  "--phi_set_dgp", size_phi_t_dgp, str(phi_dgp_tv), "--phi_set_filt", size_phi_t_filt, str(phi_filt_tv), "--n_sim",  str(kwargs['n_sim']), "--n_jobs", str(kwargs['n_jobs']), "--max_opt_iter", str(kwargs["max_opt_iter"]), "--ext_reg_dgp_set_type_tv", kwargs["ext_reg_dgp_cross_type"], "AR", str(1), str(kwargs["ext_reg_dgp_persistency"]), str(0.1), "--beta_set_dgp", str(n_reg_dgp), size_beta_t_dgp, "False",  "--beta_set_filt", str(n_reg_filt), size_beta_t_filt, "False",  "--phi_set_dgp_type_tv"] + [str(v) for v in phi_set_dgp_type_tv])


if __name__ == "__main__":
    run_sim_seq()



#####################################
# commands to get all the simulations for table 1
# python run_sim_for_table.py --n_jobs 12 --n_sim 100 --combo tphi_1reg_fphi_1reg
# python run_sim_for_table.py --n_jobs 12 --n_sim 100 --combo tphi_1reg_tphi_1reg

#####################################
# commands to get all the simulations for table 2

# python run_sim_for_table.py --n_jobs 6 --combo no_fit 2 no_fit 1  --ext_reg_dgp_persistency 0  --experiment_name "Table 2" --n_sim 50 

# python run_sim_for_table.py --n_jobs 6 --combo no_fit 2 fit_stat 1  --ext_reg_dgp_persistency 0  --experiment_name "Table 2" --n_sim 50 

# python run_sim_for_table.py --n_jobs 6 --combo no_fit 2 fit_tv 1  --ext_reg_dgp_persistency 0  --experiment_name "Table 2" --n_sim 50 

# python run_sim_for_table.py --n_jobs 6 --combo no_fit 2 no_fit 1  --ext_reg_dgp_persistency 0.98  --experiment_name "Table 2" --n_sim 50 

# python run_sim_for_table.py --n_jobs 6 --combo no_fit 2 fit_stat 1  --ext_reg_dgp_persistency 0.98  --experiment_name "Table 2" --n_sim 50 

# python run_sim_for_table.py --n_jobs 6 --combo no_fit 2 fit_tv 1  --ext_reg_dgp_persistency 0 .98 --experiment_name "Table 2" --n_sim 50 


