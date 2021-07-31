#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday July 31st 2021

"""

#%%

import click
import os
import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
from mlflow.tracking.fluent import _get_experiment_id

from ddg_utils.mlflow import _get_or_run, _get_and_set_experiment
#%%

@click.command()
@click.option("--max_opt_iter", default=10000, type=int)
@click.option("--experiment_name", type=str, default="application_eMid" )
def workflow(**kwargs):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.

    experiment = _get_and_set_experiment(f"{kwargs['experiment_name']}")

    load_and_log_data_run = _get_or_run("load_and_log_data", {}, None)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as active_run:

        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        data_npz_uri = os.path.join(load_and_log_data_run.info.artifact_uri, "data")

 
 error

if __name__ == "__main__":
    workflow()
# %%
load_and_log_data_run = _get_or_run("load_and_log_data", {}, None)
     
# %%
