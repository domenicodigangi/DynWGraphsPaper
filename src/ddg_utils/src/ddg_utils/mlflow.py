#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday July 31st 2021

"""
import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.tracking.client import MlflowClient
import logging 
import os
from urllib.request import url2pathname
from urllib.parse import urlparse, unquote
import pandas as pd

logger = logging.getLogger(__name__)

# functions from https://github.com/mlflow/mlflow/blob/master/examples/multistep_workflow/main.py


def _already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        if parameters is not None:
            for param_key, param_value in parameters.items():
                run_value = full_run.data.params.get(param_key)
                if run_value != param_value:
                    match_failed = True
                    break
            if match_failed:
                continue

        if run_info.to_proto().status != RunStatus.FINISHED:
            logger.warn(
                ("Run matched, but is not FINISHED, so skipping " "(run_id=%s, status=%s)")
                % (run_info.run_id, run_info.status)
            )
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        
        if git_commit is not None:
            if git_commit != previous_version:
                logger.warn(
                    (
                        "Run matched, but has a different source version, so skipping "
                        "(found=%s, expected=%s)"
                    )
                    % (previous_version, git_commit)
                )
                continue
        return client.get_run(run_info.run_id)
    logger.warn("No matching run has been found.")
    return None


def _get_or_run(entrypoint, parameters, git_commit, use_cache=True, use_conda=False):
    existing_run = _already_ran(entrypoint, parameters, git_commit)
    if use_cache and existing_run:
        logger.info("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
        return existing_run
    logger.info("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, use_conda=use_conda)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)



# ddg's functions

def _get_and_set_experiment(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.info("Experiment not found by name. creating")
        client = mlflow.tracking.MlflowClient()
        experiment_id = client.create_experiment(experiment_name)
        experiment = client.get_experiment(experiment_id)


    mlflow.set_experiment(experiment_name)
    logger.info(f"Setting Experiment,  Name: {experiment.name}")
    logger.info("Experiment_id: {}".format(experiment.experiment_id))
    logger.info("Artifact Location: {}".format(experiment.artifact_location))
    logger.info("Tags: {}".format(experiment.tags))
    logger.info("Lifecycle_stage: {}".format(experiment.lifecycle_stage))


    return experiment


def uri_to_path(uri):
    parsed = urlparse(uri)
    host = "{0}{0}{mnt}{0}".format(os.path.sep, mnt=parsed.netloc)
    return os.path.normpath(
        os.path.join(host, url2pathname(unquote(parsed.path)))
    )


from pathlib import Path
from types import SimpleNamespace



def get_fold_namespace(dirname, subfolds_list):
    """
    create required subfolders and return namsepace
            create required subfolders and return namsepace)
    """
    fns = SimpleNamespace()
    # set artifacts folders and subfolders
    fns.main = Path(dirname)
    for sub in subfolds_list:
        fns.__dict__[sub] = fns.main / sub
        fns.__dict__[sub].mkdir(exist_ok=True)

    return fns


def check_test_exp(kwargs):
    if kwargs["max_opt_iter"] < 500:
        logger.warning("Too few opt iter. assuming this is a test run")
        kwargs["experiment_name"] = "test"


def dict_from_run(r):
    return {**r.data.params, **r.data.metrics, **r.data.tags, **dict(r.info)}


def get_df_exp(experiment):
    all_runs = MlflowClient().search_runs(experiment.experiment_id)

    df = pd.DataFrame([dict_from_run(r) for r in all_runs])
    return df
