#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Friday August 27th 2021

preprocess data downloaded from http://www.sociopatterns.org/datasets/co-location-data-for-several-sociopatterns-data-sets/ .

"""



  
#%% Import packages
import mlflow
import click
import logging
import torch
import tempfile
import datetime
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from ddg_utils.mlflow import _get_and_set_experiment, get_fold_namespace
import pandas as pd
import pickle
from dynwgraphs.utils.tensortools import tens
from types import SimpleNamespace
import os
import scipy.sparse as sp
import tempfile
import tarfile
import requests
import gzip
import shutil

logger = logging.getLogger(__name__)


#%%

@click.command()
@click.option("--experiment_name", type=str, default="socio pattern data process" )
@click.option("--url_data", type=str, default="http://www.sociopatterns.org/wp-content/uploads/2018/12/co-presence.tar.gz")
@click.option("--url_meta_data", type=str, default="http://www.sociopatterns.org/wp-content/uploads/2018/12/metadata.tar")

# kwargs = {}
# kwargs["url_data"] = "http://www.sociopatterns.org/wp-content/uploads/2018/12/co-presence.tar.gz"
# kwargs["url_meta_data"] = "http://www.sociopatterns.org/wp-content/uploads/2018/12/metadata.tar"


def download_and_log_data(**kwargs):

    with mlflow.start_run() as run:
        mlflow.log_params(kwargs)
        # save files in temp folder, then log them as artifacts 
        # in mlflow and delete temp fold
        mlflow.set_tag("is_data_run", "y")
        with tempfile.TemporaryDirectory() as tmpdirname:
            # set artifacts folders and subfolders
            # tmp_fns = get_fold_namespace("./dev_test_data", ["raw", "compressed", "final"])
            tmp_fns = get_fold_namespace(tmpdirname, ["raw", "compressed", "final"])

            logger.info(f"Downloading data from {kwargs['url_data']}")

            r = requests.get(kwargs['url_data'])

            with open(tmp_fns.compressed / "co-presence.tar.gz", 'wb') as f:
                f.write(r.content)

            logger.info(f"Downloading data from {kwargs['url_meta_data']}")
            r = requests.get(kwargs['url_meta_data'])
            logger.info(f"extract data")
            with open(tmp_fns.compressed / "metadata.tar", 'wb') as f:
                f.write(r.content)


            with gzip.open(tmp_fns.compressed / "co-presence.tar.gz") as f_in:
                with open(tmp_fns.compressed / "co-presence.tar", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)                
                                
            with tarfile.open(tmp_fns.compressed / "co-presence.tar") as t:
                t.extractall(path=tmp_fns.raw)
            
            with tarfile.open(tmp_fns.compressed / "metadata.tar" , "r|*") as t:
                t.extractall(path=tmp_fns.raw)
            

            # only compressed data is committed to the repo. Needs to be unzipped before running the script
            logger.info(f"load dataframes ")
            all_data = { f[9:]:pd.read_csv(tmp_fns.raw / "co-presence" / f, sep = " ", skiprows=3, names = ["time", "source", "target"]) for f in os.listdir(tmp_fns.raw / "co-presence") }

            all_meta_data = { f[9:]:pd.read_csv(tmp_fns.raw / "metadata" / f, sep = "\t", skiprows=3, names = ["id", "group"]) for f in os.listdir(tmp_fns.raw / "metadata") }


            for f, df in all_data.items():
                logger.info(df.head(15))
                logger.info(all_meta_data[f].head(15))

            mlflow.log_artifacts(tmp_fns.main)


# %%

if __name__ == "__main__":
    download_and_log_data()


