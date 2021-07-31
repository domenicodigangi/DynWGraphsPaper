

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday July 7th 2021

"""


#%% import packages
import mlflow
import click
import logging
logger = logging.getLogger(__name__)
from torch.functional import split
import tempfile
import datetime
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from ddg_utils.mlflow import _get_and_set_experiment

import pandas as pd




@click.command()
# load process and log data as artifact
@click.option("--load_path_emid", help="where is the original emid data located?", type=str, default="../../../../data/emid_data/juliaFiles/Weekly_eMid_Data_from_2009_06_22_to_2015_02_27.jld" )
@click.option("--load_path_eonia", help="where is the eonia rate data located?", type=str, default="../../../../data/emid_data/csvFiles/eonia_rates.csv" )

def load_and_log_data(**kwargs):

    with mlflow.start_run() as run:
        # save files in temp folder, then log them as artifacts in mlflow and delete temp fold
        with tempfile.TemporaryDirectory() as tmpdirname:

            # set artifacts folders and subfolders
            tmp_path = Path(tmpdirname)
            data_fold = tmp_path / "data"
            data_fold.mkdir(exist_ok=True)
           
            logger.info(f"loading eMid data from {kwargs['load_path_emid']}")

            ld_data = h5py.File(kwargs['load_path_emid'], "r")
            # load networks
            w_mat_T= np.transpose(np.array(ld_data["YeMidWeekly_T"]), axes=(1, 2, 0)).astype(float)
            days_obs = np.array(ld_data["days"]).astype(int)
            months_obs = np.array(ld_data["months"]).astype(int)
            years_obs = np.array(ld_data["years"]).astype(int)
            nodes = np.array(ld_data["banksIDs"])
            Y_T = w_mat_T[:, :, 2:]
            N = Y_T.shape[0]
            T = Y_T.shape[2]
            obs_dates = [datetime.datetime(year, month, day ) for year, month, day in zip(years_obs, months_obs, days_obs)]
            obs_dates = np.array(obs_dates).astype(np.datetime64)
            obs_dates=obs_dates[2:]



            logger.info(f"loading eonia data from {kwargs['load_path_eonia']}")
            df = pd.read_csv(kwargs['load_path_eonia'], skiprows=5, header=None, usecols=[0, 1], names=['date', 'rate_eonia'])

            df = df.set_index(pd.to_datetime(df['date'])).drop(columns='date')
            df = df.loc[:obs_dates[0]]
            df = df.loc[ obs_dates[-1] :]
            df =df.sort_values(by='date')
            # weekly frequency eonia rates
            eonia_week = df.resample('W').mean()
            eonia_week.index.weekday.unique()
            all_dates = np.array(eonia_week.index)


            dens_T  = (Y_T>0).mean(axis=(0, 1))
            fig, ax = plt.subplots()
            ax.plot( all_dates, np.array(eonia_week.values))
            mlflow.log_figure(fig,  "eonia_rate.png")
            fig, ax = plt.subplots()
            ax.plot( all_dates, dens_T/dens_T.mean())
            mlflow.log_figure(fig,  "eMid_dens.png")

            np.savez(data_fold / "eMid_numpy.npz", Y_T, all_dates, eonia_week, nodes)
                
            
            # log all files and sub-folders in temp fold as artifacts            
            mlflow.log_artifacts(tmp_path)

   


#%% Run
if __name__ == "__main__":
    load_and_log_data()