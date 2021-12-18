#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Friday August 27th 2021

"""


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
from proj_utils.mlflow import _get_and_set_experiment, get_fold_namespace
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
from proj_utils.mlflow import _get_or_run, _get_and_set_experiment, uri_to_path
import scipy.sparse as sp

logger = logging.getLogger(__name__)


#%%

@click.command()
@click.option("--experiment_name", type=str, default="socio pattern data process" )
@click.option("--freq_agg", type=str, default="30m" )
@click.option("--data_name", type=str, default="InVS15" )

kwargs = {}
kwargs["data_name"] = "InVS15" 


def load_and_proc_data(**kwargs):

    kwargs["data_name"] = kwargs["data_name"].lower()
    data_run = _get_or_run("download_data", None, None)

    with mlflow.start_run() as run:
        mlflow.log_params(kwargs)
        # save files in temp folder, then log them as artifacts 
        # in mlflow and delete temp fold
        mlflow.set_tag("is_data_run", "y")
        with tempfile.TemporaryDirectory() as tmpdirname:
            # set artifacts folders and subfolders
            # tmp_fns = get_fold_namespace("./dev_test_data", ["raw", "compressed", "final"])
            
            ld_path = Path(uri_to_path(data_run.info.artifact_uri)) / "raw"
            # only compressed data is committed to the repo. Needs to be unzipped before running the script
            logger.info(f"load dataframes ")

            all_data = {f[9:-4].lower(): pd.read_csv(ld_path / "co-presence" / f, sep = " ", skiprows=3, names = ["time", "source", "target"]) for f in os.listdir(ld_path / "co-presence") if f[9:-4].lower() in kwargs["data_name"].lower()}
            
            all_meta_data = {f[9:-4].lower(): pd.read_csv(ld_path / "metadata" / f, sep = "\t", skiprows=3, names = ["id", "group"]) for f in os.listdir(ld_path / "metadata") if f[9:-4].lower() in kwargs["data_name"].lower()}

                            
            df = all_data[kwargs["data_name"]]

            logger.info(df.head(15))


            #drop duplicates
            print(f"number directed links {name} {df.shape}")
            
            meta = all_meta_data[name]

            df["datetime"] = pd.to_datetime(df["time"], origin="2013-06-24", unit="s")

            df = df.sort_values(by="datetime")


            df_lf = df.groupby(["source", "target"]).resample( "30min", on="datetime").agg({"time": "count"}).rename(columns={"time": "count"})

            mlflow.log_figure(plt.hist(df_lf["count"][df_lf["count"]!=0], log=True))


            T = df_lf.index.get_level_values("datetime").unique().shape[0]

            all_nodes_s = df_lf.index.get_level_values("source").unique().values
            all_nodes_t = df_lf.index.get_level_values("target").unique().values
            all_nodes_inds = np.unique(np.hstack((all_nodes_s, all_nodes_t)))

            df_inds = pd.DataFrame({"inds_orig": all_nodes_inds}).reset_index()

            df_lf = df_lf.reset_index()

            df_1 = df_lf.merge(df_inds, left_on="source", right_on="inds_orig" ).rename(columns = {"index": "ind_source"}).drop(columns = "inds_orig").merge(df_inds, left_on="target", right_on="inds_orig" ).rename(columns = {"index": "ind_target"}).drop(columns = "inds_orig")


            N = all_nodes_inds.shape[0]

            Y_T_list = []
            list(df_1.iloc[1:10,:].iterrows())[1][2]
            #%%
            all_T = []
            for time, df in df_1.groupby("datetime"):
                all_T.append(time)
                Y_t = torch.zeros(N, N, dtype=bool) 
                for ir, r in df.iterrows():
                    Y_t[r.ind_source, r.ind_target] = True

                Y_T_list.append(Y_t)

            Y_T = torch.stack(Y_T_list, dim=2)



            mlflow.log_artifacts(tmp_fns.main)


# %%

if __name__ == "__main__":
    load_and_proc_data()



for (name, df) in all_data.items():
name = "InVS13.dat"
df = all_data[name]


all_links = all_links.merge(meta, how = "left", left_on="source", right_on = "id").rename(columns = {"group":"group_source"}).drop(columns = "id")    

all_links = all_links.merge(meta, how = "left", left_on="target", right_on = "id").rename(columns = {"group":"group_target"}).drop(columns = "id")        

n_links = all_links.shape[0]



# def oth():
#           
#             pickle.dump(data_to_save, open(tmp_fns.data / "emid_data.pkl", "wb"))
                
            
#             # log all files and sub-folders in temp fold as artifacts            
#             mlflow.log_artifacts(tmp_fns.main)


# #%% create undirected links and drop duplicates 
# save_direct = "./data_for_K_ising/"

# name, df = list(all_data.items())[0]


#     pickle.dump(data, open(save_direct+ name[:-4] + ".pkl", "wb"))

#     print(f"saved {name} ")


# name, d = list(all_data.items())[0] 
# #from sparse matrix of 0s and 1s to full matrix of -1s and 1s 
# s_T_all_days = torch.tensor(d["links_ts_sparse"].todense(), dtype=dtype)
# s_T_all_days[s_T_all_days == 0] = -1
# inds_sorted = torch.sort(-s_T_all_days.mean(axis=0))[1]
# s_T_all_days = s_T_all_days[:, inds_sorted]

# plt.figure()
# plt.plot((s_T_all_days).mean(axis=1), ".")
# plt.figure()
# plt.plot((s_T_all_days).mean(axis=0), ".")

# df_links = d["df_links"]
# # inds_sorted = torch.sort(s_T_all_days.mean(axis=0))
# # n_links =[1] 100
# # plt.plot(inds_sorted[0],  ".")
# # s_T_all_days_active = s_T_all_days[:,inds_sorted[-n_links:]]

# # plt.plot(s_T_all_days_active.mean(axis=1), "-")

# # time lapse that defines a new day
# delta = 2000
# times= d["times"]
# plt.figure()
# plt.plot(times)
# new_day_ind = np.where(np.insert(np.diff(times)> delta, 0, True))[0] 
# plt.plot(new_day_ind, times[new_day_ind], ".r")

# day_inds = [(ind, new_day_ind[i+1]) for (i, ind) in enumerate(new_day_ind[:-1])]
# day_inds.append((new_day_ind[-1], times.shape[0]))

# s_T_per_day = [s_T_all_days[a:b, :] for (a,b) in day_inds]

# T_max = np.min([s.shape[0] for s in s_T_per_day])
