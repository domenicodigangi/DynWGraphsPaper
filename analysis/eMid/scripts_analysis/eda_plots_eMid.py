#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load data visualize networks

"""


# %% import packages
import importlib
import logging
logger = logging.getLogger(__name__)
from proj_utils import get_proj_fold
from matplotlib import pyplot as plt
import networkx as nx
from run_link_prediction_exercise_eMid_dev_tobit import load_obs
import numpy as np
from datetime import datetime
import pandas as pd
#%% 
Y_T, X_T, regr_list, net_stats = load_obs(ret_additional_stats=True)

t = 0
net_stats.dates[t].astype(object).year
graphs = [nx.convert_matrix.from_numpy_matrix(Y_T[:,:, t].numpy()) for t in range(Y_T.shape[2])]

#%%
df = pd.DataFrame({"date": net_stats.dates[2:], "graph": graphs })

df = df.resample("Y", on="date").first()
save_fold = get_proj_fold()/ "analysis" / "eMid" / "figures" / "emid_nets"
save_fold.mkdir(exist_ok=True)
for date, val in df["graph"].iteritems():
    fig = plt.figure()
    nx.draw_circular(val, node_size = 25)
    plt.title(f"eMid Network Beggining of {date.year}")
    save_path = save_fold / f"{date.year}.png"
    plt.savefig(save_path, bbox_inches='tight')

# %%
