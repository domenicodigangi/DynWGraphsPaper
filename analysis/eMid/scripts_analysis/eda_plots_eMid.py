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
graphs = [nx.convert_matrix.from_numpy_matrix(Y_T[:,:, t].numpy()) for t in range(Y_T.shape[2])]


#%%
import seaborn as sns
import scipy
x = Y_T.numpy().flatten() * 10000
x_nnz = x[x>0]

T = Y_T.shape[2]
s_T_in = Y_T.numpy().sum(axis=1) * 10000
s_T_out = Y_T.numpy().sum(axis=0) * 10000
nbanks_T = ((s_T_in>0) | (s_T_out>0)).sum(axis=0)
nlinks_T = ( Y_T.numpy()>0).sum(axis=(0,1))

avg_w_T = (s_T_in).sum(axis=0)/nlinks_T
dens_T = nlinks_T/(nbanks_T*(nbanks_T-1)) 
np.corrcoef(dens_T, avg_w_T)
scipy.stats.spearmanr(dens_T, avg_w_T)

ax = sns.lineplot(y=avg_w_T, x = net_stats.dates[2:])
ax.set_ylabel("Avg. Weight Present Links")
ax.grid()
plt.figure()
ax2 = sns.lineplot(y=dens_T, x = net_stats.dates[2:])
ax2.set_ylabel("Network Density")
ax2.grid()

#%%

s_tot_T = Y_T.numpy().mean(axis=(0,1))


g = sns.histplot(s_T_out[s_T_out>0].flatten()/T, stat="probability", bins=25, log_scale=True)
g.set_xlabel("Avg. Banks Outstanding Debt [Euro]")
plt.figure()
g = sns.histplot(s_T_in[s_T_in>0].flatten()/T , stat="probability", bins=25, log_scale=True)
g.set_xlabel("Avg. Banks Outstanding Credits [Euro]")

x_nnz.max()/1e6
x_nnz.min()/1e6
x_nnz.mean()/1e6

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
