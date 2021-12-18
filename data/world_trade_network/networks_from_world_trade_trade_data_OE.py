# -*- coding: utf-8 -*-

"""
Created on Tue Oct 22 14:46:10 2019

@author: digangi
"""



# %%
import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
fpath = "../../data/world trade web/year_origin_destination_sitc_rev2.tsv"
cty_df = pd.read_csv("../../data/world trade web/cty_map.csv", encoding='latin-1')


def pop_row_cols_mat_seq(ind, Z_T):
    tmp = Z_T[~ind, :, :]
    return tmp[:, ~ind, :]
# %% load data, country names and our country maps. Map 3 digits names into 2 digits
reader = pd.read_csv(fpath, sep='\t', low_memory= True, skiprows=[0, 1, 2], usecols=[0,1,2,4],
                 dtype={'year': np.int16, 'origin': str, 'dest': str,
                        'import_val': np.int16}, iterator=True, chunksize=10000000)

edges = pd.DataFrame()
for df in reader:
    print(df.shape)
    # group by importer exporter  pairs and year (the data are divided by product type)
    edges_it = df.groupby(by=["origin", "dest", "year"], as_index=False).agg({'export_val': np.sum })
    edges = edges.append(edges_it)


# %%Convert to Weighted adjacency matrices
# first assign an integer to each node and create edge list using those
nodes_init, inds = np.unique(np.hstack((edges['origin'], edges['dest'])), return_inverse=True)
orig, dest = np.split(inds, 2)
N = nodes_init.shape[0]

# for each year convert the list of edges of that year to an adjacency matrix
all_y = np.sort(edges["year"].unique())
T = all_y.shape[0]
wtn_T_init = np.zeros((N, N, T), dtype=np.float32)
for t, y in enumerate(all_y):
    inds_y = edges["year"] == y
    wtn_T_init[:, :, t] = sp.coo_matrix((np.array(edges["export_val"].loc[inds_y]), (orig[inds_y], dest[inds_y])),
                                   shape=(N, N)).todense()


# %% import  geographical distances and colony contingency relations
grav_data_init = pd.read_stata("../../data/world trade web/gravdata_cepii/gravdata.dta",
                          columns=["iso3_o", "iso3_d", "year", "contig", "distw", "colony"])

grav_data_init = grav_data_init.dropna()
grav_data_init["iso3_o"] = grav_data_init["iso3_o"].str.lower()
grav_data_init["iso3_d"] = grav_data_init["iso3_d"].str.lower()
# fix romania adn timor est
grav_data_init["iso3_o"].loc[grav_data_init["iso3_o"] == 'rom'] = 'rou'
grav_data_init["iso3_d"].loc[grav_data_init["iso3_d"] == 'rom'] = 'rou'
grav_data_init["iso3_o"].loc[grav_data_init["iso3_o"] == 'tmp'] = 'tls'
grav_data_init["iso3_d"].loc[grav_data_init["iso3_d"] == 'tmp'] = 'tls'


#list countries present in gravity data
nodes_grav_init = np.unique(np.hstack((grav_data_init['iso3_o'], grav_data_init['iso3_d'])))

# %% compare nodes in grav data with nodes in net data and remove those not present in net data
df1 = pd.DataFrame(data={'name': nodes_init, 'ind': range(nodes_init.shape[0]) })
df2 = pd.DataFrame(data={'name': nodes_grav_init, 'ind': range(nodes_grav_init.shape[0]) })
#list of countries that are present in both datasets
df = df1.merge(df2, how='inner', left_on="name", right_on="name") # only those present in both df1 and df2
# print removed countries
df1[~df1.ind.isin(df["ind_x"])]["name"]
cty_df[cty_df["Alpha-3"].str.lower().isin(df1[~df1.ind.isin(df["ind_x"])]["name"])]
df2.loc[~df2.ind.isin(df["ind_y"])]
cty_df[cty_df["Alpha-3"].str.lower().isin(df2[~df2.ind.isin(df["ind_y"])]["name"])]

# Remove from wtn nodes without geographical info
ind_rem_wtn = ~(df1.ind.isin(df["ind_x"])).values
wtn_T = pop_row_cols_mat_seq(ind_rem_wtn, wtn_T_init)
nodes = nodes_init[~ind_rem_wtn]
# remove contries from geographical info that are not in wtn
list_rem_grav = df2.loc[~df2.ind.isin(df["ind_y"])]["name"].values
grav_data = grav_data_init.loc[~(grav_data_init["iso3_o"].isin(list_rem_grav) |
                                 grav_data_init["iso3_d"].isin(list_rem_grav)), :]

# convert types
grav_data["iso3_o"] = grav_data["iso3_o"].astype("category")
grav_data["iso3_d"] = grav_data["iso3_d"].astype("category")
grav_data["colony"] = grav_data["colony"].astype("bool")
grav_data["contig"] = grav_data["contig"].astype("bool")
grav_data.dtypes

#list countries present and associate integer inddices to origin and destination of each link
nodes_grav, inds = np.unique(np.hstack((grav_data['iso3_o'], grav_data['iso3_d'])), return_inverse=True)
if np.prod(nodes == nodes_grav) != 1: # check that nodes lists are equal
    error
orig, dest = np.split(inds, 2)
N = nodes_grav.shape[0]
#for each year IN THE TRADE DATA convert the list of edges of that year to an adjacency matrix
all_y_grav = np.sort(grav_data["year"].unique())
T = all_y.shape[0]
dist_T = np.ones((N, N, T), dtype=np.float32)*(-1)
contig_T = np.zeros((N, N, T), dtype=np.bool)
colony_T = np.zeros((N, N, T), dtype=np.bool)
for t, y in enumerate(all_y):
    inds_y = grav_data["year"] == y
    print((t, y, np.sum(inds_y)))
    dist_T[:, :, t] = sp.coo_matrix((np.array(grav_data["distw"].loc[inds_y]), (orig[inds_y], dest[inds_y])),
                                   shape=(N, N)).todense()
    contig_T[:, :, t] = sp.coo_matrix((np.array(grav_data["contig"].loc[inds_y]), (orig[inds_y], dest[inds_y])),
                                   shape=(N, N)).todense()
    colony_T[:, :, t] = sp.coo_matrix((np.array(grav_data["colony"].loc[inds_y]), (orig[inds_y], dest[inds_y])),
                                   shape=(N, N)).todense()


# check that all countries have pairwise distances at all times
missing_dist = np.array([np.sum(dist_T[:, :, t] == 0) for t in range(T)])
missing_dist
dist_T[:, :, -2] = dist_T[:, :, -3]
dist_T[:, :, -1] = dist_T[:, :, -3]
contig = contig_T[:, :, -3]
colony = colony_T[:, :, -3]



# %% load US consumer index and prepare coefficients to adjust for inflation
#load consumer price index to account for inflation
inf_df = pd.read_csv("../../data/world trade web/CPIAUCNS.csv")
inf_df["year"] = inf_df["DATE"].apply(lambda x: np.int(x[:4]))
inf_df["month"] = inf_df["DATE"].apply(lambda x: np.int(x[5:7]))
#use index of the last month of the year
infl_scaling = (inf_df.loc[(inf_df["year"] >= all_y[0]) & (inf_df["year"] <= all_y[-1]) & (inf_df["month"] == 12), :])\
    ["CPIAUCNS"].values

scaling_infl = pd.DataFrame(data ={'year': all_y, 'scaling': (infl_scaling[-1]/infl_scaling)})

# %% select and remove very small import exporters
tot_in = (wtn_T * infl_scaling).sum(axis=(0, 2))
tot_out = (wtn_T * infl_scaling).sum(axis=(1, 2))
imp_plus_exp_thr = 50*1e9
inds_small = ((tot_in + tot_out) < imp_plus_exp_thr)
# who are we going to remove?
cty_df[cty_df["Alpha-3"].apply(lambda x:str(x).lower()).isin(nodes[inds_small])]

# Remove from matrices and names
wtn_large_T = pop_row_cols_mat_seq(inds_small, wtn_T)
nodes_large = nodes[~inds_small]

colony_large = colony[~inds_small]
contig_large = contig[~inds_small]
dist_large_T = pop_row_cols_mat_seq(inds_small,dist_T)


# %%

save_path = "./data/world_trade_network/world_trade_net_T"
np.savez(save_path, wtn_T=wtn_large_T, all_y=all_y, nodes=nodes_large, dist_T=dist_large_T, contig=contig_large,
         colony=colony_large, scaling_infl=scaling_infl)

#









