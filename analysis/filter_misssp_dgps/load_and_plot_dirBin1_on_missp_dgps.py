"""
Estimate directed sparse weighted model on WTN data
"""


#%%
import os
import numpy as np
from numpy.core.fromnumeric import size
import torch
from pathlib import Path
import pickle
import copy
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import dynwgraphs
from dynwgraphs.utils.tensortools import tens, splitVec
from dynwgraphs.dirGraphs1_dynNets import  dirBin1_sequence_ss, dirBin1_SD, dirSpW1_SD
import importlib
import torch
from torch.functional import split
importlib.reload(dynwgraphs)


#%%



degb = tens([5, N-5])
type_dgp = 'sin'
n_reg = 1
n_reg_beta_tv = 0
dim_beta = 1
dim_dist_par_un = 1


N_sample = 50
N_steps_max = 15000
opt_large_steps = 50

N_BA = N


file_path = SAVE_FOLD + '/filter_missp_dgp_dirBin1'+ \
            '_N_' + str(N) + '_T_' + str(T) + \
            '_N_steps_' + str(N_steps_max) + '_N_BA_' + str(N_BA) + \
            '_resc_score_' + str(rescale_score) + '_avoid_ovflw_fun_flag_' + str(avoid_ovflw_fun_flag) + \
            '_distr_' + distribution + \
            '_dim_beta_' + str(dim_beta) + \
            '_N_sample_' + str(N_sample) + \
            '_type_dgp_' + type_dgp + \
            '.npz'

l_dat = np.load(file_path, allow_pickle=True)

phi_T_dgp, beta_T_dgp, Y_T_dgp, X_T_dgp, w_sd_all, B_sd_all, A_sd_all, dist_par_un_sd_all,\
beta_sd_all, phi_T_sd_all, beta_ss_all, phi_T_ss_all = l_dat["arr_0"], l_dat["arr_1"], \
l_dat["arr_2"], l_dat["arr_3"], l_dat["arr_4"], l_dat["arr_5"], l_dat["arr_6"], l_dat["arr_7"], l_dat["arr_8"], \
l_dat["arr_9"], l_dat["arr_10"], l_dat["arr_11"]


#%%
plt.close()
i_plot = 1
plt.plot(phi_T_ss_all[i_plot, :, :], '.b', markersize=0.5)
plt.plot(phi_T_sd_all[i_plot, :, :], '-r', linewidth=0.5)
plt.plot(phi_T_dgp[i_plot, :], '-k', linewidth=5)
plt.legend(['Single Snapshot', 'Score Driven', 'DGP'])

#%%






