"""
Estimate directed sparse weighted model on WTN data
"""

import sys
import numpy as np
import torch
sys.path.append("./src/")
from utils import splitVec, tens
SAVE_FOLD = './data/estimates_sim_data'
from dirSpW1_dynNets import dirSpW1_dynNet_SD
import matplotlib.pyplot as plt

#%%
ovflw_lm = True
rescale_score = True
distribution = 'lognormal'
model = dirSpW1_dynNet_SD(ovflw_lm=ovflw_lm, rescale_SD=rescale_score, distribution=distribution)
N = 30
T = 100
type_dgp = 'SD'
n_reg = 2
n_reg_beta_tv = 0
dim_beta = 1
dim_dist_par_un = 1
N_sample = 50
N_steps_max = 15000
opt_large_steps = 50

N_BA = N
est_dist_par=True
const_p = 0.25#, 0.1, 0.01

file_path = SAVE_FOLD + '/filter_sd_dgp_diSpW1' + \
            '_N_' + str(N) + '_T_' + str(T) + \
            '_const_p_' + str(const_p) + \
            '_N_steps_' + str(N_steps_max) + '_N_BA_' + str(N_BA) + \
            '_resc_score_' + str(rescale_score) + '_ovflw_lm_' + str(ovflw_lm) + \
            '_distr_' + distribution + '_dim_distr_par_' + str(dim_dist_par_un) + \
            '_dim_beta_' + str(dim_beta) + \
            '_N_sample_' + str(N_sample) + \
            '_type_dgp_' + type_dgp + \
            '.npz'

print(file_path)

l_dat = np.load(file_path, allow_pickle=True)


w_dgp, B_dgp, A_dgp,\
w_sd_all, B_sd_all, A_sd_all, dist_par_un_est_all, sd_par_0_all, Y_T_dgp_all,\
Y_T_dgp_all = l_dat["arr_0"], l_dat["arr_1"], l_dat["arr_2"], l_dat["arr_3"], l_dat["arr_4"],\
              l_dat["arr_5"], l_dat["arr_6"], l_dat["arr_7"], l_dat["arr_8"],



model.cond_exp_Y(tens(w_dgp/(1-B_dgp))).sum()

tens(Y_T_dgp_all[:, :, 0, 0]).sum()

#%%
plt.close()
plt.figure(figsize=(10, 7))
plt.subplot(3, 1, 1)
plt.hist( (np.tile(w_dgp, (N_sample, 1)).transpose() - w_sd_all).reshape(-1))
plt.axvline(0, color='r')
plt.legend( ['W DGP', 'W EST'])
plt.subplot(3, 1, 2)
plt.hist(B_sd_all.reshape(-1))
plt.axvline(B_dgp[0], color='r')
plt.legend( ['B DGP', 'B EST'])
plt.subplot(3, 1, 3)
plt.hist(A_sd_all.reshape(-1))
plt.axvline(A_dgp[0], color='r')
plt.legend( ['A DGP', 'A EST'])
plt.suptitle('SD estimates of SD DGP, T = ' + str(T) +', N = ' + str(N) +  ', Distribution = ' + distribution + ', Rescaling = ' + str(rescale_score) +  '  P-link = ' + str(const_p))

#%%






