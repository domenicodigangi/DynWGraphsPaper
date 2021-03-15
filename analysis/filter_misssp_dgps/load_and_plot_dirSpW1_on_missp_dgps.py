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
distribution = 'gamma'
model = dirSpW1_dynNet_SD(ovflw_lm=ovflw_lm, rescale_SD=rescale_score, distribution=distribution)
N = 30
T = 100
type_dgp = 'ar1'
n_reg = 2
n_reg_beta_tv = 0
dim_beta = 1
dim_dist_par_un = 1
N_sample = 50
N_steps_max = 15000
opt_large_steps = 50
N_BA = N
est_dist_par=True
mse_sd = []
mse_ss = []
list_const_p = [0.9]#[0.01, 0.1, 0.25, 0.5, 0.75, 0.99]
for const_p in list_const_p:

    file_path = SAVE_FOLD + '/filter_missp_dgp_diSpW1' + \
                '_N_' + str(N) + '_T_' + str(T) + \
                '_const_p_' + str(const_p) + \
                '_N_steps_' + str(N_steps_max) + '_N_BA_' + str(N_BA) + \
                '_resc_score_' + str(rescale_score) + '_ovflw_lm_' + str(ovflw_lm) + \
                '_distr_' + distribution + '_dim_distr_par_' + str(dim_dist_par_un) + \
                '_dim_beta_' + str(dim_beta) + \
                '_N_sample_' + str(N_sample) + \
                '_type_dgp_' + type_dgp + \
                '.npz'


    l_dat = np.load(file_path, allow_pickle=True)


    phi_T_dgp, dist_par_un_dgp, beta_T_dgp, Y_T_dgp, X_T_dgp, w_sd_all, B_sd_all, A_sd_all, dist_par_un_sd_all,\
    beta_sd_all, sd_dist_par_0_all, phi_T_sd_all, dist_par_un_ss_all, beta_ss_all, phi_T_ss_all = l_dat["arr_0"], l_dat["arr_1"], \
    l_dat["arr_2"], l_dat["arr_3"], l_dat["arr_4"], l_dat["arr_5"], l_dat["arr_6"], l_dat["arr_7"], l_dat["arr_8"], \
    l_dat["arr_9"], l_dat["arr_10"], l_dat["arr_11"], l_dat["arr_12"], l_dat["arr_13"], l_dat["arr_14"]

    #% compute MSE
    delta_SD = np.zeros(0)
    delta_SS = np.zeros(0)
    for n in range(N_sample):
        delta_SD = np.append(delta_SD, np.sqrt(((phi_T_dgp[:, :-1]-phi_T_sd_all[:, 1:, n])**2).mean(axis=1)).reshape(-1))
        delta_SS = np.append(delta_SS, np.sqrt(((phi_T_dgp[:, :-1]-phi_T_ss_all[:, :-1, n])**2).mean(axis=1)).reshape(-1))
    mse_sd.append(delta_SD.mean())
    mse_ss.append(delta_SS.mean())

print(mse_sd, mse_ss)
#plt.hist(model.cond_exp_Y(tens(phi_T_dgp[:, 80])).view(-1).log10(), alpha=0.2)
#%%
#plt.close()

i_plot = 15


fig, ax = plt.subplots(2, 1)
plt.suptitle( distribution + ' distribution and constant p = ' + str(const_p))
ax[0].plot(phi_T_ss_all[i_plot, :, :], '.b', markersize=0.5)
ax[0].plot(phi_T_sd_all[i_plot, :, :], '-r', linewidth=0.5)
ax[0].plot(phi_T_dgp[i_plot, :], '-k', linewidth=5)
ax[0].legend(['Single Snapshot', 'Score Driven', 'DGP'])
ax[1].hist(beta_ss_all[0, :, :].reshape(-1), alpha=0.2)
ax[1].hist(beta_sd_all[0, :, :].reshape(-1), alpha=0.2)
ax[1].axvline(1)
ax[1].legend(['beta dgp', 'beta SS', 'beta SD'])

#%%
mse_sd
mse_ss
#%%
plt.close()
s = 0
A_s = A_sd_all[:, s]
phi_T_sd_s = phi_T_sd_all[:, :, s]
plt.plot(A_s, phi_T_sd_s.std(axis=1), '.')
#%
plt.close()
inds = phi_T_sd_s.std(axis=1)<5
plt.plot(phi_T_sd_s[inds, :].transpose())
A_s[inds]


#%%





#%%



