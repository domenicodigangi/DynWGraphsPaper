"""
Estimate directed sparse weighted model on WTN data
"""

import sys
import numpy as np
import torch
sys.path.append("./src/")
from utils import splitVec, tens
SAVE_FOLD = './data/estimates_sim_data'
from dirBin1_dynNets import dirBin1_dynNet_SD
import matplotlib.pyplot as plt

#%%
avoid_ovflw_fun_flag = True
rescale_score = False
distribution = 'bernoulli'

torch.manual_seed(2)
model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False)
N = 30
T = 100
N_sample = 50
N_steps_max = 15000

N_BA = N
n_reg = 2
n_beta_tv = 0
dim_beta = N
type_dgp = 'SD'


file_path = SAVE_FOLD + '/filter_sd_dgp_dirBin1' + \
            '_N_' + str(N) + '_T_' + str(T) + \
            '_N_steps_' + str(N_steps_max) + '_N_BA_' + str(N_BA) + \
            '_resc_score_' + str(rescale_score) + '_avoid_ovflw_fun_flag_' + str(avoid_ovflw_fun_flag) + \
            '_distr_' + distribution + \
            '_N_sample_' + str(N_sample) + \
            '_type_dgp_' + type_dgp + \
            '.npz'



l_dat = np.load(file_path, allow_pickle=True)

w_dgp, B_dgp, A_dgp, w_sd_all, B_sd_all, A_sd_all,\
Y_T_dgp = tens(l_dat["arr_0"]), tens(l_dat["arr_1"]), \
tens(l_dat["arr_2"]), tens(l_dat["arr_3"]), tens(l_dat["arr_4"]), tens(l_dat["arr_5"]), tens(l_dat["arr_6"])


phi_T, beta_sd_T, Y_T_s = model.sd_dgp(w_dgp, B_dgp, A_dgp, p_T=None, beta_const=None, X_T=None, N=N, T=T)

Y_T_dgp.sum()
#%%

delta_w = w_sd_all - np.tile(w_dgp, (N_sample, 1)).transpose()


A_sd_all
A_dgp
B_dgp
plt.close()
plt.hist(delta_w)
i_plot = 11
#plt.plot(delta_w[i_plot, :])

#%%

#%%
s=1
i_plot = 1


phi_T_dgp, o = model.sd_filt(tens(w_dgp), tens(B_dgp), tens(A_dgp), tens(Y_T_dgp[:, :, :, s ]))

phi_T_est, o = model.sd_filt(tens(w_dgp), tens(B_dgp), tens(A_dgp), tens(Y_T_dgp[:, :, :, s ]))

plt.plot(phi_T_dgp.detach().numpy().transpose(), '.b', markersize=0.5)


#%%






