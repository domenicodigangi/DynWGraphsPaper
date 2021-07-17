"""
Simulate  dirBin1 models with time varying parameters following a SD dgp and filter with SD model. save data
"""

import sys
import numpy as np
import torch
sys.path.append("./src/")
SAVE_FOLD = './data/estimates_sim_data'
from utils import tens, strIO_from_mat

from dirBin2_dynNets import dirBin2_dynNet_SD

#%
avoid_ovflw_fun_flag = True
rescale_score = False
distribution = 'bernoulli'

torch.manual_seed(2)
#%
N = 8
model = dirBin2_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False, N=N)

T = 1000
N_sample  = 50
N_steps_max = 15000

type_dgp = 'SD'

#define storage variables
w_sd_all = torch.zeros(N_sample)
B_sd_all = torch.zeros(N_sample)
A_sd_all = torch.zeros(N_sample)
Y_T_dgp_all = torch.zeros(N, N, T, N_sample)

A_dgp = torch.ones(1) * 0.05
B_dgp = torch.ones(1) * 0.95
um_dgp = -2*torch.ones(1)
model.exp_A(um_dgp)
w_dgp = um_dgp *(1-B_dgp)

for s in range(N_sample):
    # Sample the SD Dgp
    phi_T, Y_T_s = model.sd_dgp(w_dgp, B_dgp, A_dgp, N, T)

    # plt.close()
    # plt.plot(Y_T_s.sum(dim=(0, 1)))

    #score driven estimates
    B_0 = 0.9 * torch.ones(1)
    A_0 =  0.01 * torch.ones(1)
    um_0 = um_dgp #*torch.ones(1)
    w_0 =  um_0 * (1-B_0)
    w_sd_s, B_sd_s, A_sd_s,  diag = model.estimate_SD(Y_T_s,
                                                       B0=B_0,
                                                       A0=A_0,
                                                       W0=w_0,
                                                       max_opt_iter=N_steps_max,
                                                       lr=0.005,
                                                       print_flag=True, plot_flag=False,
                                                        print_every=50)

    Y_T_dgp_all[:, :, :, s] = Y_T_s.clone()
    w_sd_all[s] = w_sd_s.clone().detach()
    B_sd_all[s] = B_sd_s.clone().detach()
    A_sd_all[s] = A_sd_s.clone().detach()
    print(B_sd_s)
    print(A_sd_s)
    print(s)


file_path = SAVE_FOLD + '/filter_sd_dgp_dirBin2' + \
            '_N_' + str(N) + '_T_' + str(T) + \
            '_N_steps_' + str(N_steps_max)  + \
            '_resc_score_' + str(rescale_score) + '_avoid_ovflw_fun_flag_' + str(avoid_ovflw_fun_flag) + \
            '_N_sample_' + str(N_sample) + \
            '_type_dgp_' + type_dgp + \
            '.npz'


print(file_path)
np.savez(file_path, w_dgp.detach(), B_dgp.detach(), A_dgp.detach(),
                    w_sd_all, B_sd_all, A_sd_all, Y_T_dgp_all)



#
#
#%%
# import matplotlib.pyplot as plt
# plt.close('all')
# plt.figure()
# plt.plot(diag)
# plt.figure()
# plt.plot(Y_T_s.sum(dim=(0, 1)))

#


