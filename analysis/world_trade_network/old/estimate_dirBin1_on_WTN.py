"""
Estimate directed sparse weighted model on WTN data
"""

import sys
import numpy as np
import torch
sys.path.append("./src/")
from utils import tens

#load yearly world Trade networks
ld_data = np.load("./data/world_trade_network/world_trade_net_T.npz",
                  allow_pickle=True)
wtn_T, all_y, nodes= ld_data["wtn_T"], ld_data["all_y"], ld_data["nodes"]
dist_T, scaling_infl = ld_data["dist_T"], ld_data["scaling_infl"]

Y_T = tens(wtn_T>0)
N = Y_T.shape[0]
T = Y_T.shape[2]
SAVE_FOLD = './data/estimates_real_data/WTN'

#%%  Weighted Estimates No regressors
from dirBin1_dynNets import dirBin1_dynNet_SD

N_steps = 15000
N_BA = N
learn_rate = 0.01

model = dirBin1_dynNet_SD(ovflw_lm=True, rescale_SD=False)
phi_T_0 = torch.zeros(N*2, T)
for t in range(T):
    phi_T_0[:, t] = model.start_phi_val(Y_T[:, :, t])


B0 = torch.cat([torch.ones(N_BA) * 0.85, torch.ones(N_BA) * 0.85])
A0 = torch.cat([torch.ones(N_BA) * 0.01, torch.ones(N_BA) * 0.01])
W0 = phi_T_0.mean(dim=1)*(1-B0)

W_est, B_est, A_est, dist_par_un_est,  diag = model.estimate_SD(Y_T, B0=B0, A0=A0, W0=W0, opt_steps=N_steps,
                                                                lRate=learn_rate,
                                                                print_flag=True, print_every=200, plot_flag=False,
                                                                 est_dis_par_un=False)

file_path = SAVE_FOLD + '/WTN_dirBin1_dynNet_SD_est_test_lr_' + \
            str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
            '_N_steps_' + str(N_steps) + '_N_BA_' + str(N_BA) + '.npz'
print(file_path)
np.savez(file_path, W_est.detach(), B_est.detach(), A_est.detach(), diag, N_steps, learn_rate)
