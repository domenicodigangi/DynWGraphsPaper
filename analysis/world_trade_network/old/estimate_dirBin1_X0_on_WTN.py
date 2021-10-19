"""
Estimate directed sparse weighted model on WTN data
"""

import sys
import numpy as np
import torch

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
X_T = tens(dist_T).log().unsqueeze(2) # torch.tensor((dist_T - dist_T.mean())/dist_T.std()).unsqueeze(2)
N = Y_T.shape[0]
T = Y_T.shape[2]
SAVE_FOLD = './data/estimates_real_data/WTN'

# %%  Weighted Estimates No regressors
from dirBin1_dynNets import dirBin1_dynNet_SD

N_steps = 15000
N_BA = N
learn_rate = 0.01
n_reg=X_T.shape[2]
n_beta_tv = 0

model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False)
phi_T_0 = torch.zeros(N*2, T)
for t in range(T):
    phi_T_0[:, t] = model.start_phi_val(Y_T[:, :, t])

B0 = torch.cat([torch.ones(N_BA) * 0.85, torch.ones(N_BA) * 0.85])
A0 = torch.cat([torch.ones(N_BA) * 0.01, torch.ones(N_BA) * 0.01])
W0 = phi_T_0.mean(dim=1)*(1-B0)

for dim_beta in [1, N]:
    W_est, B_est, A_est, dist_par_un_est, beta_const_est,  diag = model.estimate_SD_X0(Y_T, X_T=X_T,
                                                                    dim_beta=dim_beta,
                                                                    n_beta_tv = n_beta_tv,
                                                                    B0=torch.cat((B0, torch.zeros(n_beta_tv))),
                                                                    A0=torch.cat((A0, torch.zeros(n_beta_tv))),
                                                                    W0=torch.cat((W0, torch.zeros(n_beta_tv))),
                                                                    beta_const_0= torch.zeros(dim_beta, n_reg-n_beta_tv),
                                                                    max_opt_iter=N_steps,
                                                                    lr=learn_rate,
                                                                    print_flag=True, print_every=200,
                                                                    plot_flag=False)


    file_path = SAVE_FOLD + '/WTN_dirBin1_X0_dynNet_SD_est_test_lr_' + \
                str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                '_N_steps_' + str(N_steps) + '_N_BA_' + str(N_BA) + \
                '_dim_beta_' + str(dim_beta) + '.npz'

    print(file_path)

    np.savez(file_path, W_est.detach(), B_est.detach(), A_est.detach(), beta_const_est.detach(),
             diag, N_steps, learn_rate)







