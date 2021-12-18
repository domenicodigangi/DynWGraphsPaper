"""
Load WTN data and estimate single snapshot versions of model for sprse Weighted networks
@author: domenico
"""
import numpy as np
import torch
import sys
sys.path.append("./src/")
from utils import tens
#load yearly world Trade networks
ld_data = np.load("./data/world_trade_network/world_trade_net_T.npz",
                  allow_pickle=True)
wtn_T, all_y, nodes= ld_data["wtn_T"], ld_data["all_y"], ld_data["nodes"]
dist_T, scaling_infl = ld_data["dist_T"], ld_data["scaling_infl"]

X_T = tens(dist_T).log().unsqueeze(2) # torch.tensor((dist_T - dist_T.mean())/dist_T.std()).unsqueeze(2)
N = wtn_T.shape[0]
T = wtn_T.shape[2]
learn_rate_phi = 0.01
Y_T = tens(wtn_T > 0)

# %% estimate and save No regressors
from dirBin1_dynNets import  dirBin1_dynNet_SD
SAVE_FOLD = './data/estimates_real_data/WTN'

model = dirBin1_dynNet_SD()
N_steps = 15000
phi_T_0 = torch.zeros(N*2, T)
for t in range(T):
    phi_T_0[:, t] = model.start_phi_val(Y_T[:, :, t])

phi_ss_est_T, diag = model.ss_filt( Y_T,  phi_T_0=phi_T_0,
                          est_dist_par=False, est_beta=False,
                          max_opt_iter=N_steps, lr=0.01, print_flag=True, plot_flag=False, print_every=200,
                          min_opt_iter=200)

file_path = SAVE_FOLD + '/WTN_dirBin1_dynNet_Single_Snap_est__lr_th' + \
            str(learn_rate)  + '_N_' + str(N) + '_T_' + str(T) + \
            '_N_steps_' + str(N_steps) + \
            '.npz'
print(file_path)
np.savez(file_path, phi_ss_est_T.detach(), diag)


#% estimate and save with regressors

model = dirBin1_dynNet_SD()
N_steps = 15000
N_steps_iter = 200
learn_rate_phi = 0.01
learn_rate_beta = 0.005


N = Y_T.shape[0]
T = Y_T.shape[2]



for dim_beta in[1, N]:

    # Set unit measure and  rescale for inflation
    Y_T = torch.tensor(tens(wtn_T > 0))
    # estimate single snapshot
    phi_ss_est_T, dist_par_un, beta_est, diag_joint = model.ss_filt_est_beta_const(Y_T,
                                       X_T=X_T, beta=torch.zeros(dim_beta,1), phi_T=phi_T_0,
                                      est_const_beta=True, dim_beta=dim_beta,
                                      opt_large_steps=N_steps//N_steps_iter, max_opt_iter_phi=N_steps_iter,
                                       lr_phi=learn_rate_phi,
                                      max_opt_iter_beta=N_steps_iter, lr_beta=learn_rate_beta,
                                      print_flag_phi=False, print_flag_beta=True,
                                      print_every=100, min_opt_iter=20)

    file_path = SAVE_FOLD + '/WTN_dirBin1_X0_dynNet_Single_Snap_est__lr_th' + \
                str(learn_rate_phi)  + '_lt_de_' + str(learn_rate_beta) + '_N_' + str(N) + '_T_' + str(T) + \
                '_N_steps_' + str(N_steps) + \
                '_dim_beta_' + str(dim_beta) + '.npz'
    print(file_path)
    np.savez(file_path, phi_ss_est_T.detach(), beta_est.detach(), diag_joint,
             N_steps, learn_rate_phi, learn_rate_beta)


