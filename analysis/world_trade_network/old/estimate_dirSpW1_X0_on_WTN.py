"""
Estimate directed sparse weighted model with distances as regressors on WTN data
"""

import sys
import numpy as np
import torch
sys.path.append("./src/")
from utils import splitVec, tens
from joblib import Parallel, delayed
import multiprocessing

#load yearly world Trade networks
ld_data = np.load("./data/world_trade_network/world_trade_net_T.npz",
                  allow_pickle=True)
wtn_T, all_y, nodes= ld_data["wtn_T"], ld_data["all_y"], ld_data["nodes"]
dist_T, scaling_infl = ld_data["dist_T"], ld_data["scaling_infl"]
N = wtn_T.shape[0]
SAVE_FOLD = './data/estimates_real_data/WTN'

#%  Weighted Estimates With regressors
import itertools
from dirSpW1_dynNets import dirSpW1_dynNet_SD
ovflw_lm = True
rescale_score = False

# Set unit measure and  rescale for inflation


def estimate_and_save_dirSpW1_X0(N_BA, learn_rate, distribution, dim_dist_par_un, dim_beta, T_test=10, unit_measure=1e6,
                                 N_steps_max=11000):
    # Set unit measure and  rescale for inflation
    Y_T = tens(wtn_T * scaling_infl[:, 1]) / unit_measure
    Y_T_train = Y_T[:, :, :-T_test]
    X_T = tens(dist_T).log().unsqueeze(2) # torch.tensor((dist_T - dist_T.mean())/dist_T.std()).unsqueeze(2)

    N = Y_T.shape[0]
    T = Y_T.shape[2]
    # %load single snapshot estimates to be used as starting point
    N_steps_load = 10000
    learn_rate_load = 0.01

    file_path = SAVE_FOLD + '/WTN_dirSpW1_X0_dynNet_Single_Snap_est__lr_' + \
                str(learn_rate_load) + '_N_' + str(N) + '_T_' + str(T) + \
                '_N_steps_' + str(N_steps_load) + \
                '_ovflw_lm_' + str(ovflw_lm) + \
                '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                '_dim_beta_' + str(dim_beta) + \
                distribution + 'distr_' + 'dim_distr_par_' + str(dim_dist_par_un) + '.npz'

    ld_est = np.load(file_path, allow_pickle=True)
    phi_ss_est_T, dist_par_un_ss, beta_ss = tens(ld_est["arr_0"]), tens(ld_est["arr_1"]), tens(ld_est["arr_2"])

    model = dirSpW1_dynNet_SD(ovflw_lm=ovflw_lm, rescale_SD=rescale_score, distribution=distribution)
    for t in range(T):
        phi_ss_est_T[:, t] = model.identify(phi_ss_est_T[:, t])
    mean_ss_phi = phi_ss_est_T[:, 0] #phi_ss_est_T.mean(dim=1)


    # define initial points of optimizer: sets also the number of A,B parameters
    B0 = torch.cat([torch.ones(N_BA) * 0.85, torch.ones(N_BA) * 0.85])
    A0 = torch.cat([torch.ones(N_BA) * 0.01, torch.ones(N_BA) * 0.01])
    UM0_i, UM0_o = splitVec(mean_ss_phi)
    W0 = torch.cat((UM0_i * (1-splitVec(B0)[0]), UM0_o *(1- splitVec(B0)[1])))
    beta0 = beta_ss
    dist_par_un_0 = dist_par_un_ss

    W_est, B_est, A_est, dist_par_un_est, beta_est, diag = model.estimate_SD_X0(Y_T_train, X_T, B0=B0, A0=A0, W0=W0,
                                                                                max_opt_iter=N_steps_max,
                                                                                lr=learn_rate,
                                                                                est_dis_par_un=True,
                                                                                dist_par_un=dist_par_un_0,
                                                                                beta_const_0=beta0,
                                                                                dim_beta=dim_beta,
                                                                                dim_dist_par_un=dim_dist_par_un,
                                                                                print_flag=True, plot_flag=False,
                                                                                print_every=200)


    file_path = SAVE_FOLD + '/WTN_dirSpW1_X0_dynNet_SD_est_lr_' + \
                str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                '_N_steps_' + str(N_steps_max) + '_N_BA_' + str(N_BA) + \
                '_resc_score_' + str(rescale_score) + '_ovflw_lm_' + str(ovflw_lm) + \
                '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                distribution + '_distr_' + '_dim_distr_par_' + str(dim_dist_par_un) + \
                '_dim_beta_' + str(dim_beta) + \
                'test_sam_last_' + str(T_test) + '.npz'
    print(file_path)
    np.savez(file_path, W_est.detach(), B_est.detach(), A_est.detach(), dist_par_un_est.detach(), beta_est.detach(),
             diag, N_steps_max, unit_measure, learn_rate)


#%%
results = Parallel(n_jobs=2)(delayed(estimate_and_save_dirSpW1_X0)(N_BA, lr, distribution, dim_dist_par_un, dim_beta,
                                                                   T_test, unit_measure, N_steps_max) \
                     for N_BA, lr, distribution, dim_dist_par_un, dim_beta, T_test, unit_measure, N_steps_max \
                     in itertools.product([N], [0.01], ['lognormal', 'gamma'], [N], [N, 1], [10], [1e6], [15000]))

#%%
estimate_and_save_dirSpW1_X0(N,0.1,'lognormal', 1,1,30,1e6, 30)