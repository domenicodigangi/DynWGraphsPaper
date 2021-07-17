"""
Estimate directed sparse weighted model on WTN data
"""

import sys
import numpy as np
import torch
sys.path.append("./src/")
SAVE_FOLD = './data/estimates_sim_data'

from dirSpW1_dynNets import dirSpW1_dynNet_SD

#%
avoid_ovflw_fun_flag = True
rescale_score = False



N = 30

type_dgp = 'SD'
n_reg = 2
n_reg_beta_tv = 0
dim_beta = 1
dim_dist_par_un = 1

N_sample = 5
N_steps_max = 15000
N_BA = N

for distribution in ['lognormal']:
    for T in [500]:
        model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=avoid_ovflw_fun_flag, rescale_SD=rescale_score, distribution=distribution)
        #const_p = 0.1
        for const_p in [0.9,   0.25,  1/N]:
            # Sample the Dgp
            Y_T_dgp_start, phi_T_dgp_Start, X_T_dgp_Start, beta_T_dgp_Start, dist_par_un_dgp_start = \
                model.sample_from_dgps(N, 5, 1, p_T=torch.ones(N, N, T)*const_p,
                                            dgp_type='sin',
                                            n_reg=n_reg, n_reg_beta_tv=n_reg_beta_tv,
                                            dim_beta=dim_beta, dim_dist_par_un=dim_dist_par_un,
                                            distribution=distribution)

            # define storage variables
            sd_par_0_all = torch.zeros(2 * N, N_sample)
            w_sd_all = torch.zeros(2 * N, N_sample)
            dist_par_un_est_all = torch.zeros(2 * N, N_sample)
            B_sd_all = torch.zeros(2 * N_BA, N_sample)
            A_sd_all = torch.zeros(2 * N_BA, N_sample)
            Y_T_dgp_all = torch.zeros(N, N, T, N_sample)
            diag_all = []

            A_dgp = torch.cat([torch.ones(N_BA) * 0.05, torch.ones(N_BA) * 0.05])
            B_dgp = torch.cat([torch.ones(N_BA) * 0.98, torch.ones(N_BA) * 0.98])


            um_sd_par = phi_T_dgp_Start[:, -1]
            w_dgp = um_sd_par * (1 - B_dgp)

            for s in range(N_sample):
                # Sample the SD Dgp
                phi_T, beta_sd_T, Y_T_s = model.sd_dgp(w_dgp, B_dgp, A_dgp, p_T=None, beta_const=None, X_T=None, N=N, T=T)

                # score driven estimates
                B_0 = B_dgp #* 0.9
                A_0 = A_dgp #* 0.9
                w_0 = w_dgp #* 0.9
                # print(Y_T_s.sum())
                w_sd_s, B_sd_s, A_sd_s, dist_par_un_est,  sd_par_0, diag = model.estimate_SD(Y_T_s,
                                                                                  B0=B_0,
                                                                                  A0=A_0,
                                                                                  W0=w_0,
                                                                                  est_dis_par_un=False,
                                                                                  max_opt_iter=N_steps_max,
                                                                                  lr=0.01,
                                                                                  print_flag=True, plot_flag=False,
                                                                                  print_every=10000)

                diag_all.append(diag)
                Y_T_dgp_all[:, :, :, s] = Y_T_s.clone()
                dist_par_un_est_all[:, s] = dist_par_un_est.clone().detach()
                w_sd_all[:, s] = w_sd_s.clone().detach()
                B_sd_all[:, s] = B_sd_s.clone().detach()
                A_sd_all[:, s] = A_sd_s.clone().detach()
                sd_par_0_all[:, s] = sd_par_0.clone().detach()
                # print(B_sd_s)
                # print(A_sd_s)
                print((s, const_p, distribution, T, rescale_score, type_dgp))

            file_path = SAVE_FOLD + '/filter_sd_dgp_diSpW1'+ \
                        '_N_' + str(N) + '_T_' + str(T) + \
                        '_const_p_' + str(const_p) + \
                        '_N_steps_' + str(N_steps_max) + '_N_BA_' + str(N_BA) + \
                        '_resc_score_' + str(rescale_score) + '_avoid_ovflw_fun_flag_' + str(avoid_ovflw_fun_flag) + \
                        '_distr_' + distribution + '_dim_distr_par_' + str(dim_dist_par_un) + \
                        '_dim_beta_' + str(dim_beta) + \
                        '_N_sample_' + str(N_sample) + \
                        '_type_dgp_' + type_dgp + \
                        '.npz'

            print(file_path)
            np.savez(file_path, w_dgp.detach(), B_dgp.detach(), A_dgp.detach(),
                     w_sd_all, B_sd_all, A_sd_all, dist_par_un_est_all, sd_par_0_all, Y_T_dgp_all)









