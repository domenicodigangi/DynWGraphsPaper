"""
Simulate dirBin1 models with time varying parameters and filter with SD model. save data
"""

import sys
import numpy as np
import torch
sys.path.append("./src/")
SAVE_FOLD = './data/estimates_sim_data'
from utils import tens
from dirBin1_dynNets import dirBin1_dynNet_SD

ovflw_lm = True
rescale_score = False
distribution = 'bernoulli'

model = dirBin1_dynNet_SD(ovflw_lm=ovflw_lm, rescale_SD=rescale_score, distribution=distribution)

N = 30
T = 100
degb = tens([5, N-5])
n_reg = 1
n_reg_beta_tv = 0
dim_beta = 1
dim_dist_par_un = 1
N_sample = 50
N_steps_max = 15000
opt_large_steps = 50
N_BA = N

for type_dgp in ['sin' ]:#'sin', 'step' # AR1 needs to be fixed
    # Sample the Dgp
    Y_T_dgp, phi_T_dgp, X_T_dgp, beta_T_dgp = \
        model.sample_from_dgps(N, T, N_sample,
                                    dgp_type=type_dgp, degb=degb,
                                    n_reg=n_reg, n_reg_beta_tv=n_reg_beta_tv,
                                    dim_beta=dim_beta)

    #define storage variables
    beta_ss_all = torch.zeros(dim_beta, n_reg, N_sample)
    phi_T_ss_all = torch.zeros(2*N, T, N_sample)
    w_sd_all = torch.zeros(2*N, N_sample)
    B_sd_all = torch.zeros(2*N_BA, N_sample)
    A_sd_all = torch.zeros(2*N_BA, N_sample)
    beta_sd_all = torch.zeros(dim_beta, n_reg, N_sample)
    dist_par_un_sd_all = torch.zeros(dim_dist_par_un, N_sample)
    phi_T_sd_all = torch.zeros(2*N, T, N_sample)

    for s in range(N_sample):
        Y_T_s = Y_T_dgp[:, :, :, s]
        beta_0 = beta_T_dgp[:, 1].view(dim_beta, n_reg)

        #single snapshot estimates
        phi_T_ss_s, dist_par_un_ss_s, beta_ss_s, diag_ss_s = \
            model.ss_filt_est_beta_const(Y_T_s, X_T=X_T_dgp, beta=None, phi_T=None,
                                         est_const_beta=True, dim_beta=1,
                                         opt_large_steps=opt_large_steps, opt_n=1,
                                         max_opt_iter_phi=N_steps_max//opt_large_steps, lr_phi=0.01,
                                         max_opt_iter_beta=N_steps_max//opt_large_steps, lr_beta=0.01,
                                         print_flag_phi=False, print_flag_beta=False, min_opt_iter=10)


        #store results ss
        phi_T_ss_all[:, :, s] = phi_T_ss_s.clone().detach()
        beta_ss_all[:, :, s] = beta_ss_s.clone().detach()

        #score driven estimates
        B_0 = 0.95 * torch.ones(2*N)
        A_0 = 0.1 * torch.ones(2*N)
        w_0 = phi_T_dgp.mean(dim=1) * (1-B_0)

        w_sd_s, B_sd_s, A_sd_s, dist_par_un_sd_s, beta_sd_s, diag_sd_s = \
            model.estimate_SD_X0(Y_T_s, X_T_dgp,
                                    dim_beta=dim_beta, n_beta_tv=0,
                                    opt_n=1, max_opt_iter=N_steps_max, lr=0.01,
                                    plot_flag=False, print_flag=False, print_every=2000,
                                    B0=B_0, A0=A_0, W0=w_0, beta_const_0=beta_0)


        phi_T_sd_s, beta_T_sd_s = model.sd_filt(w_sd_s, B_sd_s, A_sd_s, Y_T_s,
                                                beta_const=beta_T_dgp[:, 1].view(dim_beta, n_reg),
                                                X_T=X_T_dgp)

        phi_T_sd_all[:, :, s] = phi_T_sd_s.clone().detach()
        beta_sd_all[:, :, s] = beta_sd_s.clone().detach()
        w_sd_all[:, s] = w_sd_s.clone().detach()
        B_sd_all[:, s] = B_sd_s.clone().detach()
        A_sd_all[:, s] = A_sd_s.clone().detach()
        print(s)

    if False:
        plt.close()
        i_plot = 11
        plt.plot(phi_T_ss_s[i_plot, :])
        plt.plot(phi_T_sd_s[i_plot, :])
        plt.plot(phi_T_dgp[i_plot, :])


    file_path = SAVE_FOLD + '/filter_missp_dgp_dirBin1'+ \
                '_N_' + str(N) + '_T_' + str(T) + \
                '_N_steps_' + str(N_steps_max) + '_N_BA_' + str(N_BA) + \
                '_resc_score_' + str(rescale_score) + '_ovflw_lm_' + str(ovflw_lm) + \
                '_distr_' + distribution + \
                '_dim_beta_' + str(dim_beta) + \
                '_N_sample_' + str(N_sample) + \
                '_type_dgp_' + type_dgp + \
                '.npz'

    print(file_path)
    np.savez(file_path, phi_T_dgp.detach(), beta_T_dgp.detach(), Y_T_dgp.detach(), X_T_dgp.detach(),
                        w_sd_all, B_sd_all, A_sd_all, dist_par_un_sd_all, beta_sd_all, phi_T_sd_all,
                        beta_ss_all, phi_T_ss_all)









