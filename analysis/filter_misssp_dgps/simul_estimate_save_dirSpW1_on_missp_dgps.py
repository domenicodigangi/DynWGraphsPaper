"""
Estimate directed sparse weighted model on misspecified dgps
"""
import sys
import numpy as np
import torch
sys.path.append("./src/")
SAVE_FOLD = './data/estimates_sim_data'


from dirSpW1_dynNets import dirSpW1_dynNet_SD
N = 30
T = 100
rescale_score = False
ovflw_lm = True
distribution = 'lognormal'#'gamma'##'' #  #
model = dirSpW1_dynNet_SD(ovflw_lm=ovflw_lm, rescale_SD=rescale_score, distribution=distribution)

for type_dgp in ['ar1']:#]: #'ar1'
    n_reg = 2
    n_reg_beta_tv = 0
    dim_beta = 1
    dim_dist_par_un = 1
    N_sample = 50
    N_steps_max = 15000
    opt_large_steps = 50
    N_BA = N
    est_dist_par=True
    #%
    #const_p = 0.9
    for const_p in [0.9, 0.5, 0.25, 0.1, 1/N]:
        print((const_p, N_sample, type_dgp, distribution, rescale_score))
        # Sample the Dgp
        Y_T_dgp, phi_T_dgp, X_T_dgp, beta_T_dgp, dist_par_un_dgp = \
            model.sample_from_dgps(N, T, N_sample, p_T=torch.ones(N, N, T)*const_p,
                                        dgp_type=type_dgp,
                                        n_reg=n_reg, n_reg_beta_tv=n_reg_beta_tv,
                                        dim_beta=dim_beta, dim_dist_par_un=dim_dist_par_un,
                                        distribution=distribution)

        #define storage variables
        beta_ss_all = torch.zeros(dim_beta, n_reg, N_sample)
        dist_par_un_ss_all = torch.zeros(dim_dist_par_un, N_sample)
        phi_T_ss_all = torch.zeros(2*N, T, N_sample)
        sd_par_0_all = torch.zeros(2*N, N_sample)
        w_sd_all = torch.zeros(2*N, N_sample)
        B_sd_all = torch.zeros(2*N_BA, N_sample)
        A_sd_all = torch.zeros(2*N_BA, N_sample)
        beta_sd_all = torch.zeros(dim_beta, n_reg, N_sample)
        dist_par_un_sd_all = torch.zeros(dim_dist_par_un, N_sample)
        phi_T_sd_all = torch.zeros(2*N, T, N_sample)
        #s = 0
        for s in range(N_sample):
            Y_T_s = Y_T_dgp[:, :, :, s]
            beta_0 = beta_T_dgp[:, 1].view(dim_beta, n_reg)
            beta_0 = beta_0 + torch.randn(beta_0.shape)
            phi_T_0 = phi_T_dgp + torch.randn(phi_T_dgp.shape)*phi_T_dgp
            # #single snapshot estimates
            # phi_T_ss_s, dist_par_un_ss_s, beta_ss_s, diag_ss_s = \
            #     model.ss_filt_est_beta_dist_par_const(Y_T_s, X_T=X_T_dgp,
            #                                             phi_T = phi_T_0,
            #                                             beta=beta_0,
            #                                             est_const_dist_par=est_dist_par, dim_dist_par_un=dim_dist_par_un,
            #                                             est_const_beta=True, dim_beta=dim_beta,
            #                                             opt_large_steps=opt_large_steps, opt_n=1,
            #                                             opt_steps_phi=N_steps_max//opt_large_steps, lRate_phi=0.01,
            #                                             opt_steps_dist_par=N_steps_max//opt_large_steps, lRate_dist_par=0.01,
            #                                             opt_steps_beta=N_steps_max//opt_large_steps, lRate_beta=0.01,
            #                                             min_n_iter=20,
            #                                             print_flag_phi=False,
            #                                             print_flag_dist_par=False,
            #                                             print_flag_beta=False)
            #
            # #store results ss
            # phi_T_ss_all[:, :, s] = phi_T_ss_s.clone().detach()
            # dist_par_un_ss_all[:, s] = dist_par_un_ss_s.clone().detach()
            # beta_ss_all[:, :, s] = beta_ss_s.clone().detach()


            #score driven estimates
            B_0 = 0.95 * torch.ones(2*N)
            A_0 = 0.001 * torch.ones(2*N)
            w_0 = phi_T_0.mean(dim=1) * (1-B_0)
            model = dirSpW1_dynNet_SD(ovflw_lm=ovflw_lm, rescale_SD=rescale_score, distribution=distribution)

            w_sd_s, B_sd_s, A_sd_s, dist_par_un_sd_s, beta_sd_s, sd_par_0_s, diag_sd_s = \
                model.estimate_SD_X0(Y_T_s.clone().detach(), X_T_dgp.clone().detach(),
                                        dim_beta=dim_beta, n_beta_tv=0, dim_dist_par_un=dim_dist_par_un,
                                        opt_n=1, opt_steps=N_steps_max, lRate=0.005,
                                        plot_flag=False, print_flag=True, print_every=500,
                                        B0=B_0, A0=A_0, W0=w_0, beta_const_0=beta_0, est_dis_par_un=True,
                                        rel_improv_tol=1e-7, no_improv_max_count=50,
                                        min_n_iter=750, bandwidth=100, small_grad_th=1e-6)


            phi_T_sd_s, beta_T_sd_s = model.sd_filt(w_sd_s, B_sd_s, A_sd_s, Y_T_s, beta_const=beta_T_dgp[:, 1].view(dim_beta, n_reg),
                            X_T=X_T_dgp, dist_par_un=torch.zeros(1), sd_par_0=sd_par_0_s)

            phi_T_sd_all[:, :, s] = phi_T_sd_s.clone().detach()
            dist_par_un_sd_all[:, s] = dist_par_un_sd_s.clone().detach()
            beta_sd_all[:, :, s] = beta_sd_s.clone().detach()
            w_sd_all[:, s] = w_sd_s.clone().detach()
            B_sd_all[:, s] = B_sd_s.clone().detach()
            A_sd_all[:, s] = A_sd_s.clone().detach()
            sd_par_0_all[:, s] = sd_par_0_s.clone().detach()
            print(s)


        file_path = SAVE_FOLD + '/filter_missp_dgp_diSpW1'+ \
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
        np.savez(file_path, phi_T_dgp.detach(), dist_par_un_dgp.detach(), beta_T_dgp.detach(), Y_T_dgp.detach(),
                            X_T_dgp.detach(),
                            w_sd_all, B_sd_all, A_sd_all, dist_par_un_sd_all, beta_sd_all, sd_par_0_all, phi_T_sd_all,
                            dist_par_un_ss_all, beta_ss_all, phi_T_ss_all)









