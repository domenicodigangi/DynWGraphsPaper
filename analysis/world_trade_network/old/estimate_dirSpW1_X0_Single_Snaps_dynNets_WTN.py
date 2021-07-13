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


#%% estimate and save different sizes of dist_par
from dirSpW1_dynNets import  dirSpW1_dynNet_SD
SAVE_FOLD = './data/estimates_real_data/WTN'

N_steps = 15000
N_steps_iter = 100
learn_rate = 0.01
X_T = tens(dist_T).log().unsqueeze(2) # torch.tensor((dist_T - dist_T.mean())/dist_T.std()).unsqueeze(2)
N = wtn_T.shape[0]
T = wtn_T.shape[2]
ovflw_lm=True

for distribution in ['lognormal', 'gamma']:
    model = dirSpW1_dynNet_SD(ovflw_lm=ovflw_lm, distribution=distribution)
    for dim_beta in [1, N]:
        for unit_measure in [1e6]:
            for dim_dist_par_un in [N]:
                # Set unit measure and  rescale for inflation
                Y_T = torch.tensor(wtn_T * scaling_infl[:, 1])/unit_measure
                # estimate single snapshot
                phi_ss_est_T, dist_par_un, beta_est, diag = \
                    model.ss_filt_est_beta_dist_par_const(Y_T, X_T=X_T, beta=None, phi_T=None, dist_par_un=None,
                                                          est_const_dist_par=True, dim_dist_par_un=dim_dist_par_un,
                                                          est_const_beta=True, dim_beta=dim_beta,
                                                          opt_large_steps=N_steps//N_steps_iter,
                                                          max_opt_iter_phi=N_steps_iter, lr_phi=learn_rate,
                                                          max_opt_iter_dist_par=N_steps_iter, lr_dist_par=learn_rate,
                                                          max_opt_iter_beta=N_steps_iter, lr_beta=learn_rate,
                                                          print_flag_phi=False, print_flag_dist_par=True,
                                                          print_flag_beta=True,
                                                          print_every=250)

                file_path = SAVE_FOLD + '/WTN_dirSpW1_X0_dynNet_Single_Snap_est__lr_' + \
                            str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                            '_N_steps_' + str(N_steps) + \
                            '_ovflw_lm_' + str(ovflw_lm) + \
                            '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                            '_dim_beta_' + str(dim_beta) + \
                            distribution + 'distr_' + 'dim_distr_par_' + str(dim_dist_par_un) + '.npz'
                print(file_path)
                np.savez(file_path, phi_ss_est_T.detach(), dist_par_un.detach(), beta_est.detach(),
                         diag, N_steps, N_steps_iter,
                         unit_measure, learn_rate)





