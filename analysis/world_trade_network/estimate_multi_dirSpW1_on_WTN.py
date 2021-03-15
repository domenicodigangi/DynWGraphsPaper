"""
Given data Estimate multiple directed binary models in parallel
"""

import sys
import numpy as np
import torch
import os
sys.path.append("./src/")
from utils import splitVec, tens
from joblib import Parallel, delayed

#load yearly world Trade networks
ld_data = np.load("./data/world_trade_network/world_trade_net_T.npz",
                  allow_pickle=True)
wtn_T, all_y, nodes= ld_data["wtn_T"], ld_data["all_y"], ld_data["nodes"]
dist_T, scaling_infl = ld_data["dist_T"], ld_data["scaling_infl"]

fold_name = 'WTN'
SAVE_FOLD_no_reg = './data/estimates_real_data/' + fold_name


from dirSpW1_dynNets import estimate_and_save_dirSpW1_models



Y_T_all = tens(wtn_T * scaling_infl[:, 1])
Y_T = Y_T_all[:, :, 1:]

reg_name = 'distances'

SAVE_FOLD_reg = SAVE_FOLD_no_reg + '/' + reg_name
try:
    os.mkdir(SAVE_FOLD_reg)
except:
    pass

if reg_name == 'distances':
    X_T = tens(dist_T[:, :, 1:]).log().unsqueeze(2)#torch.tensor((dist_T - dist_T.mean())/dist_T.std()).unsqueeze(2)
elif reg_name == 'prev_link_t':
    X_T = Y_T_all[:, :, :-1].log().unsqueeze(2)
X_T[~torch.isfinite(X_T)] = 0

N = Y_T.shape[0]
unit_measure = 1e6
T_test = 10
ovflw_lm = True

for rescale_score in [False ]:
    def fun_to_iter(filter_type, distribution, dim_dist_par, regr_flag, dim_beta, N_steps, learn_rate):
        if regr_flag:
            SAVE_FOLD = SAVE_FOLD_reg
        else:
            SAVE_FOLD = SAVE_FOLD_no_reg
        estimate_and_save_dirSpW1_models(Y_T, distribution, dim_dist_par, filter_type, regr_flag, SAVE_FOLD,
                                         X_T=X_T, dim_beta=dim_beta, n_beta_tv=0, unit_measure=unit_measure,
                                        learn_rate=learn_rate, T_test=T_test,
                                        N_steps=N_steps, ovflw_lm=ovflw_lm, rescale_score=rescale_score,
                                         load_ss_as_0=True)

    print((reg_name, unit_measure,  'SS', T_test, ovflw_lm, rescale_score))
    all_SS_models = []
    all_SS_models.append(['SS', 'gamma', N, False, 1, 20000, 0.005])
    # all_SS_models.append(['SS', 'lognormal', N, False, 1, 20000, 0.005])
    all_SS_models.append(['SS', 'gamma', N, True, 1, 20000, 0.005])
    # all_SS_models.append(['SS', 'lognormal', N, True, 1, 20000, 0.005])
    # all_SS_models.append(['SS', 'gamma', N, True, N, 20000, 0.005])
    # all_SS_models.append(['SS', 'lognormal', N, True, N, 20000, 0.005])


    results = Parallel(n_jobs=2)(delayed(fun_to_iter)(filter_type, distribution, dim_dist_par, regr_flag, dim_beta,
                                                      N_steps, learn_rate)\
                                 for filter_type, distribution, dim_dist_par, regr_flag, dim_beta, N_steps, learn_rate\
                                        in all_SS_models)


    print((unit_measure, 'SD', T_test, ovflw_lm, rescale_score))
    all_SD_models = []
    all_SD_models.append(['SD', 'gamma', N, False, 1, 20000, 0.005])
    # all_SD_models.append(['SD', 'lognormal', N, False, 1, 20000, 0.005])
    all_SD_models.append(['SD', 'gamma', N, True, 1, 20000, 0.005])
    # all_SD_models.append(['SD', 'lognormal', N, True, 1, 50000, 0.0001])
    # all_SD_models.append(['SD', 'gamma', N, True, N, 50000, 0.0001])
    # all_SD_models.append(['SD', 'lognormal', N, True, N, 50000, 0.0001])



    results = Parallel(n_jobs=2)(delayed(fun_to_iter)(filter_type, distribution, dim_dist_par, regr_flag, dim_beta,
                                                       N_steps, learn_rate)\
                                 for filter_type, distribution, dim_dist_par, regr_flag, dim_beta, N_steps, learn_rate\
                                        in all_SD_models)


