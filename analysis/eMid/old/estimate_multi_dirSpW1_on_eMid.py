"""
Given data Estimate multiple directed binary models in parallel
"""

import os
import sys
import numpy as np
import torch
sys.path.append("./src/")
from utils import splitVec, tens
from joblib import Parallel, delayed

ld_data = np.load( "../../data/emid_data/numpyFiles/eMid_numpy.npz",
                  allow_pickle=True)

eMid_w_T, all_dates, eonia_w, nodes = ld_data["arr_0"], ld_data["arr_1"], ld_data["arr_2"], ld_data["arr_3"]

from dirSpW1_dynNets import estimate_and_save_dirSpW1_models

Y_T = tens(eMid_w_T[:, :, 1:] > 0)
N = Y_T.shape[0]
T = Y_T.shape[2]
T_test = T//5
unit_measure = 1e6
avoid_ovflw_fun_flag = True


# Define regressors
Y_T_m1 = tens(eMid_w_T[:, :, :-1] > 0).unsqueeze(2).log()
X_T_eonia = (torch.ones(N, N, T) * tens(eonia_w[1:]).squeeze()).unsqueeze(2)

X_T = torch.cat((Y_T_m1, X_T_eonia), dim=2 )




all_SS_models = []
all_SS_models.append(['SS', 'gamma', N, False, 1, 20000, 0.005])
# all_SS_models.append(['SS', 'lognormal', N, False, 1, 20000, 0.005])
all_SS_models.append(['SS', 'gamma', N, True, 1, 20000, 0.005])
# all_SS_models.append(['SS', 'lognormal', N, True, 1, 20000, 0.005])
# all_SS_models.append(['SS', 'gamma', N, True, N, 20000, 0.005])
# all_SS_models.append(['SS', 'lognormal', N, True, N, 20000, 0.005])

all_SD_models = []
all_SD_models.append(['SD', 'gamma', N, False, 1, 20000, 0.005])
# all_SD_models.append(['SD', 'lognormal', N, False, 1, 20000, 0.005])
all_SD_models.append(['SD', 'gamma', N, True, 1, 50000, 0.0001])
# all_SD_models.append(['SD', 'lognormal', N, True, 1, 50000, 0.0001])
# all_SD_models.append(['SD', 'gamma', N, True, N, 50000, 0.0001])
# all_SD_models.append(['SD', 'lognormal', N, True, N, 50000, 0.0001])




fold_name = 'eMid'
SAVE_FOLD_no_reg = './data/estimates_real_data/' + fold_name


fold_names_regs = [ 'eonia', 'prev_t_link',  'prev_t_link_eonia' ]
regs_inds = [[True, False], [False, True], [True, True]]

for rescale_score in [False, True]:
    for i in range(3):
        SAVE_FOLD = SAVE_FOLD_no_reg + '/' + fold_names_regs[i]
        try:
            os.mkdir(SAVE_FOLD)
        except:
            pass


        def fun_to_iter(filter_type, distribution, dim_dist_par, regr_flag, dim_beta, N_steps, learn_rate):
            if (not regr_flag) and (not fold_names_regs[i] == 'eonia'):
                # estimate only once the models without regressors
                pass
            else:
                if (not regr_flag) and (fold_names_regs[i] == 'eonia'):
                    SAVE_FOLD_in = [SAVE_FOLD_no_reg + '/' + name_reg for name_reg in fold_names_regs]
                else:
                    SAVE_FOLD_in = SAVE_FOLD

                estimate_and_save_dirSpW1_models(Y_T, distribution, dim_dist_par, filter_type, regr_flag, SAVE_FOLD_in,
                                                 X_T=X_T[:, :, regs_inds[i], :], dim_beta=dim_beta, n_beta_tv=0,
                                                 unit_measure=unit_measure,
                                                 learn_rate=learn_rate, T_test=T_test,
                                                 N_steps=N_steps, avoid_ovflw_fun_flag=avoid_ovflw_fun_flag, rescale_score=rescale_score,
                                                 load_ss=True)

        print((unit_measure,  'SS', T_test, avoid_ovflw_fun_flag, rescale_score, fold_names_regs[i]))

        results = Parallel(n_jobs=1)(delayed(fun_to_iter)(filter_type, distribution, dim_dist_par, regr_flag, dim_beta,
                                                          N_steps, learn_rate)\
                                     for filter_type, distribution, dim_dist_par, regr_flag, dim_beta, N_steps, learn_rate\
                                            in all_SS_models)


        print((unit_measure, 'SD', T_test, avoid_ovflw_fun_flag, rescale_score, fold_names_regs[i]))

        results = Parallel(n_jobs=1)(delayed(fun_to_iter)(filter_type, distribution, dim_dist_par, regr_flag, dim_beta,
                                                           N_steps, learn_rate)\
                                     for filter_type, distribution, dim_dist_par, regr_flag, dim_beta, N_steps, learn_rate\
                                            in all_SD_models)









