"""
Given data Estimate multiple directed binary models in parallel
"""

import sys
import numpy as np
import torch
sys.path.append("./src/")
from utils import splitVec, tens
from joblib import Parallel, delayed
import itertools

#load yearly world Trade networks
ld_data = np.load("./data/world_trade_network/world_trade_net_T.npz",
                  allow_pickle=True)
wtn_T, all_y, nodes= ld_data["wtn_T"], ld_data["all_y"], ld_data["nodes"]
dist_T, scaling_infl = ld_data["dist_T"], ld_data["scaling_infl"]

fold_name = 'WTN'
SAVE_FOLD = './data/estimates_real_data/' + fold_name

from dirBin1_dynNets import estimate_and_save_dirBin1_models

Y_T = tens(wtn_T > 0)
X_T = tens(dist_T).log().unsqueeze(2)#torch.tensor((dist_T - dist_T.mean())/dist_T.std()).unsqueeze(2)

N = Y_T.shape[0]
N_steps = 20000

# print((unit_measure,  'SS', T_test, ovflw_lm, rescale_score))
#
# all_models = []
# #all_models.append(['SS', False, 1])
# all_models.append(['SS', True, 1])
# all_models.append(['SD', False, 1])
# all_models.append(['SD', True, 1])
#
#
# # all_models.append(['SS', True, N])
# # all_models.append(['SD', True, N])
#
#
# for rescale_score in [True]:
#     def fun_to_iter(filter_type, regr_flag, dim_beta):
#         estimate_and_save_dirBin1_models(Y_T, filter_type, regr_flag, SAVE_FOLD, X_T=X_T, dim_beta=dim_beta,
#                                          N_steps=N_steps, ovflw_lm=True, T_test=10, rescale_score=rescale_score)
#
#
#     results = Parallel(n_jobs=1)(delayed(fun_to_iter)(filter_type, regr_flag, dim_beta)\
#                                  for filter_type, regr_flag, dim_beta\
#                                         in all_models )


all_models = []
# all_models.append(['SS', False, 1])
# all_models.append(['SS', True, 1])
all_models.append(['SD', False, 1])
all_models.append(['SD', True, 1])

for rescale_score in [False]:
    def fun_to_iter(filter_type, regr_flag, dim_beta):
        estimate_and_save_dirBin1_models(Y_T, filter_type, regr_flag, SAVE_FOLD, X_T=X_T, dim_beta=dim_beta,
                                         N_steps=N_steps, ovflw_lm=True, T_test=10, rescale_score=rescale_score)


    results = Parallel(n_jobs=1)(delayed(fun_to_iter)(filter_type, regr_flag, dim_beta)\
                                 for filter_type, regr_flag, dim_beta\
                                        in all_models )

