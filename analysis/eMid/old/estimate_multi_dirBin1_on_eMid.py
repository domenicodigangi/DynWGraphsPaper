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

from dirBin1_dynNets import estimate_and_save_dirBin1_models

Y_T = tens(eMid_w_T[:, :, 1:] > 0)
N = Y_T.shape[0]
T = Y_T.shape[2]
T_test = T//5
Y_T_m1 = tens(eMid_w_T[:, :, :-1] > 0).unsqueeze(2)
X_T_eonia = (torch.ones(N, N, T) * tens(eonia_w[1:]).squeeze()).unsqueeze(2)

X_T = torch.cat((Y_T_m1, X_T_eonia), dim=2 )



models_list = np.array([['SS', False, 1],\
                        ['SS', True, 1],\
                        ['SS', True, N],\
                        ['SD', False, 1],\
                        ['SD', True, 1],\
                        ['SD', True, N]])


fold_names_regs = ['prev_t_link', 'eonia', 'prev_t_link_eonia']
regs_inds = [[True, False], [False, True], [True, True]]
models_list_reg = [np.array([0, 1, 1, 0, 1, 1]) > 0,
                   np.array([0, 1, 1, 0, 1, 1]) > 0,
                   np.array([0, 0, 1, 0, 0, 1]) > 0]


fold_name = 'eMid'
SAVE_FOLD_no_reg = './data/estimates_real_data/' + fold_name
N_steps = 20000


#%%
for i in range(2,3):
    SAVE_FOLD = SAVE_FOLD_no_reg + '/' + fold_names_regs[i]
    try:
        os.mkdir(SAVE_FOLD)
    except:
        pass



    def fun_to_iter(filter_type, regr_flag, dim_beta):
        estimate_and_save_dirBin1_models(Y_T, filter_type, regr_flag, SAVE_FOLD, X_T=X_T[:, :, regs_inds[i], :],
                                         dim_beta=dim_beta,
                                         N_steps=N_steps, avoid_ovflw_fun_flag=True, T_test=T_test)

    results = Parallel(n_jobs=2)(delayed(fun_to_iter)(filter_type, regr_flag, int(dim_beta))\
                                 for filter_type, regr_flag, dim_beta\
                                        in models_list[models_list_reg[i]].tolist())

