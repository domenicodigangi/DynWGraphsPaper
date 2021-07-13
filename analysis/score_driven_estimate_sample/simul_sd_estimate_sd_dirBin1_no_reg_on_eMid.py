"""

Comparison of different estimation methods:

Estimate dirBin1 on eMid data. Then sample many times from SD dgp with those parameters and estimate each time

"""

import sys
sys.path.append("./src/")

from dirBin1_dynNets import dirBin1_dynNet_SD
from utils import splitVec, tens
import numpy as np

import torch
import pickle
#import matplotlib.pyplot as plt



#% Load real data
ld_data = np.load("../../data/emid_data/numpyFiles/eMid_numpy.npz",
                  allow_pickle=True)


eMid_w_T, all_dates, eonia_w, nodes = ld_data["arr_0"], ld_data["arr_1"], ld_data["arr_2"], ld_data["arr_3"]
Y_T = tens(eMid_w_T[:, :, 1:] > 0)
N = Y_T.shape[0]
T = Y_T.shape[2]
T_train = 100
T_test = T - T_train  # T//5

ovflw_lm = True
rescale_SD = True
fold_name = 'eMid'
SAVE_FOLD_no_reg = './data/estimates_real_data/' + fold_name
SAVE_FOLD = SAVE_FOLD_no_reg
N_steps = 2500

#% define model and load SD estimates on real data
N_BA = N
model = dirBin1_dynNet_SD(ovflw_lm=ovflw_lm, rescale_SD=rescale_SD)
opt_algo = "LBFGS"
rel_improv_tol_SD = 1e-16
min_opt_iter_SD = 100
bandwidth_SD = 250
no_improv_max_count_SD = 20
print_every = 1000
learn_rate = 0.1

T_init = 5
file_path = SAVE_FOLD + '/dirBin1_SD_est_lr_' + \
    str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + 'T_test_' + str(T_test) + \
    '_N_steps_' + str(N_steps) + "_" + opt_algo + '_N_BA_' + str(N_BA) + '.npz'

print(file_path)
l_dat = np.load(file_path, allow_pickle=True)
W_est, B_est, A_est, beta_est, sd_par_0, diag, N_steps, \
learn_rate = l_dat["arr_0"], l_dat["arr_1"], \
             l_dat["arr_2"], l_dat["arr_3"], \
             l_dat["arr_4"], l_dat["arr_5"], \
             l_dat["arr_6"], l_dat["arr_7"]


#g_norm = []
#g_mean = []
#for d in diag:
#    g_norm.append(d[2])
#    g_mean.append(d[3])
#plt.plot(g_norm)
#plt.plot([d[0] for d in diag])


W_dgp = tens(W_est)
B_dgp = tens(B_est)
# avoid integrated dgp
B_dgp[B_dgp >0.98 ] = 0.98
A_dgp = tens(A_est)
um_dgp = W_dgp/(1-B_dgp)
#sd_par_0_dgp = tens(sd_par_0)
sd_par_0_dgp = um_dgp

#for i in range(5):
#    phi_T, _, Y_T = model.sd_dgp(W_dgp, B_dgp, A_dgp, N, 200, sd_par_0=sd_par_0_dgp)
#    plt.plot(Y_T.sum(dim=(0,1)))

#%

N_steps = 15000
lrs = {"ADAMHD": [0.0001], "SGDHD": [0.005], \
          "ADAM": [0.1, 0.01 ], "LBFGS": [0.01 ]}
n_sample = 40
#T_sample = 500

opt_algo_list = ["ADAM", "LBFGS"] # ["LBFGS"] # ["ADAMHD"] # ["LBFGS"] #["SGDHD"] #  ], ]
store_par = {par_name: {opt_algo: {lr: np.zeros((2*N, n_sample)) for lr in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]} for opt_algo in opt_algo_list}
           for par_name in ["W", "B", "A"]}
 
for opt_algo in opt_algo_list:
    for lr in lrs[opt_algo]:
        for T_sample in [ 100, 500]:
            for n in range(n_sample):
                print(opt_algo)
                print(N_steps)
                print(T_sample)
                print(lr)
                print(n)
                # sample dgp with the estimated parameters
                phi_T, _, Y_T_sample = model.sd_dgp(W_dgp, B_dgp, A_dgp, N, T_sample, sd_par_0=sd_par_0_dgp)
                # repeaat the estimate
                W_est, B_est, A_est, dist_par_un_est, \
                sd_par_0, diag = model.estimate_SD(Y_T_sample.detach(), B0=B_dgp, A0=A_dgp, W0=W_dgp,
                                                     opt_n=opt_algo,
                                                     max_opt_iter=N_steps,
                                                     lr=tens(lr),
                                                     sd_par_0=sd_par_0_dgp,
                                                     print_flag=True, print_every=print_every,
                                                     plot_flag=False,
                                                     est_dis_par_un=False,
                                                     init_filt_um=False,
                                                     rel_improv_tol=rel_improv_tol_SD,
                                                     no_improv_max_count=no_improv_max_count_SD,
                                                     min_opt_iter=min_opt_iter_SD,
                                                     bandwidth=bandwidth_SD,
                                                     small_grad_th=1e-3)

                store_par["W"][opt_algo][lr][:, n] = W_est.detach().numpy()
                store_par["B"][opt_algo][lr][:, n] = B_est.detach().numpy()
                store_par["A"][opt_algo][lr][:, n] = A_est.detach().numpy()




            SAVE_FOLD = './data/estimates_sim_data/'
            file_path = SAVE_FOLD + '/dirBin1_SD_consist_test_eMid_' + \
                 '_N_' + str(N) + '_T_sample_' + str(T_sample) + \
                '_N_steps_' + str(N_steps) + "_" + opt_algo + '_N_BA_' + str(N_BA) + \
                "_"+  opt_algo + str(lr) + '_.pickle'

            with open( file_path , 'wb') as handle:
                pickle.dump(store_par, handle, protocol=pickle.HIGHEST_PROTOCOL)

SAVE_FOLD = './data/estimates_sim_data/'
file_path = SAVE_FOLD + '/dirBin1_SD_consist_test_eMid_' + \
     '_N_' + str(N) + '_T_sample_' + str(T_sample) + \
    '_N_steps_' + str(N_steps) + "_" + opt_algo + '_N_BA_' + str(N_BA) + \
    "_multi_algos_and_lr_" +  '.pickle'

with open( file_path , 'wb') as handle:
    pickle.dump(store_par, handle, protocol=pickle.HIGHEST_PROTOCOL)










