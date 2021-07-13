"""
Estimate dirBin1 on eMid data.
"""

import sys
sys.path.append("./src/")

from dirBin1_dynNets import dirBin1_dynNet_SD
from utils import splitVec, tens
import numpy as np
import torch

ld_data = np.load("../../data/emid_data/numpyFiles/eMid_numpy.npz", allow_pickle=True)

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
N_steps = 20000


#%%   # if score driven estimate only on the observations that are not in the test sample
Y_T_train = Y_T[:, :, :-T_test]
N_BA = N

model = dirBin1_dynNet_SD(ovflw_lm=ovflw_lm, rescale_SD=rescale_SD)

# estimate SS on the first few snapshots
import matplotlib.pyplot as plt

phi_T_0 = torch.zeros(N * 2, T)
for t in range(10):
    phi_T_0[:, t] = model.start_phi_from_obs(Y_T[:, :, t], n_iter = 80)

learn_rate = 0.01
rel_improv_tol_SS = 1e-6
min_opt_iter_SS = 20
bandwidth_SS = min_opt_iter_SS
no_improv_max_count_SS = 20
phi_ss_est_T = phi_T_0

file_path = SAVE_FOLD + '/dirBin1_SS_est_lr_' + \
            str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
            '_N_steps_' + str(N_steps) + \
            '.npz'
#print(file_path)

#plt.plot(phi_ss_est_T.transpose(0, 1).detach())
#np.savez(file_path, phi_ss_est_T.detach(), None, None)

#%% Estimate Score Driven
N_steps = 2500
rel_improv_tol_SD = 1e-16
min_opt_iter_SD = 500
bandwidth_SD = 250
no_improv_max_count_SD = 5
print_every = 100
T_init = 5
opt_algo = "ADAM"
learn_rate = 0.01
B0 = torch.ones(2*N_BA) * 0.98
A0 = torch.ones(2*N_BA) * 0.01
W0 = phi_ss_est_T[:, :T_init].mean(dim=1) * (1 - B0)

W_est, B_est, A_est, dist_par_un_est, sd_par_0, diag = model.estimate_SD(Y_T_train, B0=B0, A0=A0, W0=W0,
                    opt_n=opt_algo,
                                                                                                    max_opt_iter=N_steps,
                                                                                                    lr=learn_rate,
                                                                                                    sd_par_0=phi_T_0[:, :T_init].mean(dim=1),
                                                                                                    print_flag=True, print_every=print_every,
                                                                                                    plot_flag=False,
                                                                                                    est_dis_par_un=False,
                                                                                                    init_filt_um=False,
                                                                                                    rel_improv_tol=rel_improv_tol_SD,
                                                                                                    no_improv_max_count=no_improv_max_count_SD,
                                                                                                    min_opt_iter=min_opt_iter_SD,
                                                                                                    bandwidth=bandwidth_SD,
                                                                                                    small_grad_th=1e-3)

file_path = SAVE_FOLD + '/dirBin1_SD_est_lr_' + \
    str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + 'T_test_' + str(T_test) + \
    '_N_steps_' + str(N_steps) + "_" + opt_algo + '_N_BA_' + str(N_BA) + '.npz'
print(file_path)
np.savez(file_path, W_est.detach(), B_est.detach(), A_est.detach(), None,
         sd_par_0.detach(), diag, N_steps, learn_rate)

#%%
sd_par_0 = phi_T_0[:, :T_init].mean(dim=1)
rePar_0 = torch.cat((tens(W0), torch.cat((B0, A0)))).clone().detach()
n = n_b = n_a = W0.shape[0]
def obj_fun_un(Y_T, unPar, n, n_B, n_A):
    reBA = model.un2re_BA_par(unPar[n: n + n_B + n_A])
    return - model.loglike_sd_filt(unPar[:n], reBA[:n_B], reBA[n_B:n_B + n_A], Y_T,
                                   sd_par_0=sd_par_0)

def f_un(unPar):
    return obj_fun_un(Y_T_train, unPar, n, n_b, n_a)
def obj_fun_re(Y_T, rePar, n, n_B, n_A):
    reBA = rePar[n: n + n_B + n_A]
    return - model.loglike_sd_filt(rePar[:n], reBA[:n_B], reBA[n_B:n_B + n_A], Y_T,
                                   sd_par_0=sd_par_0)
def f_re(rePar):
    return obj_fun_re(Y_T_train, rePar, n, n_b, n_a)



unPar0 = torch.cat((tens(W0), model.re2un_BA_par(torch.cat((B0, A0))))).clone().detach()

#from utils import optim_torch
out_par_1, daig_1 = optim_torch(f_un, unPar0, max_opt_iter=10, opt_n="LBFGS_NEW", lr=0.01, print_every=1)

#%%


unPar_est = torch.cat((tens(W_est), model.re2un_BA_par(torch.cat((B_est, A_est))))).clone().detach()
unPar_est.requires_grad = True
loss = obj_fun_un(Y_T_train, unPar_est, n, n_b, n_a)
loss.backward()
unPar_est.grad.abs().norm()
diag[-1]


f_re(rePar_est)
f_un(unPar_est)

