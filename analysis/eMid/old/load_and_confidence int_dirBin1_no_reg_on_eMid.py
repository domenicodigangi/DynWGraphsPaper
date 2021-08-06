"""

Estimate dirBin1 on eMid data.

"""
# %%
import sys
sys.path.append("./src/")

from dirBin1_dynNets import dirBin1_dynNet_SD
from utils import splitVec, tens
import numpy as np
import torch
import matplotlib.pyplot as plt

ld_data = np.load("../../data/emid_data/numpyFiles/eMid_numpy.npz", allow_pickle=True)

eMid_w_T, all_dates, eonia_w, nodes = ld_data["arr_0"], ld_data["arr_1"], ld_data["arr_2"], ld_data["arr_3"]


Y_T = tens(eMid_w_T[:, :, 1:] > 0)
N = Y_T.shape[0]
T = Y_T.shape[2]
T_train = 100
T_test = T - T_train  # T//5

avoid_ovflw_fun_flag = True
rescale_SD = True

fold_name = 'eMid'
SAVE_FOLD_no_reg = './data/estimates_real_data/' + fold_name
SAVE_FOLD = SAVE_FOLD_no_reg
N_steps = 20000


# %%   # if score driven estimate only on the observations that are not in the test sample
Y_T_train = Y_T[:, :, :-T_test]
N_BA = N

model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=avoid_ovflw_fun_flag, rescale_SD=rescale_SD)

# estimate SS on the first few snapshots

rel_improv_tol_SS = 1e-6
min_opt_iter_SS = 20
bandwidth_SS = min_opt_iter_SS
no_improv_max_count_SS = 20

# %% SS estimates
# file_path = SAVE_FOLD + '/dirBin1_SS_est_lr_' + \
#             str(learn_rate) + '_N_' + str(N) + '_T_' + str(0) + \
#             '_N_steps_' + str(N_steps) + \
#             '.npz'
# l_dat = np.load(file_path, allow_pickle=True)



# %% SD estimates
rel_improv_tol_SD = 1e-16
min_opt_iter_SD = 15000
bandwidth_SD = 250
no_improv_max_count_SD = 500
learn_rate = 0.01
T_init = 5

file_path = SAVE_FOLD + '/dirBin1_SD_est_lr_' + \
    str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + 'T_test_' + str(T_test) + \
    '_N_steps_' + str(N_steps) + '_N_BA_' + str(N_BA) + '.npz'

print(file_path)
l_dat = np.load(file_path, allow_pickle=True)
W_est, B_est, A_est, beta_est, sd_par_0, diag, N_steps, \
learn_rate = l_dat["arr_0"], l_dat["arr_1"], \
             l_dat["arr_2"], l_dat["arr_3"], \
             l_dat["arr_4"], l_dat["arr_5"], \
             l_dat["arr_6"], l_dat["arr_7"]

W_est = tens(W_est)
B_est = tens(B_est)
A_est = tens(A_est)
sd_par_0 = tens(sd_par_0)
um_est = W_est/(1-B_est)

plt.close("all")
fig, axs = plt.subplots(2, 1)
axs[0].plot(um_est)
axs[1].plot(B_est)

g_norm = []
g_mean = []
for d in diag:
    g_norm.append(d[2])
    g_mean.append(d[3])


plt.plot(g_norm)
plt.plot([d[0] for d in diag])


# %%

# cap the unconditional means from below
inds_to_cap = um_est <-10
W_cap = W_est
W_cap[inds_to_cap] = -13 * (1-B_est[inds_to_cap])
A_cap = A_est.clone()
A_cap[A_cap > 2] = 2

phi_T, _ = model.sd_filt(tens(W_cap), tens(B_est), tens(A_est), tens(Y_T ), sd_par_0=tens(sd_par_0))
phi_T_cap, _ = model.sd_filt(tens(W_cap), tens(B_est), tens(A_cap), tens(Y_T ), sd_par_0=tens(sd_par_0))

def ll_fun(W_re, B_re, A_re):
    return model.loglike_sd_filt(W_re, B_re, A_re, tens(Y_T_train), sd_par_0=tens(sd_par_0))

lLikes = [ll_fun(tens(W_est), tens(B_est), tens(A_est)), \
            ll_fun(tens(W_cap), tens(B_est), tens(A_est)),\
            ll_fun(tens(W_cap), tens(B_est), tens(A_cap))]

print([ll.data for ll in lLikes])
phi_T_mod = phi_T.clone()
for t in range(T):
    phi_T_mod[:, t] = model.set_zero_deg_par(Y_T[:, :, t], phi_T[:, t], method="EXPECTATION", degIO=None)

plt.close("all")
fig, axs = plt.subplots(2, 1)
#axs[0].plot(phi_T.transpose(0, 1).detach())
#axs[1].plot(phi_T_cap.transpose(0, 1).detach())
axs[0].plot(phi_T[ tens(A_est).sort()[1][-10:], :].transpose(0, 1).detach())
axs[1].plot(phi_T_mod.transpose(0, 1).detach())


# %% Robust estimate of covariance of static parameters

Y_T_fun = Y_T_train

def req_grads(parTuple):
    for p in parTuple:
        p.requires_grad = True
    # sd par 0 does not require grad
    parTuple[-1].requires_grad = False

def zero_grads(parTuple):
    for p in parTuple:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()



parTuple = (W_est, B_est, A_est, sd_par_0)


# %%
from torch.autograd.functional import hessian
from torch.autograd import grad

req_grads(parTuple)
T = Y_T_fun.shape[2]
N = Y_T_fun.shape[0]
phi_T_fun, _  = model.sd_filt(parTuple[0], parTuple[1], parTuple[2], Y_T_fun, sd_par_0=parTuple[3])

t=45


def mat_A_for_cov(Y_T_fun, parTuple):
    T = Y_T_fun.shape[2]
    N = Y_T_fun.shape[0]
    req_grads(parTuple)
    T_est = 99
    outMatB = torch.zeros(N*2*3, N*2*3, T_est)
    outMatA = torch.zeros(N*2*3, N*2*3, T_est)
    parVec = torch.cat(parTuple[:3])

    for t in range(T_est-1, 1, -1):
        print((t, T))
        def logl_t(parVec):
            phi_T_up_t, _ = model.sd_filt(parVec[:2*N], parVec[2*N:4*N], parVec[4*N:6*N], Y_T_fun[:, :, :t],
                                          sd_par_0=parTuple[3])
            return model.loglike_t(Y_T_fun[:, :, t], phi_T_up_t[:, -1])
        
        s_t = grad(logl_t(parVec), parVec)[0].data
        h_t = hessian(logl_t, parVec).data
        outMatA[:, :, t] = - h_t
        outMatB[:, :, t] = s_t * s_t.unsqueeze(dim=1)
    return outMatB, outMatA

covB, covA = mat_A_for_cov(Y_T_fun, parTuple)


# %%
covA0 = covA.mean(dim=2)
covB0 = covB.mean(dim=2)

covMat = torch.matmul(covA0.cholesky_inverse(), covB0).matmul(covA0.cholesky_inverse())

ei = (covMat + torch.eye(covMat.shape[0]) ).eig()
ei[0][:,0].sort()[0].min()

covMat.diag()


m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(6*N), covMat)
m.sample()



















