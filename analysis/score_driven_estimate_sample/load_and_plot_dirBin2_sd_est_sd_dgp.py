"""
Estimate directed sparse weighted model on WTN data
"""

import sys
import numpy as np
import torch
sys.path.append("./src/")
from utils import splitVec, tens
SAVE_FOLD = './data/estimates_sim_data'
from dirBin2_dynNets import dirBin2_dynNet_SD
import matplotlib.pyplot as plt

#%%
avoid_ovflw_fun_flag = True
rescale_score = False
distribution = 'bernoulli'

torch.manual_seed(2)
N = 8
model = dirBin2_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False, N=N)

T = 1500
N_sample = 20
N_steps_max = 15000

type_dgp = 'SD'


file_path = SAVE_FOLD + '/filter_sd_dgp_dirBin2' + \
            '_N_' + str(N) + '_T_' + str(T) + \
            '_N_steps_' + str(N_steps_max)  + \
            '_resc_score_' + str(rescale_score) + '_avoid_ovflw_fun_flag_' + str(avoid_ovflw_fun_flag) + \
            '_N_sample_' + str(N_sample) + \
            '_type_dgp_' + type_dgp + \
            '.npz'





l_dat = np.load(file_path, allow_pickle=True)

w_dgp, B_dgp, A_dgp, w_sd_all, B_sd_all, A_sd_all,\
Y_T_dgp = tens(l_dat["arr_0"]), tens(l_dat["arr_1"]), \
tens(l_dat["arr_2"]), tens(l_dat["arr_3"]), tens(l_dat["arr_4"]), tens(l_dat["arr_5"]), tens(l_dat["arr_6"])

#%%
# plt.subplot(3, 1, 1)
# plt.hist(w_sd_all)
# plt.axvline(w_dgp)
# plt.subplot(3, 1, 2)
# plt.hist(B_sd_all)
# plt.axvline(B_dgp)
# plt.subplot(3, 1, 3)
# plt.hist(A_sd_all)
# plt.axvline(A_dgp)

#%% compute numerical variance covariance of estimators
from torch.autograd import grad

sd_par_est = np.stack((w_sd_all, B_sd_all, A_sd_all), axis=0)
sd_cov_est = np.cov(sd_par_est).diagonal()





#%%
N_plot = 20
hess_diag = torch.zeros(3, N_sample)

for s in range(N_plot):
    print(s)
    w, B, A = tens(w_sd_all[s]).unsqueeze(-1), tens(B_sd_all[s]).unsqueeze(-1), tens(A_sd_all[s]).unsqueeze(-1)
    Y_T = tens(Y_T_dgp[:, :, :, s])
    sd_par = torch.cat((w, B, A))
    sd_par.requires_grad = True
    like_fun = model.loglike_sd_filt(sd_par[0].unsqueeze(-1), sd_par[1].unsqueeze(-1), sd_par[2].unsqueeze(-1), Y_T)
    score = grad(like_fun, sd_par, create_graph=True)[0]
    drv2 = []
    for i in range(sd_par.shape[0]):
        tmp = score[i]
        # compute the second derivatives of the loglikelihood
        drv2.append(grad(tmp, sd_par, retain_graph=True)[0][i])
    drv2 = -torch.stack(drv2)
    hess_diag[:, s] = drv2


#%%
# plt.close('all')
plt.figure()
sd_est_var = 1/hess_diag[:, :N_plot].sqrt()
plt.subplot(3, 1, 1)
plt.hist(sd_est_var[0, :])
plt.axvline(sd_cov_est[0])
plt.subplot(3, 1, 2)
plt.hist(sd_est_var[1, :])
plt.axvline(sd_cov_est[1])
plt.subplot(3, 1, 3)
plt.hist(sd_est_var[2, :])
plt.axvline(sd_cov_est[2])


#%% test scores
model = dirBin2_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False, N=N)

s=0
w, B, A = 10*tens(w_sd_all[s]).unsqueeze(-1), tens(B_sd_all[s]).unsqueeze(-1), tens(A_sd_all[s]).unsqueeze(-1)
Y_T = tens(Y_T_dgp[:, :, :, s])
Y_t = Y_T[:, :, 13]
model.score_t(Y_t, w, backprop=False)
model.score_t(Y_t, w, backprop=True)



#%%

delta_w = w_sd_all - np.tile(w_dgp, (N_sample, 1)).transpose()

A_dgp
B_dgp
plt.close()
plt.hist(delta_w)
i_plot = 11
#plt.plot(delta_w[i_plot, :])

#%%

#%%
s=1
i_plot = 1


phi_T_dgp, o = model.sd_filt(tens(w_dgp), tens(B_dgp), tens(A_dgp), tens(Y_T_dgp[:, :, :, s]))

phi_T_est, o = model.sd_filt(tens(w_dgp), tens(B_dgp), tens(A_dgp), tens(Y_T_dgp[:, :, :, s]))

plt.plot(phi_T_dgp.detach().numpy().transpose(), '.b', markersize=0.5)


#%%






