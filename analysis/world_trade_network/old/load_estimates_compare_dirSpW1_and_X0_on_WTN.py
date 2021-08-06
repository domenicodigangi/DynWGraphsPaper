"""
Estimate directed sparse weighted model with distances as regressors on WTN data
"""

import sys
import numpy as np
import torch
sys.path.append("./src/")
import matplotlib.pyplot as plt

from utils import tens
#load yearly world Trade networks
ld_data = np.load("./data/world_trade_network/world_trade_net_T.npz",
                  allow_pickle=True)
wtn_T, all_y, nodes= ld_data["wtn_T"], ld_data["all_y"], ld_data["nodes"]
dist_T, scaling_infl = ld_data["dist_T"], ld_data["scaling_infl"]


# Set unit measure and  rescale for inflation

from dirSpW1_dynNets import dirSpW1_dynNet_SD, dirSpW1_X0_dynNet_SD
#% load and look at single snapshot estimates No regressors and fixed distr_par
unit_measure = 1e6# [1e6, 1e9, 1e12]:
Y_T = torch.tensor(wtn_T * scaling_infl[:, 1])/unit_measure
X_T = torch.tensor(dist_T/dist_T.mean())
N = Y_T.shape[0]
T = Y_T.shape[2]

SAVE_FOLD = './data/estimates_real_data/WTN'

# %% SS weighted no reg
N_steps = 10000
learn_rate = 0.01
avoid_ovflw_fun_flag = True
all_mod_log_likes = []
models_info = []
for dim_dist_par_un in [N]:
    for distribution in ['gamma', 'lognormal']:
        file_path = SAVE_FOLD + '/WTN_dirSpW1_dynNet_Single_Snap_est__lr_' + \
                    str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                    '_N_steps_' + str(N_steps) + \
                    '_avoid_ovflw_fun_flag_' + str(avoid_ovflw_fun_flag) + \
                    '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                     distribution + 'distr_' + 'dim_distr_par_' + str(dim_dist_par_un) + '.npz'

        ld_est = np.load(file_path, allow_pickle=True)
        phi_ss_est_T, dist_par_un, diag= tens(ld_est["arr_0"]), tens(ld_est["arr_1"]), tens(ld_est["arr_2"])
        #plt.plot(-diag)
        model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=avoid_ovflw_fun_flag, distribution = distribution)
        like_dirSpW1_SS = model.like_seq(Y_T, phi_ss_est_T, dist_par_un)
        all_mod_log_likes.append(like_dirSpW1_SS.item())
        models_info.append(['SS No reg', dim_dist_par_un, distribution, 0, 0, len(diag)])


N_steps = 10000
learn_rate = 0.01

dim_dist_par_un = N
dim_beta= N
for distribution in ['gamma', 'lognormal']:
    file_path = SAVE_FOLD + '/WTN_dirSpW1_X0_dynNet_Single_Snap_est__lr_' + \
                str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                '_N_steps_' + str(N_steps) + \
                '_avoid_ovflw_fun_flag_' + str(avoid_ovflw_fun_flag) + \
                '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                '_dim_beta_' + str(dim_beta) + \
                distribution + 'distr_' + 'dim_distr_par_' + str(dim_dist_par_un) + '.npz'

    ld_est = np.load(file_path, allow_pickle=True)
    phi_ss_est_T, dist_par_un, beta_est, diag= tens(ld_est["arr_0"]), tens(ld_est["arr_1"]),\
                                               tens(ld_est["arr_2"]), tens(ld_est["arr_3"])
   # plt.plot(-diag)
    model = dirSpW1_X0_dynNet_SD(avoid_ovflw_fun_flag=avoid_ovflw_fun_flag, distribution = distribution)
    like_dirSpW1_SS = model.like_seq(Y_T, phi_ss_est_T, dist_par_un, X_T = X_T, beta=beta_est)
    all_mod_log_likes.append(like_dirSpW1_SS.item())
    models_info.append(['SS With reg', dim_dist_par_un, distribution, dim_beta, 0, len(diag)])





# %%Score driven estimates without regressors
all_mod_log_likes = []
models_info = []
rescale_score = False
learn_rate = 0.01
N_BA = N
dim_dist_par_un = N
for distribution in ['gamma', 'lognormal']:
    if distribution == 'gamma':
        N_steps = 7000
    else:
        N_steps = 6000
    model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=avoid_ovflw_fun_flag, rescale_SD=rescale_score, distribution=distribution)

    file_path = SAVE_FOLD + '/WTN_dirSpW1_dynNet_SD_est_test_lr_' + \
            str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
            '_N_steps_' + str(N_steps) + '_N_BA_' + str(N_BA) + \
            '_resc_score_' + str(rescale_score) + '_avoid_ovflw_fun_flag_' + str(avoid_ovflw_fun_flag) + \
            '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
            distribution + 'distr_' + 'dim_distr_par_' + str(dim_dist_par_un) + '.npz'

    ld_est = np.load(file_path, allow_pickle=True)
    W_est, B_est, A_est = tens(ld_est["arr_0"]), tens(ld_est["arr_1"]), tens(ld_est["arr_2"])
    dist_par_un_est = tens(ld_est["arr_3"])
    diag = ld_est["arr_4"]
    plt.plot(diag)
    phi_sd_est_T = model.sd_filt_w(torch.tensor(W_est), torch.tensor(B_est), torch.tensor(A_est),
                                             Y_T, beta=None, X_T=None, dist_par_un=dist_par_un_est)
    like_dirSpW1_SD = model.loglike_sd_filt_w(torch.tensor(W_est), torch.tensor(B_est), torch.tensor(A_est),
                                          Y_T, beta=None, X_T=None, dist_par_un=dist_par_un_est)


    models_info.append(['SD No reg', dim_dist_par_un, distribution, 0, N_BA, len(diag)])
    all_mod_log_likes.append(like_dirSpW1_SD.item())
# %% Score driven estimates with regressors
N_steps = 7000
dim_beta = N
dim_dist_par_un = 1
N_BA = N
for distribution in ['gamma']:
    model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=avoid_ovflw_fun_flag, rescale_SD=rescale_score, distribution=distribution)
    #load also regressors
    file_path = SAVE_FOLD + '/WTN_dirSpW1_X0_dynNet_SD_est_test_lr_' + \
                str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                '_N_steps_' + str(N_steps) + '_N_BA_' + str(N_BA) + \
                '_resc_score_' + str(rescale_score) + '_avoid_ovflw_fun_flag_' + str(avoid_ovflw_fun_flag) + \
                '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                '_dim_beta_' + str(dim_beta) + \
                distribution + 'distr_' + 'dim_distr_par_' + str(dim_dist_par_un) + '.npz'

    ld_est = np.load(file_path, allow_pickle=True)
    W_est, B_est, A_est = tens(ld_est["arr_0"]), tens(ld_est["arr_1"]), tens(ld_est["arr_2"])
    dist_par_un_est = tens(ld_est["arr_3"])
    beta_est = ld_est["arr_4"]
    diag = ld_est["arr_5"]
    plt.plot(diag)
    phi_sd_est_T = model.sd_filt_w(torch.tensor(W_est), torch.tensor(B_est), torch.tensor(A_est),
                                                 Y_T, beta=None, X_T=None, dist_par_un=dist_par_un_est)
    like_dirSpW1_SD = model.loglike_sd_filt_w(torch.tensor(W_est), torch.tensor(B_est), torch.tensor(A_est),
                                              Y_T, beta=None, X_T=None, dist_par_un=dist_par_un_est)

    models_info.append(['SD With reg', dim_dist_par_un, distribution, dim_beta, N_BA, len(diag)])
    all_mod_log_likes.append(like_dirSpW1_SD.item())


all_mod_log_likes
# %%
tmp = np.array(diag)
rel_diff = np.diff(tmp)/tmp[1:]
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

plt.plot(moving_average(rel_diff,500))

# %%
def model_selec_crit(log_like, k, n):
    aic = 2*k - 2*log_like
    aic_c = aic + (2*k**2 + 2*k)/(n-k-1)
    bic = np.log(n)*k - 2*log_like
    return aic, aic_c, bic
all_info_crit = []
all_names = []
N_obs = N*(N-1)*T
for mod_info, like in zip(models_info, all_mod_log_likes):
    N_pars = 0
    if mod_info[4] == 0:
        N_pars += 2*N*T
    else:
        N_pars += 2*N + mod_info[4]

    N_pars += mod_info[3]
    inf_crit = model_selec_crit(like, N_pars, N_obs)
    all_info_crit.append(inf_crit)
    all_names.append(str(mod_info)[2:-1])
all_info_crit = np.array(all_info_crit)

plt.plot(all_names, all_info_crit[:, [0, 2]], '-*')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(['AIC', 'BIC'])
# %%
model_selec_crit(like_dirSpW1_X0_1_SS, 2*N*T +1, N*(N-1)*T)
model_selec_crit(like_dirSpW1_SD_it.detach().numpy(), 2*N*3, N_obs)


# %%
model.check_tot_exp_W_seq(Y_T, torch.tensor(phi_ss_est_T))
ratios_ss_T = model.check_tot_exp_W_seq(Y_T, torch.tensor(phi_ss_est_T))
t=0
Y_t=Y_T[:, :,  t]
A_t = Y_t > 0
E_Y_t = model.cond_exp_Y(tens(phi_ss_est_T[:, t]))
plt.plot(torch.log(Y_t[A_t]),  torch.log(E_Y_t[A_t]), '.')


# %% Approximate checks on the estimates
ratios_sd_T = model.check_tot_exp_W_seq(Y_T, torch.tensor(phi_sd_est_T))
plt.plot(all_y, ratios_ss_T)
plt.plot(all_y, ratios_sd_T)
A_T = Y_T>0
A_T.sum(dim=(0, 1))
dens = torch.tensor(A_T.sum(dim=(0, 1)), dtype=torch.float32)/(N*(N-1))
dens_scale=dens/dens.mean()
plt.plot(all_y,  dens_scale+(ratios_ss_T[0]- dens_scale[0]))


# %%
W = torch.tensor(W_est)
B = torch.tensor(B_est)
A = torch.tensor(A_est)
phi_est_sd_2 = model.sd_filt_w(W, B, A, Y_T, beta=None, X_T = None, alpha=torch.ones(1))



model.update_dynw_par(Y_T[:,:,0], phi_est_sd_2[:, 0], model.identify(W), B, A)
tmp = model.identify(phi_est_sd_2[:, 0])


# %% Comparison SS Binary versions
from dirBin1_dynNets import dirBin1_dynNet_SD, dirBin1_X0_dynNet_SD
A_T = tens(Y_T>0)
dim_delta = 1
learn_rate_theta = 0.1
learn_rate_delta = 0.005
N_steps = 10000
file_path = SAVE_FOLD + '/WTN_dirBin1_X0_dynNet_Single_Snap_est__lr_th' + \
            str(learn_rate_theta) + '_lt_de_' + str(learn_rate_delta) + '_N_' + str(N) + '_T_' + str(T) + \
            '_N_steps_' + str(N_steps) + \
            '_dim_delta_' + str(dim_delta) + '.npz'
ld_est = np.load(file_path, allow_pickle=True)
plt.close('all')
theta_X0_ss_est_T, delta_est, diag = tens(ld_est["arr_0"]), tens(ld_est["arr_1"]), \
                                            tens(ld_est["arr_2"])
plt.plot(-diag)

# versions no regressors
file_path = SAVE_FOLD + '/WTN_dirBin1_dynNet_Single_Snap_est__lr_th' + \
            str(0.5) + '_N_' + str(N) + '_T_' + str(T) + \
            '_N_steps_' + str(100) + '.npz'
ld_est = np.load(file_path, allow_pickle=True)

theta_ss_est_T, diag = tens(ld_est["arr_0"]), ld_est["arr_1"]

dirBin1_dynNet_SD().like_seq(A_T, theta_ss_est_T)
dirBin1_X0_dynNet_SD().like_seq(A_T, theta_X0_ss_est_T, X_T=X_T, delta=delta_est)

# %% Comparison SD Binary versions
A_T = tens(Y_T>0)
N_BA = N
dim_delta = N
learn_rate = 0.01
unit_measure = 1e12
N_steps = 6000
file_path = SAVE_FOLD + '/WTN_dirBin1_X0_dynNet_SD_est_test_lr_' + \
            str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
            '_N_steps_' + str(N_steps) + '_N_BA_' + str(N_BA) + \
            '_dim_delta_' + str(dim_delta) + \
            '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + '.npz'
ld_est = np.load(file_path, allow_pickle=True)

W_est, B_est, A_est, delta_est, diag = tens(ld_est["arr_0"]), tens(ld_est["arr_1"]), \
                                            tens(ld_est["arr_2"]), tens(ld_est["arr_3"]), tens(ld_est["arr_4"])


# Comparison Binary versions no regressors
file_path = SAVE_FOLD + '/WTN_dirBin1_dynNet_SD_est_test_lr_' + \
            str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
            '_N_steps_' + str(N_steps) + '_N_BA_' + str(N_BA) + \
            '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + '.npz'
ld_est = np.load(file_path, allow_pickle=True)

W_est, B_est, A_est,  diag = tens(ld_est["arr_0"]), tens(ld_est["arr_1"]), \
                                            tens(ld_est["arr_2"]), tens(ld_est["arr_3"])

plt.plot(-diag)
































#