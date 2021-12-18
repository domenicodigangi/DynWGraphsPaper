"""
Estimate directed sparse weighted model with distances as regressors on WTN data
"""

import sys
import numpy as np
import torch
sys.path.append("./src/")
import matplotlib.pyplot as plt
import seaborn as sns
from utils import tens
import scipy
#load yearly world Trade networks
ld_data = np.load("./data/world_trade_network/world_trade_net_T.npz",
                  allow_pickle=True)
wtn_T, all_y, nodes= ld_data["wtn_T"], ld_data["all_y"], ld_data["nodes"]
dist_T, scaling_infl = ld_data["dist_T"], ld_data["scaling_infl"]

# Set unit measure and  rescale for inflation
from dirSpW1_dynNets import dirSpW1_dynNet_SD
#% load and look at single snapshot estimates No regressors and fixed distr_par
unit_measure = 1e6# [1e6, 1e9, 1e12]:
Y_T = tens(wtn_T * scaling_infl[:, 1])/unit_measure
X_T = tens(dist_T/dist_T.mean())
X_T_new = tens((dist_T - dist_T.mean()) / dist_T.std())

N = Y_T.shape[0]
T = Y_T.shape[2]

SAVE_FOLD = './data/estimates_real_data/WTN'

# %% SS weighted no reg
N_steps = 10000
learn_rate = 0.01
avoid_ovflw_fun_flag = True
all_mod_log_likes = []
models_info = []
all_mod_phi_T = []
all_mod = []
all_mod_beta = []
for dim_dist_par_un in [N]:
    for distribution in ['gamma', 'lognormal']:
        file_path = SAVE_FOLD + '/WTN_dirSpW1_dynNet_Single_Snap_est__lr_' + \
                    str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
                    '_N_steps_' + str(N_steps) + \
                    '_avoid_ovflw_fun_flag_' + str(avoid_ovflw_fun_flag) + \
                    '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
                     distribution + 'distr_' + 'dim_distr_par_' + str(dim_dist_par_un) + '.npz'

        ld_est = np.load(file_path, allow_pickle=True)
        phi_est_T, dist_par_un, diag= tens(ld_est["arr_0"]), tens(ld_est["arr_1"]), tens(ld_est["arr_2"])
        #plt.plot(-diag)
        model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=avoid_ovflw_fun_flag, distribution = distribution)
        like_dirSpW1_SS = model.like_seq(Y_T, phi_est_T, dist_par_un)
        all_mod_log_likes.append(like_dirSpW1_SS.item())
        models_info.append(['SS No reg', dim_dist_par_un, distribution, 0, 0, len(diag)])
        all_mod_phi_T.append(phi_est_T)
        all_mod.append(model)
        all_mod_beta.append(None)

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
    phi_est_T, dist_par_un, beta_est, diag= tens(ld_est["arr_0"]), tens(ld_est["arr_1"]),\
                                               tens(ld_est["arr_2"]), tens(ld_est["arr_3"])
   # plt.plot(-diag)
    model = dirSpW1_X0_dynNet_SD(avoid_ovflw_fun_flag=avoid_ovflw_fun_flag, distribution = distribution)
    like_dirSpW1_SS = model.like_seq(Y_T, phi_est_T, dist_par_un, X_T = X_T, beta=beta_est)
    all_mod_log_likes.append(like_dirSpW1_SS.item())
    models_info.append(['SS With reg', dim_dist_par_un, distribution, dim_beta, 0, len(diag)])
    all_mod_phi_T.append(phi_est_T)
    all_mod.append(model)
    all_mod_beta.append(beta_est)

# %% general data analysis

import statsmodels.api as sm

plt.close('all')
x = Y_T[Y_T>0]
sm.qqplot(x.log(), line='45')
plt.title('log of data')




def gof_measures_w(obs, pred):
    return ((obs - pred)**2).mean().sqrt(), ((obs.log() - pred.log())**2).mean().sqrt()

def eval_w_forecast(Y_T, phi_T, T_train, A_T=None):
    T_test = T- T_train
    obs_T_test = torch.zeros(0)
    pred_T_test = torch.zeros(0)
    for t in range(T_test-1):
        t_pres = T_train + t
        t_fut= t_pres+1
        Y_fut =  Y_T[:, :, t_fut]
        if A_T is None:
            A_fut = Y_fut>0
        else:
            A_fut = A_T[:, :, t_fut]

        #assume evaluation on exact binary matrix for the moment
        A_for = A_fut
        #parameters used to forecast matrix. must depend on present data
        phi_for = phi_T[:, t_pres]
        Y_for = dirSpW1_dynNet_SD().cond_exp_Y(phi_for)

        obs_T_test = torch.cat((obs_T_test, Y_fut[A_for]), dim=0)
        pred_T_test = torch.cat((pred_T_test, Y_for[A_for]), dim=0)
    return gof_measures_w(obs_T_test, pred_T_test)

def ecdf(x, log=True):
    if log:
        x = torch.sort(x[x>0].view(-1))[0]
        p = tens(1. * np.arange(x.shape[0]) / (x.shape[0] - 1))
        plt.semilogx(x, p)
    else:
        x = torch.sort(x.view(-1))[0]
        p = tens(1. * np.arange(x.shape[0]) / (x.shape[0] - 1))
        plt.plot(x, p)
# %%

T_test = 10
T_train = T-T_test
plt.close('all')
fig, axs = plt.subplots(4, 2, figsize=(10, 15), facecolor='w', edgecolor='k')
for m_plot in range(4):
    m=m_plot #+2
    phi_T = all_mod_phi_T[m]
    model = all_mod[m]
    beta = all_mod_beta[m]
    eval_w_forecast(Y_T, phi_est_T, T-10)
    res_T = (Y_T[:, :, :T_train] - model.exp_W_seq(Y_T[:, :, :T_train], phi_T[ :, :T_train], X_T=X_T[:, :, :T_train]
                                                   , beta=beta))[Y_T[:, :, :T_train]>0]
    rel_res_T = ((Y_T - model.exp_W_seq(Y_T, phi_T, X_T=X_T, beta=beta))/Y_T)[Y_T>0]
    log_E_Y_T = torch.log10(model.exp_W_seq(Y_T[:, :, :T_train], phi_T[:, :T_train], X_T=X_T[:, :, :T_train], beta=beta))[Y_T[:, :, :T_train]>0]
    if True:
        sm.qqplot(log_res_T, line='45', fit=True, ax = axs[m,0])
        axs[m_plot,0].set_title( str(models_info[m][0:3])[2:-1] + 'constant sigma par')
        sm.qqplot(log_res_T, line='45', fit=False, ax = axs[m,1])
        axs[m_plot, 1].set_title( str(models_info[m][0:3])[2:-1] + 'constant sigma par')
    else:
        inds = Y_T[:, :, :T_train]>0
        x = Y_T[:, :, :T_train][inds].log10()
        y= log_E_Y_T
        axs[m_plot].plot(x, y, '.')
        axs[m_plot].plot(x, x, '.r')
        axs[m_plot].set_title( str(models_info[m][0:3])[2:-1] + 'constant sigma par')
        axs[m_plot].set_xlabel('log10 Y')
        axs[m_plot].set_ylabel('log10 E[Y]')




# %%

#plt.close()
ecdf(log_res_T.abs(), log=False)



ecdf_log(Y_T)

(Y_T - Y_T.mean(dim=(0, 1))).abs().sum()
res_T.abs().sum()
Y_T.abs().sum()

# %%
file_path = SAVE_FOLD + '/WTN_dirSpW1_dynNet_SD_est_lr_' + \
            str(learn_rate) + '_N_' + str(N) + '_T_' + str(T) + \
            '_N_steps_' + str(N_steps_max) + '_N_BA_' + str(N_BA) + \
            '_resc_score_' + str(rescale_score) + '_avoid_ovflw_fun_flag_' + str(avoid_ovflw_fun_flag) + \
            '_unit_' + '10e' + str(np.int(np.log10(unit_measure))) + \
            distribution + '_distr_' + '_dim_distr_par_' + str(dim_dist_par_un) + \
            'test_sam_last_' + str(T_test) + '.npz'

#