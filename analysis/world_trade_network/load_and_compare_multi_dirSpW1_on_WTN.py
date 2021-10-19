"""
Given data Estimate multiple directed binary models in parallel
"""

import sys
import numpy as np
import torch
sys.path.append("./src/")
from utils import splitVec, tens
import matplotlib.pyplot as plt


#load yearly world Trade networks
ld_data = np.load("./data/world_trade_network/world_trade_net_T.npz",
                  allow_pickle=True)
wtn_T, all_y, nodes= ld_data["wtn_T"], ld_data["all_y"], ld_data["nodes"]
dist_T, scaling_infl = ld_data["dist_T"], ld_data["scaling_infl"]

from dirSpW1_dynNets import dirSpW1_dynNet_SD, load_dirSpW1_models


fold_name = 'WTN'
SAVE_FOLD_no_reg = './data/estimates_real_data/' + fold_name

reg_name = 'distances'

SAVE_FOLD = SAVE_FOLD_no_reg + '/' +  reg_name

unit_measure = 1e6


Y_T_all = tens(wtn_T * scaling_infl[:, 1])
Y_T = Y_T_all[:, :, 1:]
T = Y_T.shape[2]
N = Y_T.shape[0]

if reg_name == 'distances':
    X_T = tens(dist_T[:, :, 1:]).log().unsqueeze(2)#torch.tensor((dist_T - dist_T.mean())/dist_T.std()).unsqueeze(2)
elif reg_name == 'prev_link_t':
    X_T = Y_T_all[:, :, :-1].log().unsqueeze(2)

T_test = 10
T_train = T - T_test
rescale_score = False

list_all_models = []
list_all_models.append(['SS', 'gamma', N, False, 0, 20000, 0.005, SAVE_FOLD_no_reg])
# list_all_models.append(['SS', 'lognormal', N, False, 0, 20000, 0.005, SAVE_FOLD_no_reg])
# list_all_models.append(['SS', 'gamma', N, True, 1, 20000, 0.005, SAVE_FOLD])
# list_all_models.append(['SS', 'lognormal', N, True, 1, 20000, 0.005, SAVE_FOLD])
# list_all_models.append(['SS', 'gamma', N, True, N, 20000, 0.005, SAVE_FOLD])
# list_all_models.append(['SS', 'lognormal', N, True, N, 20000, 0.005, SAVE_FOLD])

N_ss_mod = len(list_all_models)

list_all_models.append(['SD', 'gamma', N, False, 0, 20000, 0.005, SAVE_FOLD_no_reg])
# list_all_models.append(['SD', 'lognormal', N, False, 0, 20000, 0.005, SAVE_FOLD_no_reg])
# list_all_models.append(['SD', 'gamma', N, True, 1, 50000, 0.0001, SAVE_FOLD])
# list_all_models.append(['SD', 'lognormal', N, True, 1, 50000, 0.0001, SAVE_FOLD])
# list_all_models.append(['SD', 'gamma', N, True, N, 50000, 0.0001, SAVE_FOLD])
# list_all_models.append(['SD', 'lognormal', N, True, N, 50000, 0.0001, SAVE_FOLD])
#
N_sd_mod = len(list_all_models) - N_ss_mod

all_names = []
for mod_info in list_all_models:
    all_names.append(mod_info[0] + ' ' + mod_info[1] + ' ' + str(mod_info[4]) )
inds_gamma = [mod[1] == 'gamma' for mod in list_all_models]
inds_logn = [not  mod[1] == 'gamma' for mod in list_all_models]

# %% load all estimates
all_par_SS = []
n_pars = []
for filter_type, distribution, dim_dist_par, regr_flag, dim_beta, N_steps, learn_rate, SAVE_FOLD in \
        list_all_models[:N_ss_mod]:
    mod_par = load_dirSpW1_models(N, T, distribution, dim_dist_par, filter_type, regr_flag, SAVE_FOLD,
                                dim_beta=dim_beta, n_beta_tv=0, unit_measure=unit_measure,
                                learn_rate=learn_rate, T_test=T_test,
                                N_steps=N_steps, avoid_ovflw_fun_flag=True, rescale_score=rescale_score,
                                return_last_diag=False)
    all_par_SS.append(mod_par)
    n_reg = 0
    if regr_flag:
        n_reg = X_T.shape[2]
    n_pars.append(T_train*2*N + dim_dist_par + dim_beta * n_reg)

all_par_SD = []
for filter_type, distribution, dim_dist_par, regr_flag, dim_beta, N_steps, learn_rate, SAVE_FOLD in \
        list_all_models[N_ss_mod:]:
    mod_par = load_dirSpW1_models(N, T, distribution, dim_dist_par, filter_type, regr_flag, SAVE_FOLD,
                                dim_beta=dim_beta, n_beta_tv=0, unit_measure=1e6,
                                learn_rate=learn_rate, T_test=T_test,
                                N_steps=N_steps, avoid_ovflw_fun_flag=True, rescale_score=rescale_score,
                                return_last_diag=False)
    all_par_SD.append(mod_par)
    n_reg = 0
    if regr_flag:
        n_reg = X_T.shape[2]
    n_pars.append(3*2*N + dim_dist_par + dim_beta * n_reg)
# %% Filter all
all_exp_Y = []
all_phi_T = []
for m in range(len(all_par_SS)):
    model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True, distribution=list_all_models[:N_ss_mod][m][1], rescale_SD=False)
    phi_ss_est_T, dist_par_un, beta_est, \
    diag, N_steps, N_steps_iter, \
    unit_measure, learn_rate  = all_par_SS[m]
    if beta_est is not None:
        for_Y_T = model.exp_seq(tens(phi_ss_est_T), X_T=X_T, beta_const=tens(beta_est))
    else:
        for_Y_T = model.exp_seq(tens(phi_ss_est_T), X_T=None, beta_const=None)
    for_Y_T = torch.cat((torch.zeros(N, N, 1), for_Y_T[:, :, :-1]), dim=2)
    all_exp_Y.append(for_Y_T)
    all_phi_T.append(phi_ss_est_T)
for m in range(len(all_par_SD)):
    model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True, distribution=list_all_models[N_ss_mod:][m][1], rescale_SD=False)
    W_est, B_est, A_est, dist_par_un_est, \
    beta_est, sd_par_0, diag, N_steps, \
    unit_measure, learn_rate = all_par_SD[m]
    if beta_est is not None:
        phi_sd_est_T, beta_sd_T = model.sd_filt(tens(W_est), tens(B_est), tens(A_est), Y_T, beta_const=tens(beta_est),
                                                sd_par_0=tens(sd_par_0),
                                                X_T=X_T, dist_par_un=tens(dist_par_un_est))
        for_Y_T = model.exp_seq(tens(phi_sd_est_T), X_T=X_T, beta_const=tens(beta_est))
    else:
        model.rescale_SD = True
        phi_sd_est_T, beta_sd_T = model.sd_filt(tens(W_est), tens(B_est), tens(A_est), Y_T, beta_const=None,
                                                sd_par_0=tens(sd_par_0),
                                                X_T=None, dist_par_un=tens(dist_par_un_est))

        for_Y_T = model.exp_seq(tens(phi_sd_est_T), X_T=None, beta_const=None)
    all_exp_Y.append(for_Y_T)
    all_phi_T.append(phi_sd_est_T)


# %% Compute Gof measures in and out of sample
def gof_measures_w(obs, pred, n_rem=1):
    diff = (obs - pred).sort()[0]
    diff = diff[n_rem:-n_rem]
    log_diff = (obs.log() - pred.log()).sort()[0]
    log_diff = log_diff[n_rem:-n_rem]

    return ((diff)**2).mean().sqrt(), ((log_diff**2)).mean().sqrt()

Y_T_test = Y_T[:, :, T_train:-1]
Y_T_train= Y_T[:, :, 1:T_train]
nnz_inds_test = Y_T_test > 0
nnz_inds_train = Y_T_train > 0
obs_w_test = Y_T_test[nnz_inds_test]
obs_w_train = Y_T_train[nnz_inds_train]
gof_test = []
gof_train = []
for for_Y_T in all_exp_Y:
    for_Y_T_test = for_Y_T[:, :, T_train:-1]
    for_Y_T_train = for_Y_T[:, :, 1:T_train]
    fore_w_test = for_Y_T_test[nnz_inds_test]
    fore_w_train = for_Y_T_train[nnz_inds_train]
    gof_test.append(gof_measures_w(obs_w_test, fore_w_test) )
    gof_train.append(gof_measures_w(obs_w_train, fore_w_train) )
gof_test = np.array(gof_test)
gof_train = np.array(gof_train)

mod_x_labels = [str(list_all_models[i][0]) + ' ' + str(list_all_models[i][4]) for i in range(len(list_all_models))]
max_Y = np.array([Y.max().log10().round().item() for Y in all_exp_Y] )
max_Y_train = np.array([Y[:, :, :T_train].max().log10().round().item() for Y in all_exp_Y] )

# %%
print(gof_test[inds_gamma, 1])
print(max_Y[inds_gamma])
print(gof_test[inds_logn, 1])
print(max_Y[inds_logn])

plt.close()
plt.figure(figsize=(15, 8))
plt.plot(np.array(mod_x_labels)[inds_gamma], gof_train[inds_gamma, 1], '-*')
plt.plot(np.array(mod_x_labels)[inds_logn], gof_train[inds_logn, 1], '-*')
plt.plot(np.array(mod_x_labels)[inds_gamma], gof_test[inds_gamma, 1], '-*')
plt.plot(np.array(mod_x_labels)[inds_logn], gof_test[inds_logn, 1], '-*')
plt.xticks(rotation=45)
plt.tight_layout()
plt.title('Mean Square Error of Log Weights for Present Links', fontsize=18)
plt.legend(['Gamma In Sample', 'Lognormal In Sample', 'Gamma Out of Sample', 'Lognormal Out of Sample'])
plt.grid()
plt.tight_layout()

# %%
plt.close()
plt.plot(all_par_SD[2][6])
all_par_SD[2][1]
all_par_SD[3][4]
all_par_SS[2][2]


# %%
m=0
ind_pl = 55
plt.close('all')
plt.figure()
plt.plot(all_phi_T[m].transpose()[:, ind_pl])
# plt.plot(all_phi_T[m+6].detach().numpy().transpose()[:, ind_pl])
# plt.plot(all_phi_T[m+1].transpose()[:, ind_pl])
plt.plot(all_phi_T[m+1].detach().numpy().transpose()[:, ind_pl])
plt.legend(['SS-GAMMA', 'SD-GAMMA', 'SS-LogN', 'SD-LogN'])
s_i_T = Y_T[ind_pl, :, :].sum(dim=0)
#plt.plot(s_i_T/s_i_T.max())
all_par_SD[m][2][ind_pl]

all_par_SD[m][2].max()


# all_par_SD[m+1][2][ind_pl]
# %%

model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=False, distribution=list_all_models[len(all_par_SS):][m][1], rescale_SD=False)
model.backprop_sd = True
phi_t = tens(phi_ss_est_T[:, 0])

Y_t = Y_T[:, :, 0]
plt.close()
plt.hist(model.score_t(Y_t, phi_t)[0]*tens(A_est[:N]))
tens(W_est[:N]) + phi_t[:N] * tens(B_est[:N]) +  model.score_t(Y_t, phi_t)[0]*tens(A_est[:N])
phi_sd_est_T[:N, 0]



plt.close('all')
plt.plot(phi_ss_est_T.transpose(1, 0))
plt.plot(A_est)
plt.plot(A_est, phi_sd_est_T.std(dim=1).detach(), '.')
plt.plot(phi_sd_est_T.transpose(0,1))

# %%
#

# %%
N_plot = 50
plt.close('all')
plt.figure()
# plt.plot(all_phi_T[0].transpose()[:, 0:N_plot])
#plt.figure()
plt.plot(all_phi_T[6].detach().numpy().transpose()[:, 0:N_plot], '--')
# plt.figure()
# plt.plot(all_phi_T[1].transpose()[:, 0:N_plot])
# plt.figure()
# plt.plot(all_phi_T[7].detach().numpy().transpose()[:, 0:N_plot])

# %%
# %%

plt.close()
x = tens(range(-50, 50))/10
#plt.plot(all_y, Y_T.sum(dim=(0,1)).log())
plt.plot(x, 1- ((x).tanh()**2))

# %%





# %%

import statsmodels.api as sm

plt.close('all')
x = Y_T[Y_T>0]
sm.qqplot(x.log(), line='45')
plt.title('log of data')




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
    log_for_Y_T = torch.log10(model.exp_W_seq(Y_T[:, :, :T_train], phi_T[:, :T_train], X_T=X_T[:, :, :T_train], beta=beta))[Y_T[:, :, :T_train]>0]
    if True:
        sm.qqplot(log_res_T, line='45', fit=True, ax = axs[m,0])
        axs[m_plot,0].set_title( str(models_info[m][0:3])[2:-1] + 'constant sigma par')
        sm.qqplot(log_res_T, line='45', fit=False, ax = axs[m,1])
        axs[m_plot, 1].set_title( str(models_info[m][0:3])[2:-1] + 'constant sigma par')
    else:
        inds = Y_T[:, :, :T_train]>0
        x = Y_T[:, :, :T_train][inds].log10()
        y= log_for_Y_T
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























#


























