"""
Given data Estimate multiple directed binary models in parallel
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


sys.path.append("./src/")
from utils import splitVec, tens

#load yearly world Trade networks
ld_data = np.load("./data/world_trade_network/world_trade_net_T.npz",
                  allow_pickle=True)
wtn_T, all_y, nodes= ld_data["wtn_T"], ld_data["all_y"], ld_data["nodes"]
dist_T, scaling_infl = ld_data["dist_T"], ld_data["scaling_infl"]

fold_name = 'WTN'
SAVE_FOLD = './data/estimates_real_data/' + fold_name

from dirSpW1_dynNets import dirSpW1_dynNet_SD, load_dirSpW1_models

unit_measure = 1e6


Y_T = tens(wtn_T * scaling_infl[:, 1])/1e6
X_T = tens(dist_T).log().unsqueeze(2)#torch.tensor((dist_T - dist_T.mean())/dist_T.std()).unsqueeze(2)
T = Y_T.shape[2]
N = Y_T.shape[0]
N_steps = 20000

T_test = 10
T_train = T - T_test

list_all_models = [ ['SS', 'gamma',     N, False, 0], #0
                    ['SS', 'lognormal', N, False, 0], #1
                    ['SS', 'gamma',     N, True, 1], #2
                    ['SS', 'lognormal', N, True, 1], #3
                    ['SS', 'gamma',     N, True, N], #4
                    ['SS', 'lognormal', N, True, N], # 5
                    ['SD', 'gamma',     N, False, 0], #6
                    ['SD', 'lognormal', N, False, 0], #7
                    ['SD', 'gamma',     N, True, 1], #8
                    ['SD', 'lognormal', N, True, 1], #9
                    ['SD', 'gamma',     N, True, N], #10
                    ['SD', 'lognormal', N, True, N]] #11

all_names = []
for mod_info in list_all_models:
    all_names.append(mod_info[0] + ' ' + mod_info[1] + ' ' + str(mod_info[4]) )
inds_gamma = [  mod[1]=='gamma' for mod in list_all_models]
inds_logn = [not  mod[1]=='gamma' for mod in list_all_models]


#%% load all estimates
all_par_SS = []
n_pars = []
for filter_type, distribution, dim_dist_par, regr_flag, dim_beta in list_all_models[:6]:
    mod_par = load_dirSpW1_models(N, T, distribution, dim_dist_par, filter_type, regr_flag, SAVE_FOLD,
                                dim_beta=dim_beta, n_beta_tv=0, unit_measure=unit_measure,
                                learn_rate=0.01, T_test=T_test,
                                N_steps=N_steps, avoid_ovflw_fun_flag=True, rescale_score=False,
                                return_last_diag=False)
    all_par_SS.append(mod_par)
    n_reg = 0
    if regr_flag:
        n_reg = X_T.shape[2]
    n_pars.append(T_train*2*N + dim_dist_par + dim_beta * n_reg)

all_par_SD = []
for filter_type, distribution, dim_dist_par, regr_flag, dim_beta in list_all_models[6:]:
    mod_par = load_dirSpW1_models(N, T, distribution, dim_dist_par, filter_type, regr_flag, SAVE_FOLD,
                                dim_beta=dim_beta, n_beta_tv=0, unit_measure=1e6,
                                learn_rate=0.01, T_test=T_test,
                                N_steps=N_steps, avoid_ovflw_fun_flag=True, rescale_score=False,
                                return_last_diag=False)
    all_par_SD.append(mod_par)
    n_reg = 0
    if regr_flag:
        n_reg = X_T.shape[2]
    n_pars.append(3*2*N + dim_dist_par + dim_beta * n_reg)

all_log_l = []
for m in range(len(all_par_SS)):
    model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True, distribution=list_all_models[:len(all_par_SS)][m][1], rescale_SD=False)
    phi_ss_est_T, dist_par_un, beta_est, \
    diag, N_steps, N_steps_iter, \
    unit_measure, learn_rate  = all_par_SS[m]
    if beta_est is not None:
        log_l = model.like_seq(tens(Y_T[:, :, :T_train]), tens(phi_ss_est_T[:, :T_train]),
                               tens(dist_par_un), X_T=X_T[:, :, :, :T_train], beta=tens(beta_est))
    else:
        log_l = model.like_seq(tens(Y_T[:, :, :T_train]), tens(phi_ss_est_T[:, :T_train]), tens(dist_par_un),
                               X_T=None, beta=None)
    all_log_l.append(log_l.item())

for m in range(len(all_par_SD)):
    model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True, distribution=list_all_models[len(all_par_SS):][m][1], rescale_SD=False)
    W_est, B_est, A_est, dist_par_un_est, \
    beta_est, diag, N_steps, \
    unit_measure, learn_rate = all_par_SD[m]
    if beta_est is not None:
        log_l = model.loglike_sd_filt(tens(W_est), tens(B_est), tens(A_est), Y_T[:, :, :T_train],
                                      beta_const=tens(beta_est), X_T=X_T[:, :, :T_train],
                                      dist_par_un=tens(dist_par_un_est))
    else:
        log_l = model.loglike_sd_filt(tens(W_est), tens(B_est), tens(A_est), Y_T[:, :, :T_train],
                                      beta_const=None, X_T=None, dist_par_un=tens(dist_par_un_est))
    all_log_l.append(log_l.item())

#%%
from scipy.stats import chi2
#from utils import likel_ratio_p_val
def likel_ratio_p_val(logl_0, logl_1, df):
    G = 2 * (logl_1 - logl_0)
    p_value = chi2.sf(G, df)
    return p_value
ind_0, ind_1 = 3, 5
logl_0 = all_log_l[ind_0]
logl_1 = all_log_l[ind_1]
df = n_pars[ind_1] - n_pars[ind_0]
G = 2 * (logl_1 - logl_0)
p_value = chi2.sf(G, df)
print(list_all_models[ind_0], list_all_models[ind_1])
print(G, chi2.mean(df), df, p_value)

#%% AIC and BIC
def model_selec_crit(log_like, k, n):
    aic = 2*k - 2*log_like
    aic_c = aic + (2*k**2 + 2*k)/(n-k-1)
    bic = np.log(n)*k - 2*log_like
    return aic, aic_c, bic


aic_all =[]
aic_c_all =[]
bic_all =[]
for m in range(len(list_all_models)):
    aic, aic_c, bic = model_selec_crit(all_log_l[m], n_pars[m], T_train*N)
    aic_all.append(aic)
    aic_c_all.append(aic_c)
    bic_all.append(bic)
aic_all = np.array(aic_all)
aic_c_all = np.array(aic_c_all)
bic_all = np.array(bic_all)



plt.close()
plt.subplot(1, 2, 1)
plt.plot(np.array(all_names)[inds_gamma], np.array(aic_all)[inds_gamma], '-*')
plt.plot(np.array(all_names)[inds_gamma], np.array(bic_all)[inds_gamma], '-*')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(['AIC', 'BIC'])
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(np.array(all_names)[inds_logn], np.array(aic_all)[inds_logn], '-*')
plt.plot(np.array(all_names)[inds_logn], np.array(bic_all)[inds_logn], '-*')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(['AIC', 'BIC'])
plt.grid()

#%%


#%%

























#


























