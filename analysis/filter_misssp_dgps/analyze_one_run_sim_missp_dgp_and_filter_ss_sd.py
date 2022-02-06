# %%
from pathlib import Path
import importlib
import dynwgraphs
from dynwgraphs.utils.tensortools import splitVec, strIO_from_tens_T
import mlflow
import logging
import proj_utils
from proj_utils.mlflow import _get_and_set_experiment, check_test_exp, get_df_exp, uri_to_path
from utils_missp_sim import load_all_models_missp_sim
from proj_utils import drop_keys, pd_filt_on
from mlflow.tracking.client import MlflowClient
import pandas as pd
from run_sim_missp_dgp_and_filter_ss_sd import get_filt_mod, filt_err
import torch
import numpy as np
from matplotlib import pyplot as plt
from proj_utils.mlflow import dicts_from_run

logger = logging.getLogger(__name__)
importlib.reload(dynwgraphs)
importlib.reload(proj_utils)

#%%

# import pickle
# file_name = mod_dgp.file_names(load_path / "dgp")["parameters"]
        
# par_dict = pickle.load(open(file_name, "rb"))

# par_dict["phi_T"]
# mod_dgp.set_par_val_from_dict(par_dict)




# %%

experiment = _get_and_set_experiment("Default")

all_runs = get_df_exp(experiment, one_df=True)

run_id = "fb5a52151c38481caec5dba3f8317fd9"

row_run = all_runs[all_runs.run_id == run_id].iloc[0]


# %%
mod_filt_sd_bin, mod_filt_sd_w, mod_filt_ss_bin, mod_filt_ss_w, mod_dgp_bin, mod_dgp_bin, mod_dgp_w, obs, Y_reference = load_all_models_missp_sim(row_run)


Y_reference["Y_reference_w"].sum()
Y_reference["Y_reference_bin"].sum()


mod_dgp_w.beta_T
mod_dgp_w.sample_Y_T(A_T = mod_dgp_w.Y_T>0)

loss = mod_filt_sd_w.loglike_seq_T()


# mod_filt_w_sd.estimate()
#%%
mod_filt = mod_filt_bin_sd
mod_dgp = mod_dgp_bin


phi_T, dim_dist_par_un_T, beta_T = mod_filt.get_time_series_latent_par()

phi_T, dim_dist_par_un_T, beta_T = mod_dgp.get_time_series_latent_par()
beta_T.shape
list(range(1))

mod_filt.roll_sd_filt_train()
mod_filt.identify_sequences_tv_par()

mod_filt.T_train
mod_dgp.T_train
mod_filt.plot_phi_T()
mod_dgp.plot_phi_T()


mod_filt.phi_tv
mod_dgp.phi_tv


strIO_from_tens_T(Y_T)

mod_filt.inds_never_obs_w
mod_dgp.inds_never_obs_w

phi_to_exclude = mod_dgp.get_inds_inactive_nodes() 

phi_T, _, _= mod_filt.get_time_series_latent_par() 
plt.plot(phi_T[phi_to_exclude, :].T)
plt.plot(phi_T[~phi_to_exclude, :].T)


phi_T, _, _ = mod_dgp.get_time_series_latent_par() 
plt.plot(phi_T[phi_to_exclude, :].T)

filt_err(mod_dgp, mod_filt)
#%%

i = torch.where(~splitVec(phi_to_exclude)[0])[0][0]
x_T = X_T[0, 0, 0, :].numpy()

fig_ax = plt.subplots(2, 1)
mod_filt.plot_phi_T(i=i, fig_ax=mod_dgp.plot_phi_T(i=i, fig_ax=fig_ax))


# %%
t0 = 0
i=4
phi_T, _, _ = mod_filt.get_time_series_latent_par()
phi_i_T, phi_o_T = splitVec(phi_T)
data_i = phi_i_T[i, t0:].reshape(-1,1)
data_i_2 = phi_i_T[i+1, t0:].reshape(-1,1)
data_o = phi_o_T[i, t0:].reshape(-1,1)
data_o_2 = phi_o_T[i+1, t0:].reshape(-1,1)
x = x_T[t0:]
inds = x > - np.inf 
plt.scatter(x[inds], data_i[inds])
plt.figure()
plt.scatter(data_i, data_i_2)

print(f"{run_ind}")
plt.figure()
corr_i = [np.corrcoef(x_T, phi_i_T[i, :])[0,1] for i in range(int(row_run["n_nodes"]))]
corr_o = [np.corrcoef(x_T, phi_o_T[i, :])[0,1] for i in range(int(row_run["n_nodes"]))]
plt.hist(corr_i, alpha=0.5)
plt.hist(corr_o, alpha=0.5)

plt.figure()
beta_i, beta_o = splitVec(mod_dgp.beta_T[0])
plt.scatter(beta_i, corr_i)
plt.scatter(beta_o, corr_o)


# %%
plt.figure()
plt.scatter(mod_dgp.beta_T[0].detach(), mod_filt.beta_T[0].detach())

mod_dgp.beta_T


# %%
plt.figure()
plt.hist([np.cov(x_T, phi_i_T[i, :])[0,1] for i in range(int(row_run["n_nodes"]))], alpha=0.5)
plt.hist([np.cov(x_T, phi_o_T[i, :])[0,1] for i in range(int(row_run["n_nodes"]))], alpha=0.5)


# %%

mod_dgp.beta_T
mod_filt.beta_T

plt.plot(phi_i_T[0, :])

plt.plot(phi_i_T[0,:])
corr = np.corrcoef(np.concatenate((x_T.T, phi_T)))
corr = np.corrcoef())

tmp = np.concatenate((phi_T[14,:].unsqueeze(dim=1).T, phi_T))

np.corrcoef(tmp)[0,14]

corr[11,0]


# %%



