# %%
from pathlib import Path
import importlib
import dynwgraphs
from dynwgraphs.utils.tensortools import splitVec, strIO_from_tens_T
from dynwgraphs.dirGraphs1_dynNets import get_model_from_run_dict
from dynwgraphs.utils.dgps import get_dgp_mod_and_par
import mlflow
import logging
import ddg_utils
from ddg_utils.mlflow import _get_and_set_experiment, check_test_exp, get_df_exp, uri_to_path
from ddg_utils import drop_keys, pd_filt_on
from mlflow.tracking.client import MlflowClient
import pandas as pd
from run_sim_missp_dgp_and_filter_ss_sd import get_filt_mod, filt_err
import torch
import numpy as np
from matplotlib import pyplot as plt
from ddg_utils.mlflow import dicts_from_run

logger = logging.getLogger(__name__)
importlib.reload(dynwgraphs)
importlib.reload(ddg_utils)

#%%
def load_all_models(row_run, bin_or_w):
    load_path = Path(uri_to_path(row_run["artifact_uri"]))

    Y_T, X_T = torch.load(open(load_path / "dgp" / "obs_T_dgp.pt", "rb"))

    mod_dgp = get_model_from_run_dict("dgp", bin_or_w, Y_T, X_T, row_run)

    mod_dgp.load_par(str(load_path / "dgp")) 
    
    mod_filt_ss = get_model_from_run_dict("filt_ss", bin_or_w, Y_T, X_T, row_run)
    mod_filt_ss.load_par(str(load_path)) 
    
    mod_filt_sd = get_model_from_run_dict("filt_sd", bin_or_w, Y_T, X_T, row_run)
    mod_filt_sd.load_par(str(load_path)) 
    mod_filt_sd.roll_sd_filt_train()


    return mod_dgp, mod_filt_ss, mod_filt_sd, Y_T, X_T


import pickle
file_name = mod_dgp.file_names(load_path / "dgp")["parameters"]
        
par_dict = pickle.load(open(file_name, "rb"))

par_dict["phi_T"]
mod_dgp.set_par_val_from_dict(par_dict)




# %%

experiment = _get_and_set_experiment("sim missp filter")
experiment = _get_and_set_experiment("test")
all_runs = get_df_exp(experiment, one_df=True)

run_id = "4c61d2252b474d058c2916bb07b323c8"
row_run = all_runs[all_runs.run_id == run_id].iloc[0]



# %%
mod_dgp_bin, mod_filt_bin_ss, mod_filt_bin_sd, Y_T, X_T = load_all_models(row_run, "bin")
mod_dgp_w, mod_filt_w_ss, mod_filt_w_sd, _, _ = load_all_models(row_run, "w")


# mod_filt_w_sd.estimate()
# %%
# To Do: 
# check that phi filter and mse are now better

mod_filt = mod_filt_w_sd
mod_dgp = mod_dgp_w

mod_filt.beta_tv[0]
mod_filt.init_sd_type
mod_filt.inds_to_exclude_from_id
mod_filt.get_unc_mean(mod_filt.sd_stat_par_un_phi)

mod_dgp.identify_sequences()
mod_dgp.plot_phi_T()
phi_T[~mod_dgp.inds_to_exclude_from_id]
phi_T, beta_T, dist_par_un_T = mod_dgp.get_seq_latent_par() 


from dynwgraphs.dirGraphs1_dynNets import dirSpW1_SD

mod_dgp = dirSpW1_SD(torch.zeros(N, N, T))

mod_dgp.inds_to_exclude_from_id

mod_dgp.phi_T
mod_dgp.mod_bin = mod_dgp_bin
for p in mod_dgp.mod_bin.phi_T:
    p[:] = -5 
for p in mod_dgp.phi_T:
    p.requires_grad=False
    p[:] = 1 
mod_dgp.set_par_to_exclude_to_zero()
mod_dgp.phi_T

mod_dgp.sample_and_set_Y_T(A_T = mod_dgp.mod_bin.sample_Y_T())

mod_dgp.inds_to_exclude_from_id
(mod_dgp.Y_T>0).sum()


mod_dgp.mod_bin.phi_T
mod_dgp.mod_bin.sample_Y_T()[:,:,1].sum()
mod_dgp.sample_and_set_Y_T(A_T = mod_dgp.mod_bin.sample_Y_T())
mod_dgp.inds_to_exclude_from_id
mod_dgp.Y_T.sum()

mod_filt.roll_sd_filt_train()
mod_filt.identify_sequence()
mod_filt.phi_T[0]
mod_filt.plot_phi_T()

strIO_from_tens_T(Y_T)[~mod_filt.inds_to_exclude_from_id] 


mod_filt.inds_to_exclude_from_id
phi_T, beta_T, dist_par_un_T = mod_filt.get_seq_latent_par() 

plt.plot(phi_T[strIO_from_tens_T(Y_T) == 0, :].T)

phi_to_exclude = mod_filt.inds_to_exclude_from_id
filt_err(mod_dgp_w, mod_filt_w_sd, phi_to_exclude)


i = torch.where(~splitVec(phi_to_exclude)[0])[0][0]
x_T = X_T[0, 0, 0, :].numpy()

fig_ax = plt.subplots(2, 1)
mod_filt.plot_phi_T(i=i, fig_ax=mod_dgp.plot_phi_T(i=i, fig_ax=fig_ax))


# %%
t0 = 0
i=4
phi_T, _, _ = mod_filt.get_seq_latent_par()
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


