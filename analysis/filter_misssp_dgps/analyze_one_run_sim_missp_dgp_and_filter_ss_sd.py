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

dfs = get_df_exp(experiment)

logger.info(
    f"Staus of experiment {experiment.name}: \n {dfs['info']['status'].value_counts()}"
)

ind_fin = (dfs["info"]["status"] == "FINISHED") & (
    ~dfs["metrics"]["filt_bin_sd_actual_n_opt_iter"].isna()
)

ind_fin = ind_fin & (~ dfs["par"]["init_sd_type"].isna())

df_i = dfs["info"][ind_fin]
df_p = dfs["par"][ind_fin]
df_m = dfs["metrics"][ind_fin]

df = df_i.merge(df_p, on="run_id").merge(df_m, on="run_id")
# %%
row_run = df.iloc[0, :]
row_run.init_sd_type

mod_filt_sd_bin, mod_filt_sd_w, mod_filt_ss_bin, mod_filt_ss_w, mod_dgp_bin, mod_dgp_bin, mod_dgp_w, obs, Y_reference = load_all_models_missp_sim(row_run)


Y_reference["Y_reference_w"].sum()
Y_reference["Y_reference_bin"].sum()

phi_w_T_sd = mod_filt_sd_w.get_ts_phi()
phi_w_T_ss = mod_filt_ss_w.get_ts_phi()
phi_w_T_dgp = mod_dgp_w.get_ts_phi()

i=10
plt.plot(phi_w_T_dgp[i, :], "-k")
plt.plot(phi_w_T_sd[i, :], "-r")
plt.plot(phi_w_T_ss[i, :], ".b")