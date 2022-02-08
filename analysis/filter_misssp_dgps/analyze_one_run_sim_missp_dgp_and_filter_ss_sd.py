# %%
import importlib
import dynwgraphs
import logging
import proj_utils
from proj_utils.mlflow import _get_and_set_experiment, get_df_exp
from utils_missp_sim import load_all_models_missp_sim
from matplotlib import pyplot as plt

from dynwgraphs.utils.dgps import get_test_w_seq, _test_w_data_

logger = logging.getLogger(__name__)
importlib.reload(dynwgraphs)
importlib.reload(proj_utils)

#%%
experiment = _get_and_set_experiment("Table_1_temp")

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
inds = df["phi_set_dgp_type_tv_bin"] == "('SIN', 'ref_mat', 1.0, 0.15)"


i=1
fig, ax = plt.subplots(figsize=(25, 15))
for r, row_run in df.loc[inds, :].iloc[:50, :].iterrows():
    mod_filt_sd_bin, mod_filt_sd_w, mod_filt_ss_bin, mod_filt_ss_w, mod_dgp_bin, mod_dgp_bin, mod_dgp_w, obs, Y_reference = load_all_models_missp_sim(row_run)

    phi_w_T_sd = mod_filt_sd_w.get_ts_phi()
    phi_w_T_ss = mod_filt_ss_w.get_ts_phi()
    phi_w_T_dgp = mod_dgp_w.get_ts_phi()
    ax.plot(phi_w_T_sd[i, :], "-r")
    ax.plot(phi_w_T_ss[i, :], ".b")

ax.plot(phi_w_T_dgp[i, :], "-k", linewidth=4)
ax.grid()
# %%


# %%
