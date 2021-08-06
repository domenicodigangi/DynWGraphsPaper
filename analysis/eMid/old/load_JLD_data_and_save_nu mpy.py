"""
load data saved in julia and store them in numpy format
"""


import datetime
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
filename = "../../data/emid_data/juliaFiles/Weekly_eMid_Data_from_2009_06_22_to_2015_02_27.jld"
ld_data = h5py.File(filename, "r")

# load networks
w_mat_T= np.transpose(np.array(ld_data["YeMidWeekly_T"]), axes=(1, 2, 0)).astype(float)
days_obs = np.array(ld_data["days"]).astype(int)
months_obs = np.array(ld_data["months"]).astype(int)
years_obs = np.array(ld_data["years"]).astype(int)
nodes = np.array(ld_data["banksIDs"])
Y_T = w_mat_T[:, :, 2:]
N = Y_T.shape[0]
T = Y_T.shape[2]
obs_dates = [datetime.datetime(year, month, day ) for year, month, day in zip(years_obs, months_obs, days_obs)]
obs_dates = np.array(obs_dates).astype(np.datetime64)
obs_dates=obs_dates[2:]

# %% load eonia data
import pandas as pd
df = pd.read_csv("../../data/emid_data/csvFiles/eonia_rates.csv", skiprows=5, header=None,
                 usecols=[0, 1], names=['date', 'rate_eonia'])

df = df.set_index(pd.to_datetime(df['date'])).drop(columns='date')
df = df.loc[:obs_dates[0]]
df = df.loc[ obs_dates[-1] :]
df =df.sort_values(by='date')
# weekly frequency eonia rates
eonia_week = df.resample('W').mean()
eonia_week.index.weekday.unique()
all_dates = np.array(eonia_week.index)

obs_dates.shape
all_dates.shape

# %%
dens_T  = (Y_T>0).mean(axis=(0, 1))
plt.close()
plt.plot( all_dates, np.array(eonia_week.values))
plt.plot(all_dates, dens_T/dens_T.mean()*eonia_week.iloc[100].values)


# %%
save_path = "../../data/emid_data/numpyFiles/eMid_numpy"
np.savez(save_path, Y_T, all_dates, eonia_week, nodes)

