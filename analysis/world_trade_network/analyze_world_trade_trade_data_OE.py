import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
sys.path.append("./src/")
from utils import tens

#load yearly world Trade networks
ld_data = np.load("./data/world_trade_network/world_trade_net_T.npz",
                  allow_pickle=True)
wtn_T, all_y, nodes= ld_data["wtn_T"], ld_data["all_y"], ld_data["nodes"]
dist_T, scaling_infl = ld_data["dist_T"], ld_data["scaling_infl"]

Y_T = torch.tensor(wtn_T * scaling_infl[:, 1])
X_T = torch.tensor(dist_T/dist_T.mean())
N = wtn_T.shape[0]
T = wtn_T.shape[2]

# %%
dens_T = torch.zeros(0)
for t in range(T):
    Y_t = Y_T[:, :, t]
    A_t = tens(Y_t>0)
    dens_T = torch.cat((dens_T, A_t.mean().unsqueeze(-1)), dim=0)
    y_t = Y_t[Y_t > 0]
    if t == 0:
        stats_T = torch.stack((y_t.mean(), y_t.std(), y_t.min(), y_t.max())).unsqueeze(-1)
    else:
        stats_T = torch.cat((stats_T, torch.stack((y_t.mean(), y_t.std(), y_t.min(), y_t.max())).unsqueeze(-1)), dim=1)

plt.plot(all_y, dens_T)
plt.grid()

plt.close()
plt.plot(all_y, stats_T[3, :])
plt.grid()

plt.close()
y_plot = [55, 29, 19, 0]
for t in y_plot:
    Y_t = Y_T[:, :, t]
    A_t = tens(Y_t > 0)
    y_t = Y_t[Y_t > 0]

    #plt.close()
    plt.hist(torch.log10(y_t), alpha=0.3)
plt.legend(all_y[y_plot])
# %%
A_T = tens(Y_T>0)
plt.close()
plt.hist(A_T.sum(dim=2).view(-1)/T)

# %%
for t in range(T):
    Y_t = Y_T[:, :, t]
    A_t = tens(Y_t>0)


