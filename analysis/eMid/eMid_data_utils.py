#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday August 4th 2021

"""

import torch
from dynwgraphs.utils.tensortools import tens
import numpy as np
from types import SimpleNamespace

def get_obs_and_regr_mat_eMid(ld_data, unit_meas, regr_list_in):
    Y_T = ld_data["YeMidWeekly_T"][:, :, 2:]/unit_meas
    # lagged nets to be used as regressor
    Ytm1_T = ld_data["YeMidWeekly_T"][:, :, 1:-1].unsqueeze(dim=2)/unit_meas
    logYtm1_T = (torch.log(Ytm1_T)).nan_to_num(posinf=0, neginf=0)
    N, _, T = Y_T.shape
    X_T = torch.zeros(N, N, 1, T)
    regr_list = [""]

    if "eonia" in regr_list_in:
        X_eonia_T = tens(np.tile(ld_data["eonia_T"][:, 2:].numpy(), (N, N, 1, 1)))
        X_T = torch.cat((X_T, X_eonia_T), dim=2)
        regr_list.append("eonia")
    if "logYtm1" in regr_list_in:
        X_T = torch.cat((X_T, logYtm1_T), dim=2)
        regr_list.append("logYtm1")
    if "Atm1" in regr_list_in:
        X_T = torch.cat((X_T, Ytm1_T >0), dim=2)
        regr_list.append("Atm1")

    if X_T.shape[2] >1:
        X_T = X_T[:, :, 1:, :]
        regr_list = regr_list[1:]
    else:
        X_T = None

    net_stats = SimpleNamespace()
    net_stats.__dict__.update({
        "avg_degs_i": (Y_T > 0).sum(axis=(0)).double().mean(axis=1),
        "avg_degs_o": (Y_T > 0).sum(axis=(1)).double().mean(axis=1),
        "avg_str_i": (Y_T).sum(axis=(0)).mean(axis=1),
        "avg_str_o": (Y_T).sum(axis=(1)).mean(axis=1),
    })

    return Y_T, X_T, regr_list, net_stats
