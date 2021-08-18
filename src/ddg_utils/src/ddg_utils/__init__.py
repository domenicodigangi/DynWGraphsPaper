#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday July 31st 2021

"""

import pandas as pd
import numpy as np


def drop_keys(d:dict, keys):
    return {k: d[k] for k in d.keys() - keys}


def pd_filt_on(df, filt_dict):
    idx = np.ones(df.shape[0], bool)
    for k, v in filt_dict.items():
        if type(v) == list:
            new_idx = df[k].apply(lambda x: x in v).values
            idx = idx & new_idx
        else:
            new_idx = (df[k] == v).values
            idx = idx & new_idx
    return df[idx]
