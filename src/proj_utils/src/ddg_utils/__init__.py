#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday July 31st 2021

"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
logger = logging.getLogger(__name__)


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

def get_env_or_default(name):
    def_dict = {
        'DYNWGRAPHS_PROJ_FOLD':  'd:\\pcloud\\dynamic_networks\\repos\\dynwgraphspaper\\'
    }

    if name in os.environ.keys():
        value =  os.environ[name]
        logger.info(f"getting env var for {name} =  {value}")
    else:
        value = def_dict[name]
        logger.info(f"getting default value for {name}  =  {value}")

    return value

def get_proj_fold():
    return Path(get_env_or_default("DYNWGRAPHS_PROJ_FOLD"))
