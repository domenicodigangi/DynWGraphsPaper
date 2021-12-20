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

import yaml

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def get_env_or_conf_or_default(name):
    def_dict = {
        'PROJECT_FOLDER':  '/data/digiandomenico/phd_proj/DynWGraphsPaper'
    }
    value = None
    if name in os.environ.keys():
        value = os.environ[name]
        logger.info(f"getting env var for {name} =  {value}")
    else:
        config_path = Path(__file__).parents[3] / "proj_config.yml"
        if config_path.exists():
            config = read_yaml(config_path)
            if name in config.keys():
                value = config[name]
                logger.info(f"reading value from config file {name}  =  {value}")

    if value is None:
        value = def_dict[name]
        logger.info(f"getting default value for {name}  =  {value}")

    return value




def get_proj_fold():
    proj_fold = Path(__file__).parent.parent.parent.parent
    return proj_fold