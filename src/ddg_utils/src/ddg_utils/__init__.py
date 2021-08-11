#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday July 31st 2021

"""


def drop_keys(d:dict, keys):
    return {k: d[k] for k in d.keys() - keys}