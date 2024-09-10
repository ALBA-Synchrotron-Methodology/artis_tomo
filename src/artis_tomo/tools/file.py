#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:04:03 2018

@author: joton
"""
import glob
import os


def removePattern(pattern):

    for file in glob.glob(pattern):
        os.remove(file)
