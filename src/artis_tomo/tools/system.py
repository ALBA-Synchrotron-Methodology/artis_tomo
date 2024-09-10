#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:43:47 2019

@author: joton
"""

import os
import traceback


def getNargout():

    callInfo = traceback.extract_stack()
    callLine = str(callInfo[-3].line)
    # print(callLine)
    split_equal = callLine.split('=')
    if len(split_equal) > 1:
        split_comma = split_equal[0].split(',')
        return len(split_comma)
    else:
        return 0


def touch(file):
    with open(file, 'a'):
        os.utime(file, None)
