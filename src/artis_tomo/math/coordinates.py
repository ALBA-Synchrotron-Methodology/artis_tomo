#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coordinate systems transformations

@author: joton
"""

import numpy as np


def pol2cart(theta, ro):
    x = ro*np.cos(theta)
    y = ro*np.sin(theta)
    return (x, y)


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    return (theta, rho)