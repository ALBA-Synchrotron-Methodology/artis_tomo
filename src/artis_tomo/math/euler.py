#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Euler angles conversion

@author: joton
"""

import numpy as np
# from transforms3d.euler import euler2mat, mat2euler


def rotX2mat(alpha):

    out = np.matrix([[1,    0,              0         ],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha),  np.cos(alpha)]])
    return out


def rotY2mat(beta):

    out = np.matrix([[np.cos(beta),   0,     np.sin(beta)],
                    [    0,           1,        0        ],
                    [-np.sin(beta),   0,    np.cos(beta)]])
    return out


def rotZ2mat(gamma):

    out = np.matrix([[np.cos(gamma), -np.sin(gamma), 0 ],
                     [np.sin(gamma),  np.cos(gamma), 0 ],
                     [   0,              0,          1 ]])
    return out


def rotX2d4mat(alpha):

    out = np.matrix([[1,    0,              0,         0],
                    [0, np.cos(alpha), -np.sin(alpha), 0],
                    [0, np.sin(alpha),  np.cos(alpha), 0],
                    [0,     0,              0,         1]])
    return out


def rotY2d4mat(beta):

    out = np.matrix([[np.cos(beta),   0,    np.sin(beta), 0],
                    [    0,           1,       0        , 0],
                    [-np.sin(beta),   0,    np.cos(beta), 0],
                    [    0,           0,       0,         1]])
    return out


def rotZ2d4mat(gamma):

    out = np.matrix([[np.cos(gamma), -np.sin(gamma), 0, 0],
                     [np.sin(gamma),  np.cos(gamma), 0, 0],
                     [   0,              0,          1, 0],
                     [   0,              0,          0, 1]])
    return out


def d3matShifts(shifts):

    out = np.matrix([[1, 0, shifts[0]],
                     [0, 1, shifts[1]],
                     [0, 0,      1]])
    return out


def d4matShifts(shifts):

    out = np.matrix([[1, 0, 0, shifts[0]],
                     [0, 1, 0, shifts[1]],
                     [0, 0, 1, shifts[2]],
                     [0, 0, 0,      1]])
    return out