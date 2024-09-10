#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expand info from 1d array to 2d array.

@authors: josue Gomez, Joaquin Oton
"""

import numpy as np
from scipy import interpolate


def extendRadSym(linear_profiles, outNdim=2):
    """
    linear_profiles: Array of arrays of 1D linear profiles.
                     One for each image.
    """

    iDims = linear_profiles.shape
    nDims = len(iDims)
    xSize = iDims[0]

    if nDims == 1:
        linp = np.expand_dims(linear_profiles, axis=1)
        nProfiles = 1
    elif nDims == 2:
        linp = linear_profiles
        nProfiles = iDims[1]
    else:
        raise Exception('extendRadSym: Wrong number of dimensions for input \
                        profiles')

    oSize = 2*xSize - 1
    xC = int(oSize/2)
    xV = np.arange(oSize) - xC

    # Meshgrid consider the same xV for all the dimensions. xx is a list with
    # the arrays for all the dimension
    xx = np.array(np.meshgrid(*[xV]*outNdim))
    ro = np.sqrt((xx**2).sum(axis=0))

    xx, yy = np.meshgrid(xV, xV)

    radSym2D = np.empty((oSize,)*outNdim + (nProfiles,))

    for k in range(nProfiles):
        f = interpolate.interp1d(xV[xC:], linp[:, k], bounds_error=False,
                                 fill_value=0., kind='linear')
        radSym2D[..., k] = f(ro)

    return np.squeeze(radSym2D)


