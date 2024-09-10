#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:50:48 2017

@author: joton
"""


def wavelengthToEnergy(wavelength):
    """
    Convert the wavelength value of a EMW to its corresponding energy value.

    The wavelentgh units must be in m.
    energy = wavelengthToEnergy(lambda) returns the energy in eV units.

    """

    energy = 4.136e-15*2.998e8/wavelength
    return energy


def energyToWavelength(energy):
    """
    Convert the energy value to its corresponding wavelength value.

    The wavelentgh units must be in m.
    wavelength = energyToWavelength(lambda) returns the energy in eV units.
    """

    wavelength = 4.136e-15*2.998e8/energy
    return wavelength


def ind2sub(shape, ind, order='C'):
    """
    Convert linear indices to subscripts

    Parameters
    ----------
    shape : tuple of array dimensions
    ind:    vector of linear indices
    order : {‘C’, ‘F’}, optional
            ‘C’ means to flatten in row-major (C-style) order. ‘F’ means to
            flatten in column-major (Fortran- style) order. The default is ‘C’.

    Returns
    -------
    subs : 2d array
           Sequence of subscripts corresponding to linear indices
    """
    import numpy as np

    shape = np.array(shape)
    nDim = shape.size

    ind = np.atleast_1d(ind)

    if order == 'C':
        shape[[0, 1]] = shape[[1, 0]]

    shapecum = np.append(1, np.cumprod(shape[0:-1]))

    subs = np.zeros(ind.shape+shape.shape, dtype=int)
    for ki, idx in enumerate(ind):

        assert idx < np.prod(shape), "Index out of bounds"
        idxcum = idx/shapecum

        sub = np.zeros(nDim+1)
        tmp = 0
        for k in range(nDim-1, -1, -1):
            tmp = (tmp + sub[k+1])*shape[k]
            sub[k] = int(idxcum[k] - tmp)

        subs[ki, ...] = sub[:-1]

    if order == 'C':
        subs[:, [0, 1]] = subs[:, [1, 0]]

    return subs


def sub2ind(shape, subs, order='C'):
    """
    Convert subscripts to linear indices

    Parameters
    ----------
    shape : tuple of array dimensions
    subs : 2d array
           Sequence of subscripts indices
    order : {‘C’, ‘F’}, optional
            ‘C’ means to flatten in row-major (C-style) order. ‘F’ means to
            flatten in column-major (Fortran- style) order. The default is ‘C’.

    Returns
    -------
    ind:    vector of linear indices
    """
    import numpy as np

    subs = np.atleast_2d(subs)
    shape = np.array(shape)

    if order == 'C':
        shape[[0, 1]] = shape[[1, 0]]
        subs[:, [0, 1]] = subs[:, [1, 0]]

    shapecum = np.append(1, np.cumprod(shape[0:-1]))

    nS = subs.shape[0]
    ind = np.zeros(nS)

    for ks in range(nS):
        sub = subs[ks, :]

        assert (sub < shape).all(), "Subscripts out of bounds"

        ind[ks] = np.sum(shapecum*sub)

    return ind


