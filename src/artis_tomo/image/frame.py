#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Change array frames by padding / cropping

@author: joton
"""

import numpy as np
from artis_tomo.math.function import raisedCos


def padwithrc(vector, pad_width, iaxis, kwargs):
    """Perform a raised cosine padding function for numpy.pad."""
    value = kwargs.get('end_values', 0)
    rcpad = kwargs.get('rcpad', -1)

    if pad_width[0] > 0:
        if rcpad < 0:
            vector[:pad_width[0]] = (raisedCos(pad_width[0]) *
                                     (vector[pad_width[0]+1]-value) + value)
        else:
            rcprof = raisedCos(rcpad)
            if rcpad < pad_width[0]:
                vector[:pad_width[0]-rcpad] = value
                vector[pad_width[0]-rcpad:pad_width[0]] = \
                    (rcprof *
                     (vector[pad_width[0]+1]-value) + value)
            else:
                vector[:pad_width[0]] = \
                    (rcprof[-pad_width[0]:] *
                     (vector[pad_width[0]+1]-value) + value)

    if pad_width[1] > 0:
        if rcpad < 0 or rcpad == pad_width[1]:
            vector[-pad_width[1]:] = \
                ((raisedCos(pad_width[1])[::-1]) *
                 (vector[-pad_width[1]-1]-value) + value)
        else:
            rcprof = raisedCos(rcpad)
            if rcpad < pad_width[1]:
                vector[-pad_width[1]:-pad_width[1]+rcpad] = \
                    ((rcprof[::-1]) *
                     (vector[-pad_width[1]-1]-value) + value)
                vector[-pad_width[1]+rcpad:] = value
            else:
                vector[-pad_width[1]:] = \
                    ((raisedCos(rcpad)[:-pad_width[1]-1:-1]) *
                     (vector[-pad_width[1]-1]-value) + value)

    return vector


def padArrayCentered(arrayIn, oSize, mode=padwithrc, **kwargs):
    """
    Pad an array keeping the center pixel of the input centered.

    Parameters
    ----------
    arrayIn : array_like of rank N
        Input array
    oSize : {sequence, array_like, int}
        Output pixel length of each axis.
        (N_1, N_2, ... N_n)) unique sizes for each axis.
        (N_1,) Only applies for the first axis, keeping the rest axes at the
        same length.

    Returns
    -------
    arrayOut : ndarray
        Padded array of rank equal to `array` with shape increased
        according to `oSize`.
    padpre   : {sequence, array_like, int}
        pad width sizes applied in the beginning of each dimension.
    padpost  : {sequence, array_like, int}
        pad width sizes applied in the end of each dimension.
    """
    if mode == padwithrc:
        unsupported_kwargs = set(kwargs) - set(['end_values', 'rcpad'])
        if unsupported_kwargs:
            raise ValueError(f"unsupported keyword arguments for mode "
                             f"'padwithrc': {unsupported_kwargs}")

    kwargs['mode'] = mode

    iSize = np.array(arrayIn.shape)
    iDim = arrayIn.ndim

    aoSize = np.array(oSize)  # array oSize
    pDim = np.size(aoSize)
    pSize = np.zeros(iDim).astype(int)  # Paddedsize autoexpanded
    pSize[:pDim] = aoSize

    if pDim < iDim:
        pSize[pDim:iDim] = iSize[pDim:iDim]

    iC = np.floor(iSize/2).astype(int)
    pC = np.floor(pSize/2).astype(int)

    padpre = pC-iC
    padpost = (pSize - iSize) - padpre

    arrayOut = np.pad(arrayIn, list(zip(padpre, padpost)), **kwargs)

    return arrayOut, padpre, padpost


def cropArrayCentered(arrayIn, croppedSize):
    """
    Crop an array keeping the center pixel of the input centered.

    Parameters
    ----------
    arrayIn : array_like of rank N
        Input array
    croppedSize : {sequence, array_like, int}
        Output pixel length of each axis.
        (N_1, N_2, ... N_n)) unique sizes for each axis.
        (N_1,) Only applies for the first axis, keeping the rest axes at the
        same length.

    Returns
    -------
    arrayOut : ndarray
        Padded array of rank equal to `array` with shape increased
        according to `croppedSize`.
    """
    iSize = np.array(arrayIn.shape)
    iDim = np.size(iSize)

    cSize = np.array(croppedSize)  # array croppedSize
    cDim = np.size(cSize)

    iC = np.floor(iSize/2).astype(np.int64)
    cC = np.floor(cSize/2).astype(np.int64)

    padpre = iC - cC
    padpost = (iSize - cSize) - padpre

    slInd = [None]*iDim
    for k in range(cDim):
        slInd[k] = slice(padpre[k],
                         None if padpost[k] == 0 else -padpost[k])

    for k in range(cDim, iDim):
        slInd[k] = slice(None)

    return arrayIn[tuple(slInd)]


def padCropArrayCentered(arrayIn, oSize, mode=padwithrc, **kwargs):
    """
    Pad or crops any dimension of an array keeping the center pixel centered.

    Parameters
    ----------
    arrayIn : array_like of rank N
        Input array
    oSize : {sequence, array_like, int}
        Output pixel length of each axis.
        (N_1, N_2, ... N_n)) unique sizes for each axis.
        (N_1,) Only applies for the first axis, keeping the rest axes at the
        same length.

    Returns
    -------
    arrayOut : ndarray
        Padded array of rank equal to `array` with shape increased
        according to `oSize`.
    padpre   : {sequence, array_like, int}
        pad width sizes applied in the beginning of each dimension.
    padpost  : {sequence, array_like, int}
        pad width sizes applied in the end of each dimension.
    """
    iSize = np.array(arrayIn.shape)

    outSize = np.array(oSize)
    outDim = np.size(outSize)

    cropDimensions = outSize < iSize[:outDim]

    cropSize = cropDimensions*outSize + (~cropDimensions)*iSize[:outDim]

    if np.any(cropDimensions):
        cropArray = cropArrayCentered(arrayIn, cropSize)
    else:
        cropArray = arrayIn

    return padArrayCentered(cropArray, outSize, mode, **kwargs)
