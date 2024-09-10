#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 09:48:06 2019

@author: joton
"""

import time
from copy import copy
from tqdm import tqdm
import SharedArray as sa
import numpy as np
import mrcfile
from . import filter as ft
from . import metadataIO as mio
from ..math import statistics as mstat


def bgNormalise(fnInList, fnOutList, rad, asGroup=True, verb=False):

    boxSize = mrcfile.mmap(fnInList[0]).data.shape[0]
    mask = ~ft.masksphere(boxSize, rad)

    nPart = len(fnInList)

    if asGroup:
        if verb:
            print('Calculating mean/var for the particles group ...')
            pbar = tqdm(total=nPart, unit="Part")

        stat = np.zeros((nPart, 2))
        for k, fnPart in enumerate(fnInList):
            vol = mrcfile.open(fnPart).data
            stat[k, :] = np.array(mstat.meanVar(vol[mask]))
            if verb:
                pbar.update()

        gMean = stat[:, 0].mean()
        gVar = stat[:, 1].mean() + stat[:, 0].var()
        gStd = np.sqrt(gVar)

        if verb:
            print(f'   mean = {gMean} ; var = {gVar}')
            print('Normalising particles ...')
            pbar = tqdm(total=nPart, unit="Part")

        for fnIn, fnOut in zip(fnInList, fnOutList):
            vol = mrcfile.open(fnIn, 'r+').data
            vol = (vol - gMean)/gStd
            outFile = mrcfile.new(fnOut, overwrite=True)
            outFile.set_data(vol.astype(np.float32))
            outFile.close()
            if verb:
                pbar.update()

    else:
        if verb:
            print('Normalising particles ...')
            pbar = tqdm(total=nPart, unit="Part")

        for fnIn, fnOut in zip(fnInList, fnOutList):
            vol = mrcfile.open(fnIn, 'r+').data
            gMean, gStd = mstat.meanStd(vol[mask])
            vol = (vol - gMean)/gStd
            outFile = mrcfile.new(fnOut, overwrite=True)
            outFile.set_data(vol.astype(np.float32))
            outFile.close()
            if verb:
                pbar.update()


def bgNormaliseMask(fnStarIn, fnStarOut, fnMask, thr=1, asGroup=True, verb=False):

    dataPartIn = mio.readStarFile(fnStarIn, 'data_particles')
    dataPartOut = mio.readStarFile(fnStarOut, 'data_particles')
    mask = mrcfile.open(fnMask).data

    nPart = len(dataPartIn)

    if asGroup:
        if verb:
            print('Calculating mean/var for the particles group ...')
            pbar = tqdm(total=nPart, unit="Part")

        stat = np.zeros((nPart, 2))
        for k, fnPart in enumerate(fnInList):
            vol = mrcfile.open(fnPart).data
            stat[k, :] = np.array(mstat.meanVar(vol[mask]))
            if verb:
                pbar.update()

        gMean = stat[:, 0].mean()
        gVar = stat[:, 1].mean() + stat[:, 0].var()
        gStd = np.sqrt(gVar)

        if verb:
            print(f'   mean = {gMean} ; var = {gVar}')
            print('Normalising particles ...')
            pbar = tqdm(total=nPart, unit="Part")

        for fnIn, fnOut in zip(fnInList, fnOutList):
            vol = mrcfile.open(fnIn, 'r+').data
            vol = (vol - gMean)/gStd
            outFile = mrcfile.new(fnOut, overwrite=True)
            outFile.set_data(vol.astype(np.float32))
            outFile.close()
            if verb:
                pbar.update()

    else:
        if verb:
            print('Normalising particles ...')
            pbar = tqdm(total=nPart, unit="Part")

        for fnIn, fnOut in zip(fnInList, fnOutList):
            vol = mrcfile.open(fnIn, 'r+').data
            gMean, gStd = mstat.meanStd(vol[mask])
            vol = (vol - gMean)/gStd
            outFile = mrcfile.new(fnOut, overwrite=True)
            outFile.set_data(vol.astype(np.float32))
            outFile.close()
            if verb:
                pbar.update()


def arrayNorm(array):

    minV = array.min()
    maxV = array.max()
    return (array - array.mean())/(maxV - minV)


def binning(array, binFactor, axes=None):

    iShape = np.array(array.shape)
    nDim = len(iShape)

    if axes is None:
        axes = np.arange(nDim)

    tmpShape = list()
    cropSlices = list()

    for ax in range(nDim):
        if ax in axes:
            outSize = int(iShape[ax]/binFactor)
            tmpShape = tmpShape + [outSize, binFactor]
            endCrop = outSize*binFactor - iShape[ax]
            if endCrop < 0:
                cropSlices.append(slice(None, endCrop))
            else:
                cropSlices.append(slice(None))
        else:
            tmpShape = tmpShape + [iShape[ax]]
            cropSlices.append(slice(None))

    arrayOut = np.reshape(array[tuple(cropSlices)], tmpShape)
    for k in axes:
        arrayOut = arrayOut.mean(1+k)

    return arrayOut


def binningFourier(array, binFactor):

    dtype = array.dtype

    iShape = np.array(array.shape)
    nDim = len(iShape)
    iSize = iShape[0]
    iC = (iShape//2).astype(int)
    iSizeh = iSize/2 + 1

    halfMode = nDim > 1 and iSizeh == iShape[-1]

    arrayOut = array.copy()

    eAx = nDim-1 if halfMode else nDim

    sl = [slice(None) for k in range(nDim)]

    for ax in range(eAx):
        shape = np.array(arrayOut.shape)
        iSizeC = int(shape[ax]/2)
        shape[ax] = int(shape[ax]/binFactor)
        oSizeC = int(shape[ax]/2)

        slIn = copy(sl)
        slOut = copy(sl)
        tmpOut = np.zeros_like(arrayOut)

        # Binning applied to the first half before zero order
        slIn[ax] = slice(iSizeC)
        slOut[ax] = slice(oSizeC)
        tmpOut[tuple(slOut)] = binning(arrayOut[tuple(slIn)], binFactor, [ax])
        # Zero order is not combined with any other order
        slIn[ax] = iSizeC
        slOut[ax] = oSizeC
        tmpOut[tuple(slOut)] = arrayOut[tuple(slIn)]
        # Binning applied to the second half after zero order
        # Last index has no pair and is ignored
        slIn[ax] = slice(iSizeC+1, -1)
        slOut[ax] = slice(oSizeC+1, None)
        tmpOut[tuple(slOut)] = binning(arrayOut[tuple(slIn)], binFactor, [ax])

        arrayOut = tmpOut

    if halfMode:
        shape = np.array(arrayOut.shape)
        shape[-1] = int(shape[-2]/2 + 1)

        tmpOut = np.zeros(shape, dtype=dtype)
        # Zero order is not combined with any other order
        tmpOut[..., 0] = arrayOut[..., 0]
        tmpOut[..., 1:] = binning(arrayOut[..., 1:], binFactor, [nDim-1])

        arrayOut = tmpOut

    return arrayOut

