#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:56:51 2024

@author: joton
"""


import numpy as np
from artis_sci.image.transformation import convertToPolar
from artis_sci.image import frame as fr, filter as ft


def getTiltAngleRange(vol):

    voly = vol.sum(axis=1)

    smax = max(voly.shape)

    volypad = fr.padArrayCentered(voly, (smax,)*2)[0]
    mask = ft.maskRaisedCosineBorder2D(volypad.shape, 50)

    volyft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(volypad*mask)))

    vyftpol = convertToPolar(np.abs(volyft))
    nR, nTheta = vyftpol.shape
    nRhalf = nR//2
    thetaCenter = nTheta//2
    thetaRange = thetaCenter//2

    # We analyze a radial fraction
    angInt = vyftpol[nRhalf:int(nRhalf*1.5),
                     thetaCenter-thetaRange:thetaCenter+thetaRange].mean(0)

    thetaV = np.linspace(-180, 180, nTheta + 1)[:-1]
    thetaV = thetaV[thetaCenter-thetaRange:thetaCenter+thetaRange]
    d_theta = np.diff(thetaV[:2])[0]
    df_theta = 1/thetaV.shape[0]/d_theta

    # % Getting the angle step ##
    angIntft = np.fft.rfft(np.fft.ifftshift(angInt))

    # We use a minpos for searching the maximum in angIntft to ignore the peak around 0
    minpos = int(1/(df_theta*5))  # We expect angle step to be smaller than 5º

    maxpos = np.argmax(np.abs(angIntft[minpos:])) + minpos

    anglestep = np.round(1/(df_theta*maxpos), 1)
    ##

    # % Getting the max and min angles ##
    angRangeNoise = 5  # Angular range to estimate bg noise level
    noisePos = np.nonzero(np.abs(thetaV) > (90 - angRangeNoise))
    noiseMean = np.mean(np.abs(angInt[noisePos]))

    # Coarse indexes for extreme angles were there's signal from projections
    edgePoss = np.flatnonzero(angInt > noiseMean*2)[[0, -1]]

    # Positive values
    locProf = angInt[edgePoss[-1]-50:edgePoss[-1]]
    thr = np.mean(locProf) - np.std(locProf)
    pos0 = edgePoss[-1] - 49 + np.flatnonzero(locProf > thr)[-1]

    posPos = pos0 - 49 + np.flatnonzero(np.diff(
        angInt[pos0-50:pos0]) > 0)[-1]

    thetaMin, thetaMax = np.round(thetaV[[negPos, posPos]], 1)


    # aft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(angInt)))
    # mask = ft.maskRaisedCosineRadial(aft.shape, 9, pad= 3)
    # angIntBg = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(aft*mask))).real



    return thetaMin, thetaMax, anglestep

def selectTiltSubset(angles, nProjs):
    """
    Choose a subset of tilt angles that stays symmetric about zero.

    Picks *nProjs* angles spread evenly across *angles*, then nudges the
    selection so the two sides of the series pair up at equal absolute angles
    and the middle sits at the angle closest to zero. A subset that is
    lopsided in tilt biases anything fitted from it, which is why the even
    spacing alone is not enough.

    *nProjs* is rounded up to an odd number so that a central angle exists.

    Parameters
    ----------
    angles : sequence of float
        Tilt angles of the series, in acquisition order.
    nProjs : int
        Number of angles wanted.

    Returns
    -------
    list of int
        Indices into *angles*.
    """
    if nProjs % 2 == 0:
        nProjs += 1

    pos_list = np.linspace(0, len(angles) - 1, nProjs).astype(int).tolist()
    mangles = [abs(x) for x in angles]

    central_pos = len(pos_list) // 2
    for i in range(central_pos + 1):
        val1 = mangles[pos_list[i]]
        val2 = mangles[pos_list[-i - 1]]
        if val1 != val2:
            indx1 = [j for j in range(len(mangles)) if mangles[j] == val1]
            indx2 = [j for j in range(len(mangles)) if mangles[j] == val2]
            l1 = len(indx1)
            l2 = len(indx2)
            if l1 == 2 and l2 == 1:
                pos_list[-i - 1] = indx1[1]
            elif l2 == 2:
                pos_list[i] = indx2[0]
            else:
                l = 1
                val = val1 + 1
                while l == 1:
                    indx = [j for j in range(len(mangles)) if mangles[j] == val]
                    l = len(indx)
                    if l == 2:
                        pos_list[i] = indx[0]
                        pos_list[-i - 1] = indx[1]
                    val = val + 1

        if i == central_pos and val1 != 0.0:
            pos_list[i] = mangles.index(min(mangles))

    sel_angles = [angles[idx] for idx in pos_list]
    print("Values: ", pos_list, sel_angles)

    return pos_list


def loadTomoFromInput(input, tltfn, pxsize, flipz):
    """
    Build a tomogram object from either an IMOD directory or a stack file.

    Accepts what the correlative programs are given on the command line: a
    directory produced by IMOD, from which the metadata is read, or a single
    tilt-series file plus a separate ``.tlt`` angles file.

    Parameters
    ----------
    input : str or Path
        Directory or tilt-series file.
    tltfn : str or Path
        IMOD-style tilt angles file. Used only for the single-file case.
    pxsize : float
        Pixel size of the tilt series.
    flipz : bool
        Whether to flip the reconstruction along Z.

    Returns
    -------
    tomo : Tomogram
        The tomogram object.
    folder : Path
        Directory the data lives in, for resolving further files against.
    """
    # Imported here rather than at module level: tomo_import_imod imports
    # artis_tomo.tomo, so a module-level import would close the loop.
    from ..programs.tomo_import_imod import getTomoClass, getTomoFromFiles
    from artis_sci.io import imageIO as iio
    from pathlib import Path

    file_name = Path(input)

    if file_name.is_file():
        folder = file_name.parent
        img_arr = iio.readImageArray(file_name)
        tomo = getTomoFromFiles(img_arr, file_name, tltfn, pxsize, flipz)
    else:
        folder = file_name
        tomo = getTomoClass(folder, pxsize, flipz)

    return tomo, folder
