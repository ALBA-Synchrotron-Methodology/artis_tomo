#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:56:51 2024

@author: joton
"""


import numpy as np
from artis_tomo.image.transformation import convertToPolar
from artis_tomo.image import frame as fr, filter as ft


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