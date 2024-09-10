#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alignment of images and volumes methods

@author: joton
"""

import numpy as np
import multiprocessing
from artis_tomo.math.transforms import tr2d
from artis_tomo.image import (transformation as tr,
                     frame as fr)


def getInPlaneAlign(array, ref, padfactor=2):
    ndim = array.ndim
    if ndim == 2:
        array = array[None, :, :]

    ni, ny, nx = array.shape
    size = int(np.min([[ny, nx], ref.shape[-2:]]))
    refFT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(ref)))

    refFTP = tr.convertToPolar(np.abs(refFT))
    nR, nTheta = refFTP.shape
    nThetah = nTheta//2
    refFTP = refFTP[:, :nThetah]
    refCCconj = np.fft.fft(refFTP, axis=1).conj()

    sizePad = int(size*padfactor)
    sizePadND = (sizePad,)*2
    sizePadh = sizePad//2

    refPad = fr.padArrayCentered(ref, sizePadND, mode='constant')[0]
    refFTconj = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(refPad))).conj()

    for k, image in enumerate(array):
        imFT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
        imagePad = fr.padArrayCentered(image, sizePadND,
                                       mode='constant')[0]
        imFT2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(imagePad)))

        maxposV, angleV, ccMax = inPlaneAlignFT(imFT, imFT2, refFTconj,
                                         refCCconj, k, nR, nThetah, ni)
    shifts = -(maxposV - np.array([sizePadh, sizePadh])*np.ones((ni, 1)))
    if ndim == 2:
        shifts = np.squeeze(shifts)
    return np.rad2deg(angleV), shifts, ccMax


def inPlaneAlignFT(imFT, imFT2, refFTconj, refCCconj, k, nR, nThetah, ni):
    angl = np.zeros(ni)
    maxpos = np.zeros((ni, 2))
    
    imFTP = tr.convertToPolar(np.abs(imFT))[:, :nThetah]
    imCC = np.fft.fft(imFTP, axis=1)
    cc = np.fft.fftshift(np.fft.ifft(refCCconj*imCC, axis=1), axes=1)
    angle = -(np.argmax(np.sum(np.real(cc)[int(nR*0.1):, :], axis=0))
              - nThetah//2)/nThetah*np.pi
    ccMax = 0

    for phase in [0, np.pi]:
        m = tr2d.angles2mat(angle + phase)
        imFTRot = tr.transformRS2D(imFT2, m)
        KK = refFTconj * imFTRot
        ccA = np.abs(np.fft.fftshift(np.fft.ifftn(KK)))
        cc = np.max(ccA)
        if cc > ccMax:
            ccMax = cc
            maxpos[k, 1], maxpos[k, 0] = \
                np.unravel_index(np.flatnonzero(ccA == cc), ccA.shape)
            angl[k] = angle + phase
    return maxpos, angl, ccMax


def corr2DFourierRot(ali_array, ref_array, angle):
    if angle==0:
        imFTRot = ali_array
    else:
        m = tr2d.angles2mat(np.deg2rad(angle))
        imFTRot = tr.transformRS2D(ali_array, m)
    
    ccft = ref_array * imFTRot.conj()
    cc = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ccft))).real
    cc = cc / np.abs(cc).sum()
    return cc


def fourierInplaneAlignBF(ali_ft, ref_ft, ang_params, thr=4):
    pool = multiprocessing.Pool(thr)
    s, f, step = ang_params
    if step == 0:
        step = 1
    st = int((f - s)/step) + 1
    angList = np.linspace(s, f, st)
    ccImgList = pool.starmap(corr2DFourierRot, [(ali_ft, ref_ft, angle)
                                                 for angle in angList])
    return angList, ccImgList


def getInplaneAlignBF(array, ref, ang_params=(0, 0, 1), thr=4):
    ali_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(array)))
    ref_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(ref)))
    angList, ccImgList = fourierInplaneAlignBF(ali_ft, ref_ft, ang_params, thr)
    return angList, ccImgList


def getAlignValues(angles, ccimg, shrink=1):
    new_img_list = []
    for img in ccimg:
        shape = img.shape
        new_shape = tuple(int(shrink * x) for x in shape)
        img2 = fr.cropArrayCentered(img, new_shape)
        new_img_list.append(img2)
    
    cc = [x.max() for x in new_img_list]
    pmax = np.argmax(cc)
    imgcc_th = new_img_list[pmax]
    mpos = np.unravel_index(imgcc_th.argmax(), imgcc_th.shape)
    shifts = mpos - np.array(imgcc_th.shape) // 2
    
    return angles[pmax], shifts[::-1]
