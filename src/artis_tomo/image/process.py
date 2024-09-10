# -*- coding: utf-8 -*-
"""Diverse processing functions for 2D/3D arrays"""

import numpy as np
import skimage
import skimage.filters.thresholding as sk_thres
import skimage.transform as sktransform
import skimage.segmentation as sksegmen
from scipy import ndimage

def norm(array, mode='pos'):

    minV = array.min()
    maxV = array.max()

    if mode == 'pos':
        return (array - minV)/(maxV - minV)
    elif mode == 'mean':
        return (array - array.mean())/(maxV - minV)
    else:
        raise Exception(f'Unkown mode {mode}')


def processMosaic(img_arr, thr=0.2):
    thrmin = thr * sk_thres.threshold_minimum(img_arr)
    thr_max = sk_thres.threshold_multiotsu(img_arr)
    img_arr[0,:] = 0
    img_arr[-1,:] = 0
    img_arr[:,0] = 0
    img_arr[:,-1] = 0
    mask = sksegmen.flood(img_arr,(0,0),  tolerance=thrmin)
    mask = skimage.morphology.dilation(mask, skimage.morphology.square(40))
    mask = np.logical_not(mask)
    boundLabels = skimage.measure.label(mask)
    mask = boundLabels == np.argmax(np.bincount(boundLabels.flat)[1:]) + 1

    img_arr[img_arr < 1] = 1
    imlog = norm(-1.*np.log(img_arr))
    mean = imlog[mask].mean()
    mosaic = imlog * mask + mean * np.logical_not(mask)
    return mosaic


def filterFeatures(img, r_max=20, r_min=6):
    im_th1 = filterTophat(img, r_max)
    im_th2 = filterTophat(img, r_min)
    im_th = im_th1 - im_th2
    return im_th

def filterTophat(img, r=20):
    return ndimage.white_tophat(img, size=r)