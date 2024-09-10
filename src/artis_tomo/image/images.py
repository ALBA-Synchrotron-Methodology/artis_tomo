#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read and preprocess images

@authors: Josue Gomez & Joaquin Oton
"""
import numpy as np
import matplotlib.pyplot as plt


def readImage(fname):
    return plt.imread(fname)


def readImage2D(fname):
    img_arr = readImage(fname)
    if img_arr.ndim == 3:
        img_arr = img_arr[:, :, 0]
    elif img_arr.ndim > 3:
        raise ("Error: You must enter only 2D or 3D image files")
    return img_arr


def readImage3D(fname):
    img_arr = readImage(fname)
    if img_arr.ndim == 3:
        img_arr = img_arr[:,:,:]
    else:
        raise ("Error: You must enter 3D images only")
    return img_arr


def readDVimage(fname):
    pass

