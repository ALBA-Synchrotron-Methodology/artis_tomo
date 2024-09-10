#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read images saved in FITS file format


@author: joton
"""

try:
    from astropy.io import fits
except ImportError:
    pass

import numpy as np


__all__ = ["readFITS"]


def checkDT(data):
    if data.dtype == 'uint16':
        return data.astype(np.int16)
    return data

def readFITS(filepath):
    hdulist = fits.open(filepath,  ignore_missing_end=True) # Read the fits file
    try:
        hdulist.verify('silentfix') # Attempt to fix HDU errors (if present)
    except:
        print('----- WARNING: HDU has unfixable errors')
    data = hdulist['Primary'].data  # image data
    data = checkDT(data)
    header = hdulist['Primary'].header # Header data

    if  (np.shape(hdulist)[0] > 1): # If there is more than one HDU
        zero_bck = False
        if  (np.shape(hdulist)[0] > 1): # If there is more than one HDU
            if (type(hdulist[1]) == fits.hdu.image.ImageHDU):
                background = hdulist[1].data  # Set the second HDU as a background
                background = checkDT(background)
                print('Background found')
            else:
                print('----- WARNING: HDU has no background saved')
                zero_bck = True
        else: zero_bck = True # No background is found
        if zero_bck is True: # If there is no background
            dimensions = np.ndim(data)  # Check the shape of the data (e.g. image stack or single frame)
            if dimensions == 3:  background = np.zeros(data[0,:,:].shape) # Zero background
            elif dimensions == 2: background = np.zeros(data[:,:].shape) # Set the background frame to zeros

        data = np.subtract(data, background) # Subtract background from image
        data[data<0] = 0

    return data