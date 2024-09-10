"""
The filter module.

It contains a group of functions for filtering and padding either in real
or Fourier space.

THIS FILE IS TO BE REMOVED

@author: joton
"""

import numpy as np
from . import transformGeometry as tg
# import cv2
from scipy import interpolate as ip

from ..image.frame import (padwithrc, padArrayCentered, cropArrayCentered,
                           padCropArrayCentered)






def array2complex(array):

    out = 1j*array[..., 1]
    out += array[..., 0]
    return out


def complex2array(array):

    out = np.empty(array.shape + (2, ))
    out[..., 0] = array.real
    out[..., 1] = array.imag
    return out


# def fft(array):
#     """
#     Performs the forward Discrete Fourier transform of a 1D or 2D
#     floating-point array
#
#     Parameters
#     ----------
#     arrayIn : array_like of rank 1,2
#         Input array
#
#     Returns
#     -------
#     arrayOut : array_like of rank 1,2
#         Output array
#     """
#
#     aArray = complex2array(array)
#
#     aArrayft = cv2.dft(aArray, flags=cv2.DFT_COMPLEX_OUTPUT)
#
#     return array2complex(aArrayft)


# def ifft(array):
#     """
#     Performs the inverse Discrete Fourier transform of a 1D or 2D
#     floating-point array
#
#     Parameters
#     ----------
#     arrayIn : array_like of rank 1,2
#         Input array
#
#     Returns
#     -------
#     arrayOut : array_like of rank 1,2
#         Output array
#     """
#     aArrayft = complex2array(array)
#
#     aArray = cv2.idft(aArrayft, flags=cv2.DFT_COMPLEX_OUTPUT+cv2.DFT_SCALE)
#
#     return array2complex(aArray)
#
#
# def convFourier(array1, array2):
#     """
#     Performs the convolution operation in Fourier space between two 1D or 2D
#     floating-point arrays
#
#     Parameters
#     ----------
#     array1 : array_like of rank 1,2
#         First input array
#     array2 : array_like of rank 1,2
#         Second input array
#
#     Returns
#     -------
#     arrayOut : array_like of rank 1,2
#         Output array
#     """
#
#     aA1 = complex2array(np.fft.ifftshift(array1))
#     aA2 = complex2array(np.fft.ifftshift(array2))
#
#     aA1ft = cv2.dft(aA1, flags=cv2.DFT_COMPLEX_OUTPUT)
#     aA2ft = cv2.dft(aA2, flags=cv2.DFT_COMPLEX_OUTPUT)
#
#     aC1ft = array2complex(aA1ft)
#     aC2ft = array2complex(aA2ft)
#
#     outAft = complex2array(aC1ft*aC2ft/(aC1ft.abs()*aC2ft.abs()))
#
#     outA = np.fft.fftshift(cv2.idft(outAft,
#                                     flags=cv2.DFT_COMPLEX_OUTPUT +
#                                     cv2.DFT_SCALE))
#
#     return array2complex(outA)

def crosscorFourier(array1, array2):
    """
    Performs the convolution operation in Fourier space between two 1D or 2D
    floating-point arrays

    Parameters
    ----------
    array1 : array_like of rank 1,2
        First input array
    array2 : array_like of rank 1,2
        Second input array

    Returns
    -------
    arrayOut : array_like of rank 1,2
        Output array
    """

    aA1 = np.fft.ifftshift(array1 - np.mean(array1))
    aA2 = np.fft.ifftshift(array2 - np.mean(array2))

    aC1ft = np.fft.fft2(np.fft.ifftshift(aA1))
    aC2ft = np.fft.fft2(np.fft.ifftshift(aA2))

    outAft = aC1ft*np.conj(aC2ft)/np.sqrt(aC1ft*np.conj(aC1ft)*aC2ft*np.conj(aC2ft))

    out = np.fft.fftshift(np.fft.ifft2(outAft))

    return out.real


def boundMedianFilter(array, maskini):
    """
    Apply a median filter calculated from the sourronding pixels only for those
    pixels given by the mask

    Parameters
    ----------
    array : array_like of rank 2,3
        input array
    mask : array_like of rank 2,3
        pixels where filter is applied

    Returns
    -------
    arrayOut : array_like of same inpu array rank
        Output array
    """

    from .conversions import ind2sub
    from itertools import product

    mask = np.copy(maskini)

    shape = mask.shape

    neighbours = []

    # Relative coordinate list from -2 to 2 around the dimensions of array
    # If you think for a while you'll realize how it works
    coorList = np.array(list(product(range(-2, 3), repeat=len(shape))))

    arrayOut = np.copy(array)

    while mask.any():
        mInd = np.where(mask.flatten())[0]

        # Each pixel in the mask
        for ind in mInd:
            pixCoord = ind2sub(shape, ind)

            # Each neigbour pixel
            for nCor in coorList:
                arrayCoord = pixCoord + nCor

                if (arrayCoord < 0).any() | \
                   (arrayCoord >= np.array(shape)).any():
                    continue

                absCoord = tuple(arrayCoord.flatten())

                if mask[absCoord] == 0:
                    neighbours.append(arrayOut[absCoord])

            if len(neighbours) > 0:
                pixCoordTuple = tuple(pixCoord.flatten())
                arrayOut[pixCoordTuple] = np.median(neighbours)
                mask[pixCoordTuple] = 0
                neighbours.clear()

    return arrayOut


def varianceFilter(array, wsize):
    """
    Apply a variance filter for a window size given by wsize

    Parameters
    ----------
    array : array_like of rank 2,3
        input array
    wsize : int
        window size

    Returns
    -------
    arrayOut : array_like of same input array rank
        Output array
    """
    from scipy import ndimage

    win_mean = ndimage.uniform_filter(array, (wsize,)*2)
    win_sqr_mean = ndimage.uniform_filter(array**2, (wsize,)*2)

    return win_sqr_mean - win_mean**2


def varianceFilterNorm(array, wsize):
    """
    Apply a variance filter for a window size given by wsize

    Parameters
    ----------
    array : array_like of rank 2,3
        input array
    wsize : int
        window size

    Returns
    -------
    arrayOut : array_like of same input array rank
        Output array
    """
    from scipy import ndimage

    win_mean = ndimage.uniform_filter(array, (wsize,)*2)
    win_sqr_mean = ndimage.uniform_filter(array**2, (wsize,)*2)

    return (win_sqr_mean - win_mean**2)/win_sqr_mean



def varianceFilter3D(array, rsize):
    """
    Apply a 3D variance filter for a window size given by rsize

    Parameters
    ----------
    array : array_like of rank 3
        input array
    wsize : int
        window size

    Returns
    -------
    arrayOut : array_like of same input array rank
        Output array
    """
    iSize = np.array(array.shape)
    # Create the mask
    mask = masksphere(rsize*2+1, rsize)
    mask = mask/mask.sum()
    mask = padArrayCentered(mask, iSize)[0]


    volft = np.fft.fftn(np.fft.ifftshift(array))
    maskft = np.fft.fftn(np.fft.ifftshift(mask))
    meanfilt = np.fft.fftshift(np.fft.ifftn(volft*maskft))

    volft = np.fft.fftn(np.fft.ifftshift(array**2))
    mean2filt = np.fft.fftshift(np.fft.ifftn(volft*maskft))
    varfilt = mean2filt - meanfilt**2

    return varfilt.real, meanfilt.real


# def doseFilterPattern(dose, freqArray):
#     """Apply Standard B-factor correction."""
#     output = np.exp(-dose/4*freqArray*freqArray)

#     return output

def doseFilterPattern(dose, freqArray):
    """ The filter is as described in Grant and Grigorieff, 2015
    (DOI:10.7554/eLife.06980) and the implementation follows that in their
    "unblur" program. """

    a = 0.245
    b = -1.665
    c = 2.81

    output = np.ones(freqArray.shape)
    mask = freqArray != 0
    output[mask] = np.exp(-dose / (2 * ((a * freqArray[mask]**b) + c)))

    return output
