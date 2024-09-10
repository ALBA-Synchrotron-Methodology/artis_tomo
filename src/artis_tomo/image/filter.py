#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter tools in Real and Fourier space

@author: joton
"""

import numpy as np
from numba import njit, prange, objmode
from artis_tomo.math.function import raisedCos
from artis_tomo.math import fft
from artis_tomo.image import frame as fr
from artis_tomo.image.expand import extendRadSym


def maskRaisedCosineBorder2D(shape, pad):
    """
    Create a 2D rectangular mask with a raised cosine border.

    Parameters
    ----------
    shape : tuple
        Length in pixels of the output squared 2D array.
    pad :
        Legngth of the raised cosine profile applied at the borders.

    Returns
    -------
    arrayOut : 2D darray
        Output array of given shape.
    """
    ndim = len(shape)
    padSize = np.ones(ndim, dtype=int)*2*pad
    array = np.ones(np.array(shape)-padSize)

    return fr.padArrayCentered(array, shape)[0]


def maskrect(size, rs2, dx=1):
    """
    Create a centered 1D rectangle mask.

    Parameters
    ----------
    size : int
        Length in pixels of the output 1D array.
    rs2 : int
        Half rectangle size.
    dx : optional
        Pixel size.

    Returns
    -------
    arrayOut : 1D darray
        Output array of given size, one evaluated within the rectangle and
        zero elsewhere.
    """
    xV = np.arange(size) - size//2

    return xV**2 <= (rs2/dx)**2


def maskcirc(shape, radius, dx=1):
    """
    Create a 2D circular mask.

    Parameters
    ----------
    shape : array_like of rank N
        Length in pixels of the output squared 2D array.
    radius :
        Cut-off radius.
    dx : optional
        Pixel size.

    Returns
    -------
    arrayOut : 2D darray
        Output array of given shape, one evaluated for radius lower than
        cut-off radius and zero otherwise.
    """
    shape = np.asarray(shape)
    iC = (shape//2).astype(int)

    yV = (np.arange(shape[0]) - iC[0])
    xV = (np.arange(shape[1]) - iC[1])
    xx, yy = np.meshgrid(xV, yV)

    return (xx**2 + yy**2) <= (radius/dx)**2


def masksphere(shape, radius, dx=1):
    """
    Create a 3D spherical mask.

    Parameters
    ----------
    boxsize : array_like of rank N
        Length in pixels of the output squared 3D array.
    radius :
        Cut-off radius.
    dx : optional
        Pixel size.

    Returns
    -------
    arrayOut : 3D darray
        Output array of given boxsize, one evaluated for radius lower than
        cut-off radius and zero otherwise.
    """
    shape = np.array(shape)
    iC = np.floor(shape/2).astype(int)

    zV = (np.arange(shape[0]) - iC[0])
    yV = (np.arange(shape[1]) - iC[1])
    xV = (np.arange(shape[2]) - iC[2])
    xx, yy, zz = np.meshgrid(zV, yV, xV)

    return (xx**2 + yy**2 + zz**2) <= (radius/dx)**2


def maskRaisedCosineRadial(shape, radius, dx=1, pad=20):
    """
    Create a mask with a raised cosine border radially.

    Parameters
    ----------
    shape : tuple of length N
        Length in pixels of the output squared 2D array.
    rad :
        Length of the unitary profile before raised cosine.
    pad :
        Length of the raised cosine profile applied.

    Returns
    -------
    arrayOut : N dimensions array
        Output array of given shape.
    """
    nDim = len(shape)
    xDim2 = round(radius/dx) + 1
    if pad < 1:
        pad = 1
    profile1D = np.ones(xDim2)
    profile1D[xDim2 - pad:] = raisedCos(pad)[::-1]

    arrayNdim = extendRadSym(profile1D, nDim)

    return fr.padArrayCentered(arrayNdim, shape, 'constant')[0]


def lowPassFilter(array, dx, fc, edgewidth=10):
    """
    Apply a Fourier lowpass filter, using a rised cosine as smooth edge.

    Parameters
    ----------
    array : 1, 2 or 3D array
        Input array.
    dx : INT
        Pixel size.
    fc : FLOAT
        Cut-off frequency.
    edgewidth : INT, optional
        Width of the rised cosine of the edge. The default is 10.

    Returns
    -------
    1, 2 or 3D array
        Low pass filtered input distribution.
    """
    dims = array.shape
    nDims = len(dims)

    xDim = max(dims)
    dfx = 1/(xDim*dx)
    fcN = round(fc/dfx)

    pad = edgewidth

    arrayTmp = fr.padArrayCentered(array, (xDim,)*nDims)[0]
    ftArray = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(arrayTmp)))

    if nDims == 1:
        filt = np.ones(fcN*2-1)
        rc = raisedCos(pad)
        filt[:pad] = rc
        filt[-pad:] = rc[::-1]
        filt = fr.padArrayCentered(filt, xDim, 'constant')[0]
    else:
        filt = maskRaisedCosineRadial((xDim,)*nDims, fcN, 1, pad)

    arrayOut = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ftArray*filt)))

    if array.dtype == complex:
        if arrayOut.shape == dims:
            return arrayOut
        else:
            return fr.cropArrayCentered(arrayOut, dims)
    else:
        if arrayOut.shape == dims:
            return arrayOut.real
        else:
            return fr.cropArrayCentered(arrayOut.real, dims)


def gaussianFilter(array, dx, sigma):
    """
    Apply a Fourier lowpass filter, using a rised cosine as smooth edge.

    Parameters
    ----------
    array : 1, 2 or 3D array
        Input array.
    dx : INT
        Pixel size.
    sigma : FLOAT
        Real space std deviation. In Fourier space sigmaFS² = 1/(4*pi*sigmaRS²)
    edgewidth : INT, optional
        Width of the rised cosine of the edge. The default is 10.

    Returns
    -------
    1, 2 or 3D array
        Low pass filtered input distribution.
    """
    dims = array.shape
    nDims = len(dims)

    xDim = max(dims)
    dfx = 1/(xDim*dx)
    sigmaFS = 1/(2*np.sqrt(np.pi)*sigma)
    sigmaN = round(sigmaFS/dfx)

    # Coordinates gridding
    XXref = [np.arange(xDim) - xDim//2]
    XXmesh = np.meshgrid(*(XXref*nDims), copy=False, indexing='ij')

    radii2 = np.zeros(XXmesh[0].shape)
    for k in range(nDims):
        radii2 += XXmesh[k]**2

    gaussF = np.exp(-0.5*radii2/sigmaN**2)

    arrayTmp = fr.padArrayCentered(array, (xDim,)*nDims)[0]
    ftArray = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(arrayTmp)))

    arrayOut = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ftArray*gaussF)))

    if array.dtype == complex:
        if arrayOut.shape == dims:
            return arrayOut
        else:
            return fr.cropArrayCentered(arrayOut, dims)
    else:
        if arrayOut.shape == dims:
            return arrayOut.real
        else:
            return fr.cropArrayCentered(arrayOut.real, dims)


def griddingCorrect(array, gridCoords=None):

    if gridCoords is None:
        size = np.array(array.shape)
        sizeh = size//2
        XX = list()
        for k in range(array.ndim):
            XX.append(np.arange(size[k]) - sizeh[k])

        gridCoordsArray = np.array(np.meshgrid(*XX, copy=False))
    else:
        gridCoordsArray = np.array(gridCoords)

    ro = np.sqrt((gridCoordsArray**2).sum(axis=0))
    ro /= min(size)
    sinc2 = (np.sin(np.pi * ro) / (np.pi * ro))**2
    eps = 1e-2

    maski = np.logical_or(sinc2 < eps, ro > 1.0)
    mask = np.invert(maski)
    mask[tuple(sizeh)] = False

    arrayOut = array.copy()

    arrayOut[maski] = arrayOut[maski] / eps
    arrayOut[mask] = arrayOut[mask] / sinc2[mask]

    return arrayOut


def normalizeBg(imSS, cutoff=13, taper=9):
    """
    Normalize background projections.

    It estimates the background pattern by applying a circular raisedcosined
    edge lowpass filter to the own projection.

    Parameters
    ----------
    imSS : 2D or 3D array
        Image or stack of images.
    cutoff : Int , optional
        Fourier Cutoff frequency in Pixels. The default is 13.
    taper : Int, optional
        Length of the tapered range before cutoff. The default is 9.

    Returns
    -------
    imSSOut : 2D or 3D array
        Background normalised image or stack of images

    """
    imShape = imSS.shape
    ndim = len(imShape)

    mask = maskRaisedCosineRadial(imShape[-2:], cutoff, pad=taper)

    imSSOut = imSS.astype(float)
    if ndim == 2:
        imSSOut = imSSOut[np.newaxis, :]

    _normalizeBg(imSSOut, mask)

    if ndim == 2:
        imSSOut = np.squeeze(imSSOut)

    return imSSOut


@njit(parallel=True)
def _normalizeBg(imSS, mask):

    for k in prange(len(imSS)):
        im = imSS[k]
        # with objmode(imft='complex128[:, :]'):
        with objmode(bg='float64[:, :]'):
            imft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(im)))
            bg = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(imft*mask))).real
        imTot = im.sum()
        im = im / bg
        im = im * imTot/im.sum()

        imSS[k] = im


def wienerFilter(array, psf, SNR=1.e3):

    iShape = array.shape

    if iShape != psf.shape:
        psf = fr.padCropArrayCentered(psf, iShape)[0]

    aft = fft.rfftn(array)
    mtf = fft.rfftn(psf)

    deconvft = aft*mtf.conj()/(mtf*mtf.conj() + 1/SNR)

    decon = fft.fftshift(fft.irfftn(deconvft))

    return decon


def _statFilterKernel(stat1, stat2, array, maskft, padSize, cropSize,
                      doDisp=False):
    """
    Apply a statistical filter to compute mean, variance, and optionally dispersion.

    Parameters
    ----------
    stat1 : array_like
        Array to store the primary statistic. It holds the mean if `doDisp` is False,
        and the dispersion if `doDisp` is True.
    stat2 : array_like
        Array to store the secondary statistic. It holds the variance if `doDisp` is False.
        This parameter is ignored if `doDisp` is True.
    array : array_like
        Input array.
    maskft : array_like
        Fourier transformed mask used for convolution.
    padSize : tuple of int
        Size of the padded array.
    cropSize : tuple of int
        Size of the cropped array after filtering.
    doDisp : bool, optional
        If True, compute and store the dispersion (variance/abs(mean)) in `stat1`.
        If False, compute and store the mean in `stat1` and the variance in `stat2`.

    Notes
    -----
    This function applies a statistical filter to the input array(s), computing
    the mean and variance within a window specified by the Fourier transformed
    mask `maskft`. Optionally, it can also compute the dispersion, defined as the ratio
    of the variance to the absolute value of the mean.

    The algorithm involves padding the input array(s), performing FFT-based convolution
    to compute the mean and variance, and cropping the result to the original array size.
    When computing dispersion, mean values of zero are replaced with a small value (1.e-4)
    to avoid division by zero.

    """
    for k, profile in enumerate(array):
        arrayPad = fr.padArrayCentered(profile.astype('float'),
                                       padSize, mode='reflect')[0]

        imft = np.fft.rfftn(np.fft.ifftshift(arrayPad))
        imtmp = np.fft.fftshift(np.fft.irfftn(imft*maskft))
        mean = fr.cropArrayCentered(imtmp, cropSize)

        imft = np.fft.rfftn(np.fft.ifftshift(arrayPad**2))
        mean2 = np.fft.fftshift(np.fft.irfftn(imft*maskft))
        mean2 = fr.cropArrayCentered(mean2, cropSize)

        var = mean2 - mean**2

        if doDisp:
            mean[mean == 0] = 1.e-4
            stat1[k] = var/np.abs(mean)
        else:
            stat1[k] = mean
            stat2[k] = var


def statFilter1D(array, hsize):
    """
    Apply a 1D statistical filter with a rectangular kernel specified by rsize.

    Parameters
    ----------
    array : array_like
        Input array of rank 1 (profile) or rank 2 (stack of 1D profiles).
    hsize : int
        The half size of the filter window.

    Returns
    -------
    mean : array_like
        Array or stack of arrays containing the mean values computed from the
        input array(s).
    var : array_like or list of array_like
        Array or stack of arrays containing the variance values computed from
        the input array(s).

    Notes
    -----
    This function applies a statistical filter to the input array, computing
    the variance and mean within a rect window of half size hsize for each
    element of the array.

    The algorithm involves padding the input array, creating a rect mask
    of hal size hsize, and performing FFT-based convolution to efficiently
    compute the statistical properties. The output arrays mean and var contain
    the mean and variance values, respectively.

    """
    iSize = np.array(array.shape[-1:])
    iDim = array.ndim

    if iDim == 1:
        array = array[np.newaxis, ...]

    padSize = (iSize+hsize*2,)

    # Create the mask
    mask = maskrect(padSize[0], hsize)
    mask = mask/mask.sum()
    maskft = np.fft.rfftn(np.fft.ifftshift(mask))

    mean = np.empty_like(array)
    var = np.empty_like(array)

    _statFilterKernel(mean, var, array, maskft, padSize, iSize)

    if iDim == 1:
        mean = mean[0]
        var = var[0]

    return mean, var


def statFilter2D(array, rsize):
    """
    Apply a 2D statistical filter with a circular kernel specified by rsize.

    Parameters
    ----------
    array : array_like
        Input array of rank 2 (2D image) or rank 3 (stack of 2D images).
    rsize : int
        The radius of the filter window.

    Returns
    -------
    mean : array_like
        Array or stack of arrays containing the mean values computed from the
        input array(s).
    var : array_like
        Array or stack of arrays containing the variance values computed from
        the input array(s).

    Notes
    -----
    This function applies a statistical filter to the input array, computing
    the variance and mean within a circular window of radius rsize for each
    element of the array.

    The algorithm involves padding the input array, creating a circular mask
    of radius rsize, and performing FFT-based convolution to efficiently
    compute the statistical properties. The output arrays mean and var contain
    the mean and variance values, respectively.

    """
    iSize = np.array(array.shape[-2:])
    iDim = array.ndim
    smax = max(iSize)

    if iDim == 2:
        array = array[np.newaxis, ...]

    padSize = (smax+rsize*2,)*2

    # Create the mask
    mask = maskcirc(padSize, rsize)
    mask = mask/mask.sum()
    maskft = np.fft.rfftn(np.fft.ifftshift(mask))

    mean = np.empty_like(array)
    var = np.empty_like(array)

    _statFilterKernel(mean, var, array, maskft, padSize, iSize)

    if iDim == 2:
        mean = mean[0]
        var = var[0]

    return mean, var


def statFilter3D(array, rsize):
    """
    Apply a 3D statistical filter with a circular kernel specified by rsize.

    Parameters
    ----------
    array : array_like
        Input array of rank 3.
    rsize : int
        The radius of the filter window.

    Returns
    -------
    mean : array_like
        Array containing the mean values computed from the input array.
    var : array_like
        Array containing the variance values computed from the input array.

    Notes
    -----
    This function applies a statistical filter to the input array, computing
    the variance and mean within a circular window of radius rsize for each
    element of the array.

    The algorithm involves padding the input array, creating a circular mask
    of radius rsize, and performing FFT-based convolution to efficiently
    compute the statistical properties. The output arrays mean and var contain
    the mean and variance values, respectively.

    """
    iSize = np.array(array.shape)
    iDim = array.ndim
    smax = max(iSize)

    if iDim == 3:
        array = array[np.newaxis, ...]

    padSize = (smax+rsize*2,)*3

    # Create the mask
    mask = masksphere(padSize, rsize)
    masksum = mask.sum()
    mask = mask/masksum
    maskft = np.fft.rfftn(np.fft.ifftshift(mask))

    mean = np.empty_like(array)
    var = np.empty_like(array)

    _statFilterKernel(mean, var, array, maskft, padSize, iSize)

    if iDim == 3:
        mean = mean[0]
        var = var[0]

    return mean, var


def dispersionFilter1D(array, hsize):
    """
    Apply a 1D dispersion filter with a rectangular kernel specified by rsize.

    Parameters
    ----------
    array : array_like
        Input array of rank 1 (profile) or rank 2 (stack of 1D profiles).
    hsize : int
        The half size of the filter window.

    Returns
    -------
    mean : array_like
        Array or stack of arrays containing the mean values computed from the
        input array(s).
    var : array_like or list of array_like
        Array or stack of arrays containing the variance values computed from
        the input array(s).

    Notes
    -----
    This function applies a statistical filter to the input array, computing
    the variance and mean within a rect window of half size hsize for each
    element of the array.

    The algorithm involves padding the input array, creating a rect mask
    of hal size hsize, and performing FFT-based convolution to efficiently
    compute the statistical properties. The output arrays mean and var contain
    the mean and variance values, respectively.

    """
    iSize = np.array(array.shape[-1:])
    iDim = array.ndim

    if iDim == 1:
        array = array[np.newaxis, ...]

    padSize = (iSize+hsize*2,)

    # Create the mask
    mask = maskrect(padSize[0], hsize)
    mask = mask/mask.sum()
    maskft = np.fft.rfftn(np.fft.ifftshift(mask))

    dispersion = np.empty_like(array)

    _statFilterKernel(dispersion, None, array, maskft, padSize, iSize,
                      doDisp=True)

    if iDim == 1:
        dispersion = dispersion[0]

    return dispersion


def dispersionFilter2D(array, rsize):
    """
    Apply a 2D dispersion filter with a circular kernel specified by rsize.

    Parameters
    ----------
    array : array_like
        Input array of rank 2 (2D image) or rank 3 (stack of 2D images).
    rsize : int
        The radius of the filter window.

    Returns
    -------
    dispersion : array_like
        Array or stack of arrays containing the dispersion values computed from
        the input array(s).

    Notes
    -----
    This function applies a dispersion filter to the input array(s), computing
    the dispersion within a circular window of radius rsize for each element
    of the array.

    The algorithm involves padding the input array(s), creating a circular mask
    of radius rsize, and performing FFT-based convolution to efficiently
    compute the statistical properties. The dispersion is defined as the ratio
    of the variance to the absolute value of the mean within the filter window.
    """
    iSize = np.array(array.shape[-2:])
    iDim = array.ndim
    smax = max(iSize)

    if iDim == 2:
        array = array[np.newaxis, ...]

    padSize = (smax+rsize*2,)*2

    # Create the mask
    mask = maskcirc(padSize, rsize)
    mask = mask/mask.sum()
    maskft = np.fft.rfftn(np.fft.ifftshift(mask))

    dispersion = np.empty_like(array)

    _statFilterKernel(dispersion, None, array, maskft, padSize, iSize,
                      doDisp=True)

    if iDim == 2:
        dispersion = dispersion[0]

    return dispersion


def dispersionFilter3D(array, rsize):
    """
    Apply a 3D dispersion filter with a spherical kernel specified by rsize.

    Parameters
    ----------
    array : array_like
        Input array of rank 3 (3D volume) or rank 4 (stack of 3D volumes).
    rsize : int
        The radius of the filter window.

    Returns
    -------
    dispersion : array_like
        Array containing the dispersion values computed from the input array.

    Notes
    -----
    This function applies a dispersion filter to the input 3D array, computing
    the dispersion within a spherical window of radius rsize for each element
    of the array.

    The function first calculates the mean and variance using a spherical
    kernel by calling `statsFilter3D`. The dispersion is then defined as the
    ratio of the variance to the absolute value of the mean within the filter
    window. To avoid division by zero, mean values of zero are replaced with one.

    The algorithm relies on the `statsFilter3D` function to compute the mean
    and variance within the spherical window.

    Example
    -------
    >>> array = np.random.random((10, 10, 10))
    >>> rsize = 3
    >>> dispersion = dispersionFilter3D(array, rsize)
    """
    iSize = np.array(array.shape)
    iDim = array.ndim
    smax = max(iSize)

    if iDim == 3:
        array = array[np.newaxis, ...]

    padSize = (smax+rsize*2,)*3

    # Create the mask
    mask = masksphere(padSize, rsize)
    masksum = mask.sum()
    mask = mask/masksum
    maskft = np.fft.rfftn(np.fft.ifftshift(mask))

    dispersion = np.empty_like(array)

    _statFilterKernel(dispersion, None, array, maskft, padSize, iSize,
                      doDisp=True)

    if iDim == 3:
        dispersion = dispersion[0]

    return dispersion


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

    # from .conversions import ind2sub
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
            # pixCoord = ind2sub(shape, ind)
            pixCoord = np.array(np.unravel_index(ind, shape)).T

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