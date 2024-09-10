#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply coordinate system transformations to images and volumes

@author: joton
"""

import numpy as np
import math
from scipy import interpolate
from numba import njit, prange, objmode
from artis_tomo.math import fft
from artis_tomo.math.coordinates import pol2cart
from artis_tomo.math.transforms import TMat3D, TMat2D, tr2d
from artis_tomo.image import filter as ft
from artis_tomo.image import frame as fr


__all__ = ['transformRS',
           'transformRS2D',
           'transform',
           'transform2D',
           'convertToPolar'
           ]


def transformRS(array: np.ndarray, R: np.ndarray,
                Fourier=False, dimOrder=None):
    """
    Apply a transformation matrix to a 2D/3D array in Real space.

    Parameters
    ----------
    array : ndarray
        2D/3D input array.
    R : ndarray
        3x3 or 4x4 augmented matrix with the affine transformations.
    Fourier : bool, optional
        Asume input array is half Fourier pattern. The default is False.
    dimOrder : integers list, optional
        Order of array's dimensions. By default, X is last, Y last-1 ... so in
        2D dimOrder=[1, 0] and in 3D [2, 1, 0].

    Returns
    -------
    aRot : ndarray
        2d/3d array with transformed pattern.

    """
    ndim = array.ndim
    iSize = np.array(array.shape)
    vC = iSize//2  # Rotation center position

    if ndim != (R.shape[0] - 1):
        raise ValueError(f"Transformation matrix dimension {R.shape[0]} does"
                         f" not match array dimensions {ndim} + 1.")

    if dimOrder is None:
        dimOrder = np.arange(ndim)[::-1]  # [2, 1, 0]

    if Fourier:
        vC[dimOrder[0]] = 0

    idimOrd = np.argsort(dimOrder)  # inverse order

    # Input coordinates for interpolation
    XXref = [None]*ndim

    for k in range(ndim):
        XXref[k] = np.arange(iSize[k]) - vC[k]  # original dimensions

    # X, Y, Z coordinates of the rotated volume are the same as input. We guess
    # We're just rotating not translating
    XXmesh = np.meshgrid(*XXref, copy=False, indexing='ij')
    # We flatten by default in order='C', which match with the same default
    # order use in the reshape step after interp below
    XXoutF = [None]*(ndim+1)
    for k in range(ndim):
        XXoutF[k] = XXmesh[idimOrd[k]].ravel()
    XXoutF[-1] = np.ones(XXoutF[0].size)  # Augmented vector

    Rinv = R.I
    rotCoord = (Rinv@np.row_stack(XXoutF)).A.T

    if Fourier:
        xneg = rotCoord[:, 0] < 0
        rotCoord[xneg, :] *= -1

    rotArray = interpolate.interpn(XXref, array,
                                   rotCoord[:, dimOrder],
                                   bounds_error=False,
                                   fill_value=array.flat[0],
                                   method='linear')

    # if Fourier:
        # rotArray[xneg] = rotArray[xneg].conj()

    return rotArray.reshape(iSize)


def transform(array, R, padfactor=2, edge=-1, end_values=0, dimOrder=None):
    """
    Apply a transformation matrix to a 2D/3D array in Fourier space.

    Parameters
    ----------
    array : ndarray
        2D/3D input array.
    R : ndarray
        3x3 or 4x4 augmented matrix with the affine transformations.
    padfactor : float, optional
        Padding factor before Fourier transform. The default is 2.
    edge : int, optional
        Rised cosine edge length of Fourier mask. Not applied if negative.
        The default is -1.
    dimOrder : integers list, optional
        Order of array's dimensions. By default, X is last, Y last-1 ... so in
        2D dimOrder=[1, 0] and in 3D [2, 1, 0].

    Returns
    -------
    aRot : ndarray
        2d/3d array with transformed pattern.

    """
    ndim = array.ndim
    iSize = np.array(array.shape)
    iSizeh = iSize//2  # Rotation center position

    sizePad = int(np.max(iSize)*padfactor)
    sizePadND = (sizePad,)*ndim
    sizePadh = sizePad//2

    if dimOrder is None:
        dimOrder = np.arange(ndim)[::-1]  # [2, 1, 0]

    if end_values == 'mean':
        end_values = array.mean()

    rcpad = sizePad//32
    arrayPad = fr.padArrayCentered(array, sizePadND,
                                   end_values=end_values, rcpad=rcpad)[0]

    aFT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arrayPad)))

    if edge > 0:
        mask = ft.maskRaisedCosineRadial(sizePadND, sizePadh, pad=edge)
        aFT *= mask

    phaseShift = getPhaseShiftFourier(sizePadND, R[dimOrder, -1].A)

    R = R.copy()
    R[:-1, -1] = 0  # Only apply rotations to Fourier pattern

    aFTrot = transformRS(aFT, R, dimOrder=dimOrder)*phaseShift

    aRot = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(aFTrot))).real
    aRot = ft.griddingCorrect(aRot)
    aRot = fr.cropArrayCentered(aRot, iSize)

    return aRot


def getPhaseShiftFourier(shape, shifts):
    """
    Create a Fourier phase shift pattern in 2D/3D.

    Same sized dimensions are assumed.

    Parameters
    ----------
    shape : tuple
        Length in pixels of the output 2D/3D output array.
    shifts : floats list or 1d array
        Shifts in pixels applied to each dimension.

    Returns
    -------
    phaseShift : ndarray
        2d/3d array with phase shift pattern.

    """
    ndim = len(shape)
    size = shape[0]
    sizeh = np.asarray(size)//2

    xV = [np.arange(size) - sizeh]
    XX = np.meshgrid(*(xV*ndim), copy=False, indexing='ij')
    phaseShift = np.zeros(shape)
    for k in range(ndim):
        phaseShift += shifts[k]/size*XX[k]

    phaseShift = np.exp(-1j*2*np.pi*phaseShift)
    return phaseShift

def transformRS2D(array: np.ndarray, tmat, shifts=[0, 0],
                  Fourier=False, dimOrder=None):
    """
    Apply rotation and shifts to a 2D array in Real space.

    Parameters
    ----------
    array : ndarray
        2D input array.
    tmat : TMat3D
        3x3 or 4x4 augmented matrix with the affine transformations.
    shifts: floats list or 1d array
        X and Y shifts. Default is 0.
    Fourier : bool, optional
        Asume input array is half Fourier pattern. The default is False.
    dimOrder : integers list, optional
        Order of array's dimensions. By default, X is last, Y last-1 ... so in
        2D dimOrder=[1, 0] and in 3D [2, 1, 0].

    Returns
    -------
    aRot : ndarray
        2d array with transformed pattern.

    """
    m = tmat.matrix.copy()
    if tmat.n == 3:
        m = np.delete(m, 2, 1)
        m = np.delete(m, 2, 2)
    m_ang = TMat2D(m)
    m_sft = tr2d.shifts2mat(shifts)
    A = m_sft * m_ang
    m_arr = np.matrix(np.squeeze(A.matrix))
    return transformRS(array, m_arr, Fourier, dimOrder)


def transform2D(array: np.ndarray, tmat, shifts=[0, 0],
                padfactor=2, edge=-1, end_values=0, dimOrder=None):
    """
    Apply rotation and shifts to a 2D array in Fourier space.

    Parameters
    ----------
    array : ndarray
        2D input array.
    tmat : TMat3D
        3x3 or 4x4 augmented matrix with the affine transformations.
    shifts: floats list or 1d array
        X and Y shifts. Default is 0.
    padfactor : float, optional
        Padding factor before Fourier transform. The default is 2.
    edge : int, optional
        Rised cosine edge length of Fourier mask. Not applied if negative.
        The default is -1.
    dimOrder : integers list, optional
        Order of array's dimensions. By default, X is last, Y last-1 ... so in
        2D dimOrder=[1, 0] and in 3D [2, 1, 0].

    Returns
    -------
    aRot : ndarray
        2d array with transformed pattern.

    """
    if tmat.matrix.ndim == 3:
        tmat.matrix = np.delete(tmat.matrix, 2, 1)
        tmat.matrix = np.delete(tmat.matrix, 2, 2)
    m_sft = tr2d.shifts2mat(shifts)
    A = m_sft * tmat
    m_arr = A.matrix

    return transform(array, m_arr, padfactor, edge, end_values, dimOrder)

def getCenterSym(im):
    """ In the case of patterns with circular or spherical symmetry, we can
        find the center of the image/volume by flipping the axes and
        autoconvolving with its original function.

    Parameters
    ----------
    im : array_like of rank 1,2,3
        Input array

    Returns
    -------
    arrayOut : array_like of rank 1
        Center coordinates
    """
    from scipy.ndimage import zoom
    from scipy import signal

    imShape = np.array(im.shape)
    cv = np.floor(np.array(imShape)/2).astype(int)

    # We take largest squared ROI to detect centers close to edges
    cSize = int(np.min(imShape)//2)

    i0 = im[cv[0]-cSize:cv[0]+cSize, cv[1]-cSize:cv[1]+cSize]
    i0 = i0 - np.mean(i0)

    sf = 2  # Sampling factor to increase resolution

    if sf != 1:
        i0 = zoom(i0, sf)

    # Compute the crosscorrelation between the image and the image rotated
    # by 180 degrees (in order to find the center of the pattern later on).

    # Convolution operation implies inversion of axis in second args,
    # so it is the correlation operation with the axis inversion in 2nd arg.
    cc = signal.fftconvolve(i0, i0, 'same')

    # Position of the crosscorrelation maximum value
    peakPos = np.array(np.unravel_index(cc.argmax(), cc.shape))

    return (peakPos-cSize*sf)/sf/2+cv

def convertToPolar(array, center=None, nTheta=None, nR= None, logRadius=False):
    """
    Convert a stack of 2D images to polar coordinates.

    Parameters
    ----------
    array : 3D/2D array
        Stack of 2D images.
    center : tuple, optional
        X,Y coordinates of polar reference center. The default is None.
    nTheta : int, optional
        Dimension of angular axis. By default, pixel resolution is kept at
        largest radius.

    Returns
    -------
    arrayOut : 3D/2D
        Stack of 2D polar distribution images.

    """
    if array.ndim == 2:
        array = array[None, :, :]

    ni, ny, nx = array.shape

    if center is None:
        xC = nx//2
        yC = ny//2
    else:
        yC, xC = center

    xV = np.arange(nx) - xC
    yV = np.arange(ny) - yC

    rMax = int(np.min(np.abs([xV[[0, -1]], yV[[0, -1]]])))

    if nTheta is None:
        nTheta = round(2*np.pi*rMax)
        nTheta -= nTheta % 2
    else:
        if nTheta % 2 != 0:
            raise ValueError(f'Theta dimension size {nTheta} is not even.')

    if nR is None:
        nR = rMax
    else:
        rMax = nR

    # Polar coordinates grid
    # We exclude last pixel for 2pi, as we already have pixel for 0 degrees
    thetaV = np.linspace(0, 2*np.pi, nTheta + 1)[:-1]

    rV = np.linspace(0, rMax-1, nR)

    if logRadius:
        rV = np.exp(rV*np.log(rV[-1])/rV[-1])

    theta, ro = np.meshgrid(thetaV, rV)
    outShape = theta.shape

    xxp, yyp = pol2cart(theta, ro)
    yxp = np.row_stack((yyp.ravel(), xxp.ravel())).T

    arrayOut = np.empty((ni,) + outShape)

    _convertToPolar(yV, xV, array, yxp, arrayOut)

    arrayOut = np.squeeze(arrayOut)
    return arrayOut


@njit(parallel=True)
def _convertToPolar(yV, xV, array, yxp, arrayOut):

    nr = array.shape[0]
    outShape = arrayOut.shape[1:]

    for k in prange(nr):
        with objmode(img='float64[:, :]'):
            img = interpolate.interpn((yV, xV), array[k], yxp,
                                      bounds_error=False,
                                      fill_value=0).reshape(outShape)
        arrayOut[k] = img


@njit
def getRotatedBoxDim(inSize: np.ndarray, rotMatrix):
    """
    Return the minimum size of the output box to fit a whole rotated volume.

    Parameters
    ----------
    inSize : Array of rank 3
        Volume dimensions along (X, Y, Z) axes
    rotMatrix : 2D Array
        Rotation matrix

    Returns
    -------
    outSize : Array of rank 3
        Minimum output box size along (X, Y, Z) axes
    """

    cPos = np.int64(inSize/2)

    vertex = np.array([[    0,         0,         0    ],
                       [inSize[0],     0,         0    ],
                       [    0,     inSize[1],     0    ],
                       [    0,         0,     inSize[2]]]).T

    vertex[0, :] = vertex[0, :] - cPos[0]
    vertex[1, :] = vertex[1, :] - cPos[1]
    vertex[2, :] = vertex[2, :] - cPos[2]
    outVertex = rotMatrix@vertex

    return 2*np.int64(np.max(np.abs(outVertex), 1).A.squeeze())


def getRotatedVolumeRangeZ(inSize, tMatV: TMat3D):
    """
   Estimate the scanning z-range for a rotated volume.

   It lists the z-ranges for a list rotList of transformations applied to a
   volume of shape inSize.

   Parameters:
   - inSize (tuple): A tuple containing the dimensions of the input volume in
     the format (nz, ny, nx).
   -tMatV : List of transformation matrices
        TMat3D vector class of 4x4 augmented matrices with the affine
        transformations representing rotations to apply to the volume.

   Returns:
   - rangeZ (numpy.ndarray): An array of shape (nt, 2) containing the estimated
     minimum and maximum z-ranges for each transformation in rotList, where nt
     is the number of transformations.

   This function calculates the minimum and maximum z-positions of the volume's
   vertices after applying each transformation in rotList. The result is a
   range of z-values that covers the volume's extent in the z-direction for
   each transformation.

   Note:
   - The transformations in rotList should be represented as 4x4 transformation
     matrices.
   - The input volume is assumed to be a cuboid with dimensions (nz, ny, nx).
   - The output rangeZ is a 2D array where rangeZ[i, 0] represents the minimum
     z-value and rangeZ[i, 1] represents the maximum z-value for the i-th
     transformation in rotList.

   Example:
   >>> import numpy as np
   >>> from artis_tomo.math.transforms import tr3d
   >>> from artis_tomo.image.transformation import getRotatedVolumeRangeZ
   >>> inSize = (10, 20, 30)
   >>> rotList = tr3d.fromList([tr3d.angles2mat([0,x,0]) for x in [-10, 0, 10]])
   >>> result = getRotatedVolumeRangeZ(inSize, rotList)
   >>> print(result)
   [[-25.   0.]
    [  0.  10.]
    [ -9.  17.]]
   """
    return _getRotatedVolumeRangeZ(inSize, tMatV.matrix)


@njit(nogil=True, parallel=True)
def _getRotatedVolumeRangeZ(inSize, rotList):
    """
   Estimate the scanning z-range for a rotated volume.

   It lists the z-ranges for a list rotList of transformations applied to a
   volume of shape inSize.

   Parameters:
   - inSize (tuple): A tuple containing the dimensions of the input volume in
     the format (nz, ny, nx).
   - rotList (list): A list of transformation matrices representing rotations
     to apply to the volume.

   Returns:
   - rangeZ (numpy.ndarray): An array of shape (nt, 2) containing the estimated
     minimum and maximum z-ranges for each transformation in rotList, where nt
     is the number of transformations.

   This function calculates the minimum and maximum z-positions of the volume's
   vertices after applying each transformation in rotList. The result is a
   range of z-values that covers the volume's extent in the z-direction for
   each transformation.

   Note:
   - The transformations in rotList should be represented as 4x4 transformation
     matrices.
   - The input volume is assumed to be a cuboid with dimensions (nz, ny, nx).
   - The output rangeZ is a 2D array where rangeZ[i, 0] represents the minimum
     z-value and rangeZ[i, 1] represents the maximum z-value for the i-th
     transformation in rotList.

   Example:
   >>> import numpy as np
   >>> from artis_tomo.math.transforms import tr3d
   >>> from artis_tomo.image.transformation import _getRotatedVolumeRangeZ
   >>> inSize = (10, 20, 30)
   >>> rotList = np.asarray([tr3d.angles2mat([0,x,0]).matrix[0] for x in [-10, 0, 10]])
   >>> result = _getRotatedVolumeRangeZ(inSize, rotList)
   >>> print(result)
   [[-25.   0.]
    [  0.  10.]
    [ -9.  17.]]
   """
    nz, ny, nx = inSize
    nt = len(rotList)

    vertices = np.array([[ 0,  0,  0, 1],
                         [nx,  0,  0, 1],
                         [ 0, ny,  0, 1],
                         [ 0,  0, nz, 1],
                         [nx, ny,  0, 1],
                         [ 0, ny, nz, 1],
                         [nx,  0, nz, 1],
                         [nx, ny, nz, 1]])

    rangeZ = np.empty((nt, 2))

    for tp in prange(nt):
        Rproj = rotList[tp]

        nzmax = int(0)
        nzmin = int(0)
        for v in vertices:
            zrot = Rproj[2, 0]*v[0] + Rproj[2, 1]*v[1] + Rproj[2, 2]*v[2] +\
                       Rproj[2, 3]
            nzmax = max(nzmax, math.ceil(zrot))
            nzmin = min(nzmin, math.floor(zrot))

        rangeZ[tp, 0] = nzmin
        rangeZ[tp, 1] = nzmax

    return rangeZ


def resizeFourier(array, outShape):
    """
    Resize an N-dimensional array in Fourier space.

    Parameters
    ----------
    array : N dimensional array
        Input data array.
    outShape : array_like of rank N
        Output array size.

    Returns
    -------
    arrayOut : N dimensional array
        output data array with dimensions outShape.

    """
    inSize = np.asarray(array.shape)
    ndim = array.ndim

    outXsize2 = int(outShape[0]//2 + 1)

    axes = tuple(np.arange(ndim - 1)) if ndim > 1 else (0, )

    with fft.set_backend_from_array(array):
        aft = fft.fftshift(fft.rfftn(fft.ifftshift(array, axes)), axes)
        if ndim > 1:
            tmp_shape = np.append(outShape[:ndim-1],aft.shape[-1])
            aft = fr.padCropArrayCentered(aft, tmp_shape,
                                          mode='constant')[0]
        if outShape[0] > inSize[0]:
            pad = int(outXsize2 - inSize[0]//2)
            aft = np.pad(aft, tuple([(0, 0)]*(ndim - 1)) + ((0, pad),))
        else:
            aft = aft[..., :outXsize2]
        arrayOut = fft.fftshift(fft.irfftn(fft.ifftshift(aft, axes), outShape),
                                axes)

    return arrayOut*arrayOut.size/array.size


def rescaleFourier(array, scale, tol=0.01, maxK=2., actualScale=False,
                   keepShape=False):
    """
    Resize an N-dimensional array in Fourier S. by a pixel size scale factor.

    Output array pixel size is rescaled by factor SCALE. Array is padded to a
    greater size before Fourier resizing to get a real scale factor whose
    difference from SCALE is below required tolerance TOL. Padded size range is
    limited by ARRAY.SHAPE * maxK.

    Parameters
    ----------
    array : N dimensional array
        Input data array.
    scale : Float
        Scale factor applied to pixel size to set output shape.
    tol : Float, optional
        Relative tolerance of output scale compared to requested SCALE param.
        The default is 0.01.
    maxK: Float, optional
        High limit value of the search range to estimate best padded size to
        obtain a real scale factor with an error below TOL. The default is 2.
    actualScale: Boolean
        If True, actual  scale value is also returned. The default is False.
    keepShape: Boolen
        If True, pixel-rescaled image is returned keeping original shape.
        Otherwise, output shape is fitted to same original field of view.


    Returns
    -------
    arrayOut : N dimensional array
        output data array with dimensions outShape.
    actualScale: Float, optional
        Actual scale factor of the resized array.

    """
    inShape = np.asarray(array.shape)
    ndim = array.ndim

    inSize = inShape.max()

    def scaleError(n, _scale):
        return np.abs((n/np.round(n/_scale)-_scale)/_scale)

    sizeV = np.arange(inSize, inSize*maxK)

    idx = np.flatnonzero(scaleError(sizeV, scale) < tol)
    if idx.size == 0:
        raise Exception('No tolerance reached. Try increasing tolerance or \
                        increasing padded size search limit factor MAXK.')
    paddedSize = sizeV[idx[0]]

    inShapePadded = np.asarray([paddedSize, ]*ndim)
    outShapePadded = np.round(inShapePadded/scale).astype(int)

    arrayPadded = fr.padArrayCentered(array, inShapePadded, mode='edge')[0]
    arrayPaddedOut = resizeFourier(arrayPadded, outShapePadded)

    if keepShape:
        outShape = inShape
    else:
        outShape = np.round(inShape/scale).astype(int)

    arrayOut = fr.padCropArrayCentered(arrayPaddedOut, outShape,
                                       mode='edge')[0]

    if actualScale:
        actualScale = paddedSize/outShapePadded[0]
        return arrayOut, actualScale
    else:
        return arrayOut
