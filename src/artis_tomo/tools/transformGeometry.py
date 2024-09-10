#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 09:01:03 2017

@author: joton
"""
import tempfile
import os
import subprocess as sp
import numpy as np
from scipy import interpolate
from scipy.ndimage import interpolation as ip
from .file import removePattern

from ..math.euler import (rotX2mat as rotX,
                          rotY2mat as rotY,
                          rotZ2mat as rotZ)

from artis_tomo.image.transformation import getCenterSym

def getRotMatrix(alpha, beta, gamma):

    return rotZ(np.pi/180*gamma)*rotY(np.pi/180*beta)*rotZ(np.pi/180*alpha)


def eulerZYZ2matrixRelion(alpha, beta, gamma):

    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg = np.cos(gamma)
    sa = np.sin(alpha)
    sb = np.sin(beta)
    sg = np.sin(gamma)
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    A = np.zeros((3,3))

    A[0, 0] =  cg * cc - sg * sa
    A[0, 1] =  cg * cs + sg * ca
    A[0, 2] = -cg * sb
    A[1, 0] = -sg * cc - cg * sa
    A[1, 1] = -sg * cs + cg * ca
    A[1, 2] =  sg * sb
    A[2, 0] =  sc
    A[2, 1] =  ss
    A[2, 2] = cb

    return A


def rotateVolumeWrong(array, phi, theta, psi):

    iSize = np.array(array.shape)
    vC = np.floor(iSize/2).astype(int)

    y = np.arange(iSize[0]) - vC[0]
    x = np.arange(iSize[1]) - vC[1]
    z = np.arange(iSize[2]) - vC[2]

    xx, yy, zz = np.meshgrid(x, y, z)

    M = eulerZYZ2matrix(np.pi/180*phi, np.pi/180*theta, np.pi/180*psi).T

    outFun = interpolate.RegularGridInterpolator((y, x, z), array,
                                                 bounds_error=False,
                                                 fill_value=0)

    outCoord = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))*M

    return outFun(outCoord.A[:, [1, 0, 2]]).reshape(iSize)


def rotateVolume(array, rotMatrix, dimOrder=[1, 0, 2]):

    iSize = np.array(array.shape)
    vC = np.int64(iSize/2)  # Rotation center position

    idimOrd = np.argsort(dimOrder)  # inverse order

    # Input coordinates for interpolation
    XXref = [None]*len(iSize)

    for k in range(len(iSize)):
        XXref[k] = np.arange(iSize[k]) - vC[k]  # original dimensions

#    XXref[1] = np.arange(iSize[idimOrd[1]]) - vC[idimOrd[1]]  # Y
#    XXref[2] = np.arange(iSize[idimOrd[2]]) - vC[idimOrd[2]]  # Z
#    XX = [XXref[k] for k in idimOrd]

    # X, Y, Z coordinates of the rotated volume are the same as input. We guess
    # we're rotating cubic volumes. Otherwise code must be modified
    XXmesh = np.meshgrid(*XXref, copy=False, indexing='ij')
    # We flatten by default in order='C', which match with the same default
    # order use in the reshape step after interp below
    xf = XXmesh[idimOrd[0]].ravel()
    yf = XXmesh[idimOrd[1]].ravel()
    zf = XXmesh[idimOrd[2]].ravel()

    rMInv = rotMatrix.I
    rotCoord = (rMInv@np.row_stack((xf, yf, zf))).A.T

    rotVol = interpolate.interpn(XXref, array,
                              rotCoord[:, dimOrder],
                              bounds_error=False,
                              fill_value=array[0, 0, 0]).reshape(iSize)

    return rotVol


def rotateShiftVolume(array, rotMatrix, shifts=[0,0,0], dimOrder=[1, 0, 2],
                      shiftLast=True):

    iSize = np.array(array.shape)
    vC = np.int64(iSize/2)  # Rotation center position

    idimOrd = np.argsort(dimOrder)  # inverse order

    # Input coordinates for interpolation
    XXref = [None]*len(iSize)

    for k in range(len(iSize)):
        XXref[k] = (np.arange(iSize[k]) - vC[k]).astype(float)  # original dimensions

#    XXref[1] = np.arange(iSize[idimOrd[1]]) - vC[idimOrd[1]]  # Y
#    XXref[2] = np.arange(iSize[idimOrd[2]]) - vC[idimOrd[2]]  # Z
#    XX = [XXref[k] for k in idimOrd]

    # X, Y, Z coordinates of the rotated volume are the same as input. We guess
    # we're rotating cubic volumes. Otherwise code must be modified
    XXmesh = np.meshgrid(*XXref, copy=False, indexing='ij')
    # We flatten by default in order='C', which match with the same default
    # order use in the reshape step after interp below
    xf = XXmesh[idimOrd[0]].ravel()
    yf = XXmesh[idimOrd[1]].ravel()
    zf = XXmesh[idimOrd[2]].ravel()

    if shiftLast:
        xf = xf - shifts[0]
        yf = yf - shifts[1]
        zf = zf - shifts[2]

    rMInv = rotMatrix.I
    rotCoord = (rMInv@np.row_stack((xf, yf, zf))).A.T

    if not shiftLast:
        for k in range(len(iSize)):
            rotCoord[:, k] -= shifts[k]

    rotVol = interpolate.interpn(XXref, array,
                              rotCoord[:, dimOrder],
                              bounds_error=False,
                              fill_value=array[0, 0, 0]).reshape(iSize)

    return rotVol


def rotateProj(array, rotMatrix, dimOrder=[1, 0]):

    iSize = np.array(array.shape)
    nDim = len(iSize)
    vC = np.int64(iSize/2)  # Rotation center position

    idimOrd = np.argsort(dimOrder)  # inverse order

    # Input coordinates for interpolation
    XXref = [None]*nDim

    for k in range(nDim):
        XXref[k] = np.arange(iSize[k]) - vC[k]  # original dimensions

#    XXref[1] = np.arange(iSize[idimOrd[1]]) - vC[idimOrd[1]]  # Y
#    XXref[2] = np.arange(iSize[idimOrd[2]]) - vC[idimOrd[2]]  # Z
#    XX = [XXref[k] for k in idimOrd]

    # X, Y, Z coordinates of the rotated volume are the same as input. We guess
    # we're rotating cubic volumes. Otherwise code must be modified
    XXmesh = np.meshgrid(*XXref, copy=False, indexing='ij')
    # We flatten by default in order='C', which match with the same default
    # order use in the reshape step after interp below
    xf = XXmesh[idimOrd[0]].ravel()
    yf = XXmesh[idimOrd[1]].ravel()

    rMInv = rotMatrix[0:2, 0:2].I
    rotCoord = (rMInv@np.row_stack((xf, yf))).A.T

    rotVol = interpolate.interpn(XXref, array,
                                 rotCoord[:, dimOrder],
                                 bounds_error=False,
                                 fill_value=array[0, 0],
                                 method='linear').reshape(iSize)

    return rotVol


def getPeakPos(profile):

    diffprof = np.diff(profile)
    minpos = np.argmin(diffprof)
    maxpos = np.argmax(diffprof)

    zeropos = minpos + \
        (maxpos - minpos)/(diffprof[maxpos] - diffprof[minpos])*\
        (- diffprof[minpos])
    return zeropos


def getCenterSymFromSumProfile(im):

    xdim = im.shape[0]
    xdh = xdim//2

    xprofile = np.sum(im, 0)
    yprofile = np.sum(im, 1)
    xc0 = np.argmin(xprofile)
    yc0 = np.argmin(yprofile)
    xc = getPeakPos(xprofile)
    yc = getPeakPos(yprofile)

    atol = 2
    if (not np.isclose(xc, xc0, atol=atol)) or \
       (not np.isclose(yc, yc0, atol=atol)):
        r2 = (xc-xdh)**2 + (yc-xdh)**2
        r02 = (xc0-xdh)**2 + (yc0-xdh)**2

        if r02 < r2:
            return (yc0, xc0)

    return (yc, xc)



def calculateShiftEFTEM(fnRef, fnIm, eftemDict):

    command = ('java -cp {FIJIHOME}/ImageJ-linux64:{FIJIHOME}/plugins/' +
               'TomoJ_2.32-jar-with-dependencies.jar:{FIJIHOME}/plugins/' +
               'Eftem_TomoJ_1.04.jar eftemtomoj/' +
               'EFTEM_TomoJ ').format(**eftemDict)
    command += '-tsSignal {} 1 1 -tsBg {} 2 1 -align NMI 0'.format(fnRef, fnIm)

    subp = sp.Popen(command.split(), stdin=sp.PIPE,
                    stdout=sp.PIPE, stderr=sp.PIPE)

    return subp.communicate()


def xyRealignEFTEM(array):

    from ..io.imageIO import writeMRC

    eftemDict = {'FIJIHOME': '/home/joton/opt/Fiji.app'}

    mywd = os.getcwd()
    os.chdir(tempfile.tempdir)

    nIm = array.shape[2]

    iC = int(nIm/2)

    dshifts = np.zeros((nIm, 2))

    fnRef = tempfile.mktemp() + '.mrc'
    fnIm = tempfile.mktemp() + '.mrc'
    fnTmp = fnIm
    fnShifts = fnIm + '_aligned.transf'

#    writeMRC(array[..., 0], fnRef)
    writeMRC(array[..., 0], fnIm)

    for k in range(1, nIm):

        fnTmp = fnRef
        fnRef = fnIm
        fnIm = fnTmp

        writeMRC(array[..., k], fnIm)

        calculateShiftEFTEM(fnRef, fnIm, eftemDict)
        dshifts[k, :] = np.loadtxt(fnShifts)

    shifts = dshifts.cumsum(axis=0)

    shifts -= np.ones((nIm, 1))*shifts[iC, ...]

#    writeMRC(array[..., iC], fnRef)
#    tmpIdx = np.arange(nIm)
#    for k in tmpIdx[tmpIdx != iC]:
#        writeMRC(array[..., k], fnIm)
#        calculateShiftEFTEM(fnRef, fnIm, eftemDict)
#        dshifts[k, :] = np.loadtxt(fnShifts)

    removePattern(fnRef+'*')
    removePattern(fnIm+'*')
    os.chdir(mywd)

    arrayOut = np.empty(array.shape)

    for k in range(nIm):
        arrayOut[..., k] = ip.shift(array[..., k], shifts[k, :], mode='mirror')

    return arrayOut


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


def sphereSurfacePointsAngles(nPoints, thetaMax=np.pi):
    """
    Calculate positions around a spherical surface uniformily distributed

    Parameters
    ----------
    nPoints    : Integer number of points

    thetaMax   : maximum value for theta angle. Default = pi

    Returns
    -------
    angles     :    Array of rank 2
                 Column 1 is Theta angle (Elevation)
                 Column 2 is Phi angle (Azimuth)

    """

    indices = np.arange(0, nPoints, dtype=float)  # + 0.5
    thetaTot = np.arccos(1 - 2*indices/(nPoints-1))
    theta = thetaTot[thetaTot <= thetaMax]
    iMax = len(theta)

    # Phi step is the gold ratio to avoid points to match in phi
    phi = np.pi * (1 + 5**0.5) * indices[:iMax]
    phi = phi % (2*np.pi)

    angles = np.zeros((iMax, 2))
    angles[...] = np.column_stack([theta, phi])

    return angles


def sphereSurfacePoints(nPoints, thetaMax=np.pi):
    """
    Calculate positions around a spherical surface uniformily distributed

    Parameters
    ----------
    nPoints    : Integer number of points

    thetaMax   : maximum value for theta angle. Default = pi

    Returns
    -------
    points     : Coordinates of points over a spherical surface of radius 1

    """
    angles = sphereSurfacePointsAngles(nPoints, thetaMax)
    theta = angles[:, 0]
    phi = angles[:, 1]

    points = np.column_stack([np.cos(phi) * np.sin(theta),
                              np.sin(phi) * np.sin(theta),
                              np.cos(theta)])

    return points


def sphereSurfaceAngleStep(angStep, thetaMax=np.pi):
    """
    Calculate positions around a spherical surface uniformily distributed

    Parameters
    ----------
    angStep    : angular separation among points

    thetaMax   : maximum value for theta angle. Default = pi

    Returns
    -------
    points     : Coordinates of points over a spherical surface of radius 1

    """
    nPoints = round(4*(180**2)/(np.pi*angStep**2))
    indices = np.arange(0, nPoints, dtype=float)  # + 0.5
    thetaTot = np.arccos(1 - 2*indices/(nPoints-1))
    theta = thetaTot[thetaTot <= thetaMax]
    iMax = len(theta)
    points = np.zeros((iMax, 3))

    # Phi step is the gold ratio to avoid points to match in phi
    phi = np.pi * (1 + 5**0.5) * indices[:iMax]
    phi = phi % (2*np.pi)

    points[...] = np.column_stack([np.cos(phi) * np.sin(theta),
                                   np.sin(phi) * np.sin(theta),
                                   np.cos(theta)])

    return points
