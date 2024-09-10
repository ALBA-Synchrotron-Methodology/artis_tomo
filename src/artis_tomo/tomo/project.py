"""
Module for simulated projections.

@author: joton
"""
import numba

import numpy as np
import math
import pywt
from numba import njit, prange, cuda
from ..utils.gpu import gpu
from joblib import Parallel, delayed

from artis_tomo.math.transforms import tr3d, TMat3D
from ..math import fft, framework as fw
from ..image.transformation import getRotatedVolumeRangeZ
from ..tools.parallel import splitTasks


def mergeWavelet(array):

    if array.ndim == 1:
        array = array[None, :, :]

    ni = array.shape[0]

    wvList = list()
    for image in array:
        wvList.append(pywt.wavedecn(image, 'db20'))

    nw = len(wvList[0])

    wvOut = wvList[0]

    aCoef = wvOut[0]
    for k in range(1, ni):

        np.maximum(aCoef, wvList[k][0], out=aCoef[...])

        for kl in range(1, nw):  # level
            for kd in ['ad', 'da', 'dd']:  # detail
                dCoef = wvOut[kl][kd]
                np.maximum(dCoef, wvList[k][kl][kd], out=dCoef[...])

    arrayOut = pywt.waverecn(wvOut, 'db20')

    return arrayOut


def mergeFourier(array):

    if array.ndim == 1:
        array = array[None, :, :]
    ni = array.shape[0]

    fFT = np.empty(array.shape, dtype=np.complex128)

    for k in range(ni):
        fFT[...] = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(array[k])))

    fFToout = np.max(fFT, 0)

    mean = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fFToout))))

    return mean


def projectOneRS(array, tMatV: TMat3D = None, projSize=None, center=[0, 0, 0],
                 nThreads=-1):
    """
    Project in R.S. along z direction after applying a transformation matrix.

    Array axes order is assumed as Z, Y and X.

    Parameters
    ----------
    array : ndarray
        3D input array volume.
    tMatV : List of transformation matrices
        TMat3D vector class of 4x4 augmented matrices with the affine
        transformations. If None, the identity matrix is used.
    projSize : ints list, optional
        X, Y dimensions of projection. If None, X, Y input volume dimensions
        are used instead.
    center : floats list, optional
        Relative X, Y and Z coordinates of rotation center regarding to
        volume center.
    nThreads: int, optional
            Number of threads used in parallel. If -1, total number of
            available-threads/nProcs is used.

    Returns
    -------
    proj : ndarray
        2D array with transformed pattern.

    """
    if tMatV is None:
        tMatV = tr3d.empty()
    elif isinstance(tMatV, np.ndarray):
        tMatV = TMat3D(tMatV)

    if nThreads > 0:
        numba.set_num_threads(nThreads)

    nt = len(tMatV)
    dims = np.array(array.shape)
    nz, ny, nx = dims
    iSizehV = dims//2  # Default Rotation center position

    if projSize is None:
        projStkSize = np.array([nt, ny, nx])
    else:
        projStkSize = np.asarray((nt,) + projSize)
    sizeh = projStkSize[1:]//2

    centerProj = tr3d.shifts2mat([*(sizeh[::-1] + center[:2]), 0])
    centerVol = tr3d.shifts2mat(-(iSizehV[::-1] + center))

    # Trans matrices from Volume coords to Projection coords.
    RvpMatV = centerProj * tMatV * centerVol
    # Transformations from Proj. to Volume
    RinvMatV = RvpMatV.inv

    zRanges = getRotatedVolumeRangeZ(array.shape, RvpMatV)

    xp = fw.frame.from_array(array)
    fwName = xp.get_frameInfo()

    projStk = xp.empty(projStkSize, np.float32)
    if fwName == 'cupy':
        RinvArrayD = xp.to_device(RinvMatV.matrix.astype(np.float32))
        zRangesD = xp.to_device(zRanges.astype(np.float32))

        threadsperblock = gpu.getThrBlk(3)
        blockspergrid_x = math.ceil(nx / threadsperblock[0])
        blockspergrid_y = math.ceil(ny / threadsperblock[1])
        blockspergrid_z = math.ceil(nt / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

        _projectRSCuda[blockspergrid, threadsperblock](array, projStk,
                                                       RinvArrayD, zRangesD)
    elif fwName == 'numpy':
        _projectRS(array, projStk, RinvMatV.matrix, zRanges)
    else:
        raise ValueError(f'Project is not currently implemented for {fwName}.')

    return projStk


def backProjectOneRS(array, tMatV: TMat3D, volSize=None):
    """
    Backproject in R.S.

    Array axes order is assumed as Z, Y and X.

    Parameters
    ----------
    array : ndarray
        3D input array. Stack of image projections. First axis is image index.
    tMatV : List of transformation matrices
        TMat3D vector class of 4x4 augmented matrices with the affine
        transformations.
    volSize : ints list, optional
        Z, Y and X dimensions of reconstructed volume. If None, (X, Y, X)
        dimensions from projections are used instead.
    shifts : floats list, optional
        X, Y and Z offsets applied to volume frame.

    Returns
    -------
    rec : 3d array
        Reconstructed volume.

    """
    dims = np.array(array.shape)
    nt, ny, nx = dims

    projSize = np.array([ny, nx])
    sizeh = projSize//2

    if volSize is None:
        volSize = np.array([nx, ny, nx])
    else:
        volSize = np.asarray(volSize)
    vSizeh = volSize//2

    centerProj = tr3d.shifts2mat([*(sizeh[::-1]), 0])
    centerVol = tr3d.shifts2mat(-vSizeh[::-1])

    RvpMatV = centerProj * tMatV * centerVol

    xp = fw.frame.from_array(array)
    fwName = xp.get_frameInfo()

    if fwName == 'cupy':
        arrayD = cuda.to_device(array.astype(np.float32))
        RvpArrayD = cuda.to_device(RvpMatV.matrix.astype(np.float32))
        recD = cuda.device_array(volSize, np.float32)

        threadsperblock = gpu.getThrBlk(3)
        blockspergrid_x = math.ceil(volSize[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(volSize[1] / threadsperblock[1])
        blockspergrid_z = math.ceil(volSize[2] / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

        _backProjectRSCuda[blockspergrid,
                           threadsperblock](arrayD, recD, RvpArrayD)
        rec = recD.copy_to_host()

    elif fwName == 'numpy':
        rec = np.zeros(volSize)
        _backProjectRS(array, rec, RvpMatV.matrix)
    else:
        raise ValueError(f'Backproject is not currently implemented for {fwName}.')

    return rec


# Projection interpolating in inverted direction
@njit(nogil=True)
def _projectRSinv(array, proj, Rproj):

    nz, ny, nx = np.array(array.shape)
    nyp, nxp = proj.shape
    weight = np.zeros(proj.shape)

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                xproj = Rproj[0, 0]*x + Rproj[0, 1]*y + Rproj[0, 2]*z \
                        + Rproj[0, 3]
                yproj = Rproj[1, 0]*x + Rproj[1, 1]*y + Rproj[1, 2]*z \
                    + Rproj[1, 3]

                x0 = math.floor(xproj)
                y0 = math.floor(yproj)
                value = array[z, y, x]

                if value > 0:

                    for ky in range(2):
                        yp = y0 + ky
                        if yp >= nyp or yp < 0:
                            continue
                        wy = 1. - np.abs(yproj - yp)

                        for kx in range(2):
                            xp = x0 + kx
                            if xp >= nxp or xp < 0:
                                continue
                            wx = 1. - np.abs(xproj - xp)
                            proj[yp, xp] += wy*wx*value
                            weight[yp, xp] += wy*wx

    # proj[...] = proj / weight
    # proj[...] = weight

    # for k in range(proj.size):
        # if weight.flat[k] > 0:
            # proj.flat[k] = proj.flat[k] / weight.flat[k]


@njit(nogil=True, parallel=True)
def _projectRS(array, projStk, RinvMatV, zRanges):

    nz, ny, nx = array.shape
    ntp, nyp, nxp = projStk.shape

    for tp in range(ntp):
        Rinv = RinvMatV[tp]
        proj = projStk[tp, :, :]

        nzpmin, nzpmax = zRanges[tp]

        for yp in prange(nyp):
            xvoly = Rinv[0, 1]*yp + Rinv[0, 3]
            yvoly = Rinv[1, 1]*yp + Rinv[1, 3]
            zvoly = Rinv[2, 1]*yp + Rinv[2, 3]

            for xp in range(nxp):
                xvolyx = xvoly + Rinv[0, 0]*xp
                yvolyx = yvoly + Rinv[1, 0]*xp
                zvolyx = zvoly + Rinv[2, 0]*xp

                tmp = 0.

                for zp in range(nzpmin, nzpmax):
                    xvol = xvolyx + Rinv[0, 2]*zp
                    yvol = yvolyx + Rinv[1, 2]*zp
                    zvol = zvolyx + Rinv[2, 2]*zp
                    x0 = math.floor(xvol)
                    y0 = math.floor(yvol)
                    z0 = math.floor(zvol)

                    if x0 < -1 or y0 < -1 or z0 < -1 or \
                       x0 > nx-1 or y0 > ny-1 or z0 > nz-1:
                        continue

                    for kz in [0, 1]:
                        z = z0 + kz
                        if z > -1 and z < nz:
                            wz = 1. - np.abs(zvol - z)
                            if wz == 0.:
                                continue
                        else:
                            continue

                        for ky in [0, 1]:
                            y = y0 + ky
                            if y > -1 and y < ny:
                                wy = 1. - np.abs(yvol - y)
                                if wy == 0.:
                                    continue
                            else:
                                continue

                            for kx in [0, 1]:
                                x = x0 + kx
                                if x > -1 and x < nx:
                                    wx = 1. - np.abs(xvol - x)
                                    if wx == 0.:
                                        continue
                                else:
                                    continue

                                value = array[z, y, x]
                                tmp += wz*wy*wx*value

                proj[yp, xp] = tmp


@cuda.jit
def _projectRSCuda(array, projStk, RinvMatV, zRanges):

    nz, ny, nx = array.shape
    ntp, nyp, nxp = projStk.shape

    xp, yp, tp = cuda.grid(3)

    if tp < ntp and yp < nyp and xp < nxp:

        Rinv = RinvMatV[tp]
        proj = projStk[tp, :, :]

        nzpmin, nzpmax = zRanges[tp]

        xvolyx = Rinv[0, 0]*xp + Rinv[0, 1]*yp + Rinv[0, 3]
        yvolyx = Rinv[1, 0]*xp + Rinv[1, 1]*yp + Rinv[1, 3]
        zvolyx = Rinv[2, 0]*xp + Rinv[2, 1]*yp + Rinv[2, 3]

        tmp = 0.

        for zp in range(nzpmin, nzpmax):
            xvol = xvolyx + Rinv[0, 2]*zp
            yvol = yvolyx + Rinv[1, 2]*zp
            zvol = zvolyx + Rinv[2, 2]*zp
            x0 = math.floor(xvol)
            y0 = math.floor(yvol)
            z0 = math.floor(zvol)

            if x0 < -1 or y0 < -1 or z0 < -1 or \
               x0 > nx-1 or y0 > ny-1 or z0 > nz-1:
                continue

            for kz in [0, 1]:
                z = int(z0 + kz)
                if z > -1 and z < nz:
                    wz = 1. - abs(zvol - z)
                    if wz == 0.:
                        continue
                else:
                    continue

                for ky in [0, 1]:
                    y = int(y0 + ky)
                    if y > -1 and y < ny:
                        wy = 1. - abs(yvol - y)
                        if wy == 0.:
                            continue
                    else:
                        continue

                    for kx in [0, 1]:
                        x = int(x0 + kx)
                        if x > -1 and x < nx:
                            wx = 1. - abs(xvol - x)
                            if wx == 0.:
                                continue
                        else:
                            continue

                        value = array[z, y, x]
                        tmp += wz*wy*wx*value

        proj[yp, xp] = tmp


@njit(nogil=True, parallel=True)
def _backProjectRS(array, rec, RvpMatV):

    nz, ny, nx = np.array(rec.shape)
    # nzh = nz//2
    # nyh = ny//2
    # nxh = nx//2

    ntp, nyp, nxp = array.shape

    # rmax2 = (nxp//2 - 20)**2

    for z in prange(nz):
        # r2z = (z - nzh)**2
        for tp in range(ntp):
            Rproj = RvpMatV[tp]
            proj = array[tp, :, :]

            xprojz = Rproj[0, 2]*z + Rproj[0, 3]
            yprojz = Rproj[1, 2]*z + Rproj[1, 3]

            for y in range(ny):
                # r2zy = r2z + (y - nyh)**2
                xprojzy = xprojz + Rproj[0, 1]*y
                yprojzy = yprojz + Rproj[1, 1]*y

                for x in range(nx):

                    # r2 = r2zy + (x - nxh)**2
                    # print(f'r2={r2} - rmax2={rmax2}')
                    # if r2 > rmax2:
                        # rec[z, y, x] = 0
                        # continue

                    xproj = xprojzy + Rproj[0, 0]*x
                    yproj = yprojzy + Rproj[1, 0]*x

                    x0 = math.floor(xproj)
                    y0 = math.floor(yproj)

                    tmp = 0.

                    for ky in [0, 1]:
                        yp = y0 + ky
                        if yp > -1 and yp < nyp:
                            wy = 1. - np.abs(yproj - yp)
                            if wy == 0:
                                continue
                        else:
                            continue

                        for kx in [0, 1]:
                            xp = x0 + kx
                            if xp > -1 and xp < nxp:
                                wx = 1. - np.abs(xproj - xp)
                                if wx == 0:
                                    continue
                            else:
                                continue

                            tmp += wy*wx*proj[yp, xp]

                    rec[z, y, x] += tmp


@cuda.jit
def _backProjectRSCuda(array, rec, RvpMatV):

    nz, ny, nx = rec.shape
    ntp, nyp, nxp = array.shape

    x, y, z = cuda.grid(3)

    if z < nz and y < ny and x < nx:
        tmp = 0.
        for tp in range(ntp):
            Rproj = RvpMatV[tp]
            proj = array[tp, :, :]

            # xprojz = Rproj[0, 2]*z + Rproj[0, 3]
            # yprojz = Rproj[1, 2]*z + Rproj[1, 3]
            # xprojzy = xprojz + Rproj[0, 1]*y
            # yprojzy = yprojz + Rproj[1, 1]*y

            xproj = Rproj[0, 0]*x + Rproj[0, 1]*y + Rproj[0, 2]*z + Rproj[0, 3]
            yproj = Rproj[1, 0]*x + Rproj[1, 1]*y + Rproj[1, 2]*z + Rproj[1, 3]

            x0 = math.floor(xproj)
            y0 = math.floor(yproj)

            for ky in [0, 1]:
                yp = int(y0 + ky)
                if yp > -1 and yp < nyp:
                    wy = 1. - abs(yproj - yp)
                    if wy == 0:
                        continue
                else:
                    continue

                for kx in [0, 1]:
                    xp = int(x0 + kx)
                    if xp > -1 and xp < nxp:
                        wx = 1. - abs(xproj - xp)
                        if wx == 0:
                            continue
                    else:
                        continue

                    tmp += wy*wx*proj[yp, xp]

        rec[z, y, x] += tmp






def projectRS(array, tMatV: TMat3D, projSize=None, center=[0, 0, 0],
              nProcs=1, nThreads=-1, pPool=None):
    """
    Project in R.S. along z direction for a list of transformation matrices.

    Array axes order is assumed as Z, Y and X.

    Parameters
    ----------
    array : ndarray
        3D input array.
    tMatV : List of transformation matrices
        TMat3D vector class of 4x4 augmented matrices with the affine
        transformations.
    projSize : ints list, optional
        X, Y dimensions of projection. If None, X, Y input volume dimensions
        are used instead.
    center : floats list, optional
        Relative X, Y and Z coordinates of rotation center regarding to
        volume center.
    nProcs : int, optional
            Number of projections processed in parallel. If -1, total number
            of available cpu cores is used.
    nThreads: int, optional
            Number of threads used in every single processed projection. If -1,
            total number of available/nProcs is used.
    pPool : joblib parallel pool, optional
                If not provided, internal pool is created.

    Returns
    -------
    proj : List of ndarrays
        List of 2D arrays with transformed pattern.

    """
    if isinstance(tMatV, np.ndarray):
        tMatV = TMat3D(tMatV)
    nt = len(tMatV)

    if pPool is None:
        pPool = Parallel(n_jobs=nProcs)

    if nThreads < 0:
        nThreads = numba.config.NUMBA_NUM_THREADS//pPool.n_jobs

    tRangeIni, tRangeEnd = splitTasks(nt, pPool.n_jobs)
    slcs = [slice(tRangeIni[k], tRangeEnd[k]) for k in range(pPool.n_jobs)]

    projs = pPool(delayed(projectOneRS)
                  (array, tMatV[slcs[k]], projSize, center,
                   nThreads) for k in range(pPool.n_jobs))

    return projs[0]


def backProjectRS(projs, tMatV: TMat3D, volSize=None, nProcs=1, nThreads=-1,
                  pPool=None):
    """
    Backproject in R.S.

    Array axes order is assumed as Z, Y and X.

    Parameters
    ----------
    projs : ndarray
        3D input array. Stack of image projections. First axis is image index.
    tMatV : List of transformation matrices
        TMat3D vector class of 4x4 augmented matrices with the affine
        transformations.
    volSize : ints list, optional
        Z, Y and X dimensions of reconstructed volume. If None, (X, Y, X)
        dimensions from projections are used instead.
    nProcs : int, optional
            Number of image sets processed in parallel. If -1, total number
            of available cpu cores is used.
    nThreads: int, optional
            Number of threads used in parallel. If -1, total number of
            available-threads/nProcs is used.
    pPool : joblib parallel pool, optional
                If not provided, internal pool is created.

    Returns
    -------
    rec : 3d array
        Reconstructed volume.

    """
    nt = len(tMatV)
    nx = projs.shape[2]

    if pPool is None:
        pPool = Parallel(n_jobs=nProcs)

    nThreadsIni = numba.get_num_threads()
    if nThreads < 0:
        nThreads = numba.config.NUMBA_NUM_THREADS//pPool.n_jobs
    numba.set_num_threads(nThreads)

    tRangeIni, tRangeEnd = splitTasks(nt, pPool.n_jobs)
    slcs = [slice(tRangeIni[k], tRangeEnd[k]) for k in range(pPool.n_jobs)]

    tmpvol = pPool(delayed(backProjectOneRS)
                   (projs[slcs[k]], tMatV[slcs[k]], volSize)
                   for k in range(pPool.n_jobs))

    rec = tmpvol[0]

    if pPool.n_jobs > 1:
        for k in range(1, pPool.n_jobs):
            rec += tmpvol[k]

    numba.set_num_threads(nThreadsIni)

    del tmpvol
    rec /= (nt*nx)

    return rec
