"""
Module for tomogram reconstruction.

@author: joton
"""
import numpy as np
import numba
import math
from artis_tomo.math.transforms import tr3d, TMat3D
# from ..math import fft
from artis_tomo.image.transformation import getRotatedVolumeRangeZ
from artis_tomo.image import filter as ft
from artis_tomo.utils.gpu import gpu
from artis_tomo.math import framework as fw
from artis_tomo.tomo.filter import bpFilter, bpExactFilter
from .project import (projectRS, backProjectRS, _projectRS, _backProjectRS,
                      _projectRSCuda, _backProjectRSCuda)

__all__ = ['reconstruct', 'getPSFRecon', 'getNormRecon']

# bpNormK = 1./1.63  # BP Normalization factor for nz/nx = 1


def reconstruct(projsIni, tMatV: TMat3D, algorithm='fbp', initVolume=None,
                filterName='ramp', volSize=None, nIter=1, reg=0.1, tol=5e-2,
                center=[0, 0], shifts=[0, 0, 0], nProcs=1, nThreads=-1,
                pPool=None):
    """
    Reconstruct tomograms.

    It can use different approaches for any projection direction.

    Parameters
    ----------
    projsIni : ndarray
        3D input array. Stack of image projections. First axis is image index.
    tMatV : List of transformation matrices
        TMat3D vector class of 4x4 augmented matrices with the affine
        transformations.
    algorithm : str, optional
        Reconstruction algorithm. Options are fbp, art, sirt, cg.
        The default is 'fbp'.
    filterName : str, optional
        Filter to compensate back projection reconstruction weights. It can be
        ramp, shepp, cosine, hamming, hanning or None. The default is
        'ramp'.
    initVolume : ndarray, optional
        3D volume array as initial reconstruction. The default is None.
    volSize : int list, optional
        Z, Y and X dimensions of reconstructed volume. If None, (X, Y, X)
        dimensions from projections are used. The default is None.
    nIter : int, optional
        Number of iterations. The default is 1.
    center : int list, optional
        Tilt axis [X, Y] position from projections center pixel.
        The default is [0, 0].
    shifts : int list, optional
        [X, Y, Z] offsets applied to volume frame. The default is [0, 0, 0].
    nProcs : int, optional
        Number of image sets processed in parallel. If -1, total number
        of available cpu cores is used. The default is 1.
    nThreads : int, optional
        Number of threads used in parallel. If -1, total number of
        available-threads/nProcs is used. The default is -1.
    pPool : joblib parallel pool, optional
        If not provided, internal pool is created. The default is None.

    Returns
    -------
    rec : ndarray
        3d reconstructed volume array.

    """
    if isinstance(tMatV, np.ndarray):
        tMatV = TMat3D(tMatV)

    centerProj = tr3d.shifts2mat([*(center), 0])
    centerVol = tr3d.shifts2mat(-(np.asarray([*(center), 0]) + shifts))

    RcsMatV = centerProj*tMatV*centerVol

    if filterName is None:
        projs = projsIni
    else:
        projs = bpFilter(projsIni, filterName)

    if initVolume is None or algorithm == 'fbp':
        rec = backProjectRS(projs, RcsMatV, volSize, nProcs, nThreads, pPool)
        initVolume = rec

    if algorithm != 'fbp':

        if algorithm in _recFuns:
            rec = reconITER(projsIni, RcsMatV, algorithm, initVolume,
                            filterName, nIter, reg, tol, nProcs, nThreads,
                            pPool)

        else:
            raise Exception(f'Algorithm {algorithm} not yet implemented in '
                            'Artis (former xpytools). Please, contact '
                            'developers if you really require it.')

    return rec


def reconITER(projsIni, tMatV: TMat3D, algorithm, initVolume, filterName, nIter=15,
              reg=0.1, tol=1e-2, nProcs=1, nThreads=-1, pPool=None):

    nt = projsIni.shape[0]
    nz, ny, nx = initVolume.shape

    projSize = np.asarray([ny, nx])
    sizeh = projSize//2

    volSize = np.asarray([nz, ny, nx])
    vSizeh = volSize//2

    centerProj = tr3d.shifts2mat([*(sizeh[::-1]), 0])
    centerVol = tr3d.shifts2mat(-vSizeh[::-1])

    RvpMatV = centerProj * tMatV * centerVol
    RinvMatV = RvpMatV.inv

    zRanges = getRotatedVolumeRangeZ(volSize, RvpMatV)

    # ATA = pr.backProjectRS(prtemp, RprojsIn, volSize=volsize, filterName=filtername)

    # knorm = reg/(nt*nx)*bpNormK
    knorm = reg/(nt*nx)
    errorNorm = 1./np.linalg.norm(projsIni)

    recfun = _getRecFun(algorithm)

    _recFuns['filter'] = bpFilter

    cp = fw.frame.from_array(projsIni)
    fName = cp.get_frameInfo()

    if fName == 'cupy':
        # print('reconITER-gpu')

        threadsperblock = gpu.getThrBlk(3)
        blockspergrid_x = math.ceil(nx / threadsperblock[0])
        blockspergrid_y = math.ceil(ny / threadsperblock[1])
        blockspergrid_z = math.ceil(nz / threadsperblock[2])
        blockspergrid_t = math.ceil(nt / threadsperblock[2])

        bpgProj = (blockspergrid_x, blockspergrid_y, blockspergrid_t)
        bpgBProj = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

        recD = cp.to_device(initVolume, dtype=np.float32)
        projsD = cp.to_device(projsIni, dtype=np.float32)
        RvpD = cp.to_device(RvpMatV.matrix, dtype=np.float32)
        RinvD = cp.to_device(RinvMatV.matrix, dtype=np.float32)
        zRangesD = cp.to_device(zRanges, dtype=np.float32)

        prSimD = cp.empty_like(projsIni, np.float32)
        prSimD[0, 0, 0] = cp.nan
        rectmpD = cp.empty_like(initVolume, np.float32)

        _recFuns['project'] = _projectRSCuda[bpgProj, threadsperblock]
        _recFuns['backproject'] = _backProjectRSCuda[bpgBProj, threadsperblock]

        for i in range(nIter):
            rms = recfun(projsD, recD, RvpD, RinvD, zRangesD, filterName,
                         knorm, prSimD, rectmpD)

            error = rms*errorNorm
            print(f'Iter={i}, RMSerror={error}, tol={tol}')
            if error < tol:
                break

        rec = cp.to_device(recD, 'cpu')

    elif fName == 'numpy':
        # print('reconITER-nogpu')
        rec = initVolume.copy()
        prSim = np.empty_like(projsIni)
        rectmp = np.empty_like(rec)

        _recFuns['project'] = _projectRS
        _recFuns['backproject'] = _backProjectRS

        for i in range(nIter):
            rms = recfun(projsIni, rec, RvpMatV.matrix, RinvMatV.matrix, zRanges,
                         filterName, knorm, prSim, rectmp)

            error = rms*errorNorm
            print(f'Iter={i}, RMSerror={error}, tol={tol}')
            if error < tol:
                break

    return rec


_recFuns = {}


def registerRecFunIter(label):
    def inner(func):
        global _recFuns
        _recFuns[label] = func
        return func
    return inner


def _getRecFun(algorithm):

    # algFun = {'art': reconART
    #           }
    global _recFuns

    if algorithm in _recFuns:
        return _recFuns[algorithm]
    else:
        raise Exception(f'Tomo.recon: algorithm {algorithm} not implemented.')


@registerRecFunIter('art')
def reconART(projs, rec, RvpList, RinvList, zRanges, filterName, reg,
             projTmp, recTmp):

    xp = fw.frame.from_array(projs)

    if xp.isnan(projTmp[0, 0, 0]):
        projTmp[0, 0, 0] = 0

    nt = len(projs)
    projOrd = xp.argsort(xp.abs(xp.arange(nt)-nt//2 + 0.25))[::-1]

    for k in range(nt):
        ko = projOrd[k]
        # print(f'project {k}: ko={ko}')
        proj1 = projTmp[ko][None, ...]
        Rinv1 = RinvList[ko][None, ...]
        zRange1 = zRanges[ko][None, :]

        _recFuns['project'](rec, proj1, Rinv1, zRange1)
        proj1 -= projs[ko]

        if 'filter' in _recFuns and filterName is not None:
            # print('Filter  one')
            proj1[...] = _recFuns['filter'](proj1, filterName)

        # print('Reconstruct one')
        recTmp.fill(0)
        _recFuns['backproject'](proj1, recTmp, RvpList[ko][None, ...])

        rec -= reg * recTmp
        # rec[rec < 0] = 0

    _recFuns['project'](rec, projTmp, RinvList, zRanges)
    projTmp -= projs

    rms = np.linalg.norm(projTmp)

    return rms


@registerRecFunIter('sirt')
def reconSIRT(projs, rec, RvpList, RinvList, zRanges, filterName, reg,
              projTmp, recTmp):

    xp = fw.frame.from_array(projs)

    if xp.isnan(projTmp[0, 0, 0]):
        projTmp[0, 0, 0] = 0
        _recFuns['project'](rec, projTmp, RinvList, zRanges)
        projTmp -= projs

    if 'filter' in _recFuns and filterName is not None:
        projTmp[...] = _recFuns['filter'](projTmp, filterName)
    recTmp.fill(0)
    _recFuns['backproject'](projTmp, recTmp, RvpList)

    rec -= reg * recTmp
    # rec[rec < 0] = 0

    _recFuns['project'](rec, projTmp, RinvList, zRanges)
    projTmp -= projs

    rms = np.linalg.norm(projTmp)

    return rms


def getPSFRecon(projs, RprojList, volSize, nProcs=1, nThreads=-1, pPool=None):

    nt, ny, nx = projs.shape
    psfProjs = np.zeros((nt, ny, nx))
    psfProjs[:, ny//2, nx//2] = 1.

    return backProjectRS(psfProjs, RprojList, volSize, nProcs, nThreads, pPool)


def getNormRecon(rec, RprojList, filterName='ramp', projSize=None, volSize=None,
               nProcs=1, nThreads=-1, pPool=None):

    projs = projectRS(np.ones_like(rec), RprojList, projSize, nProcs=nProcs,
                      nThreads=nThreads, pPool=pPool)
    print(projs.shape)
    if filterName is not None:
        projs = bpFilter(projs, filterName)
    if volSize is None:
        volSize = rec.shape
    # ATA = pr.backProjectRS(prtempf, RprojsIn, volSize=volsize)
    return backProjectRS(projs, RprojList, volSize=volSize)
