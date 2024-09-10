#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:22:45 2021

@author: joton
"""

import numpy as np
import mrcfile
import h5py
import numba
from artis_tomo.tools.parallel import ProgressParallel as Parallel
from artis_tomo.utils.parser.docopt import docopt
from artis_tomo.tools.metadataIO import writeImodTransformFile
from artis_tomo.tomo.alignment import tiltAlignOptFlow, applyAlign
from artis_tomo.image.filter import normalizeBg


def tiltAlignProgram():
    """{progName}: align tilt series projections.

    Estimate fiducialess local alignment based on Optical Flow

    Usage:
      {progName} -i <tilt_series> -o <fnOut_root> [options]

    Arguments:
        -i <tilt_series>  MRC stack file.
         -o <fnOut_root>  Rootname used for output files aligned stack and
                          transformation matrices.
    Options:
           -a <tlt_file>  IMOD's style tilt file.
             --ref <idx>  Frame index to be used as reference. By default,
                          projection at 0º will be used. [Default: -1]
           --dataset <idx>  If the input is an HDF5 the path to the dataset
                            must be specified.
           --xrange <xr>  X range used to average alignment between
                          consecutive projections around the center of the
                          tilt axis projection. [Default: 10]
           --yrange <yr>  Y range used to average alignment between
                          consecutive projections around the center of the
                          tilt axis  projection. By default, it uses the whole
                          dimension. [Default: -1]
        --center <cx,cy>  X,Y position of the reference image point to be used
                          as alignment center. If negative, the image center
                          is used. [Default: -1,-1]
           --radius <rn>  Radius of the window considered around each pixel in
                          Optical Flow. [Default: 32]
            --bgnorm_out  Apply background normalization to fix brightfield
                          non-uniformities in output aligned projections.
             --bgnorm_of  Apply background normalization before
                          estimating Optical Flow local shifts.
                   --log  Apply logarithm to projections.
           --j <threads>  Number of threads por parallel computing. If -1,
                          it uses all available cores. [Default: -1]


                 --debug  Plot frames alignment and vector shift map.
               -h --help  Show this screen.
            -v --version  Show version.

    """
    arguments = docopt(tiltAlignProgram.__doc__)
    # print(arguments)

    # Prepare input/output files
    fnStkIn = arguments.get('-i')
    fnTlt = arguments.get('-a')
    h5DataPath = arguments.get('--dataset')
    fnOutRoot = arguments.get('-o')

    # Read data
    if fnStkIn.endswith(".mrc"):
        with mrcfile.open(fnStkIn, permissive=True) as mrc_f:
            stk = mrc_f.data
    elif fnStkIn.endswith((".hdf5", ".h5")):
        with h5py.File(fnStkIn, 'r') as hdf5_f:
            stk = hdf5_f[h5DataPath][()]
    else:
        raise Exception(
            "Input stack format is not correct. It must be mrc or hdf5."
        )
    nTilt = stk.shape[0]

    # Read (optional) IMOD-style tilt angles file
    if fnTlt is not None:
        tiltList = np.loadtxt(fnTlt)
        assert nTilt == len(tiltList), "Tilt number mismatch between .tlt" +\
            f" {len(tiltList)} and stack {nTilt}."
    else:
        tiltList = np.arange(nTilt) - nTilt//2

    # Align
    stkAli, transList = tilt_align(
        stk,
        tiltList,
        refId = int(arguments.get('--ref')),
        centerStr = arguments.get('--center'),
        xRange = int(arguments.get('--xrange')),
        yRange = int(arguments.get('--yrange')),
        radius = int(arguments.get('--radius')),
        normBgOF = arguments.get('--bgnorm_of'),
        normBgOut = arguments.get('--bgnorm_out'),
        applyLog = arguments.get('--log'),
        debug = arguments.get('--debug'),
        nProcs = int(arguments.get('--j'))
    )

    # Save
    fnXFOut = fnOutRoot + '.xf'
    fnStkOut = fnOutRoot + ('.mrc' if fnStkIn.endswith(".mrc") else ".hdf5")

    print('- Writing output files:')
    print(f'   - IMOD transformation file: {fnXFOut}')
    print(f'   - Aligned stack: {fnStkOut}')

    # Trnasformations
    writeImodTransformFile(fnXFOut, transList)

    # Aligned stack
    if fnStkOut.endswith(".mrc"):
        with mrcfile.new(fnStkOut, overwrite=True) as mrc:
            mrc.set_data(stkAli.astype(np.float32))
    else:
        with h5py.File(fnStkOut, 'w') as hdf5_f_out:
            with h5py.File(fnStkIn, 'r') as hdf5_f_in:
                for _, ds in hdf5_f_in.items():
                    hdf5_f_out.copy(ds, ds.name)

            hdf5_f_out.create_dataset(h5DataPath + '_aligned', data=stkAli)
            hdf5_f_out.create_dataset(
                h5DataPath + '_transformations', data=np.stack(transList)
            )
    

def tilt_align(
        stk: np.ndarray,
        tiltList: np.ndarray,
        refId: int = -1,
        centerStr: str =  "-1,-1",
        xRange: int = 10,
        yRange: int = -1,
        radius: int = 32,
        normBgOF: bool = False,
        normBgOut: bool = False,
        applyLog: bool = False,
        debug: bool = False,
        nProcs: int = -1
):
    # Parallel configuration
    if nProcs > 0:
        numba.set_num_threads(nProcs)

    pPool = Parallel(n_jobs=nProcs, unit='Projs.')

    # Set reference index
    if refId < 0:
        refId = np.argmin(np.abs(tiltList))

    # Normalize stack
    if normBgOF or normBgOut:
        print('- Background normalization ...')
        stkNorm = normalizeBg(stk, 9)
    stkOF = stkNorm if normBgOF else stk
    stkIni = stkNorm if normBgOut else stk

    # Determine center
    cX, cY = [float(v) for v in centerStr.split(',')]

    # Align
    print('- Estimating local shifts ...')
    shifts = tiltAlignOptFlow(stkOF, refId=refId, normBg=False,
                              xRange=xRange, yRange=yRange,
                              xCenter=cX, yCenter=cY, radius=radius,
                              nProcs=nProcs, pPool=pPool, debug=debug)

    print('- Applying alignment ...')
    stkAli = applyAlign(stkIni, shifts, nProcs=nProcs, pPool=pPool)

    # Apply logarithm to the aligned stack
    if applyLog:
        print('- Applying logarithm ...')
        stkAli = np.log(stkAli)

    transList = list()
    for shift in shifts:
        transM = np.identity(3)
        transM[:2, 2] = -shift
        transList.append(transM)

    return stkAli, np.stack(transList)
