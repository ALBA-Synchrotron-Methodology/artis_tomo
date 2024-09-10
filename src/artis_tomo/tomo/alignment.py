#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:22:45 2021

@author: joton
"""

import numpy as np
from skimage.registration import optical_flow_ilk
from skimage.registration import phase_cross_correlation
from skimage import transform as tf
from joblib import Parallel, delayed
from ..image.filter import normalizeBg
from ..image.frame import padwithrc

import matplotlib.pyplot as plt
plt.ion()


def getOptFlowShifts(array, radius=32, nProcs=1, pPool=None):
    """
    Return XY shifts between consecutive stacked images.

    Parameters
    ----------
    array : ndarray, shape ([P], M, N)
        Stack of 2D images. First (optional) index is image ID

    radius : int, optional
        Radius of the window considered around each pixel.

    nProcs : int, optional
        Number of processes for paralellizing purposes. If -1, total number of
        available cpu cores is used.

    pPool : joblib parallel pool, optional
            If not provided, internal pool is created.

    Returns
    -------
    v, u : 3D arrays. 2D stacks of local Y and X shifts between images. First
           index length is N-1 images.

    """
    nt, nr, nc = array.shape
    nDims = (nt-1, nr, nc)

    vA = np.empty(nDims)
    uA = np.empty(nDims)

    if nProcs == 1:

        for k in range(0, nt-1):
            vA[k], uA[k] = optical_flow_ilk(array[k], array[k+1],
                                            radius=radius, num_warp=1)
    else:

        if pPool is None:
            pPool = Parallel(n_jobs=nProcs)

        result = pPool(delayed(optical_flow_ilk)(array[k], array[k+1],
                                                 radius=radius, num_warp=1)
                       for k in range(0, nt-1))

        for k in range(0, nt-1):
            vA[k], uA[k] = result[k]

    return vA, uA


def tiltAlignOptFlow(array, refId=-1, normBg=True, xRange=10, yRange=10,
                     xCenter=-1, yCenter=-1, radius=32,
                     nProcs=1, pPool=None, debug=False):
    """
    Tilt series alignment based on local Optical Flow shifts.

    Parameters
    ----------
    array : 3D array
        Stack of 2D projections. First index is image.
    refId : int, optional
        Index of image used as reference. If negative, center image is selected.
        The default is -1.
    normBg : boolean, optional
        Apply background normalization before OF process. The default is True.
    xRange : int, optional
        Half window size to average X shift. The default is 10.
    yRange : int, optional
        Half window size to average Y shift. If negative, half image height
         minus 50 is used. The default is 10.
    xCenter : int, optional
        X position of the reference image point to be used as alignment center.
        If negative, the image center is used. The default is -1.
    yCenter : TYPE, optional
        Y position of the reference image point to be used as alignment center.
        If negative, the image center is used. The default is -1.
    radius : int, optional
        Radius of the window considered around each pixel in OF.
    nProcs : int, optional
        Number of parallel processes. The default is 1.
    pPool : TYPE, optional
        joblib workers pool. If not provided it's created internally.
        The default is None.

    Returns
    -------
    shifts : 2D array
        XY shifts to align input array. Reference image shifts are zero.

    """
    nt, nr, nc = array.shape

    tC = nt//2
    xC = nc//2
    if xC % 2 == 1:
        xC -= 0.5  # To match IMOD reference system
    yC = nr//2

    if refId < 0:
        refId = tC
    if yRange < 0:
        yRange = nr//2 - 50
    if xCenter < 0:
        xCenter = xC
    if yCenter < 0:
        yCenter = yC

    roi = getROI((nr, nc), (yCenter, xCenter), (2*yRange, 2*xRange),
                 mode='margin', margin=50)

    if normBg:
        array = normalizeBg(array, 9)

    mean = array.mean((1, 2))[:, None, None]
    std = array.std((1, 2))[:, None, None]
    array = (array - mean)/std

    vA, uA = getOptFlowShifts(array, radius, nProcs, pPool)

    shifts = np.zeros((nt, 2))
    shifts[refId, :] = [xCenter - xC, yCenter - yC]

    center = np.zeros((nt, 2))
    center[refId, :] = [xCenter, yCenter]

    if debug:
        fig, ax = plt.subplots(2, 2, num=1, clear=True)

    for k in range(refId, nt-1):

        (xCenterK, yCenterK) = [xC, yC] + shifts[k, :].astype(int)

        roi = getROI((nr, nc), (yCenterK, xCenterK), (2*yRange, 2*xRange),
                     mode='margin', margin=30)

        shifts[k+1, 0] = shifts[k, 0] + np.mean(uA[k][roi])
        shifts[k+1, 1] = shifts[k, 1] + np.mean(vA[k][roi])

        center[k+1, :] = [xC, yC] + shifts[k+1, :].astype(int)

        if debug:
            _ploFrametOF(array[k], array[k+1], roi, uA[k], vA[k],
                         centerRef=center[k, :], fig_ax=(fig, ax),
                         label=f'Projections {k} and {k+1}')

    for k in range(refId, 0, -1):

        (xCenterK, yCenterK) = [xC, yC] + shifts[k, :].astype(int)

        roi = getROI((nr, nc), (yCenterK, xCenterK), (2*yRange, 2*xRange),
                     mode='margin', margin=30)

        shifts[k-1, 0] = shifts[k, 0] - np.mean(uA[k-1][roi])
        shifts[k-1, 1] = shifts[k, 1] - np.mean(vA[k-1][roi])

        center[k-1, :] = [xC, yC] + shifts[k-1, :].astype(int)

        if debug:
            _ploFrametOF(array[k], array[k-1], roi, -uA[k-1], -vA[k-1],
                         centerRef=center[k, :], fig_ax=(fig, ax),
                         label=f'Projections {k} and {k-1}')

    if debug:
        plt.close(fig)

    return shifts


def getROI(shape, center, window, mode='default', margin=[0, ]):
    """
    Return tuple of slices with ROI to access in arrays.

    Parameters
    ----------
    shape : tuple
        Array shape to get the ROI.
    center : tuple
        Coordinates of ROI center.
    window : tuple
        ROI width for each dimension. If (N,) is provided it's used for all
        dimensions.
    mode : string, optional
        How to proceed if ROI indices are out of shape. Options are
            - crop: negative and > shape indexes are truncated.
            - margin: keeps the ROI within the array shape by shifting the
                      center and add a margin from the edge.
            - default: negative and > shape indexes are not modified.
    margin : list, optional
        Minimum gap between ROI window and edges per dimension. If [N,] is
        provided it's used for all dimensions. The default is [0, ].

    Returns
    -------
    roiSlice : tuple
        Indexes slices per dimension to obtain an array ROI.

    """
    shape = np.array(shape)
    nDims = len(shape)

    wnDims = len(window)
    if wnDims < nDims:
        window = np.concatenate([window, np.ones(nDims - wnDims)*window[-1]])

    margin = np.atleast_1d(margin)
    mnDims = len(margin)
    if mnDims < nDims:
        margin = np.concatenate([margin, np.ones(nDims - mnDims)*margin[-1]])

    window2 = np.array(window)//2

    roi = np.zeros((nDims, 2), dtype=int)

    for k in range(nDims):
        posIni = center[k] - window2[k]  # Initial index
        posEnd = posIni + window[k]  # Final index

        if mode == 'crop':
            if posIni < 0:
                posIni = 0
            if posEnd > shape[k]:
                posEnd = shape[k]
        elif mode == 'margin':
            if posIni < margin[k]:
                posIni = margin[k]
                posEnd = posIni + window[k]
            elif posEnd > shape[k] - margin[k]:
                posEnd = shape[k] - margin[k]
                posIni = posEnd - window[k]
        elif mode == 'default':
            pass

        roi[k, 0] = posIni
        roi[k, 1] = posEnd

    roiSlice = tuple(slice(*x) for x in roi)

    return roiSlice


def applyAlign(array, shifts, taper=32, nProcs=1, pPool=None):
    """
    Apply shifts to align 2D images stack.

    Parameters
    ----------
    array : 3D array
        Stack of 2D projections.
    shifts : 2D array
        XY shifts to align input array. shifts[:, 0] are X values.
    taper : int, optional
        Number of pixels from the edge to estimate mean and use it for tapering.
        If zero, shifted image is padded with zeros. The default is 32.
    nProcs : int, optional
        Number of parallel processes. The default is 1.
    pPool : TYPE, optional
        joblib workers pool. If not provided it's created internally.
        The default is None.

    Returns
    -------
    arrayOut : 3D array
        Stack of aligned 2D projections.

    """
    nt = array.shape[0]

    arrayOut = np.empty_like(array)

    if nProcs == 1:
        for k in range(nt):
            arrayOut[k] = _applyAlignOne(array[k], shifts[k], taper)
    else:
        if pPool is None:
            pPool = Parallel(n_jobs=nProcs)

        result = pPool(delayed(_applyAlignOne)(array[k], shifts[k], taper)
                       for k in range(nt))

        for k in range(nt):
            arrayOut[k] = result[k]
    return arrayOut


def _applyAlignOne(image, shifts, taper):

    doTaper = taper > 0
    if doTaper:
        dims = image.shape
        pads = list()
        for shift in reversed(shifts):  # shifts[0] == X
            shiftInt = int(np.ceil(np.abs(shift)))
            pads.append((shiftInt, 0) if shift < 0 else (0, shiftInt))

        roiSlice = tuple(slice(pad[0], pad[0]+dims[k])
                         for k, pad in enumerate(pads))

        # imgTmp = np.pad(image, pads, mode='mean', stat_length=(taper,))
        imgTmp = np.pad(image, pads, mode=padwithrc,
                        end_values=image.mean(), rcpad=taper)
    else:
        imgTmp = image

    tform = tf.SimilarityTransform(translation=shifts)
    imgTmp = tf.warp(imgTmp, tform, order=5)

    if doTaper:
        imgTmp = imgTmp[roiSlice]

    return imgTmp




# TODO: def alignIter(prj, ang, fdir='.', iters=10, pad=(0, 0),
#         blur=True, center=None, algorithm='sirt',
#         upsample_factor=10, rin=0.5, rout=0.8,
#         save=False, debug=True):
#     """
#     Aligns the projection image stack using the joint
#     re-projection algorithm :cite:`Gursoy:17`.

#     Parameters
#     ----------
#     prj : ndarray
#         3D stack of projection images. The first dimension
#         is projection axis, second and third dimensions are
#         the x- and y-axes of the projection image, respectively.
#     ang : ndarray
#         Projection angles in radians as an array.
#     iters : scalar, optional
#         Number of iterations of the algorithm.
#     pad : list-like, optional
#         Padding for projection images in x and y-axes.
#     blur : bool, optional
#         Blurs the edge of the image before registration.
#     center: array, optional
#         Location of rotation axis.
#     algorithm : {str, function}
#         One of the following string values.

#         'art'
#             Algebraic reconstruction technique :cite:`Kak:98`.
#         'gridrec'
#             Fourier grid reconstruction algorithm :cite:`Dowd:99`,
#             :cite:`Rivers:06`.
#         'mlem'
#             Maximum-likelihood expectation maximization algorithm
#             :cite:`Dempster:77`.
#         'sirt'
#             Simultaneous algebraic reconstruction technique.
#         'tv'
#             Total Variation reconstruction technique
#             :cite:`Chambolle:11`.
#         'grad'
#             Gradient descent method with a constant step size

#     upsample_factor : integer, optional
#         The upsampling factor. Registration accuracy is
#         inversely propotional to upsample_factor.
#     rin : scalar, optional
#         The inner radius of blur function. Pixels inside
#         rin is set to one.
#     rout : scalar, optional
#         The outer radius of blur function. Pixels outside
#         rout is set to zero.
#     save : bool, optional
#         Saves projections and corresponding reconstruction
#         for each algorithm iteration.
#     debug : book, optional
#         Provides debugging info such as iterations and error.

#     Returns
#     -------
#     ndarray
#         3D stack of projection images with jitter.
#     ndarray
#         Error array for each iteration.
#     """

#     # Needs scaling for skimage float operations.
#     prj, scl = scale(prj)

#     # Shift arrays
#     sx = np.zeros((prj.shape[0]))
#     sy = np.zeros((prj.shape[0]))

#     conv = np.zeros((iters))

#     # Pad images.
#     npad = ((0, 0), (pad[1], pad[1]), (pad[0], pad[0]))
#     prj = np.pad(prj, npad, mode='constant', constant_values=0)

#     # Initialization of reconstruction.
#     rec = 1e-12 * np.ones((prj.shape[1], prj.shape[2], prj.shape[2]))

#     extra_kwargs = {}
#     if algorithm != 'gridrec':
#         extra_kwargs['num_iter'] = 1

#     # Register each image frame-by-frame.
#     for n in range(iters):

#         if np.mod(n, 1) == 0:
#             _rec = rec

#         # Reconstruct image.
#         rec = recon(prj, ang, center=center, algorithm=algorithm,
#                     init_recon=_rec, **extra_kwargs)

#         # Re-project data and obtain simulated data.
#         sim = project(rec, ang, center=center, pad=False)

#         # Blur edges.
#         if blur:
#             _prj = blur_edges(prj, rin, rout)
#             _sim = blur_edges(sim, rin, rout)
#         else:
#             _prj = prj
#             _sim = sim

#         # Initialize error matrix per iteration.
#         err = np.zeros((prj.shape[0]))

#         # For each projection
#         for m in range(prj.shape[0]):

#             # Register current projection in sub-pixel precision
#             shift, error, diffphase = phase_cross_correlation(
#                     _prj[m], _sim[m], upsample_factor=upsample_factor)
#             err[m] = np.sqrt(shift[0]*shift[0] + shift[1]*shift[1])
#             sx[m] += shift[0]
#             sy[m] += shift[1]

#             # Register current image with the simulated one
#             tform = tf.SimilarityTransform(translation=(shift[1], shift[0]))
#             prj[m] = tf.warp(prj[m], tform, order=5)

#         if debug:
#             print('iter=' + str(n) + ', err=' + str(np.linalg.norm(err)))
#             conv[n] = np.linalg.norm(err)

#         if save:
#             write_tiff(prj, 'tmp/iters/prj', n)
#             write_tiff(sim, 'tmp/iters/sim', n)
#             write_tiff(rec, 'tmp/iters/rec', n)

#     # Re-normalize data
#     prj *= scl
#     return prj, sx, sy, conv


def _ploFrametOF(imref, image, roi, u=None, v=None,
                 centerRef=None, center=None, fig_ax=None, label=None):
    nr, nc = imref.shape

    if u is None or v is None:
        v, u = optical_flow_ilk(imref, image)

    xoffset = np.mean(u[roi])
    yoffset = np.mean(v[roi])

    imref = imref - imref.min()
    image = image - image.min()

    # Register current image with the simulated one
    tform = tf.SimilarityTransform(translation=(xoffset, yoffset))
    image1_Nowarp = tf.warp(image, tform, order=5)

    # build an RGB image with the unregistered sequence
    seq_im = np.zeros((nr, nc, 3))
    seq_im[..., 0] = image
    seq_im[..., 1] = imref
    seq_im[..., 2] = imref

    # build an RGB image with the registered sequence
    target_im = np.zeros((nr, nc, 3))
    target_im[..., 0] = image1_Nowarp
    target_im[..., 1] = imref
    target_im[..., 2] = imref

    # --- Compute flow magnitude
    # norm = np.sqrt(u ** 2 + v ** 2)

    # OF Plotting
    if fig_ax is None:
        fig, ax = plt.subplots(2, 2, num=1, clear=True)
    else:
        fig, ax = fig_ax
        for ar in ax:
            for ac in ar:
                ac.cla()

    ax[0][0].imshow(seq_im/seq_im.max())
    ax[0][0].set_title("Unaligned sequence")
    ax[0][0].set_axis_off()
    if centerRef is not None:
        ax[0][0].plot(*centerRef, 'og')
        center = centerRef + np.array([xoffset, yoffset])
        ax[0][0].plot(*center, 'xr')

    ax[0][1].imshow(target_im/target_im.max())
    ax[0][1].set_title("Aligned sequence")
    # ax[0][1].plot([xC-xRange]*2, [0, nr], 'r')
    # ax[0][1].plot([xC+xRange]*2, [0, nr], 'r')
    ax[0][1].set_axis_off()

    if centerRef is not None:
        ax[0][1].plot(*centerRef, 'xr')
    ax[0][1].plot([roi[1].start, roi[1].stop], [roi[0].start]*2, 'r')
    ax[0][1].plot([roi[1].start, roi[1].stop], [roi[0].stop]*2, 'r')
    ax[0][1].plot([roi[1].start]*2, [roi[0].start, roi[0].stop], 'r')
    ax[0][1].plot([roi[1].stop]*2, [roi[0].start, roi[0].stop], 'r')

    nvec = 20  # Number of vectors to be displayed along each image dimension
    step = max(nr//nvec, nc//nvec)

    y, x = np.mgrid[:nr:step, :nc:step]
    u_ = u[::step, ::step]
    v_ = v[::step, ::step]

    ax[1][0].imshow(imref)
    ax[1][0].quiver(x, y, u_, v_, color='r', units='dots',
                    angles='xy', scale_units='xy', lw=3)
    ax[1][0].set_title("Shifts vector field")
    # ax[1][0].plot([xC-xRange]*2, [0, nr], 'r')
    # ax[1][0].plot([xC+xRange]*2, [0, nr], 'r')
    ax[1][0].set_axis_off()

    ax[1][1].plot(np.mean(u, 0))
    ax[1][1].plot(np.mean(v, 1))
    ax[1][1].set_title("X, Y shifts - avg")
    ax[1][1].legend(['X shift per column', 'Y shift per row'])
    # ax[1][2].set_axis_off()

    # fig.tight_layout()
    if label:
        fig.suptitle(label)
    # plt.pause(1)
    fig.show()
    fig.canvas.draw()
    plt.waitforbuttonpress()
