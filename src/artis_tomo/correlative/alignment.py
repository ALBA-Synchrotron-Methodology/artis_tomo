#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-modal alignment steps of the correlative pipeline.

Registering a fluorescence volume against a soft X-ray tomogram is done in
three stages, each narrowing the search for the next:

1. :func:`alignFluoToMosaic` -- a fluorescence projection against the X-ray
   mosaic, searching the full 360 degrees in plane.
2. :func:`alignTomoToMosaic` -- the tomogram's zero-degree projection against
   the same mosaic, giving the tomogram's place in the mosaic frame.
3. :func:`alignTomoToFluoVolume` -- refines the 3D orientation, perturbing it
   over a spherical cap and scoring each candidate by back-projecting the
   cross-correlations of a subset of tilts.

The two modalities have no intensity relationship, so every correlation is
between feature-size-filtered images rather than the raw data; that is what
makes the cross-modal match possible at all.
"""

import copy
import time

import numpy as np

from artis_sci.image import frame
from artis_sci.image import alignment as ali
from artis_sci.image import transformation as tr
from artis_sci.image.filter import filterFeaturesPipeline
from artis_sci.io import imageIO as iio
from artis_sci.math import transforms as tf
from artis_sci.math.transforms import sphereSurfaceMatrices

from ..tomo import project as pr
from ..tomo.project import projectVolume
from ..tomo.utils import selectTiltSubset
from .preprocessing import (normalizeSXTMosaic, preprocessFluoImage,
                            preprocessMosaic, preprocessTomoProjection)


def saveDebugImage(img, debugDir, name, ext='.tif'):
    """
    Write an intermediate image, if a debug directory was given.

    The single place in this subpackage that touches the filesystem. Passing
    ``None`` for *debugDir* disables it, which is the normal case.

    Parameters
    ----------
    img : array_like
        Image or stack to write.
    debugDir : str or Path or None
        Directory to write into, or None to do nothing.
    name : str
        File name without extension.
    ext : str, optional
        Extension. Default '.tif'; the volume and stack dumps use '.mrc'.
    """
    if debugDir is None:
        return None

    return iio.writeImage(img, debugDir, name, ext=ext)


def _featureFilter(img, pix, dim_range, preprocess=None, feat_filter='dot',
                   **preprocess_kw):
    """Feature-size filter, in the form the steps below want it."""
    return filterFeaturesPipeline(img, pix, dim_range, preprocess=preprocess,
                                  method=feat_filter, **preprocess_kw)


def alignFluoToMosaic(fnmosaic, fluovol, tol, useGPU, flPix, mPix, ang_params,
                      dim_range, thr, preprocess=None, feat_filter='dot',
                      preprocess_kw=None, debugDir=None):
    """
    Step 1: align a fluorescence projection to the X-ray mosaic.

    Both are reduced to their features in the target size range before
    correlating, over a full in-plane angular search.

    Parameters
    ----------
    fnmosaic : str or Path
        Mosaic image file, raw transmission.
    fluovol : 3D array_like
        Fluorescence volume.
    tol : float
        Segmentation tolerance for the mosaic normalisation.
    useGPU : int or None
        CUDA device for the projection, or None for CPU.
    flPix, mPix : float
        Fluorescence and mosaic pixel sizes.
    ang_params : tuple
        (start, stop, step) of the in-plane search, in degrees.
    dim_range : tuple
        (min, max) feature size to keep, in the same units as the pixel sizes.
    thr : int
        Worker count for the correlation.
    preprocess : str or None
        Intensity preprocessing before the size filter.
    feat_filter : str
        Size filter method.
    preprocess_kw : dict or None
        Extra arguments for the preprocessing.
    debugDir : str or Path or None
        Where to write intermediates, or None.

    Returns
    -------
    TMat3D
        Fluorescence to mosaic transformation, shifts in real-space units.
    """
    if preprocess_kw is None:
        preprocess_kw = {}

    start_imp = time.time()
    mosaic = iio.readImage2D(fnmosaic).astype(float)
    mosaic = normalizeSXTMosaic(mosaic, tol)
    mosaic_scaled = tr.rescaleFourier(mosaic, flPix / mPix)
    mosaic_fil = _featureFilter(mosaic_scaled, flPix, dim_range, preprocess,
                               feat_filter, **preprocess_kw)
    xrSizeSc = mosaic_fil.shape

    img_fluo = projectVolume(fluovol, useGPU=useGPU, label='Proj FLUO')[0]
    img_fluo = preprocessFluoImage(img_fluo, xrSizeSc)
    print("finishing import images. "
          f"time elapsed: {round(time.time() - start_imp, 2)} s")

    start_proc = time.time()
    img_fluo_fil = _featureFilter(img_fluo, flPix, dim_range, None,
                                 feat_filter, **preprocess_kw)
    print("finishing processing images. "
          f"time elapsed: {round(time.time() - start_proc, 2)} s")

    # The fluorescence is what moves; the mosaic is the reference.
    ang_fm, sft_fm, _ = ali.alignInplaneBF(img_fluo_fil, mosaic_fil,
                                           ang_params, thr=thr, shrink=0.5)

    sft_ls = [sft_fm[0], sft_fm[1], 0]
    sft_abs = [x * flPix for x in sft_ls]
    ang_fm = [np.deg2rad(ang_fm), 0, 0]

    m_scaled = tf.tr3d.angles2mat(ang_fm, shifts=sft_ls)
    fluofix = tr.transformRS2D(img_fluo, m_scaled)
    m_fm_mo = tf.tr3d.angles2mat(ang_fm, sft_abs)

    saveDebugImage(mosaic_scaled, debugDir, "step1_mosaic_scaled")
    saveDebugImage(mosaic, debugDir, "step1_mosaic_processed")
    saveDebugImage(mosaic_fil, debugDir, "step1_mosaic_tophat_toAlign")
    saveDebugImage(img_fluo_fil, debugDir, "step1_fluo_tophat_toAlign")
    saveDebugImage(fluofix, debugDir, "step1_fluo_correlated")
    saveDebugImage(img_fluo, debugDir, "step1_fluo_original")

    return m_fm_mo


def alignFluoImageToMosaic(fnmosaic, fluoimg, tol, useGPU, flPix, mPix,
                           ang_params, dim_range, thr, preprocess=None,
                           feat_filter='dot', preprocess_kw=None,
                           debugDir=None):
    """
    Align a fluorescence *image* to the X-ray mosaic, for the 2D pipeline.

    The 2D counterpart of :func:`alignFluoToMosaic`, differing in more than the
    input being an image rather than a volume: the mosaic is resampled before
    being converted to absorption contrast rather than after, and the
    fluorescence is put through the same intensity preprocessing as the mosaic
    instead of skipping it.

    Parameters
    ----------
    fnmosaic : str or Path
        Mosaic image file, raw transmission.
    fluoimg : 2D array_like
        Fluorescence image.
    tol : float
        Segmentation tolerance for the mosaic normalisation.
    useGPU : int or None
        Unused here, kept for symmetry with the 3D step.
    flPix, mPix : float
        Fluorescence and mosaic pixel sizes.
    ang_params : tuple
        (start, stop, step) of the in-plane search, in degrees.
    dim_range : tuple
        (min, max) feature size to keep.
    thr : int
        Worker count for the correlation.
    preprocess : str or None
        Intensity preprocessing before the size filter.
    feat_filter : str
        Size filter method.
    preprocess_kw : dict or None
        Extra arguments for the preprocessing.
    debugDir : str or Path or None
        Where to write the filtered inputs and the original, or None.

    Returns
    -------
    m_fm_mo : TMat3D
        Fluorescence to mosaic transformation, shifts in real-space units.
    mosaic : 2D ndarray
        The resampled, normalised mosaic.
    fluoCorrelated : 2D ndarray
        The fluorescence image moved onto the mosaic.
    """
    if preprocess_kw is None:
        preprocess_kw = {}

    start_imp = time.time()
    mosaic = iio.readImage2D(fnmosaic).astype(float)
    mosaic = tr.rescaleFourier(mosaic, flPix / mPix)
    mosaic = normalizeSXTMosaic(mosaic, tol)
    mosaic_fil = _featureFilter(mosaic, flPix, dim_range, preprocess,
                               feat_filter, **preprocess_kw)
    xrSizeSc = mosaic_fil.shape

    img_fluo = preprocessFluoImage(fluoimg, xrSizeSc, pix=flPix, sigma=5)
    print("finishing import images. "
          f"time elapsed: {round(time.time() - start_imp, 2)} s")

    start_proc = time.time()
    img_fluo_fil = _featureFilter(img_fluo, flPix, dim_range, preprocess,
                                 feat_filter, **preprocess_kw)
    print("finishing processing images. "
          f"time elapsed: {round(time.time() - start_proc, 2)} s")

    # The fluorescence is what moves; the mosaic is the reference.
    ang_fm, sft_fm, _ = ali.alignInplaneBF(img_fluo_fil, mosaic_fil,
                                           ang_params, thr=thr, shrink=0.5)

    sft_ls = [sft_fm[0], sft_fm[1], 0]
    sft_abs = [x * flPix for x in sft_ls]
    ang_fm = [np.deg2rad(ang_fm), 0, 0]

    m_scaled = tf.tr3d.angles2mat(ang_fm, shifts=sft_ls)
    fluofix = tr.transformRS2D(img_fluo, m_scaled)
    m_fm_mo = tf.tr3d.angles2mat(ang_fm, sft_abs)

    saveDebugImage(mosaic_fil, debugDir, "mosaic_tophat_toAlign")
    saveDebugImage(img_fluo_fil, debugDir, "fluo_tophat_toAlign")
    saveDebugImage(img_fluo, debugDir, "fluo_original")

    return m_fm_mo, mosaic, fluofix


def alignTomoToMosaic(fnmosaic, tomo, noref_params, xrpix, mPix, flPix, tol,
                      thr, debugDir=None):
    """
    Step 2: align the tomogram's zero-degree projection to the mosaic.

    Correlates without an angular search -- the tilt series and the mosaic come
    from the same microscope, so only the shift is unknown. The alignment is
    stored on *tomo*.

    Parameters
    ----------
    fnmosaic : str or Path
        Mosaic image file, raw transmission.
    tomo : Tomogram
        Tomogram, whose alignment matrix this sets.
    noref_params : tuple
        Angular search parameters, normally (0, 0, 1) for shift only.
    xrpix, mPix, flPix : float
        X-ray, mosaic and fluorescence pixel sizes.
    tol : float
        Segmentation tolerance for the mosaic normalisation.
    thr : int
        Worker count for the correlation.
    debugDir : str or Path or None
        Where to write intermediates, or None.

    Returns
    -------
    tomo : Tomogram
        With its alignment matrix set, shifts in real-space units.
    mosaic_scaled : 2D ndarray
        The mosaic as correlated against.
    """
    imgXr = iio.readImage2D(fnmosaic).astype(float)
    mosaic = normalizeSXTMosaic(imgXr, tol)
    mosaic_scaled = preprocessMosaic(mosaic, flPix, mPix)

    tomoCentral = tomo.getImage('center')
    tomo_img_scaled = preprocessTomoProjection(tomoCentral, flPix, xrpix,
                                               mosaic_scaled.shape)

    # The tomogram projection is what moves; the mosaic is the reference.
    ang_th, sft_th, ccimg = ali.alignInplaneBF(tomo_img_scaled, mosaic_scaled,
                                               noref_params, thr=thr,
                                               shrink=0.8, ccTophat=True)

    saveDebugImage(ccimg, debugDir, "step2_cc_central")

    ang_th = [np.deg2rad(ang_th), 0, 0]
    sft_th = list(sft_th)
    sft_th.append(0)
    sft_real = [x * flPix for x in sft_th]
    m_st_mo_real = tf.tr3d.angles2mat(ang_th, sft_real)

    tomo.setAlignTMat(m_st_mo_real)

    if debugDir is not None:
        # The alignment overlaid on the *unprocessed* mosaic, which is easier to
        # judge by eye than the flattened version actually correlated against.
        mosaic_orig = iio.readImage2D(fnmosaic).astype(float)
        mosaic_orig_scaled = tr.rescaleFourier(mosaic_orig, flPix / mPix)

        tomoCentral = -1. * tomo.getImage('center')
        tomoCentral = tomoCentral - tomoCentral.min()
        tomo_img_vis = tr.rescaleFourier(tomoCentral, flPix / xrpix)
        tomo_img_vis = frame.padCropArrayCentered(tomo_img_vis,
                                                  mosaic_orig_scaled.shape,
                                                  mode='constant')[0]

        # The stored matrix is in real-space units; transforming pixels needs
        # it in pixels, so convert a copy rather than the tomogram's own.
        m_st_mo_pix = copy.deepcopy(m_st_mo_real)
        m_st_mo_pix.matrix[:, 0:3, 3] = m_st_mo_pix.matrix[:, 0:3, 3] / flPix
        st_fix = tr.transformRS2D(tomo_img_vis, m_st_mo_pix)
        print("MATRICES :", m_st_mo_pix, m_st_mo_real)

        saveDebugImage(mosaic_orig_scaled, debugDir,
                       "step2_mosaic_orig_fluoScale")
        saveDebugImage(tomo_img_vis, debugDir, "step2_xrprojcenter_fluoScale")
        saveDebugImage(st_fix, debugDir,
                       "step2_xrprojcenter_fluoScale_mosaicAligned")

    return tomo, mosaic_scaled


def alignTomoToFluoVolume(fluovol, tomo, xrPix, flPix, perturb, projs,
                          ref_params, dim_range, m_fm_mo, useGPU, thr,
                          preprocess=None, feat_filter='dot',
                          preprocess_kw=None, debugDir=None, verbose=False):
    """
    Step 3: refine the tomogram's 3D orientation against the fluorescence.

    Starts from the pre-alignment of steps 1 and 2, refines the in-plane angle
    at zero tilt, then searches orientations spread over a spherical cap. Each
    candidate is scored by projecting the fluorescence volume along the tilt
    geometry, correlating against the real tilt images, and back-projecting
    those correlations into a volume whose peak measures the fit.

    .. note::
       The shift columns of *m_fm_mo* and of the tomogram's alignment matrix
       are divided by *flPix* in place, converting them from real-space to
       pixel units. Callers that need the real-space values afterwards must
       copy them beforehand.

    Parameters
    ----------
    fluovol : 3D array_like
        Fluorescence volume.
    tomo : Tomogram
        Tomogram, aligned to the mosaic by step 2.
    xrPix, flPix : float
        X-ray and fluorescence pixel sizes.
    perturb : tuple
        (angular step, maximum polar angle) of the orientation search, degrees.
    projs : int
        How many tilts to use for scoring.
    ref_params : tuple
        (start, stop, step) of the in-plane refinement, in degrees.
    dim_range : tuple
        (min, max) feature size to keep.
    m_fm_mo : TMat3D
        Fluorescence to mosaic transformation from step 1.
    useGPU : int or None
        CUDA device for the projections, or None for CPU.
    thr : int
        Worker count for the correlations.
    preprocess : str or None
        Intensity preprocessing before the size filter.
    feat_filter : str
        Size filter method.
    preprocess_kw : dict or None
        Extra arguments for the preprocessing.
    debugDir : str or Path or None
        Where to write intermediates, or None.
    verbose : bool
        Print the intermediate transformations and per-perturbation detail.

    Returns
    -------
    TMat3D
        Tomogram volume to fluorescence volume transformation, shifts in pixel
        units.
    """
    if preprocess_kw is None:
        preprocess_kw = {}

    # Stack to mosaic, and fluo to mosaic, in pixel units. Both are modified in
    # place, as documented above.
    m_st_mo = tomo.getAlignTmat()
    m_st_mo.matrix[:, 0:3, 3] = m_st_mo.matrix[:, 0:3, 3] / flPix
    m_fm_mo.matrix[:, 0:3, 3] = m_fm_mo.matrix[:, 0:3, 3] / flPix

    # Stack to fluo
    m_st_fm = m_fm_mo.inv * m_st_mo

    imfluo = projectVolume(fluovol, useGPU=useGPU, label='Proj FLUO')[0]

    # The X-ray projection is filtered, then rescaled to the fluorescence
    # sampling and pre-aligned with what steps 1 and 2 found.
    imst0 = tomo.getImage('center')
    imst0_filt = _featureFilter(imst0, xrPix, dim_range, preprocess,
                               feat_filter, **preprocess_kw)
    imst0_filt_sc = preprocessTomoProjection(imst0_filt, flPix, xrPix,
                                             imfluo.shape)
    imst0_preali_filt = tr.transformRS2D(imst0_filt_sc, m_st_fm)

    # Both modalities are feature-filtered before correlating, as in step 1, so
    # only features in the target size range drive the match.
    imfluo_filt = _featureFilter(imfluo, flPix, dim_range, preprocess,
                                feat_filter, **preprocess_kw)

    # The X-ray projection is what moves; the fluorescence is the reference.
    ang_th, sft_th, cc_im = ali.alignInplaneBF(
        imst0_preali_filt - imst0_preali_filt.min(),
        imfluo_filt - imfluo_filt.min(),
        ref_params, thr=thr, shrink=0.05, ccTophat=True)

    sft_ls = [sft_th[0], sft_th[1], 0]
    ang_fm = [np.deg2rad(ang_th), 0, 0]
    m_corrected = tf.tr3d.angles2mat(ang_fm, sft_ls)
    m_st_fm = m_corrected * m_st_fm

    if verbose:
        print("DEBUG STEP3: Inplane prealignment refine search angles: ",
              ref_params)
        print("DEBUG STEP3: Inplane prealignment: ",
              "\n - Angle: ", ang_th,
              "\n - Shifts: ", sft_th)
        print("DEBUG STEP3: imst0 to imfluo transform Correction: ",
              m_corrected)
        print("DEBUG STEP3: imst0 to imfluo full transform: ", m_st_fm)

    if debugDir is not None:
        saveDebugImage(imfluo, debugDir, "step3_fluoproj0_orig")

        imst0_preali = preprocessTomoProjection(imst0, flPix, xrPix,
                                                imfluo.shape)
        imst0_preali = tr.transformRS2D(imst0_preali, m_st_fm)
        saveDebugImage(imst0_preali, debugDir, "step3_xrproj0_prealigned")

        saveDebugImage(imfluo_filt, debugDir, "step3_fluoproj0_filtered")
        saveDebugImage(imst0_preali_filt, debugDir,
                       "step3_xrproj0_filtered_prealigned")
        imst0_filt_aligned = tr.transformRS2D(imst0_preali_filt, m_corrected)
        saveDebugImage(imst0_filt_aligned, debugDir,
                       "step3_xrproj0_filtered_fluoAligned")
        saveDebugImage(cc_im, debugDir, "step3_prealign_cc_image")

        # Inverted and background-shifted, purely so the overlay is legible
        imst0_vis = tr.rescaleFourier(-1. * imst0, flPix / xrPix)
        imst0_vis -= imst0_vis.min()
        imst0_vis = frame.padCropArrayCentered(imst0_vis, imfluo.shape,
                                               end_values=imst0_vis.min(),
                                               rcpad=2)[0]
        imst0_vis_aligned = tr.transformRS2D(imst0_vis, m_st_fm)
        saveDebugImage(imst0_vis_aligned, debugDir, "step3_xrproj0_fluoAligned")

    # ---------------------------- 3D refinement ----------------------------
    m_pertub = sphereSurfaceMatrices(perturb[0], perturb[1])
    proj_list = selectTiltSubset(tomo.getTiltAngles(), projs)
    zoom_tf = xrPix / flPix
    m_rec = tomo.getRecTMat()[proj_list]
    m_rec.matrix[:, 0:3, 3] = m_rec.matrix[:, 0:3, 3] * zoom_tf
    m_rec_ang = tf.tr3d.removeShifts(m_rec)

    tomo_ts = tomo.getTiltseries()[proj_list]

    count = 0
    ccmax_ls = []
    shift_ls = []
    angle_ls = []

    if verbose:
        print("Number of perturbations: ", len(m_pertub))

    for m in m_pertub:
        if verbose:
            print("Perturbation matrix: ", m)

        st_list = []
        fm_list = []
        cc_list = []
        count += 1

        m_rec_perturb = m_rec * m * m_st_fm.inv
        fluo_projstk = projectVolume(fluovol, m_rec_perturb, useGPU=useGPU,
                                     label='Proj FLUO')

        for imfluo_p, img_ts in zip(fluo_projstk, tomo_ts):
            # The correlation that drives the 3D fit, again between filtered
            # versions of the synthetic fluorescence projection and the real
            # tilt image.
            img_ts_fil = _featureFilter(img_ts, xrPix, dim_range, preprocess,
                                       feat_filter, **preprocess_kw)
            img_ts_fil = preprocessTomoProjection(img_ts_fil, flPix, xrPix,
                                                  imfluo_p.shape)
            imfluo_p_filt = _featureFilter(imfluo_p, flPix, dim_range,
                                          preprocess, feat_filter,
                                          **preprocess_kw)

            start_align = time.time()
            # The tilt image is what moves; the fluorescence projection is
            # the reference.
            ang, sft, cc_img = ali.alignInplaneBF(img_ts_fil, imfluo_p_filt,
                                                  (0, 0, 1), thr=thr,
                                                  shrink=0.05, ccTophat=True)
            finish_align = time.time()

            sft_st = [sft[0], sft[1], 0]
            ang_st = [np.deg2rad(ang), 0, 0]
            mat_corr = tf.tr3d.angles2mat(ang_st, sft_st)

            start_trans2d = time.time()
            img_ts1 = tr.transformRS2D(img_ts, mat_corr)
            finish_trans2d = time.time()

            st_list.append(img_ts1)
            fm_list.append(imfluo_p)
            cc_list.append(cc_img)

        st_stack = np.stack(st_list)
        fm_stack = np.stack(fm_list)
        cc_stack = np.stack(cc_list)

        # The correlations are back-projected with the tilt geometry, so the
        # peak of the resulting volume scores this orientation in 3D.
        cc_shape = cc_stack.shape
        vol_shape = (cc_shape[1], int(0.2 * cc_shape[2]),
                     int(0.2 * cc_shape[1]))
        rec_cc = pr.backProjectRS(cc_stack, m_rec_ang, vol_shape)

        shifts = ali.getAlignValue(rec_cc, 0.05)
        sft_th = shifts[::-1]

        print("Shifts: ", shifts, sft_th)
        print("ALIGNING. "
              f"time elapsed: {round(finish_align - start_align, 2)} s")
        print("TRANSFORM 2D. "
              f"time elapsed: {round(finish_trans2d - start_trans2d, 2)} s")

        ang, _ = m.mat2params(axes=tf.SZYZ)

        ccmax_ls.append(rec_cc.argmax())
        shift_ls.append(sft_th)
        angle_ls.append(np.rad2deg(ang))

        if debugDir is not None:
            saveDebugImage(rec_cc, debugDir, "step3_vol_CC_%s" % count,
                           ext=".mrc")
            saveDebugImage(cc_stack, debugDir, "step3_stack_CC_%s" % count,
                           ext=".mrc")
            saveDebugImage(fm_stack, debugDir, "step3_stack_FM_%s" % count,
                           ext=".mrc")
            saveDebugImage(st_stack, debugDir, "step3_stack_TS_%s" % count,
                           ext=".mrc")

    posmax = ccmax_ls.index(max(ccmax_ls))
    print("posmax: ", posmax)
    print(m_pertub[posmax])
    print("angles and shifts: ", angle_ls[posmax], shift_ls[posmax])

    m_shift_corrected = tf.tr3d.shifts2mat(shift_ls[posmax])
    m_trans_tomo = m_st_fm * m_pertub[posmax].inv * m_shift_corrected
    print("Matrix to apply to tomo volume: ", m_trans_tomo)

    return m_trans_tomo


def applyTransformToVolumes(outRoot, fluovol, tomofn, flPix, xrPix,
                            m_trans_tomo):
    """
    Write out both volumes in each other's frame.

    Produces the tomogram resampled into the fluorescence frame, and the
    fluorescence volume resampled back into the tomogram's, so the result can
    be inspected either way round.

    Parameters
    ----------
    outRoot : str or Path
        Directory to write into.
    fluovol : 3D array_like
        Fluorescence volume.
    tomofn : str or Path
        Reconstructed tomogram MRC file.
    flPix, xrPix : float
        Fluorescence and X-ray pixel sizes.
    m_trans_tomo : TMat3D
        Tomogram to fluorescence transformation, shifts in pixel units.
    """
    import mrcfile

    fl_shape = fluovol.shape
    tomo_vol = mrcfile.open(tomofn).data
    xr_shape = tomo_vol.shape

    tomo_vol = tomo_vol - tomo_vol.min()
    tomo_vol_scaled = tr.rescaleFourier(tomo_vol, flPix / xrPix)
    tomo_vol_scaled = frame.padCropArrayCentered(tomo_vol_scaled, fl_shape,
                                                 mode='constant')[0]

    matrix = m_trans_tomo.matrix[0]
    tomo_vol_corr = tr.transformRS(tomo_vol_scaled, matrix)

    iio.writeImage(tomo_vol_corr, outRoot, "xrvol_fluoAligned", ext=".tif")
    iio.writeImage(fluovol, outRoot, "fluovol", ext=".tif")

    del tomo_vol, tomo_vol_scaled, tomo_vol_corr

    # The inverse is applied in two parts: the shifts first, then the rotation
    # on a temporarily enlarged volume, so a rotated corner is not clipped.
    v_shifts = matrix[:3, 3]
    m_shifts = tf.tr3d.angles2mat([0, 0, 0], -v_shifts)
    matrix[:3, 3] = 0
    matrix_inv = np.linalg.inv(matrix)

    fluovol_ali = tr.transformRS(fluovol, m_shifts.matrix[0])
    tmp_shape = (np.array(xr_shape) * (xrPix / flPix) * np.sqrt(2)).astype(int)
    fluovol_ali_sc = frame.padCropArrayCentered(fluovol_ali, tmp_shape,
                                                mode='constant')[0]
    fluovol_ali = tr.transformRS(fluovol_ali_sc, matrix_inv)

    fluovol_ali_sc = tr.rescaleFourier(fluovol_ali, xrPix / flPix)
    fluovol_ali_sc = frame.padCropArrayCentered(fluovol_ali_sc, xr_shape,
                                                mode='constant')[0]

    iio.writeImage(fluovol_ali_sc, outRoot, "fluovol_xrAligned", ext=".tif")
