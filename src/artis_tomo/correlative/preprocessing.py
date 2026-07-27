#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing for correlative multi-modal microscopy pipelines.

Functions here are modality-aware: they assume knowledge of the physical signal
origin -- soft X-ray transmission, fluorescence emission, a stitched mosaic --
and prepare each modality so that images from different ones can be correlated
against each other. The generic operations they build on live in
``artis_sci.image``.
"""

import numpy as np
import skimage
import skimage.filters.thresholding as sk_thres
import skimage.segmentation as sksegmen

from artis_sci.image import frame
from artis_sci.image import filter as ft
from artis_sci.image import transformation as tr
from artis_sci.image.exposure import norm
from artis_sci.io import imageIO as iio
from artis_sci.math import transforms as tf

from ..tomo.project import projectVolume


def normalizeSXTMosaic(img_arr, thr=0.2):
    """
    Preprocess a soft X-ray transmission mosaic image to absorption contrast.

    Applies threshold-based flood-fill segmentation to isolate the cell
    region, converts from transmission to linear absorption coefficient
    (via -log), and normalizes the result so that background pixels are
    set to the mean absorption of the cell region.

    Parameters
    ----------
    img_arr : 2D array_like
        Raw X-ray transmission mosaic image (pixel values proportional to
        transmitted intensity).
    thr : float, optional
        Fraction of the minimum threshold used to define the background for
        the flood-fill segmentation.  Default: 0.2.

    Returns
    -------
    ndarray
        Normalized absorption image with background set to the mean cell value.
    """
    thrmin = thr * sk_thres.threshold_minimum(img_arr)
    img_arr[0, :]  = 0
    img_arr[-1, :] = 0
    img_arr[:, 0]  = 0
    img_arr[:, -1] = 0
    mask = sksegmen.flood(img_arr, (0, 0), tolerance=thrmin)
    mask = skimage.morphology.dilation(mask, skimage.morphology.square(40))
    mask = np.logical_not(mask)
    boundLabels = skimage.measure.label(mask)
    mask = boundLabels == np.argmax(np.bincount(boundLabels.flat)[1:]) + 1

    img_arr[img_arr < 1] = 1
    imlog = norm(-1. * np.log(img_arr))
    mean  = imlog[mask].mean()
    mosaic = imlog * mask + mean * np.logical_not(mask)
    return mosaic


def readFluoVolume(fname, channel=0, flip=True, zoom=2):
    """
    Read a fluorescence volume and put it on an isotropic-ish sampling.

    Handles the two formats these come in, DeltaVision ``.dv`` (multi-channel)
    and TIFF. The volume is padded along Z and then stretched by *zoom*, since
    fluorescence stacks are acquired with a coarser Z spacing than XY. The Y
    flip accounts for the fluorescence camera's handedness relative to the
    X-ray data.

    Parameters
    ----------
    fname : str or Path
        Volume file.
    channel : int, optional
        Channel to read from a multi-channel file.
    flip : bool, optional
        Flip along Y. Default True.
    zoom : float, optional
        Z stretch factor. Default 2.

    Returns
    -------
    3D ndarray
    """
    from pathlib import Path

    ext = Path(fname).suffix
    if ext == '.dv':
        file = iio.imopen(fname, extension=".dv")
        fl_vol = file.read(C=channel)
        file.close()
    elif ext in ['.tif', '.tiff']:
        fl_vol = iio.readImage3D(fname)
    else:
        raise ValueError(f"{fname}: expected a .dv or .tif fluorescence "
                         "volume.")

    shape = np.array(fl_vol.shape) * np.array([zoom, 1, 1])
    fl_vol = frame.padArrayCentered(fl_vol, shape)[0]

    mat = np.array([[1, 0,          0, 0],
                    [0, -1 if flip else 1, 0, 0],
                    [0, 0,       zoom, 0],
                    [0, 0,          0, 1]])

    m = tf.TMat3D(mat)
    fl_vol = tr.transformRS(fl_vol, np.squeeze(m.matrix))

    return fl_vol


def readFluoImage(fname, channel=0, flip=True, zoom=2):
    """
    Read fluorescence data and return a single 2D image.

    A 3D file is projected along the beam direction after the same Z stretch
    and flip as :func:`readFluoVolume`; a 2D file is only transformed. This is
    what the 2D correlative pipeline consumes.

    Parameters
    ----------
    fname : str or Path
        Fluorescence file, 2D or 3D.
    channel : int, optional
        Channel to read from a multi-channel file.
    flip : bool, optional
        Flip along Y. Default True.
    zoom : float, optional
        Z stretch factor. Default 2.

    Returns
    -------
    2D ndarray
    """
    mat = np.array([[1, 0,          0, 0],
                    [0, -1 if flip else 1, 0, 0],
                    [0, 0,       zoom, 0],
                    [0, 0,          0, 1]])
    m = tf.TMat3D(mat)

    file = iio.imopen(fname)
    img_arr = file.read()

    if img_arr.ndim >= 3:
        img_arr = file.read(C=channel)
        shape = np.array(img_arr.shape) * np.array([zoom, 1, 1])
        img_arr = frame.padArrayCentered(img_arr, shape)[0]
        fl_img = projectVolume(img_arr, m, label='Proj FLUO')[0]
    else:
        fl_img = tr.transformRS2D(img_arr, m)

    file.close()

    return fl_img


def preprocessFluoImage(img, shape, pix=None, sigma=None):
    """
    Fit a fluorescence image to the shape it will be correlated against.

    Parameters
    ----------
    img : 2D array_like
        Fluorescence image, typically a projection of the volume.
    shape : tuple
        Target shape, that of the X-ray image being correlated against.
    pix : float, optional
        Pixel size, required when *sigma* is given.
    sigma : float or None, optional
        Gaussian smoothing applied before fitting. ``None`` (default) skips it.
        The 2D pipeline smooths with sigma=5; the 3D pipeline does not, since
        the feature-size filter it applies later already suppresses the noise
        this was for.

    Returns
    -------
    2D ndarray
        Image padded or cropped to *shape*.
    """
    if sigma is not None:
        img = ft.gaussianFilter(img, pix, sigma)

    return frame.padCropArrayCentered(img, shape, rcpad=24)[0]


def preprocessTomoProjection(img, new_pix, tomoPix, mosaic_size):
    """
    Prepare an X-ray tomogram projection for correlation.

    Removes the mean, tapers the border so the rescaling does not ring, then
    resamples to *new_pix* and fits the result to *mosaic_size*.

    Parameters
    ----------
    img : 2D array_like
        Projection through the X-ray tomogram.
    new_pix : float
        Pixel size to resample to, that of the data being correlated against.
    tomoPix : float
        Pixel size of *img*.
    mosaic_size : tuple
        Target shape.

    Returns
    -------
    2D ndarray
    """
    img = img.copy()
    img = img - img.mean()

    shape = img.shape
    mask = ft.maskRaisedCosineBorder2D(shape, 16*2)
    img = img * mask
    tomo_img_scaled = tr.rescaleFourier(img, new_pix / tomoPix)
    tomo_img_scaled = frame.padCropArrayCentered(tomo_img_scaled,
                                                 mosaic_size, rcpad=24)[0]

    return tomo_img_scaled


def preprocessMosaic(img, new_pix, old_pix):
    """
    Prepare a mosaic for correlation, flattening its brightest outliers.

    Tapers the border, resamples to *new_pix*, flattens the background, then
    replaces everything above the 99.5th percentile with the image mean. Very
    bright features -- lipid droplets, ice contamination -- would otherwise
    dominate the cross-correlation regardless of whether they are the features
    being matched.

    Parameters
    ----------
    img : 2D array_like
        Mosaic image, already converted to absorption contrast.
    new_pix : float
        Pixel size to resample to.
    old_pix : float
        Pixel size of *img*.

    Returns
    -------
    2D ndarray
    """
    shape = img.shape
    mask = ft.maskRaisedCosineBorder2D(shape, 20)
    img = img * mask
    img = tr.rescaleFourier(img, new_pix / old_pix)
    img = ft.normalizeBg(img)
    mean = img.mean()
    quan = np.quantile(img, 0.995)
    print("QUANTILE: ", quan)
    mask = img >= quan
    mask = skimage.morphology.dilation(mask, skimage.morphology.disk(5))
    img[mask] = mean

    return img
