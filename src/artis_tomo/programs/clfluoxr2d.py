#!/usr/bin/env python3_artis_tomo
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 2023

@authors: Josue Gomez  & Joaquin Oton
"""
import time
import numpy as np
from pathlib import Path
from artis_tomo.io import imageIO as iio
from artis_tomo.math import transforms as tf
from artis_tomo.math import framework as fw
from artis_tomo.tomo import project as pr
from artis_tomo.utils.parser import argparse
from artis_tomo.image import (process, frame, alignment, filter,
                            transformation as tr)


parser = argparse.ArgumentParser(description='align fluorecense and'
                                             ' x-ray mosaic images.')

required = parser.add_argument_group('Required Arguments')

required.add_argument('-i', '--input', required=True,
                      help='X-ray mosaic image.')
required.add_argument('-f', '--fluoimg', required=True,
                      help='Either cryo fluorescence volume or image.')
required.add_argument('--mpix', required=True, type=float,
                      help='X-ray mosaic pixel size')
required.add_argument('--flpix', required=True, type=float,
                      help='XY fluorescence volume pixel size')
required.add_argument('-o', '--oroot', required=True,
                      help='Rootname used for output files.')

xrayoptions = parser.add_argument_group('X-ray')
xrayoptions.add_argument('--tol', type=float, default=0.5,
                         help='value to mask automatically non-interest '
                              'regions of X-ray image.')

fluoptions = parser.add_argument_group('Fluorescence')
fluoptions.add_argument('--flipFluo', default=False,
                     action='store_true',
                     help='Flip Y axis to correct fluorescence volume '
                          'handiness.')
fluoptions.add_argument('--channel', type=int, default=0,
                      help='Channel for cryosim volume if is an input.')
fluoptions.add_argument('--flzpix', type=float, default=1.0 ,
                        help='Z fluorescence volume pixel size if fluorescence '
                             'volume if is an input.')

options = parser.add_argument_group('Correlation options')
options.add_argument('--range', nargs=2, metavar=('MIN', 'MAX'),
                     type=float, default=[400, 1000],
                     help='Range (in nm) to define filtered features to align.')
options.add_argument('--global_step', type=float, default=1.0,
                     help='global searching step to correlate images')

resources = parser.add_argument_group('Resources')
resources.add_argument('--gpu', type=int, const=0, nargs='?',
                     help="Use GPU acceleration. Select gpu device number "
                          "(default: 0) ")
resources.add_argument('--nproc', type=int, default=4,
                         help='Number of angles processed in  parallel. If '
                         'negative, it uses all available cores')

def _saveImages(img, oroot, filename, ext='.tif', ntype=np.float32):
    base_name = filename + ext
    img_fn = Path(oroot, base_name)
    img_fn.with_suffix('')
    iio.imwrite(img_fn, img.astype(ntype))

def _readImage(fname, *kwargs):
    file = iio.imopen(fname, *kwargs)
    img = file.read()
    file.close()
    return img

def _imread2d(fname):
    img_arr = _readImage(fname)
    if img_arr.ndim == 3:
        img_arr = img_arr[:, :, 0]
    elif img_arr.ndim > 3:
        raise ("Error: You must enter only 2D or 3D image files")
    return img_arr

def _imread3d(fname):
    img_arr = _readImage(fname)
    if img_arr.ndim == 3:
        img_arr = img_arr[:, :, :]
    else:
        raise ("Error: You must enter 3D images only")
    return img_arr

def _getfluo(fname, channel=0, flip=True, zoom=2):
    if flip:
        mat = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, zoom, 0],
                        [0, 0, 0, 1]])
    else:
        mat = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, zoom, 0],
                        [0, 0, 0, 1]])
    m = tf.TMat3D(mat)
    
    file = iio.imopen(fname)
    img_arr = file.read()
    if img_arr.ndim >= 3:
        img_arr = file.read(C=channel)
        shape = np.array(img_arr.shape) * np.array([zoom, 1, 1])
        img_arr = frame.padArrayCentered(img_arr, shape)[0]
        fl_img = _getimgfluo(img_arr, m)[0]
        
    else:
        fl_img = tr.transformRS2D(img_arr, m)
    file.close()
    return fl_img

def _getimgfluo(fl_vol, m=None, useGPU=None):
    xp = fw.frame()
    if useGPU is not None:
        xp.set_device(f'cuda:{useGPU}', 'cupy')
        fl_vol = xp.to_device(fl_vol)
    if m is None:
        m = tf.tr3d.empty(1)
    start_proj_fluo = time.time()
    fl_img = pr.projectRS(fl_vol, m)
    finish_proj_fluo = time.time()
    fl_img = xp.to_device(fl_img, 'cpu')
    time_proj = finish_proj_fluo - start_proj_fluo
    print("Proj FLUO. "
          f"time elapsed: {round(time_proj, 2)} s")
    return fl_img

def _apply_tophat_imgcc(imgcc):
    imgcc_th_list = [process.filterTophat(im, 10) for im in imgcc]
    return imgcc_th_list

def _alignImages(img_template, img_correlate, ang,
                 ccTH=False, shr=0.8, thr=1):
    anglist, ccimglist = alignment.getInplaneAlignBF(img_correlate,
                                                     img_template,
                                                     ang,
                                                     thr)
    if ccTH:
        ccimglist = _apply_tophat_imgcc(ccimglist)
    angle, shift = alignment.getAlignValues(anglist, ccimglist, shr)
    return angle, shift, ccimglist

def _filterFeaturesStep(img, pix, dim_range):
    rmin, rmax = dim_range
    px_min = round(rmin / pix)
    px_max = round(rmax / pix)
    print("TOPHAT ", px_max, px_min)
    img_fil = process.filterFeatures(img, px_max, px_min)
    return img_fil

def _preproces_img_fluo(img_fluo, flPix, xrSizeSc, useGPU=None):
    #Note: img_fl must be 2d-array, otherwise gaussianFilter will get crazy :D
    img_fil_fl = filter.gaussianFilter(img_fluo, flPix, 5)
    img_fluo = frame.padCropArrayCentered(img_fil_fl, xrSizeSc, rcpad=24)[0]
    return img_fluo

def align_fl2d_mos_step(fnmosaic, fluoimg, tol, useGPU,
                        flPix, mPix, ang_params, dim_range, thr):
    start_imp_imgs = time.time()
    mosaic = _imread2d(fnmosaic).astype(float)
    mosaic = tr.rescaleFourier(mosaic, flPix / mPix)
    mosaic = process.processMosaic(mosaic, tol)
    mosaic_fil = _filterFeaturesStep(mosaic, flPix, dim_range)
    xrSizeSc = mosaic_fil.shape
    
    img_fluo = _preproces_img_fluo(fluoimg, flPix, xrSizeSc, useGPU)
    finish_imp_imgs = time.time()
    tim_elap_imp = finish_imp_imgs - start_imp_imgs
    print(f"finishing import images. time elapsed: {round(tim_elap_imp, 2)} s")
    start_proc_imgs = time.time()
    img_fluo_fil = _filterFeaturesStep(img_fluo, flPix, dim_range)
    
    finish_proc_imgs = time.time()
    tim_elap_proc = finish_proc_imgs - start_proc_imgs
    print("finishing processing images. "
          f"time elapsed: {round(tim_elap_proc, 2)} s")
    ang_fm, sft_fm, ccimglist = _alignImages(mosaic_fil, img_fluo_fil,
                                             ang_params, shr=0.5, thr=thr)
    sft_ls = [sft_fm[0], sft_fm[1], 0]
    sft_abs = [x * flPix for x in sft_ls]
    ang_fm = [np.deg2rad(ang_fm), 0, 0]
    m_scaled = tf.tr3d.angles2mat(ang_fm, shifts=sft_ls)
    fluofix = tr.transformRS2D(img_fluo, m_scaled)
    m_fm_mo = tf.tr3d.angles2mat(ang_fm, sft_abs)
    images = [mosaic, mosaic_fil, img_fluo_fil, fluofix, img_fluo]
    
    return m_fm_mo, images


def clfluoxr2dProgram():
    args = parser.parse_args()
    input = args.input
    fluofn = args.fluoimg
    flPix = args.flpix
    flzPix = args.flzpix
    mPix = args.mpix
    fnOutRoot = args.oroot
    
    channel = args.channel
    zoom = flzPix / flPix
    
    flipFluo = args.flipFluo
    
    tol = args.tol
    dim_range = args.range
    global_step = args.global_step
    thr = args.nproc
    useGPU = args.gpu
    global_params = (0, 359, global_step)
    debug = False
    fluoimg = _getfluo(fluofn, channel, flipFluo, zoom)
    # -------------------------------------------------------------------------
    # Step 1: Align CryoSIM proj to SXT mosaic.
    m_fm_mo_sr_mo, images = align_fl2d_mos_step(input, fluoimg, tol,
                                                useGPU, flPix, mPix,
                                                global_params, dim_range, thr)

    _saveImages(images[0], fnOutRoot, "mosaic_scaled")
    _saveImages(images[3], fnOutRoot, "fluo_correlated")

    if debug:
        _saveImages(images[1], fnOutRoot, "mosaic_tophat_toAlign")
        _saveImages(images[2], fnOutRoot, "fluo_tophat_toAlign")
        _saveImages(images[4], fnOutRoot, "fluo_original")