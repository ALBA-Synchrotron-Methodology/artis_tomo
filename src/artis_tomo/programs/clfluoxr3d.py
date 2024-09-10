#!/usr/bin/env python3_artis_tomo
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 2023

@authors: Josue Gomez  & Joaquin Oton
"""
import copy
import time
import mrcfile
import numpy as np
from pathlib import Path
from artis_tomo.utils.parser import argparse
from artis_tomo.math import framework as fw
from artis_tomo.io import imageIO as iio
from artis_tomo.image import (process, frame, alignment, filter,
                            transformation as tr)
from artis_tomo.math import transforms as tf
from artis_tomo.programs.tomo_import_imod import getTomoClass, getTomoFromFiles
from artis_tomo.tomo import project as pr

parser = argparse.ArgumentParser(description='correlation between fluorecense '
                                             '3DSIM and X-ray tomograms '
                                             'volumes.')
required = parser.add_argument_group('Required Arguments')
required.add_argument('-i', '--input', required=True,
                      help='Input could be either IMOD directory, X-ray tilt '
                           'series (aligned or not) or a reconstructed '
                           'tomogram volume. If either X-ray tilt series or '
                           'tomogram volume is provided, the angles file with'
                           ' --tlt should be provided.')
required.add_argument('-m', '--mosaic', required=True,
                      help='X-ray mosaic image.')
required.add_argument('-f', '--fluovol', required=True,
                      help='cryo fluorescence volume.')
required.add_argument('--xrpix', required=True, type=float,
                      help='X-ray input images/volume pixel size')
required.add_argument('--mpix', required=True, type=float,
                      help='X-ray mosaic pixel size')
required.add_argument('--flpix', required=True, type=float,
                      help='XY fluorescence volume pixel size')
required.add_argument('--flzpix', required=True, type=float,
                      help='Z fluorescence volume pixel size')
required.add_argument('-o', '--oroot', required=True,
                      help='Rootname used for output files.')

xrayoptions = parser.add_argument_group('X-ray')
xrayoptions.add_argument('--tlt', metavar='FILE',
                         help='IMOD style tilt angles file')
xrayoptions.add_argument('--flipZ', action='store_true',
                         help='Flip Z axis for X-ray tomogram volume '
                           'reconstruction to match fluorescence ')
xrayoptions.add_argument('--tol', type=float, default=0.5,
                         help='value to mask automatically non-interest '
                              'regions of X-ray image.')
xrayoptions.add_argument('--projs', type=int, default=9,
                         help='Set how number of projections used to 3D align'
                           'fluorescence volume with the reconstructed '
                           'X-ray tomogram volume')
xrayoptions.add_argument('--tomofn', metavar='FILE',
                         help='Reconstructed tomogram volume filename to create'
                           'aligned output. If not provided, input should '
                           'be a volume.')

fluoptions = parser.add_argument_group('Fluorescence')
fluoptions.add_argument('--channel', type=int, default=0,
                      help='Channel for cryosim volume.')
fluoptions.add_argument('--flipFluo', default=False,
                     action='store_true',
                     help='Flip Y axis to correct fluorescence volume '
                          'handiness.')

options = parser.add_argument_group('Correlation options')
options.add_argument('--range', nargs=2, metavar=('MIN', 'MAX'),
                     type=float, default=[400, 1000],
                     help='Range (in nm) to define filtered features to align.')
options.add_argument('--global_step', type=float, default=1.0,
                     help='global searching step to correlate images')
options.add_argument('--perturb',nargs=2, metavar=('STEP', 'THETA'),
                     type=float, default=[1, 3],
                      help='Maximum polar theta angle and angle step for 3D '
                           'angular search.')

resources = parser.add_argument_group('Resources')
resources.add_argument('--gpu', type=int, const=0, nargs='?',
                     help="Use GPU acceleration. Select gpu device number \
                     (default: 0) ")
resources.add_argument('--nproc', type=int, default=4,
                         help='Number of angles processed in  parallel. If '
                         'negative, it uses all available cores')

def _readImage(fname, *kwargs):
    file = iio.imopen(fname, *kwargs)
    img = file.read()
    file.close()
    return img

def _saveImages(img, oroot, filename, ext='.tif', ntype=np.float32):
    base_name = filename + ext
    img_fn = Path(oroot, base_name)
    img_fn.with_suffix('')
    iio.imwrite(img_fn, img.astype(ntype))

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

def _getTomoFromFiles(fn, tltfn, xrpix, flipz):
    img_arr = _readImage(fn)
    tomo = getTomoFromFiles(img_arr, fn, tltfn, xrpix, flipz)
    return tomo

def _get_input_files(input, tltfn, xrpix, flipz):
    file_name = Path(input)
    if file_name.is_file():
        fn = file_name.parent
        tomo = _getTomoFromFiles(file_name, tltfn, xrpix, flipz)
    else:
        fn = file_name
        tomo = getTomoClass(fn, xrpix, flipz)
    return tomo, fn


def _flip_images(img):
    mat_flip = np.array([[1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    m_flip = tf.TMat3D(mat_flip)
    img_flip = tr.transformRS(img, np.matrix(np.squeeze(m_flip.matrix)))
    return img_flip


def _getvolfluo(fname, channel=0, flip=True, zoom=2):
    file = iio.imopen(fname, extension=".dv")
    fl_vol =  file.read(C=channel)
    file.close()
    shape = np.array(fl_vol.shape) * np.array([zoom,1,1])
    fl_vol = frame.padArrayCentered(fl_vol, shape)[0]
    
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
    
    m_flip = tf.TMat3D(mat)
    fl_vol = tr.transformRS(fl_vol, np.matrix(np.squeeze(m_flip.matrix)))
    return fl_vol

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

def _preproces_img_fluo(fluovol, flPix, xrSizeSc, useGPU=None):
    img_fl = _getimgfluo(fluovol, useGPU=useGPU)[0]
    #Note: img_fl must be 2d-array, otherwise gaussianFilter will get crazy :D
    img_fil_fl = filter.gaussianFilter(img_fl, flPix, 5)
    img_fluo = frame.padCropArrayCentered(img_fil_fl, xrSizeSc, rcpad=24)[0]
    return img_fluo

def _preprocesTomoImg(img, new_pix, tomoPix, mosaic_size):
    img = img.copy()
    img[img <= 0.] = 1
    imlog = -1. * np.log(img)
    mean = imlog.mean()
    
    img = imlog - mean
    shape = img.shape
    mask = filter.maskRaisedCosineBorder2D(shape, 16)
    img = img * mask
    tomo_img_scaled = tr.rescaleFourier(img, new_pix / tomoPix)
    tomo_img_scaled = frame.padCropArrayCentered(tomo_img_scaled,
                                                 mosaic_size, rcpad=24)[0]
    # tomo_img_scaled = filter.gaussianFilter(tomo_img_scaled, new_pix, 5)
    return tomo_img_scaled

def _filterFeaturesStep(img, pix, dim_range):
    rmin, rmax = dim_range
    px_min = round(rmin / pix)
    px_max = round(rmax / pix)
    print("TOPHAT ", px_max, px_min)
    img_fil = process.filterFeatures(img, px_max, px_min)
    return img_fil

def _preprocesMosImg(img, new_pix, old_pix):
    import skimage
    shape = img.shape
    mask = filter.maskRaisedCosineBorder2D(shape, 20)
    img = img * mask
    img = tr.rescaleFourier(img, new_pix / old_pix)
    img = filter.normalizeBg(img)
    mean = img.mean()
    quan = np.quantile(img, 0.995)
    print("QUANTILE: ", quan)
    mask = img >= quan
    mask = skimage.morphology.dilation(mask, skimage.morphology.disk(5))
    img[mask] = mean
    return img

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

def sphereSurfaceMatrixAngleStep(perturb):
    """
    Calculate matrix transformation around a spherical surface
    uniformily distributed

    Parameters
    ----------
    angStep    : angular separation among points
    thetaMax   : maximum value for theta angle. Default 5º

    Returns
    -------
    transform matrices     : Transformation matrices for an unit vector over
                             a spherical surface
    """
    angStep, thetaMax = perturb
    thetaMax = thetaMax * np.pi / 180
    nPoints = round(4 * (180 ** 2) / (np.pi * angStep ** 2))
    indices = np.arange(0, nPoints, dtype=float)  # + 0.5
    thetaTot = np.arccos(1 - 2 * indices / (nPoints - 1))
    theta = thetaTot[thetaTot <= thetaMax]
    iMax = len(theta)
    
    # Phi step is the gold ratio to avoid points to match in phi
    phi = np.pi * (1 + 5 ** 0.5) * indices[:iMax]
    phi = phi % (2 * np.pi)
    mX = [tf.tr3d.rotXmat(ang) for ang in theta]
    mZ = [tf.tr3d.rotZmat(ang) for ang in phi]
    l = len(mX)
    m = tf.tr3d.empty(l)
    for x, z, c in zip(mX, mZ, range(l)):
        mult = z.matrix @ x.matrix @ z.inv.matrix
        ml = tf.TMat3D(mult)
        m.set(ml, c)
    return m

def multElemByElem(mat1, mat2):
    count = 0
    size = len(mat1)
    tmat_stack = tf.tr3d.empty(size)
    for m1, m2 in zip(mat1, mat2):
        mat = m1 * m2
        tmat_stack.set(mat, count)
        count += 1
    return tmat_stack

def multNested(mat1, mat2):
    size = len(mat1) * len(mat2)
    tmat_stack = tf.tr3d.empty(size)
    count = 0
    for m1 in mat1:
        for m_sph_s in mat2:
            mat = m1 * m_sph_s
            tmat_stack.set(mat, count)
            count += 1
    return tmat_stack

def _apply_tophat_imgcc(imgcc):
    imgcc_th_list = [process.filterTophat(im, 10) for im in imgcc]
    return imgcc_th_list
    

def _get_values(angles, ccimg):
    new_img_list = []
    for img in ccimg:
        shape = img.shape
        new_shape = tuple(int(0.9 * x) for x in shape)
        img2 = frame.padCropArrayCentered(img, new_shape, rcpad=1)[0]
        new_img_list.append(img2)
    
    cc = [x.max() for x in new_img_list]
    pmax = np.argmax(cc)
    imgcc_th = new_img_list[pmax]
    x, y = np.unravel_index(imgcc_th.argmax(), imgcc_th.shape)
    imgmask = np.zeros(imgcc_th.shape)
    imgmask[x - 30:x + 30, y - 30:y + 30] = 1
    imgcc_th = imgcc_th * imgmask
    mpos = np.unravel_index(imgcc_th.argmax(), imgcc_th.shape)
    shifts = mpos - np.array(imgcc_th.shape) // 2
    return angles[pmax], shifts[::-1], new_img_list


def _get_projs(angles, projs):
    if projs % 2 == 0:
        projs += 1
    pos_list = np.linspace(0, len(angles)-1, projs).astype(int).tolist()
    mangles = [abs(x) for x in angles]
    # min_ind =mangles.index(min(mangles))
    # max_ind = [i for i in range(len(mangles)) if mangles[i] == max(mangles)]
    
    central_pos = len(pos_list)//2
    for i in range(central_pos+1):
        val1 = mangles[pos_list[i]]
        val2 = mangles[pos_list[-i-1]]
        if val1 != val2:
            indx1 = [j for j in range(len(mangles)) if mangles[j] == val1]
            indx2 = [j for j in range(len(mangles)) if mangles[j] == val2]
            l1 = len(indx1)
            l2 = len(indx2)
            if l1 == 2 and l2 == 1:
                pos_list[-i - 1] = indx1[1]
            elif l2 == 2:
                pos_list[i] = indx2[0]
            else:
                l = 1
                val = val1 + 1
                while l == 1:
                    indx = [j for j in range(len(mangles)) if mangles[j] == val]
                    l = len(indx)
                    if l == 2:
                        pos_list[i] = indx[0]
                        pos_list[-i - 1] = indx[1]
                    val = val + 1
        
        if i == central_pos and val1 != 0.0:
            pos_list[i] = mangles.index(min(mangles))
    
    sel_angles = [angles[idx] for idx in pos_list]
    print("Values: ", pos_list, sel_angles)
    return pos_list


def align_fl2d_mos_step(fnmosaic, fluovol, tol, useGPU,
                        flPix, mPix, ang_params, dim_range, thr):
    start_imp_imgs = time.time()
    mosaic = _imread2d(fnmosaic).astype(float)
    mosaic = process.processMosaic(mosaic, tol)
    mosaic_scaled = tr.rescaleFourier(mosaic, flPix / mPix)
    mosaic_fil = _filterFeaturesStep(mosaic_scaled, flPix, dim_range)
    xrSizeSc = mosaic_fil.shape
    
    img_fluo = _preproces_img_fluo(fluovol, flPix, xrSizeSc, useGPU)
    finish_imp_imgs = time.time()
    tim_elap_imp = finish_imp_imgs - start_imp_imgs
    print(f"finishing import images. time elapsed: {round(tim_elap_imp, 2)} s")
    start_proc_imgs = time.time()
    img_fluo_fil = _filterFeaturesStep(img_fluo, flPix, dim_range)
    
    # raise Exception
    finish_proc_imgs = time.time()
    tim_elap_proc = finish_proc_imgs - start_proc_imgs
    print("finishing processing images. "
          f"time elapsed: {round(tim_elap_proc, 2)} s")
    ang_fm, sft_fm, ccimglist = _alignImages(mosaic_fil, img_fluo_fil,
                                             ang_params, shr=0.5,  thr=thr)
    sft_ls = [sft_fm[0], sft_fm[1], 0]
    sft_abs = [x * flPix for x in sft_ls]
    ang_fm = [np.deg2rad(ang_fm), 0, 0]
    m_scaled = tf.tr3d.angles2mat(ang_fm, shifts=sft_ls)
    fluofix = tr.transformRS2D(img_fluo, m_scaled)
    m_fm_mo = tf.tr3d.angles2mat(ang_fm, sft_abs)
    images = [mosaic_scaled, mosaic, mosaic_fil, img_fluo_fil, fluofix, img_fluo]
    
    return m_fm_mo, images


def align_st_mos_step(fnmosaic, tomo, noref_params,
                      xrpix, mPix, flPix, tol, thr, fnOutRoot):
        
    imgXr = _imread2d(fnmosaic).astype(float)
    mosaic = process.processMosaic(imgXr, tol)
    mosaic_scaled = _preprocesMosImg(mosaic, flPix, mPix)
    
    tomoCentral = tomo.getImage('center')
    tomo_img_scaled = _preprocesTomoImg(tomoCentral, flPix, xrpix,
                                        mosaic_scaled.shape)
    ang_th, sft_th, ccimglist = _alignImages(mosaic_scaled, tomo_img_scaled,
                                  noref_params, ccTH=True, shr=0.8,
                                  thr=thr)
    _saveImages(ccimglist[0], fnOutRoot, "cc_central")
    # ang_th, sft_th, _ = _get_values(ang_th, ccimglist)
    ang_th = [np.deg2rad(ang_th), 0, 0]
    sft_th = list(map(lambda x: x, sft_th))
    sft_th.append(0)
    # m_st_mo_pix = tf.tr3d.angles2mat(ang_th, sft_th)
    sft_real = [x * flPix for x in sft_th]
    # print("sft_real", sft_real, sft_th)
    m_st_mo_real = tf.tr3d.angles2mat(ang_th, sft_real)
    
    tomo.setAlignTMat(m_st_mo_real)
    return tomo, mosaic_scaled


def align_st_fv_step(fnOutRoot, fluovol, tomo, xrPix, flPix,
                     perturb, projs, ref_params ,m_fm_mo, useGPU, thr, debug):
    # stack to mosaic
    m_st_mo = tomo.getAlignTmat()
    m_st_mo.matrix[:, 0:3, 3] = m_st_mo.matrix[:, 0:3, 3] / flPix
    m_fm_mo.matrix[:, 0:3, 3] = m_fm_mo.matrix[:, 0:3, 3] / flPix
    
    #stack to fluo
    m_st_fm = m_fm_mo.inv * m_st_mo
    # print("M m_fm_mo", m_fm_mo)
    # print("M m_st_mo", m_st_mo)
    # print("M m_st_fm", m_st_fm)
    # # for 0º, find best inplane angle and apply to others (Kino's idea)
    img_fl = _getimgfluo(fluovol, useGPU=useGPU)[0]
    img_fluo = filter.gaussianFilter(img_fl, flPix, 5)

    tomoCentral = tomo.getImage('center')
    tomo_img_scaled = _preprocesTomoImg(tomoCentral, flPix, xrPix,
                                        img_fluo.shape)
    st_fix = tr.transformRS2D(tomo_img_scaled, m_st_fm)
    if debug:
        _saveImages(st_fix, fnOutRoot, "central_st_fitted")
    
    ang_th, sft_th, cclist = _alignImages(img_fluo, st_fix, ref_params,
                                  shr=0.05, ccTH=True, thr=thr)

    sft_ls = [sft_th[0], sft_th[1], 0]
    ang_fm = [np.deg2rad(ang_th), 0, 0]
    m_corrected = tf.tr3d.angles2mat(ang_fm, sft_ls)
    # print("M Corrected: ", m_corrected)
    m_st_fm = m_corrected * m_st_fm
    # m_st_fm_sr_fm_ang = tf.tr3d.removeShifts(m_st_fm_sr_fm)dale

    # #Visualization
    st_fix = tr.rescaleFourier(tomoCentral, flPix / xrPix)
    st_central = frame.padCropArrayCentered(st_fix, img_fluo.shape,
                                            end_values=tomo_img_scaled.min(),
                                            rcpad=2)[0]
    st_trans = tr.transformRS2D(st_central, m_st_fm)
    # print("MATRIX: ", m_st_fm)

    cc = [x.max() for x in cclist]
    pmax = np.argmax(cc)
    
    if debug:
        _saveImages(cclist[pmax], fnOutRoot, "cc_image")
        _saveImages(img_fluo, fnOutRoot, "fluo_original")
        _saveImages(st_trans, fnOutRoot, "central_st_correlated")
    
    # ------------------3D-------------------------------
    m_pertub = sphereSurfaceMatrixAngleStep(perturb)
    proj_list = _get_projs(tomo.getTiltAngles(), projs)
    zoom_tf = xrPix / flPix
    m_rec = tomo.getRecTMat()[proj_list]
    m_rec.matrix[:, 0:3, 3] = m_rec.matrix[:, 0:3, 3] * zoom_tf
    m_rec_ang = tf.tr3d.removeShifts(m_rec)

    tomo_ts = tomo.getTiltseries()[proj_list]

    count = 0
    ccmax_ls = []
    shift_ls = []
    angle_ls = []
    for m in m_pertub:
        st_list = []
        fm_list = []
        cc_list = []
        count += 1
        m_rec_perturb = m_rec * m * m_st_fm.inv
        start_proj_fluo = time.time()
        fluo_projstk = _getimgfluo(fluovol, m_rec_perturb, useGPU=useGPU)
        finish_proj_fluo = time.time()
        for img_fluo, img_ts in zip(fluo_projstk, tomo_ts):
            img_ts = _preprocesTomoImg(img_ts, flPix, xrPix, img_fluo.shape)
            img_fluo = filter.gaussianFilter(img_fluo, flPix, 5)
            start_align = time.time()
            
            angs = (0, 0, 1)
            ang, sft, cc_img_list_def = _alignImages(img_fluo, img_ts, angs,
                                                     ccTH=True, thr=thr)
            sft_st = [sft[0], sft[1], 0]
            ang_st = [np.deg2rad(ang), 0, 0]
            mat_corr = tf.tr3d.angles2mat(ang_st, sft_st)
            # print("Mat_corr: ", mat_corr)
            finish_align = time.time()
            start_trans2d = time.time()
            img_ts1 = tr.transformRS2D(img_ts, mat_corr)
            finish_trans2d = time.time()
            st_list.append(img_ts1)
            fm_list.append(img_fluo)
            cc_list.append(cc_img_list_def[0])
        
        st_stack = np.stack(st_list)
        fm_stack = np.stack(fm_list)
        cc_stack = np.stack(cc_list)
        cc_shape = cc_stack.shape
        vol_shape = tuple((cc_shape[1],int(0.2 * cc_shape[2]), int(0.2 * cc_shape[1])))
        # print("SHAPE CC vol: ", vol_shape)
        rec_cc = pr.backProjectRS(cc_stack, m_rec_ang, vol_shape)
        mpos = np.unravel_index(rec_cc.argmax(), rec_cc.shape)
        shifts = mpos - np.array(rec_cc.shape) // 2
        sft_th = shifts[::-1]
        time_proj = finish_proj_fluo - start_proj_fluo
        time_align = finish_align - start_align
        time_transform = finish_trans2d - start_trans2d

        print("Shifts: ", shifts, sft_th)
        print("Proj FLUO. "
              f"time elapsed: {round(time_proj, 2)} s")
        print("ALIGNING. "
              f"time elapsed: {round(time_align, 2)} s")
        print("TRANSFORM 2D. "
              f"time elapsed: {round(time_transform, 2)} s")
        ang, _ = m.mat2params(axes=tf.SZYZ)
        
        ccmax_ls.append(rec_cc.argmax())
        shift_ls.append(sft_th)
        angle_ls.append( np.rad2deg(ang))
        if debug:
            _saveImages(rec_cc, fnOutRoot, "vol_CC_%s" % count, ext=".mrc")
            _saveImages(cc_stack, fnOutRoot, "stack_CC_%s" %count, ext=".mrc")
            _saveImages(fm_stack, fnOutRoot, "stack_FM_%s" %count, ext=".mrc")
            _saveImages(st_stack, fnOutRoot, "stack_TS_%s" %count, ext=".mrc")
    posmax = ccmax_ls.index(max(ccmax_ls))
    print("posmax: ", posmax)
    print(m_pertub[posmax])
    print("angles and shifts: ", angle_ls[posmax], shift_ls[posmax])
    shifts_max = shift_ls[posmax]
    m_shift_corrected = tf.tr3d.shifts2mat(shifts_max)
    m_trans_tomo = m_st_fm  * m_pertub[posmax].inv * m_shift_corrected
    print("Matrix to apply to tomo volume: ", m_trans_tomo)
    return m_trans_tomo


def apply_matrix_tomo(fnOutRoot, fluovol, tomofn, flPix, xrPix, m_trans_tomo):
    fl_shape = fluovol.shape
    tomo_vol = mrcfile.open(tomofn).data
    tomo_vol_scaled = tr.rescaleFourier(tomo_vol, flPix / xrPix)
    tomo_vol_scaled = frame.padCropArrayCentered(tomo_vol_scaled,
                                                 fl_shape, rcpad=1)[0]
    m_trans_tomo_np_matrix = np.matrix(np.squeeze(m_trans_tomo.matrix))
    tomo_vol_corr = tr.transformRS(tomo_vol_scaled, m_trans_tomo_np_matrix)

    _saveImages(tomo_vol_corr, fnOutRoot, "tomo_corr", ext=".mrc")
    _saveImages(fluovol, fnOutRoot, "fluo_vol", ext=".mrc")
    #------------------------------------------------------------
    # return "Change when function is ready"


def clfluoxr3dProgram():
    args = parser.parse_args()
    input = args.input
    fnmosaic = args.mosaic
    fluofn = args.fluovol
    xrPix = args.xrpix
    flPix = args.flpix
    flzPix = args.flzpix
    mPix = args.mpix
    fnOutRoot = args.oroot
    
    channel = args.channel
    zoom = flzPix / flPix
    
    flipFluo = args.flipFluo
    flipz = args.flipZ
    
    tltfn = args.tlt
    tomofn = args.tomofn
    tol = args.tol
    dim_range = args.range
    projs = args.projs
    perturb = args.perturb
    global_step = args.global_step
    thr = args.nproc
    useGPU = args.gpu
    global_params = (0, 359, global_step)
    refine_params = (-5, 5, 0.5)
    noref_params = (0, 0, 1)
    debug = False
    fluovol = _getvolfluo(fluofn, channel, flipFluo, zoom)
    # -------------------------------------------------------------------------
    # Step 1: Align CryoSIM proj to SXT mosaic.
    m_fm_mo_sr_mo, images = align_fl2d_mos_step(fnmosaic, fluovol, tol,
                                                useGPU, flPix, mPix,
                                                global_params, dim_range, thr)
    if debug:
        _saveImages(images[0], fnOutRoot,"mosaic_scaled")
        _saveImages(images[1], fnOutRoot, "mosaic_processed")
        _saveImages(images[2], fnOutRoot, "mosaic_tophat_toAlign")
        _saveImages(images[3], fnOutRoot, "fluo_tophat_toAlign")
        _saveImages(images[4], fnOutRoot, "fluo_correlated")
        _saveImages(images[5], fnOutRoot, "fluo_original")
    
    # -------------------------------------------------------------------------
    # Step 2: align tomogram proj at 0º to SXT mosaic.
    tomo, input_dir = _get_input_files(input, tltfn, xrPix, flipz)
    
    tomo, mosaic_scaled = align_st_mos_step(fnmosaic, tomo, noref_params, xrPix,
                                            mPix, flPix, tol, thr, fnOutRoot)
    if debug:
        tomoCentral = tomo.getImage('center')
        tomo_img_scaled = _preprocesTomoImg(tomoCentral, flPix,
                                            xrPix, mosaic_scaled.shape)
        m_st_mo_real = tomo.getAlignTmat()
        m_st_mo_pix = copy.deepcopy(m_st_mo_real)
        m_st_mo_pix.matrix[:, 0:3, 3] = m_st_mo_pix.matrix[:, 0:3, 3] / flPix
        st_fix = tr.transformRS2D(tomo_img_scaled, m_st_mo_pix)
        print("MATRICES :", m_st_mo_pix, m_st_mo_real)
        _saveImages(mosaic_scaled, fnOutRoot, "mosaic")
        _saveImages(tomo_img_scaled, fnOutRoot, "st_tomo01_orig")
        _saveImages(st_fix, fnOutRoot, "st_tomo01_fix")
    
    # -------------------------------------------------------------------------
    # Step 3: align tomogram to Fluorescense 3D image.
    m_trans_tomo = align_st_fv_step(fnOutRoot, fluovol, tomo, xrPix,
                                    flPix, perturb, projs, refine_params,
                                    m_fm_mo_sr_mo, useGPU, thr, debug)
    
    apply_matrix_tomo(fnOutRoot, fluovol, tomofn, flPix, xrPix, m_trans_tomo)
    
    # fluo_vol = _getimgfluo(fluovol, channel)[0]
    # tomo_vol = mrcfile.open(tomofn).data
    # tomo_vol_scaled = tr.rescaleFourier(tomo_vol, flPix / xrPix)
    # tomo_vol_scaled = frame.padCropArrayCentered(tomo_vol_scaled,
    #                                              fluo_vol.shape, rcpad=24)[0]
    # tomo_vol_corr = pr.
    print("Matrix to apply: ", m_trans_tomo)
    