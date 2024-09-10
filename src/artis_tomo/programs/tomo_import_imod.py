#!/usr/bin/env python3_artis_tomo
# -*- coding: utf-8 -*-
"""
Created on THu May 04 2023

@authors: Josue Gomez  & Joaquin Oton
"""
import time
import numpy as np
from pathlib import Path

import transforms3d.affines

from artis_tomo.utils.parser import argparse
from artis_tomo.tomo import tomogram as tm
from artis_tomo.math import transforms as tf
from artis_tomo.io import imageIO as io
from artis_tomo.tomo import project as pr


parser = argparse.ArgumentParser(description='import imod alignment '
                                             'parameters')
required = parser.add_argument_group('Arguments')
required.add_argument('-i', '--iroot', required=True,
                      help='Folder with imod files for a single tomogram')
required.add_argument('-p', '--pix', required=True, type=float,
                      help='tilt series images pixel size')

options = parser.add_argument_group('Options')
options.add_argument('-f', '--flipYZ', type=bool, default=True,
                      help='Interchange the Y and Z coordinates')
options.add_argument('-z', '--flipZ', type=bool, default=False,
                      help='are tilt series hand inverted?')

def getLineValue(filename, pattern, splt=1):
    f = open(filename, mode='r')
    lines = f.readlines()
    listPatt = list(filter(lambda x: pattern in x, lines))
    if listPatt != []:
        val = list(filter(lambda x: pattern in x, lines))[0].split()[splt]
    else:
        val = None
    f.close()
    return val

def getIniMat(size):
    return tf.tr3d.empty(size)

def importImodProgram():
    args = parser.parse_args()
    fn = args.iroot
    pxzise = args.pix
    flipz = args.flipZ
    tomo = getTomoClass(fn, pxzise, flipz)
    tomoDict = tomo.exportDict()
    fn2 = tomo.getFilename()[:-4] + '.h5'
    io.savetoh5File(fn2, tomoDict)

def _get_xfmatrix(transfn, size):
    if transfn.exists():
        f = open(transfn, mode='r')
        mlines = f.readlines()
        xf_m_st = getIniMat(size)
        for i, m in enumerate(mlines):
            shifts = np.array(list(map(float, m.split()))[4:])
            shifts = np.append(-shifts, 0)
            shifts_mat = tf.tr3d.shifts2mat(shifts)
            ang_list = list(map(float, m.split()))[:4]
            m = tf.tr3d.empty()
            m.matrix[0, :2, :2] = np.array(ang_list).reshape(2, 2)
            m = m.inv
            m *= shifts_mat
            xf_m_st.set(m, i)
        f.close()
    else:
        xf_m_st = getIniMat(size)
    return xf_m_st


def _get_tltmatrix(fntlt):
    if fntlt.exists():
        f2 = open(fntlt, mode='r')
        tilt_list = list(map(float, f2.readlines()))
        size = len(tilt_list)
        tilt_m_st = getIniMat(size)
        for i, ang in enumerate(tilt_list):
            ang = np.pi * ang / 180
            m = tf.tr3d.rotYmat(ang)
            tilt_m_st.set(m, i)
        f2.close()
        return tilt_m_st, tilt_list
    else:
        raise Exception("file %s not found. It must be exists" % fntlt)

def getTomoClass(fn, pxsize, flipz):
    newst = Path(fn, 'newst.com')
    tilt = Path(fn, 'tilt.com')
    if Path.exists(newst) and Path.exists(tilt):
        tomo = tm.Tomogram(pxsize)
        stfn = Path(fn, getLineValue(newst, 'InputFile'))
        tomo.setFilename(str(stfn))
        size = tomo.getSize()
    
        transfn = Path(fn, getLineValue(newst, 'TransformFile'))
        xf_m_st = _get_xfmatrix(transfn, size)
        bin = getLineValue(newst, 'BinByFactor')
        tomo.setBinning(int(bin))
    
        roi = (int(getLineValue(tilt, 'FULLIMAGE')),
               int(getLineValue(tilt, 'FULLIMAGE', 2)), 0)
        tomo.setRoi(roi)
        tomo.setThickness(int(getLineValue(tilt, 'THICKNESS')))
        shiftx = getLineValue(tilt, 'SHIFT')
        shiftz = getLineValue(tilt, 'SHIFT', 2)
        shifts = (int(shiftx) if shiftx is not None else 0, 0,
                  int(shiftz) if shiftz is not None else 0)
        tomo.setShifts(shifts)
    
        tiltfile = getLineValue(tilt, 'TILTFILE')
        fntlt = Path(fn, tiltfile)
        tilt_m_st, tilt_list = _get_tltmatrix(fntlt)
        tomo.setTiltAngles(tilt_list)

        (xdim, ydim, _) = tomo.getRoi()
        thck = tomo.getThickness()
        # shift = np.array([(xdim - 1)/ 2., (ydim - 1) / 2., thck])
        rot_centerX = -0.5 * ((xdim + 1) % 2)
        rot_centerY = -0.5 * ((ydim + 1) % 2)
        rot_centerZ = -0.5 * ((thck + 1) % 2)
        
        orig3D_shifts = np.array([rot_centerX, rot_centerY, rot_centerZ])
        mOrig3D = tf.tr3d.shifts2mat(orig3D_shifts)
    
        if (flipz):
            mat_flip = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, -1, 0],
                                 [0, 0, 0, 1]])
        else:
            mat_flip = np.eye(4)
        mat_flip = tf.TMat3D(mat_flip)
    
        (offx, offy, offz) = tomo.getShifts()
        offsets = np.array([-offx, 0, -offz])
        mat_off = tf.tr3d.shifts2mat(offsets)
    
        tmat_stack = getIniMat(size)
        smat = mOrig3D.inv * mat_flip * mat_off
        for i in range(size):
            mat = mOrig3D * xf_m_st[i] * tilt_m_st[i] * smat
            tmat_stack.set(mat, i)
        tomo.setRecTMat(tmat_stack)
        return tomo
    else:
        raise ('iroot must be a valid directory that contains all '\
               'imod asociated files such as newst.com and tilt.com')


def getTomoFromFiles(img_arr, fn, tltfn, xrpix, flipz):
    basename = fn.name.split(".")[0]
    fn_base = Path(fn.parent, basename)
    tomo = tm.Tomogram(xrpix)
    shape = img_arr.shape
    # For a 3D array we assume is a stack if nz <= 250.
    print("SHAPE: ", shape)
    
    if shape[0] <= 250:
        tomo.setFilename(str(fn))
        size = tomo.getSize()
        transfn = Path(str(fn_base) +  ".xf")
        xf_m_st = _get_xfmatrix(transfn, size)
        
        fntlt = Path(tltfn)
        tilt_m_st, tilt_list = _get_tltmatrix(fntlt)
        tomo.setTiltAngles(tilt_list)
    else:
        fntlt = Path(tltfn)
        if fntlt.exists():
           tilt_m_st, tilt_list = _get_tltmatrix(fntlt)
        else:
            # implement when angles are not given
            raise Exception("not given tlt file will be implememnted later")
        
        tomo.setTiltAngles(tilt_list)
        img_projs = pr.projectRS(img_arr, tilt_m_st)
        imgs_fn = Path(fn_base + ".ali")
        imgs_fn.with_suffix('')
        io.imwrite(imgs_fn, img_projs.astype(np.float32))
        tomo.setFilename(str(imgs_fn))
        size = tomo.getSize()
        xf_m_st = getIniMat(size)
    
    tomo.setBinning(1)
    roi = (0, 0, 0)
    tomo.setRoi(roi)
    tomo.setThickness(int(size))
    shifts = (0, 0, 0)
    tomo.setShifts(shifts)
    
    (xdim, ydim, _) = tomo.getRoi()
    thck = tomo.getThickness()
    
    rot_centerX = -0.5 * ((xdim + 1) % 2)
    rot_centerY = -0.5 * ((ydim + 1) % 2)
    rot_centerZ = -0.5 * ((thck + 1) % 2)
    
    orig3D_shifts = np.array([rot_centerX, rot_centerY, rot_centerZ])
    mOrig3D = tf.tr3d.shifts2mat(orig3D_shifts)
    
    if (flipz):
        mat_flip = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]])
    else:
        mat_flip = np.eye(4)
    mat_flip = tf.TMat3D(mat_flip)
    
    (offx, offy, offz) = tomo.getShifts()
    offsets = np.array([-offx, 0, -offz])
    mat_off = tf.tr3d.shifts2mat(offsets)
    
    tmat_stack = getIniMat(size)
    smat = mOrig3D.inv * mat_flip * mat_off
    for i in range(size):
        mat = mOrig3D * xf_m_st[i] * tilt_m_st[i] * smat
        tmat_stack.set(mat, i)
    tomo.setRecTMat(tmat_stack)
    return tomo