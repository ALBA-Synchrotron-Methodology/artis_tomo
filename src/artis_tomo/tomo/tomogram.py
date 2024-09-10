#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tomogram class for tomography

@authors: Josue Gomez & Joaquin Oton
"""
import numpy as np
import artis_tomo.io.imageIO as iio
from artis_tomo.math import transforms as tf

class Tomogram():
    def __init__(self, pxsize):
        self._stfn = str()
        self._pxsize = pxsize
        self._tilt_series = None
        self._rec_tmats = tf.TMat3D(np.eye(4))
        self._align_mo_tmats = tf.TMat3D(np.eye(4))
        self._binning = float()
        self._roi = tuple()
        self._shift = tuple()
        self._thickness = int(0)
        self._angs = list()

    def getFilename(self):
        return self._stfn
    
    def setFilename(self, fn):
        if type(fn) is str:
            self._stfn = fn
            self.setTiltseries()
        else:
            raise ValueError('filename must be a string')

    def getSamplingRate(self):
        return self._pxsize
    
    def setSamplingRate(self, value):
        self._pxsize = value
    
    def getBinning(self):
        return self._binning
    
    def setBinning(self, value):
        self._binning = value
    
    def setRoi(self, val):
        if type(val) is tuple and len(val) == 3:
            self._roi = val
        else:
            raise Exception('roi must be a tuple with 3 values.')

    def getRoi(self):
        return self._roi

    def setShifts(self, val):
        if type(val) is tuple and len(val) == 3:
            self._shifts = val
        else:
            raise Exception ('shifts must be a tuple with 3 values.')

    def getSize(self):
        return self.getTiltseries().shape[0]
    
    def getDimensions(self):
        return self.getTiltseries().shape
    
    def getShifts(self):
        return self._shifts
    
    def setThickness(self, val):
        self._thickness = val
    
    def getThickness(self):
        return self._thickness
    
    def getTiltseries(self):
        return self._tilt_series
    
    def getTiltAngles(self):
        return self._angs
    
    def getImage(self, order):
        if order=='max':
            indx = len(self._angs)
        elif order=='min':
            indx = 0
        elif order=='center':
            indx = self._angs.index(min(list(map(abs, self._angs))))
        elif type(order)==int:
            indx = order
        else:
            raise ("Error: Enter a number, \'max\', \'min\' or \'center\' ")
        
        return self._tilt_series[indx]
    
    def getRecTmatImg(self, order):
        if order=='max':
            indx = len(self._angs)
        elif order=='min':
            indx = 0
        elif order=='center':
            indx = self._angs.index(min(list(map(abs, self._angs))))
        elif type(order)==int:
            indx = order
        else:
            raise ("Error: You must enter an integer,"
                   " \'max\', \'min\' or \'center\' ")
        return self._rec_tmats[indx]
        
    def setTiltseries(self, img_arr=None):
        if img_arr is None:
            fn = self.getFilename()
            img_arr = iio.imread(fn)
            
            if img_arr.ndim == 3:
                img_arr = img_arr[:, :, :]
            else:
                raise ("Error: You must enter only an image stack")
        self._tilt_series = img_arr

    def getRecTMat(self, order=None):
        if order is not None:
            if order=='max':
                indx = len(self._angs)
            elif order=='center':
                indx = self._angs.index(min(list(map(abs, self._angs))))
            elif type(order)==int:
                indx = order
            return self.getRecTMat()[indx]
        else:
            return self._rec_tmats
    
    def setRecTMat(self, m):
        self._rec_tmats = m
    
    def getAlignTmat(self):
        return self._align_mo_tmats

    def setAlignTMat(self, m):
        self._align_mo_tmats = m

    def setTiltAngles(self, anglist):
        self._angs = anglist
    
    def exportDict(self):
        flMat = self.getAlignTmat().matrix
        return dict(fn=self.getFilename(),
                    pxsize=self.getSamplingRate(),
                    recmats=self.getRecTMat().matrix,
                    fluomats=flMat,
                    bin=self.getBinning(),
                    roi=self.getRoi(),
                    shifts=self.getShifts(),
                    thickness=self.getThickness(),
                    angs=self.getTiltAngles())
    
    def importDict(self, dic):
        self.setFilename(dic['fn'])
        self.setSamplingRate(dic['pxzise'])
        self.setBinning(dic['bin'])
        self.setRoi(tuple(map(tuple, dic['roi'])))
        self.setThickness(dic['thickness'])
        self.setShifts(tuple(map(tuple, dic['shifts'])))
        self.setTiltAngles(dic['angs'])
        if dic['recmats'] == 0:
            self.setRecTMat(None)
        else:
            tmat = tf.TMat3D(dic['recmats'])
            self.setRecTMat(tmat)
        
        if dic['fluomats'] == 0:
            self.setAlignTMat(None)
        else:
            tmat = tf.TMat3D(dic['fluomats'])
            self.setAlignTMat(tmat)
