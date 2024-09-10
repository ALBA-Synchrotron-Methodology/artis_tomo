"""
Custom Mistral struct in HDF5 container.

@author: joton
"""

import numpy as np
import h5py
from artis_tomo.math.statistics import meanStd
from artis_tomo.image import (filter as ft,
                     frame as fr)

__all__ = ['mistralDict', 'MistralFile']

mistralDict = {'data':  '/NXtomo/instrument/sample/data',
               'angles': 'NXtomo/data/rotation_angle',
               'exptime': 'NXtomo/instrument/sample/ExpTimes',
               'current': 'NXtomo/instrument/sample/current',
               'iniIm': 'NXtomo/instrument/sample/0_degrees_initial_image',
               'endIm': 'NXtomo/instrument/sample/0_degrees_final_image',
               'ff_data': 'NXtomo/instrument/bright_field/data',
               'ff_exptime': 'NXtomo/instrument/bright_field/ExpTimes',
               'ff_current': 'NXtomo/instrument/bright_field/current'
               }


class MistralFile():
    """Class to handle hdf5 files created by Mistral microscope."""

    def __init__(self, fName):

        self.h5File = h5py.File(fName)

# To call h5py.File methods as own methods
    def __getattr__(self, name):
        """Native h5py class methods are kept in MistralFile class."""
        try:
            return getattr(self.h5File, name)
        except AttributeError:
            raise AttributeError(
             "'%s' object has no attribute '%s'" % (type(self).__name__, name))

    # To use [] directly in MistralFile class as h5py.File
    def __getitem__(self, arg):
        """
        h5py class items can be called directly in MistralFile class.

        Example:
        -------
                    mFile = MistralFile('datafile.h5')
                    data = mFile['dataset']
        """
        return self.h5File[arg]

    def __contains__(self, name):
        """Search content redirected to h5File."""
        return self.h5File.__contains__(name)

    def processFlafield(self):
        """Calculate the avg flatfield to normalize stack projections."""
        ffShape = self.h5File[mistralDict['ff_data']].shape
        nFF = ffShape[0]

        if mistralDict['ff_exptime'] in self.h5File:
            ffExp = self.h5File[mistralDict['ff_exptime']]
        else:
            ffExp = np.ones(nFF)

        if mistralDict['ff_current'] in self.h5File:
            ffCur = self.h5File[mistralDict['ff_current']]
        else:
            ffCur = np.ones(nFF)

        ffmean = np.zeros(ffShape[1:])

        for k in range(nFF):
            ffmean += self.h5File[mistralDict['ff_data']][k, ...] / \
                      (ffExp[k]*ffCur[k])

        ffmean /= nFF

        if self.applyCrop:
            ffmean = fr.cropArrayCentered(ffmean, self.outSize[0:2])

        self.flatfield = ffmean

    def preprocess(self, cropSize=(0, 0), kfilter=3, applyFilter=True):
        """
        Prepare calculations to normalize image projections.

        Parameters
        ----------
        cropSize : pixels to be cropped at each side in Y and X dimension
                    (cY, cX). Default: (0, 0)
        kfilter:    Bad pixels from camera sensor are corrected by a boundaries
                    median filter for those whose value < mean -kfilter*std.
                    Default: 3
        applyFilter : Apply boundaries median filter.
                      Default: True
        """
        self.applyCrop = cropSize > (0, 0)
        self.cropSize = cropSize
        self.applyFilter = applyFilter
        self.kfilter = kfilter

        self.dataShape = np.array(self.h5File[mistralDict['data']].shape)
        self.nProj = self.dataShape[0]

        self.outSize = self.dataShape[[1, 2, 0]] - \
            2 * np.array(self.cropSize+(0, ))

        self.processFlafield()

        if self.applyFilter:
            mean, std = meanStd(self.flatfield)
            thr = mean - self.kfilter*std
            self.mask = self.flatfield < thr
            self.flatfield = ft.boundMedianFilter(self.flatfield, self.mask)

        if mistralDict['exptime'] in self.h5File:
            dataExp = self.h5File[mistralDict['exptime']]
        else:
            dataExp = np.ones(self.nProj)

        if mistralDict['current'] in self.h5File:
            dataCur = self.h5File[mistralDict['current']]
        else:
            dataCur = np.ones(self.nProj)

        self.dataExp = dataExp
        self.dataCur = dataCur

    def getProj(self, idx):
        """
        Return one corrected single projection no flatfield normalized.

        Parameters
        ----------
        idx : Stack image index (Indices begins in 0)

        Returns
        -------
        image : 2d array
        """
        assert idx < self.nProj, 'Selected image number out of range'

        proj = self.h5File[mistralDict['data']][idx, ...] / \
            (self.dataExp[idx]*self.dataCur[idx])
        if self.applyCrop:
            proj = fr.cropArrayCentered(proj, self.outSize[0:2])
        if self.applyFilter:
            proj = ft.boundMedianFilter(proj, self.mask)

        return proj

    def getProjNorm(self, idx):
        """
        Return one single projection corrected by exposition and current and
        flatfield normalized.

        Parameters
        ----------
        idx : Stack image index (Indices begins in 0)

        Returns
        -------
        image : 2d array
        """
        return self.getProj(idx)/self.flatfield

    def getStackNorm(self):
        """
        Return the whole image projections stack corrected and normalized.

        Returns
        -------
        image : 3d array
        """
        stackOut = np.empty(self.outSize)

        for k in range(self.nProj):

            if self.applyCrop:
                stackOut[..., k] = fr.cropArrayCentered(
                    self.h5File[mistralDict['data']][k, ...],
                    self.outSize[0:2])
            else:
                stackOut[..., k] = self.h5File[mistralDict['data']][k, ...]

            stackOut[..., k] /= (self.flatfield * self.dataExp[k] *
                                 self.dataCur[k])

            if self.applyFilter:
                stackOut[..., k] = ft.boundMedianFilter(stackOut[..., k],
                                                        self.mask)

        return stackOut
