"""
Generic tomo I/O functions.

@author: joton
"""
import os
import numpy as np
import mrcfile
from artis_tomo.io.image_formats.mistral_h5 import MistralFile
from artis_tomo.utils.struct import StructBase

__all__ = ['readTomo']


def readTomo(fnIn):

    ext = os.path.splitext(fnIn)[1]

    if ext == '.mrc':
        array = mrcfile.open(fnIn).data
    elif ext in ['.hdf5', '.h5']:
        mistralh5 = MistralFile(fnIn)
        dataset = 'TomoNormalized/TomoNormalized'
        if dataset in mistralh5:
            array = mistralh5[dataset][...]
        else:
            mistralh5.preprocess()
            array = mistralh5.getStackNorm()
    else:
        raise Exception(f'Unknown tomo file format {ext}.')

    if array.ndim < 3:
        array = array[np.newaxis, :, :]

    return array

""" Not need it at this moment"""
# class tomoStruct(StructBase):
#
#     attribs = ['fn', 'pxsize', 'ts', 'recmats', 'fluomats', 'bin',
#                'roi', 'shifts', 'thickness', 'angs']
#
#     def __init__(self, inputdict: dict = None):
#         """
#         Initialize psfStruct and, optionally, from dictionary.
#
#         If inputdict is passed, it must contain all required attribs:
#             'fn', 'pxsize', 'ts', 'recmats', 'fluomats', 'bin',
#                'roi', 'shifts', 'thickness', 'angs'
#
#         """
#         super(tomoStruct, self).__init__(inputdict)
#
#     @classmethod
#     def tomo2Dict(cls, tomo):
#         return cls(dict(fn=tomo.getFilename(),pxsize=tomo.getSamplingRate(),
#                         recmats=tomo.getRecTMat().getMatrix(),
#                         fluomats=tomo.getFluoTmat().getMatrix(),
#                         bin=tomo.getBinning(),
#                         roi=tomo.getRoi(),
#                         shifts=tomo.getShifts(),
#                         thickness=tomo.getThickness(),
#                         angs = tomo.getTiltAngles()))
    
        
