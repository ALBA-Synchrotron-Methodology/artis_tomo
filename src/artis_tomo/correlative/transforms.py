#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistence of the transformations a correlative alignment produces.

Written out so that the expensive stages of a run can be skipped on the next
one: steps 1 and 2 of the 3D pipeline depend only on the mosaic and the
fluorescence data, so once they are known the 3D refinement can be re-run on
its own with different parameters.
"""

from pathlib import Path

import numpy as np

from artis_sci.io.imageIO import savetoh5File, loadfromh5File
from artis_sci.math import transforms as tf


TRANSFORMS_FILENAME = 'clfluoxr3d_transforms.h5'


def saveTransforms(root, fluoToMosaic, tomoCentralToMosaic, tomoToFluoVol,
                   filename=TRANSFORMS_FILENAME):
    """
    Write the three transformations of a 3D correlative run to HDF5.

    Parameters
    ----------
    root : str or Path
        Directory to write into.
    fluoToMosaic : TMat3D
        Fluorescence projection to mosaic, from step 1.
    tomoCentralToMosaic : TMat3D
        Central tomogram projection to mosaic, from step 2.
    tomoToFluoVol : TMat3D
        Tomogram volume to fluorescence volume, the result of step 3.
    filename : str, optional
        Name of the file within *root*.

    Returns
    -------
    Path
        The file written.
    """
    out_fn = Path(root, filename)

    savetoh5File(out_fn, {
        'fluo_to_mosaic': np.squeeze(fluoToMosaic.matrix),
        'tomocentral_to_mosaic': np.squeeze(tomoCentralToMosaic.matrix),
        'tomo_to_fluo_vol': np.squeeze(tomoToFluoVol.matrix),
    })

    print(f"Transformation matrices saved to: {out_fn}")

    return out_fn


def loadTransforms(fname):
    """
    Read back the transformations needed to skip steps 1 and 2.

    Only the two that those steps produce are returned; the step 3 result is
    stored for reference but is what a re-run recomputes.

    Parameters
    ----------
    fname : str or Path
        HDF5 file written by :func:`saveTransforms`.

    Returns
    -------
    fluoToMosaic, tomoCentralToMosaic : TMat3D
    """
    data = loadfromh5File(fname)

    fluoToMosaic = tf.TMat3D(np.asarray(data['fluo_to_mosaic']))
    tomoCentralToMosaic = tf.TMat3D(np.asarray(data['tomocentral_to_mosaic']))

    print(f"Transformation matrices loaded from: {fname}")

    return fluoToMosaic, tomoCentralToMosaic
