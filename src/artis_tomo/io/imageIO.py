#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:01:24 2017

@author: joton
"""

import h5py
import numpy as np
import mrcfile as mrc
from artis_tomo.utils.dict import AttributeDict

import importlib
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

# Mistral custom h5 reader
from .image_formats.mistral_h5 import *
# Tomo custom functions
from .tomo import *

if importlib.util.find_spec('em') is not None:
    import em as emcore
    emcoreExist = True
elif importlib.util.find_spec('emcore') is not None:
    import emcore
    emcoreExist = True
else:
    emcoreExist = False

from .image_formats.image_fits import readFITS

#  Import imageio library adding mrcfile plugin
import imageio as _iio
# We integrate v3 methods in our imageIO module
from imageio.v3 import imread, imwrite, imiter, improps, immeta

_plugins = _iio.config.plugins
_extensions = _iio.config.extensions
_my_extension_list = []

# mrcfile plugins register
_plugins.known_plugins["mrcfile"] = _plugins.PluginConfig(
    name="mrcfile",
    class_name="MrcfilePlugin",
    module_name="artis_tomo.io.image_formats.mrcfile",
)
for ext in ['.mrc', '.mrcs', '.st', '.ali', '.rec']:
    _my_extension_list.append(
        _extensions.FileExtension(name="MRC file",
                                  extension=ext,
                                  priority=["mrcfile"]))

# dvfile plugins register
_plugins.known_plugins["dvfile"] = _plugins.PluginConfig(
    name="dvfile",
    class_name="dvPlugin",
    module_name="artis_tomo.io.image_formats.dvfile",
)
_my_extension_list.append(
    _extensions.FileExtension(name="DV file",
                              extension='.dv',
                              priority=["dvfile"]))

# Adding new extensions to imageio known_extensions as first option
for ext in _my_extension_list:
    if ext.extension not in _extensions.known_extensions:
        _extensions.known_extensions[ext.extension] = list()
    _extensions.known_extensions[ext.extension].insert(0, ext)

# Replace tifffile plugin by our custom plugin where class init method ignores
# unrecognized parameters
_plugins.known_plugins['tifffile'].module_name =\
    "artis_tomo.io.image_formats.tifffile"
__all__ = ["imopen", "imread", "imwrite", "imiter", "improps", "immeta",
           "readMRC", "writeMRC", "mrcMmapTry"]


# We replace imageio.v3.imopen to set default io_mode to "r"
def imopen(
    uri,
    io_mode='r',
    *,
    plugin=None,
    extension=None,
    format_hint=None,
    legacy_mode=False,
    **kwargs,
):
    """
    Open an ImageResource.

    .. warning::
        This warning is for pypy users. If you are not using a context manager,
        remember to deconstruct the returned plugin to avoid leaking the file
        handle to an unclosed file.

    Parameters
    ----------
    uri : str or pathlib.Path or bytes or file or Request
        The :doc:`ImageResource <../../user_guide/requests>` to load the
        image from.
    io_mode : str
        The mode in which the file is opened. Possible values are::

            ``r`` - open the file for reading
            ``w`` - open the file for writing

        Depreciated since v2.9:
        A second character can be added to give the reader a hint on what
        the user expects. This will be ignored by new plugins and will
        only have an effect on legacy plugins. Possible values are::

            ``i`` for a single image,
            ``I`` for multiple images,
            ``v`` for a single volume,
            ``V`` for multiple volumes,
            ``?`` for don't care (default)

    plugin : str, Plugin, or None
        The plugin to use. If set to None (default) imopen will perform a
        search for a matching plugin. If not None, this takes priority over
        the provided format hint.
    extension : str
        If not None, treat the provided ImageResource as if it had the given
        extension. This affects the order in which backends are considered, and
        when writing this may also influence the format used when encoding.
    format_hint : str
        Deprecated. Use `extension` instead.
    legacy_mode : bool
        If true (default) use the v2 behavior when searching for a suitable
        plugin. This will ignore v3 plugins and will check ``plugin``
        against known extensions if no plugin with the given name can be found.
    **kwargs : Any
        Additional keyword arguments will be passed to the plugin upon
        construction.

    Notes
    -----
    Registered plugins are controlled via the ``known_plugins`` dict in
    ``imageio.config``.

    Passing a ``Request`` as the uri is only supported if ``legacy_mode``
    is ``True``. In this case ``io_mode`` is ignored.

    Using the kwarg ``format_hint`` does not enforce the given format. It merely
    provides a `hint` to the selection process and plugin. The selection
    processes uses this hint for optimization; however, a plugin's decision how
    to read a ImageResource will - typically - still be based on the content of
    the resource.


    Examples
    --------
    >>> import artis_tomo.io.imageio as iio
    >>> with iio.imopen("/path/to/image.png") as file:
    >>>     im = file.read()

    >>> file = iio.imopen("/path/to/image.png")
    >>> im = file.read()

    >>> # Direct reading
    >>> im = iio.imread("/path/to/image.png", "r")

    >>> with iio.imopen("/path/to/output.jpg", "w") as file:
    >>>     file.write(im)

    >>> # Direct writing
    >>> iio.imwrite("/path/to/output.jpg", im)

    """
    return _iio.v3.imopen(uri, io_mode, plugin=plugin, extension=extension,
                          format_hint=format_hint, legacy_mode=legacy_mode,
                          **kwargs)


def readMRC(filename):
    """
    Read an array from a MRC file

    Parameters
    ----------
    filename : MRC type file name

    Returns
    -------
    images : 2d/3d array
           2d Array in case of single image and 3d array in case of stack of
           images or volume array
    """

    return np.moveaxis(mrc.open(filename).data, 0, -1)


def readEM(filename):

    if not emcoreExist:
        raise Exception('EM-CORE module not detected. Please install!!')

    im = emcore.Image()
    im.read(emcore.ImageLocation(filename))
    return np.array(im)


def writeMRC(array, filename):
    """
    Write an array to a MRC file format

    Parameters
    ----------
    array :     2d/3d array
    filename :  MRC type file name
    """

    file = mrc.new(filename, overwrite=True)
    file.set_data(np.moveaxis(array.astype(np.float32), -1, 0))
    file.close()


def mrcMmapTry(fn):

    try:
        mrcout = mrc.mmap(fn)
    except Exception as err:
        print(f'mrcMmapTry: {err} ... Reading in permissive mode')
        mrcout = mrc.mmap(fn, permissive=True)
    return mrcout




def savetoh5File(fname, dic, root=''):
    """
    Save a dictionary struct of arrays in a hdf5 file.

    Parameters
    ----------
    dic   : dict structure to be stored
    fname : hdf5 filename
    """
    f = h5py.File(fname, 'w')
    grp = f if len(root) == 0 else f.create_group(root)
    dict2h5(dic, grp)
    f.close()


def dict2h5(dic, grp):
    """
    Save a dictionary struct of arrays in a hdf5 group object.

    Parameters
    ----------
    dic   : dict structure to be stored
    grp     : hdf5 group object
    """
    if isinstance(dic, dict):
        for key, value in dic.items():
            if isinstance(value, (dict, list)):
                newGrp = grp.create_group(key)
                dict2h5(value, newGrp)
            else:
                grp.create_dataset(key, data=value)
    elif isinstance(dic, list):
        for ind, value in enumerate(dic):
            label = f'##listItem##{ind}'
            if type(value) in (list, dict):
                newGrp = grp.create_group(label)
                dict2h5(value, newGrp)
            else:
                grp.create_dataset(label, data=value)
    elif isinstance(dic, np.ndarray):
        grp.create_dataset('##array##', data=dic)


def loadfromh5File(fname, trans=False):
    """
    Load a hdf5 struct in a dictionary struct of arrays.

    Parameters
    ----------
    fname : hdf5 input filename
    trans : transpose 2d datasets when stored in arrays.
            Default = False

    Returns
    -------
    dic   : dict structure
    """
    f = h5py.File(fname, 'r')
    dictOut = h52dict(f, trans)
    f.close()
    return dictOut


def h52dict(grp, trans=False):
    """
    Load a hdf5 group struct in a dictionary struct of arrays.

    Parameters
    ----------
    grp   : hdf5 group object
    trans : transpose 2d datasets when stored in arrays.
            Default = False

    Returns
    -------
    dic   : dict structure
    """
    if '##listItem##' in list(grp.keys())[0]:
        listOut = [None]*len(grp)

        for key, value in grp.items():
            ind = int(key[12:])

            if grp.get(key, getclass=True) is h5py._hl.dataset.Dataset:
                listOut[ind] = np.transpose(value[...]) \
                    if (trans and type(value) is np.ndarray) else value[...]
            else:
                listOut[ind] = h52dict(value, trans)

        return listOut

    elif '##array##' in list(grp.keys())[0]:
        value = list(grp.values())[0][...]
        return np.transpose(value) if trans else value

    else:
        dictOut = AttributeDict()
        for key, value in grp.items():

            if grp.get(key, getclass=True) is h5py._hl.dataset.Dataset:
                dictOut[key] = np.transpose(value[...]) \
                    if (trans and type(value) is np.ndarray) else value[...]
                if dictOut[key].dtype == object:
                    dictOut[key] = str(dictOut[key])
            else:
                dictOut[key] = h52dict(value, trans)

        return dictOut
