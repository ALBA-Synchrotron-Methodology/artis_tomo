# -*- coding: utf-8 -*-

""" Read/Write images using Delta vision microscope image format.

Backend Library: `mrc <https://github.com/tlambert03/mrc>`_

Plugin that wraps the mrc library. MRC is designed to read the MRC variant
used by deltavision microscopes (.dv) and the IVE library from UCSF. For the
MRC2014 file format frequently used for structural biology, see mrcfile.

Parameters
----------
request : Request
    A request object representing the resource to be operated on.

Methods
-------
.. autosummary::
    :toctree: _plugins/mrcfile
    dvPlugin.read
    dvPlugin.write
    dvPlugin.iter
    dvPlugin.get_meta
"""

from typing import Optional, Dict, Any, Iterator
import numpy as np
from imageio.core.request import Request, IOMode
from imageio.core.v3_plugin_api import PluginV3 #, ImageProperties
from .imageio_plugin_api import ImageProperties
import warnings
from imageio.typing import ArrayLike

try:
    import mrc as _mrc
except ImportError:
    warnings.warn("If you need support for DV file formats, install "
                  "the mrc module directly:"
                  " `pip install mrc`")
    raise ImportError


class dvPlugin(PluginV3):

    def __init__(self, request: Request, mmap=False) -> None:
        """Instantiate a new DVFile Plugin Object.

        Parameters
        ----------
        request : {Request}
            A request object representing the resource to be operated on.

        mmap: Boolean
            Open a memory-mapped file (fast for large files).
            Default True.
        """
        super().__init__(request)

        self._image = None

        self._mmap = mmap

        if request.mode.io_mode == IOMode.read:

            self._image = _mrc.DVFile(request.get_local_filename())
            self._image.data.flags['WRITEABLE'] = False
        else:
            self._image = _mrc.Mrc2(request.get_local_filename(), mode='w+')

    def close(self) -> None:
        if self._image:
            self._image.close()

        self._request.finish()

    def read(self, *, index: int = None, C: int = None,
             T: int = None) -> np.ndarray:
        """
        Parse the given URI and creates a ndarray from it.

        Parameters
        ----------
        index : int
            If the ImageResource contains multiple Z images, and index is an
            integer, select the index-th z-slice from among them and return it.
            If index is None, this plugin reads the full image stack.
        C : int
            If the ImageResource contains multiple C channels, and index is an
            integer, select the C-th channel from among them and return it.
            If C is None, this plugin reads all the channels.
        T : int
            If the ImageResource contains multiple T timestamps images, and T
            is an integer, select the T-th snapshot from among them and return it.
            If T is None, this plugin reads all the timestamps.

        Returns
        -------
        ndimage : ndarray
            A numpy array containing the loaded image data

        """
        axes = self._image.axes
        sldic = dict()
        slNone = slice(None)
        sldic['Z'] = slNone if index is None else index
        sldic['C'] = slNone if C is None else C
        sldic['T'] = slNone if T is None else T
        sldic['Y'] = slNone
        sldic['X'] = slNone

        sl = []
        for c in axes:
            sl.append(sldic[c])

        image = np.squeeze(self._image.data[tuple(sl)])

        if not self._mmap:
            image = image.copy()

        return image

    def iter(self, *, C: int = None, T: int = None) -> Iterator[np.ndarray]:
        """
        Iterate over all z-slices from the DVfile.

        Parameters
        ----------
        C : int
            If the ImageResource contains multiple C channels, and index is an
            integer, select the C-th channel from among them and return it.
            If C is None, this plugin reads all the channels.
        T : int
            If the ImageResource contains multiple T timestamps images, and T
            is an integer, select the T-th snapshot from among them and return it.
            If T is None, this plugin reads all the timestamps.

        Yields
        ------
        ndimage : np.ndarray
            A ndimage.
        """
        axes = self._image.axes
        zId = axes.find('Z')
        nimg = self._image.shape[zId]

        sldic = dict()
        slNone = slice(None)
        sldic['Z'] = 0
        sldic['C'] = slNone if C is None else C
        sldic['T'] = slNone if T is None else T
        sldic['Y'] = slNone
        sldic['X'] = slNone

        sl = []
        for c in axes:
            sl.append(sldic[c])

        for nz in range(nimg):
            sl[zId] = nz
            image = np.squeeze(self._image.data[tuple(sl)])

            if not self._mmap:
                image = image.copy()

            yield image

    def write(self, ndimage: ArrayLike, *, pixel_size=None, pixel_size_z=None,
              channels=None, zAxisOrder='wtz', hdr=None, calcMMM=True,
              metadata=None) -> Optional[bytes]:
        """
        Write an ndimage to the URI specified in path.

        If the URI points to a file on the current host and the file does not
        yet exist it will be created. If the file exists already, it will be
        appended if possible; otherwise, it will be replaced.

        Parameters
        ----------
        image : ndarray or list
            The ndimage to write. If a list is given each element is expected to
            be an ndimage.
        pixel_size : float
            Pixel size value set in header for X and Y dimensions.
        pixel_size_z : float
            Pixel size value set in header for Z dimension.
        channels : List[int, ...]
            List of wavelength that define each channel in the array.
         zAxisOrder : str
             Set the dimensional order for time, wave (channel) and z axes.
             Use zAxisOrder if arr.ndim > 3. Examples:
                 4D: time,z,y,x          -->  zAxisOrder= 't z'
                 4D: wave,z,y,x          -->  zAxisOrder= 'w z'
                 5D: time, wave, z,y,x   -->  zAxisOrder= 't,z,w'
             Spaces,commas,dots,minuses  are ignored. If zAxisOrder None:
                  3D: 'z'
                  4D: 'tz'
                  5D: 'wtz'
        hdr : MRC header struct
            To use already read headers or customized after creating them using
            mrc.makeHdrArray().
            If hdr is not None: copy all fields(except 'Num',...)
        calcMMM: Bool
            Calculate min,max,mean of data set and set hdr field. Default True.
        metadata : Dict
            Fields to overwrite in the header, accepts all field names in hdr.

        Example
        -------

        import imageio as iio

        data = np.random.rand(3, 1, 10, 128, 128)
        f = iio.imopen('test_save.dv', 'w')
        f.write(data, zAxisOrder='wtz', pixel_size=1.5, pixel_size_z=3.2,
                channels=[488, 513,612,0,0])

        """
        if isinstance(ndimage, list):
            ndimage = np.stack(ndimage, axis=0)
            is_batch = True
        else:
            ndimage = np.asarray(ndimage)

        self._initHdrForArr(ndimage, zAxisOrder)

        if calcMMM:
            _mrc.mrc.calculate_mmm(ndimage, self._image)

        if metadata is None:
            metadata = {}

        if pixel_size is not None:
            metadata['dx'] = pixel_size
            metadata['dy'] = pixel_size
        if pixel_size_z is not None:
            metadata['dz'] = pixel_size
        if channels is not None:
            metadata['wave'] = channels

        if len(metadata) > 0:
            _mrc.mrc.add_metadata(metadata, self._image.hdr)

        self._image.writeHeader()
        self._image.writeStack(ndimage)

        return None

    def _initHdrForArr(self, arr, zAxisOrder=None):

        if zAxisOrder is None:
            zAxisOrder = _mrc.mrc.pick_zAxisOrder(arr)
        else:
            import re

            # remove delimiter characters '-., '
            zAxisOrder = re.sub("[-., ]", "", zAxisOrder)

        mrcmode = _mrc.mrc.dtype2MrcMode(arr.dtype.type)
        _mrc.mrc.init_simple(self._image.hdr, mrcmode, arr.shape)

        if arr.ndim == 2:
            pass
        elif arr.ndim == 3:
            if zAxisOrder[-1] == "z":
                self._image.hdr.ImgSequence = 0
            elif zAxisOrder[-1] == "w":
                self._image.hdr.ImgSequence = 1
                self._image.hdr.NumWaves = arr.shape[-3]
            elif zAxisOrder[-1] == "t":
                self._image.hdr.ImgSequence = 2
                self._image.hdr.NumTimes = arr.shape[-3]
            else:
                raise ValueError("unsupported axis order")
        elif arr.ndim == 4:
            if zAxisOrder[-2:] == "zt":
                self._image.hdr.ImgSequence = 2
                self._image.hdr.NumTimes = arr.shape[-3]
            elif zAxisOrder[-2:] == "tz":
                self._image.hdr.ImgSequence = 0
                self._image.hdr.NumTimes = arr.shape[-4]
            elif zAxisOrder[-2:] == "wz":
                self._image.hdr.ImgSequence = 0
                self._image.hdr.NumWaves = arr.shape[-4]
            elif zAxisOrder[-2:] == "zw":
                self._image.hdr.ImgSequence = 1
                self._image.hdr.NumWaves = arr.shape[-3]
            else:
                raise ValueError("unsupported axis order")
        elif arr.ndim == 5:
            # hdr.ImgSequence (0 = ZTW, 1 = WZT, 2 = ZWT)
            # zAxisOrder     ["wtz", "tzw", "twz"]
            if zAxisOrder == "wtz":
                self._image.hdr.ImgSequence = 0
                self._image.hdr.NumTimes = arr.shape[-4]
                self._image.hdr.NumWaves = arr.shape[-5]
            elif zAxisOrder == "tzw":
                self._image.hdr.ImgSequence = 1
                self._image.hdr.NumTimes = arr.shape[-5]
                self._image.hdr.NumWaves = arr.shape[-3]
            elif zAxisOrder == "twz":
                self._image.hdr.ImgSequence = 2
                self._image.hdr.NumTimes = arr.shape[-5]
                self._image.hdr.NumWaves = arr.shape[-4]
            else:
                raise ValueError(f"unsupported axis order {zAxisOrder}")
        else:
            raise ValueError("unsupported array ndim")
        self._image._initWhenHdrArraySet()


    def get_meta(self, *, index=0, C=0, T=0) -> Dict[str, Any]:
        """Read metadata.

        Parameters
        ----------
        index : int
            If the ImageResource contains multiple Z images, and index is an
            integer, select the index-th z-slice from among them and return it.
        C : int
            If the ImageResource contains multiple C channels, and index is an
            integer, select the C-th channel from among them and return it.
        T : int
            If the ImageResource contains multiple T timestamps images, and T
            is an integer, select the T-th snapshot from among them and return it.

        Returns
        -------
        metadata : dict
            A dictionary of format-specific metadata.
        """
        values = {'Z': index, 'C': C, 'T': T}
        axes = self._image.axes[:3]
        label = ''

        for c in axes:
            label += f'{c}{values[c]}'

        return self._image.ext_hdr._asdict()[label]

    def metadata(self) -> Dict[str, Any]:
        """Read metadata.

        Returns
        -------
        metadata : dict
            A dictionary of format-specific metadata.
        """
        metadata = self._image.hdr._asdict()

        return metadata

    def properties(self) -> ImageProperties:
        """
        Standardized metadata.

        Returns
        -------
        properties : ImageProperties
            A dataclass filled with standardized image metadata.

        """
        imProp = ImageProperties(shape=self._image.shape,
                                 dtype=self._image.data.dtype,
                                 n_images=1,
                                 n_channels=self._image.hdr.nc,
                                 n_times=self._image.hdr.nt, is_batch=False)

        return imProp

