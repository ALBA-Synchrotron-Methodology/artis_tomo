# -*- coding: utf-8 -*-

""" Read/Write images using mrcfile.

Backend Library: `mrcfile <https://mrcfile.readthedocs.io/en/stable/>`_

Plugin that wraps the the mrcfile library. Mrcfile is a Python implementation
of the MRC2014 file format, which is used in structural biology to store image
and volume data. For the complete list of features and supported formats
please refer to mrcfiles official docs (see the Backend Library link).

Parameters
----------
request : Request
    A request object representing the resource to be operated on.

Methods
-------
.. autosummary::
    :toctree: _plugins/mrcfile
    mrcfilePlugin.read
    mrcfilePlugin.write
    mrcfilePlugin.iter
    mrcfilePlugin.get_meta
"""

from typing import Optional, Dict, Any, Iterator
import numpy as np
from imageio.core.request import Request, IOMode
from imageio.core.v3_plugin_api import PluginV3
import warnings
from imageio.typing import ArrayLike
from .imageio_plugin_api import ImageProperties


try:
    import mrcfile as _mrcfile
except ImportError:
    warnings.warn("If you need support for mrcfile formats, install "
                  "the mrcfile module directly:"
                  " `pip install mrcfile`")
    raise ImportError


class MrcfilePlugin(PluginV3):

    def __init__(self, request: Request, mmap=False) -> None:
        """Instantiate a new Mrcfile Plugin Object.

        Parameters
        ----------
        request : {Request}
            A request object representing the resource to be operated on.

        mmap: Boolean
            Open a memory-mapped file (fast for large files).
            Default True.
        """
        super().__init__(request)

        self._image: _mrcfile.mrcfile.Mrcfile = None
        self._mmap = mmap

        if request.mode.io_mode == IOMode.read:

            if self._mmap:
                self._image = _mrcfile.open(request.get_local_filename(),
                                            permissive=True)
            else:
                self._image = _mrcfile.mmap(request.get_local_filename(),
                                        permissive=True)

        else:
            self._image = _mrcfile.new(request.get_local_filename(),
                                       overwrite=True)

    def close(self) -> None:
        if self._image:
            self._image.close()

        self._request.finish()

    def read(self, *, index: int = None, is_batch: bool = None) -> np.ndarray:
        """
        Parse the given URI and creates a ndarray from it.

        Parameters
        ----------
        index : int
            If the ImageResource contains multiple ndimages, and index is an
            integer, select the index-th ndimage from among them and return it.
            If index is None, this plugin reads the full image stack.
        is_batch : bool
            Explicitly tell the reader that ``image`` is a batch of images/volumes
            (True) or not (False). If None, the reader will guess this from the
            header or ``image.shape``.

        Returns
        -------
        ndimage : ndarray
            A numpy array containing the loaded image data

        """
        if index is None:
            image = self._image.data
        elif self._image.is_image_stack() or self._image.is_volume_stack() or\
            is_batch is True:
            image = self._image.data[index]
        else:
            raise Exception('Mrcfile is not a stack of images/volumes.')

        return image

    def iter(self) -> Iterator[np.ndarray]:
        """
        Iterate over all ndimages/volumes from the MRC file.

        Yields
        ------
        ndimage : np.ndarray
            A decoded ndimage.
        """
        if self._image.is_image_stack() or self._image.is_volume_stack():
            for im in self._image.data:
                yield im
        else:
            raise Exception('Mrcfile is not a stack of images/volumes.')


    def write(self, ndimage: ArrayLike, *, is_batch: bool = False,
              pixel_size=None, **kwargs) -> Optional[bytes]:
        """
        Write an ndimage to the URI specified in path.
        If the URI points to a file on the current host and the file does not
        yet exist it will be created. If the file exists already, it will be
        appended if possible; otherwise, it will be replaced.
        If necessary, the image is broken down along the leading dimension to
        fit into individual frames of the chosen format. If the format doesn't
        support multiple frames, and IOError is raised.
        Parameters
        ----------
        image : ndarray or list
            The ndimage to write. If a list is given each element is expected to
            be an ndimage.
        mode : str
            Specify the image's color format. If None (default), the mode is
            inferred from the array's shape and dtype. Possible modes can be
            found at:
            https://Mrcfile.readthedocs.io/en/stable/handbook/concepts.html#modes
        format : str
            Optional format override. If omitted, the format to use is
            determined from the filename extension. If a file object was used
            instead of a filename, this parameter must always be used.
        is_batch : bool
            Explicitly tell the writer that ``image`` is a batch of images
            (True) or not (False). If None, the writer will guess this from the
            provided ``mode`` or ``image.shape``. While the latter often works,
            it may cause problems for small images due to aliasing of spatial
            and color-channel axes.
        kwargs : ...
            Extra arguments to pass to Mrcfile. If a writer doesn't recognise an
            option, it is silently ignored. The available options are described
            in Mrcfile's `image format documentation
            <https://Mrcfile.readthedocs.io/en/stable/handbook/image-file-formats.html>`_
            for each writer.
        Notes
        -----
        When writing batches of very narrow (2-4 pixels wide) gray images set
        the ``mode`` explicitly to avoid the batch being identified as a colored
        image.
        """
        if isinstance(ndimage, list):
            ndimage = np.stack(ndimage, axis=0)
            is_batch = True
        else:
            ndimage = np.asarray(ndimage)

        # check if ndimage is a batch of images/volumes
        if ndimage.ndim == 2:
            is_batch = False
        elif ndimage.ndim == 3 and ndimage.shape[0] == 1:
            ndimage = np.squeeze(ndimage)
            is_batch = False
        elif ndimage.ndim == 4:
            if ndimage.shape[:2] == (1, 1):
                ndimage = np.squeeze(ndimage)
                is_batch = False
            else:
                is_batch = True

        self._image.set_data(ndimage)

        if pixel_size is not None:
            self._image.voxel_size = (pixel_size, )*3

        if ndimage.ndim == 3:
            if is_batch:
                self._image.set_image_stack()
            else:
                self._image.set_volume()

        return None

    def get_meta(self, *, index=0) -> Dict[str, Any]:
        return self.metadata(index=index)

    def metadata(self, index: int = None) -> Dict[str, Any]:
        """Read ndimage metadata.
        Parameters
        ----------
        index : {integer, None}
            If the ImageResource contains multiple ndimages, and index is an
            integer, select the index-th ndimage from among them and return its
            metadata. If index is an ellipsis (...), read and return global
            metadata. If index is None, this plugin reads metadata from the
            first image of the file (index=0) unless the image is a GIF or APNG,
            in which case global metadata is read (index=...).
        exclude_applied : bool
            If True, exclude metadata fields that are applied to the image while
            reading. For example, if the binary data contains a rotation flag,
            the image is rotated by default and the rotation flag is excluded
            from the metadata to avoid confusion.
        Returns
        -------
        metadata : dict
            A dictionary of format-specific metadata.
        """

        header = self._image.header
        metadata = {name: header[name] for name in header.dtype.names}

        return metadata

    def properties(self, index: int = None) -> ImageProperties:
        """
        Standardized ndimage metadata.

        Parameters
        ----------
        index : int
            If the ImageResource contains multiple ndimages, and index is an
            integer, select the index-th ndimage from among them and return its
            properties. If index is an ellipsis (...), read and return the
            properties of all ndimages in the file stacked along a new batch
            dimension. If index is None, this plugin reads and returns the
            properties of the first image (index=0) unless the image is a GIF or
            APNG, in which case it reads and returns the properties all images
            (index=...).

        Returns
        -------
        properties : ImageProperties
            A dataclass filled with standardized image metadata.

        Notes
        -----
        This does not decode pixel data and is 394fast for large images.
        """
        is_batch = False

        height = int(self._image.header.ny)
        width = int(self._image.header.nx)

        shape = (height, width)

        nz: int = self._image.header.nz
        mz:  int = self._image.header.mz

        if nz > 1:
            depth = int(mz)
            nimg = int(nz/mz)
            if depth > 1:
                if nimg > 1:  # Stack of volumes
                    shape = (nimg, depth, *shape)
                    is_batch = True
                else:  # Single volume
                    shape = (depth, *shape)
            else:  # Stack of images
                shape = (nimg, *shape)
                is_batch = True

        return ImageProperties(
            shape=shape,
            n_images=nimg,
            dtype=self._image.data.dtype,
            is_batch=is_batch,
        )
