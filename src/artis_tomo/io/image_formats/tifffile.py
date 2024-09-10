"""Read/Write TIFF files using tifffile.

Custom version of origian imageio.tifffile plugin.

"""

from imageio.core.request import Request
from imageio.plugins.tifffile_v3 import TifffilePlugin as _TifffilePlugin
import warnings


class TifffilePlugin(_TifffilePlugin):
    """Custom support for tifffile as backend.

    We fix initialization to remove other parameters that may be used by other
    file formats without interrupting the program.

    Parameters
    ----------
    request : iio.Request
        A request object that represents the users intent. It provides a
        standard interface for a plugin to access the various ImageResources.
        Check the docs for details.
    kwargs : Any
        Additional kwargs are forwarded to tifffile's constructor, i.e.
        to ``TiffFile`` for reading or ``TiffWriter`` for writing.

    """

    def __init__(self, request: Request, **kwargs) -> None:

        if 'mmap' in kwargs:
            warnings.warn("Tifffile reader does not support mmap reading. "
                          "Ignoring it.")

            del kwargs['mmap']

        super().__init__(request, **kwargs)
