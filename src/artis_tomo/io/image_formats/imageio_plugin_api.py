#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 19:43:57 2023

@author: joton
"""

from imageio.core.v3_plugin_api import ImageProperties as _ImageProperties
from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageProperties(_ImageProperties):
    """Standardized Metadata

    ImageProperties represent a set of standardized metadata that is available
    under the same name for every supported format. If the ImageResource (or
    format) does not specify the value, a sensible default value is chosen
    instead.

    Attributes
    ----------
    shape : Tuple[int, ...]
        The shape of the loaded ndimage.
    dtype : np.dtype
        The dtype of the loaded ndimage.
    n_images : int
        Number of images in the file if ``index=...``, `None` for single images.
    is_batch : bool
        If True, the first dimension of the ndimage represents a batch dimension
        along which several images are stacked.
    spacing : Tuple
        A tuple describing the spacing between pixels along each axis of the
        ndimage. If the spacing is uniform along an axis the value corresponding
        to that axis is a single float. If the spacing is non-uniform, the value
        corresponding to that axis is a tuple in which the i-th element
        indicates the spacing between the i-th and (i+1)-th pixel along that
        axis.

    """

    n_channels: Optional[int] = None
    n_times: Optional[int] = None
