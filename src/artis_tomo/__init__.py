# -*- coding: utf-8 -*-
# Copyright (c) 2017 Joaquín Otón
# This software is distributed under a MIT License. See LICENSE.txt.

"""
artis_tomo
=======

"""
from numba.core import config
config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = True

from .version import __version__
