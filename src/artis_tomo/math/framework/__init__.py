#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 20:11:47 2023

@author: joton
"""

from ._frameworks import frameworks

frameworks.add_framework(['._numpy',
                          '._cupy',
                          '._torch'],
                          'artis_tomo.math.framework')

from artis_tomo.math.framework._frame import frame
from artis_tomo.math.framework._array import Array

# cpu = DirectDevice('cpu')
# if 'cuda:0' in frame.deviceFrames:
#     gpu = DirectDevice('cuda:0')


