#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlative
===========

Multi-modal correlative microscopy: registering fluorescence data against soft
X-ray tomography.

The code here is modality-aware, assuming knowledge of the physical origin of
each signal, which is what distinguishes it from the generic image processing in
``artis_sci``.

- :mod:`~artis_tomo.correlative.preprocessing` prepares each modality for
  correlation against the others.
- :mod:`~artis_tomo.correlative.alignment` holds the three alignment stages and
  applies the result.
- :mod:`~artis_tomo.correlative.transforms` reads and writes the
  transformations, so the expensive stages can be skipped on a re-run.
"""

from . import preprocessing
from . import alignment
from . import transforms
