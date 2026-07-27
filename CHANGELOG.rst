Changelog
=========

..
    Template
    0.3.0 (2024-08-28)
    -------------------

    **Added**
    - Introduced a new `feature_x` module for advanced data analysis.
    - Added support for Python 3.11.
    - Integrated logging capabilities into the `data_processor` module.

    **Changed**
    - Refactored the `config_loader` to improve performance.
    - Updated dependencies: `numpy` to 1.25.0, `pandas` to 2.1.0.

    **Fixed**
    - Resolved issue #42 where the `data_parser` would crash on empty input.
    - Corrected typos in the documentation.

    **Deprecated**
    - Marked the `legacy_function` in `utils.py` as deprecated; it will be removed in version 0.4.0.

1.1.0 (2026-07-27)
------------------

**Added**

- ``correlative``: the multi-modal correlative pipeline, previously spread
  through the ``clfluoxr`` programs. ``preprocessing`` prepares each modality
  for correlation against the others (``readFluoVolume``, ``readFluoImage``,
  ``preprocessFluoImage``, ``preprocessTomoProjection``, ``preprocessMosaic``,
  ``normalizeSXTMosaic``); ``alignment`` holds the three alignment stages and
  ``applyTransformToVolumes``; ``transforms`` reads and writes the resulting
  transformations, so a re-run can skip the stages that only depend on the
  mosaic and the fluorescence data.
- ``tomo.project.projectVolume``, wrapping ``projectRS`` with the GPU device
  handling around it.
- ``tomo.utils.selectTiltSubset`` and ``tomo.utils.loadTomoFromInput``.
- ``ars_corr_fluo_3d``: ``--feat_preprocess``, ``--winsorize_q`` and
  ``--feat_filter`` select how features are isolated before each cross-modality
  correlation, and ``--tmat`` reuses transformations from a previous run to skip
  steps 1 and 2. ``ars_corr_fluo_2d`` gains the same feature-filter options.

**Changed**

- Image, math, io, tools and utils now come from the new ``artis_sci`` package
  instead of being carried here, so they can be shared with other domain
  packages without depending on tomography. ``install_requires`` is reduced to
  ``artis_sci >= 1.1``, which brings the previous dependencies transitively.
- ``ars_corr_fluo_3d`` feature-filters both modalities before every
  cross-modality correlation rather than only smoothing the fluorescence, which
  is what lets features of a chosen size drive the match between two signals
  with no intensity relationship.
- ``programs`` modules now hold the command-line definition and nothing else;
  the processing they used to carry lives in ``correlative`` and ``tomo``.

**Removed**

- The ``artis_tomo.image``, ``artis_tomo.math``, ``artis_tomo.io``,
  ``artis_tomo.tools`` and ``artis_tomo.utils`` submodules. Code importing them
  should import from ``artis_sci`` instead, for example
  ``from artis_sci.image import filter``. The functions themselves are
  unchanged.

1.0.2 (2026-05-20)
------------------

**Fixed**

- Crash on import when cupy is installed but the CUDA driver is missing or
  incompatible. The device list is now built defensively and an unusable
  backend is skipped instead of taking down the framework registry.

1.0.1 (2025-05-13)
------------------

**Added**

- Module program `tilt_align` for external usage.

1.0.0 (2024-09-10)
------------------

- Initial release.

