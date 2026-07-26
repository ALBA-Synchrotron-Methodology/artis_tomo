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

