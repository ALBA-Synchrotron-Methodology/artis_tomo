Artis Tomo
==========

Computational methods for tomography and microscopy.

Description
-----------

**Artis Tomo** is a collection of algorithms, methodologies and programs to
process and analyse imaging and tomographic data from X-ray and electron
microscopies, among others.

Source code
-----------

**Artis Tomo** is open source and is available at GitHub_.

Installation
------------

It is available at PyPi_ and can be installed with pip as:

.. code-block:: bash

    $ pip install artis-tomo


GPU acceleration
----------------

**Artis Tomo** is also compatible with GPU acceleration. In case you use a
machine with an Nvidia GPU and CUDA capability, you can make it available by
creating a conda environment:

.. code-block:: bash

    conda create -n artis-tomo python=3.10
    conda activate artis-tomo
    conda install -c conda-forge "numpy=1.26" cudatoolkit=11.8 cupy=13
    conda install rapidsai::cubinlinker conda-forge::ptxcompiler
    pip install "numpy==1.26.*" artis-tomo

Or a micromamba environment:

.. code-block:: bash

    micromamba create -n artis-tomo python=3.10
    micromamba activate artis-tomo
    micromamba install -c conda-forge "numpy=1.26" cudatoolkit=11.8 cupy=13
    pip install ptxcompiler-cu11 cubinlinker-cu11 --extra-index-url=https://pypi.nvidia.com
    pip install "numpy==1.26.*" artis-tomo

numpy is pinned in **both** steps on purpose. Pinning only the conda step lets
the later ``pip install`` pull a numpy 2.x wheel over it, and pinning only pip
leaves conda free to move numpy on the next ``install``/``update``. See
`Version compatibility`_ for why 1.26 is the ceiling.


Version compatibility
~~~~~~~~~~~~~~~~~~~~~

Two independent GPU paths are used: **cupy**, for array operations on the
framework backend, and **numba.cuda**, for the compiled kernels. They locate
CUDA libraries differently, so a setup where one works and the other does not
is entirely possible — check both.

This combination has been verified end to end on an NVIDIA L40
(compute capability 8.9):

=============  ===========  ===================================================
Component      Version      Notes
=============  ===========  ===================================================
Python         3.10         3.9-3.11 expected to work
numpy          1.26         **Must be < 1.27**, see below
numba          0.59.1       Requires ``llvmlite >=0.42,<0.43``
scipy          1.11.4       Predates numpy 2 support; keep with numpy 1.26
cupy           13.6         Accepts numpy >=1.22,<2.6
cudatoolkit    11.8         Provides ``libnvvm.so.4``, needed by numba
NVIDIA driver  575.57.08    Exposes CUDA 12.9; newer than the toolkit is fine
=============  ===========  ===================================================

**Why numpy is capped.** ``artis_sci`` pins ``numpy<1.27`` because the code has
not been validated against numpy 2, and that cap reaches here through the
dependency. numba 0.59 also declared ``numpy>=1.22,<1.27``, but numba 0.60 and
later accept numpy 2, so the cap is now the binding constraint rather than
numba. Lifting it means validating numba, scipy and numpy 2 together, then
raising the cap and the scipy floor (1.13 or newer) in one go.

**Do not let conda and pip both manage numpy.** If pip downgrades numpy to
satisfy numba, the conda metadata still records the version it installed, so
``conda list``/``micromamba list`` and the actual runtime disagree, and a later
``install``/``update`` can silently restore the incompatible version. Check
what is really imported with:

.. code-block:: bash

    python -c "import numpy, numba; print(numpy.__version__, numba.__version__)"

**The driver may be newer than the toolkit.** A CUDA 12.x driver runs a CUDA
11.8 build fine, so there is no need to match them. ``ptxcompiler-cu11`` and
``cubinlinker-cu11`` provide the minor-version compatibility numba needs to
target recent architectures from a CUDA 11 toolkit.

Verify both GPU paths after installing:

.. code-block:: bash

    python -c "import cupy; print('cupy devices:', cupy.cuda.runtime.getDeviceCount())"
    python -c "from numba import cuda; print('numba cuda:', cuda.is_available()); cuda.detect()"


Troubleshooting: CUDA_ERROR_STUB_LIBRARY
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``numba.cuda.is_available()`` returns ``False`` and ``cuda.detect()`` fails
with ``CUDA_ERROR_STUB_LIBRARY (34)`` at ``cuInit``, while cupy keeps working,
then a CUDA *stub* library is shadowing the real driver on ``LD_LIBRARY_PATH``.
This is a machine configuration problem rather than an **Artis Tomo** one.

Quick check:

.. code-block:: bash

    echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep stubs

Any match is the cause. ``docs/cuda_stub_library_issue.rst`` explains the
problem, why it hits numba but not cupy, and how to fix it both system-wide and
per environment; it is written to be handed to whoever administers the machine.


Programs
--------

.. code-block:: bash

    ars_corr_fluo_2d
    ars_corr_fluo_3d
    ars_tomo_import_imod


Usage
-----

.. code-block:: bash

    ars_corr_fluo_3d -i imod_folder -m xray_mosaic.tiff -f 3DSIM_BR_FL_SIR.dv --xrpix 10
    --mpix 20 --flpix 60 --flzpix 120 --channel 1 --range 600 1200 -o corr3d/ --gpu 0 --nproc 8 --tomofn xray_nice_sirt_rec.mrc


License
-------

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.


.. _GitLab: https://gitlab.com/scimet/artis_tomo
.. _GitHub: https://github.com/ALBA-Synchrotron-Methodology/artis_tomo

.. _PyPi: https://pypi.org/project/artis-tomo
