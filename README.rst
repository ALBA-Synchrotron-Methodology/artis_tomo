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

    conda create -n artis-tomo
    conda activate artis-tomo
    conda install -c conda-forge cudatoolkit cupy
    conda install rapidsai::cubinlinker conda-forge::ptxcompiler
    pip install artis-tomo

Or a micromamba environment:

.. code-block:: bash

    micromamba create -n artis-tomo python=3.9
    micromamba activate artis-tomo
    micromamba install -c conda-forge cudatoolkit cupy
    pip install ptxcompiler-cu11 cubinlinker-cu11 --extra-index-url=https://pypi.nvidia.com
    pip install artis-tomo


When installing cupy or cudatoolkit, try the most suitable versions
for your hardware.


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
