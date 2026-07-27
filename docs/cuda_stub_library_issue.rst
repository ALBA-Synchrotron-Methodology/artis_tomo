CUDA stub libraries on LD_LIBRARY_PATH break GPU applications
=============================================================

:Audience: system administrators of GPU compute nodes
:Scope: any node where a CUDA toolkit ``stubs`` directory is on the default
        ``LD_LIBRARY_PATH``
:Impact: numba, PyTorch, TensorFlow, JAX and anything else that opens the CUDA
         driver by its unversioned name cannot use the GPU at all

This document describes a machine configuration problem, not an application
bug. It was diagnosed while running **Artis Tomo**, but it affects every CUDA
application on the node, so it is written to stand on its own.


Summary
-------

The CUDA toolkit ships *stub* libraries for linking. Their ``libcuda.so`` is a
placeholder: it exports the driver API so a compiler can resolve symbols at
build time, but every entry point fails at run time by design. It must never be
on the runtime library search path.

On an affected node a system profile script appends the stubs directory to
``LD_LIBRARY_PATH`` for every login shell. The dynamic loader then hands out the
stub instead of the real driver, and GPU initialisation fails.

**The fix is to remove the stubs directory from ``LD_LIBRARY_PATH``.** Nothing
needs it there; it is only useful to the compiler, which locates it on its own.


Symptom
-------

Applications that open the driver by its unversioned name fail at
initialisation. With numba:

.. code-block:: text

    >>> from numba import cuda
    >>> cuda.is_available()
    False
    >>> cuda.detect()
    CudaSupportError: Error at driver init:
    Call to cuInit results in CUDA_ERROR_STUB_LIBRARY (34)

``nvidia-smi`` works, the driver is loaded, and the GPU is idle and healthy, so
the node looks fine. Error 34 is CUDA's way of saying "you called into a stub
library", which is a strong and specific signal: it is not a driver fault, a
permissions problem, or a version mismatch.


Why some libraries fail and others do not
-----------------------------------------

This is the confusing part, and the reason the problem is easy to misattribute
to the application.

The real driver is installed by the NVIDIA driver package and is normally
present under three names:

.. code-block:: text

    /usr/lib64/libcuda.so.<driver version>    the actual library, ~90 MB
    /usr/lib64/libcuda.so.1                   SONAME symlink
    /usr/lib64/libcuda.so                     development symlink

The stubs directory contains **only** the unversioned name:

.. code-block:: text

    <cuda>/targets/<arch>/lib/stubs/libcuda.so    placeholder, tens of kB

So the outcome depends purely on which name a library asks for:

- Asks for ``libcuda.so`` — matches the stub, which appears earlier on
  ``LD_LIBRARY_PATH`` than ``/usr/lib64``. **Fails.** numba does this.
- Asks for ``libcuda.so.1`` — no such file in the stubs directory, so the search
  continues and finds the real driver. **Works.** cupy does this.

The practical consequence: on an affected node cupy reports the GPU correctly
and runs kernels, while numba insists there is no usable CUDA device. Two GPU
libraries in the same environment disagree about whether the machine has a GPU.
That looks like an application or packaging fault and is normally investigated
as one, which is why this is worth documenting explicitly.

A stub is easy to tell from the real thing by size alone:

.. code-block:: bash

    ls -l /usr/lib64/libcuda.so.1                              # ~90 MB
    ls -l /usr/local/cuda/targets/*/lib/stubs/libcuda.so       # tens of kB


Diagnosis
---------

Check whether any stubs directory is on the search path:

.. code-block:: bash

    echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep stubs

Any output confirms the problem. To find which file sets it:

.. code-block:: bash

    grep -rn stubs /etc/profile.d/ /etc/ld.so.conf.d/ /etc/environment 2>/dev/null

Also worth checking module files (``/usr/share/modulefiles``, ``/etc/lmod``) and
any site scripts sourced at login.

Confirm which library a process actually resolves:

.. code-block:: bash

    python -c "import ctypes; ctypes.CDLL('libcuda.so'); print('loaded')"
    # then, to see the path that was used:
    LD_DEBUG=libs python -c "import ctypes; ctypes.CDLL('libcuda.so')" 2>&1 | grep -i libcuda


Resolution
----------

**System-wide, preferred.** Remove the stubs component from whatever sets it.
Typically a line such as:

.. code-block:: bash

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/targets/x86_64-linux/lib/stubs

becomes:

.. code-block:: bash

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/targets/x86_64-linux/lib

Keep the ``lib`` directory if applications rely on the toolkit runtime
libraries; only the ``stubs`` component is harmful. No CUDA application needs
stubs at run time, and anything that links against them at build time finds them
through the toolkit layout or an explicit ``-L`` flag, not through
``LD_LIBRARY_PATH``.

The same applies to ``/etc/ld.so.conf.d/``: a stubs directory must never appear
there. If one does, remove it and run ``ldconfig``.

**Per environment, as an interim measure.** Where the system file cannot be
changed immediately, users can strip the entry on environment activation. For
conda/mamba environments, ``$CONDA_PREFIX/etc/conda/activate.d/10-drop-cuda-stubs.sh``:

.. code-block:: bash

    #!/bin/sh
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        LD_LIBRARY_PATH=$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' \
            | grep -v '/stubs$' | grep -v '^$' | paste -sd:)
        export LD_LIBRARY_PATH
    fi

A narrower option is to point numba directly at the real driver with
``export NUMBA_CUDA_DRIVER=/usr/lib64/libcuda.so.1``. That fixes numba only and
hardcodes a path, so prefer removing the stubs directory, which fixes every
affected library at once.


Verification
------------

After the change, in a fresh login shell:

.. code-block:: bash

    echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep stubs   # expect no output

    python -c "from numba import cuda; print(cuda.is_available()); cuda.detect()"
    python -c "import cupy; print('cupy devices:', cupy.cuda.runtime.getDeviceCount())"

``cuda.is_available()`` should be ``True`` and ``cuda.detect()`` should list the
devices as ``[SUPPORTED]``. A launched kernel is the conclusive check:

.. code-block:: bash

    python - <<'EOF'
    import numpy as np
    from numba import cuda

    @cuda.jit
    def add_one(a):
        i = cuda.grid(1)
        if i < a.size:
            a[i] += 1.0

    d = cuda.to_device(np.zeros(1024))
    add_one[8, 128](d)
    assert (d.copy_to_host() == 1.0).all()
    print("kernel launch OK")
    EOF


Observed instance
-----------------

Recorded for reference; adjust paths to the node in question.

===================  =========================================================
Item                 Value
===================  =========================================================
Node role            GPU compute node, NVIDIA L40 (compute capability 8.9)
Driver               575.57.08, exposing CUDA 12.9
Toolkit              CUDA 12.9 in ``/usr/local/cuda``
Offending file       ``/etc/profile.d/10-gpfspath.sh``, line 8
Offending component  ``/usr/local/cuda/targets/x86_64-linux/lib/stubs``
Stub size            70,368 bytes
Real driver          ``/usr/lib64/libcuda.so.575.57.08``, 92,316,728 bytes
Symptom              numba ``CUDA_ERROR_STUB_LIBRARY (34)``; cupy unaffected
===================  =========================================================

The line reads:

.. code-block:: bash

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/targets/x86_64-linux/lib/stubs

Removing the trailing ``:/usr/local/cuda/targets/x86_64-linux/lib/stubs``
resolves it. Verified on this node: with the component removed, numba reports
the L40 as supported and kernels launch and return correct results, with no
other configuration change.
