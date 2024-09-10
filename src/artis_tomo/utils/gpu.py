"""
Numba Cuda related stuff.

@author: joton
"""
from numba.core import config
config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = True
from numba import cuda

import numpy as _np
from contextvars import ContextVar
from artis_tomo.math import framework



class _metagpu(type):
    @property
    def doGPU(cls):
        return cls._doGPU.get()


class gpu(metaclass=_metagpu):
    """Check available GPU cards and info."""

    def _getDeviceAvail():
        """Check if Cuda GPU cards devices are available."""
        try:
            lst = cuda.gpus.lst
            # _default = cuda.get_current_device()
        except Exception:
            lst = []
        return len(lst) > 0

    _devAvail = _getDeviceAvail()
    _frameAvail = hasattr(framework, 'gpu')

    avail = _np.logical_and(_devAvail, _frameAvail)

    usage = False

    _doGPU = ContextVar('doGPU', default=usage)

    @classmethod
    def setUsage(cls, usage):
        """Activate GPU usage if available."""

        if usage and not cls.avail:
            msg = "GPU cannot be used. "
            n = 0
            if not cls._devAvail:
                msg += "Device "
                n += 1
                if not cls._frameAvail:
                    msg = 'and '
            if not cls._frameAvail:
                msg += "Math framework "
                n += 1
            if n > 1:
                msg += 'have '
            else:
                msg += 'has '

            msg += 'not been detected.'

            raise Exception(msg)

        cls.usage = usage
        cls._doGPU.set(usage)


    @classmethod
    def default(cls):
        """Print default GPU card info."""
        _default = cuda.get_current_device()

        if cls.avail:
            print("name = %s" % _default.name)
            print("maxThreadsPerBlock = %s" % str(_default.MAX_THREADS_PER_BLOCK))
            print("maxBlockDimX = %s" % str(_default.MAX_BLOCK_DIM_X))
            print("maxBlockDimY = %s" % str(_default.MAX_BLOCK_DIM_Y))
            print("maxBlockDimZ = %s" % str(_default.MAX_BLOCK_DIM_Z))
            print("maxGridDimX = %s" % str(_default.MAX_GRID_DIM_X))
            print("maxGridDimY = %s" % str(_default.MAX_GRID_DIM_Y))
            print("maxGridDimZ = %s" % str(_default.MAX_GRID_DIM_Z))
            print("maxSharedMemoryPerBlock = %s" % str(_default.MAX_SHARED_MEMORY_PER_BLOCK))
            print("asyncEngineCount = %s" % str(_default.ASYNC_ENGINE_COUNT))
            print("canMapHostMemory = %s" % str(_default.CAN_MAP_HOST_MEMORY))
            print("multiProcessorCount = %s" % str(_default.MULTIPROCESSOR_COUNT))
            print("warpSize = %s" % str(_default.WARP_SIZE))
            print("unifiedAddressing = %s" % str(_default.UNIFIED_ADDRESSING))
            print("pciBusID = %s" % str(_default.PCI_BUS_ID))
            print("pciDeviceID = %s" % str(_default.PCI_DEVICE_ID))
        else:
            print('No GPU card available.')

    def getThrBlk(ndim):
        """Return threads blok per dimension automatically."""
        _default = cuda.get_current_device()

        maxThr = int(_np.log2(_default.MAX_THREADS_PER_BLOCK))

        if ndim == 1:
            return 2**maxThr
        elif ndim == 2:
            return tuple(_np.ones(2, int)*2**(maxThr//2))
        elif ndim == 3:
            nx = int(maxThr*0.4)
            nz = int(maxThr - 2*nx)
            return tuple(2**_np.array([nx, nx, nz]))










