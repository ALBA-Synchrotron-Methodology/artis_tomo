"""CUFFT backend and wrapper module."""

from artis_tomo.math.fft._backend import _ScipyBackendCXTBase
import cupyx.scipy.fft as _cufft


class _Backend(_ScipyBackendCXTBase):
    """Custom FFT backend for cupy."""

    @staticmethod
    def __ua_function__(method, args, kw):
        """Wrap fft function calls to remove 'workers' parameter."""
        # print(f' cufftw._Backend.__ua_function__: calling {method.__name__}')

        if 'workers' in kw:
            del(kw['workers'])
        return getattr(_cufft, method.__name__)(*args, **kw)


from cupyx.scipy.fft import *
