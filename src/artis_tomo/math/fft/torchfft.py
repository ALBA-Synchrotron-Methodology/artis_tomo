"""TORCH FFt backend and wrapper."""

from artis_tomo.math.fft._backend import _ScipyBackendCXTBase
import torch.fft as _tcfft


class _Backend(_ScipyBackendCXTBase):
    """Custom FFT backend for torch."""

    @staticmethod
    def __ua_function__(method, args, kw):
        """Wrap fft function calls to remove 'workers' parameter."""
        # print(f' torchfft._Backend.__ua_function__: calling {method.__name__}')

        if 'workers' in kw:
            del(kw['workers'])
        return getattr(_tcfft, method.__name__)(*args, **kw)


from torch.fft import *
