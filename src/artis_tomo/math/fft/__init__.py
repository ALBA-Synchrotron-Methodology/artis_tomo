"""
Fast Fourier Transform Module.

By default, it is scipy.fft module, which is faster for a single FFT operation
than numpy.fft.

3rd FFT backends can also be used. If available, cupy and torch backends ared
added to be used through the scipy backend context manager.

"""

import numpy as np
from scipy.fft import *
import scipy.fft._backend as _sp_backend
from artis_tomo.math.fft import pyfftw
from artis_tomo.math.framework import frameworks

_typed_backends = {}
_named_backends = {}


if 'cupy' in frameworks.frameworks:
    from . import cufft
    _typed_backends[frameworks.frameworks['cupy'].
                    _interface._arrayClass] = cufft._Backend
    _named_backends['cupy'] = cufft._Backend

if 'torch' in frameworks.frameworks:
    from . import torchfft
    _typed_backends[frameworks.frameworks['torch'].
                    _interface._arrayClass] = torchfft._Backend
    _named_backends['torch'] = torchfft._Backend

_typed_backends[np.ndarray] = pyfftw._Backend
_named_backends['numpy'] = pyfftw._Backend


# def set_backend(backend, threads=-1):
#     pyfftw.config.NUM_THREADS = threads
#     return backend(_sp_backend.set_backend(backend, only=True))


def set_backend_from_array(array, threads=-1):
    """
    Context manager to set the backend within a fixed scope from array type.

    Upon entering the ``with`` statement, the backend associated to the array
    class will be selected. Upon exit, the backend is reset to scipy.

    Parameters
    ----------
    array : object
        Array object to be used as input for FFT functions.
        Can be a numpy array or a cupy array/torch tensor in case those modules
        are also available.
    threads : int, optional
        Threads number for parallel processing in CPU devices. If -1 all
        threads are used. The default is -1.

    Examples
    --------
    >>> import artis.math.framework as fw
    >>> from artis.math import fft
    >>> xp = fw.frame()
    >>> A = xp.ones((16, 16), device='cuda')
    >>> with fft.set_backend_from_array(A):
    >>>     AT = fft.fft(A)
    """
    pyfftw.config.NUM_THREADS = threads

    backend = _typed_backends.get(type(array), None)
    if backend is None:
        raise TypeError(f'Not supported array type {type(array)} to set backend')

    return backend(_sp_backend.set_backend(backend, only=True))


def set_framework_backend(framename, threads=-1):
    """
    Context manager to set the backend within a fixed scope from framename.

    Upon entering the ``with`` statement, the backend associated to the array
    class will be selected. Upon exit, the backend is reset to scipy.

    Parameters
    ----------
    framename : string
        Name of the frame to be used for FFT calculations. 'numpy' is expected
        to be always present while 'cupy' and 'torch' are optionally available
        if modules are installed.
    threads : int, optional
        Threads number for parallel processing in CPU devices. If -1 all
        threads are used. The default is -1.

    Examples
    --------
    >>> import artis.math.framework as fw
    >>> from artis.math import fft
    >>> xp = fw.frame()
    >>> A = xp.ones((16, 16), device='cuda', framename='cupy')
    >>> with fft.set_framework_backend('cupy'):
    >>>     AT = fft.fft(A)
    """
    pyfftw.config.NUM_THREADS = threads
    backend = _named_backends[framename]
    return backend(_sp_backend.set_backend(backend, only=True))
