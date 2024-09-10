"""PYFFTW backend and wrapper module."""

from artis_tomo.math.fft._backend import _ScipyBackendCXTBase
from scipy.fft import (hfft2, ihfft2, hfftn, ihfftn,
                       fftshift, ifftshift, fftfreq, rfftfreq,
                       get_workers, set_workers)

import pyfftw as _pyfftw

__all__ = ['fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
           'hfft', 'ihfft', 'hfft2', 'ihfft2', 'hfftn', 'ihfftn',
           'dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn',
           'fftshift', 'ifftshift', 'fftfreq', 'rfftfreq', 'get_workers',
           'set_workers', 'next_fast_len']


_implemented = {}


def _wrapper(method, args, kw):
    """
    Wrap fft function calls to move 'workers' parameter to 'threads'.

    Original pyfftw backend for scipy is far slower. It seems not to keep
    builders long enough.
    """
    # print(f'pyfftw._wrapper: {method}')

    array = args[0]

    if method not in _implemented or \
            array.shape != _implemented[method].input_array.shape:

        if 'workers' in kw:
            kw['threads'] = kw.pop('workers')

        if array.flags.writeable:
            argsb = args

        else:
            # We create a copy of the input array to be set as internal
            # builder._input_array to be reused for following input arrays as
            # they'll be copied internally in the FFTW class.
            # If builder is created with a nonwriteable array and used with
            # writeable arrays, it'll be less efficient.
            atmp = _pyfftw.empty_aligned(shape=array.shape, dtype=array.dtype)
            atmp[:] = array
            argsb = (atmp, ) + args[1:]

            # Array is set to None as it's already been declared in the constructor
            # as writeable, so it's used internally for the first time.
            array = None

        # print(f'pyfftw._wrapper: method {method} created')
        _implemented[method] = getattr(_pyfftw.builders, method)(*argsb, **kw)
        return _implemented[method]()

    # In case nonwriteable, we reuse the previously created image for input
    if not array.flags.writeable:
         _implemented[method].input_array[:] = array
         array = None

    return _implemented[method](array)


def __getattr__(name):
    """Wrap fft functions called directly from this interface."""
    # print(f'pyfftw getattrib: {name}')

    def fun(*args, **kw):
        # print(f' pyfftw._wrapper: calling {name}')
        return _wrapper(name, args, kw)

    return fun


def clear():
    _implemented.clear()


class _Backend(_ScipyBackendCXTBase):
    """Custom FFT backend for cupy."""

    @staticmethod
    def __ua_function__(method, args, kw):
        """Wrap fft functions called by scipy backend context manager."""
        # print(f' pyfftw._Backend.__ua_function__: calling {method.__name__}')

        return _wrapper(method.__name__, args, kw)

    @classmethod
    def __exit__(cls, type, value, traceback):
        """Once the context finishes, clear builders."""
        # print('pyfftw._Backend.__exit__')
        _implemented.clear()
