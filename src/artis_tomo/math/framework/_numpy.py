"""Numpy framework interface."""

import numpy as _np
from ._interface import _Wrapper
from ...utils.functools import rgetattr


class _interface(_Wrapper):

    _framename = 'numpy'
    _frame = _np
    _arrayClass = _np.ndarray
    _devices = ['cpu']

    @classmethod
    def get_device(cls, array):
        return cls.devices[0]

    @classmethod
    def to_device(cls, array, device, iniFrame, dtype=None):
        """
        Convert any framework array to numpy array.

        Parameters
        ----------
        array : ndarray
            Input array related to any framework.
        device : String
            Target device for this framework. In this case, numpy, only accepts
            'cpu' devices.
        iniFrame : framelib instance
            Framework of the input array. To be considered for conversion.
        dtype : TYPE, optional
            Dtype for output array. The default is None.

        Returns
        -------
        ndarray
            Copy or view of input array, depending on iniFrame.

        """
        if device not in cls._devices:
            raise Exception(f' Device {device} not supported by numpy.')

        if iniFrame._interface.framename == 'cupy':
            return iniFrame._interface.frame.asnumpy(array)
        if iniFrame._interface.framename == 'torch':
            return _np.asarray(array.cpu(), dtype)
        else:
            return _np.asarray(array, dtype)

    @staticmethod
    def _func(attribList, device, *args, **kwargs):

        return rgetattr(_np, attribList)(*args, **kwargs)
