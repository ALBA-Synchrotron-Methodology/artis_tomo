
import cupy as _cp
from ._interface import _Wrapper
from ...utils.functools import rgetattr
from artis_tomo.math.framework._array import Array

""" Cupy framework interface"""


# _DEVICE_CONSTRUCTOR = {
#     # standard ones
#     _cp.empty,
#     _cp.ones,
#     _cp.arange,
#     _cp.eye,
#     _cp.fft.fftfreq,
#     _cp.fft.rfftfreq,
#     _cp.full,
#     _cp.fill_diagonal,
#     _cp.linspace,
#     _cp.logspace,
#     _cp.ones,
#     _cp.random,
#     _cp.zeros,
#     _cp.asarray
# }


class _interface(_Wrapper):

    _framename = 'cupy'
    _frame = _cp
    _arrayClass = _cp.ndarray
    _devices = [f'cuda:{n}' for n in range(_cp.cuda.runtime.getDeviceCount())]

    @classmethod
    def get_device(cls, array):
        return f'cuda:{array.device.id}'

    @classmethod
    def to_device(cls, array, device, iniFrame=None, dtype=None):

        if 'cuda' in device:
            if len(device) < 5:
                dev = 0
            else:
                dev = int(device[5:])
        else:
            raise Exception(f'Device {device} not available. '
                            'Expected one of {cls._devices}')

        with _cp.cuda.Device(dev):
            return _cp.asarray(array, dtype)

    @staticmethod
    def _func(attribList, device, *args, **kwargs):

        newargs = ()

        # This ArtisArray extraction to cupy is when cupy is called from Frame,
        # not from np._array_ufuncs_ . KINO: I think this should go in Frame
        for arg in args:
            if isinstance(arg, Array):
                newargs = newargs + (arg.array, )
            else:
                newargs = newargs + (arg, )
        for key, obj in kwargs.items():
            if isinstance(obj, Array):
                kwargs[key] = obj.array

        with _cp.cuda.Device(device[-1]):
            # if attribList[0] == 'fft':
                # print('fft caca')
                # return args[0]*2
            return rgetattr(_cp, attribList)(*newargs, **kwargs)
