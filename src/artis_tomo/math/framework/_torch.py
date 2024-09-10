
import torch as _tr
from contextlib import contextmanager
from ._interface import _Wrapper
from ...utils.functools import rgetattr
from artis_tomo.math.framework._array import Array
from artis_tomo.math.framework import frameworks


### Check if context manager issue in pytoch has finally been included :
### https://github.com/pytorch/pytorch/issues/82296
_DEVICE_CONSTRUCTOR = {
    # standard ones
    _tr.empty,
    _tr.empty_strided,
    _tr.empty_quantized,
    _tr.ones,
    _tr.arange,
    _tr.bartlett_window,
    _tr.blackman_window,
    _tr.eye,
    _tr.fft.fftfreq,
    _tr.fft.rfftfreq,
    _tr.full,
    _tr.fill,
    _tr.hamming_window,
    _tr.hann_window,
    _tr.kaiser_window,
    _tr.linspace,
    _tr.logspace,
    # _tr.nested_tensor,
    # _tr.normal,
    _tr.ones,
    _tr.rand,
    _tr.randn,
    _tr.randint,
    _tr.randperm,
    _tr.range,
    _tr.sparse_coo_tensor,
    _tr.sparse_compressed_tensor,
    _tr.sparse_csr_tensor,
    _tr.sparse_csc_tensor,
    _tr.sparse_bsr_tensor,
    _tr.sparse_bsc_tensor,
    _tr.tril_indices,
    _tr.triu_indices,
    _tr.vander,
    _tr.zeros,
    _tr.asarray,
    # weird ones
    _tr.tensor,
    _tr.as_tensor,
}

## Wrapping methods of Numpy API for Tensor class

# def implements(numpy_function):
#     """Register an __array_function__ implementation for MyArray objects."""
#     def decorator(func):
#         HANDLED_FUNCTIONS[numpy_function] = func
#         return func
#     return decorator


# NUMPY API IN TORCH FRAMEWORK: methods
_tr.array = _tr.asarray

_NUMPY_API_TENSOR_WRAPPER = {
    'power': [_tr.pow, 0],  # Called from framework
    'iscomplex': [_tr.is_complex, 0],
    'equal': ['__eq__', 1]  # Called from Tensor class
    }

def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

    fname = ufunc.__name__
    if hasattr(ufunc, '__module__'):
        module = ufunc.__module__.split('.')[1:]
    else:
        module = []
    fname = module + [fname]
    fname_str = '.'.join(fname)

    # print(f'inputs={inputs}')
    # print(f'kwargs={kwargs}')
    # print(f'_torch:__array_ufunc__: {fname}')

    if 'out' in kwargs:
        kwargs['out'] = kwargs['out'][0]

    fun, mode = _NUMPY_API_TENSOR_WRAPPER.get(fname_str, [None, None])
    if fun is None:
        fun = rgetattr(_tr, fname)
    elif mode == 1:
        fun = getattr(self, fun)
        inputs = inputs[1:]

    # print(f'inputs={inputs}')
    # print(f'kwargs={kwargs}')
    newinputs = ()
    for obj in inputs:
        # convert to own device (to assert comparison to non torch arrays)
        if not isinstance(obj, _tr.Tensor) and \
                type(obj) in frameworks._arrayToFrame:
            obj = _tr.asarray(obj, device=self.device.type)

        newinputs += (obj, )

    # print(f'newinputs={newinputs}')
    # print(f'kwargs={kwargs}')
    return fun(*newinputs, **kwargs)


def __array_function__(self, func, types, *args, **kwargs):
    # print(f'args={args}')
    # print(f'kwargs={kwargs}')

    args = args[0]
    return self.__array_ufunc__(func, None, *args, **kwargs)


setattr(_tr.Tensor, '__array_ufunc__', __array_ufunc__)
setattr(_tr.Tensor, '__array_function__', __array_function__)


class _DeviceMode(_tr.overrides.TorchFunctionMode):
    """Object to build a context manager calling push method."""

    def __init__(self, device):
        self.device = _tr.device(device)

    def __torch_function__(self, func, types, args=(), kwargs={}):

        if func in _DEVICE_CONSTRUCTOR and kwargs.get('device') is None:
            kwargs['device'] = self.device

        return func(*args, **kwargs)


class _interface(_Wrapper):

    _framename = 'torch'
    _frame = _tr
    _arrayClass = _tr.Tensor

    _devices = [f'cuda:{n}' for n in range(_tr.cuda.device_count())]
    _devices += ['cpu']

    @classmethod
    def get_device(cls, array):
        return array.device.type

    @classmethod
    def to_device(cls, array, device, iniFrame=None, dtype=None):
        if device not in cls._devices:
            raise Exception(f'Device {device} not available. '
                            'Expected one of {cls._devices}')
        return _tr.asarray(array, device=device, dtype=dtype)

    @contextmanager
    @classmethod
    def deviceMode(cls, device):
        """Set a context manager for a specific device.

        Select default device to work in a block. Torch does not include it.
        Note not all functions are acepted. Check torch.overrides.get_ignored_functions()
        to list ignored functions.
        """
        if device in cls._devices:
            with _DeviceMode.push(_tr.device(device)):
                yield

    @staticmethod
    def _func(attribList, device, *args, **kwargs):
        """Torch functions which constructs in a device are set by argument."""

        func = rgetattr(_tr, attribList)

        if func in _DEVICE_CONSTRUCTOR:
            if kwargs.get('device') is None:
                kwargs['device'] = device

        newargs = ()
        for arg in args:
            if isinstance(arg, Array):
                newargs = newargs + (arg.array, )
            else:
                newargs = newargs + (arg, )
        for key, obj in kwargs.items():
            if isinstance(obj, Array):
                kwargs[key] = obj.array

        return func(*args, **kwargs)

