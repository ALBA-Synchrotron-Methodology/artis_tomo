"""
Framework manager.

frame class allows to switch among several devices (cpu, cuda) using different
library frameworks as numpy, cupy, torch ...
"""

import importlib
from contextlib import contextmanager
from ...utils.functools import rgetattr
from ._frameworks import frameworks

# import numpy as np


class frame():
    """
    Create an instance of a framework manager.

    It can switch to any of the registered devices and available frameworks.

    It also allows for arrays conversion among devices and frameworks.

    Attributes
    ----------
    frameworks : dict
        Dictionary of registered frame libraries.
    deviceFrames : dict
        Dictionary with available frameworks for each registered device.
    _arrayToFrame : dict
        Dictionary with framelibs for each ndarray/tensor class.

    Methods
    -------
    add_framework(moduleNameList, package):
        Register modules that define a wrapper interface object.
    __init__(self, device='cpu', framename='default'):
        Initialize the framework manager to a specific device/framework.
    from_array(array):
        Initialize the framework manager from array device and framework.
    set_device(device, framename='default'):
        Select a specific device/framework.
    get_frameInfo(array=None, mode=0):
        Retrieve information about the underlying mathematical framework and
        the device associated with the array.
    _get_device(array=None):
        Retrieve the device and framelib associated with the given array.
    _get_deviceSuffix(device):
        Complete the device index if required.
    _get_deviceFramelib(device, framename='default'):
        Get the selected framework for the device.
    to_device(array, device=None, framename=None, dtype=None):
        Send the array to a new array in the device using the framework given
        by framename.
    fft_backend(threads=-1):
        Context manager for specific FFT backends.
    __getattr__(name):
        Catch the sequence of called attributes.
    __call__(*args, **kwargs):
        Redirect function calls to the selected framework.
    __dir__():
        Return a list of attribute names of the active framework.
    __str__():
        Print the active device/framework.
    __repr__():
        Print the active device/framework.
    """

    device = None
    framename = None
    framework = None

    def __init__(self, device='cpu', framename=None):
        """
        Initialize the framework manager to specific device/framework.

        Parameters
        ----------
        device : String, optional
            Active device. The default is 'cpu'.
        framename : String, optional
            Framework name to process with the device. Only frameworks
            supported by the device are allowed. If 'None', the first
            registered framework for this device is used.
            The default is None.

        """
        # print("__init__")
        self._attribList = []
        self.default = {}

        for dev, _framenames in frameworks.deviceFrames.items():
            self.default[dev] = _framenames[0]

        self.set_device(device, framename)

    @classmethod
    def from_array(cls, array):
        """
        Initialize the framework manager from array device and framework.

        Parameters
        ----------
        array : ArrayClass of any registered framework
            Array to obtain device/framework.

        Returns
        -------
        frame manager
            Framework manager activated in same array's device/framework.

        """
        framelib = frameworks._arrayToFrame[type(array)]
        device = framelib._interface.get_device(array)
        return cls(device, framelib._interface.framename)

    def set_device(self, device, framename=None):
        """
        Select specific device/framework.

        Parameters
        ----------
        device : String, optional
            Active device. The default is 'cpu'.
        framename : String, optional
            Framework name to process with the device. Only frameworks
            supported by the device are allowed. If 'default', the last
            selected framework for this same device is used.
            The default is 'default'.

        """
        self.device = frameworks._get_deviceSuffix(device)

        if framename is None:
            self.framename = self.default[self.device]
        else:
            self.framename = framename

        self.framework = self._get_deviceFramelib(self.device, self.framename)

    def get_frameInfo(self, array=None, mode=0):
        """
        Retrieves information about the underlying mathematical framework and
        the device associated with the array.

        Parameters:
        -----------
        array : array/tensor-like object, optional
            The input array to retrieve the information from. If not provided,
            the default device and framelib associated with the class instance
            are used.

        mode : int, optional
            Specifies the type of information to retrieve:
            - mode = 0: Returns the name of the mathematical framework (framename).
            - mode = 1: Returns the device on which computations are performed.
            - mode = 2: Returns both the framename and the device as a list.
            - mode = 3: Returns the framelib object associated with the array.

        Returns:
        --------
        info : str or list or object
            The requested information based on the specified mode:
            - mode = 0: Returns the framename as a string.
            - mode = 1: Returns the device as a string.
            - mode = 2: Returns [framename, device] as a list of strings.
            - mode = 3: Returns the framelib object associated with the array.

        Notes:
        ------
        The device refers to the hardware device on which the computations are
        performed (e.g., CPU or GPU).

        The framelib represents the underlying mathematical framework or
        library used for array operations.

        If an array is provided, its type is determined, and the corresponding
        framelib and device are retrieved. Otherwise, the framelib and device
        associated with the class instance are used.

        The framelib and device values are specific to the mathematical
        framework being used and should be set appropriately during class
        initialization.
        """

        device, framelib = self._get_device(array)
        framename = framelib._interface.framename

        if mode == 0:
            return framename
        elif mode == 1:
            return device
        elif mode == 2:
            return [framename, device]
        elif mode == 3:
            return framelib

    def _get_device(self, array=None):
        """
        Retrieves the device and framelib associated with the given array.

        Parameters:
        -----------
        array : array/tensor-like object, optional
            The input array to get the device and framelib from. If not provided,
            the default device and framelib associated with the class instance
            are returned.

        Returns:
        --------
        [device, framelib] : list
            A list containing the device and framelib associated with the input
            array or the default device and framelib if no array is provided.

        Notes:
        ------
        The device refers to the hardware device on which the computations are
        performed (e.g., CPU or GPU), while framelib represents the underlying
        mathematical framework or library used for array operations.

        If an array is provided, its type is determined and the corresponding
        framelib and device are retrieved. Otherwise, the framelib and device
        associated with the class instance are returned.

        The framelib and device values are specific to the mathematical framework
        being used and should be set appropriately during class initialization.
        """
        if array is None:
            framelib = self.framework
            device = self.device
        else:
            framelib = frameworks._arrayToFrame[type(array)]
            device = framelib._interface.get_device(array)

        return [device, framelib]

    def _get_deviceFramelib(self, device, framename='default'):
        """
        Get the selected framework for device.

        Parameters
        ----------
        device : String
            Device name.
        framename : String, optional
            Framework name to process with the device. If 'default',
            the last selected framework for this same device is returned.
            The default is 'default'.

        Raises
        ------
        Exception
            It raises an exception in case the the framename is not registered
            for the device.

        Returns
        -------
        frame : Framework library
            Mathematical framework library compatible with device.

        """
        deviceList = frameworks.deviceFrames[device]

        if framename == 'default':
            framename = self.default[device]
        elif framename not in deviceList:
            raise Exception(f'Framework {framename} not available for device '
                            f'{device}. Expected one of {deviceList}')

        framelib = frameworks.frameworks[framename]

        return framelib

    def to_device(self, array, device=None, framename=None, dtype=None):
        """
        Send array to a new array in device using framework given by framename.

        Parameters
        ----------
        array : ArrayClass of any registered framework
            Array to be copied to another device/framework.
        device : String
            Device name.
        framename : String, optional
            Framework name to process with the device. If 'default',
            the last selected framework for this same device is returned.
            The default is 'default'.

        Returns
        -------
        ArrayClass of selected device/framework
            New array object. If selected device matches the input array
            device, the new array object is a view of the input one.

        """
        framelib = None

        if device is None:
            device = self.device
            if framename is None:
                framelib = self.framework

        device = frameworks._get_deviceSuffix(device)

        if framelib is None:
            if framename is None:
                framename = self.default[device]

            framelib = self._get_deviceFramelib(device, framename)

        prevFramelib = frameworks._arrayToFrame[type(array)]

        return framelib._interface.to_device(array, device, prevFramelib,
                                             dtype)

    @contextmanager
    def fft_backend(self, threads=-1):
        """
        Context manager for specific FFT backends.

        Parameters
        ----------
        threads : int, optional
            Threads number for parallel processing in CPU devices. If -1 all
            threads are used. The default is -1.

        Yields
        ------
        fft : FFT library interface.
            Library interface for active framework.

        Examples
        --------
        >>> import artis.math.framework as fw
        >>> xp = fw.frame('cuda', 'cupy')
        >>> with xp.fft_backend() as fft:
        >>>     A = xp.ones((16, 16))
        >>>     AT = fft.rfft(A)

        """
        from .. import fft

        with fft.set_framework_backend(self.framework._interface.framename,
                                       threads=threads):
            yield fft

    def __getattr__(self, name):
        """Catch the sequence of called attributes.

        Framework functions/attributes can be called and it stores the full
        sequence before calling the selected framework function/attribute.

        """
        attList = self._attribList
        attListExists = len(attList) > 0

        modulePath = attList[0].copy() if attListExists else []

        # print(f'__getattr__: name={name} - attList={attList}')

        # Attrib name to get the pointer to module/object or attrib
        if name == '_get':
            if attListExists:
                del attList[0]
            return rgetattr(self.framework._interface.frame, modulePath)
        # For Ipython console purposes
        elif name in ['getdoc', 'size', 'shape', 'float', 'compute'] or name[0] == '_':
            if attListExists:
                del attList[0]
            return rgetattr(self.framework._interface.frame,
                            modulePath + [name])
        # elif name == 'nan':
            # return getattr(self.framework._interface.frame, name)

        attribExists = name in dir(rgetattr(self.framework._interface.frame,
                                            modulePath))
        if attListExists and attribExists:
            attList[0].append(name)
        elif attribExists or name in dir(self.framework._interface.frame):
            attList.insert(0, [name])
            attribExists = True  # In case it exists in framework root.
        else:
            raise AttributeError(f"module {self.framework._interface.framename!r} "
                                  f"has no attribute {name!r}")

        # To directly return attributes like nan or type values(float32, ...)
        if attribExists:
            attrib = rgetattr(self.framework._interface.frame, attList[0])
            if isinstance(attrib, (float, type)):
                del attList[0]
                return attrib

        # print(f'__getattr__:attList={self._attribList}')

        return self

    def __current_attrib__(self):

        print(f'__current_attrib__:attList={self._attribList}')
        return self


    def __call__(self, *args, **kwargs):
        """
        Redirect function calls to selected framework.

        Framework functions/attributes can be called through this frame class.

        In case an array constructor method is called, active device/framework
        is used. Device/framework can also be selected by setting them as
        arguments. Here, we remove them from args and pass as parameters to
        the framework interface function caller.

        """
        framelib = None
        device = None
        framename = None

        if 'device' in kwargs:
            device = frameworks._get_deviceSuffix(kwargs['device'])
            del kwargs['device']

            if 'framename' in kwargs:
                framename = kwargs['framename']
                del kwargs['framename']
            else:
                framename = 'default'

            device = frameworks._get_deviceSuffix(device)
            framelib = self._get_deviceFramelib(device, framename)

        # elif type(args[0]) in self._arrayToFrame:
        #     framelib = self._arrayToFrame[type(args[0])]
        #     device = framelib._interface.get_device(args[0])

        else:
            device = self.device

            if 'framename' in kwargs:
                framename = kwargs['framename']
                del kwargs['framename']

                framelib = self._get_deviceFramelib(device, framename)
            else:
                framelib = self.framework

        attribList = self._attribList[0]
        del self._attribList[0]

        # print("__call__")

        return framelib._interface._func(attribList, device, *args, **kwargs)

    def __dir__(self):
        """Return list of attribute names of active framework.

        For Ipython console purposes.
        """
        if len(self._attribList) > 0:
            return dir(rgetattr(self.framework._interface.frame,
                                self._attribList[0]))

        return dir(self.framework._interface.frame)


    def __str__(self):
        """Print the active device/framework."""
        return f'ArtisFramework({self.device}, {self.framename})'

    def __repr__(self):
        """Print the active device/framwork."""
        print(self._attribList)
        # self._attribList.clear()
        return self.__str__()


class DirectDevice():
    """Create a framework manager instance based on a specific device.

    The target framework can be selected within the device.

    """

    def __init__(self, device):
        """
        Initialize the class.

        Parameters
        ----------
        device : String
            Device to be used.

        """
        self.device = device
        self._frame = frame(device)

    def __getattr__(self, name):
        """Redirect attributes to active framework library."""
        return getattr(self._frame, name)

    def set_frame(self, framename):
        """
        Change active framework.

        Parameters
        ----------
        framename : String
            Framework name to process with the device.

        """
        self._frame.set_device(self.device, framename)

    def __dir__(self):
        """Return list of attribute names of active framework.

        For Ipython console purpoes.
        """
        return dir(self._frame.framelib._interface.frame)


