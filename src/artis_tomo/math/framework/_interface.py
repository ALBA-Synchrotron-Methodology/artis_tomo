"""Interface wrapper baseclass for mathematical frameworks."""

import abc


class _Wrapper_meta(abc.ABCMeta):
    pass

    @property
    def framename(cls):
        """Name of the framework set in attribute cls_framename."""
        try:
            return cls._framename
        except Exception:
            raise NotImplementedError(f"{cls} does not implement _framename.")

    @property
    def frame(cls):
        """Framework library."""
        try:
            return cls._frame
        except Exception:
            raise NotImplementedError(f"{cls} does not implement _frame.")

    @property
    def arrayClass(cls):
        """Class of the ndarray of the framework."""
        try:
            return cls._arrayClass
        except Exception:
            raise NotImplementedError(f"{cls} does not implement _arrayClass.")

    @property
    def devices(cls):
        """List of supported devices."""
        try:
            return cls._devices
        except Exception:
            raise NotImplementedError(f"{cls} does not implement _devices.")

    def __getattr__(cls, name):
        return getattr(cls.frame, name)
        """Redirect attributes to framework library."""

    def __dir__(cls):
        return dir(cls.frame)
        """Return attributes of framework library."""


class _Wrapper(metaclass=_Wrapper_meta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_device') and
                callable(subclass.get_device) and
                hasattr(subclass, 'to_device') and
                callable(subclass.to_device) and
                hasattr(subclass, '_func') and
                callable(subclass._func) or
                NotImplemented)

    @classmethod
    @abc.abstractmethod
    def get_device(cls, array):
        """Return current device of the array."""
        raise NotImplementedError(f"{cls} does not implement this method.")

    @classmethod
    @abc.abstractmethod
    def to_device(cls, array, device, iniFrame=None, dtype=None):
        """
        Copy input array to a new array in this framework.

        If array.device is accessible from this framework, a view is
        created instead.
        """
        raise NotImplementedError(f"{cls} does not implement this method.")

    @classmethod
    @abc.abstractmethod
    def _func(cls, attribList, device, *args, **kwargs):
        """
        Call methods within the framwework.

        Method nested name is given by attribList. It uses the selected device
        with passed arguments.
        """
        raise NotImplementedError(f"{cls} does not implement this method.")
