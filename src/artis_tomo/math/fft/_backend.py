"""Interface wrapper baseclass for FFT context manager backends."""


class _ScipyBackendCXTBase:
    """
    Base class to customize scipy FFT backends context managers.

    This allows pre and post processing when running a scipy.fft backend.

    Preprocessing can be added in __init__ or __enter__ methods.

    Postprocessing can be added in __exit__ method.

    If a wrapper is required when calling fft methods, it must be added in
    __ua_function__ method.

    """

    __ua_domain__ = 'numpy.scipy.fft'

    @classmethod
    def __init__(cls, backendCXTManager):
        """
        Initialize context manager.

        It uses the original scipy backend context manager to allow
        modify init, enter and exit methods.

        Parameters
        ----------
        backendCXTManager : A backend context manager
            This is context manager given by
            scipy.fft._backend.set_backend(backend) which will be modified.

        """
        # print('_ScipyBackendCXTBase.__init__')
        cls.backendCXTManager = backendCXTManager

    @classmethod
    def __enter__(cls):
        """Context manager enter method."""
        # print('_ScipyBackendCXTBase.__enter__')
        return cls.backendCXTManager.__enter__()

    @classmethod
    def __exit__(cls, type, value, traceback):
        """Context manager exit method."""
        # print('_ScipyBackendCXTBase.__exit__')
        return cls.backendCXTManager.__exit__()

    @staticmethod
    def __ua_function__(method, args, kw):
        """Backend call functions method."""
        # print('_ScipyBackendCXTBase.__ua_function__')
        raise NotImplementedError
