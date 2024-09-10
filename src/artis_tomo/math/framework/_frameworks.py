#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 20:26:46 2023

@author: joton
"""
import importlib
from contextlib import contextmanager
from ...utils.functools import rgetattr

class frameworks():

    frameworks = {}
    _arrayToFrame = {}
    deviceFrames = {}

    @classmethod
    def add_framework(cls, moduleNameList, package):
        """
        Register modules that define a wrapper interface object.

        Parameters
        ----------
        moduleNameList : List of strings
            Module names which contain the mathematical framework interface
            class named _interface. Example: '._numpy', '._cupy']
        package : String
            Package to use as the anchor point from which to resolve the
            relative import of the modules. Example: 'artis.math.framework'

        """
        for moduleName in moduleNameList:

            try:
                framelib = importlib.import_module(moduleName, package)
                framename = framelib._interface.framename
                if framename not in cls.frameworks:
                    cls.frameworks[framename] = framelib
                    cls._arrayToFrame[framelib._interface.arrayClass] = framelib

                    for device in framelib._interface.devices:
                        if device not in cls.deviceFrames:
                            cls.deviceFrames[device] = []
                        cls.deviceFrames[device].append(framename)

            except ImportError:  # as err:
                # print(err)
                pass

    @classmethod
    def _get_deviceSuffix(cls, device):
        """
        Complete device index if required.

        If device name does not include suffix index ":X" it its append using
        default index 0.

        It also checks the device is registered.

        Parameters
        ----------
        device : String
            Device name.

        Raises
        ------
        Exception
            It raises an exception in case the device is not registered.

        Returns
        -------
        device : String
            Device name including index number.

        """
        if device == 'cuda':
            device += ':0'
        if device not in cls.deviceFrames:
            raise Exception(f'Device {device} not available. '
                            f'Expected one of {list(cls.deviceFrames.keys())}')
        return device
