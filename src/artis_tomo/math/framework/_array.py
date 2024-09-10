#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 20:05:45 2023

@author: joton
"""
from numpy.lib import mixins
from ...utils.functools import rgetattr
from typing import overload
from ._frame import frame
from ._frameworks import frameworks

class Array(mixins.NDArrayOperatorsMixin):

    def __init__(self, array=None, device=None, framename=None, copy=False):

        self.array = None
        self._arrays = {}
        self.frame = frame()

        if array is not None:
            if type(array) in frameworks._arrayToFrame or \
               isinstance(array, self.__class__):
                if device is None and framename is None:
                    if isinstance(array, self.__class__):
                        framename = array.frame.framename
                        device = array.frame.device
                        array = array.array
                    else:
                        framename, device = self.frame.get_frameInfo(array,
                                                                     mode=2)
                elif isinstance(array, self.__class__):
                    array = array.array

                self.frame.set_device(device, framename)
                if copy:
                    self._arrays[framename] = self.frame.array(array, copy=True)
                else:
                    self._arrays[framename] = self.frame.array(array)
            else:
                if device is not None:
                    self.frame.set_device(device, framename)
                self._arrays[self.frame.framename] = self.frame.asarray(array)

            self.array = self._arrays[self.frame.framename]

        elif device is not None:
                self.frame.set_device(device, framename)

    def to_frame(self, framename, dtype=None, copy=False):

        self.to_device(self.frame.device, framename, dtype, copy)

    @overload
    def to_device(self, array, dtype=None):
        ...

    def to_device(self, device: str, framename:str=None, dtype=None, keep=False):

        if isinstance(device, self.__class__):
            array = device
            device = array.frame.device
            framename = array.frame.framename

        framenameIni = self.frame.framename
        arrayIni = self.array

        self.frame.set_device(device, framename)
        framename = self.frame.framename

        if framenameIni == framename:
            return

        self._arrays[framename] = self.frame.to_device(arrayIni, dtype=dtype)
        self.array = self._arrays[framename]

        if not keep:
            del self._arrays[framenameIni]

    def __extract_arrays_recursive(self, arg):

        if isinstance(arg, self.__class__):
            if self.frame.device != arg.frame.device or \
               self.frame.framename != arg.frame.framename:
                return self.frame.to_device(arg.array)
            else:
                return arg.array
        elif isinstance(arg, tuple):
            newargs = ()
            for obj in arg:
                newargs += (self.__extract_arrays_recursive(obj), )
            return newargs
        else:
            return arg

    def __extract_types__(self, types):

        newtypes = ()
        for typ in types:
            if typ == self.__class__:
                newtypes += (self.array.__class__, )
            else:
                newtypes += (typ, )

        return newtypes

    def __extract_arrays__(self, inputs, kwargs):

        inputs = self.__extract_arrays_recursive(inputs)

        # if isinstance(inputs, tuple):
        #     print('0')
        #     newinputs = ()
        #     for obj in inputs:
        #         print(type(obj))
        #         if isinstance(obj, self.__class__):
        #             print('1')
        #             if self.frame.device != obj.frame.device or \
        #                self.frame.framename != obj.frame.framename:
        #                 obj = self.frame.to_device(obj.array)
        #                 print('2')
        #             else:
        #                 print('3')
        #                 obj = obj.array
        #         # convert to own device
        #         # elif not isinstance(obj, frameworks.frameworks[
        #         #         self.frame.framename]._interface.arrayClass) and \
        #         #         type(obj) in frameworks._arrayToFrame:
        #         #     obj = self.frame.to_device(obj)

        #         newinputs += (obj, )
        # else:
        #     print('4')

        #     if isinstance(inputs, self.__class__):
        #         if self.frame.device != inputs.frame.device or \
        #            self.frame.framename != inputs.frame.framename:
        #             newinputs = self.frame.to_device(inputs.array)
        #         else:
        #             newinputs = obj.array

        for key, obj in kwargs.items():
            if isinstance(obj, tuple):
                newObj = ()
                for obj2 in obj:
                    if id(obj2) == id(self):
                        newObj += (obj2.array,)
                    elif isinstance(obj2, self.__class__):
                        if self.frame.device != obj2.frame.device and \
                           self.frame.framename != obj2.frame.framename:
                            newObj += (self.frame.to_device(obj2.array), )
                        else:
                            newObj += (obj2.array,)
                kwargs[key] = newObj
            elif isinstance(obj, self.__class__):
                if self.frame.device != obj.frame.device and \
                   self.frame.framename != obj.frame.framename:
                    kwargs[key] = self.frame.to_device(obj.array)
                else:
                    kwargs[key] = obj.array

            #     #Cupy allows for more than one output. It must be a tuple
            # if key == 'out' and self.frame.framename == 'cupy':
            #     kwargs[key] = (kwargs[key], )
            # Torch requires out not to be a tuple
            # if key == 'out' and self.frame.framename == 'torch':
            #     kwargs[key] = kwargs[key][0]
        return inputs, kwargs

    def __getitem__(self, arg):
        return self.__class__(self.array[arg])

    def __setitem__(self, arg, value):
        self.array[arg] = value

    def __array__(self):
        return self.array.__array__()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        array = self.array
        ufuncExists = '__array_ufunc__' in dir(array)

        # print(f'__array_ufunc__: {ufunc}')
        # print(f'inputs={inputs}')
        # print(f'kwargs={kwargs}')

        # if not ufuncExists:
        #     inputs = inputs[1:]

        inputs, kwargs = self.__extract_arrays__(inputs, kwargs)

        # print(f'inputs={inputs}')
        # print(f'kwargs={kwargs}')

        if ufuncExists:
            output = array.__array_ufunc__(ufunc, method, *inputs, **kwargs)

            if 'out' in kwargs:
                return self
            return self.__class__(output)

        # else:
        #     if 'out' in kwargs:
        #         kwargs['out'] = kwargs['out'][0]
        #     fun = getattr(self.frame.framework._interface.frame, ufunc.__name__)
        #     return self.__class__(fun(self.array, *inputs, **kwargs))

    def __array_function__(self, func, types, *args, **kwargs):

        # print("ArtisArray.__array_function__")

        # print(f'args={args}')
        # print(f'kwargs={kwargs}')

        args, kwargs = self.__extract_arrays__(args, kwargs)

        # print(f'args={args}')
        # print(f'kwargs={kwargs}')

        # args = args[0]
        # print(f'types={types}')
        types = self.__extract_types__(types)
        # print(f'types={types}')

        output = self.array.__array_function__(func, types, *args, **kwargs)
        return self.__class__(output)

        # try:
        #     fun = rgetattr(self.frame.framework._interface,
        #                    module + [func.__name__])
        # except AttributeError:
        #     return NotImplemented

        # return self.__class__(fun(*args, **kwargs))

    def __array_wrap__(self, obj, context=None):
        """
        Special hook for ufuncs.

        Wraps the numpy array and sets the mask according to context.

        """
        print('__array_wrap__')
        if isinstance(obj, self.__class__):  # for in-place operations
            result = obj
        elif type(obj) in self.frame._arrayToFrame:
            result = self.__class__(obj)

        return result

    def __array_finalize__(self, obj):
        """
        Special hook for ufuncs.

        Wraps the numpy array and sets the mask according to context.

        """
        print('__array_finalize__')
        if isinstance(obj, self.__class__):  # for in-place operations
            result = obj
        elif type(obj) in self.frame._arrayToFrame:
            result = self.__class__(obj)

        return result

    def __getattr__(self, name):
        # print('__getattr__')
        attrib = getattr(self.array, name)

        if callable(attrib):
            def func(*args, **kwargs):
                result = attrib(*args, **kwargs)
                arrayClass = frameworks.frameworks[self.frame.framename]._interface.arrayClass
                if isinstance(result, arrayClass):
                    result = self.__class__(result)
                return result

            return func
        else:
            return attrib

    # def __eq__(self, obj):

    #     if isinstance(obj, self.__class__):
    #         obj == obj.array

    #     return self.__class__(self.array == obj)

    def __dir__(self):
        return self.array.__dir__()

    def __str__(self):
        """Print the active device/framwork."""
        return str(self.array)

    def __repr__(self):
        framename = self.frame.framename
        device = self.frame.device
        array = self.array
        mystr = f"ArtisArray({device}, {framename})\n" + str(array)
        return mystr
