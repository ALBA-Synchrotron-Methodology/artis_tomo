#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom dict objects

@author: joton
"""

# class AttributeDict(dict):
#     __slots__ = ()
#     __getattr__ = dict.__getitem__
#     __setattr__ = dict.__setitem__


class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class RecursiveDict(dict):
    def get(self, *args, **kwargs):
        nargs = len(args)
        if nargs == 2:
            default = args[-1]
        else:
            default = None

        keys = args[0]
        if not isinstance(keys, list):
            keys = [keys]

        obj = super()
        for key in keys:
            obj = obj.get(key, default)
            if not isinstance(obj, dict):
                break

        return obj

    def merge_dict(self, dictmain):
        """
        Merges local dict with dictmain dictionary recursively.

        Dictmain overwrites local dict in case of key conflicts.

        Parameters:
        dictmain (dict): The second dictionary, whose values will be overwritten

        Returns:
        dict: The merged dictionary.
        """
        merged = super().copy()  # Start with dict1's keys and values
        for key, value in dictmain.items():
            if key in merged and isinstance(merged[key],
                                            dict) and isinstance(value,
                                                                 dict):
                # Recursively merge if both are dictionaries
                merged[key] = self._merge_dicts(merged[key], value)
            else:
                # Otherwise, overwrite with dictmain's value
                merged[key] = value
        return self.__class__(merged)

    @classmethod
    def _merge_dicts(cls, dict1, dict2):
        """
            Merges two dictionaries recursively.

             Dict2 overwrites dict1 in case of key conflicts.

            Parameters:
            dict1 (dict): The first dictionary.
            dict2 (dict): The second dictionary, whose values will
                          overwrite those in dict1.

            Returns:
            dict: The merged dictionary.
            """
        merged = dict1.copy()  # Start with dict1's keys and values
        for key, value in dict2.items():
            if key in merged and isinstance(merged[key],
                                            dict) and isinstance(value,
                                                                 dict):
                # Recursively merge if both are dictionaries
                merged[key] = cls._merge_dicts(merged[key], value)
            else:
                # Otherwise, overwrite with dict2's value
                merged[key] = value
        return merged
