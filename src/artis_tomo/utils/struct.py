"""
Structure class.

@author: joton
"""
import numpy as np
from .dict import AttributeDict


class StructBase(AttributeDict):
    """
    Base structure class based on dictionary object.

    Dictionary keys are also strcture attributes. A list of custom attribs can
    be declared to specify the structure.

    """

    attribs = None
    __children = {}

    def __init__(self, inputdict: dict = None):
        """
        Initialize Struct and, optionally, from dictionary.

        If inputdict is passed, it must contain all required attribs.

        """
        super(StructBase, self).__init__()

        if self.attribs is not None:
            if inputdict is None:
                for key in self.attribs:
                    self[key] = None
            else:
                for key in self.attribs:
                    if key in inputdict:
                        self[key] = inputdict[key]
                    else:
                        raise Exception(f'psfStruct: missing {key} attrib in '
                                        'input dictinoray.')

    def __init_subclass__(cls):
        cls.__children[cls.__name__] = cls

    @classmethod
    def get_children(cls):
        return cls.__children


    def __repr__(self, level=0):

        rep = ''
        indent = ' '*2*level

        for k, v in self.items():

            if isinstance(v, np.ndarray):
                vt = f'Array {v.shape}\n'
            elif isinstance(v, StructBase):
                vt = 'Struct\n' + v.__repr__(level + 1)
            else:
                vt = f'{v}\n'

            rep += indent + f'{k}: {vt}'

        return rep
