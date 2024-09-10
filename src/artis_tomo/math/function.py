"""
Mathematical functions declaration.

@author: joton
"""

import numpy as np


def raisedCos(Ni):
    """
    Create a raised cosine profile.

    Parameters
    ----------
    Ni : int
        Profile size in pixels.

    Returns
    -------
    rc : 1D array
        Raised cosine profile , from 0 to 1.

    """
    rc = (np.sin(np.linspace(-np.pi/2, np.pi/2, Ni)) + 1)/2

    return rc
