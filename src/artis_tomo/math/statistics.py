"""
Statistic functions.

@author: joton
"""

import numpy as np


def meanVar(array):

    m = np.mean(array)
    flat = array.flatten()
    power = np.dot(flat, flat)/array.size

    return m, power - m**2


def meanStd(array):

    m, var = meanVar(array)
    return m, np.sqrt(var)


def getCustomDistRandom(histlist):

    xp = np.insert(histlist[0], 0, 0).cumsum()
    xp = xp/xp[-1]
    fp = histlist[1]

    def randfun(*argv):
        return np.interp(np.random.rand(*argv), xp, fp)

    return randfun
