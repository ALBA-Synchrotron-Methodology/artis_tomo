#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Euler angles conversion with x,y,z translations as homogeneus coordinates

@author: joton
"""

import numpy as np
import transforms3d.euler
from transforms3d.euler import euler2mat, mat2euler

SXYZ = 'sxyz'
SXYX = 'sxyx'
SXZY = 'sxzy'
SXZX = 'sxzx'
SYZX = 'syzx'
SYZY = 'syzy'
SYXZ = 'syxz'
SYXY = 'syxy'
SZXY = 'szxy'
SZXZ = 'szxz'
SZYX = 'szyx'
SZYZ = 'szyz'
RZYX = 'rzyx'
RXYX = 'rxyx'
RYZX = 'ryzx'
RXZX = 'rxzx'
RXZY = 'rxzy'
RZYZ = 'ryzy'
RZXY = 'rzxy'
RYXY = 'ryxy'
RYXZ = 'ryxz'
RZXZ = 'rzxz'
RXYZ = 'rxyz'


class TMat3D:
    """
    A class for managing a 3D transformation matrix that describes affine transformations.

    Affine transformations include translation, rotation, scaling, shearing, and reflection.

    Attributes
    ----------
    n : int
        The dimension of the transformation matrix (3 for TMat3D).
    matrix : numpy.ndarray
        The transformation matrix representing the affine transformation.

    Methods
    -------
    __init__(self, array)
        Initialize the TMat3D object with a given array representing the transformation matrix.
    mat2params(self, zoom=False, axes=SZYZ)
        Convert the transformation matrix to parameters for translation, rotation, and scaling.
    __mul__(self, other)
        Multiply the TMat3D object with another TMat3D object or scalar.
    __str__(self)
        Return a string representation of the TMat3D object.
    __repr__(self)
        Return a string representation of the TMat3D object.

    Properties
    ----------
    inv
        Get the inverse of the TMat3D object.
    """

    n = 3

    def __init__(self, array):
        """
        Initialize the TMat3D object with a given array representing the transformation matrix.

        Parameters
        ----------
        array : numpy.ndarray
            The transformation matrix array.

        Raises
        ------
        ValueError
            If the input array is not an augmented matrix of affine transformations.
        """
        if array.ndim == 2:
            array = array[None, ...]
        if array.shape[-2:] == (self.n+1, self.n+1) and \
           np.all(array[:, self.n, :self.n-1] == 0) and \
           np.all(array[:, self.n, self.n] == 1):
            self._matrix = array
        else:
            raise ValueError("Input array is not an augmented matrix of affine transformations.")

    @property
    def matrix(self):
        """
        Get the transformation matrix.

        Returns
        -------
        numpy.ndarray
            The transformation matrix.
        """
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        """
        Set the transformation matrix.

        Parameters
        ----------
        value : numpy.ndarray
            The new transformation matrix.
        """
        self._matrix = np.array(value)

    def get(self, n):

        return self.__class__(self._matrix[n])

    def set(self, tMat, n):
        if len(tMat) == 1:
            self._matrix[n] = tMat.matrix
        else:
            raise ValueError(f"Input  TMat matrix length {len(tMat)} is not 1.")

    @property
    def inv(self):
        """
        Get the inverse of the TMat3D object.

        Returns
        -------
        TMat3D
            The inverse of the TMat3D object.
        """
        return self.__class__(np.linalg.inv(self._matrix))

    @property
    def shifts(self):
        """
        Get the shifts of the TMat3D object.

        Returns
        -------
        shifts : 1d array
            Shifts from the transformation matrix.
        """
        return self._matrix[:, :-1, -1]

    @shifts.setter
    def shifts(self, value):
        """
        Set the shifts of the TMat3D object.

        Parameter
        -------
        shifts : 1d array
            Shifts to set in the transformation matrix.
        """
        return
        self._matrix[:, :-1, -1] = value

    def mat2params(self, zoom=False, axes=SZYZ):
        """
        Convert the transformation matrix to parameters for translation, rotation, and scaling.

        Parameters
        ----------
        zoom : bool, optional
            Flag indicating whether to include zoom parameters (default is False).
        axes : ..., optional
            The axes convention for rotation (default is SZYZ).

        Returns
        -------
        tuple
            A tuple containing the angles, shifts, and zoom parameters (if enabled).
        """
        n = len(self._matrix)
        out = [None]*n

        for k in range(n):
            out[k] = self._mat2paramsOne(self._matrix[k], zoom, axes)

        if n > 1:
            return out
        else:
            return out[0]
    
    @classmethod
    def _mat2paramsOne(cls, matrix, zoom=False, axes=SZYZ):

        shifts, R, Z, _ = transforms3d.affines.decompose(matrix)
        angles = mat2euler(R, axes)
        # IF SZYZ scheme, force rotY angle in range [0, 180º]
        if axes == SZYZ and angles[1] < 0:
            angles = cls._fixAnglesZYZ(angles)

        shifts = [i for i in shifts]
        if zoom:
            Z = [i for i in Z]
            return angles, shifts, Z
        else:
            return angles, shifts

    @staticmethod
    def _fixAnglesZYZ(angles):
        """
        For scheme ZYZ, fix angles to keep rotY to be in range [0, PI]

        Parameters
        ----------
        angles : 1D array, list
            List of Euler angles.

        Returns
        -------
        angles : list
            Fixed Euler angles

        """
        rot = angles[0] + np.pi
        if rot > np.pi:
            rot -= 2.*np.pi
        psi = angles[2] - np.pi
        if psi <= -np.pi:
            psi += 2.*np.pi

        return (rot, -angles[1], psi)

    def __len__(self):
        return len(self._matrix)

    def __getitem__(self, key):
        if isinstance(key, (int, slice, list)):
            return self.__class__(self._matrix[key])
        else:
            raise ValueError("Only one index is supported.")

    def __setitem__(self, key, newvalue):
        if isinstance(key, int):
            if isinstance(newvalue, self.__class__) and len(newvalue) == 1:
                self._matrix[key] = newvalue.matrix
            else:
                raise ValueError("Input is not a TMat matrix of size 1.")
        elif isinstance(key, slice):
            raise NotImplementedError("Setting several matrices at once is "
                                      "not currently implemented.")
        else:
            raise ValueError("Only one index is supported.")

    def __add__(self, other):
        if isinstance(other, self.__class__):
            out = np.concatenate((self._matrix, other._matrix), 0)
        else:
            raise NotImplementedError(f"Addition of {type(other)} "
                                      "is not currently implemented.")
        return self.__class__(out)

    def __mul__(self, other):
        """
        Multiply the TMat3D object with another TMat3D object or scalar.

        Parameters
        ----------
        other : TMat3D or scalar
            The object or scalar to multiply with.

        Returns
        -------
        TMat3D
            The result of the multiplication.

        Raises
        ------
        NotImplementedError
            If multiplication by `other` is not implemented.
        """
        if isinstance(other, self.__class__):
            leno = len(other)
            lens = len(self._matrix)
            if leno == 1 or lens == 1 or leno == lens:
                out = np.matmul(self._matrix, other._matrix)
            else:
                raise NotImplementedError(f"Length of external TMat is not 1 "
                    f"or does not match own length of {len(self._matrix)} "
                    "elements.")

        elif isinstance(other, float) or isinstance(other, int):
            zoom = np.diag([other, ]*self.n + [1, ])
            out = np.matmul(self._matrix, zoom)

        elif isinstance(other, np.ndarray) and len(other) == self.n:
            out = np.dot(self._matrix, np.append(other, 1))
            return out[:, :-1]
        else:
            raise NotImplementedError(f"Multiplication by {type(other)} "
                                      "is not currently implemented.")
        return self.__class__(out)

    def __rmul__(self, other):
        """
        Multiply the TMat3D object from the left with scalar value as zoom.

        Parameters
        ----------
        other : scalar
            The scalar to compose a zoom matrix to multiply with.

        Returns
        -------
        TMat3D
            The result of the multiplication.

        Raises
        ------
        NotImplementedError
            If multiplication by `other` is not implemented.
        """
        if isinstance(other, float) or isinstance(other, int):
            zoom = np.diag([other, ]*self.n + [1, ])
            out = np.matmul(zoom, self._matrix)
        else:
            raise NotImplementedError(f"Multiplication by {type(other)} "
                                       "is not currently implemented.")
        return self.__class__(out)

    def __str__(self):
        """
        Return a string representation of the TMat3D object.

        Returns
        -------
        str
            The string representation of the TMat3D object.
        """
        return f'Artis TransMatrix{self.n}D(\n{self._matrix})'

    def __repr__(self):
        """
        Return a string representation of the TMat3D object.

        Returns
        -------
        str
            The string representation of the TMat3D object.
        """
        return self.__str__()


class TMat2D(TMat3D):
    """
    A subclass of TMat3D representing a 2D transformation matrix.

    Inherits all methods and attributes from TMat3D.

    Attributes
    ----------
    n : int
        The dimension of the transformation matrix (2 for TMat2D).

    Methods
    -------
    mat2params(self, zoom=False)
        Convert the transformation matrix to parameters for translation, rotation, and scaling.
    """

    n = 2

    def mat2params(self, zoom=False):
        """
        Convert the transformation matrix to parameters for translation, rotation, and scaling.

        Parameters
        ----------
        zoom : bool, optional
            Flag indicating whether to include zoom parameters (default is False).

        Returns
        -------
        tuple
            A tuple containing the angle, shifts, and zoom parameters (if enabled).
        """
        n = len(self._matrix)
        out = [None]*n

        for k in range(n):
            out[k] = self._mat2paramsOne(self._matrix[k], zoom)

        if n > 1:
            return out
        else:
            return out[0]

    @staticmethod
    def _mat2paramsOne(matrix, zoom=False):

        shifts, R, Z, _ = transforms3d.affines.decompose(matrix)
        angle = np.arctan2(R[1, 0], R[0, 0])
        shifts = [i for i in shifts]
        if zoom:
            Z = [i for i in Z]
            return angle, shifts, Z
        else:
            return angle, shifts


class _trBase:
    """
    A namespace for managing the construction of custom 3D transformation matrix objects.

    Attributes
    ----------
    fw : module, optional
        The module to use for array operations (default is numpy).
    n : int
        The dimension of the transformation matrix (3 for tr3d).

    Methods
    -------
    rotXmat(alpha)
        Create a transformation matrix for a rotation around the X-axis.
    rotYmat(beta)
        Create a transformation matrix for a rotation around the Y-axis.
    rotZmat(gamma)
        Create a transformation matrix for a rotation around the Z-axis.
    angles2mat(angles, shifts=None, zoom=None, axes=SZYZ)
        Create a transformation matrix from rotation angles.
    shifts2mat(shifts, zoom=None)
        Create a transformation matrix from translation shifts.
    """

    fw = np
    n = None
    _TMat = None

    @classmethod
    def shifts2mat(cls, shifts, zoom=None):
        """
        Create a transformation matrix from translation shifts.

        Parameters
        ----------
        shifts : list
            A list containing the translation shifts.
        zoom : list, optional
            A list containing the zoom parameters (default is None).

        Returns
        -------
        TMat3D
            The transformation matrix created from the translation shifts.
        """
        m = np.eye(cls.n+1)
        if len(shifts) != cls.n:
            raise ValueError(f"Bad shifts length for {cls.n}d trans matrix.")
        m[:cls.n, cls.n] = shifts
        if zoom is not None:
            if len(zoom) != cls.n:
                raise ValueError(f"Bad zoom length for {cls.n}d trans matrix.")
            m[range(cls.n), range(cls.n)] *= zoom

        return cls._TMat(cls.fw.array(m))

    @classmethod
    def empty(cls, n=1):
        array = cls.fw.concatenate([cls.fw.identity(cls.n+1)[None, ...]]*n, 0)
        return cls._TMat(array)

    @classmethod
    def fromList(cls, tMatList):
        n = len(tMatList)
        array = cls.fw.zeros((n, cls.n+1, cls.n+1))
        array[:, cls.n, cls.n] = 1
        mat = cls._TMat(array)

        for k, tMat in enumerate(tMatList):
            mat[k] = tMat

        return mat

class tr3d(_trBase):
    """
    A namespace for managing the construction of custom 3D transformation matrix objects.

    Attributes
    ----------
    fw : module, optional
        The module to use for array operations (default is numpy).
    n : int
        The dimension of the transformation matrix (3 for tr3d).

    Methods
    -------
    rotXmat(alpha)
        Create a transformation matrix for a rotation around the X-axis.
    rotYmat(beta)
        Create a transformation matrix for a rotation around the Y-axis.
    rotZmat(gamma)
        Create a transformation matrix for a rotation around the Z-axis.
    angles2mat(angles, shifts=None, zoom=None, axes=SZYZ)
        Create a transformation matrix from rotation angles.
    shifts2mat(shifts, zoom=None)
        Create a transformation matrix from translation shifts.
    """

    fw = np
    n = 3
    _TMat = TMat3D
    
    @classmethod
    def rotXmat(cls, alpha):
        """
        Create a transformation matrix for a rotation around the X-axis.

        Parameters
        ----------
        alpha : float
            The rotation angle in radians.

        Returns
        -------
        TMat3D
            The transformation matrix representing the rotation around the X-axis.
        """
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        out = cls.fw.array([[1,  0,   0, 0],
                            [0, ca, -sa, 0],
                            [0, sa,  ca, 0],
                            [0,  0,   0, 1]])
        return cls._TMat(out)

    @classmethod
    def rotYmat(cls, beta):
        """
        Create a transformation matrix for a rotation around the Y-axis.

        Parameters
        ----------
        beta : float
            The rotation angle in radians.

        Returns
        -------
        TMat3D
            The transformation matrix representing the rotation around the Y-axis.
        """
        cb = np.cos(beta)
        sb = np.sin(beta)

        out = cls.fw.array([[ cb, 0, sb, 0],
                            [  0, 1,  0, 0],
                            [-sb, 0, cb, 0],
                            [  0, 0,  0, 1]])
        return cls._TMat(out)

    @classmethod
    def rotZmat(cls, gamma):
        """
        Create a transformation matrix for a rotation around the Z-axis.

        Parameters
        ----------
        gamma : float
            The rotation angle in radians.

        Returns
        -------
        TMat3D
            The transformation matrix representing the rotation around the Z-axis.
        """
        cg = np.cos(gamma)
        sg = np.sin(gamma)

        out = cls.fw.array([[cg, -sg, 0, 0],
                            [sg,  cg, 0, 0],
                            [ 0,   0, 1, 0],
                            [ 0,   0, 0, 1]])
        return cls._TMat(out)

    @classmethod
    def angles2mat(cls, angles, shifts=None, zoom=None, axes=SZYZ):
        """
        Create a transformation matrix from rotation angles.

        Parameters
        ----------
        angles : tuple
            A tuple containing the rotation angles.
        shifts : list, optional
            A list containing the translation shifts (default is None).
        zoom : list, optional
            A list containing the zoom parameters (default is None).
        axes : str or tuple, optional
            The rotation sequence (default is SZYZ).

        Returns
        -------
        TMat3D
            The transformation matrix created from the rotation angles.
        """
        R = euler2mat(*angles, axes)
        if zoom is not None:
            if len(zoom) != cls.n:
                raise ValueError(f"Bad zoom length for {cls.n}d trans matrix.")
            Z = np.diag(zoom)
            R = np.dot(R, Z)

        A = np.eye(cls.n+1)
        A[:cls.n, :cls.n] = R
        if shifts is not None:
            A[:cls.n, cls.n] = shifts
        return cls._TMat(cls.fw.array(A))

    @classmethod
    def removeShifts(cls, obj):
        arr = obj.matrix.copy()
        arr[:, 0:cls.n, cls.n] = 0
        out = cls._TMat(cls.fw.array(arr))
        return out

    @classmethod
    def removeAngles(cls, obj):
        arr = obj.matrix.copy()
        arr[:, 0:cls.n, 0:cls.n] = np.eye(cls.n)
        out = cls._TMat(cls.fw.array(arr))
        return out


class tr2d(_trBase):
    """
    A namespace for managing the construction of custom 2D transformation matrix objects.

    Attributes
    ----------
    fw : module, optional
        The module to use for array operations (default is numpy).
    n : int
        The dimension of the transformation matrix (2 for tr2d).

    Methods
    -------
    _rot2mat(gamma)
        Create a transformation matrix for a rotation.
    """

    fw = np
    n = 2
    _TMat = TMat2D

    @classmethod
    def _rot2mat(cls, gamma):
        """
        Create a transformation matrix for a rotation.

        Parameters
        ----------
        gamma : float
            The rotation angle in radians.

        Returns
        -------
        np.ndarray
            The transformation matrix representing the rotation.
        """
        cg = np.cos(gamma)
        sg = np.sin(gamma)

        out = cls.fw.array([[cg, -sg],
                            [sg,  cg]])
        return out

    @classmethod
    def angles2mat(cls, angle, shifts=None, zoom=None):
        # angle = np.deg2rad(np.array(angle))
        R = cls._rot2mat(angle)
        if zoom is not None:
            if len(zoom) != cls.n:
                raise ValueError(f"Bad zoom length for {cls.n}d trans matrix.")
            Z = np.diag(zoom)
            R = np.dot(R, Z)

        A = np.eye(cls.n+1)
        A[:cls.n, :cls.n] = R
        if shifts is not None:
            A[:cls.n, cls.n] = shifts
        return cls._TMat(A)
    

# TODO: to be removed
def rotX2mat(alpha):

    out = np.matrix([[1,    0,              0         ],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha),  np.cos(alpha)]])
    return out


def rotY2mat(beta):

    out = np.matrix([[np.cos(beta),   0,     np.sin(beta)],
                    [    0,           1,        0        ],
                    [-np.sin(beta),   0,    np.cos(beta)]])
    return out


def rotZ2mat(gamma):

    out = np.matrix([[np.cos(gamma), -np.sin(gamma), 0 ],
                     [np.sin(gamma),  np.cos(gamma), 0 ],
                     [   0,              0,          1 ]])
    return out





def d3matShifts(shifts):

    out = np.matrix([[1, 0, shifts[0]],
                     [0, 1, shifts[1]],
                     [0, 0,      1]])
    return out


def d4matShifts(shifts):

    out = np.matrix([[1, 0, 0, shifts[0]],
                     [0, 1, 0, shifts[1]],
                     [0, 0, 1, shifts[2]],
                     [0, 0, 0,      1]])
    return out