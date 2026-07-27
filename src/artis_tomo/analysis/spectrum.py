# Copyright (C) 2025 Joaquín Otón
#
# This file is part of artis_tomo
#
# artis is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# artis is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with artis.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np


def getMaskCorrectedFSC(array1, array2, mask, pixelSize,                       
                        nIter=1, rndThr=0.8, rndOffset=4):   
    """
        Compute the mask-corrected Fourier Shell Correlation (FSC) between two 
        N-Dimensional arrays.

        This function calculates the unmasked FSC, masked FSC, phase-randomized 
        FSC, and mask-corrected FSC between two input volumes, correcting for 
        the effects of masking using phase randomization as described in cryo-EM
        literature.

        Parameters
        ----------
        array1 : np.ndarray
            First 3D input array (volume).
        array2 : np.ndarray
            Second 3D input array (volume).
        mask : np.ndarray
            3D mask array of the same shape as the input arrays.
        pixelSize : float
            Pixel size in Angstroms.
        nIter : int, optional
            Number of phase randomization iterations to average (default is 1).
        rndThr : float, optional
            FSC threshold below which phase randomization is applied (default is 0.8).
        rndOffset : int, optional
            Offset in shells after the randomization threshold for mask correction (default is 4).

        Returns
        -------
        out : np.ndarray
            Array of shape (nShells, 5), where columns are:
                0: Spatial frequency (1/Angstrom)
                1: Unmasked FSC
                2: Masked FSC
                3: Phase-randomized FSC
                4: Mask-corrected FSC

        Notes
        -----
        - The function prints the pixel index and corresponding resolution at which phase randomization is applied.
        - The mask-corrected FSC is computed using the method described by Chen et al., 2013 (doi:10.1038/nmeth.2477).
        - Input arrays and mask must be cubic and of the same shape.
    """
    # Check that input arrays and mask are cubic and have the same shape
    if array1.shape != array2.shape or array1.shape != mask.shape:
        raise ValueError("Input arrays and mask must have the same shape.")
    if len(set(array1.shape)) != 1:
        raise ValueError("Input arrays and mask must be cubic (all dimensions equal).")

    unmaskedArray1 = np.fft.fftshift(np.fft.fftn(array1))
    unmaskedArray2 = np.fft.fftshift(np.fft.fftn(array2))

    maskedArray1 = np.fft.fftshift(np.fft.fftn(array1*mask))
    maskedArray2 = np.fft.fftshift(np.fft.fftn(array2*mask))

    ndim = array1.ndim
    iSize = array1.shape[0]
    iSize2 = iSize//2

    xv = np.arange(iSize) - iSize2
    XXmeshlist = np.meshgrid(*((xv,)*ndim), copy=False, indexing='ij')

    radii = np.zeros((iSize,)*ndim)

    for XXmesh in XXmeshlist:
        radii += XXmesh**2
    radii = np.sqrt(radii)

    nShells = iSize2 + 1
    shellList = [None]*nShells

    for k in range(nShells):
        shellList[k] = ((radii >= k) * (radii < (k+1))).astype(bool)

    unmaskedCCF = np.real(unmaskedArray1 * unmaskedArray2.conj())
    unmaskedInt1 = np.real(unmaskedArray1 * unmaskedArray1.conj())
    unmaskedInt2 = np.real(unmaskedArray2 * unmaskedArray2.conj())

    maskedCCF = np.real(maskedArray1 * maskedArray2.conj())
    maskedInt1 = np.real(maskedArray1 * maskedArray1.conj())
    maskedInt2 = np.real(maskedArray2 * maskedArray2.conj())

    unmaskedCCF_1D = np.zeros(nShells)
    unmaskedInt1_1D = np.zeros(nShells)
    unmaskedInt2_1D = np.zeros(nShells)

    maskedCCF_1D = np.zeros(nShells)
    maskedInt1_1D = np.zeros(nShells)
    maskedInt2_1D = np.zeros(nShells)

    for k in range(nShells):
        unmaskedCCF_1D[k] = unmaskedCCF[shellList[k]].sum()
        unmaskedInt1_1D[k] = unmaskedInt1[shellList[k]].sum()
        unmaskedInt2_1D[k] = unmaskedInt2[shellList[k]].sum()
        maskedCCF_1D[k] = maskedCCF[shellList[k]].sum()
        maskedInt1_1D[k] = maskedInt1[shellList[k]].sum()
        maskedInt2_1D[k] = maskedInt2[shellList[k]].sum()

# FSC profiles
    unmaskedFSC = unmaskedCCF_1D/np.sqrt(unmaskedInt1_1D * unmaskedInt2_1D)
    maskedFSC = maskedCCF_1D/np.sqrt(maskedInt1_1D * maskedInt2_1D)

# Index for phase rondimization
    rndIdx = np.where(unmaskedFSC < rndThr)[0]

    if len(rndIdx) == 0:
        rndIdx = nShells - 1
    elif rndIdx[0] > 0:
        rndIdx = rndIdx[0]
    else:
        rndIdx = rndIdx[1]

    print(f'Phase randomization at pixel {rndIdx} =\
          {(iSize*pixelSize)/rndIdx:.2g} Å')

    rndMask = radii >= rndIdx
    nRnd = rndMask.sum()
    angleArray1 = np.angle(unmaskedArray1)
    angleArray2 = np.angle(unmaskedArray2)
    absArray1 = np.abs(unmaskedArray1)
    absArray2 = np.abs(unmaskedArray2)

# We repeat the phase randomization to average

    correctedFSCum = np.zeros(nShells)
    phaseRndFSCum = np.zeros(nShells)

    for k in range(nIter):
        rndAngleArray1 = angleArray1.copy()
        rndAngleArray2 = angleArray2.copy()

        rndAngleArray1[rndMask] = np.random.rand(nRnd)*2*np.pi
        rndAngleArray2[rndMask] = np.random.rand(nRnd)*2*np.pi

        rndArray1FT = absArray1 * np.exp(1j*rndAngleArray1)
        rndArray2FT = absArray2 * np.exp(1j*rndAngleArray2)

        rndArray1 = np.fft.ifftn(np.fft.ifftshift(rndArray1FT))
        rndArray2 = np.fft.ifftn(np.fft.ifftshift(rndArray2FT))

        maskedRndArray1FT = np.fft.fftshift(np.fft.fftn(rndArray1*mask))
        maskedRndArray2FT = np.fft.fftshift(np.fft.fftn(rndArray2*mask))

        maskedRndCCF = np.real(maskedRndArray1FT * maskedRndArray2FT.conj())
        maskedRndInt1 = np.real(maskedRndArray1FT * maskedRndArray1FT.conj())
        maskedRndInt2 = np.real(maskedRndArray2FT * maskedRndArray2FT.conj())

        maskedRndCCF_1D = np.zeros(nShells)
        maskedRndInt1_1D = np.zeros(nShells)
        maskedRndInt2_1D = np.zeros(nShells)

        for k in range(nShells):
            maskedRndCCF_1D[k] = maskedRndCCF[shellList[k]].sum()
            maskedRndInt1_1D[k] = maskedRndInt1[shellList[k]].sum()
            maskedRndInt2_1D[k] = maskedRndInt2[shellList[k]].sum()

        maskedRndFSC = maskedRndCCF_1D / \
            np.sqrt(maskedRndInt1_1D * maskedRndInt2_1D)

        # Mask-corrected FSC
        correctedFSCTmp = maskedFSC.copy()
        correctedFSCTmp[rndIdx+rndOffset:] = \
            (maskedFSC[rndIdx+rndOffset:] - maskedRndFSC[rndIdx+rndOffset:]) /\
            (1 - maskedRndFSC[rndIdx+rndOffset:])
        correctedFSCum += correctedFSCTmp
        phaseRndFSCum += maskedRndFSC

    correctedFSC = correctedFSCum/nIter
    phaseRndFSC = phaseRndFSCum/nIter

    out = np.zeros((nShells, 6))
    out[:, 0] = np.arange(nShells) / (iSize*pixelSize)
    out[:, 1] = unmaskedFSC
    out[:, 2] = maskedFSC
    out[:, 3] = phaseRndFSC
    out[:, 4] = correctedFSC
    out[:, 5] = maskedInt1_1D/maskedInt1_1D[0]  # Normalization check

    return out