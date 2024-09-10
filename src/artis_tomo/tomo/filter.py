#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter tools for tomography

@author: joton
"""

import numpy as np
from ..math import fft, framework as fw
from artis_tomo.image import (frame as fr,
                            transformation as tr,
                            filter as ft)

def bpExactFilter(array, nt, dt=1):

    np = fw.frame.from_array(array)


    dims = array.ndim

    if dims == 2:
        array = array[None, :, :]

    nt, nr, nc = array.shape

    nfc = nc//2 + 1

    cr = int(np.ceil(1/np.arctan(np.deg2rad(dt))))  # Crowthers radius

    cr = nfc
    filtFS = np.empty((nfc), complex)
    filtFS[:cr] = np.linspace(0, nt-1, cr)
    filtFS[cr:] = nt - 1
    filtFS = (filtFS + 1)
    filtFS /= filtFS[0]

    filtered = np.empty_like(array)

    with fft.set_backend_from_array(array):
        for k in range(nt):
            aft = fft.rfft(array[k], axis=1)
            filtered[k] = fft.irfft(aft*filtFS, axis=1)

    if dims == 2:
        filtered = np.squeeze(filtered)

    return filtered


def bpFilter(array, filterName='ramp', nThreads=-1):

    xnp = fw.frame.from_array(array)

    dims = array.ndim

    if dims == 2:
        array = array[None, :, :]

    nt, nr, nc = array.shape

    filtFS = xnp.to_device(getBPFilter(nc, filterName))
    filtered = xnp.empty_like(array)

    with xnp.fft_backend(nThreads) as fft:
        for k in range(nt):
            aft = fft.rfft(array[k], axis=1)
            filtered[k] = fft.irfft(aft*filtFS, n=nc, axis=1)

    if dims == 2:
        filtered = xnp.squeeze(filtered)

    return filtered


def getBPFilter(size, filterName='ramp'):

    center = size//2
    pos = np.roll(np.arange(size) - center, center)
    fpos = np.arange(center+1)/size
    fRS = np.zeros(size)
    fRS[0] = 0.25
    fRS[1::2] = -1 / (np.pi * pos)[1::2] ** 2

    # Computing the ramp filter from the fourier transform of its
    # frequency domain representation lessens artifacts and removes a
    # small bias as explained in AC Kak, M Slaney, "Principles of Computerized
    # Tomographic Imaging", IEEE Press 1988, Chap 3. Equation 61

    fFS = 2 * np.fft.rfft(fRS)  # ramp filter

    if filterName == "ramp":
        pass
    elif filterName == "shepp":  # shepp-logan
        # Start from first element to avoid divide by zero
        omega = np.pi * fpos[1:]
        fFS[1:] *= np.sin(omega) / omega
    elif filterName == "cosine":
        freq = fpos*np.pi
        cosine_filter = np.cos(freq)
        fFS *= cosine_filter
    elif filterName == "hamming":
        fFS *= np.hamming(size)[center::-1]
    elif filterName == "hanning":
        fFS *= np.hanning(size)[center::-1]
    elif filterName is None:
        fFS[:] = 1
    else:
        raise Exception(f'ERROR: Unknown filter name {filterName}.')

    fFS /= fFS[0]

    return fFS

def deconv_wiener(stack: np.ndarray, dx: float, psf: np.ndarray, psf_dx: float,
        kw: float, pad: int = 20, fc: float = -1):
    """Apply Wiener filter to the given stack of images.

    Parameters
    ----------
    stack : np.ndarray
        Stack to deconvolve.
    dx : float
        Pixel size of the stack.
    psf : np.ndarray
        Consensed PSF to used in the filter.
    psf_dx : float
        Pixel size of the PSF.
    kw : float
        Wiener parameter.
    pad : int, optional
        Padding to add to the images and PSF, by default 20
    fc : float, optional
        Cut off to apply while filtering experimental MTFs. If -1 this filter
        is not applied, by default -1.

    Returns
    -------
    np.ndarray
        Array of the deconvolved stack.
    """
    # Extract information and prepare input stack of images by padding them
    stack_size = stack.shape
    num_dims = len(stack_size)
    num_imgs = (num_dims == 2) * 1 + (num_dims == 3) * stack_size[0]
    extended_size = np.max(stack_size) + pad * 2
    padded_size = extended_size * np.array([1, 1])
    stack, _, _ = fr.padArrayCentered(stack, (
    stack_size[0], padded_size[0], padded_size[1]))
    x_center = extended_size // 2

    # Rescale PSF
    psf = tr.rescaleFourier(psf, scale=dx / psf_dx)

    # Expand and PSF
    psf, _, _ = fr.padCropArrayCentered(psf, stack.shape[1:])
    cos_radial_mask = ft.maskRaisedCosineRadial(shape=psf.shape,
                                                radius=x_center - pad - 5,
                                                pad=20)
    psf *= cos_radial_mask

    # H_exp = Experimental Transfer Function
    H_exp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
    H_exp = H_exp / (np.abs(H_exp[x_center, x_center]))

    # Correcting for frequencies greater than 1 (just in case of coherence
    # plateu). If some element is bigger than 1, then we correct the freqs.
    positive_gradients_positions = np.nonzero(np.abs(H_exp) > 1)
    if np.size(positive_gradients_positions) != 0:
        xv = np.arange(
            extended_size) - x_center  # Spatial coordinates along dimension x

        # Radial distances corresponding to each point in the spatial grid
        [x_grid, y_grid] = np.meshgrid(xv, xv)
        radial_grid = np.sqrt(x_grid ** 2 + y_grid ** 2)

        # Determine minimum and maximum radial distance for correction
        min_radial_dist = np.ceil(
            np.max(radial_grid[positive_gradients_positions]))

        H_exp_low_freq = H_exp.copy()

        # Identify positions where radial distance is less than min_radial_dist
        # then normalize these positions
        radial_min_poss = np.nonzero(radial_grid < min_radial_dist)
        H_exp_low_freq[radial_min_poss] = H_exp_low_freq[radial_min_poss] / (
            np.abs(H_exp_low_freq[radial_min_poss]))

        # TODO: Technical debt: Change it to work with the hermitic fourier
        tmp_pad = pad
        if min_radial_dist < pad:
            tmp_pad = min_radial_dist
        low_freq_mask = ft.maskRaisedCosineRadial(H_exp_low_freq.shape,
                                                  radius=min_radial_dist,
                                                  pad=int(tmp_pad))
        H_exp = H_exp_low_freq * low_freq_mask + H_exp * (1 - low_freq_mask)

    # Filter experimental MTFs
    if fc > 0:
        dfx = 1.0 / float(
            dx) / extended_size  # Spatial freq. resolution along dim x
        fourier_mask = ft.maskRaisedCosineRadial(shape=H_exp.shape, radius=fc,
                                                 dx=dfx, pad=pad)
        H_exp = H_exp * fourier_mask

    # Wiener filters
    wiener_filter = np.conjugate(H_exp) / (H_exp * np.conjugate(H_exp) + kw)
    wiener_filter = wiener_filter / np.abs(wiener_filter[x_center, x_center])
    wiener_filter = np.fft.ifftshift(wiener_filter)

    # Apply filter and deconvolute
    deconv_stack = np.zeros((np.shape(stack)), dtype=complex)
    for i_img in range(num_imgs):
        # Compute FFT of stack[i_img]
        fft_i_img = np.fft.fft2(np.fft.ifftshift(stack[i_img]))

        # Apply Wiener filter and recover real space image
        freq_filtered_img = fft_i_img * wiener_filter
        deconv_stack[i_img] = np.fft.fftshift(np.fft.ifft2(freq_filtered_img))

    deconv_stack = np.real(fr.cropArrayCentered(deconv_stack, stack_size))

    return deconv_stack