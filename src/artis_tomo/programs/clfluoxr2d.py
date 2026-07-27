#!/usr/bin/env python3_artis_tomo
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 2023

@authors: Josue Gomez  & Joaquin Oton
"""
from artis_sci.io import imageIO as iio
from artis_sci.utils.parser import argparse

from artis_tomo.correlative.alignment import alignFluoImageToMosaic
from artis_tomo.correlative.preprocessing import readFluoImage

parser = argparse.ArgumentParser(description='align fluorecense and'
                                             ' x-ray mosaic images.')

required = parser.add_argument_group('Required Arguments')

required.add_argument('-i', '--input', required=True,
                      help='X-ray mosaic image.')
required.add_argument('-f', '--fluoimg', required=True,
                      help='Either cryo fluorescence volume or image.')
required.add_argument('--mpix', required=True, type=float,
                      help='X-ray mosaic pixel size')
required.add_argument('--flpix', required=True, type=float,
                      help='XY fluorescence volume pixel size')
required.add_argument('-o', '--oroot', required=True,
                      help='Rootname used for output files.')

xrayoptions = parser.add_argument_group('X-ray')
xrayoptions.add_argument('--tol', type=float, default=0.5,
                         help='value to mask automatically non-interest '
                              'regions of X-ray image.')

fluoptions = parser.add_argument_group('Fluorescence')
fluoptions.add_argument('--flipFluo', default=False,
                     action='store_true',
                     help='Flip Y axis to correct fluorescence volume '
                          'handiness.')
fluoptions.add_argument('--channel', type=int, default=0,
                      help='Channel for cryosim volume if is an input.')
fluoptions.add_argument('--flzpix', type=float, default=1.0 ,
                        help='Z fluorescence volume pixel size if fluorescence '
                             'volume if is an input.')

options = parser.add_argument_group('Correlation options')
options.add_argument('--range', nargs=2, metavar=('MIN', 'MAX'),
                     type=float, default=[400, 1000],
                     help='Range (in nm) to define filtered features to align.')
options.add_argument('--global_step', type=float, default=1.0,
                     help='global searching step to correlate images')
options.add_argument('--feat_preprocess',
                     choices=['none', 'winsorize', 'local_norm'],
                     default='none',
                     help='Intensity preprocessing applied before the feature '
                          'size filter to reduce outlier dominance (e.g., lipid '
                          'droplets). "winsorize": clip at quantile --winsorize_q. '
                          '"local_norm": z-score normalise within r_max window. '
                          'Default: none.')
options.add_argument('--winsorize_q', type=float, default=0.98,
                     help='Upper quantile for winsorization, only used with '
                          '--feat_preprocess=winsorize. Default: 0.98.')
options.add_argument('--feat_filter',
                     choices=['dot', 'dog', 'butterworth', 'frangi'],
                     default='dot',
                     help='Feature size filter method. "dot": difference of '
                          'top-hats (morphological, default). "dog": difference '
                          'of Gaussians (linear). "butterworth": Fourier band-pass '
                          'with steep roll-off. "frangi": Hessian-based shape '
                          'filter, best for tubular structures (mitochondria).')

resources = parser.add_argument_group('Resources')
resources.add_argument('--gpu', type=int, const=0, nargs='?',
                     help="Use GPU acceleration. Select gpu device number "
                          "(default: 0) ")
resources.add_argument('--nproc', type=int, default=4,
                         help='Number of angles processed in  parallel. If '
                         'negative, it uses all available cores')


def clfluoxr2dProgram():
    args = parser.parse_args()

    zoom = args.flzpix / args.flpix
    global_params = (0, 359, args.global_step)

    feat_preprocess = None if args.feat_preprocess == 'none' \
        else args.feat_preprocess
    preprocess_kw = ({'q': args.winsorize_q}
                     if feat_preprocess == 'winsorize' else {})

    fluoimg = readFluoImage(args.fluoimg, args.channel, args.flipFluo, zoom)

    # Step 1: align the CryoSIM projection to the SXT mosaic.
    m_fm_mo, mosaic, fluoCorrelated = alignFluoImageToMosaic(
        args.input, fluoimg, args.tol, args.gpu, args.flpix, args.mpix,
        global_params, args.range, args.nproc, preprocess=feat_preprocess,
        feat_filter=args.feat_filter, preprocess_kw=preprocess_kw)

    iio.writeImage(mosaic, args.oroot, "mosaic_scaled")
    iio.writeImage(fluoCorrelated, args.oroot, "fluo_correlated")
