#!/usr/bin/env python3_artis_tomo
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 2023

@authors: Josue Gomez  & Joaquin Oton
"""
import copy
from pathlib import Path

from artis_sci.utils.parser import argparse

from artis_tomo.correlative.alignment import (alignFluoToMosaic,
                                              alignTomoToMosaic,
                                              alignTomoToFluoVolume,
                                              applyTransformToVolumes)
from artis_tomo.correlative.preprocessing import readFluoVolume
from artis_tomo.correlative.transforms import saveTransforms, loadTransforms
from artis_tomo.tomo.utils import loadTomoFromInput

parser = argparse.ArgumentParser(description='correlation between fluorecense '
                                             '3DSIM and X-ray tomograms '
                                             'volumes.')
required = parser.add_argument_group('Required Arguments')
required.add_argument('-i', '--input', required=True,
                      help='Input could be either IMOD directory, X-ray tilt '
                           'series (aligned or not) or a reconstructed '
                           'tomogram volume. If either X-ray tilt series or '
                           'tomogram volume is provided, the angles file with'
                           ' --tlt should be provided.')
required.add_argument('-m', '--mosaic', required=True,
                      help='X-ray mosaic image.')
required.add_argument('-f', '--fluovol', required=True,
                      help='cryo fluorescence volume.')
required.add_argument('--xrpix', required=True, type=float,
                      help='X-ray input images/volume pixel size')
required.add_argument('--mpix', required=True, type=float,
                      help='X-ray mosaic pixel size')
required.add_argument('--flpix', required=True, type=float,
                      help='XY fluorescence volume pixel size')
required.add_argument('--flzpix', required=True, type=float,
                      help='Z fluorescence volume pixel size')
required.add_argument('-o', '--oroot', required=True,
                      help='Rootname used for output files.')

xrayoptions = parser.add_argument_group('X-ray')
xrayoptions.add_argument('--tlt', metavar='FILE',
                         help='IMOD style tilt angles file')
xrayoptions.add_argument('--flipZ', action='store_true',
                         help='Flip Z axis for X-ray tomogram volume '
                           'reconstruction to match fluorescence ')
xrayoptions.add_argument('--tol', type=float, default=0.5,
                         help='value to mask automatically non-interest '
                              'regions of X-ray image.')
xrayoptions.add_argument('--projs', type=int, default=9,
                         help='Set how number of projections used to 3D align'
                           'fluorescence volume with the reconstructed '
                           'X-ray tomogram volume')
xrayoptions.add_argument('--tomofn', metavar='FILE',
                         help='Reconstructed tomogram volume filename to create'
                           'aligned output. If not provided, input should '
                           'be a volume.')

fluoptions = parser.add_argument_group('Fluorescence')
fluoptions.add_argument('--channel', type=int, default=0,
                      help='Channel for cryosim volume.')
fluoptions.add_argument('--flipFluo', default=False,
                     action='store_true',
                     help='Flip Y axis to correct fluorescence volume '
                          'handiness.')

options = parser.add_argument_group('Correlation options')
options.add_argument('--range', nargs=2, metavar=('MIN', 'MAX'),
                     type=float, default=[400, 1000],
                     help='Range (in nm) to define filtered features to align.')
options.add_argument('--global_step', type=float, default=1.0,
                     help='global searching step to correlate images')
options.add_argument('--perturb',nargs=2, metavar=('STEP', 'THETA'),
                     type=float, default=[1, 3],
                      help='Maximum polar theta angle and angle step for 3D '
                           'angular search.')
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
options.add_argument('--debug', action='store_true',
                     help='Enable debug mode to save intermediate images')
options.add_argument('--tmat', metavar='FILE',
                     help='HDF5 file with pre-computed transformation matrices '
                          '(fluo_to_mosaic, tomocentral_to_mosaic). '
                          'If provided, steps 1 and 2 are skipped.')

resources = parser.add_argument_group('Resources')
resources.add_argument('--gpu', type=int, const=0, nargs='?',
                     help="Use GPU acceleration. Select gpu device number \
                     (default: 0) ")
resources.add_argument('--nproc', type=int, default=4,
                         help='Number of angles processed in  parallel. If '
                         'negative, it uses all available cores')


def clfluoxr3dProgram():
    args = parser.parse_args()

    zoom = args.flzpix / args.flpix
    global_params = (0, 359, args.global_step)
    refine_params = (-5, 5, 0.5)
    noref_params = (0, 0, 1)

    debugDir = args.oroot if args.debug else None

    if args.debug:
        print("DEBUG MODE: Saving intermediate images for inspection.")
        print(f"Input parameters:\n"
              f"  Input: {args.input}\n"
              f"  Mosaic: {args.mosaic}\n"
              f"  Fluorescence volume: {args.fluovol}\n"
              f"  Tomogram: {args.tomofn}\n"
              f"  Output root: {args.oroot}\n"
              f"  Pixel sizes: flPix={args.flpix}, flzPix={args.flzpix}, "
              f"mPix={args.mpix}, xrPix={args.xrpix}\n"
              f"  Channel: {args.channel}, Zoom: {zoom}\n"
              f"  Flip Fluorescence: {args.flipFluo}, Flip Z: {args.flipZ}\n"
              f"  TLT file: {args.tlt}, Tolerance: {args.tol}\n"
              f"  Dim range: {args.range}, Projections: {args.projs}, "
              f"Perturb: {args.perturb}\n"
              f"  Global step: {args.global_step}, Threshold: {args.nproc}, "
              f"Use GPU: {args.gpu}\n"
              f"  Feature preprocessing: {args.feat_preprocess}, "
              f"Feature filter: {args.feat_filter}\n"
              f"  Debug: {args.debug}, Transformation matrix file: {args.tmat}")

    feat_preprocess = None if args.feat_preprocess == 'none' \
        else args.feat_preprocess
    preprocess_kw = ({'q': args.winsorize_q}
                     if feat_preprocess == 'winsorize' else {})

    fluovol = readFluoVolume(args.fluovol, args.channel, args.flipFluo, zoom)

    Path(args.oroot).mkdir(parents=True, exist_ok=True)

    if args.tmat is not None:
        # Steps 1 and 2 depend only on the mosaic and the fluorescence data, so
        # a previous run's result can be reused to go straight to step 3.
        print(f"- Loading transformation matrices from {args.tmat}, "
              "skipping steps 1 and 2...")
        m_fm_mo, m_tomocentral_mo = loadTransforms(args.tmat)
        tomo, _ = loadTomoFromInput(args.input, args.tlt, args.xrpix,
                                    args.flipZ)
        tomo.setAlignTMat(m_tomocentral_mo)
    else:
        print("- Align CryoSIM proj to SXT mosaic...")
        m_fm_mo = alignFluoToMosaic(args.mosaic, fluovol, args.tol, args.gpu,
                                    args.flpix, args.mpix, global_params,
                                    args.range, args.nproc,
                                    preprocess=None, feat_filter='dot',
                                    preprocess_kw=None, debugDir=debugDir)

        print("- Align tomogram proj at 0º to SXT mosaic...")
        tomo, _ = loadTomoFromInput(args.input, args.tlt, args.xrpix,
                                    args.flipZ)
        tomo, _ = alignTomoToMosaic(args.mosaic, tomo, noref_params,
                                    args.xrpix, args.mpix, args.flpix,
                                    args.tol, args.nproc, debugDir=debugDir)

    print("- Align tomogram to Fluorescense 3D image...")

    # Step 3 converts these to pixel units in place, so keep the real-space
    # values for the file written at the end.
    m_fm_mo_realspace = copy.deepcopy(m_fm_mo)
    m_tomocentral_realspace = copy.deepcopy(tomo.getAlignTmat())

    m_trans_tomo = alignTomoToFluoVolume(fluovol, tomo, args.xrpix, args.flpix,
                                         args.perturb, args.projs,
                                         refine_params, args.range, m_fm_mo,
                                         args.gpu, args.nproc,
                                         preprocess=feat_preprocess,
                                         feat_filter=args.feat_filter,
                                         preprocess_kw=preprocess_kw,
                                         debugDir=debugDir,
                                         verbose=args.debug)

    applyTransformToVolumes(args.oroot, fluovol, args.tomofn, args.flpix,
                            args.xrpix, m_trans_tomo)

    # Everything in the file is in real-space units; step 3 returns pixels.
    m_trans_tomo_realspace = copy.deepcopy(m_trans_tomo)
    m_trans_tomo_realspace.matrix[:, 0:3, 3] = (
        m_trans_tomo_realspace.matrix[:, 0:3, 3] * args.flpix)

    saveTransforms(args.oroot, m_fm_mo_realspace, m_tomocentral_realspace,
                   m_trans_tomo_realspace)

    print("Matrix to apply: ", m_trans_tomo)
