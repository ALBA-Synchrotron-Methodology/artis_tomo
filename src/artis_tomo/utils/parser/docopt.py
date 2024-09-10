#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom docopt parser

@author: joton
"""


import sys
from os.path import basename
from docopt import docopt as docoptBase
from ...version import __version__


def docopt(doc, *args, **kwargs):
    """Use custom docopt with package version."""
    progName = basename(sys.argv[0])
    if 'version' not in kwargs:
        kwargs['version'] = __version__
    return docoptBase(doc.format(progName=progName), *args, **kwargs)
