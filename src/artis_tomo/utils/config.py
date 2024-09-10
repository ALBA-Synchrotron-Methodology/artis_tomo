"""
Read custom config files

"""

import os
import os.path as path
import yaml
from .dict import RecursiveDict as rdict

configFnGlob = os.getenv('ARS_CFG_GLOBAL')
configFnUser = os.getenv('ARS_CFG_USER')
configFnUserDft = path.join(path.expanduser("~"), '.config', 'artis.conf')

# Global configuration read
if configFnGlob and path.isfile(configFnGlob):
    _paramsGlob = rdict(
        yaml.load(open(configFnGlob, "r"), Loader=yaml.FullLoader))
else:
    _paramsGlob = rdict()

# Explicit user config file overwrites the default one
if not configFnUser:
    configFnUser = configFnUserDft

if configFnUser and path.isfile(configFnUser):
    _paramsUsr = rdict(yaml.load(open(configFnUser, "r"), Loader=yaml.FullLoader))
else:
    _paramsUsr = rdict()

params = _paramsGlob.merge_dict(_paramsUsr)