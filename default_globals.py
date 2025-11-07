'''
Storing GLOBALS that are accessed from multiple files.
'''

import os, sys
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c
import numpy as np


DEFAULT_H0 = 67.9
DEFAULT_OM0 = 0.3065
COSMO = FlatLambdaCDM(H0=DEFAULT_H0, Om0=DEFAULT_OM0)
SPEED_OF_LIGHT_KMS = c.to('km/s').value

CALC_LOGLLH_AT_N_POINTS = 1000
LOG_LLH_X_AX = np.linspace(0.0001, 0.9999, CALC_LOGLLH_AT_N_POINTS)

COLORS = ['orangered', 'navy', 'teal', 'goldenrod', 'indigo', 'crimson']

# DEFAULT_SKY_AREA_LOW = 1e2
# DEFAULT_SKY_AREA_HIGH = 1e4
# DEFAULT_LUMDIST_RELERR = 0.1

# DEFAULT_N_POSTERIOR_SAMPLES = int(5e4)
# DEFAULT_N_CPU = 1
# DEFAULT_POSTERIOR_OUTDIR = os.path.join(sys.path[0], "output/default_posteriors")
# RCOM_SCALE = 10