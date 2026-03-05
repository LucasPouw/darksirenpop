'''
Storing GLOBALS that are accessed from multiple files.
'''

import os, sys
from astropy.cosmology import Planck15
from astropy.constants import c
import numpy as np

POST_DIR = '/home/lucas/Documents/PhD/fagn_posteriors'
PLOT_DIR = '/home/lucas/Documents/PhD/darksirenpop/plots'

DEFAULT_H0 = 67.9
DEFAULT_OM0 = 0.3065
COSMO = Planck15.clone(H0=DEFAULT_H0, Om0=DEFAULT_OM0)
SPEED_OF_LIGHT_KMS = c.to('km/s').value

CALC_LOGLLH_AT_N_POINTS = 1000
LOG_LLH_X_AX = np.linspace(0.0001, 0.9999, CALC_LOGLLH_AT_N_POINTS)

COLORS = ['orangered', 'navy', 'teal', 'goldenrod', 'hotpink', 'indigo', 'crimson']
