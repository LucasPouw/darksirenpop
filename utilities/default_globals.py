'''
Storing GLOBALS that are used in multiple notebooks

I'm forcing my data structure onto you, the user, with how I coded my paths with f-strings here. I hope you like it :) 
You can change it of course, the goal is to ONLY have to change paths in here if anything changes
'''

from astropy.cosmology import Planck15
from astropy.constants import c
import numpy as np

DEFAULT_H0 = 67.9
DEFAULT_OM0 = 0.3065
COSMO = Planck15.clone(H0=DEFAULT_H0, Om0=DEFAULT_OM0)
SPEED_OF_LIGHT_KMS = c.to('km/s').value

CALC_LOGLLH_AT_N_POINTS = 1000
LOG_LLH_X_AX = np.linspace(0.0001, 0.9999, CALC_LOGLLH_AT_N_POINTS)

COLORS = ['orangered', 'navy', 'teal', 'goldenrod', 'hotpink', 'indigo', 'crimson']


### PATHS TO DIRECTORIES ###
REPOSITORY_DIR = '/home/lucas/Documents/PhD/darksirenpop'
GENERATED_DATA_DIR = '/home/lucas/Documents/PhD/generated_data'
GW_DATA_DIR = '/home/lucas/Documents/PhD/gw_data'
EM_DATA_DIR = '/home/lucas/Documents/PhD/agn_data'

# Subdirectories of GW_DATA_DIR
POPSUMMARY_DIR = f'{GW_DATA_DIR}/popsummary_files'
GWTC2P1 = f'{GW_DATA_DIR}/PEsamples_gwtc2p1'
GWTC3P0 = f'{GW_DATA_DIR}/PEsamples_gwtc3p0'
GWTC4P0 = f'{GW_DATA_DIR}/PEsamples_gwtc4p0'
GWTC5P0 = f'{GW_DATA_DIR}/PEsamples_gwtc5p0'

# Subdirectories of GENERATED_DATA_DIR
AGN_DIST_DIR = f'{GENERATED_DATA_DIR}/em'
JSON_DIR = f'{GENERATED_DATA_DIR}/jsons'
FAGN_POST_DIR = f'{GENERATED_DATA_DIR}/fagn_posteriors'
MOCK_DATA_DIR = f'{GENERATED_DATA_DIR}/mock_gws'
PDET_EM_DIR = f"{GENERATED_DATA_DIR}/compare_PdetEM_methods"

REWEIGHT_GWTC5_DIR = f'{GENERATED_DATA_DIR}/gw/reweighted-gwtc5'
REWEIGHT_GWTC5_SAMPLES = f'{REWEIGHT_GWTC5_DIR}/samples'
REWEIGHT_GWTC5_SKYMAPS = f'{REWEIGHT_GWTC5_DIR}/skymaps'
REWEIGHT_GWTC5_SKYMAP_STATS = f'{REWEIGHT_GWTC5_DIR}/skymap_stats'
REWEIGHT_GWTC5_SKYMAP_EVALS = f'{REWEIGHT_GWTC5_DIR}/real_skymaps_evaluated/'  # Directory to store all evaluated redshift posteriors from real data

# Subdirectories of REPOSITORY_DIR
PLOT_DIR = f'{REPOSITORY_DIR}/plots'
MOCK_DIR = f"{REPOSITORY_DIR}/mock"
PDET_DIR = f"{MOCK_DIR}/pdet"

### PATHS TO FILES ###

# GW data
LVK_HYPERPOSTERIOR_PATH = f"{POPSUMMARY_DIR}/gwtc5_updated_madau_dickinson_mmax_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_redshift_MadauDickinsonRedshift_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_popsummary_result.h5"
INJECTIONS_PATH = f'{GW_DATA_DIR}/injection_samples_essick/mixture-semi_o1_o2-real_o3_o4a_o4b-polar_spins_20260410130052UTC-clipped.hdf'  # zenodo_get 19500052
GWOSC_SUMMARY = f'{GW_DATA_DIR}/events.csv'  # csv file downloaded from gwosc.org containing SNR and FAR for all events
V90_CDF_PATH = f'{GENERATED_DATA_DIR}/v90_cdf_LVK.npy'

# EM data
QUAIA_PATH = f"{AGN_DIST_DIR}/quaia_zleq3_withlumcorr.csv"
QUAIA_FITS_PATH = f'{EM_DATA_DIR}/quaia_G20.5.fits'
DR16Q_PATH = f'{EM_DATA_DIR}/dr16q_prop_May01_2024.fits.gz'

# .json files
PE_JSON_PATH_LVK_PATH = f'{REWEIGHT_GWTC5_DIR}/real_PEsamples_gwtc5.json'  # .json file that stores paths to PE samples published by LVK
PE_JSON_PATH_REWEIGHT_PATH = f'{REWEIGHT_GWTC5_DIR}/real_PEsamples_reweight_gwtc5.json'  # .json file that stores paths to reweighted PE samples
SKYMAP_JSON_PATH = f'{REWEIGHT_GWTC5_DIR}/real_skymaps_reweight_gwtc5.json'  # .json that contains paths to skymaps
SKYMAP_EVALS_JSON_PATH = f'{REWEIGHT_GWTC5_DIR}/real_skymaps_evaluated.json'  # .json that contains paths to redshift posteriors
SKYMAP_CW_EVALS_JSON_PATH = f'{REWEIGHT_GWTC5_DIR}/real_cw_skymaps_evaluated.json'  # .json that contains paths to completeness-weighted redshift posteriors
