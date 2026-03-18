'''
Controls and prepares all mock data for the inference of f_agn.
'''

import numpy as np
from tqdm import tqdm
from astropy.coordinates import SkyCoord
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import h5py
from default_globals import *
import astropy.units as u
import os
import matplotlib.pyplot as plt
from scipy import stats
import healpy as hp
from redshift_utils import *
import pandas as pd
from scipy.interpolate import interp1d
import time
import glob
from priors import BBH_powerlaw_gaussian


################################################### GLOBALS ###################################################

VERBOSE = False

THREADING = True  # Parallellized over data realizations
N_WORKERS = 16

REAL_DATA = False  # Real or mock flag
USE_SKYMAPS = True  # Instead of posterior samples
SOURCE_FRAME_MASS_PRIOR = None  # BBH_powerlaw_gaussian()

N_REALIZATIONS = 200  # Number of realizations
BATCH = 150  # Number of GWs per realization
TRUE_FAGNS = np.tile(0.5, N_REALIZATIONS)

DIRECTORY_ID = 'all'
ALL_TRUE_SOURCES = np.genfromtxt(f'/home/lucas/Documents/PhD/true_r_theta_phi_{DIRECTORY_ID}.txt', delimiter=',')
SKYMAP_DIR = f"./skymaps_{DIRECTORY_ID}/"
SAMPLES_DIR = f"./posterior_samples_{DIRECTORY_ID}/"

AGN_DIST_DIR = '/home/lucas/Documents/PhD/darksirenpop/agn_distribution'
CATALOG_PATH = "/home/lucas/Documents/PhD/agn_data/Quaia_z15.csv"

SKYMAP_CL = 0.999

ZMIN = 1e-4  # Some buffer for astropy's lowest possible value
ZMAX = 1.5   # Maximum true redshift for GWs, such that p_rate(z > ZMAX) = 0. Needs to correspond to the input data, it is not enforced automatically!
AGN_ZMAX = 10  # Maximum true redshift for AGN
AGN_ZCUT = 1  # Redshift cut of the AGN catalog, defines the redshift above which f_c(z)=0

QLF = 'kulkarni'  # QLF \in [kulkarni, shenA, shenB]
AGN_ZPRIOR = f'46.5_{QLF}'  # Valid: 'positive_redshift', 'uniform_comoving_volume', '44.5_<QLF>', '45.0_<QLF>', '45.5_<QLF>', '46.0_<QLF>', '46.5_<QLF>'

LUM_THRESH = 'zero_upto_cut'  # Valid: '44.5', '45.0', '45.5', '46.0', '46.5' (V25 completeness bins), 'zero' (complete catalog), 'zero_upto_cut' (complete catalog up to a redshift cut), 'inf' (empty catalog)

MASK_GALACTIC_PLANE = True
CMAP_PATH = "./completeness_map.fits"  # Healpix map with 1 if the sky pixel is within the survey footprint and 0 everywhere else.
PLOT_CMAP = False  # Plot the survey footprint map
CMAP_NSIDE = 64

ADD_NAGN_TO_CAT = int(3.5e5)  # Lower bound, since we prioritize simulating a catalog following the input distribution. This may require more than this number of AGN.
ASSUME_PERFECT_REDSHIFT = False  # If true, neglect AGN redshift error
AGN_ZERROR = 'quaia'  # Valid: Float, 'quaia' or False

CORRECT_TIME_DILATION = True  # Adds 1/(1+z) weighting to GW redshift sampling.
MERGER_RATE = 'madau'  # For the ALT hypothesis! Either 'uniform' or 'madau'.

LINAX = True  # If Z_INTEGRAL_AX is a geomspace instead of linspace, make False
AGN_ZPRIOR_NORM_AX = np.linspace(ZMIN, AGN_ZMAX, 1024+1)

#################################################################################################################


if not AGN_ZERROR and not ASSUME_PERFECT_REDSHIFT:
    sys.exit('Stop trying to break my code.')


if (LUM_THRESH in ['44.5', '45.0', '45.5', '46.0', '46.5']) & (AGN_ZCUT < 1.3125):
    raise ValueError(f"V25 completeness bins require AGN_ZCUT to be higher than V25's highest z-bin, which is 1.3125. Got: {AGN_ZCUT}")


if THREADING:
    os.environ["OMP_NUM_THREADS"] = "1"  # Just in case


REALIZED_FAGNS = np.random.binomial(BATCH, TRUE_FAGNS) / BATCH  # Observed f_agn fluctuates around the true value
N_TRUE_FAGNS = len(TRUE_FAGNS)


COMDIST_MIN = COSMO.comoving_distance(ZMIN).value
COMDIST_MAX = COSMO.comoving_distance(ZMAX).value
AGN_COMDIST_MAX = COSMO.comoving_distance(AGN_ZMAX).value


# Quaia completeness (rows = bins, cols = thresholds)
THRESHOLD_MAP = {"46.5": 0, "46.0": 1, "45.5": 2, "45.0": 3, "44.5": 4}
Z_EDGES = np.array([0.0000, 0.1875, 0.3750, 0.5625, 0.7500, 0.9375, 1.1250, 1.3125, AGN_ZCUT, AGN_ZMAX])
QUAIA_C_VALS = np.array([
                    [0.000, 0.000, 0.229, 0.945, 0.718],
                    [1.000, 1.000, 1.000, 1.000, 0.781],
                    [1.000, 1.000, 1.000, 1.000, 0.408],
                    [1.000, 0.891, 1.000, 0.681, 0.211],
                    [1.000, 1.000, 0.994, 0.429, 0.138],
                    [1.000, 1.000, 0.837, 0.258, 0.085],
                    [0.927, 0.940, 0.576, 0.179, 0.060],
                    [1.000, 1.000, 0.482, 0.155, 0.053],
                    [0., 0., 0., 0., 0.]
                ])

def v25_selection_function(z):
    completeness_zvals = np.array(QUAIA_C_VALS[:, THRESHOLD_MAP[LUM_THRESH]])
    bin_idx = np.digitize(z, Z_EDGES) - 1
    bin_idx[bin_idx == len(completeness_zvals)] = len(completeness_zvals) - 1
    return completeness_zvals[bin_idx.astype(np.int32)]


if AGN_ZPRIOR[:4] != LUM_THRESH:
    print(f'WARNING: You are performing an analysis assuming log10(Lbol) >= {LUM_THRESH}, while the AGN redshift posteriors in your catalog may have an inconsistent prior: {AGN_ZPRIOR}')


if AGN_ZERROR == 'quaia':
    quaia_errors = pd.read_csv(CATALOG_PATH)["redshift_quaia_err"]  # Load into memory and sample later


if MERGER_RATE == 'madau':
    MERGER_RATE_EVOLUTION = merger_rate_madau_dickinson
    MERGER_RATE_KWARGS = {}
elif MERGER_RATE == 'uniform':
    MERGER_RATE_EVOLUTION = merger_rate_uniform
    MERGER_RATE_KWARGS = {}


if REAL_DATA:
    FAGN_POSTERIOR_FNAME = f'p26_post_realdata_{REAL_DATA}_useskymap_{USE_SKYMAPS}_rate_{MERGER_RATE}_timedil_{CORRECT_TIME_DILATION}_agnZprior_{AGN_ZPRIOR}_Lthresh_{LUM_THRESH}_perfz_{ASSUME_PERFECT_REDSHIFT}_GPmask_{MASK_GALACTIC_PLANE}_CL_{SKYMAP_CL}_gwZmax_{ZMAX}_agnZcut_{AGN_ZCUT}'
else:
    FAGN_POSTERIOR_FNAME = f'p26_post_realdata_{REAL_DATA}_useskymap_{USE_SKYMAPS}_rate_{MERGER_RATE}_timedil_{CORRECT_TIME_DILATION}_agnZprior_{AGN_ZPRIOR}_Lthresh_{LUM_THRESH}_perfz_{ASSUME_PERFECT_REDSHIFT}_GPmask_{MASK_GALACTIC_PLANE}_addAGN_{ADD_NAGN_TO_CAT}_nreal_{N_REALIZATIONS}_batch_{BATCH}_CL_{SKYMAP_CL}_agnZerr_{AGN_ZERROR}_gwZmax_{ZMAX}_agnZcut_{AGN_ZCUT}'
inp = None
while inp not in ['y', 'Y', 'yes', 'Yes', 'n', 'N', 'no', 'No']:
    inp = input('Have you changed the posterior filename? (y/n)')
if inp not in ['y', 'Y', 'yes', 'Yes']:
    sys.exit('Please adjust the posterior filename accordingly.') 


def get_agn_zprior():
    """
    'positive_redshift', 'uniform_comoving_volume', '44.5', '45.0', '45.5', '46.0', '46.5'
    """

    if AGN_ZPRIOR == 'uniform_comoving_volume':
        return lambda z: uniform_comoving_prior(z)
    
    elif str(AGN_ZPRIOR[:4]) in ['44.5', '45.0', '45.5', '46.0', '46.5']:
        filename = f'{AGN_DIST_DIR}/agn_redshift_pdf_{AGN_ZPRIOR}.npy'
        if VERBOSE:
            print(f'Loading AGN redshift distribution calculated from QLF from file: {filename}')
        z, n = np.load(filename)
        return interp1d(z, n, bounds_error=False, fill_value=0)
    
    elif AGN_ZPRIOR == 'positive_redshift':  # Equivalent to a uniform-in-redshift prior. Redundant since the Z_INTEGRAL_AX is >0, but this way we enforce a conscious decision on the prior.
        return lambda z: z_cut(-z, zcut=0)
    
    else:
        sys.exit(f'AGN redshift prior not recognized: {AGN_ZPRIOR}. \nExiting...')
    

AGN_ZPRIOR_FUNCTION = get_agn_zprior()


def get_z_integral_ax(at_least_N=1, npoints_min=512):
    '''This function assumes the smallest scale in the problem is the AGN redshift posterior.'''
    if AGN_ZERROR == 'quaia':
        smallest_error = np.min(quaia_errors)
    else:
        smallest_error = AGN_ZERROR

    if smallest_error == 0:  # This only happens when neglecting the z-error, use 512+1 points as default
        return np.linspace(ZMIN, ZMAX, npoints_min + 1)
    else:
        npoints = int(2**np.ceil(np.log2(at_least_N * (ZMAX - ZMIN) / smallest_error)))  # Enforces at least N points within 1 sigma of the AGN posteriors
        print(f'Requiring at least {npoints} in redshift integral axis to capture all AGN information for smallest error: {smallest_error}.')
        npoints = max(npoints, npoints_min)
        return np.linspace(ZMIN, ZMAX, npoints + 1)
    

Z_INTEGRAL_AX = get_z_integral_ax()  # Sets the resolution of the redshift prior, should capture all information of AGN posteriors, see Gray et al. 2022, 2023

inp = None
while inp not in ['y', 'Y', 'yes', 'Yes', 'n', 'N', 'no', 'No']:
    inp = input(f'Redshift integral has {len(Z_INTEGRAL_AX)} points between z={ZMIN} and z={ZMAX}. Do you accept this? (y/n)')
if inp not in ['y', 'Y', 'yes', 'Yes']:
    sys.exit('Please adjust the resolution accordingly.') 


def get_observed_redshift_from_rcom(agn_rcom):
    true_agn_redshift = fast_z_at_value(COSMO.comoving_distance, agn_rcom * u.Mpc)

    # Sample from Quaia or make all errors the same (which requires AGN_ZERROR to be a float)?
    if AGN_ZERROR == 'quaia':
        if VERBOSE:
            print('Sampling AGN redshift errors from Quaia')
        agn_redshift_err = np.random.choice(quaia_errors, size=len(agn_rcom))
    else:
        agn_redshift_err = np.tile(AGN_ZERROR, len(agn_rcom))

    # Perfect measurement or not?
    if not AGN_ZERROR:
        if VERBOSE:
            print('No AGN redshift errors')
        obs_agn_redshift = true_agn_redshift
    else:
        if VERBOSE:
            print('Scattering AGN according to their redshift errors')
        obs_agn_redshift = stats.truncnorm.rvs(size=len(agn_rcom), 
                                                a=(ZMIN - true_agn_redshift) / agn_redshift_err, 
                                                b=(np.inf - true_agn_redshift) / agn_redshift_err, 
                                                loc=true_agn_redshift, 
                                                scale=agn_redshift_err)
    if VERBOSE:
        print(f'Complete catalog has {np.sum(obs_agn_redshift > AGN_ZMAX)} REALIZED AGN redshifts above AGN_ZMAX. These are thrown away.')
        print(f'Complete catalog has {np.sum(obs_agn_redshift < ZMAX)} REALIZED AGN redshifts below GW_ZMAX.')
    return obs_agn_redshift, agn_redshift_err


def make_redshift_selection(obs_agn_redshift):
    '''
    AGN catalogs can be redshift-incomplete. This function returns a masking array that does the selection based on the data realization of AGN redshifts.
    '''
    if LUM_THRESH == 'zero':  # No redshift selection
        z_selection_function = lambda z: 1
        redshift_incomplete_mask = np.ones_like(obs_agn_redshift, dtype=bool)

    elif LUM_THRESH == 'zero_upto_cut':
        z_selection_function = lambda z: z_cut(z, zcut=AGN_ZCUT)
        redshift_incomplete_mask = obs_agn_redshift < AGN_ZCUT
    
    elif LUM_THRESH == 'inf':
        z_selection_function = lambda z: 0
        redshift_incomplete_mask = np.zeros_like(obs_agn_redshift, dtype=bool)

    else:
        z_selection_function = v25_selection_function

        c_per_zbin = np.array(QUAIA_C_VALS[:, THRESHOLD_MAP[LUM_THRESH]])
        redshift_incomplete_mask = np.zeros_like(obs_agn_redshift, dtype=bool)
        for i, c_in_bin in enumerate(c_per_zbin):
            z_low, z_high = Z_EDGES[i], Z_EDGES[i + 1]
            agn_in_bin = np.where((obs_agn_redshift > z_low) & (obs_agn_redshift < z_high))[0]
            keep_these = np.random.choice(np.arange(len(agn_in_bin)), size=round(c_in_bin * len(agn_in_bin)), replace=False)
            redshift_incomplete_mask[agn_in_bin[keep_these]] = True

    return z_selection_function, redshift_incomplete_mask


def make_latitude_selection(agn_ra, agn_dec, obs_agn_rlum):
    '''
    Latitude completeness map is for indicating which sky area is surveyed (c=1) and which is not (c=0).
    '''
    npix = hp.nside2npix(CMAP_NSIDE)
    theta, phi = hp.pix2ang(CMAP_NSIDE, np.arange(npix), nest=True)
    map_coord = SkyCoord(phi * u.rad, (np.pi * 0.5 - theta) * u.rad)
    map_b = map_coord.galactic.b.degree
    outside_galactic_plane_pix = np.logical_or(map_b > 10, map_b < -10)

    completeness_map = np.tile(1., npix)
    b = SkyCoord(agn_ra * u.rad, agn_dec * u.rad, obs_agn_rlum * u.Mpc).galactic.b.degree
    if MASK_GALACTIC_PLANE:
        latitude_mask = np.logical_or(b > 10, b < -10)
        completeness_map[~outside_galactic_plane_pix] = 0
    else:
        latitude_mask = np.ones_like(b, dtype=bool)
    
    if PLOT_CMAP:
        hp.mollview(
                    completeness_map,
                    nest=True,
                    coord="G",               # plot in Galactic coords
                    title="Mask: 1 outside |b|<=10°, 0 inside",
                    cmap="coolwarm",
                    min=0, max=1
                )
        hp.graticule()
        plt.savefig(f'{PLOT_DIR}/cmap.pdf', bbox_inches='tight')
        plt.close()
    return latitude_mask, completeness_map


def make_incomplete_catalog(agn_ra, agn_dec, obs_agn_rlum, obs_agn_redshift):
    z_selection_function, redshift_incomplete_mask = make_redshift_selection(obs_agn_redshift)  # Making a redshift-incomplete catalog
    latitude_mask, completeness_map = make_latitude_selection(agn_ra, agn_dec, obs_agn_rlum)
    incomplete_catalog_mask = (latitude_mask & redshift_incomplete_mask)
    if VERBOSE:
        print(f'Observed {np.sum(incomplete_catalog_mask)} AGN from realizations, of which {np.sum(obs_agn_redshift[incomplete_catalog_mask] < ZMAX)} below GW_ZMAX. Average completeness below GW_ZMAX: {np.sum(obs_agn_redshift[incomplete_catalog_mask] < ZMAX) / np.sum(obs_agn_redshift < ZMAX):.5f}')
    return incomplete_catalog_mask, z_selection_function, completeness_map


def compute_and_save_posteriors_hdf5(filename, all_agn_z, all_agn_z_err):
    '''
    AGN redshift posteriors are modelled as truncnorms on [0, inf) with a uniform-in-comoving-volume redshift prior.
    The posteriors are then evaluated on Z_INTEGRAL_AX, which is what is necessary for the crossmatch.
    '''
    n_norm = 100
    n_agn = len(all_agn_z)
    n_z = len(Z_INTEGRAL_AX)  # Only need to save the posterior evaluated at this axis
    chunk_size = int(1e6 / n_z)
    with h5py.File(filename, "w") as f:
        dset = f.create_dataset("agn_redshift_posteriors", shape=(n_agn, n_z), dtype=np.float64)

        if VERBOSE:
            iterchunks = tqdm( range(0, n_agn, chunk_size) )
        else:
            iterchunks = range(0, n_agn, chunk_size)

        for start in iterchunks:
            end = min(start + chunk_size, n_agn)
            z_chunk = all_agn_z[start:end]
            zerr_chunk = all_agn_z_err[start:end]
            mu = z_chunk[:, None]
            sigma = zerr_chunk[:, None]

            # Build per-AGN normalization axes
            t = np.linspace(0, 1, n_norm)[None, :]
            z_norm_ax = (mu - 10*sigma) + t * (20*sigma)

            likelihood = lambda z: stats.truncnorm.pdf(
                z,
                a=(ZMIN - mu) / sigma,
                b=(np.inf - mu) / sigma,
                loc=mu,
                scale=sigma
            )
            posteriors_unnorm = lambda z: likelihood(z) * z_cut(z, zcut=AGN_ZMAX) * AGN_ZPRIOR_FUNCTION(z)
            z_norms = np.trapezoid(posteriors_unnorm(z_norm_ax), z_norm_ax, axis=1)
            posteriors = posteriors_unnorm(Z_INTEGRAL_AX) / z_norms[:, None]

            dset[start:end, :] = posteriors
    if VERBOSE:
        print(f"All AGN posteriors written to {filename}")
    return


def get_agn_posteriors(fagn_idx, obs_agn_redshift, agn_redshift_err, label, replace_old_file=True):
    '''
    To save computation time, the AGN posteriors are calculated and evaluated on the z-integral axis once and kept in memory.
    '''

    if ASSUME_PERFECT_REDSHIFT:
        return np.empty(1), 1
    
    else:
        posterior_path = f'./precompute_posteriors/agn_posteriors_precompute_gwZmax_{ZMAX}_prior_{AGN_ZPRIOR}_{fagn_idx}_{label}.hdf5'

        if not os.path.exists(posterior_path):
            compute_and_save_posteriors_hdf5(posterior_path, obs_agn_redshift, agn_redshift_err)
        elif os.path.exists(posterior_path) & replace_old_file:
            os.remove(posterior_path)
            compute_and_save_posteriors_hdf5(posterior_path, obs_agn_redshift, agn_redshift_err)

        # Keep ~few GB in memory, this is typically faster than reading random slices
        with h5py.File(posterior_path, "r") as f:
            agn_posterior_dset = f["agn_redshift_posteriors"][()]
        sum_of_posteriors = np.sum(agn_posterior_dset, axis=0)  # Sum of posteriors is required to normalize the in-catalog population prior

        return agn_posterior_dset, sum_of_posteriors
