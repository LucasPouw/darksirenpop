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
from scipy.integrate import romb
from scipy import stats
import healpy as hp
from redshift_utils import z_cut, merger_rate_madau, merger_rate_uniform, merger_rate, uniform_comoving_prior, fast_z_at_value
import pandas as pd
from scipy.interpolate import interp1d
import time
import glob


os.environ["OMP_NUM_THREADS"] = "1"
N_WORKERS = 16

VERBOSE = True

N_REALIZATIONS = 100
BATCH = 100#len(np.array(glob.glob('/home/lucas/Documents/PhD/mockstats/' + 'gw*.dat'))) 
TRUE_FAGNS = np.tile(0.5, N_REALIZATIONS)
REALIZED_FAGNS = np.random.binomial(BATCH, TRUE_FAGNS) / BATCH  # Observed f_agn fluctuates around the true value
N_TRUE_FAGNS = len(TRUE_FAGNS)

CMAP_PATH = "./completeness_map.fits"
DIRECTORY_ID = 'all'
AGN_DIST_DIR = '/home/lucas/Documents/PhD/darksirenpop/agn_distribution'
SKYMAP_DIR = f"./skymaps_{DIRECTORY_ID}/"
SKYMAP_CL = 0.999

AGN_ZMAX = 7.  # Maximum true redshift for AGN
AGN_ZCUT = 1.5  # Redshift cut of the AGN catalog, defines the redshift above which f_c(z)=0
AGN_ZPRIOR = 'uniform_comoving_volume'  # Valid: 'positive_redshift', 'uniform_comoving_volume', '44.5', '45.0', '45.5', '46.0', '46.5'

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
LUM_THRESH = '46.0'  #'45.5'  # str or False for complete catalog
if AGN_ZPRIOR != LUM_THRESH:
    print(f'WARNING: You are performing an analysis assuming log10(Lbol) >= {LUM_THRESH}, while the AGN redshift posteriors in your catalog may have an inconsistent prior: {AGN_ZPRIOR}')

REDSHIFT_SELECTION_FUNCTION = 'binned'  # 'binned' or 'continuous'
MASK_GALACTIC_PLANE = True
PLOT_CMAP = False
CMAP_NSIDE = 64

ADD_NAGN_TO_CAT = int(1e2)  # Lower bound, since we prioritize a uniform map, which may require more AGN
ASSUME_PERFECT_REDSHIFT = False
AGN_ZERROR = 'quaia'
if AGN_ZERROR == 'quaia':
    quaia_errors = pd.read_csv("/home/lucas/Documents/PhD/Quaia_z15.csv")["redshift_quaia_err"]  # Load into memory and sample later

REAL_DATA = True
if REAL_DATA:
    FAGN_POSTERIOR_FNAME = f'p26_likelihood_posteriors_realdata_{REAL_DATA}_agnZprior_{AGN_ZPRIOR}_lumthresh_{LUM_THRESH}_perfectz_{ASSUME_PERFECT_REDSHIFT}_galplanemask_{MASK_GALACTIC_PLANE}_skymapCL_{SKYMAP_CL}'
else:
    FAGN_POSTERIOR_FNAME = f'p26_likelihood_posteriors_realdata_{REAL_DATA}_agnZprior_{AGN_ZPRIOR}_lumthresh_{LUM_THRESH}_perfectz_{ASSUME_PERFECT_REDSHIFT}_galplanemask_{MASK_GALACTIC_PLANE}_addAGN_{ADD_NAGN_TO_CAT}_nrealizations_{N_REALIZATIONS}_batch_{BATCH}_skymapCL_{SKYMAP_CL}_agnZerror_{AGN_ZERROR}'
inp = None
while inp not in ['y', 'Y', 'yes', 'Yes', 'n', 'N', 'no', 'No']:
    inp = input('Have you changed the posterior filename? (y/n)')
if inp not in ['y', 'Y', 'yes', 'Yes']:
    sys.exit('Please adjust the posterior filename accordingly.') 

MERGER_RATE_EVOLUTION = merger_rate_uniform
MERGER_RATE_KWARGS = {}

ZMIN = 1e-4  # Some buffer for astropy's lowest possible value
ZMAX = 1.5   # Maximum true redshift for GWs, such that p_rate(z > ZMAX) = 0
# assert AGN_ZMAX >= ZMAX, 'Need AGN at least as deep as GWs can go, otherwise the population prior is not evaluated on the correct axis.'  # Don't think this is true anymore
COMDIST_MIN = COSMO.comoving_distance(ZMIN).value
COMDIST_MAX = COSMO.comoving_distance(ZMAX).value
AGN_COMDIST_MAX = COSMO.comoving_distance(AGN_ZMAX).value

Z_INTEGRAL_AX = np.linspace(ZMIN, ZMAX, int(512)+1)  # Sets the resolution of the redshift prior, should capture all information of AGN posteriors, see Gray et al. 2022, 2023
LINAX = True  # If geomspace instead of linspace, make False


def get_agn_zprior():
    """
    'positive_redshift', 'uniform_comoving_volume', '44.5', '45.0', '45.5', '46.0', '46.5'
    """

    if AGN_ZPRIOR == 'uniform_comoving_volume':
        return lambda z: uniform_comoving_prior(z)
    
    elif str(AGN_ZPRIOR) in ['44.5', '45.0', '45.5', '46.0', '46.5']:
        filename = f'{AGN_DIST_DIR}/agn_redshift_pdf_{AGN_ZPRIOR}.npy'
        if VERBOSE:
            print(f'Loading AGN redshift distribution calculated from QLF from file: {filename}')
        z, n = np.load(filename)
        return interp1d(z, n, bounds_error=False, fill_value=0)
    
    elif AGN_ZPRIOR == 'positive_redshift':  # Equivalent to a uniform-in-redshift prior. Redundant since the Z_INTEGRAL_AX is >0, but this way we enforce a conscious decision on the prior.
        return lambda z: z_cut(-z, zcut=0)
    
    else:
        sys.exit(f'AGN redshift prior not recognized: {AGN_ZPRIOR}. \nExiting...')


def get_agn_z_posterior_norm_ax(at_least_N=10):
    if AGN_ZERROR == 'quaia':
        smallest_error = np.min(quaia_errors)
    else:
        smallest_error = AGN_ZERROR
    npoints = int(2**np.ceil(np.log2(at_least_N * (AGN_ZMAX - ZMIN) / smallest_error)))  # Enforces at least N points within 1 sigma in normalization of AGN posteriors
    return np.linspace(ZMIN, AGN_ZMAX, npoints + 1)


AGN_ZPRIOR_FUNCTION = get_agn_zprior()

if not ASSUME_PERFECT_REDSHIFT:
    AGN_POSTERIOR_NORM_AX = get_agn_z_posterior_norm_ax()


def get_observed_redshift_from_rcom(agn_rcom):
    true_agn_redshift = fast_z_at_value(COSMO.comoving_distance, agn_rcom * u.Mpc)

    # Sample from Quaia or make all errors the same?
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
    AGN catalogs can be redshift-incomplete. This function returns a masking array that does the selection based on the observed AGN redshifts.
    '''
    if not LUM_THRESH:
        c_per_zbin = np.tile(1, len(Z_EDGES) - 1)
        redshift_incomplete_mask = np.ones_like(obs_agn_redshift, dtype=bool)
    else:
        c_per_zbin = np.array(QUAIA_C_VALS[:, THRESHOLD_MAP[LUM_THRESH]])
        redshift_incomplete_mask = np.zeros_like(obs_agn_redshift, dtype=bool)
        for i, c_in_bin in enumerate(c_per_zbin):
            z_low, z_high = Z_EDGES[i], Z_EDGES[i + 1]
            agn_in_bin = np.where((obs_agn_redshift > z_low) & (obs_agn_redshift < z_high))[0]
            keep_these = np.random.choice(np.arange(len(agn_in_bin)), size=round(c_in_bin * len(agn_in_bin)), replace=False)
            redshift_incomplete_mask[agn_in_bin[keep_these]] = True
    if VERBOSE:
        print(c_per_zbin, 'Z-COMPLETENESS')
    return c_per_zbin, redshift_incomplete_mask


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
        plt.figure()
        hp.mollview(
                    completeness_map,
                    nest=True,
                    coord="G",               # plot in Galactic coords
                    title="Mask: 1 outside |b|<=10°, 0 inside",
                    cmap="coolwarm",
                    min=0, max=1
                )
        hp.graticule()
        plt.savefig('cmap.pdf', bbox_inches='tight')
        plt.close()
    return latitude_mask, completeness_map


def make_incomplete_catalog(agn_ra, agn_dec, obs_agn_rlum, obs_agn_redshift):
    c_per_zbin, redshift_incomplete_mask = make_redshift_selection(obs_agn_redshift)  # Making a redshift-incomplete catalog
    latitude_mask, completeness_map = make_latitude_selection(agn_ra, agn_dec, obs_agn_rlum)
    if ASSUME_PERFECT_REDSHIFT:
        incomplete_catalog_mask = (latitude_mask & redshift_incomplete_mask & (obs_agn_redshift < ZMAX))  # Only AGN below ZMAX will 
        if VERBOSE:
            print(f'Observed {np.sum(incomplete_catalog_mask)} AGN from realizations, all below GW_ZMAX since we assume a perfect AGN redshift so those above do not contribute. Average completeness below GW_ZMAX: {np.sum(obs_agn_redshift[incomplete_catalog_mask] < ZMAX) / np.sum(obs_agn_redshift < ZMAX):.5f}')
    else:
        incomplete_catalog_mask = (latitude_mask & redshift_incomplete_mask)  # To emulate V25's selection of Quaia, include  & (obs_agn_redshift < ZMAX). TODO: P26 will have to test & (obs_agn_redshift < AGN_ZMAX)
        if VERBOSE:
            print(f'Observed {np.sum(incomplete_catalog_mask)} AGN from realizations, of which {np.sum(obs_agn_redshift[incomplete_catalog_mask] < ZMAX)} below GW_ZMAX. Average completeness below GW_ZMAX: {np.sum(obs_agn_redshift[incomplete_catalog_mask] < ZMAX) / np.sum(obs_agn_redshift < ZMAX):.5f}')
    return incomplete_catalog_mask, c_per_zbin, completeness_map


def compute_and_save_posteriors_hdf5(filename, all_agn_z, all_agn_z_err):
    '''
    AGN redshift posteriors are modelled as truncnorms on [0, inf) with a uniform-in-comoving-volume redshift prior.
    The posteriors are normalized on agn_posterior_norm_ax, which goes up to AGN_ZMAX.
    The posteriors are then evaluated on Z_INTEGRAL_AX, which is what is necessary for the crossmatch.
    '''

    maxdiff = np.max(np.diff(AGN_POSTERIOR_NORM_AX))
    thresh = np.min(all_agn_z_err) / 10
    assert maxdiff < thresh, f'AGN normalization array is too coarse to capture AGN distribution fully. Got {maxdiff:.3e}, need {thresh:.3e}.'

    n_agn = len(all_agn_z)
    n_z = len(Z_INTEGRAL_AX)  # Only need to save the posterior evaluated at this axis
    chunk_size = int(1e6 / n_z)
    dx = np.diff(AGN_POSTERIOR_NORM_AX)[0]  # For romb integration on linear axis
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

            likelihood = lambda z: stats.truncnorm.pdf(
                z,
                a=(ZMIN - z_chunk[:, None]) / zerr_chunk[:, None],
                b=(np.inf - z_chunk[:, None]) / zerr_chunk[:, None],
                loc=z_chunk[:, None],
                scale=zerr_chunk[:, None]
            )

            agn_posteriors_unnorm = lambda z: likelihood(z) * z_cut(z, zcut=AGN_ZMAX) * AGN_ZPRIOR_FUNCTION(z)
            z_norms = romb(agn_posteriors_unnorm(AGN_POSTERIOR_NORM_AX), dx=dx)
            posteriors = agn_posteriors_unnorm(Z_INTEGRAL_AX) / z_norms[:, None]  # Save the posterior evaluated at the relevant axis

            dset[start:end, :] = posteriors
    if VERBOSE:
        print(f"All AGN posteriors written to {filename}")
    return


def get_agn_posteriors_and_zprior_normalization(fagn_idx, obs_agn_redshift, agn_redshift_err, label, replace_old_file=True):
    '''
    To save computation time, the AGN posteriors are calculated and evaluated on the z-integral axis once and kept in memory.

    Also returns the normalization of the LOS zprior, which involves the sum of all posteriors.
    '''

    if ASSUME_PERFECT_REDSHIFT:
        agn_posterior_dset = np.empty(1)
        redshift_population_prior_normalization = np.sum(merger_rate(obs_agn_redshift[obs_agn_redshift < ZMAX], MERGER_RATE_EVOLUTION, **MERGER_RATE_KWARGS))  # TODO: check if this is correct when varying the merger rate evolution
        sum_of_posteriors = 1
    else:
        posterior_path = f'./precompute_posteriors/agn_posteriors_precompute_prior_{AGN_ZPRIOR}_{fagn_idx}_{label}.hdf5'

        if not os.path.exists(posterior_path):
            compute_and_save_posteriors_hdf5(posterior_path, obs_agn_redshift, agn_redshift_err)
        elif os.path.exists(posterior_path) & replace_old_file:
            os.remove(posterior_path)
            compute_and_save_posteriors_hdf5(posterior_path, obs_agn_redshift, agn_redshift_err)

        # Keep ~few GB in memory, this is typically faster than reading random slices
        with h5py.File(posterior_path, "r") as f:
            agn_posterior_dset = f["agn_redshift_posteriors"][()]
        sum_of_posteriors = np.sum(agn_posterior_dset, axis=0)  # Sum of posteriors is required to normalize the in-catalog population prior

        if LINAX:
            dz = np.diff(Z_INTEGRAL_AX)[0]
            jacobian = 1
        else:
            dz = np.diff(np.log10(Z_INTEGRAL_AX))[0]
            jacobian = Z_INTEGRAL_AX * np.log(10)
        
        redshift_population_prior_normalization = romb(sum_of_posteriors * merger_rate(Z_INTEGRAL_AX, MERGER_RATE_EVOLUTION, **MERGER_RATE_KWARGS) * z_cut(Z_INTEGRAL_AX, zcut=ZMAX) * jacobian, dx=dz)

    return agn_posterior_dset, redshift_population_prior_normalization, sum_of_posteriors
