import numpy as np
from utils import uniform_shell_sampler, make_nice_plots, fast_z_at_value, spherical2cartesian, cartesian2spherical
from tqdm import tqdm
from default_arguments import DEFAULT_COSMOLOGY
import astropy.units as u
from scipy.integrate import quad, romb
from astropy.constants import c
from concurrent.futures import as_completed, ThreadPoolExecutor
from multiprocessing import cpu_count
# from scipy.interpolate import CubicSpline
import traceback
import time
import sys, os
from priors import *
from scipy.stats import vonmises_fisher
from scipy.optimize import root_scalar
from scipy.special import erf
import h5py
from astropy.table import Table
import shutil
import glob
import scipy.stats as stats

make_nice_plots()

#### NOTES ####

"""
AGN can scatter to negative comoving distances, though this is relatively rare: ~500 / 100k at sigma = 0.2 -- update: checked with truncnorm AGN errors between [0, inf) and AGN catalog up to 2xCOMDIST_MAX and no change in bias observed.
Are LOS zpriors correctly calculated given the AGN posteriors with prior that includes p_rate(z > zmax) = 0 ?

Are my posterior samples correct? No induced prior (Jacobian) that needs to be accounted for?
"""

########################### INPUT PARAMETERS #################################

COSMO = DEFAULT_COSMOLOGY
SPEED_OF_LIGHT_KMS = c.to('km/s').value

MAKE_SKYMAPS = True  # Only implemented in combination with USE_ONLY_3DLOC = True
SKYMAP_DIR = './skymaps_moc_500_xd'
POST_SAMPS_DIR = './posterior_samples_moc_500_xd'
CAT_DIR = './catalogs_moc_500_xd'

USE_ONLY_SKY = False
USE_ONLY_3DLOC = True
USE_ONLY_REDSHIFT = False
USE_ONLY_MASS = False
USE_MASS_AND_REDSHIFT = False
ASSUME_PERFECT_REDSHIFT_MEASUREMENT = False

SKY_AREA_LOW = 100
SKY_AREA_HIGH = 10000

SKYMAP_CL = 0.9  # ALWAYS CHECK, MAKE 1 UNLESS YOU WANT TO CHANGE THE LIKELIHOOD FUNCTION

N_POSTERIOR_SAMPLES = int(1e4)

# AGN_ZERROR = float(0.05)  # Absolute error for now, same for all AGN, posterior = Gaussian * dVdz prior / norm TODO: Verify that this likelihood is fine for Quaia
AGN_ZERROR = int(0)

AGN_MASS_MODEL = PrimaryMass_gaussian(mu_g=25)  # PrimaryMass_gaussian(mu_g=60, sigma_g=4)
ALT_MASS_MODEL = PrimaryMass_powerlaw_gaussian()  # PrimaryMass_gaussian(mu_g=30)  # PrimaryMass_powerlaw_gaussian(lambda_peak=0, alpha=1.4)
AGNMODELSTR = 'G'
ALTMODELSTR = 'G'
PRIMARY_MASS_ERROR = 10  # Solar masses
PRIMARY_MASS_INTEGRAL_AXIS = np.geomspace(1e-4, 200, 8192*2+1)  # TODO: Make this not hard-coded, it currently only works because of the specific chosen populations

ZMIN = 1e-4
ZMAX = 1.5  # p_rate(z > ZMAX) = 0
GW_DIST_ERR = 0.2  # Relative error

LUMDIST_MIN = COSMO.luminosity_distance(ZMIN).value
LUMDIST_MAX = COSMO.luminosity_distance(ZMAX).value

COMDIST_MIN = COSMO.comoving_distance(ZMIN).value
COMDIST_MAX = COSMO.comoving_distance(ZMAX).value  # Maximum comoving distance in Mpc

### Redshift integral resolutions strongly influence accuracy of results, this is the first place to look when encountering biases!!! ###
# LOS_ZPRIOR_Z_ARRAY = np.geomspace(ZMIN, ZMAX, 8193)  # Sets the resolution of the redshift prior, should capture all information of AGN posteriors, see Gray et al. 2022, 2023
# if AGN_ZERROR:
#     assert np.max( np.diff(LOS_ZPRIOR_Z_ARRAY)[0] ) < AGN_ZERROR, 'LOS zprior resolution is too coarse to capture AGN distribution fully.'
# S_ALT_Z_INTEGRAL_AX = np.geomspace(ZMIN, ZMAX, 4097)
# ZNORM_ROMB_AXIS = np.geomspace(ZMIN / 10, 10 * ZMAX, 513)  # Normalizing the GW redshift posterior is done numerically, this is the range on which that happens.

LOS_ZPRIOR_Z_ARRAY = np.geomspace(ZMIN, ZMAX, 8192*2+1)  # Sets the resolution of the redshift prior, should capture all information of AGN posteriors, see Gray et al. 2022, 2023
if AGN_ZERROR:
    assert np.max( np.diff(LOS_ZPRIOR_Z_ARRAY)[0] ) < AGN_ZERROR, 'LOS zprior resolution is too coarse to capture AGN distribution fully.'
S_ALT_Z_INTEGRAL_AX = np.geomspace(ZMIN, ZMAX, 8192*2+1)
ZNORM_ROMB_AXIS = np.geomspace(ZMIN / 10, 10 * ZMAX, 8192*2+1)  # Normalizing the GW redshift posterior is done numerically, this is the range on which that happens.

VOLUME = 4 / 3 * np.pi * COMDIST_MAX**3
AGN_NUMDENS = 500 / VOLUME
BATCH = int(500)
N_TRIALS = 1
MAX_N_FAGNS = 6
CALC_LOGLLH_AT_N_POINTS = 1000
# GW_CHUNK = 100  # Optimized for my own system - vectorize operations for this many GWs 
# LLH_CHUNK = 10  # Optimized for my own system - vectorize operations for this many values of f_agn

NAGN = int( np.ceil(AGN_NUMDENS * VOLUME) )

# Panicky factor to generate AGN in a larger volume to check boundary issues in USE_ONLY_3DLOC case
AAA = 1
AGN_NUMDENS /= AAA**3
print(f'\nPANICK FACTOR = {AAA}\n')

if USE_ONLY_MASS:
    fname = f'posteriors_SKYMAP_CL_{SKYMAP_CL}_BATCH_{BATCH}_PRIMARY_MASS_ERROR_{PRIMARY_MASS_ERROR}_AGNMODEL_{AGNMODELSTR}_ALTMODEL_{ALTMODELSTR}'
elif USE_ONLY_REDSHIFT:
    fname = f'posteriors_SKYMAP_CL_{SKYMAP_CL}_AGNZERROR_{AGN_ZERROR}_ASSUMENOERR_{ASSUME_PERFECT_REDSHIFT_MEASUREMENT}_BATCH_{BATCH}_GWDISTERR_{GW_DIST_ERR}_ZMAX_{ZMAX}_NAGN_{NAGN}'
elif USE_MASS_AND_REDSHIFT:
    fname = f'posteriors_SKYMAP_CL_{SKYMAP_CL}_AGNZERROR_{AGN_ZERROR}_ASSUMENOERR_{ASSUME_PERFECT_REDSHIFT_MEASUREMENT}_BATCH_{BATCH}_GWDISTERR_{GW_DIST_ERR}_ZMAX_{ZMAX}_NAGN_{NAGN}_PRIMARY_MASS_ERROR_{PRIMARY_MASS_ERROR}_AGNMODEL_{AGNMODELSTR}_ALTMODEL_{ALTMODELSTR}'
elif USE_ONLY_SKY:
    fname = f'posteriors_SKYMAP_CL_{SKYMAP_CL}_SKY_AREA_LOW_{SKY_AREA_LOW}_SKY_AREA_HIGH_{SKY_AREA_LOW}_NAGN_{NAGN}_BATCH_{BATCH}'
elif USE_ONLY_3DLOC:
    fname = f'posteriors_SKYMAP_CL_{SKYMAP_CL}_AGNZERROR_{AGN_ZERROR}_NAGN_{NAGN}_BATCH_{BATCH}'

# NCPU = cpu_count()
NCPU = 1

assert NAGN >= BATCH, f'Every AGN-origin GW must come from a unique AGN. Got {NAGN} AGN and {BATCH} AGN-origin GWs.'
print(f'#AGN is rounded from {AGN_NUMDENS * VOLUME} to {NAGN}, giving a number density of {NAGN / VOLUME:.3e}. Target was {AGN_NUMDENS:.3e}.')
NGW_ALT = BATCH
NGW_AGN = BATCH
N_TRUE_FAGNS = min(BATCH + 1, MAX_N_FAGNS)  # Cannot create more f_agn values than BATCH+1 and don't want to generate more than MAX_N_FAGNS

LOG_LLH_X_AX = np.linspace(0.0001, 0.9999, CALC_LOGLLH_AT_N_POINTS)

# USE_GW_SELECTION_EFFECTS = False
# LUMDIST_THRESH_GW = LUMDIST_MAX  # Luminosity distance threshold in Mpc

USE_N_AGN_EVENTS = np.arange(0, BATCH + 1, int(BATCH / (N_TRUE_FAGNS - 1)), dtype=np.int32)
TRUE_FAGNS = USE_N_AGN_EVENTS / BATCH

####################################################################


####################### DATA GENERATION #########################

def gaussian_unnorm(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2)


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)


def dVdz_unnorm(z):
    '''Assuming flat LCDM'''
    Omega_m = COSMO.Om0
    Omega_Lambda = 1 - Omega_m
    E_of_z = np.sqrt((1 + z)**3 * Omega_m + Omega_Lambda)
    com_vol = ((1 + z) * COSMO.angular_diameter_distance(z).value)**2 / E_of_z
    return com_vol


func = lambda z: dVdz_unnorm(z)
DVDZ_NORM_POP = quad(func, ZMIN, ZMAX)[0]  # Normalize up to ZMAX, since p_rate(z > ZMAX) = 0. TODO: Could maybe change to Romberg later
# DVDZ_NORM_GW = quad(func, ZMIN, ZMAX * 10)[0]


def dVdz_populationprior(z):
    z = np.atleast_1d(z)
    result = np.zeros_like(z)
    below_thresh = (z < ZMAX) & (z > ZMIN)
    result[below_thresh] = dVdz_unnorm(z[below_thresh]) / DVDZ_NORM_POP
    return result


# def dVdz_GWprior(z):
#     z = np.atleast_1d(z)
#     result = np.zeros_like(z)
#     below_thresh = (z < ZMAX * 10) & (z > ZMIN)
#     result[below_thresh] = dVdz_unnorm(z[below_thresh]) / DVDZ_NORM_GW
#     return result


def unnormed_lumdist_distribution(d, dobs, sigma):
    return np.exp(-(dobs/d - 1)**2 / (2 * sigma**2)) / d


def unnormed_redshift_distribution(z, dobs, sigma):
    dl = COSMO.luminosity_distance(z).value
    H_z = COSMO.H(z).value  # H(z) in km/s/Mpc
    chi_z = (1 + z) * dl
    dDL_dz = chi_z + (1 + z) * (SPEED_OF_LIGHT_KMS / H_z)
    return unnormed_lumdist_distribution(dl, dobs, sigma) / dDL_dz


def gw_redshift_posterior(z, dobs):
    # TODO: add dVdz_prior to redshift posterior. Did single naive test with 30% GW error, seemed to cause upwards bias.
    # NOTE: Could be integration error due to wrong integration bounds?. Oh hm or it is because of the z in [0, 1.5] range, while the posterior should extend past 1.5...
    # NOTE: Doesn't seem to work either, let it go for now

    func2norm = lambda x: unnormed_redshift_distribution(x, dobs, sigma=GW_DIST_ERR) * x * np.log(10)  # * dVdz_GWprior(x)
    norm = romb(y=func2norm(ZNORM_ROMB_AXIS), dx=np.diff(np.log10(ZNORM_ROMB_AXIS))[0])
    try:
        return unnormed_redshift_distribution(z, dobs, sigma=GW_DIST_ERR) / norm[:,np.newaxis]  # * dVdz_GWprior(z)
    except IndexError:  # I hate this, but it happens when plotting diagnostics for a single dobs
        return unnormed_redshift_distribution(z, dobs, sigma=GW_DIST_ERR) / norm  # * dVdz_GWprior(z)


def process_gw_redshift(rlum_obs, agn_z_obs):
    """
    Calculates the evidence for hypothesis x in {agn, alt} as S_x = int p(z|d) p_x(z) dz = int p(z|d) p_x(z) z log(10) d(log10 z)
    """

    if ASSUME_PERFECT_REDSHIFT_MEASUREMENT:
        S_agn = np.sum( gw_redshift_posterior(z=agn_z_obs, dobs=rlum_obs[:,np.newaxis]), axis=1 ) / NAGN

    else:
        agn_posteriors = gaussian_unnorm(x=LOS_ZPRIOR_Z_ARRAY, mu=agn_z_obs[:,np.newaxis], sigma=AGN_ZERROR) * dVdz_populationprior(LOS_ZPRIOR_Z_ARRAY)
        norm = romb(y=agn_posteriors * LOS_ZPRIOR_Z_ARRAY * np.log(10),
                    dx=np.diff(np.log10(LOS_ZPRIOR_Z_ARRAY))[0])
        
        # LOS_zprior = np.sum(agn_posteriors / norm[:,np.newaxis], axis=0) / NAGN

        LOS_zprior = np.zeros_like(LOS_ZPRIOR_Z_ARRAY)  # p_agn(z)
        for z in agn_z_obs:  # Not vectorized, because NAGN * len(LOS_ZPRIOR_Z_ARRAY) could be large
            agn_posterior = gaussian_unnorm(x=LOS_ZPRIOR_Z_ARRAY, mu=z, sigma=AGN_ZERROR) * dVdz_populationprior(LOS_ZPRIOR_Z_ARRAY)
            norm = romb(y=agn_posterior * LOS_ZPRIOR_Z_ARRAY * np.log(10),
                        dx=np.diff(np.log10(LOS_ZPRIOR_Z_ARRAY))[0])
            LOS_zprior += agn_posterior / norm  # Assume Gaussian redshift posteriors
        LOS_zprior /= NAGN  # Normalize

        # import matplotlib.pyplot as plt
        # for rlum in rlum_obs:
        #     rlum = COSMO.luminosity_distance(0.55).value
        #     fig, pdf_ax = plt.subplots(figsize=(8,6))
        #     pop_ax = pdf_ax.twinx()
        #     plt.plot(LOS_ZPRIOR_Z_ARRAY, gw_redshift_posterior(z=LOS_ZPRIOR_Z_ARRAY, dobs=rlum) * LOS_zprior, color='orangered', label=r'$p(z|d)p_{\rm agn}(z)$', linewidth=3, linestyle='dashed')
        #     plt.plot(LOS_ZPRIOR_Z_ARRAY, gw_redshift_posterior(z=LOS_ZPRIOR_Z_ARRAY, dobs=rlum) * dVdz_populationprior(LOS_ZPRIOR_Z_ARRAY), color='teal', label=r'$p(z|d)p_{\rm alt}(z)$', linewidth=3, linestyle='dashed')
        #     pdf_ax.plot(LOS_ZPRIOR_Z_ARRAY, gw_redshift_posterior(z=LOS_ZPRIOR_Z_ARRAY, dobs=rlum), color='goldenrod', label=r'$p(z|d)$', linewidth=3)
        #     pdf_ax.plot(LOS_ZPRIOR_Z_ARRAY, LOS_zprior, label=r'$p_{\rm agn}(z)$', color='crimson', linewidth=3)
        #     pdf_ax.plot(LOS_ZPRIOR_Z_ARRAY, dVdz_populationprior(LOS_ZPRIOR_Z_ARRAY), label=r'$p_{\rm alt}(z)$', color='indigo', linewidth=3)
            
        #     pdf_ax.set_xlabel('Redshift')
        #     pdf_ax.set_ylabel('Probability density')
        #     pop_ax.set_ylabel(r'$p(z|d)p_{\rm pop}(z)$')

        #     pdf_ax.set_xlim(0, 1.5)
        #     pdf_ax.set_ylim(0, 5)
        #     pop_ax.set_ylim(0, 5)

        #     lines1, labels1 = pdf_ax.get_legend_handles_labels()
        #     lines2, labels2 = pop_ax.get_legend_handles_labels()
        #     pdf_ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=1)
            
        #     # plt.title(f'Sagn={p:.3f}, Salt={p_alt_integral(rlum):.3f}')

        #     plt.savefig('redshift_populations.pdf', bbox_inches='tight')
        #     time.sleep(15)
        #     plt.close()
        
        S_agn = romb(gw_redshift_posterior(z=LOS_ZPRIOR_Z_ARRAY, dobs=rlum_obs[:,np.newaxis]) * LOS_zprior * LOS_ZPRIOR_Z_ARRAY * np.log(10), 
                     dx=np.diff(np.log10(LOS_ZPRIOR_Z_ARRAY))[0])
    
    S_alt = romb(y=gw_redshift_posterior(S_ALT_Z_INTEGRAL_AX, rlum_obs[:,np.newaxis]) * dVdz_populationprior(S_ALT_Z_INTEGRAL_AX) * S_ALT_Z_INTEGRAL_AX * np.log(10),
                 dx=np.diff(np.log10(S_ALT_Z_INTEGRAL_AX))[0])
    
    return S_agn, S_alt


def gw_mass_posterior(m, mobs):
    "Currently only primary mass posterior, to be extended to primary+secondary mass joint posterior"
    return gaussian(m, mu=mobs, sigma=PRIMARY_MASS_ERROR)  # No prior necessary


def process_gw_mass(mobs):

    S_agn = romb(y=gw_mass_posterior(PRIMARY_MASS_INTEGRAL_AXIS, mobs[:,np.newaxis]) * AGN_MASS_MODEL.joint_prob(PRIMARY_MASS_INTEGRAL_AXIS) * PRIMARY_MASS_INTEGRAL_AXIS * np.log(10), 
                 dx=np.diff(np.log10(PRIMARY_MASS_INTEGRAL_AXIS))[0])

    S_alt = romb(y=gw_mass_posterior(PRIMARY_MASS_INTEGRAL_AXIS, mobs[:,np.newaxis]) * ALT_MASS_MODEL.joint_prob(PRIMARY_MASS_INTEGRAL_AXIS) * PRIMARY_MASS_INTEGRAL_AXIS * np.log(10),
                 dx=np.diff(np.log10(PRIMARY_MASS_INTEGRAL_AXIS))[0])
    
    return S_agn, S_alt


def get_observed_primary_mass(n_agn_events):
    '''This data generation is independent of redshift'''
    true_primary_mass_gw_agn = AGN_MASS_MODEL.sample(n_agn_events)[0]
    true_primary_mass_gw_alt = ALT_MASS_MODEL.sample(BATCH - n_agn_events)[0]
    true_primary_mass_gw = np.hstack((true_primary_mass_gw_agn, true_primary_mass_gw_alt))

    obs_primary_mass_gw = np.random.normal(loc=true_primary_mass_gw, scale=PRIMARY_MASS_ERROR, size=BATCH)
    return obs_primary_mass_gw


def get_observed_gw_rlum_agn_z(n_agn_events):
    # AGN catalog
    true_agn_rcom, _, _ = uniform_shell_sampler(COMDIST_MIN, COMDIST_MAX, NAGN)  # NO AGN NEEDED ABOVE RMAX_GW, p_rate = 0
    true_agn_z = fast_z_at_value(COSMO.comoving_distance, true_agn_rcom * u.Mpc)

    if not AGN_ZERROR:
        obs_agn_z = true_agn_z
    else:
        obs_agn_z = np.random.normal(loc=true_agn_z, scale=AGN_ZERROR, size=NAGN)

    # GWs & likelihood
    true_rcom_gw_agn = np.random.choice(true_agn_rcom, size=n_agn_events, replace=False)  # Do not generate GWs from the same AGN (see Hitchhiker's guide) - I think it only matters when doing selection effects?
    true_rcom_gw_alt, _, _ = uniform_shell_sampler(COMDIST_MIN, COMDIST_MAX, BATCH - n_agn_events)

    true_rcom_gw = np.hstack((true_rcom_gw_agn, true_rcom_gw_alt))
    true_z_gw = fast_z_at_value(COSMO.comoving_distance, true_rcom_gw * u.Mpc)
    true_rlum_gw = COSMO.luminosity_distance(true_z_gw).value

    # Measure
    obs_rlum_gw = true_rlum_gw * (1. + GW_DIST_ERR * np.random.normal(size=BATCH))  # TODO: Remove scatter and see what happens

    return obs_rlum_gw, obs_agn_z


def get_observed_gw_skypos_agn_catalog(n_agn_events):
    """
    Only sky position info
    Always assume perfectly measured AGN sky positions
    """
    # AGN catalog
    true_agn_rcom, true_agn_theta, true_agn_phi = uniform_shell_sampler(COMDIST_MIN, COMDIST_MAX, NAGN)  # NO AGN NEEDED ABOVE RMAX_GW, p_rate = 0
    true_agn_x, true_agn_y, true_agn_z = spherical2cartesian(true_agn_rcom, true_agn_theta, true_agn_phi)

    obs_agn_rcom, obs_agn_theta, obs_agn_phi = true_agn_rcom, true_agn_theta, true_agn_phi
    obs_agn_x, obs_agn_y, obs_agn_z = spherical2cartesian(obs_agn_rcom, obs_agn_theta, obs_agn_phi)

    # GWs & likelihood
    agn_generating_gw = np.random.choice(np.arange(NAGN), size=n_agn_events, replace=False)  # Do not generate GWs from the same AGN (see Hitchhiker's guide) - I think it only matters when doing selection effects?
    true_x_gw_agn, true_y_gw_agn, true_z_gw_agn = true_agn_x[agn_generating_gw], true_agn_y[agn_generating_gw], true_agn_z[agn_generating_gw]
    true_rcom_gw_alt, true_theta_gw_alt, true_phi_gw_alt = uniform_shell_sampler(COMDIST_MIN, COMDIST_MAX, BATCH - n_agn_events)
    true_x_gw_alt, true_y_gw_alt, true_z_gw_alt = spherical2cartesian(true_rcom_gw_alt, true_theta_gw_alt, true_phi_gw_alt)

    true_x_gw, true_y_gw, true_z_gw = np.hstack((true_x_gw_agn, true_x_gw_alt)), np.hstack((true_y_gw_agn, true_y_gw_alt)), np.hstack((true_z_gw_agn, true_z_gw_alt))

    # Skymap sizes
    sky_areas = np.random.uniform(low=SKY_AREA_LOW, high=SKY_AREA_HIGH, size=BATCH) * (np.pi / 180)**2  # Deg^2 to sr
    vonmises_kappas = 2 * np.pi * 1.14 / sky_areas

    skymap_centers = np.zeros((BATCH, 3))
    for i in range(BATCH):
        mu_true = np.array([true_x_gw[i], true_y_gw[i], true_z_gw[i]])  # True centers
        skymap_centers[i,:] = vonmises_fisher.rvs(mu_true / np.linalg.norm(mu_true), vonmises_kappas[i], size=1)  # Observed centers

    return skymap_centers, vonmises_kappas, obs_agn_x, obs_agn_y, obs_agn_z


def process_gw_skypos(skymap_centers, vonmises_kappas, obs_agn_x, obs_agn_y, obs_agn_z):
    S_agn = np.zeros(BATCH)
    for i in range(BATCH):
        for j in range(NAGN):
            agn_pos = np.array([obs_agn_x[j], obs_agn_y[j], obs_agn_z[j]])
            S_agn[i] += vonmises_fisher.pdf(x=agn_pos / np.linalg.norm(agn_pos), mu=skymap_centers[i], kappa=vonmises_kappas[i])
    S_agn /= NAGN
    S_alt = np.tile(1 / (4 * np.pi), BATCH)
    return S_agn, S_alt


def sample_sigmas(n_events=BATCH, mean=50, std=50):
    # Estimated by eye, seems to match observations quite well
    return np.abs( np.random.normal(loc=mean, scale=std, size=n_events) )
    

def cl_volume_from_sigma(sigmas):
    '''Calculates length of CL% vector when x, y, z are normally distributed with the same sigma'''
    radii = np.zeros(len(sigmas))
    for i, sigma in enumerate(sigmas):
        vector_length_cdf = lambda x: erf(x / (sigma * np.sqrt(2))) - np.sqrt(2) / (sigma * np.sqrt(np.pi)) * x * np.exp(-x**2 / (2 * sigma**2)) - SKYMAP_CL
        radii[i] = root_scalar(vector_length_cdf, bracket=[0, 100 * sigma], method='bisect').root
    return 4 * np.pi * radii**3 / 3, radii


def get_observed_gw_3dloc_agn_catalog(n_agn_events, fagn_idx, trial_idx):

    # AGN catalog
    true_agn_rcom, true_agn_theta, true_agn_phi = uniform_shell_sampler(COMDIST_MIN, AAA * COMDIST_MAX, NAGN)  # NO AGN NEEDED ABOVE RMAX_GW, p_rate = 0

    if MAKE_SKYMAPS:  # TODO: save only the true AGN positions for testing
        with h5py.File(CAT_DIR + f'/mockcat_{trial_idx}_{fagn_idx}.hdf5', "w") as f:
            f.create_dataset('comoving_distance', data=true_agn_rcom)
            f.create_dataset('ra', data=true_agn_phi)
            f.create_dataset('dec', data=np.pi * 0.5 - true_agn_theta)

    true_agn_x, true_agn_y, true_agn_z = spherical2cartesian(true_agn_rcom, true_agn_theta, true_agn_phi)
    true_agn_redshift = fast_z_at_value(COSMO.comoving_distance, true_agn_rcom * u.Mpc)

    if not AGN_ZERROR:
        obs_agn_redshift = true_agn_redshift
    else:
        obs_agn_redshift = np.random.normal(loc=true_agn_redshift, scale=AGN_ZERROR, size=NAGN)
        # obs_agn_redshift = stats.truncnorm.rvs(size=NAGN, a=(ZMIN - true_agn_redshift) / AGN_ZERROR, b=(np.inf - true_agn_redshift) / AGN_ZERROR, loc=true_agn_redshift, scale=AGN_ZERROR)

    obs_agn_rcom, obs_agn_theta, obs_agn_phi = COSMO.comoving_distance(obs_agn_redshift).value, true_agn_theta, true_agn_phi
    obs_agn_x, obs_agn_y, obs_agn_z = spherical2cartesian(obs_agn_rcom, obs_agn_theta, obs_agn_phi)

    # GWs & likelihood
    agn_possibly_generating_gw = np.arange(NAGN)[true_agn_rcom < COMDIST_MAX]
    agn_generating_gw = np.random.choice(agn_possibly_generating_gw, size=n_agn_events, replace=False)  # Do not generate GWs from the same AGN (see Hitchhiker's guide) - I think it only matters when doing selection effects?
    true_x_gw_agn, true_y_gw_agn, true_z_gw_agn = true_agn_x[agn_generating_gw], true_agn_y[agn_generating_gw], true_agn_z[agn_generating_gw]
    
    true_rcom_gw_alt, true_theta_gw_alt, true_phi_gw_alt = uniform_shell_sampler(COMDIST_MIN, COMDIST_MAX, BATCH - n_agn_events)
    true_x_gw_alt, true_y_gw_alt, true_z_gw_alt = spherical2cartesian(true_rcom_gw_alt, true_theta_gw_alt, true_phi_gw_alt)

    true_x_gw, true_y_gw, true_z_gw = np.hstack((true_x_gw_agn, true_x_gw_alt)), np.hstack((true_y_gw_agn, true_y_gw_alt)), np.hstack((true_z_gw_agn, true_z_gw_alt))

    locvol_sigmas = sample_sigmas()  # 3D spherical Gaussian with same sigma in each cartesian coordinate
    _, locvol_radius = cl_volume_from_sigma(locvol_sigmas)

    obs_gw_x, obs_gw_y, obs_gw_z = np.random.normal(loc=true_x_gw, scale=locvol_sigmas, size=BATCH), np.random.normal(loc=true_y_gw, scale=locvol_sigmas, size=BATCH), np.random.normal(loc=true_z_gw, scale=locvol_sigmas, size=BATCH)

    return obs_gw_x, obs_gw_y, obs_gw_z, locvol_radius, locvol_sigmas, obs_agn_x, obs_agn_y, obs_agn_z


def process_gw_3dloc(obs_gw_x, obs_gw_y, obs_gw_z, locvol_radius, locvol_sigmas, obs_agn_x, obs_agn_y, obs_agn_z):

    p_cl = np.zeros(BATCH)
    for i, gw_x in enumerate(obs_gw_x):
        gw_y, gw_z = obs_gw_y[i], obs_gw_z[i]
        agn_in_locvol = ((gw_x - obs_agn_x)**2 + (gw_y - obs_agn_y)**2 + (gw_z - obs_agn_z)**2) < locvol_radius[i]**2
        # print(np.sum(agn_in_locvol))

        p_cl[i] = np.sum( gaussian(x=obs_agn_x[agn_in_locvol], mu=gw_x, sigma=locvol_sigmas[i]) * gaussian(x=obs_agn_y[agn_in_locvol], mu=gw_y, sigma=locvol_sigmas[i]) * gaussian(x=obs_agn_z[agn_in_locvol], mu=gw_z, sigma=locvol_sigmas[i]) )

    S_agn = p_cl / AGN_NUMDENS
    S_alt = np.tile(SKYMAP_CL, BATCH)

    # print(S_agn, S_alt)
    return S_agn, S_alt


def make_skymaps_and_catalog(n_agn_events, fagn_idx, trial_idx):
    obs_gw_x, obs_gw_y, obs_gw_z, locvol_radius, locvol_sigmas, obs_agn_x, obs_agn_y, obs_agn_z = get_observed_gw_3dloc_agn_catalog(n_agn_events, fagn_idx, trial_idx)

    # First make posterior samples .hdf5 files
    for i in range(BATCH):
        posterior_samples_x = np.random.normal(loc=obs_gw_x[i], scale=locvol_sigmas[i], size=N_POSTERIOR_SAMPLES)
        posterior_samples_y = np.random.normal(loc=obs_gw_y[i], scale=locvol_sigmas[i], size=N_POSTERIOR_SAMPLES)
        posterior_samples_z = np.random.normal(loc=obs_gw_z[i], scale=locvol_sigmas[i], size=N_POSTERIOR_SAMPLES)

        # TODO: ASK LORENZO ABOUT INDUCED PRIOR IN THIS CASE
        rcom_samples, theta_samples, phi_samples = cartesian2spherical(posterior_samples_x, posterior_samples_y, posterior_samples_z)
        dec_samples = 0.5 * np.pi - theta_samples
        redshift_samples = fast_z_at_value(COSMO.comoving_distance, rcom_samples * u.Mpc)
        rlum_samples = COSMO.luminosity_distance(redshift_samples).value

        samples_table = Table([phi_samples, dec_samples, rlum_samples, rcom_samples, redshift_samples], 
                                    names=('ra', 'dec', 'luminosity_distance', 'comoving_distance', 'redshift'))
        filename = os.path.join(POST_SAMPS_DIR, f"gw_{trial_idx}_{fagn_idx}_{i:05d}.h5")

        with h5py.File(filename, "a") as f:
            mock_group = f.require_group("mock")  # Takes place of approximant in real GW data
            mock_group.create_dataset('posterior_samples', data=samples_table)
    
    # Then process all files into skymaps, this is the slow part
    print('Verifying OMP_NUM_THREADS:')
    os.system("echo $OMP_NUM_THREADS")

    for _, infile in enumerate(glob.glob(f'{POST_SAMPS_DIR}/gw_{trial_idx}_{fagn_idx}_*.h5')):
        print(f'Processing: {infile}')
        outfile = f'skymap_{trial_idx}_{fagn_idx}_{infile[-8:-3]}.fits.gz'
        print(f'Output: {SKYMAP_DIR}/{outfile}')

        os.system(f"ligo-skymap-from-samples {infile} --fitsoutname {outfile} --outdir {SKYMAP_DIR} --jobs {os.cpu_count()}")
    
    os.system(f"rm -rf {SKYMAP_DIR}/skypost.obj")
    
    return locvol_radius, obs_agn_x, obs_agn_y, obs_agn_z


def process_skymaps(locvol_radius, obs_agn_x, obs_agn_y, obs_agn_z):
    return np.ones(BATCH), np.ones(BATCH)


def check_directory(directory):
    if os.path.isdir(directory):
        if len(os.listdir(directory)) != 0:
            inp = None
            while inp not in ['y', 'yes', 'n', 'no']:
                inp = input(f'Found existing data in output directory: `{directory}`. DELETE existing data? (y/n)')

            if inp in ['y', 'yes']:
                print('Erasing existing data...')
                shutil.rmtree(directory)
                os.mkdir(directory)
            else:
                sys.exit('Not removing data. Please run again with a new output directory.')
    else:
        os.mkdir(directory)


def generate_and_process_universe_realization(fagn, fagn_idx, trial_idx):

    n_agn_events = int(fagn * BATCH)

    if USE_ONLY_MASS:
        obs_primary_mass_gw = get_observed_primary_mass(n_agn_events)
        return process_gw_mass(obs_primary_mass_gw)

    elif USE_ONLY_REDSHIFT:
        obs_rlum_gw, obs_agn_z = get_observed_gw_rlum_agn_z(n_agn_events)
        return process_gw_redshift(obs_rlum_gw, obs_agn_z)

    elif USE_MASS_AND_REDSHIFT:
        obs_primary_mass_gw = get_observed_primary_mass(n_agn_events)
        S_agn_mass, S_alt_mass = process_gw_mass(obs_primary_mass_gw)

        obs_rlum_gw, obs_agn_z = get_observed_gw_rlum_agn_z(n_agn_events)
        S_agn_redshift, S_alt_redshift = process_gw_redshift(obs_rlum_gw, obs_agn_z)
        return S_agn_mass * S_agn_redshift, S_alt_mass * S_alt_redshift  # Assuming the mass posterior is independent of redshift!!
    
    elif USE_ONLY_SKY:
        skymap_centers, vonmises_kappas, obs_agn_x, obs_agn_y, obs_agn_z = get_observed_gw_skypos_agn_catalog(n_agn_events)
        return process_gw_skypos(skymap_centers, vonmises_kappas, obs_agn_x, obs_agn_y, obs_agn_z)
    
    elif USE_ONLY_3DLOC:

        if not MAKE_SKYMAPS:
            obs_gw_x, obs_gw_y, obs_gw_z, locvol_radius, locvol_sigmas, obs_agn_x, obs_agn_y, obs_agn_z = get_observed_gw_3dloc_agn_catalog(n_agn_events, fagn_idx, trial_idx)
            return process_gw_3dloc(obs_gw_x, obs_gw_y, obs_gw_z, locvol_radius, locvol_sigmas, obs_agn_x, obs_agn_y, obs_agn_z)
        else:
            locvol_radius, obs_agn_x, obs_agn_y, obs_agn_z = make_skymaps_and_catalog(n_agn_events, fagn_idx, trial_idx)
            return process_skymaps(locvol_radius, obs_agn_x, obs_agn_y, obs_agn_z)


####################### ANALYSIS #########################


def full_analysis_likelihood_thread(trial_idx):
    
    log_llh = np.zeros((len(LOG_LLH_X_AX), N_TRUE_FAGNS))
    for fagn_idx, fagn_true in enumerate(TRUE_FAGNS):
        # if fagn_idx <= 1:
        #     continue

        # fagn_realized = np.random.binomial(BATCH, fagn_true) / BATCH  # Observed f_agn fluctuates around the true value
        fagn_realized = fagn_true
        print('FAGN = FAGN_TRUE')
        print(f"fagn realized: {fagn_realized}, fagn true: {fagn_true}")
        S_agn, S_alt = generate_and_process_universe_realization(fagn=fagn_realized, fagn_idx=fagn_idx, trial_idx=trial_idx)
        loglike = np.log(S_agn[:,None] * SKYMAP_CL * LOG_LLH_X_AX[None,:] + S_alt[:,None] * (1 - SKYMAP_CL * LOG_LLH_X_AX[None,:]))
        
        log_llh[:,fagn_idx] = np.sum(loglike, axis=0)  # sum over all GWs

    return trial_idx, log_llh


if __name__ == '__main__':

    posteriors = np.zeros((N_TRIALS, CALC_LOGLLH_AT_N_POINTS, N_TRUE_FAGNS))

    if not MAKE_SKYMAPS:  # Then we use multiple cores for multiple Universe realizations
        with ThreadPoolExecutor(max_workers=NCPU) as executor:
            future_to_index = {executor.submit(full_analysis_likelihood_thread, index): index for index in range(N_TRIALS)}
            for future in tqdm(as_completed(future_to_index), total=N_TRIALS):
                try:
                    trial_idx, log_llh = future.result(timeout=20)
                    posteriors[trial_idx,:,:] = log_llh
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error processing trial {future_to_index[future]}: {e}")

    else:  # We will need all the cores for generating skymaps
        check_directory(POST_SAMPS_DIR)
        check_directory(SKYMAP_DIR)
        check_directory(CAT_DIR)
        os.environ["OMP_NUM_THREADS"] = "1"  # Important for proper threading when making skymaps
        
        for trial_idx in range(N_TRIALS):
            _, log_llh = full_analysis_likelihood_thread(trial_idx)
            posteriors[trial_idx,:,:] = log_llh

    np.save(os.path.join(sys.path[0], fname), posteriors)
