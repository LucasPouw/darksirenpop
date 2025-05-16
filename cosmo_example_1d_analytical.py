import numpy as np
import matplotlib.pyplot as plt
from utils import uniform_shell_sampler, make_nice_plots, fast_z_at_value, histogram_pdf
from tqdm import tqdm
from default_arguments import DEFAULT_COSMOLOGY
import astropy.units as u
from scipy.integrate import quad, simpson
from scipy.stats import gaussian_kde
from astropy.constants import c
from concurrent.futures import as_completed, ThreadPoolExecutor
from multiprocessing import cpu_count
import time
from scipy.interpolate import interp1d
import traceback

make_nice_plots()

####################################################################
SPEED_OF_LIGHT_KMS = c.to('km/s').value
GW_DIST_ERR = 0.01  # Relative error
COMDIST_MAX = 100  # Maximum comoving distance in Mpc
VOLUME = 4 / 3 * np.pi * COMDIST_MAX**3
AGN_NUMDENS = 100 / VOLUME
BATCH = int(100)
N_POST_SAMPS = int(1e4)
N_MC_SAMPS = int(1e4)
N_TRIALS = 100
MAX_N_FAGNS = 21
CALC_LOGLLH_AT_N_POINTS = 1000
GW_CHUNK = 100  # Optimized for my own system - vectorize operations for this many GWs 
LLH_CHUNK = 10  # Optimized for my own system - vectorize operations for this many values of f_agn

fname = 'posteriors_replaceFalse_relerr0.01_100AGN_rel6'
NCPU = cpu_count()

# All subsequent globals are derived from the parameters above.
NAGN = int( np.ceil(AGN_NUMDENS * VOLUME) )

assert NAGN >= BATCH, f'Every AGN-origin GW must come from a unique AGN. Got {NAGN} AGN and {BATCH} AGN-origin GWs.'
print(f'#AGN is rounded from {AGN_NUMDENS * VOLUME} to {NAGN}, giving a number density of {NAGN / VOLUME:.3e}. Target was {AGN_NUMDENS:.3e}.')
NGW_ALT = BATCH
NGW_AGN = BATCH
N_TRUE_FAGNS = min(BATCH + 1, MAX_N_FAGNS)  # Cannot create more f_agn values than BATCH+1 and don't want to generate more than MAX_N_FAGNS
LOG_LLH_X_AX = np.linspace(0.0001, 0.9999, CALC_LOGLLH_AT_N_POINTS)

COSMO = DEFAULT_COSMOLOGY
ZMIN = 1e-4
ZMAX = fast_z_at_value(COSMO.comoving_distance, COMDIST_MAX * u.Mpc)
Z_INTEGRAL_AX = np.linspace(ZMIN, ZMAX, 1000)
LUMDIST_MIN = COSMO.luminosity_distance(ZMIN).value
LUMDIST_MAX = COSMO.luminosity_distance(ZMAX).value

COMDIST_MIN = COSMO.comoving_distance(ZMIN).value
rr = np.linspace(COMDIST_MIN, COMDIST_MAX, 1000)

USE_GW_SELECTION_EFFECTS = False
LUMDIST_THRESH_GW = LUMDIST_MAX  # Luminosity distance threshold in Mpc

## These lines are for making the underlying truth equal to the actual realization of that truth - TODO: CURRENTLY BINOMIAL OPTION DOESN'T WORK
USE_N_AGN_EVENTS = np.arange(0, BATCH + 1, int(BATCH / (N_TRUE_FAGNS - 1)), dtype=np.int32)
TRUE_FAGNS = USE_N_AGN_EVENTS / BATCH
REALIZED_FAGNS = USE_N_AGN_EVENTS / BATCH  # Realization of the truth
####################################################################


def dVdz_unnorm(z, cosmo):
    '''Assuming flat LCDM'''
    Omega_m = cosmo.Om0
    Omega_Lambda = 1 - Omega_m
    E_of_z = np.sqrt((1 + z)**3 * Omega_m + Omega_Lambda)
    com_vol = ((1 + z) * cosmo.angular_diameter_distance(z).value)**2 / E_of_z
    return com_vol


func = lambda z: dVdz_unnorm(z, COSMO)
NORM = quad(func, ZMIN, ZMAX)[0]  # NORMALIZE UP TO ZMAX, since p_rate(z > ZMAX) = 0


def dVdz_prior(z, norm=NORM, cosmo=COSMO):
    z = np.atleast_1d(z)
    result = np.zeros_like(z)
    below_thresh = (z < ZMAX) & (z > ZMIN)
    result[below_thresh] = dVdz_unnorm(z[below_thresh], cosmo) / norm
    return result


def unnormed_lumdist_distribution(d, dobs, sigma):
    return np.exp(-(dobs/d - 1)**2 / (2 * sigma**2)) / d


def unnormed_redshift_distribution(z, dobs, sigma):
    dl = COSMO.luminosity_distance(z).value
    H_z = COSMO.H(z).value  # H(z) in km/s/Mpc
    chi_z = (1 + z) * dl
    dDL_dz = chi_z + (1 + z) * (SPEED_OF_LIGHT_KMS / H_z)
    return unnormed_lumdist_distribution(dl, dobs, sigma) / dDL_dz


n_norms = 1000
redshift_norm_z_axis = np.geomspace(ZMIN / 10, 10 * ZMAX, 10000)
znorms = np.zeros(n_norms)
znorms_quad = np.zeros(n_norms)
trial_lumdist = np.linspace(LUMDIST_MIN, 2 * LUMDIST_MAX, n_norms)  # 0 and 10*lumdist_max
print('Precalculating redshift posterior normalizations...')
for i, dobs in tqdm(enumerate(trial_lumdist)):
    func = lambda x: unnormed_redshift_distribution(x, dobs, sigma=GW_DIST_ERR)
    znorms[i] = simpson(y=func(redshift_norm_z_axis), x=redshift_norm_z_axis)
ZNORMS_INTERP = interp1d(trial_lumdist, znorms)


def redshift_distribution(z, dobs, sigma=GW_DIST_ERR):
    # Try simple Gaussian first
    func = lambda x: unnormed_redshift_distribution(x, dobs, sigma)
    norm = ZNORMS_INTERP(dobs)
    return func(z) / norm


def redshift_posterior_times_alt_prior(z, dobs, sigma=GW_DIST_ERR):
    return redshift_distribution(z, dobs, sigma) * dVdz_prior(z)


def p_alt_integral(dobs, redshift_integral_axis=Z_INTEGRAL_AX, gw_dist_err=GW_DIST_ERR):
    func = lambda x: redshift_posterior_times_alt_prior(x, dobs, gw_dist_err)
    return simpson(y=func(redshift_integral_axis), x=redshift_integral_axis)


LUMDIST_AT_MAX_REDSHIFT = COSMO.luminosity_distance(999).value  # For removing errors in generating posteriors later


def process_gw(rlum_obs, agn_z):
    p_agn = np.sum( redshift_distribution(agn_z[:,np.newaxis], rlum_obs), axis=0 ) / len(agn_z)
    p_alt = p_alt_integral(rlum_obs[:,np.newaxis])
    return p_agn, p_alt


def generate_and_process_universe_realization(fagn, batch=BATCH):
    # AGN catalog
    agn_rcom, _, _ = uniform_shell_sampler(COMDIST_MIN, COMDIST_MAX, NAGN)  # NO AGN NEEDED ABOVE RMAX_GW, p_rate = 0
    agn_z = fast_z_at_value(COSMO.comoving_distance, agn_rcom * u.Mpc)

    n_agn_events = int(fagn * batch)

    # GWs & likelihood
    # GENERATE GWS UP TO RMAX_GW
    true_rcom_gw_agn = np.random.choice(agn_rcom, size=n_agn_events, replace=False)
    true_rcom_gw_alt, _, _ = uniform_shell_sampler(COMDIST_MIN, COMDIST_MAX, batch - n_agn_events)

    true_rcom_gw = np.hstack((true_rcom_gw_agn, true_rcom_gw_alt))
    true_z_gw = fast_z_at_value(COSMO.comoving_distance, true_rcom_gw * u.Mpc)
    true_rlum_gw = COSMO.luminosity_distance(true_z_gw).value

    # MEASURE
    obs_rlum_gw = true_rlum_gw * (1. + GW_DIST_ERR * np.random.normal(size=batch))  # Remove scatter and see what happens

    return process_gw(obs_rlum_gw, agn_z)


def full_analysis_likelihood_thread(index, batch=BATCH, N_true_fagns=N_TRUE_FAGNS, log_llh_x_ax=LOG_LLH_X_AX):
    
    log_llh = np.zeros((len(log_llh_x_ax), N_true_fagns))
    for i, fagn_true in enumerate(TRUE_FAGNS):
        # draw fagn from a binomial distribution
        # fagn_realized = np.random.binomial(batch, fagn_true) / batch
        fagn_realized = fagn_true
        # print(f"fagn realized: {fagn_realized}, fagn true: {fagn}")
        p_agn, p_alt = generate_and_process_universe_realization(fagn=fagn_realized)
        loglike = np.log(p_agn[:,None] * log_llh_x_ax[None,:] + p_alt[:,None] * (1 - log_llh_x_ax[None,:]))
        
        log_llh[:,i] = np.sum(loglike, axis=0)  # sum over all GWs

    return index, log_llh


if __name__ == '__main__':
    # estimation_arr = np.zeros((N_TRIALS, N_TRUE_FAGNS))
    posteriors = np.zeros((N_TRIALS, CALC_LOGLLH_AT_N_POINTS, N_TRUE_FAGNS))

    # for i in tqdm(range(N_TRIALS)):
    #     # _, estimation_arr[i,:] = full_analysis_likelihood_thread(i)
    #     _, log_llh = full_analysis_likelihood_thread(i)
    #     posteriors[i,:,:] = log_llh


    with ThreadPoolExecutor(max_workers=NCPU) as executor:
        future_to_index = {executor.submit(full_analysis_likelihood_thread, index): index for index in range(N_TRIALS)}
        for future in tqdm(as_completed(future_to_index), total=N_TRIALS):
            try:
                i, log_llh = future.result(timeout=20)
                posteriors[i,:,:] = log_llh
            except Exception as e:
                traceback.print_exc()
                print(f"Error processing event {future_to_index[future]}: {e}")

    np.save(fname, posteriors)
