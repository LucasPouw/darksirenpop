import numpy as np
import matplotlib.pyplot as plt
from utils import uniform_shell_sampler, make_nice_plots, fast_z_at_value
from tqdm import tqdm
from default_arguments import DEFAULT_COSMOLOGY
import astropy.units as u
from scipy.integrate import quad
from scipy.stats import gaussian_kde
from astropy.constants import c
from concurrent.futures import as_completed, ThreadPoolExecutor
from multiprocessing import cpu_count
import time

make_nice_plots()

####################################################################
SPEED_OF_LIGHT_KMS = c.to('km/s').value
GW_DIST_ERR = 0.1  # Relative error
COMDIST_MIN = 0
COMDIST_MAX = 100  # Maximum comoving distance in Mpc
NAGN = int(1e2)
N_POST_SAMPS = int(1e5)
N_MC_SAMPS = int(1e4)
rr = np.linspace(COMDIST_MIN, COMDIST_MAX, 1000)
CHUNK = int(100)
NGW_ALT = CHUNK
NGW_AGN = CHUNK

COSMO = DEFAULT_COSMOLOGY
ZMIN = 0
ZMAX = fast_z_at_value(COSMO.comoving_distance, COMDIST_MAX * u.Mpc)
LUMDIST_MIN = COSMO.luminosity_distance(ZMIN).value
LUMDIST_MAX = COSMO.luminosity_distance(ZMAX).value

USE_GW_SELECTION_EFFECTS = False
LUMDIST_THRESH_GW = LUMDIST_MAX  # Luminosity distance threshold in Mpc

NCPU = 1  # cpu_count()
####################################################################


use_N_gws = CHUNK
n_trials = 300
max_N_fagns = 51
N_true_fagns = min(use_N_gws+1, max_N_fagns)    # Cannot create more f_agn values than use_N_gws+1 and don't want to generate more than max_N_fagns
calc_logllh_at_N_points = 1000                  # Only change if you want higher resolution, but why would you?
log_llh_x_ax = np.linspace(0.0001, 0.9999, calc_logllh_at_N_points)
gw_chunk_size = 100  # Optimized for my own system - vectorize operations for this many GWs 
llh_chunk_size = 10  # Optimized for my own system - vectorize operations for this many values of f_agn
p_det = 1


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
    below_thresh = z < ZMAX
    result[below_thresh] = dVdz_unnorm(z[below_thresh], cosmo) / norm
    return result


LUMDIST_AT_MAX_REDSHIFT = COSMO.luminosity_distance(999).value


def generate_posterior(rlum_obs, rlum_relerr=GW_DIST_ERR, n_posterior_samples=N_POST_SAMPS):
    # Importance resampling of distances
    dtrue_postsamps = rlum_obs / (1 + rlum_relerr * np.random.normal(size=4 * n_posterior_samples))
    neg = dtrue_postsamps < 0
    # if np.sum(neg) != 0:
    #     print(f'Removing {np.sum(neg)} negative luminosity distance samples.')
    dtrue_postsamps = dtrue_postsamps[~neg]  # WARNING: Negative values are very rare, (20% (30%) error, 50k (100k) postsamps, 1 (180) negative samp), so just remove them. But be aware!
    weights = dtrue_postsamps / np.sum(dtrue_postsamps)  # Importance weights proportional to d
    lumdist_samples = np.random.choice(dtrue_postsamps, size=2 * n_posterior_samples, p=weights)
    
    n_samps_above_max_z = np.sum(lumdist_samples > LUMDIST_AT_MAX_REDSHIFT)
    if n_samps_above_max_z != 0:
        lumdist_samples = lumdist_samples[lumdist_samples > LUMDIST_AT_MAX_REDSHIFT]
        print(f'Removing {n_samps_above_max_z} samples at too high luminosity distance ({LUMDIST_AT_MAX_REDSHIFT:.2f}).')
    
    # Redshift reweighting
    z_samples = fast_z_at_value(COSMO.luminosity_distance, lumdist_samples * u.Mpc)
    H_z = COSMO.H(z_samples).value  # H(z) in km/s/Mpc
    chi_z = (1 + z_samples) * lumdist_samples
    dDL_dz = chi_z + (1 + z_samples) * (SPEED_OF_LIGHT_KMS / H_z)  # c = 3e5 km/s
    z_weights = 1 / dDL_dz
    z_weights /= np.sum(z_weights)
    z_samples = np.random.choice(z_samples, n_posterior_samples, p=z_weights)
    return z_samples


def process_gw(index, rlum_obs, agn_z, n_mc_samps=N_MC_SAMPS):
    
    # Takes ~0.08 s/it for 1e5 posterior samples
    gw_z_posterior = generate_posterior(rlum_obs)

    # Avoid outliers messing up the KDE!
    trimmed = gw_z_posterior[(gw_z_posterior > np.percentile(gw_z_posterior, 0.1)) & (gw_z_posterior < np.percentile(gw_z_posterior, 99.9))]
    kde_gw_z_posterior = gaussian_kde(trimmed)

    # Takes ~0.13 s/it for 100 AGN
    p_agn = np.sum(kde_gw_z_posterior(agn_z)) / len(agn_z)

    mc_samps = np.random.choice(gw_z_posterior, size=n_mc_samps)
    mc_samps_below_thresh = mc_samps[mc_samps < ZMAX]
    p_alt = np.sum( dVdz_prior(mc_samps_below_thresh) ) / n_mc_samps
    return index, p_agn, p_alt


def process_gw_thread(obs_rlum, agn_z, zmax=ZMAX, ncpu=NCPU):
    ngw = len(obs_rlum)
    p_agn = np.zeros(ngw)
    p_alt = np.zeros(ngw)
    error_idx_agn = []
    with ThreadPoolExecutor(max_workers=ncpu) as executor:
        future_to_index = {executor.submit(
                                        process_gw, 
                                        index, 
                                        obs_rlum,
                                        agn_z[agn_z < zmax]
                                    ): index for index, obs_rlum in enumerate(obs_rlum)
                                    }
        
        for future in as_completed(future_to_index):
            try:
                i, p_agn_result, p_alt_result = future.result(timeout=20)
                p_agn[i] = p_agn_result
                p_alt[i] = p_alt_result
            except Exception as e:
                print(f"Error processing event {future_to_index[future]}: {e}")
                error_idx_agn.append(future_to_index[future])
    return p_agn, p_alt


def generate_and_process_chunk_gws(ngw=CHUNK):
    # AGN catalog
    agn_rcom, _, _ = uniform_shell_sampler(COMDIST_MIN, COMDIST_MAX, NAGN)  # NO AGN NEEDED ABOVE RMAX_GW, p_rate = 0
    agn_z = fast_z_at_value(COSMO.comoving_distance, agn_rcom * u.Mpc)

    # GWs & likelihood
    # GENERATE GWS UP TO RMAX_GW
    true_rcom_gw_alt, _, _ = uniform_shell_sampler(COMDIST_MIN, COMDIST_MAX, ngw)
    true_rcom_gw_agn = np.random.choice(agn_rcom, ngw)

    true_z_gw_alt = fast_z_at_value(COSMO.comoving_distance, true_rcom_gw_alt * u.Mpc)
    true_z_gw_agn = fast_z_at_value(COSMO.comoving_distance, true_rcom_gw_agn * u.Mpc)

    true_rlum_gw_alt = COSMO.luminosity_distance(true_z_gw_alt).value
    true_rlum_gw_agn = COSMO.luminosity_distance(true_z_gw_agn).value

    # MEASURE
    obs_rlum_gw_alt = true_rlum_gw_alt * (1. + GW_DIST_ERR * np.random.normal(size=ngw))
    obs_rlum_gw_agn = true_rlum_gw_agn * (1. + GW_DIST_ERR * np.random.normal(size=ngw))


    p_agn_agn_gws, p_alt_agn_gws = process_gw_thread(obs_rlum_gw_agn, agn_z)
    p_agn_alt_gws, p_alt_alt_gws = process_gw_thread(obs_rlum_gw_alt, agn_z)

    # p_agn_agn_gws = np.zeros(ngw)
    # p_alt_agn_gws = np.zeros(ngw)
    # p_agn_alt_gws = np.zeros(ngw)
    # p_alt_alt_gws = np.zeros(ngw)
    # for i, obs_rlum in enumerate(obs_rlum_gw_agn):
    #     _, p_agn_agn_gws[i], p_alt_agn_gws[i] = process_gw(i, obs_rlum, agn_z, n_mc_samps=N_MC_SAMPS)
    
    # for i, obs_rlum in enumerate(obs_rlum_gw_alt):
    #     _, p_agn_alt_gws[i], p_alt_alt_gws[i] = process_gw(i, obs_rlum, agn_z, n_mc_samps=N_MC_SAMPS)

    return p_agn_agn_gws, p_alt_agn_gws, p_agn_alt_gws, p_alt_alt_gws


def chunked_llh_processing(total_cw_prob_agn, total_cw_prob_alt, total_prob_alt, fagn_times_fobsc):
    log_llh_numerator = np.zeros((calc_logllh_at_N_points, N_true_fagns))
    for i in range(int(use_N_gws / gw_chunk_size)):
        gw_start, gw_stop = int(i * gw_chunk_size), int((i + 1) * gw_chunk_size)
        cw_agn_prob_chunk = total_cw_prob_agn[:, gw_start:gw_stop, :]
        cw_alt_prob_chunk = total_cw_prob_alt[:, gw_start:gw_stop, :]
        alt_prob_chunk = total_prob_alt[:, gw_start:gw_stop, :]

        for j in range(int(calc_logllh_at_N_points / llh_chunk_size)):
            llh_start, llh_stop = int(j * llh_chunk_size), int((j + 1) * llh_chunk_size)
            fagn_chunk = fagn_times_fobsc[llh_start:llh_stop, ...]
            fagn_times_cw_p_agn = fagn_chunk * cw_agn_prob_chunk
            fagn_times_cw_p_alt = fagn_chunk * cw_alt_prob_chunk
            alt_prob_chunk_rightshape = np.ones_like(fagn_chunk) * alt_prob_chunk  # TODO: This shape correction doesn't seem to matter

            log_prob = np.log(fagn_times_cw_p_agn + alt_prob_chunk_rightshape - fagn_times_cw_p_alt)
            log_llh_numerator[llh_start:llh_stop, :] += np.sum(log_prob, axis=1)
    return log_llh_numerator


## These two lines are for making the underlying truth equal to the actual realization of that truth
use_N_agn_events = np.arange(0, use_N_gws + 1, int(use_N_gws / (N_true_fagns-1)), dtype=np.int32)
true_fagns = use_N_agn_events / use_N_gws
use_N_alt_events = use_N_gws - use_N_agn_events
realized_fagns = use_N_agn_events / use_N_gws  # Realization of the truth


def full_analysis_likelihood_thread(index):
    p_agn_agn_gws, p_alt_agn_gws, p_agn_alt_gws, p_alt_alt_gws = generate_and_process_chunk_gws()

    ### Some translations to reuse my code ###
    cw_pagn = np.append(p_agn_agn_gws, p_agn_alt_gws)
    cw_palt = np.append(p_alt_agn_gws, p_alt_alt_gws)
    palt = cw_palt  # c = 1
    agn_events = np.ones(len(cw_pagn), dtype=bool)
    agn_events[NGW_ALT:] = 0
    alt_events = ~agn_events
    ##########################################
    
    if (use_N_gws > gw_chunk_size) & (calc_logllh_at_N_points > llh_chunk_size):
        print('Chunking...')

    ## Use these lines for binomial realization of truth TODO: get this working properly
    # true_fagns = np.linspace(0, 1, N_true_fagns)  # Underlying truth
    # use_N_agn_events = np.random.binomial(n=use_N_gws, p=true_fagns)  # Make random realization of a universe with a true fagn
    # use_N_alt_events = use_N_gws - use_N_agn_events
    # realized_fagns = use_N_agn_events / use_N_gws  # Realization of the truth

    agn_idx = np.zeros((N_true_fagns, use_N_gws), dtype=int)
    alt_idx = np.zeros((N_true_fagns, use_N_gws), dtype=int)
    for k in range(N_true_fagns):
        arr = np.arange(use_N_gws)
        np.random.shuffle(arr)
        agn_idx[k,:] = arr
        np.random.shuffle(arr)
        alt_idx[k,:] = arr + NGW_AGN
    idx = np.where(np.arange(use_N_gws) < use_N_agn_events[:, None], agn_idx, alt_idx)  # Shape (N_true_fagns, use_N_gws)

    fagn_times_fobsc = log_llh_x_ax[:, np.newaxis, np.newaxis]
    total_cw_prob_agn = cw_pagn[idx].T[np.newaxis,...]
    total_cw_prob_alt = cw_palt[idx].T[np.newaxis,...]
    total_prob_alt = palt[idx].T[np.newaxis,...]

    if (use_N_gws > gw_chunk_size) & (calc_logllh_at_N_points > llh_chunk_size):  # Chunking to avoid too large arrays in memory
        log_llh_numerator = chunked_llh_processing(total_cw_prob_agn, total_cw_prob_alt, total_prob_alt, fagn_times_fobsc)
    else:
        log_llh_numerator_per_event = np.log(fagn_times_fobsc * total_cw_prob_agn + total_prob_alt - fagn_times_fobsc * total_cw_prob_alt)
        log_llh_numerator = np.sum(log_llh_numerator_per_event, axis=1 )

    log_llh_denominator = use_N_gws * np.log(p_det)  # GW SELECTION EFFECTS
    log_llh = log_llh_numerator - log_llh_denominator
    return index, log_llh_x_ax[np.argmax(log_llh, axis=0)]  # TODO: change to interpolation


if __name__ == '__main__':
    estimation_arr = np.zeros((n_trials, N_true_fagns))
    for i in tqdm(range(n_trials)):
        _, estimation_arr[i,:] = full_analysis_likelihood_thread(i)
    # with ThreadPoolExecutor(max_workers=NCPU) as executor:
    #     future_to_index = {executor.submit(full_analysis_likelihood_thread, index): index for index in range(n_trials)}
    #     for future in tqdm(as_completed(future_to_index), total=n_trials):
    #         try:
    #             i, estimates = future.result(timeout=20)
    #             estimation_arr[i,:] = estimates
    #         except Exception as e:
    #             print(f"Error processing event {future_to_index[future]}: {e}")


    fagn_medians = np.median(estimation_arr, axis=0)
    q016 = np.quantile(estimation_arr, 0.16, axis=0)
    q084 = np.quantile(estimation_arr, 0.84, axis=0)

    plt.figure(figsize=(8,8))
    plt.plot(true_fagns, fagn_medians, color='red', linewidth=3)
    plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), linestyle='dashed', color='black', zorder=6, linewidth=3)
    plt.fill_between(true_fagns, q016, q084, color='red', alpha=0.3)
    plt.xlabel(r'$f_{\rm agn, true}$')
    plt.ylabel(r'$f_{\rm agn, estim}$')
    plt.grid()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig('relerr300.pdf')
    plt.show()
