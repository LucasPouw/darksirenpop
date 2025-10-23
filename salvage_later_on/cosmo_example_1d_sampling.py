import numpy as np
import matplotlib.pyplot as plt
from utils import uniform_shell_sampler, make_nice_plots, fast_z_at_value, histogram_pdf
from tqdm import tqdm
from default_globals import *
import astropy.units as u
from scipy.integrate import quad, simpson
from scipy.stats import gaussian_kde
from concurrent.futures import as_completed, ThreadPoolExecutor
from multiprocessing import cpu_count
import time
from scipy.interpolate import interp1d
import traceback

make_nice_plots()

####################################################################
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


LUMDIST_AT_MAX_REDSHIFT = COSMO.luminosity_distance(999).value  # For removing errors in generating posteriors later


def generate_posterior(rlum_obs, rlum_relerr=GW_DIST_ERR, n_posterior_samples=N_POST_SAMPS):
    # Importance resampling of distances
    dtrue_postsamps = rlum_obs / (1 + rlum_relerr * np.random.normal(size=4 * n_posterior_samples))
    neg = dtrue_postsamps < LUMDIST_MIN
    # if np.sum(neg) != 0:
    #     print(f'Removing {np.sum(neg)} negative luminosity distance samples.')
    dtrue_postsamps = dtrue_postsamps[~neg]  # WARNING: Negative values are very rare, (20% (30%) error, 50k (100k) postsamps, 1 (180) negative samp), so just remove them. But be aware!
    weights = dtrue_postsamps / np.sum(dtrue_postsamps)  # Importance weights proportional to d
    # start = time.time()
    lumdist_samples = np.random.choice(dtrue_postsamps, size=2 * n_posterior_samples, p=weights)
    # print('resampling dl', time.time() - start)
    
    n_samps_above_max_z = np.sum(lumdist_samples > LUMDIST_AT_MAX_REDSHIFT)
    if n_samps_above_max_z != 0:
        lumdist_samples = lumdist_samples[lumdist_samples < LUMDIST_AT_MAX_REDSHIFT]
        print(f'Removing {n_samps_above_max_z} samples at too high luminosity distance ({LUMDIST_AT_MAX_REDSHIFT:.2f}).')
    
    # Redshift reweighting
    # start = time.time()
    z_samples = fast_z_at_value(COSMO.luminosity_distance, lumdist_samples * u.Mpc)
    # print('fast z', time.time() - start)
    H_z = COSMO.H(z_samples).value  # H(z) in km/s/Mpc
    chi_z = (1 + z_samples) * lumdist_samples
    dDL_dz = chi_z + (1 + z_samples) * (SPEED_OF_LIGHT_KMS / H_z)  # c = 3e5 km/s
    z_weights = 1 / dDL_dz
    z_weights /= np.sum(z_weights)
    # start = time.time()
    z_samples = np.random.choice(z_samples, n_posterior_samples, p=z_weights)
    # print('resampling z', time.time() - start)
    return z_samples


# def process_gw(index, rlum_obs, agn_z, n_mc_samps=N_MC_SAMPS):
    
#     # start = time.time()
#     gw_z_posterior = generate_posterior(rlum_obs)
#     # print("posterior generation took", time.time() - start)

#     # start = time.time()
#     # Avoid outliers messing up the KDE!
#     trimmed = gw_z_posterior[(gw_z_posterior > np.percentile(gw_z_posterior, 0.1)) & (gw_z_posterior < np.percentile(gw_z_posterior, 99.9))]

#     if len(trimmed) < 1e4:
#         kde_gw_z_posterior = gaussian_kde(trimmed)
#         kde_vals = kde_gw_z_posterior(agn_z)
#     else:
#         # print('Histogram PDF inference is not tested yet. Check first for unbiased results using gaussian_kde.')
#         kde_vals = histogram_pdf(trimmed, agn_z)
    
#     p_agn = np.sum(kde_vals) / len(agn_z)

#     # start = time.time()
#     # kde_gw_z_posterior = gaussian_kde(trimmed)
#     # # Takes ~0.13 s/it for 100 AGN
#     # kde_vals = kde_gw_z_posterior.evaluate(agn_z)
#     # p_agn = np.sum(kde_vals) / len(agn_z)
#     # print('gaussian_kde:', time.time() - start)

#     # start = time.time()
#     # ckde_vals = ckd_tree_kde_evaluation(data=trimmed, evaluation_points=agn_z)
#     # p_agn_ckdtree = np.sum( ckde_vals ) / len(agn_z)
#     # print('cKDTree:', time.time() - start)

#     # start = time.time()
#     # hkde_vals = histogram_pdf(trimmed, agn_z)
#     # p_agn = np.sum( hkde_vals ) / len(agn_z)
#     # print('histogram:', time.time() - start)

#     # idx = np.argsort(agn_z)

#     # plt.figure()
#     # plt.plot(agn_z[idx], kde_vals[idx], color='blue')
#     # plt.plot(agn_z[idx], ckde_vals[idx], color='red')
#     # plt.plot(agn_z[idx], hkde_vals[idx], color='black')
#     # plt.hist(trimmed, density=True, bins=50)
#     # plt.savefig('kde_comparison.pdf')
#     # plt.close()

#     # start = time.time()
#     mc_samps = np.random.choice(gw_z_posterior, size=n_mc_samps)
#     mc_samps_below_thresh = mc_samps[mc_samps < ZMAX]
#     p_alt = np.sum( dVdz_prior(mc_samps_below_thresh) ) / n_mc_samps
#     # print('the rest took', time.time() - start)
#     return index, p_agn, p_alt


# def process_gw_thread(obs_rlum, agn_z, zmax=ZMAX, ncpu=NCPU):
#     ngw = len(obs_rlum)
#     p_agn = np.zeros(ngw)
#     p_alt = np.zeros(ngw)
#     error_idx_agn = []
#     with ThreadPoolExecutor(max_workers=ncpu) as executor:
#         future_to_index = {executor.submit(
#                                         process_gw, 
#                                         index, 
#                                         obs_rlum,
#                                         agn_z[agn_z < zmax]
#                                     ): index for index, obs_rlum in enumerate(obs_rlum)
#                                     }
        
#         for future in as_completed(future_to_index):
#             try:
#                 i, p_agn_result, p_alt_result = future.result(timeout=20)
#                 p_agn[i] = p_agn_result
#                 p_alt[i] = p_alt_result
#             except Exception as e:
#                 print(f"Error processing event {future_to_index[future]}: {e}")
#                 error_idx_agn.append(future_to_index[future])
#     return p_agn, p_alt


# def generate_and_process_chunk_gws(ngw=BATCH):
#     # AGN catalog
#     agn_rcom, _, _ = uniform_shell_sampler(COMDIST_MIN, COMDIST_MAX, NAGN)  # NO AGN NEEDED ABOVE RMAX_GW, p_rate = 0
#     agn_z = fast_z_at_value(COSMO.comoving_distance, agn_rcom * u.Mpc)

#     # GWs & likelihood
#     # GENERATE GWS UP TO RMAX_GW
#     true_rcom_gw_alt, _, _ = uniform_shell_sampler(COMDIST_MIN, COMDIST_MAX, ngw)
#     true_rcom_gw_agn = np.random.choice(agn_rcom, ngw, replace=False)

#     true_z_gw_alt = fast_z_at_value(COSMO.comoving_distance, true_rcom_gw_alt * u.Mpc)
#     true_z_gw_agn = fast_z_at_value(COSMO.comoving_distance, true_rcom_gw_agn * u.Mpc)

#     true_rlum_gw_alt = COSMO.luminosity_distance(true_z_gw_alt).value
#     true_rlum_gw_agn = COSMO.luminosity_distance(true_z_gw_agn).value

#     # MEASURE
#     obs_rlum_gw_alt = true_rlum_gw_alt * (1. + GW_DIST_ERR * np.random.normal(size=ngw))
#     obs_rlum_gw_agn = true_rlum_gw_agn * (1. + GW_DIST_ERR * np.random.normal(size=ngw))

#     p_agn_agn_gws, p_alt_agn_gws = process_gw_thread(obs_rlum_gw_agn, agn_z)
#     p_agn_alt_gws, p_alt_alt_gws = process_gw_thread(obs_rlum_gw_alt, agn_z)

#     # p_agn_agn_gws = np.zeros(ngw)
#     # p_alt_agn_gws = np.zeros(ngw)
#     # p_agn_alt_gws = np.zeros(ngw)
#     # p_alt_alt_gws = np.zeros(ngw)
#     # for i, obs_rlum in enumerate(obs_rlum_gw_agn):
#     #     _, p_agn_agn_gws[i], p_alt_agn_gws[i] = process_gw(i, obs_rlum, agn_z, n_mc_samps=N_MC_SAMPS)
    
#     # for i, obs_rlum in enumerate(obs_rlum_gw_alt):
#     #     _, p_agn_alt_gws[i], p_alt_alt_gws[i] = process_gw(i, obs_rlum, agn_z, n_mc_samps=N_MC_SAMPS)

#     return p_agn_agn_gws, p_alt_agn_gws, p_agn_alt_gws, p_alt_alt_gws


# def chunked_llh_processing(total_cw_prob_agn, total_cw_prob_alt, total_prob_alt, fagn_times_fobsc):
#     log_llh_numerator = np.zeros((CALC_LOGLLH_AT_N_POINTS, N_TRUE_FAGNS))
#     for i in range(int(BATCH / GW_CHUNK)):
#         gw_start, gw_stop = int(i * GW_CHUNK), int((i + 1) * GW_CHUNK)
#         cw_agn_prob_chunk = total_cw_prob_agn[:, gw_start:gw_stop, :]
#         cw_alt_prob_chunk = total_cw_prob_alt[:, gw_start:gw_stop, :]
#         alt_prob_chunk = total_prob_alt[:, gw_start:gw_stop, :]

#         for j in range(int(CALC_LOGLLH_AT_N_POINTS / LLH_CHUNK)):
#             llh_start, llh_stop = int(j * LLH_CHUNK), int((j + 1) * LLH_CHUNK)
#             fagn_chunk = fagn_times_fobsc[llh_start:llh_stop, ...]
#             fagn_times_cw_p_agn = fagn_chunk * cw_agn_prob_chunk
#             fagn_times_cw_p_alt = fagn_chunk * cw_alt_prob_chunk
#             alt_prob_chunk_rightshape = np.ones_like(fagn_chunk) * alt_prob_chunk  # TODO: This shape correction doesn't seem to matter

#             log_prob = np.log(fagn_times_cw_p_agn + alt_prob_chunk_rightshape - fagn_times_cw_p_alt)
#             log_llh_numerator[llh_start:llh_stop, :] += np.sum(log_prob, axis=1)
#     return log_llh_numerator


# def full_analysis_likelihood_thread(index, batch=BATCH, N_true_fagns=N_TRUE_FAGNS, use_N_agn_events=USE_N_AGN_EVENTS, log_llh_x_ax=LOG_LLH_X_AX):
#     # start = time.time()
#     p_agn_agn_gws, p_alt_agn_gws, p_agn_alt_gws, p_alt_alt_gws = generate_and_process_chunk_gws()
#     # print('Total realization and processing took', time.time() - start)

#     ### Some translations to reuse my code ###
#     cw_pagn = np.append(p_agn_agn_gws, p_agn_alt_gws)
#     cw_palt = np.append(p_alt_agn_gws, p_alt_alt_gws)
#     palt = cw_palt  # c = 1
#     agn_events = np.ones(len(cw_pagn), dtype=bool)
#     agn_events[batch:] = 0
#     alt_events = ~agn_events
#     ##########################################
    
#     chunking = False
#     if (batch > GW_CHUNK) & (len(log_llh_x_ax) > LLH_CHUNK):  # Chunking to avoid too large arrays in memory
#         print('Chunking...')
#         chunking = True

#     ## Use these lines for binomial realization of truth TODO: get this working properly
#     # true_fagns = np.linspace(0, 1, N_true_fagns)  # Underlying truth
#     # use_N_agn_events = np.random.binomial(n=use_N_gws, p=true_fagns)  # Make random realization of a universe with a true fagn
#     # use_N_alt_events = use_N_gws - use_N_agn_events
#     # realized_fagns = use_N_agn_events / use_N_gws  # Realization of the truth

#     agn_idx = np.zeros((N_true_fagns, batch), dtype=int)
#     alt_idx = np.zeros((N_true_fagns, batch), dtype=int)
#     for k in range(N_true_fagns):
#         arr = np.arange(batch)
#         np.random.shuffle(arr)
#         agn_idx[k,:] = arr
#         np.random.shuffle(arr)
#         alt_idx[k,:] = arr + batch
#     idx = np.where(np.arange(batch) < use_N_agn_events[:, None], agn_idx, alt_idx)  # Shape (N_true_fagns, use_N_gws)

#     fagn_times_fobsc = log_llh_x_ax[:, np.newaxis, np.newaxis]
#     total_cw_prob_agn = cw_pagn[idx].T[np.newaxis,...]
#     total_cw_prob_alt = cw_palt[idx].T[np.newaxis,...]
#     total_prob_alt = palt[idx].T[np.newaxis,...]

#     if chunking:
#         log_llh_numerator = chunked_llh_processing(total_cw_prob_agn, total_cw_prob_alt, total_prob_alt, fagn_times_fobsc)
#     else:
#         log_llh_numerator_per_event = np.log(fagn_times_fobsc * total_cw_prob_agn + total_prob_alt - fagn_times_fobsc * total_cw_prob_alt)
#         log_llh_numerator = np.sum(log_llh_numerator_per_event, axis=1 )

#     log_llh_denominator = 0  # TODO: GW SELECTION EFFECTS
#     log_llh = log_llh_numerator - log_llh_denominator
#     # print('Start to finish took', time.time() - start)
#     return index, log_llh  # log_llh_x_ax[np.argmax(log_llh, axis=0)]  # TODO: change to interpolation


def process_gw(all_rlum_obs, agn_z, n_mc_samps=N_MC_SAMPS):

    p_agn = np.zeros_like(all_rlum_obs)
    p_alt = np.zeros_like(all_rlum_obs)
    for i, rlum_obs in enumerate(all_rlum_obs):
        gw_z_posterior = generate_posterior(rlum_obs)

        # Avoid outliers messing up the KDE!
        trimmed = gw_z_posterior[(gw_z_posterior > np.percentile(gw_z_posterior, 0.1)) & (gw_z_posterior < np.percentile(gw_z_posterior, 99.9))]

        if len(trimmed) < 1e4:
            kde_gw_z_posterior = gaussian_kde(trimmed)
            kde_vals = kde_gw_z_posterior(agn_z)
        else:
            # print('Histogram PDF inference is not tested yet. Check first for unbiased results using gaussian_kde, but note that evaluating a KDE is very slow.')
            kde_vals = histogram_pdf(trimmed, agn_z)
        
        p_agn[i] = np.sum(kde_vals) / len(agn_z)

        mc_samps = np.random.choice(gw_z_posterior, size=n_mc_samps)
        mc_samps_below_thresh = mc_samps[mc_samps < ZMAX]
        p_alt[i] = np.sum( dVdz_prior(mc_samps_below_thresh) ) / n_mc_samps

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
