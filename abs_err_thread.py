import numpy as np
import matplotlib.pyplot as plt
from utils import uniform_shell_sampler, make_nice_plots
from scipy import stats
from tqdm import tqdm
from concurrent.futures import as_completed, ThreadPoolExecutor
from multiprocessing import cpu_count
import traceback
import sys

make_nice_plots()


def dVdr_prior(r, rmax):
    '''Uniform in volume distribution on radius, normalized between 0 and rmax.'''
    return np.where((r < rmax) & (r > 0), r**2 / rmax**3 * 3, 0)


####################################################################
# For simulations:
DIST_ERR_DIVBY_RMAX = 0.001 # Delta R / Rmax
RMAX_GW = 1
RMIN = 0
VOLUME = 4 / 3 * np.pi * RMAX_GW**3
AGN_NUMDENS = 1e3 / VOLUME
BATCH = int(100)
N_MC_SAMPS = int(1e4)
N_TRIALS = 10000  #int( 100 * (DIST_ERR_DIVBY_RMAX / 0.01) * (AGN_NUMDENS / (100 / (4 / 3 * np.pi * 100**3))) * (BATCH / 100) )
# For log llh maximization:
MAX_N_FAGNS = 51
CALC_LOGLLH_AT_N_POINTS = 1000 # Only change if you want higher resolution

fname = 'posteriors_replaceFalse_relerr0.001_1kAGN_evalprior'

NCPU = cpu_count()

# All subsequent globals are derived from the parameters above.
GW_DIST_ERR = DIST_ERR_DIVBY_RMAX * RMAX_GW  # Absolute distance error
RTHRESH_GW = RMAX_GW
NAGN = int( np.ceil(AGN_NUMDENS * VOLUME) )
assert NAGN >= BATCH, 'Every AGN-origin GW must come from a unique AGN. Got {NAGN} AGN and {BATCH} AGN-origin GWs.'
print(f'#AGN is rounded from {AGN_NUMDENS * VOLUME} to {NAGN}, giving a number density of {NAGN / VOLUME:.3e}. Target was {AGN_NUMDENS:.3e}.')
NGW_ALT = BATCH
NGW_AGN = BATCH
N_TRUE_FAGNS = min(BATCH + 1, MAX_N_FAGNS)  # Cannot create more f_agn values than BATCH+1 and don't want to generate more than MAX_N_FAGNS
LOG_LLH_X_AX = np.linspace(0.0001, 0.9999, CALC_LOGLLH_AT_N_POINTS)

## These lines are for making the underlying truth equal to the actual realization of that truth - TODO: CURRENTLY BINOMIAL OPTION DOESN'T WORK
USE_N_AGN_EVENTS = np.arange(0, BATCH + 1, int(BATCH / (N_TRUE_FAGNS - 1)), dtype=np.int32)
TRUE_FAGNS = USE_N_AGN_EVENTS / BATCH
REALIZED_FAGNS = USE_N_AGN_EVENTS / BATCH  # Realization of the truth
rr = np.linspace(RMIN, RMAX_GW, 1000)  # x-axis for plotting
####################################################################


def process_gw_batch(obs_pos_gw, agn_pos, batch=BATCH, gw_dist_err=GW_DIST_ERR, rmax=RMAX_GW, n_mc_samps=N_MC_SAMPS):
    '''Currently tested for batch size of 100 GWs. Performance could be worse for different batch sizes.'''

    posterior = lambda x: stats.norm.pdf(x, loc=obs_pos_gw, scale=gw_dist_err)
    # posterior = lambda x: stats.truncnorm.pdf(x, (RMIN - obs_pos_gw) / gw_dist_err, (rmax - obs_pos_gw) / gw_dist_err)
    p_agn = np.sum( posterior(agn_pos[:,np.newaxis]), axis=0 ) / len(agn_pos)

    mc_samps = np.random.normal(loc=obs_pos_gw, scale=gw_dist_err, size=(n_mc_samps, batch))
    # p_alt = np.zeros(batch)
    # for i in range(batch):
    #     mc = mc_samps[(mc_samps[:,i] > RMIN) & (mc_samps[:,i] < rmax), i]
    #     p_alt[i] = np.sum(dVdr_prior(mc, rmax)) / len(mc)
    p_alt = np.sum( dVdr_prior(mc_samps, rmax), axis=0) / n_mc_samps  # NORMALIZE UP TO RMAX_GW, since p_rate(r > RMAX_GW) = 0
    # print(p_agn, p_alt)


    # r_pos = ( np.random.uniform(size=(n_mc_samps, batch), low=0.0, high=rmax**3) )**(1/3)
    # p_alt = np.sum( posterior(r_pos), axis=0 ) / len(r_pos)
    return p_agn, p_alt


def generate_and_process_universe_realization(rmin=RMIN, rmax=RMAX_GW, nagn=NAGN, batch=BATCH, gw_dist_err=GW_DIST_ERR):
    # GENERATE AGN CATALOG
    agn_pos, _, _ = uniform_shell_sampler(rmin, rmax, nagn)  # NO AGN NEEDED ABOVE RMAX_GW, p_rate = 0

    # GENERATE GWS UP TO RMAX_GW
    true_pos_gw_agn = np.random.choice(agn_pos, batch, replace=False)
    true_pos_gw_alt, _, _ = uniform_shell_sampler(rmin, rmax, batch)

    # MEASURE GW LOCATIONS
    # obs_pos_gw_agn = stats.truncnorm.rvs((rmin - true_pos_gw_agn) / gw_dist_err, (rmax - true_pos_gw_agn) / gw_dist_err)
    # obs_pos_gw_alt = stats.truncnorm.rvs((rmin - true_pos_gw_alt) / gw_dist_err, (rmax - true_pos_gw_alt) / gw_dist_err)
    obs_pos_gw_agn = np.random.normal(loc=true_pos_gw_agn, scale=gw_dist_err)
    obs_pos_gw_alt = np.random.normal(loc=true_pos_gw_alt, scale=gw_dist_err)

    # CALCULATE HYPOTHESIS EVIDENCE
    p_agn_agn_gws, p_alt_agn_gws = process_gw_batch(obs_pos_gw_agn, agn_pos)
    p_agn_alt_gws, p_alt_alt_gws = process_gw_batch(obs_pos_gw_alt, agn_pos)

    return p_agn_agn_gws, p_alt_agn_gws, p_agn_alt_gws, p_alt_alt_gws


def full_analysis_likelihood_thread(index, batch=BATCH, N_true_fagns=N_TRUE_FAGNS, use_N_agn_events=USE_N_AGN_EVENTS, log_llh_x_ax=LOG_LLH_X_AX):
    p_agn_agn_gws, p_alt_agn_gws, p_agn_alt_gws, p_alt_alt_gws = generate_and_process_universe_realization()

    ### Some translations to reuse some old code ###
    cw_pagn = np.append(p_agn_agn_gws, p_agn_alt_gws)  # Completeness-weighted AGN hypothesis evidence
    cw_palt = np.append(p_alt_agn_gws, p_alt_alt_gws)
    palt = cw_palt  # c = 1
    agn_events = np.ones(len(cw_pagn), dtype=bool)  # Mask to select GW events from AGN
    agn_events[batch:] = 0
    ##########################################

    ## Use these lines for binomial realization of truth - TODO: get this working properly
    # true_fagns = np.linspace(0, 1, N_true_fagns)  # Underlying truth
    # use_N_agn_events = np.random.binomial(n=batch, p=true_fagns)  # Make random realization of a universe with a true fagn
    # use_N_alt_events = batch - use_N_agn_events
    # realized_fagns = use_N_agn_events / batch  # Realization of the truth

    agn_idx = np.zeros((N_true_fagns, batch), dtype=int)
    alt_idx = np.zeros((N_true_fagns, batch), dtype=int)
    for k in range(N_true_fagns):
        arr = np.arange(batch)
        np.random.shuffle(arr)
        agn_idx[k,:] = arr
        np.random.shuffle(arr)
        alt_idx[k,:] = arr + batch  # Assumes array of len 2*batch, first batch are AGN events, second batch are ALT events
    idx = np.where(np.arange(batch) < use_N_agn_events[:, None], agn_idx, alt_idx)  # Shape (N_true_fagns, batch)

    # f_agn is calculated N_TRUE_FAGNS times (ax 2), using BATCH gws (ax 1), each time giving a likelihood function evaluated at CALC_LOGLLH_AT_N_POINTS points (ax 0)
    fagn_times_fobsc = log_llh_x_ax[:, np.newaxis, np.newaxis]
    total_cw_prob_agn = cw_pagn[idx].T[np.newaxis,...]
    total_cw_prob_alt = cw_palt[idx].T[np.newaxis,...]
    total_prob_alt = palt[idx].T[np.newaxis,...]

    log_llh_numerator_per_event = np.log(fagn_times_fobsc * total_cw_prob_agn + total_prob_alt - fagn_times_fobsc * total_cw_prob_alt)
    log_llh_numerator = np.sum(log_llh_numerator_per_event, axis=1 )  # Sum over GW events
    log_llh_denominator = 0  # TODO: gw selection effects
    log_llh = log_llh_numerator - log_llh_denominator

    return index, log_llh

    # return index, log_llh_x_ax[np.argmax(log_llh, axis=0)]  # Maximize along logllh axis - TODO: change to interpolation


if __name__ == '__main__':

    # estimation_arr = np.zeros((N_TRIALS, N_TRUE_FAGNS))
    posteriors = np.zeros((N_TRIALS, CALC_LOGLLH_AT_N_POINTS, N_TRUE_FAGNS))

    # If you don't want to use threading, uncomment this
    # for i in range(N_TRIALS):
    #     _, estimation_arr[i,:] = full_analysis_likelihood_thread(i)

    # If you don't want to use threading, comment this
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
    # combined_posteriors = np.sum(posteriors, axis=0)

    # estimated_fagns = LOG_LLH_X_AX[np.argmax(combined_posteriors, axis=0)]

    # # fagn_medians = np.median(combined_posteriors, axis=0)
    # # q016 = np.quantile(combined_posteriors, 0.16, axis=0)
    # # q084 = np.quantile(combined_posteriors, 0.84, axis=0)

    # plt.figure(figsize=(8,8))
    # plt.plot(TRUE_FAGNS, estimated_fagns, color='red', linewidth=3)
    # plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), linestyle='dashed', color='black', zorder=6, linewidth=3)
    # # plt.fill_between(TRUE_FAGNS, q016, q084, color='red', alpha=0.3)
    # plt.xlabel(r'$f_{\rm agn, true}$')
    # plt.ylabel(r'$f_{\rm agn, estim}$')
    # plt.grid()
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.savefig('abserr.pdf')
    # plt.show()

    # relative_estimates = estimated_fagns - TRUE_FAGNS
    # # q016 = np.quantile(relative_estimates, 0.16, axis=0)
    # # q084 = np.quantile(relative_estimates, 0.84, axis=0)
    # plt.figure(figsize=(8,8))
    # plt.plot(TRUE_FAGNS, np.median(relative_estimates, axis=0), color='red', linewidth=3, label='Median')
    # plt.plot(TRUE_FAGNS, np.mean(relative_estimates, axis=0), color='blue', linewidth=3, label='Mean')
    # plt.plot(np.linspace(0,1,100), np.zeros(100), linestyle='dashed', color='black', zorder=6, linewidth=3)
    # # plt.fill_between(TRUE_FAGNS, q016, q084, color='red', alpha=0.3, label=r'$68\%$ CI')
    # plt.xlabel(r'$f_{\rm agn, true}$')
    # plt.ylabel(r'$f_{\rm agn, estim} - f_{\rm agn, true}$')
    # plt.grid()
    # plt.legend()
    # plt.xlim(0, 1)
    # plt.savefig('abserr_diff.pdf')
    # plt.show()

    # # print(np.std(relative_estimates, axis=0), np.mean(np.std(relative_estimates, axis=0)), np.median(np.std(relative_estimates, axis=0)))