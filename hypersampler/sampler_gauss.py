import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from darksirenpop.run import *
from darksirenpop.utilities.redshift_utils import *
from scipy.integrate import romb
from darksirenpop.utilities.default_globals import *
from darksirenpop.utilities.gw_selection_effects import get_alpha_alt
from darksirenpop.likelihood import *
import emcee

from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# Define constants and priors
ncores = 5
ndim = 4
nwalkers = 100
nsteps = 10000
LTHRESH_STRING = '45.5'

par_names = np.array(['b', 'c', 'd', 'fagn'])
par_names_latex = np.array([r'$\gamma$', r'$1 + z_{\rm peak}$', r'$\alpha$', r'$f_{\rm agn}$'])
# par_names = np.array(['b', 'c', 'd', 'fagn', 'zmax'])
# par_names_latex = np.array([r'$\gamma$', r'$1 + z_{\rm peak}$', r'$\alpha$', r'$f_{\rm agn}$', r'$z_{\rm max}$'])


######### GAUSSIAN PRIORS ON MD + UNIFORM ON FAGN #########
filename = f"/home/lucas/Documents/PhD/generated_data/alt_origin_hyperparameters/sampler_gauss_priors_withcat_{LTHRESH_STRING}.h5"
backend = emcee.backends.HDFBackend(filename)
# reset if starting a new run
# backend.reset(nwalkers, ndim)

PRIOR_MU = np.array([3.3, 2.55, 6.1])  # b, c, d = low-z PL slope, 1 + z_peak, high-z PL slope
PRIOR_SIGMA = np.array([0.2, 0.09, 0.2])
FAGN_PRIOR = [0, 1]

def log_prior(theta):
    theta = np.asarray(theta)
    gauss_params = theta[:3]
    fagn = theta[3]

    # Gaussian priors for first 3 parameters
    logp_gauss = -0.5 * np.sum(
        ((gauss_params - PRIOR_MU) / PRIOR_SIGMA) ** 2
        + np.log(2 * np.pi * PRIOR_SIGMA**2)
    )

    # Uniform prior for fagn
    if FAGN_PRIOR[0] <= fagn <= FAGN_PRIOR[1]:
        logp_fagn = 0.0   # constant, can omit
    else:
        logp_fagn = -np.inf

    return logp_gauss + logp_fagn


# Initial positions
pos = np.zeros((nwalkers, ndim))
pos[:, :len(PRIOR_MU)] = np.random.normal(PRIOR_MU, PRIOR_SIGMA, size=(nwalkers, len(PRIOR_MU)))
pos[:, -1] = np.random.uniform(FAGN_PRIOR[0], FAGN_PRIOR[1], nwalkers)

####################################


######### UNIFORM PRIORS #########
# filename = "/home/lucas/Documents/PhD/darksirenpop/sampler_uniform_priors_withcat.h5"
# backend = emcee.backends.HDFBackend(filename)
# # reset if starting a new run
# # backend.reset(nwalkers, ndim)

# b_prior = [-10, 10]
# c_prior = [1, 3.5]
# d_prior = [0, 10]
# fagn_prior = [0, 1]
# # zmax_prior = [2, 10]

# # MCMC_PRIOR = np.array([b_prior, c_prior, d_prior, fagn_prior, zmax_prior])
# MCMC_PRIOR = np.array([b_prior, c_prior, d_prior, fagn_prior])

# def log_prior(theta):
#     '''Uniform prior in all parameters'''
#     for i in range(0, len(theta)):
#         if (theta[i] < MCMC_PRIOR[i, 0]) or (theta[i] > MCMC_PRIOR[i, 1]):
#             return -np.inf
#     return 0.0

# # Initial positions
# pos = np.zeros((nwalkers, ndim))
# pos[:, 0] = np.random.uniform(b_prior[0], b_prior[1], nwalkers)
# pos[:, 1] = np.random.uniform(c_prior[0], c_prior[1], nwalkers)
# pos[:, 2] = np.random.uniform(d_prior[0], d_prior[1], nwalkers)
# pos[:, 3] = np.random.uniform(fagn_prior[0], fagn_prior[1], nwalkers)
# # pos[:, 4] = np.random.uniform(zmax_prior[0], zmax_prior[1], nwalkers)

####################################


# def log_prior(theta):
#     '''Gaussian prior in all parameters'''
#     theta = np.asarray(theta)

#     return -0.5 * np.sum(
#         ((theta - prior_mu) / prior_sigma) ** 2
#         + np.log(2 * np.pi * prior_sigma**2)
#     )


np.seterr(divide='ignore')

zmodel = 'madau'
thresh = f'{LTHRESH_STRING}_kulkarni'
label = '_zleq3_final_kmax5_gwtc5_luminformed_smoothcorr'
agn_json_path = f'/home/lucas/Documents/PhD/generated_data/jsons/real_output_{zmodel}{label}.json'
# json_emptycat = f'/home/lucas/Documents/PhD/gw_data/real_output_nocat_{zmodel}{label}.json'

with open(agn_json_path, "r") as f:
    data = json.load(f)
    GW_EVIDENCE_DICT = data[str(thresh)]


def log_likelihood(theta):
    agn_zcut = 3.0  # 1.5

    b, c, d, fagn = theta
    rate_param_dict = {'b': b, 'c': c, 'd': d}

    cfg = Config()
    cfg.RATE_PARAMETERS = rate_param_dict
    cfg.VERBOSE = False
    cfg.THREADING = False
    cfg.REAL_DATA = True
    cfg.OUTFILE = 'none'
    cfg.AGN_DIST_DIR = '/home/lucas/Documents/PhD/generated_data/em'
    cfg.CATALOG_PATH = '/home/lucas/Documents/PhD/generated_data/em/quaia_zleq3_withlumcorr.csv'  # '/home/lucas/Documents/PhD/agn_data/Quaia_z15.csv'
    cfg.CMAP_PATH = '/home/lucas/Documents/PhD/completeness_map.fits'
    cfg.ZMIN = 0.000001
    cfg.ZMAX = 10
    cfg.AGN_ZMAX = 10
    cfg.AGN_ZCUT = agn_zcut
    cfg.AGN_ZPRIOR = LTHRESH_STRING
    cfg.LUM_THRESH = LTHRESH_STRING
    cfg.MASK_GALACTIC_PLANE = True
    cfg.ASSUME_PERFECT_REDSHIFT = False
    cfg.AGN_ZERROR = 'quaia'
    cfg.CORRECT_TIME_DILATION = True
    cfg.MERGER_RATE = 'madau'
    cfg.LABEL = 'samplinggwtc5'
    cfg.finalize()



    with open(cfg.JSON_PATH, "r") as f:
        gw_path_dict = json.load(f)
    gw_keys = list(gw_path_dict.keys())
    gw_fnames = [gw_path_dict[key] for key in gw_keys]

    ### Prepare functions for in the likelihood ###
    if cfg.CORRECT_TIME_DILATION:
        time_dilation_func = lambda z: time_dilation_correction(z)
    else: 
        time_dilation_func = lambda z: np.ones_like(z)
    time_dilation = time_dilation_func(cfg.Z_INTEGRAL_AX)
    zcut = z_cut(cfg.Z_INTEGRAL_AX, zcut=cfg.ZMAX)
    zrate_alt = merger_rate(cfg.Z_INTEGRAL_AX, cfg.MERGER_RATE_EVOLUTION, **cfg.MERGER_RATE_KWARGS)
    p_rate_of_z_alt = time_dilation * zrate_alt * zcut
    PEprior_func = lambda z: uniform_comoving_prior(z, cosmo=cfg.COSMO)
    PEprior = PEprior_func(cfg.Z_INTEGRAL_AX)
    dz, jacobian = get_dz_and_jacobian(cfg)

    alpha_alt = get_alpha_alt(snr_thr=cfg.SNR_THR, 
                              far_thr=cfg.FAR_THR, 
                              alt_rate_model=cfg.MERGER_RATE, 
                              alt_rate_parameters=cfg.RATE_PARAMETERS, 
                              zmax=cfg.ZMAX,
                              agn_dist_dir=cfg.AGN_DIST_DIR)
    alpha_agn = GW_EVIDENCE_DICT[gw_keys[0]]['alpha_agn']  # Same for all events


    ### Calculate the integrals in the likelihood ###
    Ngws = len(gw_fnames)  # Due to selection effects not always the same number
    S_agn_incat = np.zeros(Ngws)
    S_agn_outofcat = np.zeros(Ngws)
    S_alt = np.zeros(Ngws)
    for gw_idx, _ in enumerate(gw_fnames):
        gwkey = gw_keys[gw_idx]

        with open(cfg.REAL_ZPOSTS_JSON_PATH, "r") as f:
            gw_zpost_path_dict = json.load(f)
            gw_zpost_path = gw_zpost_path_dict[gwkey]
        with open(cfg.REAL_CW_ZPOSTS_JSON_PATH, "r") as f:
            gw_zpost_cw_path_dict = json.load(f)
            gw_zpost_cw_path = gw_zpost_cw_path_dict[gwkey]
        z, p = np.load(gw_zpost_path)
        z_cw, p_cw = np.load(gw_zpost_cw_path)
        gwpost_interp = CubicSpline(z, p, extrapolate=False)
        gw_redshift_posterior_marginalized_evaluated = gwpost_interp(cfg.Z_INTEGRAL_AX)
        gw_redshift_posterior_marginalized_evaluated[np.isnan(gw_redshift_posterior_marginalized_evaluated)] = 0  # NaNs outside extrapolation range changed to zeros

        if not cfg.MASK_GALACTIC_PLANE:
            gw_redshift_posterior_marginalized_cw_evaluated = gw_redshift_posterior_marginalized_evaluated.copy()
        else:
            gwpost_interp_cw = CubicSpline(z_cw, p_cw, extrapolate=False)
            gw_redshift_posterior_marginalized_cw_evaluated = gwpost_interp_cw(cfg.Z_INTEGRAL_AX)
            gw_redshift_posterior_marginalized_cw_evaluated[np.isnan(gw_redshift_posterior_marginalized_cw_evaluated)] = 0  # NaNs outside extrapolation range changed to zeros'

        background_alt_distribution = uniform_comoving_prior(cfg.Z_INTEGRAL_AX, cosmo=cfg.COSMO)
        alt_redshift_population_prior = background_alt_distribution * p_rate_of_z_alt
        alt_redshift_population_prior /= romb(alt_redshift_population_prior * jacobian, dx=dz)
        salt = romb(y=gw_redshift_posterior_marginalized_evaluated / PEprior * alt_redshift_population_prior * jacobian, dx=dz)

        sagn_incat = GW_EVIDENCE_DICT[gwkey]['S_agn_incat']
        sagn_outofcat = GW_EVIDENCE_DICT[gwkey]['S_agn_outcat']

        S_agn_incat[gw_idx] = sagn_incat
        S_agn_outofcat[gw_idx] = sagn_outofcat
        S_alt[gw_idx] = salt

    ### Evaluate the likelihood ###
    S_agn_incat = S_agn_incat[~np.isnan(S_agn_incat)]
    S_agn_outofcat = S_agn_outofcat[~np.isnan(S_agn_outofcat)]
    S_alt = S_alt[~np.isnan(S_alt)]

    loglike = np.log(cfg.LOG_LLH_X_AX[None,:] * (S_agn_incat[:,None] + S_agn_outofcat[:,None] - S_alt[:,None]) + S_alt[:,None])
    loglike = np.sum(loglike, axis=0)  # Sum over all GWs
    loglike -= Ngws * np.log(alpha_agn * cfg.LOG_LLH_X_AX + alpha_alt * (1 - cfg.LOG_LLH_X_AX))  # Correct selection effects

    interped_log_llh = CubicSpline(LOG_LLH_X_AX, loglike)

    return interped_log_llh(fagn)


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend)
# sampler.run_mcmc(pos, nsteps, progress=True)

with Pool(processes=ncores, maxtasksperchild=50) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, backend=backend)
    # sampler.run_mcmc(pos, nsteps, progress=True)

    try:
        state = sampler.get_last_sample()
    except AttributeError:
        state = pos

    for i, state in tqdm(enumerate(sampler.sample(state, iterations=nsteps))):
        continue
        # if (i % 2 == 0) and (i != 0) :
            # print(f"Step {i + 1}/{nsteps}")

            # get_diagnostics(sampler, burnin=0, acceptance_fraction_low=0, acceptance_fraction_high=1)
