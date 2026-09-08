from ligo.skymap.io.fits import read_sky_map
from ligo.skymap import moc

from pathlib import Path

from darksirenpop.utilities.gw_selection_effects import alpha
from darksirenpop.utilities.mockdata_utils import *

import sys, os
import h5py
import healpy as hp
import numpy as np
import astropy_healpix as ah
import glob
import json
import time

from scipy.integrate import romb
from scipy.interpolate import interp1d, CubicSpline


def get_dz_and_jacobian(cfg):
    if cfg.LINAX:
        dz = np.diff(cfg.Z_INTEGRAL_AX)[0]
        jacobian = 1
    else:
        dz = np.diff(np.log10(cfg.Z_INTEGRAL_AX))[0]
        jacobian = cfg.Z_INTEGRAL_AX * np.log(10)
    return dz, jacobian


# from numba import njit, prange
# @njit(parallel=True, cache=True)
# def compute_integrand(agn_posteriors, gw_posteriors, agn_pix_labels, dP_dA_per_agn, n_zbins):
#     integrand = np.zeros(n_zbins, dtype=np.float32)
#     for z in prange(n_zbins):
#         acc = np.float32(0.0)
#         for a in range(agn_posteriors.shape[0]):
#             acc += dP_dA_per_agn[a] * gw_posteriors[z, agn_pix_labels[a]] * agn_posteriors[a, z]
#         integrand[z] = acc
#     return integrand


def get_gw_zpost(filename, cfg, from_agn_hdf5=None, from_alt_hdf5=None, from_agn_cw_hdf5=None, from_alt_cw_hdf5=None, gwkey=None):
    '''
    Load pre-calculated GW redshift posterior and interpolate with scipy.interpolate.CubicSpline to the redshift integral axis.
    We make a distinction between the total redshift posterior and the posterior weighted by the survey footprint.
    '''

    if cfg.REAL_DATA:  # TODO: Put evaluated sky maps for real data in single hdf5-file as well. Or else load the jsons outside the loop, to avoid opening and closing many times per run.
        with open(cfg.REAL_ZPOSTS_JSON_PATH, "r") as f:
            gw_zpost_path_dict = json.load(f)
            gw_zpost_path = gw_zpost_path_dict[gwkey]

        with open(cfg.REAL_CW_ZPOSTS_JSON_PATH, "r") as f:
            gw_zpost_cw_path_dict = json.load(f)
            gw_zpost_cw_path = gw_zpost_cw_path_dict[gwkey]

        z, p = np.load(gw_zpost_path)
        z_cw, p_cw = np.load(gw_zpost_cw_path)

    else:
        agn_or_alt = filename.split('/')[-2]
        if agn_or_alt == 'agn':
            zpost_file = from_agn_hdf5
            cw_zpost_file = from_agn_cw_hdf5
        elif agn_or_alt == 'alt':
            zpost_file = from_alt_hdf5
            cw_zpost_file = from_alt_cw_hdf5
        else:
            sys.exit(f'Do not recognize subdirectory: {agn_or_alt}. Expected "agn" or "alt".')

        gw_id = filename[-13:-8]
        z = zpost_file[str(gw_id)]['eval_ax'][:]
        p = zpost_file[str(gw_id)]['posterior'][:]
        z_cw = cw_zpost_file[str(gw_id)]['eval_ax'][:]
        p_cw = cw_zpost_file[str(gw_id)]['posterior'][:]
    
    gwpost_interp = CubicSpline(z, p, extrapolate=False)
    gw_redshift_posterior_marginalized_evaluated = gwpost_interp(cfg.Z_INTEGRAL_AX)
    gw_redshift_posterior_marginalized_evaluated[np.isnan(gw_redshift_posterior_marginalized_evaluated)] = 0  # NaNs outside extrapolation range changed to zeros

    if not cfg.MASK_GALACTIC_PLANE:
        gw_redshift_posterior_marginalized_cw_evaluated = gw_redshift_posterior_marginalized_evaluated.copy()
    else:
        gwpost_interp_cw = CubicSpline(z_cw, p_cw, extrapolate=False)
        gw_redshift_posterior_marginalized_cw_evaluated = gwpost_interp_cw(cfg.Z_INTEGRAL_AX)
        gw_redshift_posterior_marginalized_cw_evaluated[np.isnan(gw_redshift_posterior_marginalized_cw_evaluated)] = 0  # NaNs outside extrapolation range changed to zeros

    return gw_redshift_posterior_marginalized_evaluated, gw_redshift_posterior_marginalized_cw_evaluated


def calculate_evidence(
            cfg,
            filename,
            agn_posterior_dset,
            agn_ra, 
            agn_dec,
            agn_redshift,
            p_rate_of_z_agn_func,
            p_rate_of_z_agn,
            p_rate_of_z_alt,
            PEprior_func,
            PEprior,
            fc_of_z,
            average_completeness,
            sky_coverage,
            normed_agn_background_dist,
            nagn_norm,
            agn_population_prior_normalization,
            from_agn_hdf5, 
            from_alt_hdf5, 
            from_agn_cw_hdf5, 
            from_alt_cw_hdf5,
            gwkey
        ):

    dz, jacobian = get_dz_and_jacobian(cfg)

    sky_map = read_sky_map(filename, moc=True)
    sky_map = np.flipud(np.sort(sky_map, order="PROBDENSITY"))
    
    # Unpacking skymap
    norm = sky_map["DISTNORM"]      # Ansatz norm in 1/Mpc^2
    norm[np.isinf(norm)] = 0        # Infs are observed to happen rarely in low-probability sky regions, this line avoids nans later if using CL->1
    bad_pixels = np.isnan(norm)     # NaNs occur very rarely. Seen coinciding with sigma = inf

    norm = norm[~bad_pixels]
    dP_dA = sky_map["PROBDENSITY"][~bad_pixels]  # Probdens in 1/sr
    mu = sky_map["DISTMU"][~bad_pixels]          # Ansatz mean in Mpc
    sigma = sky_map["DISTSIGMA"][~bad_pixels]    # Ansatz width in Mpc
    skymap_uniq = sky_map["UNIQ"][~bad_pixels]

    if np.sum(np.isnan(dP_dA)) > 0:
        print('BAD SKYMAP')
        return np.nan, np.nan, np.nan

    # Find the pixels that contain AGN
    order, ipix = moc.uniq2nest(skymap_uniq)
    max_order = np.max(order)
    max_nside = ah.level_to_nside(max_order)
    max_ipix = ipix << np.int64(2 * (max_order - order))

    agn_theta = 0.5 * np.pi - agn_dec
    agn_phi = agn_ra
    agn_pix = hp.ang2pix(max_nside, agn_theta, agn_phi, nest=True)
    i = np.argsort(max_ipix)
    gw_pixidx_at_agn_locs = i[np.digitize(agn_pix, max_ipix[i]) - 1]  # Indeces that indicate skymap pixels that contain an AGN

    del agn_theta
    del agn_phi
    del agn_ra
    del agn_dec
    del agn_pix

    dA = moc.uniq2pixarea(skymap_uniq)  # Pixel areas in sr
    dP = dP_dA * dA  # Dimensionless probability density in each pixel
    cumprob = np.cumsum(dP)
    cumprob[cumprob > 1] = 1.  # Correcting floating point error which could cause issues when skymap_cl == 1
    searched_prob_at_agn_locs = cumprob[gw_pixidx_at_agn_locs]

    # Getting only relevant AGN and pixels from the skymap
    agn_within_cl_mask = (searched_prob_at_agn_locs <= cfg.SKYMAP_CL)
    nagn_within_cl = np.sum(agn_within_cl_mask)

    # Load pre-calculated GW redshift posteriors
    if cfg.FLAT_GW_POSTERIORS:  # For testing
        gw_redshift_posterior_marginalized_evaluated = PEprior.copy()
        gw_redshift_posterior_marginalized_cw_evaluated = PEprior.copy() * sky_coverage
    else:
        gw_redshift_posterior_marginalized_evaluated, gw_redshift_posterior_marginalized_cw_evaluated = get_gw_zpost(filename, 
                                                                                                                     cfg, 
                                                                                                                     from_agn_hdf5=from_agn_hdf5, 
                                                                                                                     from_alt_hdf5=from_alt_hdf5, 
                                                                                                                     from_agn_cw_hdf5=from_agn_cw_hdf5, 
                                                                                                                     from_alt_cw_hdf5=from_alt_cw_hdf5, 
                                                                                                                     gwkey=gwkey)

    ####################### Integrals #######################

    ### Alternative-origin population part ###
    
    # int dz p(z|d_gw)/PEprior(z) * p_pop(z | \conj{A}, \conj{G}): unif. in com.vol.
    background_alt_distribution = uniform_comoving_prior(cfg.Z_INTEGRAL_AX, cosmo=cfg.COSMO)
    alt_redshift_population_prior = background_alt_distribution * p_rate_of_z_alt
    alt_redshift_population_prior /= romb(alt_redshift_population_prior * jacobian, dx=dz)
    S_alt = romb(y=gw_redshift_posterior_marginalized_evaluated / PEprior * alt_redshift_population_prior * jacobian, dx=dz)

    ### AGN-origin population part ###

    # Out-of-catalogue part 

    # Calculated in 2 parts, otherwise, S_agn_outofcat could become slightly negative. TODO: test again how important this even is after other bug fixes
    a = romb(y=gw_redshift_posterior_marginalized_evaluated / PEprior * p_rate_of_z_agn * normed_agn_background_dist * jacobian, dx=dz)
    b = romb(y=fc_of_z * gw_redshift_posterior_marginalized_cw_evaluated / PEprior * p_rate_of_z_agn * normed_agn_background_dist * jacobian, dx=dz)
    S_agn_outofcat = (a - b) / agn_population_prior_normalization

    # S_agn_outofcat = romb(y=(gw_redshift_posterior_marginalized_evaluated - fc_of_z * gw_redshift_posterior_marginalized_cw_evaluated) / PEprior * p_rate_of_z_agn * normed_agn_background_dist * jacobian, dx=dz) / agn_population_prior_normalization   
    if S_agn_outofcat < 0:
        print(f'GOT NEGATIVE: {filename}, {S_agn_outofcat}')
    
    # In-catalogue part
    if (nagn_within_cl == 0) or (nagn_norm == 0):
        S_agn_incat = 0
        return S_agn_incat, S_agn_outofcat, S_alt
    
    gw_pixidx_at_agn_locs_within_cl = gw_pixidx_at_agn_locs[agn_within_cl_mask]
    unique_gw_pixidx_containing_agn = np.unique(gw_pixidx_at_agn_locs_within_cl)  # We only need to consider the GW pixels with catalog support
    distnorm_allpix, distmu_allpix, distsigma_allpix = norm[unique_gw_pixidx_containing_agn], mu[unique_gw_pixidx_containing_agn], sigma[unique_gw_pixidx_containing_agn]
    # print(f'Found {nagn_within_cl} AGN within {skymap_cl} CL in {len(unique_gw_pixidx_containing_agn)} pixels')
    
    if cfg.ASSUME_PERFECT_REDSHIFT:  # Delta-function AGN posteriors make the calculations easier

        agn_redshifts_within_cl = agn_redshift[agn_within_cl_mask]
        agn_posterior_idx = np.arange(nagn_within_cl)

        S_agn_incat = 0
        for i, gw_idx in enumerate(unique_gw_pixidx_containing_agn):
            norm_in_pix, mu_in_pix, sig_in_pix = distnorm_allpix[i], distmu_allpix[i], distsigma_allpix[i]
            gw_redshift_posterior_in_pix = lambda z: redshift_pdf_given_lumdist_pdf(z, LOS_lumdist_ansatz, distnorm=norm_in_pix, distmu=mu_in_pix, distsigma=sig_in_pix, cosmo=cfg.COSMO)
            agn_posterior_idx_in_pix = agn_posterior_idx[gw_pixidx_at_agn_locs_within_cl == gw_idx]
            selected_agn_redshifts = agn_redshifts_within_cl[agn_posterior_idx_in_pix]

            if cfg.FLAT_GW_POSTERIORS:
                gw_redshift_posterior_in_pix = PEprior_func
                dP_dA[gw_idx] = 1 / (4 * np.pi)
            
            # p(s|z) * p_gw(z) * p_gw(Omega) / pi_PE(z), evaluated at AGN position because of delta-function AGN posteriors, sum contributions of all AGN in this pixel
            S_agn_incat += dP_dA[gw_idx] * np.sum( p_rate_of_z_agn_func(selected_agn_redshifts) * gw_redshift_posterior_in_pix(selected_agn_redshifts) / PEprior_func(selected_agn_redshifts) )

        S_agn_incat *= 4 * np.pi * average_completeness / nagn_norm / agn_population_prior_normalization

    else:  # AGN have z-errors, need to use their full posteriors
        gw_redshift_posterior_in_allpix = redshift_pdf_given_lumdist_pdf(cfg.Z_INTEGRAL_AX[:,np.newaxis], 
                                                                         LOS_lumdist_ansatz, 
                                                                         distnorm=distnorm_allpix[np.newaxis, :], 
                                                                         distmu=distmu_allpix[np.newaxis, :], 
                                                                         distsigma=distsigma_allpix[np.newaxis, :], 
                                                                         cosmo=cfg.COSMO)  # Vectorized evaluation of the GW posteriors for all unique relevant pixels - requires sufficient RAM to comfortably handle arrays of (npix with agn)*len(z-array) elements        

        agn_redshift_posteriors_in_cl = agn_posterior_dset[agn_within_cl_mask,:]  # Loading the AGN posteriors
        agn_posterior_idx = np.arange(nagn_within_cl)

        # Building p_pop(z|A,G) * p_GW(z|d)
        integrand = np.zeros_like(cfg.Z_INTEGRAL_AX)  # AGN posteriors weighted by GW sky posterior, to be integrated over redshift
        # LOSzprior = np.zeros_like(cfg.Z_INTEGRAL_AX)  # Needed for normalization of population prior

        for i, gw_idx in enumerate(unique_gw_pixidx_containing_agn):
            gw_redshift_posterior_in_pix = gw_redshift_posterior_in_allpix[:, i]
            # gw_redshift_posterior_in_pix = redshift_pdf_given_lumdist_pdf(cfg.Z_INTEGRAL_AX, LOS_lumdist_ansatz, distnorm=distnorm_allpix[i], distmu=distmu_allpix[i], distsigma=distsigma_allpix[i], cosmo=cfg.COSMO)
            # print(len(distnorm_allpix), len(cfg.Z_INTEGRAL_AX))

            if cfg.FLAT_GW_POSTERIORS:
                gw_redshift_posterior_in_pix = PEprior.copy()
                dP_dA[gw_idx] = 1 / (4 * np.pi)

            agn_posterior_idx_in_pix = agn_posterior_idx[gw_pixidx_at_agn_locs_within_cl == gw_idx]
            agn_redshift_posteriors_in_pix = agn_redshift_posteriors_in_cl[agn_posterior_idx_in_pix, :]

            # The population prior consists of AGN posteriors, modulated by redshift evolving merger rates (done later)
            sum_of_agn_posteriors = np.sum(agn_redshift_posteriors_in_pix, axis=0)

            del agn_redshift_posteriors_in_pix

            # LOSzprior += sum_of_agn_posteriors
            integrand += dP_dA[gw_idx] * gw_redshift_posterior_in_pix * sum_of_agn_posteriors

        del gw_redshift_posterior_in_allpix
        del agn_redshift_posteriors_in_cl

        # Normalize
        integrand /= nagn_norm
        # LOSzprior /= nagn_norm

        # Calculate evidence
        S_agn_incat = romb(integrand * p_rate_of_z_agn / PEprior * jacobian, dx=dz) * 4 * np.pi * average_completeness / agn_population_prior_normalization  # 1/4pi from PEprior does not cancel, since the AGN sky posterior is delta(Omega_i - Omega)

    return S_agn_incat, S_agn_outofcat, S_alt


def load_quaia(fagn_idx, cfg):
    '''
    Load Quaia and select sources outside the galactic plane and above the specified bolometric luminosity
    '''
    if cfg.LUM_THRESH == 'inf':
        return np.empty((0, len(cfg.Z_INTEGRAL_AX))), np.empty(0), np.empty(0), np.empty(0), lambda z: np.zeros_like(z)

    with h5py.File(f'{cfg.AGN_DIST_DIR}/quaia_zleq{cfg.AGN_ZCUT}_{cfg.LUM_THRESH}_{cfg.QLF}.hdf5', 'r') as f:
        agn_redshift = f['redshift'][()]
        agn_ra = np.deg2rad( f['ra'][()] )
        agn_dec = np.deg2rad( f['dec'][()] )
        agn_posterior_dset = f['agn_redshift_posteriors'][()]

    filename = f'{cfg.AGN_DIST_DIR}/completeness_zleq{cfg.AGN_ZCUT}_{cfg.LUM_THRESH}_{cfg.QLF}.npy'
    if cfg.VERBOSE:
        print(f'Loading continuous selection function calculated from QLF from file: {filename}')
    z, fc_of_z = np.load(filename)
    c_above_1 = fc_of_z > 1
    fc_of_z[c_above_1] = 1.
    redshift_completeness = interp1d(z, fc_of_z, bounds_error=False, fill_value=0)

    return agn_posterior_dset, agn_ra, agn_dec, agn_redshift, redshift_completeness


def prepare_functions(cfg, redshift_completeness):
    if cfg.CORRECT_TIME_DILATION:
        time_dilation_func = lambda z: time_dilation_correction(z)
    else: 
        time_dilation_func = lambda z: np.ones_like(z)
    time_dilation = time_dilation_func(cfg.Z_INTEGRAL_AX)

    p_rate_of_z_agn_func = lambda z: time_dilation_func(z) * z_cut(z, zcut=cfg.ZMAX)
    p_rate_of_z_agn = p_rate_of_z_agn_func(cfg.Z_INTEGRAL_AX)

    zcut = z_cut(cfg.Z_INTEGRAL_AX, zcut=cfg.ZMAX)
    zrate_alt = merger_rate(cfg.Z_INTEGRAL_AX, cfg.MERGER_RATE_EVOLUTION, **cfg.MERGER_RATE_KWARGS)
    p_rate_of_z_alt = time_dilation * zrate_alt * zcut

    PEprior_func = lambda z: uniform_comoving_prior(z, cosmo=cfg.COSMO)  # PE prior hard-coded, but consistent with mock and real data TODO: generalize?
    PEprior = PEprior_func(cfg.Z_INTEGRAL_AX)
    
    dz, jacobian = get_dz_and_jacobian(cfg)
    normed_agn_background_dist = cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX) / romb(cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX) * jacobian, dx=dz)  # 1/4pi cancels with sky position PEprior

    # # Get survey footprint
    # skymap_theta, skymap_phi = moc.uniq2ang(sky_map['UNIQ'])
    # cmap_nside = hp.npix2nside(len(completeness_map))
    # pix_idx = hp.ang2pix(cmap_nside, skymap_theta, skymap_phi, nest=True)
    # cmap_vals_in_gw_skymap = completeness_map[pix_idx]
    # surveyed = (cmap_vals_in_gw_skymap != 0)
    # sky_coverage = np.sum(dA[surveyed]) / np.sum(dA)
    if cfg.MASK_GALACTIC_PLANE:
        sky_coverage = 1 - np.sin(np.deg2rad(10))  # FIXME: Hard-coded for now
    else:
        sky_coverage = 1.
    
    fc_of_z = redshift_completeness(cfg.Z_INTEGRAL_AX)
    average_redshift_completeness = romb(fc_of_z * normed_agn_background_dist * jacobian, dx=dz)
    average_completeness = average_redshift_completeness * sky_coverage

    fc_and_rate_weighted_agn_background_dist = (1 - fc_of_z) * normed_agn_background_dist * p_rate_of_z_agn
    return (average_completeness, 
            average_redshift_completeness, 
            sky_coverage, 
            fc_of_z, 
            p_rate_of_z_agn, 
            p_rate_of_z_alt, 
            p_rate_of_z_agn_func, 
            PEprior, 
            PEprior_func, 
            normed_agn_background_dist, 
            fc_and_rate_weighted_agn_background_dist, 
            jacobian, 
            dz)


def calculate_normalizations(cfg, 
                             agn_posterior_dset,
                             obs_agn_redshift, 
                             average_redshift_completeness,
                             p_rate_of_z_agn,
                             p_rate_of_z_agn_func,
                             fc_and_rate_weighted_agn_background_dist,
                             jacobian,
                             dz
                             ):
    '''
    There are three normalizations to calculate:

    1. The number of AGN by which to normalize the in-catalog redshift prior
    This is given by the number of AGN that could host a GW (AGN above ZMAX should not contribute to the z-prior, they have a weight of 0).

    2. The normalization of the full redshift population prior
    This needs to be done numerically, because of the 1/(1 + z) pre-factor

    3. The normalization of the likelihood, due to the limited GW detection efficiency
    The AGN-origin and alternative-origin GWs have different redshift distributions, and therefore different detection efficiencies.
    '''

    # In the real data case, the detection efficiency is calculated with an injection campaign
    # For mock data, we use the Pdet function obtained from calc_mock_pdet.ipynb
    if not cfg.REAL_DATA:  
        alpha_alt = cfg.ALPHA_ALT
        Pdet = cfg.PDET
        pdet = Pdet(cfg.Z_INTEGRAL_AX)
        pdet[np.isnan(pdet)] = 0

    # Get zprior normalizations, dealing with delta-function AGN posteriors (then assume_perfect_redshift == True) and empty catalogues (then nagn_norm == 0)
    if cfg.ASSUME_PERFECT_REDSHIFT:
        agn_below_zmax_mask = obs_agn_redshift < cfg.ZMAX
        agn_below_zmax = obs_agn_redshift[agn_below_zmax_mask]
        nagn_norm = np.sum(agn_below_zmax_mask)

        if nagn_norm == 0:
            agn_population_prior_normalization = romb(fc_and_rate_weighted_agn_background_dist * jacobian, dx=dz)
            if not cfg.REAL_DATA:
                alpha_agn = romb(pdet * fc_and_rate_weighted_agn_background_dist * jacobian, dx=dz) / agn_population_prior_normalization
        else:            
            agn_population_prior_normalization = average_redshift_completeness * np.sum(p_rate_of_z_agn_func(agn_below_zmax)) / nagn_norm + romb(agn_population_prior_rate_weighted * jacobian, dx=dz)
            if not cfg.REAL_DATA:
                pdet_at_agnz = Pdet(agn_below_zmax)
                pdet_at_agnz[np.isnan(pdet_at_agnz)] = 0
                
                alpha_agn = np.sum( pdet_at_agnz * average_redshift_completeness * p_rate_of_z_agn_func(agn_below_zmax) / nagn_norm )
                alpha_agn += romb(pdet * fc_and_rate_weighted_agn_background_dist * jacobian, dx=dz)
                alpha_agn /= agn_population_prior_normalization

    else:
        sum_of_all_agn_posteriors = np.sum(agn_posterior_dset, axis=0)
        nagn_norm = romb(sum_of_all_agn_posteriors, dx=dz)

        if nagn_norm == 0:
            agn_population_prior_normalization = romb(fc_and_rate_weighted_agn_background_dist * jacobian, dx=dz)
            if not cfg.REAL_DATA:
                alpha_agn = romb(pdet * fc_and_rate_weighted_agn_background_dist * jacobian, dx=dz) / agn_population_prior_normalization
        else:
            # p_rate_of_z_agn imposes a redshift cut in the GW population, up to which the pop. is normalized. Therefore agn_population_prior only has to be evaluated at redshifts up to this cut.
            agn_population_prior_rate_weighted = p_rate_of_z_agn * average_redshift_completeness * sum_of_all_agn_posteriors / nagn_norm + fc_and_rate_weighted_agn_background_dist
            agn_population_prior_normalization = romb(agn_population_prior_rate_weighted * jacobian, dx=dz)
            if not cfg.REAL_DATA:
                alpha_agn = romb(pdet * agn_population_prior_rate_weighted * jacobian, dx=dz) / agn_population_prior_normalization
    
    if cfg.REAL_DATA:  # FIXME 26 Aug 2026: alpha_agn calculation seems wrong when changing ZMAX from 10 to 6, but only when using a catalogue, not when using empty catalogue. 
                        # UPDATE 7 Sep 2026: Could have been issue with calculating alpha in my notebook, not in this code. TODO: Check if the bug is still there!
        if cfg.LUM_THRESH == 'inf' or cfg.AGN_ZCUT == 0:  # The empty-catalogue case
            alpha_alt, alpha_agn, _ = alpha(fagn=cfg.LOG_LLH_X_AX, snr_thr=cfg.SNR_THR, far_thr=cfg.FAR_THR, agn_zpop=f'emptycat_{cfg.AGN_ZPRIOR.split('_')[0]}', 
                                            alt_rate_model=cfg.MERGER_RATE, alt_rate_parameters=cfg.RATE_PARAMETERS, zmax=cfg.ZMAX, agn_dist_dir=cfg.AGN_DIST_DIR)
        else:
            zpop = interp1d(cfg.Z_INTEGRAL_AX, agn_population_prior_rate_weighted / agn_population_prior_normalization, bounds_error=False, fill_value=0)
            alpha_alt, alpha_agn, _ = alpha(fagn=cfg.LOG_LLH_X_AX, snr_thr=cfg.SNR_THR, far_thr=cfg.FAR_THR, agn_zpop=zpop, 
                                            alt_rate_model=cfg.MERGER_RATE, alt_rate_parameters=cfg.RATE_PARAMETERS, zmax=cfg.ZMAX, agn_dist_dir=cfg.AGN_DIST_DIR)

    return nagn_norm, agn_population_prior_normalization, alpha_agn, alpha_alt


#############################################################
####################### MAIN FUNCTION #######################
#############################################################


def process_one_fagn(fagn_idx, cfg):

    ### Load the AGN catalogue ###
    if cfg.REAL_DATA:
        agn_posterior_dset, agn_ra, agn_dec, obs_agn_redshift, redshift_completeness = load_quaia(fagn_idx, cfg)
        with open(cfg.JSON_PATH, "r") as f:
            gw_path_dict = json.load(f)
        gw_keys = list(gw_path_dict.keys())
        gw_fnames = [gw_path_dict[key] for key in gw_keys]
    else:
        print(f'\nRealization {fagn_idx + 1}/{cfg.N_REALIZATIONS}')
        # Give every process a unique seed -- TODO: save the seeds somewhere
        seed = np.random.SeedSequence().generate_state(1)[0]
        np.random.seed(seed)
        gw_fnames, agn_posterior_dset, agn_ra, agn_dec, obs_agn_redshift, redshift_completeness = make_mock_agn_catalog(fagn_idx, cfg)

    ### Prepare functions for in the likelihood ###
    (average_completeness, 
     average_redshift_completeness, 
     sky_coverage, 
     fc_of_z, 
     p_rate_of_z_agn, 
     p_rate_of_z_alt, 
     p_rate_of_z_agn_func, 
     PEprior, 
     PEprior_func, 
     normed_agn_background_dist, 
     fc_and_rate_weighted_agn_background_dist, 
     jacobian, 
     dz
     ) = prepare_functions(cfg, redshift_completeness)

    nagn_norm, agn_population_prior_normalization, alpha_agn, alpha_alt = calculate_normalizations(cfg, 
                                                                                                    agn_posterior_dset,
                                                                                                    obs_agn_redshift, 
                                                                                                    average_redshift_completeness,
                                                                                                    p_rate_of_z_agn,
                                                                                                    p_rate_of_z_agn_func,
                                                                                                    fc_and_rate_weighted_agn_background_dist,
                                                                                                    jacobian,
                                                                                                    dz)

    ### Calculate the integrals in the likelihood ###
    Ngws = len(gw_fnames)  # Due to selection effects not always the same number
    S_agn_incat = np.zeros(Ngws)
    S_agn_outofcat = np.zeros(Ngws)
    S_alt = np.zeros(Ngws)

    if cfg.REAL_DATA:
        hdf5_files = [None] * 4  # Evaluated sky maps still stored as separate .npz files for real data
    else:
        subdir = '/'.join(gw_fnames[0].split('/')[:-3])
        gw_zpost_path_agn = f'{subdir}/skymaps_evaluated/agn/zpost_gpmask_False_skymapcl_{cfg.SKYMAP_CL}_cmapnside_{cfg.CMAP_NSIDE}.h5'
        gw_zpost_cw_path_agn = f'{subdir}/skymaps_evaluated/agn/zpost_gpmask_True_skymapcl_{cfg.SKYMAP_CL}_cmapnside_{cfg.CMAP_NSIDE}.h5'
        gw_zpost_path_alt = f'{subdir}/skymaps_evaluated/alt/zpost_gpmask_False_skymapcl_{cfg.SKYMAP_CL}_cmapnside_{cfg.CMAP_NSIDE}.h5'
        gw_zpost_cw_path_alt = f'{subdir}/skymaps_evaluated/alt/zpost_gpmask_True_skymapcl_{cfg.SKYMAP_CL}_cmapnside_{cfg.CMAP_NSIDE}.h5'
        hdf5_files = [h5py.File(path, "r") for path in (gw_zpost_path_agn, gw_zpost_path_alt, gw_zpost_cw_path_agn, gw_zpost_cw_path_alt)]

    try:
        from_agn_hdf5, from_alt_hdf5, from_agn_cw_hdf5, from_alt_cw_hdf5 = hdf5_files
        for gw_idx, filename in enumerate(gw_fnames):

            if cfg.REAL_DATA:
                gwkey = gw_keys[gw_idx]
            else:
                gwkey = get_id_from_fname(filename)

            if cfg.VERBOSE:
                print(f'({gw_idx+1}/{Ngws})')

            if cfg.USE_SKYMAPS:
                sagn_incat, sagn_outofcat, salt = calculate_evidence(
                                                                    cfg=cfg, filename=filename,
                                                                    agn_posterior_dset=agn_posterior_dset, agn_ra=agn_ra, agn_dec=agn_dec,
                                                                    agn_redshift=obs_agn_redshift, p_rate_of_z_agn_func=p_rate_of_z_agn_func,
                                                                    p_rate_of_z_agn=p_rate_of_z_agn, p_rate_of_z_alt=p_rate_of_z_alt,
                                                                    PEprior_func=PEprior_func, PEprior=PEprior, fc_of_z=fc_of_z,
                                                                    average_completeness=average_completeness, sky_coverage=sky_coverage,
                                                                    normed_agn_background_dist=normed_agn_background_dist, nagn_norm=nagn_norm,
                                                                    agn_population_prior_normalization=agn_population_prior_normalization,
                                                                    from_agn_hdf5=from_agn_hdf5, from_alt_hdf5=from_alt_hdf5,
                                                                    from_agn_cw_hdf5=from_agn_cw_hdf5, from_alt_cw_hdf5=from_alt_cw_hdf5, gwkey=gwkey
                                                                )
                if np.isnan(sagn_incat) and np.isnan(sagn_outofcat) and np.isnan(salt):
                    print(f'All evidences are NaN: {filename}')
            else:
                raise NotImplementedError('Only inference using GW sky maps is currently implemented.')

            S_agn_incat[gw_idx] = sagn_incat
            S_agn_outofcat[gw_idx] = sagn_outofcat
            S_alt[gw_idx] = salt

            if cfg.REAL_DATA & (cfg.REAL_POSTERIOR_JSON_DIR != 'none'):  # Store terms in likelihood in .json file, allows for per-event analysis in post.
                if cfg.LUM_THRESH == 'inf' or cfg.AGN_ZCUT == 0:
                    suffix = '_nocat' 
                else:
                    suffix = ''
                json_path = f'{cfg.REAL_POSTERIOR_JSON_DIR}/realdata_posterior{suffix}_{cfg.MERGER_RATE}'

                if cfg.LABEL != 'none':
                    suffix2 = f'_{cfg.LABEL}'
                else:
                    suffix2 = ''
                json_path = f'{json_path}{suffix2}.json' 

                output_file = Path(json_path)

                # Load existing data
                if output_file.exists():
                    data = json.loads(output_file.read_text())
                else:
                    data = {}

                if cfg.AGN_ZPRIOR not in data.keys():  # If this AGN subpopulation has not yet been analyzed, add it to the dictionary
                    data[cfg.AGN_ZPRIOR] = {}
                datadict = data[cfg.AGN_ZPRIOR]  # Modify subdictionary
                datadict[gwkey] = {'S_agn_incat': sagn_incat, 'S_agn_outcat': sagn_outofcat, 'S_alt': salt, 'alpha_agn': alpha_agn, 'alpha_alt': alpha_alt}
                
                output_file.write_text(json.dumps(data, indent=2))

            if cfg.VERBOSE:
                print(f'S_alt: {salt}, S_incat: {sagn_incat}, S_outcat: {sagn_outofcat}, S_agn: {sagn_incat + sagn_outofcat}\n')
                neg = np.sum((cfg.LOG_LLH_X_AX * (sagn_incat + sagn_outofcat - salt) + salt) < 0)
                if neg != 0:
                    print(f'Got {neg} negative values.')
                
    finally:  # If code crashes, still close hdf5 files
        for hdf5_file in hdf5_files:
            if hdf5_file is not None:
                hdf5_file.close()

    del agn_posterior_dset  # Free up the memory asap
    
    ### Evaluate the likelihood ###
    S_agn_incat = S_agn_incat[~np.isnan(S_agn_incat)]
    S_agn_outofcat = S_agn_outofcat[~np.isnan(S_agn_outofcat)]
    S_alt = S_alt[~np.isnan(S_alt)]

    loglike = np.log(cfg.LOG_LLH_X_AX[None,:] * (S_agn_incat[:,None] + S_agn_outofcat[:,None] - S_alt[:,None]) + S_alt[:,None])
    total_loglike = np.sum(loglike, axis=0)  # Sum over all GWs
    total_loglike -= Ngws * np.log(alpha_agn * cfg.LOG_LLH_X_AX + alpha_alt * (1 - cfg.LOG_LLH_X_AX))  # Correct selection effects

    nans = np.isnan(loglike)
    if np.sum(nans) != 0:
        print('Got NaNs:')
        arr = np.ones_like(cfg.LOG_LLH_X_AX)
        print((arr[None,:] * S_agn_incat[:,None])[nans])
        print((arr[None,:] * S_agn_outofcat[:,None])[nans])
        print((arr[None,:] * S_alt[:,None])[nans])

    return fagn_idx, total_loglike
