# from ligo.skymap.io.fits import read_sky_map
# from ligo.skymap import moc

# from pathlib import Path

# from darksirenpop.utilities.gw_selection_effects import alpha
from darksirenpop.utilities.redshift_utils import *
from darksirenpop.utilities.redshift_utils import _CHI_INTERP, _DL_INTERP
from darksirenpop.utilities.utils import uniform_shell_sampler, sample_spherical_angles, truncnorm_pdf_inplace

from tqdm import tqdm
import sys, os
import h5py
import healpy as hp
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import astropy_healpix as ah
import glob
# import json
# import time

from scipy.integrate import romb
from scipy.interpolate import interp1d  #, CubicSpline
from scipy import stats

import astropy.units as u
from astropy.coordinates import SkyCoord


def v25_selection_function(z, cfg):
    completeness_zvals = np.array(cfg.QUAIA_C_VALS[:, cfg.THRESHOLD_MAP[cfg.LUM_THRESH]])
    bin_idx = np.digitize(z, cfg.Z_EDGES) - 1
    bin_idx[bin_idx == len(completeness_zvals)] = len(completeness_zvals) - 1
    return completeness_zvals[bin_idx.astype(np.int32)]


def get_observed_redshift_from_rcom(agn_rcom, cfg):
    true_agn_redshift = fast_z_at_value(cfg.COSMO.comoving_distance, agn_rcom * u.Mpc)

    # Sample from Quaia or make all errors the same (which requires AGN_ZERROR to be a float)?
    if cfg.AGN_ZERROR == 'quaia':
        if cfg.VERBOSE:
            print('Sampling AGN redshift errors from Quaia')
        agn_redshift_err = np.random.choice(cfg.quaia_errors, size=len(agn_rcom))
    else:
        agn_redshift_err = np.tile(cfg.AGN_ZERROR, len(agn_rcom))

    # Perfect measurement or not?
    if not cfg.AGN_ZERROR:
        if cfg.VERBOSE:
            print('No AGN redshift errors')
        obs_agn_redshift = true_agn_redshift
    else:
        if cfg.VERBOSE:
            print('Scattering AGN according to their redshift errors')
        obs_agn_redshift = stats.truncnorm.rvs(size=len(agn_rcom), 
                                                a=(cfg.ZMIN - true_agn_redshift) / agn_redshift_err, 
                                                b=(np.inf - true_agn_redshift) / agn_redshift_err, 
                                                loc=true_agn_redshift, 
                                                scale=agn_redshift_err)
    if cfg.VERBOSE:
        print(f'Complete catalog has {np.sum(obs_agn_redshift > cfg.AGN_ZMAX)} REALIZED AGN redshifts above AGN_ZMAX.')
        print(f'Complete catalog has {np.sum(obs_agn_redshift < cfg.ZMAX)} REALIZED AGN redshifts below GW_ZMAX.')
    return obs_agn_redshift, agn_redshift_err


def make_redshift_selection(obs_agn_redshift, cfg):
    '''
    AGN catalogs can be redshift-incomplete. This function returns a masking array that does the selection based on the data realization of AGN redshifts.
    '''
    if cfg.LUM_THRESH == 'zero':  # No redshift selection
        z_selection_function = lambda z: np.ones_like(z)
        redshift_incomplete_mask = np.ones_like(obs_agn_redshift, dtype=bool)

    elif cfg.LUM_THRESH == 'zero_upto_cut':
        z_selection_function = lambda z: z_cut(z, zcut=cfg.AGN_ZCUT)
        redshift_incomplete_mask = obs_agn_redshift < cfg.AGN_ZCUT
    
    elif cfg.LUM_THRESH == 'inf':
        z_selection_function = lambda z: np.zeros_like(z)
        redshift_incomplete_mask = np.zeros_like(obs_agn_redshift, dtype=bool)

    else:
        z_selection_function = lambda z: v25_selection_function(z, cfg=cfg)

        c_per_zbin = np.array(cfg.QUAIA_C_VALS[:, cfg.THRESHOLD_MAP[cfg.LUM_THRESH]])
        redshift_incomplete_mask = np.zeros_like(obs_agn_redshift, dtype=bool)
        for i, c_in_bin in enumerate(c_per_zbin):
            z_low, z_high = cfg.Z_EDGES[i], cfg.Z_EDGES[i + 1]
            agn_in_bin = np.where((obs_agn_redshift > z_low) & (obs_agn_redshift < z_high))[0]
            keep_these = np.random.choice(np.arange(len(agn_in_bin)), size=round(c_in_bin * len(agn_in_bin)), replace=False)
            redshift_incomplete_mask[agn_in_bin[keep_these]] = True

    return z_selection_function, redshift_incomplete_mask


def make_latitude_selection(agn_ra, agn_dec, obs_agn_rlum, cfg):
    '''
    Latitude completeness map is for indicating which sky area is surveyed (c=1) and which is not (c=0).
    '''
    npix = hp.nside2npix(cfg.CMAP_NSIDE)
    theta, phi = hp.pix2ang(cfg.CMAP_NSIDE, np.arange(npix), nest=True)
    map_coord = SkyCoord(phi * u.rad, (np.pi * 0.5 - theta) * u.rad)
    map_b = map_coord.galactic.b.degree
    outside_galactic_plane_pix = np.logical_or(map_b > 10, map_b < -10)

    completeness_map = np.tile(1., npix)
    b = SkyCoord(agn_ra * u.rad, agn_dec * u.rad, obs_agn_rlum * u.Mpc).galactic.b.degree
    if cfg.MASK_GALACTIC_PLANE:
        latitude_mask = np.logical_or(b > 10, b < -10)
        completeness_map[~outside_galactic_plane_pix] = 0
    else:
        latitude_mask = np.ones_like(b, dtype=bool)
    
    if cfg.PLOT_CMAP:
        hp.mollview(
                    completeness_map,
                    nest=True,
                    coord="G",               # plot in Galactic coords
                    title="Mask: 1 outside |b|<=10°, 0 inside",
                    cmap="coolwarm",
                    min=0, max=1
                )
        hp.graticule()
        plt.savefig(f'{cfg.PLOT_DIR}/cmap.pdf', bbox_inches='tight')
        plt.close()
    return latitude_mask, completeness_map


def make_incomplete_catalog(agn_ra, agn_dec, obs_agn_rlum, obs_agn_redshift, cfg):
    z_selection_function, redshift_incomplete_mask = make_redshift_selection(obs_agn_redshift, cfg=cfg)  # Making a redshift-incomplete catalog
    latitude_mask, completeness_map = make_latitude_selection(agn_ra, agn_dec, obs_agn_rlum, cfg=cfg)
    incomplete_catalog_mask = (latitude_mask & redshift_incomplete_mask)
    if cfg.VERBOSE:
        print(f'Observed {np.sum(incomplete_catalog_mask)} AGN from realizations, of which {np.sum(obs_agn_redshift[incomplete_catalog_mask] < cfg.ZMAX)} below GW_ZMAX. Average completeness below GW_ZMAX: {np.sum(obs_agn_redshift[incomplete_catalog_mask] < cfg.ZMAX) / np.sum(obs_agn_redshift < cfg.ZMAX):.5f}')
    return incomplete_catalog_mask, z_selection_function, completeness_map


def compute_agn_posteriors_chunk(start, end, all_agn_z, all_agn_z_err, cfg, n_norm=100):
    '''
    Compute a chunk of AGN posteriors. This computation is vectorized.
    AGN redshift posteriors are modelled as truncnorms on [0, inf) with a QLF-based redshift prior.
    The posteriors are then evaluated on Z_INTEGRAL_AX, which is what is necessary for the crossmatch.
    '''
    
    z_chunk = all_agn_z[start:end]
    zerr_chunk = all_agn_z_err[start:end]
    mu = z_chunk[:, None]
    sigma = zerr_chunk[:, None]
    
    # Build per-AGN normalization axes
    t = np.linspace(0, 1, n_norm)[None, :]
    z_norm_ax = np.maximum(mu - 10*sigma, cfg.ZMIN) + t * (20 * sigma)
    
    # Get normalization of all posteriors
    posteriors_unnorm = truncnorm_pdf_inplace(z_norm_ax, mu, sigma, zmin=cfg.ZMIN)
    posteriors_unnorm *= z_cut(z_norm_ax, zcut=cfg.AGN_ZMAX)
    posteriors_unnorm *= cfg.AGN_ZPRIOR_FUNCTION(z_norm_ax)
    z_norms = np.trapezoid(posteriors_unnorm, z_norm_ax, axis=1)
    
    # Get evaluation of posteriors on the desired axis -- memory expensive
    posteriors = truncnorm_pdf_inplace(cfg.Z_INTEGRAL_AX, mu, sigma, zmin=cfg.ZMIN)
    posteriors *= z_cut(cfg.Z_INTEGRAL_AX, zcut=cfg.AGN_ZMAX)
    posteriors *= cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX)
    posteriors /= z_norms[:, None]
    return posteriors


# FIXME: function never called - test this before removing
# def compute_and_save_posteriors_hdf5(filename, all_agn_z, all_agn_z_err, cfg, n_norm=100):
#     '''
#     For real data, we should only have to do this computation once and reuse the stored values.
#     For mock, this is not useful, since we use different AGN catalogues each time.
#     '''

#     n_agn = len(all_agn_z)
#     n_z = len(cfg.Z_INTEGRAL_AX)  # Only need to save the posterior evaluated at this axis
#     chunk_size = int(1e6 / n_z)
#     with h5py.File(filename, "w") as f:
#         dset = f.create_dataset("agn_redshift_posteriors", shape=(n_agn, n_z), dtype=np.float32)

#         if cfg.VERBOSE:
#             iterchunks = tqdm( range(0, n_agn, chunk_size) )
#         else:
#             iterchunks = range(0, n_agn, chunk_size)

#         for start in iterchunks:
#             end = min(start + chunk_size, n_agn)
#             posteriors = compute_agn_posteriors_chunk(start, end, all_agn_z, all_agn_z_err, cfg, n_norm)
#             dset[start:end, :] = posteriors

#     if cfg.VERBOSE:
#         print(f"All AGN posteriors written to {filename}")
    
#     return


def get_agn_posteriors(fagn_idx, obs_agn_redshift, agn_redshift_err, label, cfg, replace_old_file=True, n_norm=100):
    '''
    To save computation time, the AGN posteriors are calculated and evaluated on the z-integral axis once and kept in memory.
    '''

    if cfg.ASSUME_PERFECT_REDSHIFT:
        return np.empty(0), 1
    
    else:

        #################### FIXME: cfg.REAL_DATA is never true when this function is called - remove this block & check! ####################

        # if cfg.REAL_DATA:  # Not used anymore as of 25-06-2026
        #     posterior_path = f'{cfg.AGN_DIST_DIR}/quaia_zleq{cfg.AGN_ZCUT}_{cfg.AGN_ZPRIOR}.hdf5'
        # else:
        #     posterior_path = f'./precompute_posteriors/agn_posteriors_precompute_gwZmax_{cfg.ZMAX}_prior_{cfg.AGN_ZPRIOR}_{fagn_idx}_{label}.hdf5'

        # if cfg.REAL_DATA:  # The real AGN catalogue doesn't change, so we can compute it once and store it (although you can still choose to recompute using the replace_old_file flag)
        #     if not os.path.exists(posterior_path):
        #         compute_and_save_posteriors_hdf5(posterior_path, obs_agn_redshift, agn_redshift_err, cfg, n_norm=n_norm)
        #     elif os.path.exists(posterior_path) & replace_old_file:
        #         os.remove(posterior_path)
        #         compute_and_save_posteriors_hdf5(posterior_path, obs_agn_redshift, agn_redshift_err, cfg, n_norm=n_norm)

        #     # Keep ~few GB in memory, this is typically faster than reading random slices
        #     with h5py.File(posterior_path, "r") as f:
        #         agn_posterior_dset = f["agn_redshift_posteriors"][()]
            
        #     if agn_posterior_dset.shape[1] != len(cfg.Z_INTEGRAL_AX):
        #         sys.exit(f'AGN redshift posteriors evaluated on the wrong axis. dset has len {agn_posterior_dset.shape[1]}, but z-ax requires {len(cfg.Z_INTEGRAL_AX)} Exiting...')
        #########################################################################################################
        
        # else:  
        # Just compute and immediately keep in memory.
        agn_posterior_dset = compute_agn_posteriors_chunk(start=0, end=len(obs_agn_redshift), all_agn_z=obs_agn_redshift, all_agn_z_err=agn_redshift_err, cfg=cfg, n_norm=n_norm)

        sum_of_posteriors = np.sum(agn_posterior_dset, axis=0)

        return agn_posterior_dset, sum_of_posteriors


def get_id_from_fname(fname):
    file_type = fname.split('/')[-1].split('_')[-4]
    if file_type == 'gw':
        return fname[-8:-3]
    elif file_type == 'skymap':
        return fname[-13:-8]
    else:
        sys.exit(f'Extracted the following file type from file name and did not recognize: {file_type}')
    return 


def get_fnames(ids, file_type, cfg):
    '''file_type either skymap or samples'''
    if file_type == 'samples':
        label = 'gw'
        dir = cfg.SAMPLES_DIR
    elif file_type == 'skymap':
        label = 'skymap'
        dir = cfg.SKYMAP_DIR
    else:
        sys.exit(f'Do not recognize file type: {file_type}. Choose between str(skymap) or str(samples).')

    fnames = []
    for id in ids:
        s = f'{dir}{label}_0_0_{id:05d}.fits.gz'
        fnames.append(s)
    return np.array(fnames)


def get_gw_fnames_resampled(fagn_realized, cfg):
    '''
    Currently assuming ALT GW hosts are always distributed uniform in comoving volume, but rate evolution can be set.

    Warning: Resampling of a finite amount of GW data will cause biases when analyzing many data realizations due to duplicate GWs
    '''
    
    agn_rcom = cfg.ALL_TRUE_SOURCES[:,1]
    agn_z = fast_z_at_value(cfg.COSMO.comoving_distance, agn_rcom * u.Mpc)

    # Make target GW-from-AGN population, which is normalized on Z_INTEGRAL_AX
    norm = romb(cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX), dx=np.diff(cfg.Z_INTEGRAL_AX)[0])
    target_population = lambda z: cfg.AGN_ZPRIOR_FUNCTION(z) / norm

    weights = target_population(agn_z) / uniform_comoving_prior(agn_z, cosmo=cfg.COSMO)  # Divide out the distribution of the mock data
    if cfg.CORRECT_TIME_DILATION:
        weights *= 1 / (1 + agn_z)
    from_agn_population = np.random.choice(np.arange(len(agn_z)), p=weights / np.sum(weights), size=round(fagn_realized * cfg.BATCH))

    need_these = cfg.TRUE_SOURCE_IDENTIFIERS[from_agn_population].astype(int)
    gw_fnames_from_agn = get_fnames(need_these, file_type=cfg.FILE_TYPE, cfg=cfg)

    # GW-from-ALT population
    if not cfg.CORRECT_TIME_DILATION and (cfg.MERGER_RATE == 'uniform'):  # Target population is equal to mock data population
        gw_fnames_from_alt = np.random.choice(cfg.ALL_GW_FNAMES, size=cfg.BATCH - gw_fnames_from_agn.shape[0], replace=False)
    else:
        weights = merger_rate(agn_z, cfg.MERGER_RATE_EVOLUTION, **cfg.MERGER_RATE_KWARGS)
        if cfg.CORRECT_TIME_DILATION:
            weights *= 1 / (1 + agn_z)
        from_alt_population = np.random.choice(np.arange(len(agn_z)), p=weights / np.sum(weights), size=cfg.BATCH - gw_fnames_from_agn.shape[0])
        
        need_these = cfg.TRUE_SOURCE_IDENTIFIERS[from_alt_population].astype(int)
        gw_fnames_from_alt = get_fnames(need_these, file_type=cfg.FILE_TYPE, cfg=cfg)
    
    gw_fnames = np.append(gw_fnames_from_agn, gw_fnames_from_alt)

    return gw_fnames, gw_fnames_from_agn


def fill_catalog_to_complete(agn_ra, agn_dec, agn_rcom, cfg):
    '''To preserve overall distribution, need to add AGN above COMDIST_MAX, where the GW-hosting AGN are.'''
    if cfg.AGN_ZPRIOR == 'uniform_comoving_volume':
        n2complete = int(round(len(agn_ra) * ( (cfg.AGN_COMDIST_MAX / cfg.COMDIST_MAX)**3 - 1)))
        new_rcom, new_theta, new_phi = uniform_shell_sampler(cfg.COMDIST_MAX, cfg.AGN_COMDIST_MAX, n2complete)

    else:
        rcom_integrate_ax = np.linspace(cfg.COMDIST_MIN, cfg.AGN_COMDIST_MAX, 1024*4+1)
        total = romb(comdist_pdf_given_redshift_pdf(rcom_integrate_ax, cfg.AGN_ZPRIOR_FUNCTION, cosmo=cfg.COSMO), dx=np.diff(rcom_integrate_ax)[0])
        rcom_integrate_ax = np.linspace(cfg.COMDIST_MIN, cfg.COMDIST_MAX, 1024*4+1)
        current = romb(comdist_pdf_given_redshift_pdf(rcom_integrate_ax, cfg.AGN_ZPRIOR_FUNCTION, cosmo=cfg.COSMO), dx=np.diff(rcom_integrate_ax)[0])

        n2complete = int(round(len(agn_ra) * ( total / current - 1)))
        if n2complete != 0:

            new_theta, new_phi = sample_spherical_angles(n2complete)

            norm = romb(cfg.AGN_ZPRIOR_FUNCTION(cfg.AGN_ZPRIOR_NORM_AX) * (1 - z_cut(cfg.AGN_ZPRIOR_NORM_AX, zcut=cfg.ZMAX)), dx=np.diff(cfg.AGN_ZPRIOR_NORM_AX)[0])
            target_population = lambda z: cfg.AGN_ZPRIOR_FUNCTION(z) * (1 - z_cut(z, zcut=cfg.ZMAX)) / norm
            cdf = np.cumsum(target_population(cfg.AGN_ZPRIOR_NORM_AX))
            cdf /= cdf[-1]
            unif = np.random.rand(n2complete)
            new_z = np.interp(unif, cdf, cfg.AGN_ZPRIOR_NORM_AX)
            new_rcom = _CHI_INTERP(new_z)  #cfg.COSMO.comoving_distance(new_z).value

            agn_ra = np.append(agn_ra, new_phi)
            agn_dec = np.append(agn_dec, np.pi * 0.5 - new_theta)
            agn_rcom = np.append(agn_rcom, new_rcom)

    if cfg.VERBOSE:
        print(f'Adding {n2complete} AGN above GW zmax ({cfg.ZMAX}) to get a catalog with distribution: {cfg.AGN_ZPRIOR}.')

    return agn_ra, agn_dec, agn_rcom, n2complete


def add_agn_propto_z(agn_ra, agn_dec, agn_rcom, nsamps, cfg):
    '''
    Add samples from p(z) ~ pi_agn(z) * z / (1 + z) such that the total sample of AGN follows pi_agn(z). This is
    necessary because the GW-hosting AGN follow q(z) ~ pi_agn(z) / (1 + z).
    '''
    new_theta, new_phi = sample_spherical_angles(nsamps)

    norm = romb(cfg.AGN_ZPRIOR_FUNCTION(cfg.AGN_ZPRIOR_NORM_AX) * cfg.AGN_ZPRIOR_NORM_AX / (1 + cfg.AGN_ZPRIOR_NORM_AX), dx=np.diff(cfg.AGN_ZPRIOR_NORM_AX)[0])
    target_population = lambda z: cfg.AGN_ZPRIOR_FUNCTION(z) * cfg.AGN_ZPRIOR_NORM_AX  / (1 + cfg.AGN_ZPRIOR_NORM_AX) / norm
    cdf = np.cumsum(target_population(cfg.AGN_ZPRIOR_NORM_AX))
    cdf /= cdf[-1]
    unif = np.random.rand(nsamps)
    new_z = np.interp(unif, cdf, cfg.AGN_ZPRIOR_NORM_AX)
    new_rcom = _CHI_INTERP(new_z) #cfg.COSMO.comoving_distance(new_z).value

    agn_ra = np.append(agn_ra, new_phi)
    agn_dec = np.append(agn_dec, np.pi * 0.5 - new_theta)
    agn_rcom = np.append(agn_rcom, new_rcom)

    return agn_ra, agn_dec, agn_rcom


def add_agn_to_catalog(agn_ra, agn_dec, agn_rcom, nsamps, cfg):
    '''As background noise'''

    if cfg.AGN_ZPRIOR == 'uniform_comoving_volume':
        new_rcom, new_theta, new_phi = uniform_shell_sampler(cfg.COMDIST_MIN, cfg.AGN_COMDIST_MAX, nsamps)
    
    else:
        new_theta, new_phi = sample_spherical_angles(nsamps)

        norm = romb(cfg.AGN_ZPRIOR_FUNCTION(cfg.AGN_ZPRIOR_NORM_AX), dx=np.diff(cfg.AGN_ZPRIOR_NORM_AX)[0])
        target_population = lambda z: cfg.AGN_ZPRIOR_FUNCTION(z) / norm
        cdf = np.cumsum(target_population(cfg.AGN_ZPRIOR_NORM_AX))
        cdf /= cdf[-1]
        unif = np.random.rand(nsamps)
        new_z = np.interp(unif, cdf, cfg.AGN_ZPRIOR_NORM_AX)
        new_rcom = _CHI_INTERP(new_z) #cfg.COSMO.comoving_distance(new_z).value

    agn_ra = np.append(agn_ra, new_phi)
    agn_dec = np.append(agn_dec, np.pi * 0.5 - new_theta)
    agn_rcom = np.append(agn_rcom, new_rcom)

    return agn_ra, agn_dec, agn_rcom


def get_mock_gw_sources(fagn_idx, cfg):
    output_directory = glob.glob(f'{cfg.MOCKDATA_ROOT}/output_run_{fagn_idx + 1}_*')[0]

    gw_fnames_from_agn = glob.glob(f'{output_directory}/skymaps/agn/skymap*.fits.gz')
    gw_fnames_from_alt = glob.glob(f'{output_directory}/skymaps/alt/skymap*.fits.gz')
    gw_fnames = np.append(gw_fnames_from_agn, gw_fnames_from_alt)
    gw_identifiers = sorted(np.array([get_id_from_fname(f) for f in gw_fnames_from_agn]).astype(int))
    
    if len(gw_identifiers) > 0:  # If there are GWs from AGN in the data set
        true_sources = np.genfromtxt(f'{output_directory}/true_gw_coords/agn/true_r_theta_phi.txt', delimiter=',')  # There are only positions of GW-generating AGN in this file, no need to sort and search
        true_sources = np.atleast_2d(true_sources)
    else:
        true_sources = np.empty((0, 5))
    
    agn_ra, agn_dec, agn_rcom = true_sources[:,3], 0.5 * np.pi - true_sources[:,2], true_sources[:,1]

    sources_of_gw_nondetections = np.genfromtxt(f'{output_directory}/true_gw_coords_nondetections/agn/true_r_theta_phi.txt', delimiter=',')
    agn_ra_nd, agn_dec_nd, agn_rcom_nd = sources_of_gw_nondetections[:,2], 0.5 * np.pi - sources_of_gw_nondetections[:,1], sources_of_gw_nondetections[:,0]
    
    agn_ra = np.append(agn_ra, agn_ra_nd)
    agn_dec = np.append(agn_dec, agn_dec_nd)
    agn_rcom = np.append(agn_rcom, agn_rcom_nd)

    return gw_fnames, agn_ra, agn_dec, agn_rcom


def make_mock_agn_catalog(fagn_idx, fagn_realized, cfg):
    '''
    Make the incomplete AGN catalog on the fly, given the source coordinates of GWs from AGN. Returns the GW filenames, catalog and estimated selection function.
    '''

    ### Get true source coordinates for GWs from AGN to put in the AGN catalog ###
    if cfg.MOCKDATA_ROOT == None:  # Some steps are already done in the config
        gw_fnames, gw_fnames_from_agn = get_gw_fnames_resampled(fagn_realized, cfg=cfg)
        gw_identifiers = sorted(np.array([get_id_from_fname(f) for f in gw_fnames_from_agn]).astype(int))
        true_sources = cfg.ALL_TRUE_SOURCES[np.searchsorted(cfg.TRUE_SOURCE_IDENTIFIERS, gw_identifiers)]  # Get all positions of GW-generating AGN

        agn_ra, agn_dec, agn_rcom = true_sources[:,3], 0.5 * np.pi - true_sources[:,2], true_sources[:,1]

    else:  # Folders are unique per realization, so get them on the fly
        gw_fnames, agn_ra, agn_dec, agn_rcom = get_mock_gw_sources(fagn_idx, cfg)

    ### Complete catalog to preserve proper distribution, i.e., without overdensity below cfg.ZMAX due to adding GW-generating AGN first ###
    agn_ra_complete, agn_dec_complete, agn_rcom_complete, n2complete = fill_catalog_to_complete(agn_ra, agn_dec, agn_rcom, cfg=cfg)

    # print('Testing', np.mean(fast_z_at_value(COSMO.comoving_distance, agn_rcom * u.Mpc)))
    agn_ra_complete, agn_dec_complete, agn_rcom_complete = add_agn_propto_z(agn_ra_complete, agn_dec_complete, agn_rcom_complete, int(np.mean(fast_z_at_value(COSMO.comoving_distance, agn_rcom * u.Mpc)) * len(agn_ra_complete)), cfg)
    ############################################################################

    
    if cfg.ADD_NAGN_TO_CAT > n2complete + len(agn_ra_complete):  # Add uncorrelated AGN as background
        if cfg.VERBOSE:
            print(f'Adding {cfg.ADD_NAGN_TO_CAT - n2complete - len(agn_ra_complete)} more AGN.')

        agn_ra_complete, agn_dec_complete, agn_rcom_complete = add_agn_to_catalog(agn_ra_complete, agn_dec_complete, agn_rcom_complete, cfg.ADD_NAGN_TO_CAT - n2complete - len(agn_ra_complete), cfg=cfg)
    
    # print(len(agn_ra_complete), 'final number')
    # plt.figure()
    # # plt.hist(fast_z_at_value(COSMO.comoving_distance, agn_rcom * u.Mpc), density=True, bins=np.linspace(0, 10, 100), histtype='step', label='GW origins')
    # # plt.hist(fast_z_at_value(COSMO.comoving_distance, agn_rcom_complete * u.Mpc), density=True, bins=np.linspace(0, 10, 100), histtype='step', label='Completed')
    # plt.plot(cfg.AGN_ZPRIOR_NORM_AX, cfg.AGN_ZPRIOR_FUNCTION(cfg.AGN_ZPRIOR_NORM_AX), color='black')
    # plt.hist(fast_z_at_value(COSMO.comoving_distance, agn_rcom_complete * u.Mpc), density=True, bins=np.linspace(0, 10, 100), histtype='step', label='Noise added')
    # plt.legend()
    # plt.show()
    # sys.exit(1)

    if len(agn_rcom_complete) == 0:
        obs_agn_redshift_complete, agn_redshift_err_complete = np.empty_like(agn_rcom_complete), np.empty_like(agn_rcom_complete)
        obs_agn_rlum_complete = np.empty_like(agn_rcom_complete)
    else:
        obs_agn_redshift_complete, agn_redshift_err_complete = get_observed_redshift_from_rcom(agn_rcom_complete, cfg=cfg)
        obs_agn_rlum_complete = _DL_INTERP(obs_agn_redshift_complete) #cfg.COSMO.luminosity_distance(obs_agn_redshift_complete).value

    ### Make an incomplete AGN catalog from these coordinates ###
    incomplete_catalog_mask, z_selection_function, completeness_map = make_incomplete_catalog(agn_ra_complete, agn_dec_complete, obs_agn_rlum_complete, obs_agn_redshift_complete, cfg=cfg)
    agn_ra = agn_ra_complete[incomplete_catalog_mask]
    agn_dec = agn_dec_complete[incomplete_catalog_mask]
    obs_agn_redshift = obs_agn_redshift_complete[incomplete_catalog_mask]
    agn_redshift_err = agn_redshift_err_complete[incomplete_catalog_mask]
    # obs_agn_rlum = obs_agn_rlum_complete[incomplete_catalog_mask]

    agn_posterior_dset, sum_of_posteriors_incomplete = get_agn_posteriors(fagn_idx, obs_agn_redshift, agn_redshift_err, label='INCOMPLETE', cfg=cfg)

    ### Characterize the redshift-completeness ###
    if cfg.ASSUME_PERFECT_REDSHIFT or cfg.LUM_THRESH == 'inf':
        redshift_completeness = z_selection_function  # TODO: Should actually still measure from the data, instead of just feeding the selection function in

    else:  # Measure the selection function from the data realization
        latitude_mask, _ = make_latitude_selection(agn_ra_complete, agn_dec_complete, obs_agn_rlum_complete, cfg=cfg)  # Measure completeness in the surveyed sky area
        expected_distribution = np.sum(latitude_mask) * cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX) / romb(cfg.AGN_ZPRIOR_FUNCTION(cfg.AGN_ZPRIOR_NORM_AX), dx=np.diff(cfg.AGN_ZPRIOR_NORM_AX)[0])
        no_zero = (expected_distribution != 0)

        redshift_agn_selection_function = np.zeros_like(expected_distribution)
        redshift_agn_selection_function[no_zero] = sum_of_posteriors_incomplete[no_zero] / expected_distribution[no_zero]
        redshift_agn_selection_function[redshift_agn_selection_function > 1] = 1
        redshift_completeness = interp1d(cfg.Z_INTEGRAL_AX, redshift_agn_selection_function, bounds_error=False, fill_value=0)

    ### True selection function requires selection function and likelihood. Single cut + Gaussian likelihood gives the following expression:
    # # redshift_completeness_singlecut = lambda z: stats.norm.cdf(cfg.AGN_ZCUT, loc=z, scale=cfg.AGN_ZERROR)
    
    # if cfg.LUM_THRESH == 'zero_upto_cut':
    #     cbins = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0])
    # else:
    #     cbins = np.array(cfg.QUAIA_C_VALS[:, cfg.THRESHOLD_MAP[cfg.LUM_THRESH]])
    # step1 = lambda z: cbins[0] * (stats.truncnorm.cdf(cfg.Z_EDGES[1], loc=z, scale=cfg.AGN_ZERROR, a=(0 - z) / cfg.AGN_ZERROR, b = np.inf) - stats.truncnorm.cdf(cfg.Z_EDGES[0], loc=z, scale=cfg.AGN_ZERROR, a=(0 - z) / cfg.AGN_ZERROR, b = np.inf))
    # step2 = lambda z: cbins[1] * (stats.truncnorm.cdf(cfg.Z_EDGES[2], loc=z, scale=cfg.AGN_ZERROR, a=(0 - z) / cfg.AGN_ZERROR, b = np.inf) - stats.truncnorm.cdf(cfg.Z_EDGES[1], loc=z, scale=cfg.AGN_ZERROR, a=(0 - z) / cfg.AGN_ZERROR, b = np.inf))
    # step3 = lambda z: cbins[2] * (stats.truncnorm.cdf(cfg.Z_EDGES[3], loc=z, scale=cfg.AGN_ZERROR, a=(0 - z) / cfg.AGN_ZERROR, b = np.inf) - stats.truncnorm.cdf(cfg.Z_EDGES[2], loc=z, scale=cfg.AGN_ZERROR, a=(0 - z) / cfg.AGN_ZERROR, b = np.inf))
    # step4 = lambda z: cbins[3] * (stats.truncnorm.cdf(cfg.Z_EDGES[4], loc=z, scale=cfg.AGN_ZERROR, a=(0 - z) / cfg.AGN_ZERROR, b = np.inf) - stats.truncnorm.cdf(cfg.Z_EDGES[3], loc=z, scale=cfg.AGN_ZERROR, a=(0 - z) / cfg.AGN_ZERROR, b = np.inf))
    # step5 = lambda z: cbins[4] * (stats.truncnorm.cdf(cfg.Z_EDGES[5], loc=z, scale=cfg.AGN_ZERROR, a=(0 - z) / cfg.AGN_ZERROR, b = np.inf) - stats.truncnorm.cdf(cfg.Z_EDGES[4], loc=z, scale=cfg.AGN_ZERROR, a=(0 - z) / cfg.AGN_ZERROR, b = np.inf))
    # step6 = lambda z: cbins[5] * (stats.truncnorm.cdf(cfg.Z_EDGES[6], loc=z, scale=cfg.AGN_ZERROR, a=(0 - z) / cfg.AGN_ZERROR, b = np.inf) - stats.truncnorm.cdf(cfg.Z_EDGES[5], loc=z, scale=cfg.AGN_ZERROR, a=(0 - z) / cfg.AGN_ZERROR, b = np.inf))
    # step7 = lambda z: cbins[6] * (stats.truncnorm.cdf(cfg.Z_EDGES[7], loc=z, scale=cfg.AGN_ZERROR, a=(0 - z) / cfg.AGN_ZERROR, b = np.inf) - stats.truncnorm.cdf(cfg.Z_EDGES[6], loc=z, scale=cfg.AGN_ZERROR, a=(0 - z) / cfg.AGN_ZERROR, b = np.inf))
    # step8 = lambda z: cbins[7] * (stats.truncnorm.cdf(cfg.Z_EDGES[8], loc=z, scale=cfg.AGN_ZERROR, a=(0 - z) / cfg.AGN_ZERROR, b = np.inf) - stats.truncnorm.cdf(cfg.Z_EDGES[7], loc=z, scale=cfg.AGN_ZERROR, a=(0 - z) / cfg.AGN_ZERROR, b = np.inf))
    
    # def redshift_completenessx(z):
    #     return step1(z) + step2(z) + step3(z) + step4(z) + step5(z) + step6(z) + step7(z) + step8(z)

    # ## Estimate from redshift means in bins
    # arr = np.array([0.00000e+00 ,2.65000e+02 ,1.78800e+03, 8.48500e+03, 3.04290e+04, 8.31410e+04, 1.73429e+05, 3.52603e+05])  # Expected from sim with 10^7 AGN at 10^46.5 erg/s without z-error
    # expected_nagn_in_bin = np.around(arr / 1e7 * len(agn_rcom_complete))
    # observed_nagn_in_bin, bins, _ = plt.hist(obs_agn_redshift, histtype='step', linewidth=2, bins=np.linspace(0, 1.5, 9))
    # plt.close()

    # fc = np.zeros_like(expected_nagn_in_bin)
    # fc[expected_nagn_in_bin != 0] = observed_nagn_in_bin[expected_nagn_in_bin != 0] / expected_nagn_in_bin[expected_nagn_in_bin != 0] / (1 - np.sin(np.deg2rad(10)))
    # fc[fc > 1] = 1

    # def make_fc_lookup(bins, fc):
    #     bins = np.asarray(bins)
    #     fc = np.asarray(fc)

    #     def fc_of_z(z):
    #         z = np.asarray(z)

    #         # Find bin indices
    #         idx = np.digitize(z, bins) - 1

    #         # Handle out-of-range values
    #         idx[idx < 0] = 0
    #         idx[idx >= len(fc)] = len(fc) - 1

    #         res = fc[idx]
    #         res[z >= 1.5] = 0

    #         return res

    #     return fc_of_z

    # redshift_completeness_v25 = make_fc_lookup(bins, fc)

    # TRUTH = redshift_completenessx(cfg.Z_INTEGRAL_AX) * cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX)
    # P26 = redshift_completeness(cfg.Z_INTEGRAL_AX) * cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX)
    # V25 = redshift_completeness_v25(cfg.Z_INTEGRAL_AX) * cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX)

    # from utils import make_nice_plots
    # make_nice_plots()

    # # np.save(f'V25_{cfg.AGN_ZERROR}.npy', V25 - TRUTH)
    # # np.save(f'P26_{cfg.AGN_ZERROR}.npy', P26 - TRUTH)
    # # np.save(f'True_{cfg.AGN_ZERROR}.npy', TRUTH)

    # plt.figure(figsize=(8,6))
    # plt.plot(cfg.Z_INTEGRAL_AX, V25 - TRUTH, color='teal', linewidth=3, label=r'V25')
    # plt.plot(cfg.Z_INTEGRAL_AX, P26 - TRUTH, color='crimson', linewidth=1, label=r'P26')
    # plt.xlabel('Redshift')
    # plt.ylabel(r'$\Delta\!\left[P^{\rm EM}_{\rm det}(z)\,\pi_{\rm agn}(z)\right]$')
    # plt.legend()
    # plt.xlim(0, 2)
    # plt.show()
    # sys.exit(1)

    # # plt.figure()
    # # plt.plot(cfg.Z_INTEGRAL_AX, redshift_completeness_v25(cfg.Z_INTEGRAL_AX) * cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX), linewidth=2, label='V25')
    # # plt.plot(cfg.Z_INTEGRAL_AX, redshift_completenessx(cfg.Z_INTEGRAL_AX) * cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX), label='True', color='black', linewidth=3)
    # # plt.plot(cfg.Z_INTEGRAL_AX, redshift_completeness(cfg.Z_INTEGRAL_AX) * cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX), linewidth=2, label='P26')
    # # plt.xlabel('Redshift')
    # # plt.ylabel('Pdet(z) * Ppop(z)')
    # # plt.legend()
    # # plt.xlim(0, 2)
    # # plt.show()

    # # sys.exit(1)
    return gw_fnames, agn_posterior_dset, agn_ra, agn_dec, obs_agn_redshift, redshift_completeness