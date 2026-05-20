from ligo.skymap.io.fits import read_sky_map
from ligo.skymap import moc

from redshift_utils import *
from redshift_utils import _CHI_INTERP, _DL_INTERP
from utils import uniform_shell_sampler, sample_spherical_angles, truncnorm_pdf_inplace

from tqdm import tqdm
import sys, os
import h5py
import healpy as hp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy_healpix as ah
import glob
import json
import time

from scipy.integrate import romb
from scipy.interpolate import interp1d, CubicSpline
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
    AGN redshift posteriors are modelled as truncnorms on [0, inf) with a uniform-in-comoving-volume redshift prior.
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


def compute_and_save_posteriors_hdf5(filename, all_agn_z, all_agn_z_err, cfg, n_norm=100):
    '''
    For real data, we should only have to do this computation once and reuse the stored values.
    For mock, this is not useful, since we use different AGN catalogues each time.
    '''

    n_agn = len(all_agn_z)
    n_z = len(cfg.Z_INTEGRAL_AX)  # Only need to save the posterior evaluated at this axis
    chunk_size = int(1e6 / n_z)
    with h5py.File(filename, "w") as f:
        dset = f.create_dataset("agn_redshift_posteriors", shape=(n_agn, n_z), dtype=np.float64)

        if cfg.VERBOSE:
            iterchunks = tqdm( range(0, n_agn, chunk_size) )
        else:
            iterchunks = range(0, n_agn, chunk_size)

        for start in iterchunks:
            end = min(start + chunk_size, n_agn)
            posteriors = compute_agn_posteriors_chunk(start, end, all_agn_z, all_agn_z_err, cfg, n_norm)
            dset[start:end, :] = posteriors

    if cfg.VERBOSE:
        print(f"All AGN posteriors written to {filename}")
    
    return


def get_agn_posteriors(fagn_idx, obs_agn_redshift, agn_redshift_err, label, cfg, replace_old_file=True, n_norm=100):
    '''
    To save computation time, the AGN posteriors are calculated and evaluated on the z-integral axis once and kept in memory.
    '''

    if cfg.ASSUME_PERFECT_REDSHIFT:
        return np.empty(0), 1
    
    else:
        posterior_path = f'./precompute_posteriors/agn_posteriors_precompute_gwZmax_{cfg.ZMAX}_prior_{cfg.AGN_ZPRIOR}_{fagn_idx}_{label}.hdf5'

        if cfg.REAL_DATA:  # The real AGN catalogue doesn't change, so we can compute it once and store it (although you can still choose to recompute using the replace_old_file flag)
            if not os.path.exists(posterior_path):
                compute_and_save_posteriors_hdf5(posterior_path, obs_agn_redshift, agn_redshift_err, cfg, n_norm=n_norm)
            elif os.path.exists(posterior_path) & replace_old_file:
                os.remove(posterior_path)
                compute_and_save_posteriors_hdf5(posterior_path, obs_agn_redshift, agn_redshift_err, cfg, n_norm=n_norm)

            # Keep ~few GB in memory, this is typically faster than reading random slices
            with h5py.File(posterior_path, "r") as f:
                agn_posterior_dset = f["agn_redshift_posteriors"][()]
        
        else:  # Just compute and immediately keep in memory.
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


########################################################################################################################################################


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

    if cfg.REAL_DATA:
        with open(cfg.REAL_ZPOSTS_JSON_PATH, "r") as f:
            gw_zpost_path_dict = json.load(f)
            gw_zpost_path = gw_zpost_path_dict[gwkey]
        
        with open(cfg.REAL_CW_ZPOSTS_JSON_PATH, "r") as f:
            gw_zpost_cw_path_dict = json.load(f)
            gw_zpost_cw_path = gw_zpost_cw_path_dict[gwkey]
        
        z, p = np.load(gw_zpost_path)
        z_cw, p_cw = np.load(gw_zpost_cw_path)

    else:
        gw_id = filename[-13:-8]

        if cfg.MOCKDATA_ROOT == None:
            sys.exit('Option outdated: cfg.MOCKDATA_ROOT == None. Run code from a mock data root directory!')
            gw_zpost_path=f'{cfg.GW_ZPOST_DIR}zpost_{gw_id}_gpmask_False_skymapcl_{cfg.SKYMAP_CL}_cmapnside_{cfg.CMAP_NSIDE}.npy'
            gw_zpost_cw_path=f'{cfg.GW_ZPOST_DIR}zpost_{gw_id}_gpmask_True_skymapcl_{cfg.SKYMAP_CL}_cmapnside_{cfg.CMAP_NSIDE}.npy'
        else:
            agn_or_alt = filename.split('/')[-2]
            # output_run = filename.split('/')[-4]
            # gw_zpost_path = f'{cfg.MOCKDATA_ROOT}/{output_run}/skymaps_evaluated/{agn_or_alt}/zpost_{gw_id}_gpmask_False_skymapcl_{cfg.SKYMAP_CL}_cmapnside_{cfg.CMAP_NSIDE}.npy'
            # gw_zpost_cw_path = f'{cfg.MOCKDATA_ROOT}/{output_run}/skymaps_evaluated/{agn_or_alt}/zpost_{gw_id}_gpmask_True_skymapcl_{cfg.SKYMAP_CL}_cmapnside_{cfg.CMAP_NSIDE}.npy'

            if agn_or_alt == 'agn':
                zpost_file = from_agn_hdf5
                cw_zpost_file = from_agn_cw_hdf5
            elif agn_or_alt == 'alt':
                zpost_file = from_alt_hdf5
                cw_zpost_file = from_alt_cw_hdf5
            else:
                sys.exit(f'Do not recognize subdirectory: {agn_or_alt}. Expected "agn" or "alt".')

            # gw_zpost_path = f'{cfg.MOCKDATA_ROOT}/{output_run}/skymaps_evaluated/{agn_or_alt}/zpost_gpmask_False_skymapcl_{cfg.SKYMAP_CL}_cmapnside_{cfg.CMAP_NSIDE}.h5'
            # gw_zpost_cw_path = f'{cfg.MOCKDATA_ROOT}/{output_run}/skymaps_evaluated/{agn_or_alt}/zpost_gpmask_True_skymapcl_{cfg.SKYMAP_CL}_cmapnside_{cfg.CMAP_NSIDE}.h5'
    
    
        z = zpost_file[str(gw_id)]['eval_ax'][:]
        p = zpost_file[str(gw_id)]['posterior'][:]

        z_cw = cw_zpost_file[str(gw_id)]['eval_ax'][:]
        p_cw = cw_zpost_file[str(gw_id)]['posterior'][:]
    
    gwpost_interp = CubicSpline(z, p, extrapolate=False)
    gw_redshift_posterior_marginalized_evaluated = gwpost_interp(cfg.Z_INTEGRAL_AX)
    gw_redshift_posterior_marginalized_evaluated[np.isnan(gw_redshift_posterior_marginalized_evaluated)] = 0  # NaNs outside extrapolation range changed to zeros

    # if not cfg.MASK_GALACTIC_PLANE:
    #     gw_redshift_posterior_marginalized_cw_evaluated = gw_redshift_posterior_marginalized_evaluated.copy()
    # else:
    #     # z, p = np.load(gw_zpost_cw_path)
        
    gwpost_interp_cw = CubicSpline(z_cw, p_cw, extrapolate=False)
    gw_redshift_posterior_marginalized_cw_evaluated = gwpost_interp_cw(cfg.Z_INTEGRAL_AX)
    gw_redshift_posterior_marginalized_cw_evaluated[np.isnan(gw_redshift_posterior_marginalized_cw_evaluated)] = 0  # NaNs outside extrapolation range changed to zeros'

    return gw_redshift_posterior_marginalized_evaluated, gw_redshift_posterior_marginalized_cw_evaluated


def crossmatch(
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
        # plt.figure()
        # plt.plot(z, p)
        # plt.show()
        # plt.figure()
        # plt.plot(cfg.Z_INTEGRAL_AX, fc_of_z)
        # plt.show()
        # plt.figure()
        # plt.plot(cfg.Z_INTEGRAL_AX, gw_redshift_posterior_marginalized_evaluated)
        # plt.show()
        # plt.figure()
        # plt.plot(cfg.Z_INTEGRAL_AX, gw_redshift_posterior_marginalized_cw_evaluated)
        # plt.show()
        # plt.figure()
        # plt.plot(cfg.Z_INTEGRAL_AX, gw_redshift_posterior_marginalized_evaluated - fc_of_z * gw_redshift_posterior_marginalized_cw_evaluated)
        # plt.show()
    
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

        # def print_var_sizes(namespace, top=None):
        #     items = [
        #         (name, type(val).__name__, sys.getsizeof(val))
        #         for name, val in namespace.items()
        #         if not name.startswith("__")
        #     ]

        #     items.sort(key=lambda x: x[2], reverse=True)

        #     if top:
        #         items = items[:top]

        #     for name, typ, size in items:
        #         size_mb = size / (1024 ** 2)
        #         print(f"{name:20} {typ:15} {size_mb:8.3f} MB")
        #     return

        # print_var_sizes(locals())
        # sys.exit(1)

        # t = time.time()
        # agn_pix_labels = np.searchsorted(unique_gw_pixidx_containing_agn,
        #                                 gw_pixidx_at_agn_locs_within_cl).astype(np.int64)
        # dP_dA_per_agn  = dP_dA[unique_gw_pixidx_containing_agn[agn_pix_labels]].astype(np.float32)
        # integrand_new = compute_integrand(
        #     agn_redshift_posteriors_in_cl,
        #     gw_redshift_posterior_in_allpix,
        #     agn_pix_labels,
        #     dP_dA_per_agn,
        #     16385
        # )
        # print(np.sum(np.isclose(integrand, integrand_new)) == len(integrand))
        # print(time.time() - t, 'new\n')


        # Normalize
        integrand /= nagn_norm
        # LOSzprior /= nagn_norm

        # Calculate evidence
        S_agn_incat = romb(integrand * p_rate_of_z_agn / PEprior * jacobian, dx=dz) * 4 * np.pi * average_completeness / agn_population_prior_normalization  # 1/4pi from PEprior does not cancel, since the AGN sky posterior is delta(Omega_i - Omega)

    return S_agn_incat, S_agn_outofcat, S_alt


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

    ### Complete catalog to preserve proper distribution, i.e., without overdensity below cfg.ZMAX due to adding GW-generating AGN first ###
    agn_ra_complete, agn_dec_complete, agn_rcom_complete, n2complete = fill_catalog_to_complete(agn_ra, agn_dec, agn_rcom, cfg=cfg)
    ############################################################################
    # print(len(agn_ra_complete), len(agn_ra))
    # plt.figure()
    # plt.hist(fast_z_at_value(COSMO.comoving_distance, agn_rcom * u.Mpc), density=True, bins=15)
    # plt.plot(cfg.AGN_ZPRIOR_NORM_AX, cfg.AGN_ZPRIOR_FUNCTION(cfg.AGN_ZPRIOR_NORM_AX))
    # plt.show()
    # sys.exit(1)
    
    if cfg.ADD_NAGN_TO_CAT > n2complete + len(agn_ra_complete):  # Add uncorrelated AGN as background
        if cfg.VERBOSE:
            print(f'Adding {cfg.ADD_NAGN_TO_CAT - n2complete - len(agn_ra_complete)} more AGN.')

        agn_ra_complete, agn_dec_complete, agn_rcom_complete = add_agn_to_catalog(agn_ra_complete, agn_dec_complete, agn_rcom_complete, cfg.ADD_NAGN_TO_CAT - n2complete - len(agn_ra_complete), cfg=cfg)

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
        redshift_completeness = z_selection_function

    else:  # Measure the selection function from the data realization
        latitude_mask, _ = make_latitude_selection(agn_ra_complete, agn_dec_complete, obs_agn_rlum_complete, cfg=cfg)  # Measure completeness in the surveyed sky area
        expected_distribution = np.sum(latitude_mask) * cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX) / romb(cfg.AGN_ZPRIOR_FUNCTION(cfg.AGN_ZPRIOR_NORM_AX), dx=np.diff(cfg.AGN_ZPRIOR_NORM_AX)[0])
        no_zero = (expected_distribution != 0)

        redshift_agn_selection_function = np.zeros_like(expected_distribution)
        redshift_agn_selection_function[no_zero] = sum_of_posteriors_incomplete[no_zero] / expected_distribution[no_zero]
        redshift_agn_selection_function[redshift_agn_selection_function > 1] = 1
        redshift_completeness = interp1d(cfg.Z_INTEGRAL_AX, redshift_agn_selection_function, bounds_error=False, fill_value=0)

    ### True selection function requires selection function and likelihood. Single cut + Gaussian likelihood gives the following expression:
    # redshift_completeness = lambda z: stats.norm.cdf(cfg.AGN_ZCUT, loc=z, scale=cfg.AGN_ZERROR)
    
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
    
    # plt.figure()
    # plt.plot(cfg.Z_INTEGRAL_AX, redshift_completenessx(cfg.Z_INTEGRAL_AX) * cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX), label='True')
    # # plt.plot(cfg.Z_INTEGRAL_AX, z_selection_function(cfg.Z_INTEGRAL_AX), label='Selection function')
    # plt.plot(cfg.Z_INTEGRAL_AX, redshift_completeness(cfg.Z_INTEGRAL_AX) * cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX), label='P26')
    # plt.xlabel('Redshift')
    # plt.ylabel('Pdet(z) * Ppop(z)')
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.plot(cfg.Z_INTEGRAL_AX, redshift_completenessx(cfg.Z_INTEGRAL_AX), label='True')
    # plt.plot(cfg.Z_INTEGRAL_AX, z_selection_function(cfg.Z_INTEGRAL_AX), label='Selection function')
    # plt.plot(cfg.Z_INTEGRAL_AX, redshift_completeness(cfg.Z_INTEGRAL_AX), label='P26')
    # plt.xlabel('Redshift')
    # plt.ylabel('Completeness')
    # plt.legend()
    # plt.show()
    

    ### Estimate from redshift means and bins
#     arr = np.array([
#     [0.0000e+00, 3.0000e+01, 1.9800e+02, 8.2600e+02, 2.9710e+03, 8.2060e+03, 1.8934e+04, 3.5073e+04],
#     [3.0000e+00, 2.2000e+01, 1.7100e+02, 8.5500e+02, 2.9970e+03, 8.3530e+03, 1.8657e+04, 3.5271e+04],
#     [3.0000e+00, 2.2000e+01, 1.5700e+02, 8.2600e+02, 3.0460e+03, 8.5020e+03, 1.8827e+04, 3.5055e+04]
# ])

#     expected_nagn_in_bin = np.around(arr.mean(axis=0) / 1e6 * len(agn_rcom_complete))

#     observed_nagn_in_bin, bins, _ = plt.hist(obs_agn_redshift, histtype='step', linewidth=2, bins=np.linspace(0, 1.5, 9))
#     plt.close()

#     fc = np.zeros_like(expected_nagn_in_bin)
#     fc[expected_nagn_in_bin != 0] = observed_nagn_in_bin[expected_nagn_in_bin != 0] / expected_nagn_in_bin[expected_nagn_in_bin != 0] / (1 - np.sin(np.deg2rad(10)))
#     fc[fc > 1] = 1

#     def make_fc_lookup(bins, fc):
#         bins = np.asarray(bins)
#         fc = np.asarray(fc)

#         def fc_of_z(z):
#             z = np.asarray(z)

#             # Find bin indices
#             idx = np.digitize(z, bins) - 1

#             # Handle out-of-range values
#             idx[idx < 0] = 0
#             idx[idx >= len(fc)] = len(fc) - 1

#             res = fc[idx]
#             res[z > 1.5] = 0

#             return res

#         return fc_of_z

#     redshift_completeness_v25 = make_fc_lookup(bins, fc)

#     plt.figure()
#     plt.plot(cfg.Z_INTEGRAL_AX, redshift_completenessx(cfg.Z_INTEGRAL_AX), label='True', color='black', linewidth=3)
#     plt.plot(cfg.Z_INTEGRAL_AX, redshift_completeness_v25(cfg.Z_INTEGRAL_AX), linewidth=2, label='V25')
#     plt.plot(cfg.Z_INTEGRAL_AX, redshift_completeness(cfg.Z_INTEGRAL_AX), linewidth=2, label='P26')
#     plt.xlabel('Redshift')
#     plt.ylabel('Pdet(z)')
#     plt.legend()
#     plt.show()

#     plt.figure()
#     plt.plot(cfg.Z_INTEGRAL_AX, redshift_completenessx(cfg.Z_INTEGRAL_AX) * cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX), label='True', color='black', linewidth=3)
#     plt.plot(cfg.Z_INTEGRAL_AX, redshift_completeness_v25(cfg.Z_INTEGRAL_AX) * cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX), linewidth=2, label='V25')
#     plt.plot(cfg.Z_INTEGRAL_AX, redshift_completeness(cfg.Z_INTEGRAL_AX) * cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX), linewidth=2, label='P26')
#     plt.xlabel('Redshift')
#     plt.ylabel('Pdet(z) * Ppop(z)')
#     plt.legend()
#     plt.show()

#     sys.exit(1)

    # redshift_completeness = z_selection_function

    # print(np.sum(fast_z_at_value(COSMO.comoving_distance, agn_rcom_complete * u.Mpc) < 0.4))
    # print(np.sum(fast_z_at_value(COSMO.comoving_distance, agn_rcom_complete * u.Mpc) > 0.4))
    # zzz = fast_z_at_value(COSMO.comoving_distance, agn_rcom_complete * u.Mpc)
    # print(np.min(agn_rcom_complete), np.max(agn_rcom_complete))
    # print(np.min(zzz), np.max(zzz))

    # print(np.sum(obs_agn_redshift < 0.4))
    # print(np.sum(obs_agn_redshift > 0.4))
    # plt.figure()
    # plt.plot(Z_INTEGRAL_AX, redshift_completeness(Z_INTEGRAL_AX), label='Estimated')
    # plt.plot(Z_INTEGRAL_AX, z_selection_function(Z_INTEGRAL_AX), label='Input')
    # plt.xlabel('Redshift')
    # plt.ylabel('Completeness')
    # plt.legend()
    # plt.show()
    # sys.exit(1)
    return gw_fnames, agn_posterior_dset, agn_ra, agn_dec, obs_agn_redshift, redshift_completeness


def load_quaia(fagn_idx, cfg):
    '''
    Load Quaia and select sources outside the galactic plane and above the specified bolometric luminosity
    '''
    if cfg.LUM_THRESH == 'inf':
        return np.empty((0, len(cfg.Z_INTEGRAL_AX))), np.empty(0), np.empty(0), np.empty(0), lambda z: np.zeros_like(z)

    df = pd.read_csv(cfg.CATALOG_PATH)
    cols = ["redshift_quaia", "redshift_quaia_err", "ra", "dec", "b", "loglbol_corr"]
    data = df[cols]
    b              = data["b"].to_numpy()
    loglbol_corr   = data["loglbol_corr"].to_numpy()

    outside_galactic_plane = np.logical_or((b > 10), (b < -10))
    above_lbol_thresh = loglbol_corr >= float(cfg.LUM_THRESH)

    b                  = b[outside_galactic_plane & above_lbol_thresh]
    loglbol_corr       = loglbol_corr[outside_galactic_plane & above_lbol_thresh]
    agn_redshift       = data["redshift_quaia"].to_numpy()[outside_galactic_plane & above_lbol_thresh]
    agn_redshift_err   = data["redshift_quaia_err"].to_numpy()[outside_galactic_plane & above_lbol_thresh]
    agn_ra             = np.deg2rad( data["ra"].to_numpy()[outside_galactic_plane & above_lbol_thresh] )
    agn_dec            = np.deg2rad( data["dec"].to_numpy()[outside_galactic_plane & above_lbol_thresh] )
    # agn_rlum           = COSMO.luminosity_distance(agn_redshift).value

    agn_posterior_dset, _ = get_agn_posteriors(fagn_idx, agn_redshift, agn_redshift_err, label=cfg.LUM_THRESH, cfg=cfg, replace_old_file=False)
    # _, c_per_zbin, completeness_map = make_incomplete_catalog(agn_ra, agn_dec, agn_rlum, agn_redshift)  # Quaia is already redshift incomplete, but convenient to get completeness maps this way

    filename = f'{cfg.AGN_DIST_DIR}/completeness_{cfg.LUM_THRESH}_{cfg.QLF}.npy'
    if cfg.VERBOSE:
        print(f'Loading continuous selection function calculated from QLF from file: {filename}')
    z, fc_of_z = np.load(filename)
    c_above_1 = fc_of_z > 1
    fc_of_z[c_above_1] = 1.
    redshift_completeness = interp1d(z, fc_of_z, bounds_error=False, fill_value=0)

    return agn_posterior_dset, agn_ra, agn_dec, agn_redshift, redshift_completeness


########################################################################################################################################################


def process_one_fagn(fagn_idx, fagn_realized, cfg):
    
    if cfg.REAL_DATA:
        agn_posterior_dset, agn_ra, agn_dec, obs_agn_redshift, redshift_completeness = load_quaia(fagn_idx, cfg)

        with open(cfg.JSON_PATH, "r") as f:
            gw_path_dict = json.load(f)
        gw_keys = list(gw_path_dict.keys())
        gw_fnames = [gw_path_dict[key] for key in gw_keys]
    
    else:
        print(f'\nRealization {fagn_idx + 1}/{cfg.N_REALIZATIONS}: fagn = {fagn_realized}')

        # Give every process a unique seed -- TODO: save the seeds somewhere
        seed = np.random.SeedSequence().generate_state(1)[0]
        np.random.seed(seed)
        
        gw_fnames, agn_posterior_dset, agn_ra, agn_dec, obs_agn_redshift, redshift_completeness = make_mock_agn_catalog(fagn_idx, fagn_realized, cfg)


    ### Prepare functions for in the likelihood ###
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

    PEprior_func = lambda z: uniform_comoving_prior(z, cosmo=cfg.COSMO)
    PEprior = PEprior_func(cfg.Z_INTEGRAL_AX)
    # PEprior = redshift_pdf_given_lumdist_pdf(z_integral_ax, lumdist_pdf=lambda dl: dl**2)
    
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
        sky_coverage = 1 - np.sin(np.deg2rad(10))
    else:
        sky_coverage = 1.
    
    fc_of_z = redshift_completeness(cfg.Z_INTEGRAL_AX)
    average_redshift_completeness = romb(fc_of_z * normed_agn_background_dist * jacobian, dx=dz)
    average_completeness = average_redshift_completeness * sky_coverage

    alpha_alt = cfg.ALPHA_ALT
    Pdet = cfg.PDET
    pdet = Pdet(cfg.Z_INTEGRAL_AX)
    pdet[np.isnan(pdet)] = 0

    # Get zprior normalizations, dealing with delta-function AGN posteriors (then assume_perfect_redshift == True) and empty catalogues (then total_n_agn == 0)
    if cfg.ASSUME_PERFECT_REDSHIFT:
        agn_below_zmax_mask = obs_agn_redshift < cfg.ZMAX
        agn_below_zmax = obs_agn_redshift[agn_below_zmax_mask]
        nagn_norm = np.sum(agn_below_zmax_mask)

        if nagn_norm == 0:
            agn_population_prior_normalization = romb((1 - fc_of_z) * normed_agn_background_dist * p_rate_of_z_agn * jacobian, dx=dz)

            alpha_agn = romb(pdet * (1 - fc_of_z) * normed_agn_background_dist * p_rate_of_z_agn * jacobian / agn_population_prior_normalization, dx=dz)

        else:            
            agn_population_prior_normalization = average_redshift_completeness * np.sum(p_rate_of_z_agn_func(agn_below_zmax)) / nagn_norm + romb((1 - fc_of_z) * normed_agn_background_dist * p_rate_of_z_agn * jacobian, dx=dz)
    
            pdet_at_agnz = Pdet(agn_below_zmax)
            pdet_at_agnz[np.isnan(pdet_at_agnz)] = 0
            alpha_agn = np.sum( pdet_at_agnz * average_redshift_completeness * p_rate_of_z_agn_func(agn_below_zmax) / nagn_norm )
            alpha_agn += romb(pdet * (1 - fc_of_z) * normed_agn_background_dist * p_rate_of_z_agn * jacobian, dx=dz)
            alpha_agn /= agn_population_prior_normalization
    
    else:
        sum_of_all_agn_posteriors = np.sum(agn_posterior_dset, axis=0)
        nagn_norm = romb(sum_of_all_agn_posteriors, dx=dz)

        if nagn_norm == 0:
            agn_population_prior_normalization = romb((1 - fc_of_z) * normed_agn_background_dist * p_rate_of_z_agn * jacobian, dx=dz)

            alpha_agn = romb(pdet * (1 - fc_of_z) * normed_agn_background_dist * p_rate_of_z_agn * jacobian, dx=dz) / agn_population_prior_normalization
        else:
            # p_rate_of_z_agn imposes a redshift cut in the GW population, up to which the pop. is normalized. Therefore agn_population_prior only has to be evaluated at redshifts up to this cut.
            agn_population_prior = average_redshift_completeness * sum_of_all_agn_posteriors / nagn_norm + (1 - fc_of_z) * normed_agn_background_dist 
            agn_population_prior_rate_weighted = agn_population_prior * p_rate_of_z_agn
            agn_population_prior_normalization = romb(agn_population_prior_rate_weighted * jacobian, dx=dz)

            alpha_agn = romb(pdet * agn_population_prior_rate_weighted * jacobian, dx=dz) / agn_population_prior_normalization

    print(alpha_agn, 'ALPHA_AGN')

    ### Calculate the integrals in the likelihood ###
    Ngws = len(gw_fnames)  # Due to selection effects not always the same number
    S_agn_incat = np.zeros(Ngws)
    S_agn_outofcat = np.zeros(Ngws)
    S_alt = np.zeros(Ngws)

    # S_agn_incat_dict = {}
    # S_agn_outofcat_dict = {}
    # S_alt_dict = {}
    # from_agn_dict = {}

    subdir = '/'.join(gw_fnames[0].split('/')[:-3])
    gw_zpost_path_agn = f'{subdir}/skymaps_evaluated/agn/zpost_gpmask_False_skymapcl_{cfg.SKYMAP_CL}_cmapnside_{cfg.CMAP_NSIDE}.h5'
    gw_zpost_cw_path_agn = f'{subdir}/skymaps_evaluated/agn/zpost_gpmask_True_skymapcl_{cfg.SKYMAP_CL}_cmapnside_{cfg.CMAP_NSIDE}.h5'
    gw_zpost_path_alt = f'{subdir}/skymaps_evaluated/alt/zpost_gpmask_False_skymapcl_{cfg.SKYMAP_CL}_cmapnside_{cfg.CMAP_NSIDE}.h5'
    gw_zpost_cw_path_alt = f'{subdir}/skymaps_evaluated/alt/zpost_gpmask_True_skymapcl_{cfg.SKYMAP_CL}_cmapnside_{cfg.CMAP_NSIDE}.h5'

    with h5py.File(gw_zpost_path_agn, "r") as from_agn_hdf5, \
        h5py.File(gw_zpost_path_alt, "r") as from_alt_hdf5, \
        h5py.File(gw_zpost_cw_path_agn, "r") as from_agn_cw_hdf5, \
        h5py.File(gw_zpost_cw_path_alt, "r") as from_alt_cw_hdf5:

        for gw_idx, filename in enumerate(gw_fnames):
            
            if cfg.REAL_DATA:
                gwkey = gw_keys[gw_idx]
            else:
                gwkey = get_id_from_fname(filename)

            if cfg.VERBOSE:
                print(f'({gw_idx+1}/{len(gw_fnames)})')

            if cfg.USE_SKYMAPS:
                sagn_incat, sagn_outofcat, salt = crossmatch(
                                                            cfg=cfg,
                                                            filename=filename,
                                                            agn_posterior_dset=agn_posterior_dset,              # AGN data (needed when using AGN z-errors)
                                                            agn_ra=agn_ra,                                      # AGN data (needed when neglecting AGN z-errors)
                                                            agn_dec=agn_dec,                                    # AGN data (needed when neglecting AGN z-errors)
                                                            agn_redshift=obs_agn_redshift,                       # AGN data (needed when neglecting AGN z-errors)
                                                            p_rate_of_z_agn_func=p_rate_of_z_agn_func,
                                                            p_rate_of_z_agn=p_rate_of_z_agn,
                                                            p_rate_of_z_alt=p_rate_of_z_alt,
                                                            PEprior_func=PEprior_func,
                                                            PEprior=PEprior,
                                                            fc_of_z=fc_of_z,
                                                            average_completeness=average_completeness,
                                                            sky_coverage=sky_coverage,
                                                            normed_agn_background_dist=normed_agn_background_dist,
                                                            nagn_norm=nagn_norm,
                                                            agn_population_prior_normalization=agn_population_prior_normalization,
                                                            from_agn_hdf5=from_agn_hdf5, 
                                                            from_alt_hdf5=from_alt_hdf5, 
                                                            from_agn_cw_hdf5=from_agn_cw_hdf5, 
                                                            from_alt_cw_hdf5=from_alt_cw_hdf5,
                                                            gwkey=gwkey
                                                        )                      
                if np.isnan(sagn_incat) and np.isnan(sagn_outofcat) and np.isnan(salt):
                    print(filename)
            else:
                NotImplementedError('Only inference using GW skymaps is currently tested.')
                # if cfg.VERBOSE:
                #     print(f'Using GW posterior samples!')
                # with h5py.File(filename, 'r') as posterior_samples:
                #     sagn_incat, sagn_outofcat, salt = crossmatch_from_samples_p26(posterior_samples=posterior_samples, 
                #                                                                   z_integral_ax=cfg.Z_INTEGRAL_AX,
                #                                                                   agn_posterior_dset=agn_posterior_dset,
                #                                                                   agn_ra=agn_ra,
                #                                                                   agn_dec=agn_dec,
                #                                                                   completeness_map=completeness_map,
                #                                                                   redshift_completeness=redshift_completeness,
                #                                                                   gw_zcut=cfg.ZMAX,
                #                                                                   merger_rate_func=cfg.MERGER_RATE_EVOLUTION,
                #                                                                   correct_time_dilation=cfg.CORRECT_TIME_DILATION,
                #                                                                   background_agn_distribution=cfg.AGN_ZPRIOR_FUNCTION,
                #                                                                   linax=cfg.LINAX,
                #                                                                   minpix=30,
                #                                                                   skymap_cl=cfg.SKYMAP_CL,
                #                                                                   minsamps=100,
                #                                                                   **cfg.MERGER_RATE_KWARGS)
    
            S_agn_incat[gw_idx] = sagn_incat
            S_agn_outofcat[gw_idx] = sagn_outofcat
            S_alt[gw_idx] = salt

            # S_agn_incat_dict[gwkey] = sagn_incat
            # S_agn_outofcat_dict[gwkey] = sagn_outofcat
            # S_alt_dict[gwkey] = salt

            if cfg.REAL_DATA:
                from pathlib import Path

                if cfg.LUM_THRESH == 'inf':
                    json_path = '/home/lucas/Documents/PhD/gw_data/real_output_nocat.json'
                else:
                    json_path = '/home/lucas/Documents/PhD/gw_data/real_output.json'

                output_file = Path(json_path)

                # Load existing data
                if output_file.exists():
                    data = json.loads(output_file.read_text())
                else:
                    data = {}

                if not cfg.AGN_ZPRIOR in data.keys():
                    data[cfg.AGN_ZPRIOR] = {}

                datadict = data[cfg.AGN_ZPRIOR] 
                datadict[gwkey]= {'S_agn_incat': sagn_incat, 'S_agn_outcat': sagn_outofcat, 'S_alt': salt}
                
                output_file.write_text(json.dumps(data, indent=2))

            # if int(key) in gw_identifiers:
            #     from_agn_dict[key] = True
            # else:
            #     from_agn_dict[key] = False

            if cfg.VERBOSE:
                print(f'S_alt: {salt}, S_incat: {sagn_incat}, S_outcat: {sagn_outofcat}, S_agn: {sagn_incat + sagn_outofcat}, negative values: {np.sum((cfg.LOG_LLH_X_AX * (sagn_incat + sagn_outofcat - salt) + salt) < 0)}\n')
    
    del agn_posterior_dset  # Free up the memory asap

    
    ### Evaluate the likelihood ###
    S_agn_incat = S_agn_incat[~np.isnan(S_agn_incat)]
    S_agn_outofcat = S_agn_outofcat[~np.isnan(S_agn_outofcat)]
    S_alt = S_alt[~np.isnan(S_alt)]

    loglike = np.log(cfg.SKYMAP_CL * cfg.LOG_LLH_X_AX[None,:] * (S_agn_incat[:,None] + S_agn_outofcat[:,None] - S_alt[:,None]) + S_alt[:,None])
    total_loglike = np.sum(loglike, axis=0) - Ngws * np.log(alpha_agn * cfg.LOG_LLH_X_AX + alpha_alt * (1 - cfg.LOG_LLH_X_AX))  # Sum over all GWs

    nans = np.isnan(loglike)
    if np.sum(nans) != 0:
        print('Got NaNs:')
        arr = np.ones_like(cfg.LOG_LLH_X_AX)
        print((arr[None,:] * S_agn_incat[:,None])[nans])
        print((arr[None,:] * S_agn_outofcat[:,None])[nans])
        print((arr[None,:] * S_alt[:,None])[nans])

    return fagn_idx, total_loglike
