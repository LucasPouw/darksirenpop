from darksirenpop.utilities.redshift_utils import *
from darksirenpop.utilities.redshift_utils import _CHI_INTERP, _DL_INTERP
from darksirenpop.utilities.utils import uniform_shell_sampler, sample_spherical_angles, truncnorm_pdf_inplace

import sys
import healpy as hp
import numpy as np
import glob

from scipy.integrate import romb
from scipy.interpolate import interp1d
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
    Returns the TRUE redshift selection function (which we will approximate later) and a mask of observed AGN
    '''
    if cfg.LUM_THRESH == 'zero':  # No redshift selection
        z_selection_function = lambda z: np.ones_like(z)
        redshift_incomplete_mask = np.ones_like(obs_agn_redshift, dtype=bool)

    elif cfg.LUM_THRESH == 'zero_upto_cut':  # Only detect AGN below redshift AGN_ZCUT
        z_selection_function = lambda z: z_cut(z, zcut=cfg.AGN_ZCUT)
        redshift_incomplete_mask = obs_agn_redshift < cfg.AGN_ZCUT
    
    elif cfg.LUM_THRESH == 'inf':  # Empty catalogue
        z_selection_function = lambda z: np.zeros_like(z)
        redshift_incomplete_mask = np.zeros_like(obs_agn_redshift, dtype=bool)

    else:  # We select sources according to the ESTIMATED selection function of V25. 
        z_selection_function = lambda z: v25_selection_function(z, cfg=cfg)  # This is the true selection function if the AGN z-errors are zero, otherwise, it should be smoothed by the likelihood (see compare_PdetEM_methods.ipynb)

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
    Currently only supports selection of sources outside Galactic plane, or no selection.
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
        hp.mollview(completeness_map, nest=True, coord="G", cmap="coolwarm", min=0, max=1)
        hp.graticule()
        plt.savefig(f'{cfg.PLOT_DIR}/cmap.pdf', bbox_inches='tight')
        plt.close()
    return latitude_mask, completeness_map


def make_incomplete_catalog(agn_ra, agn_dec, obs_agn_rlum, obs_agn_redshift, cfg):
    z_selection_function, redshift_incomplete_mask = make_redshift_selection(obs_agn_redshift, cfg=cfg)  # Making a redshift-incomplete catalog
    latitude_mask, completeness_map = make_latitude_selection(agn_ra, agn_dec, obs_agn_rlum, cfg=cfg)  # Making a sky-incomplete catalog
    incomplete_catalog_mask = (latitude_mask & redshift_incomplete_mask)
    if cfg.VERBOSE:
        print(f'Observed {np.sum(incomplete_catalog_mask)} AGN from realizations, of which {np.sum(obs_agn_redshift[incomplete_catalog_mask] < cfg.ZMAX)} below GW_ZMAX. Average completeness below GW_ZMAX: {np.sum(obs_agn_redshift[incomplete_catalog_mask] < cfg.ZMAX) / np.sum(obs_agn_redshift < cfg.ZMAX):.5f}')
    return incomplete_catalog_mask, z_selection_function, completeness_map


def compute_agn_posteriors_chunk(start, end, all_agn_z, all_agn_z_err, cfg, n_norm=100):
    '''
    Compute a chunk of AGN posteriors. This computation is vectorized.
    AGN redshift posteriors are modelled as truncnorms on [0, inf) with a QLF-based redshift prior.
    The posteriors are then evaluated on Z_INTEGRAL_AX, which is what is necessary for the integrals
    in the likelihood.
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
    
    # Get evaluation of posteriors on the desired axis --> memory-expensive part
    posteriors = truncnorm_pdf_inplace(cfg.Z_INTEGRAL_AX, mu, sigma, zmin=cfg.ZMIN)
    posteriors *= z_cut(cfg.Z_INTEGRAL_AX, zcut=cfg.AGN_ZMAX)
    posteriors *= cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX)
    posteriors /= z_norms[:, None]
    return posteriors


def get_agn_posteriors(obs_agn_redshift, agn_redshift_err, cfg, n_norm=100):
    '''
    To save computation time, the AGN posteriors are pre-calculated and evaluated on the z-integral axis once and kept in memory.
    THIS TAKES A LOT OF MEMORY IF THE Z-INTEGRAL IS VERY HIGH RESOLUTION WITH MANY AGN!
    '''
    if cfg.ASSUME_PERFECT_REDSHIFT:
        return np.empty(0), 1
    else:
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


# 9 Sept 2026: This function is currently unused, but it may be at some point when AGN_ZMAX != ZMAX. But why would you do that?
# def fill_catalog_to_complete(agn_ra, agn_dec, agn_rcom, cfg):
#     '''
#     If GWs are generated up to a redshift (ZMAX) that is lower than the maximum redshift of AGN (AGN_ZMAX),
#     there needs to be a population of non-host AGN in the catalog to preserve the overall AGN distribution.
#     This function does that.
#     '''
#     # Uniform case can be done quicker, but never used so might as well remove clutter --> TODO
#     # if cfg.AGN_ZPRIOR == 'uniform_comoving_volume':
#     #     n2complete = int(round(len(agn_ra) * ( (cfg.AGN_COMDIST_MAX / cfg.COMDIST_MAX)**3 - 1)))
#     #     new_rcom, new_theta, new_phi = uniform_shell_sampler(cfg.COMDIST_MAX, cfg.AGN_COMDIST_MAX, n2complete)
#     # else:

#     # Compare total number of AGN expected up to AGN_COMDIST_MAX versus the total number we got up to COMDIST_MAX.
#     # Since we take the ratio, we don't need to calculate the actual number, only this integral which is proportional to it.
#     rcom_integrate_ax = np.linspace(cfg.COMDIST_MIN, cfg.AGN_COMDIST_MAX, 1024*4+1)
#     total = romb(comdist_pdf_given_redshift_pdf(rcom_integrate_ax, cfg.AGN_ZPRIOR_FUNCTION, cosmo=cfg.COSMO), dx=np.diff(rcom_integrate_ax)[0])

#     rcom_integrate_ax = np.linspace(cfg.COMDIST_MIN, cfg.COMDIST_MAX, 1024*4+1)
#     current = romb(comdist_pdf_given_redshift_pdf(rcom_integrate_ax, cfg.AGN_ZPRIOR_FUNCTION, cosmo=cfg.COSMO), dx=np.diff(rcom_integrate_ax)[0])

#     n2complete = int(round(len(agn_ra) * ( total / current - 1)))
#     if n2complete != 0:

#         new_theta, new_phi = sample_spherical_angles(n2complete)

#         norm = romb(cfg.AGN_ZPRIOR_FUNCTION(cfg.AGN_ZPRIOR_NORM_AX) * (1 - z_cut(cfg.AGN_ZPRIOR_NORM_AX, zcut=cfg.ZMAX)), dx=np.diff(cfg.AGN_ZPRIOR_NORM_AX)[0])
#         target_population = lambda z: cfg.AGN_ZPRIOR_FUNCTION(z) * (1 - z_cut(z, zcut=cfg.ZMAX)) / norm
#         cdf = np.cumsum(target_population(cfg.AGN_ZPRIOR_NORM_AX))
#         cdf /= cdf[-1]
#         unif = np.random.rand(n2complete)
#         new_z = np.interp(unif, cdf, cfg.AGN_ZPRIOR_NORM_AX)
#         new_rcom = _CHI_INTERP(new_z)  #cfg.COSMO.comoving_distance(new_z).value

#         agn_ra = np.append(agn_ra, new_phi)
#         agn_dec = np.append(agn_dec, np.pi * 0.5 - new_theta)
#         agn_rcom = np.append(agn_rcom, new_rcom)

#     if cfg.VERBOSE:
#         print(f'Adding {n2complete} AGN above GW zmax ({cfg.ZMAX}) to get a catalog with distribution: {cfg.AGN_ZPRIOR}.')

#     return agn_ra, agn_dec, agn_rcom, n2complete


def add_agn_propto_z(agn_ra, agn_dec, agn_rcom, nsamps, cfg):
    '''
    Add samples from p(z) ~ pi_agn(z) * z / (1 + z) such that the total sample of AGN follows pi_agn(z). This is
    necessary because the GW-hosting AGN follow q(z) ~ pi_agn(z) / (1 + z).

    FIXME:
    This is not ideal (and not general), and could be avoided by allowing AGN to host multiple GWs. Because then we can just generate
    mock AGN positions and draw GW hosts from that list with replacement. I didn't do this because it may cause a bias,
    however that bias may be small: https://arxiv.org/abs/2212.08694. May be worth doing something about in the future.
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
    '''Add AGN as background noise to our analysis.'''

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
    '''
    Extracts GW skymap paths (gw_fnames) + RA, Dec and comoving distance (Mpc) from mock data directory.
    '''
    output_directory = glob.glob(f'{cfg.MOCKDATA_ROOT}/output_run_{fagn_idx + 1}_*')[0]

    gw_fnames_from_agn = glob.glob(f'{output_directory}/skymaps/agn/skymap*.fits.gz')
    gw_fnames_from_alt = glob.glob(f'{output_directory}/skymaps/alt/skymap*.fits.gz')
    gw_fnames = np.append(gw_fnames_from_agn, gw_fnames_from_alt)
    gw_identifiers = sorted(np.array([get_id_from_fname(f) for f in gw_fnames_from_agn]).astype(int))
    
    if len(gw_identifiers) > 0:  # If there are GWs from AGN in the data set
        true_sources = np.genfromtxt(f'{output_directory}/true_gw_coords/agn/true_r_theta_phi.txt', delimiter=',')
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


def make_mock_agn_catalog(fagn_idx, cfg):
    '''
    Make an incomplete AGN catalog, given the source coordinates of GWs from AGN. Returns the GW filenames, catalog and estimated selection function.
    '''

    gw_fnames, agn_ra, agn_dec, agn_rcom = get_mock_gw_sources(fagn_idx, cfg)  # Get true source coordinates for GWs from AGN to put in the AGN catalog

    ### Complete catalog to preserve proper distribution, i.e., without overdensity below cfg.ZMAX due to adding GW-generating AGN first ###

    # Check if we need to add AGN that cannot host a GW (only happens if COMDIST_MAX < AGN_COMDIST_MAX)
    if cfg.AGN_COMDIST_MAX == cfg.COMDIST_MAX:
        agn_ra_complete, agn_dec_complete, agn_rcom_complete, n2complete = agn_ra, agn_dec, agn_rcom, 0
    else:
        raise ValueError(f'Currently only tested case for AGN_ZMAX = ZMAX, so all AGN can generate GWs.')

        # TODO: Test code below before removing ValueError (but first really ask yourself why you would need that). 
        # I think the AGN added here are not distributed as pi(z) / (1 + z), which is inconsistent with the GW-hosting AGN. 
        # Honestly why did I ever code it this way.
        # agn_ra_complete, agn_dec_complete, agn_rcom_complete, n2complete = fill_catalog_to_complete(agn_ra, agn_dec, agn_rcom, cfg=cfg)

    # Correct for overdensity caused by GW-hosting AGN. Currently, the catalog follows pi(z) / (1 + z), but that needs to be pi(z).
    # This may cause the total number of AGN to exceed the requested cfg.ADD_NAGN_TO_CAT, but oh well.
    agn_ra_complete, agn_dec_complete, agn_rcom_complete = add_agn_propto_z(agn_ra_complete, 
                                                                            agn_dec_complete, 
                                                                            agn_rcom_complete, 
                                                                            int(np.mean(fast_z_at_value(COSMO.comoving_distance, agn_rcom * u.Mpc)) * len(agn_ra_complete)),  # Source: trust me bro
                                                                            cfg)

    # Add uncorrelated AGN as background, up to the requested total number
    if cfg.ADD_NAGN_TO_CAT > n2complete + len(agn_ra_complete):  
        if cfg.VERBOSE:
            print(f'Adding {cfg.ADD_NAGN_TO_CAT - n2complete - len(agn_ra_complete)} more AGN.')

        agn_ra_complete, agn_dec_complete, agn_rcom_complete = add_agn_to_catalog(agn_ra_complete, agn_dec_complete, agn_rcom_complete, cfg.ADD_NAGN_TO_CAT - n2complete - len(agn_ra_complete), cfg=cfg)

    ### Make AGN observations ###

    if len(agn_rcom_complete) == 0:
        obs_agn_redshift_complete, agn_redshift_err_complete = np.empty_like(agn_rcom_complete), np.empty_like(agn_rcom_complete)
        obs_agn_rlum_complete = np.empty_like(agn_rcom_complete)
    else:
        obs_agn_redshift_complete, agn_redshift_err_complete = get_observed_redshift_from_rcom(agn_rcom_complete, cfg=cfg)
        obs_agn_rlum_complete = _DL_INTERP(obs_agn_redshift_complete) #cfg.COSMO.luminosity_distance(obs_agn_redshift_complete).value

    if cfg.VERBOSE:
        print(f'Number of AGN in complete catalog: {len(agn_rcom_complete)}')

    ### Make an incomplete AGN catalog from these coordinates ###

    incomplete_catalog_mask, z_selection_function, completeness_map = make_incomplete_catalog(agn_ra_complete, agn_dec_complete, obs_agn_rlum_complete, obs_agn_redshift_complete, cfg=cfg)
    agn_ra = agn_ra_complete[incomplete_catalog_mask]
    agn_dec = agn_dec_complete[incomplete_catalog_mask]
    obs_agn_redshift = obs_agn_redshift_complete[incomplete_catalog_mask]
    agn_redshift_err = agn_redshift_err_complete[incomplete_catalog_mask]

    agn_posterior_dset, sum_of_posteriors_incomplete = get_agn_posteriors(obs_agn_redshift, agn_redshift_err, cfg=cfg)

    ### Characterize the redshift-completeness ###

    if cfg.ASSUME_PERFECT_REDSHIFT or cfg.LUM_THRESH == 'inf':
        # TODO: Should actually still measure from the data if AGN_ZERROR is not 0, 
        # instead of just feeding the true selection function in: we would only recover the true selection function if the redshift errors are zero.
        redshift_completeness = z_selection_function  

    else:  # Measure the selection function from the data realization
        latitude_mask, _ = make_latitude_selection(agn_ra_complete, agn_dec_complete, obs_agn_rlum_complete, cfg=cfg)  # Measure completeness in the surveyed sky area
        expected_distribution = np.sum(latitude_mask) * cfg.AGN_ZPRIOR_FUNCTION(cfg.Z_INTEGRAL_AX) / romb(cfg.AGN_ZPRIOR_FUNCTION(cfg.AGN_ZPRIOR_NORM_AX), dx=np.diff(cfg.AGN_ZPRIOR_NORM_AX)[0])
        no_zero = (expected_distribution != 0)

        redshift_agn_selection_function = np.zeros_like(expected_distribution)
        redshift_agn_selection_function[no_zero] = sum_of_posteriors_incomplete[no_zero] / expected_distribution[no_zero]
        redshift_agn_selection_function[redshift_agn_selection_function > 1] = 1
        redshift_completeness = interp1d(cfg.Z_INTEGRAL_AX, redshift_agn_selection_function, bounds_error=False, fill_value=0)

    return gw_fnames, agn_posterior_dset, agn_ra, agn_dec, obs_agn_redshift, redshift_completeness
