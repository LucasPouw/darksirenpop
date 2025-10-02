import astropy_healpix as ah
import healpy as hp
import numpy as np
from ligo.skymap import distance, moc
import time
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c
from scipy.integrate import romb
from numba import njit, prange
import h5py
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from default_arguments import DEFAULT_COSMOLOGY as COSMO

# from collections import namedtuple
# from tqdm import tqdm
# import os
# from astropy import units as u
# from astropy.coordinates import ICRS, SkyCoord, SphericalRepresentation
# import sys
# from scipy import stats
# import matplotlib.pyplot as plt


INV_SQRT2PI = 1.0 / np.sqrt(2.0 * np.pi)
SPEED_OF_LIGHT_KMS = c.to('km/s').value

@njit(parallel=True, fastmath=True)
def allsky_marginal_lumdist_distribution(dl_array, dP, norm, mu, sigma):
    M = dl_array.shape[0]
    N = mu.shape[0]
    out = np.empty(M)

    for j in prange(M):                 # parallel over dl_array
        dl = dl_array[j]
        s = 0.0
        for i in range(N):              # fast inner loop
            diff = (dl - mu[i]) / sigma[i]
            gauss = np.exp(-0.5 * diff * diff) * INV_SQRT2PI / sigma[i]
            s += dP[i] * norm[i] * dl * dl * gauss
        out[j] = s
    return out


# def allsky_marginal_lumdist_distribution(dl_array, dP, norm, mu, sigma):
#     dl_array = np.atleast_1d(dl_array)             # shape (M,)
#     # Broadcast: (M,1) vs (N,)
#     dl = dl_array[:, None]                         # (M,1)
#     mu = mu[None, :]                               # (1,N)
#     sigma = sigma[None, :]                         # (1,N)
#     norm = norm[None, :]
#     dP = dP[None, :]

#     gauss = np.exp(-0.5 * ((dl - mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))
#     result = np.sum(dP * norm * (dl**2) * gauss, axis=1)  # sum over N
#     return result


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)


def LOS_lumdist_ansatz(dl, distnorm, distmu, distsigma):
    'The ansatz is normalized on dL [0, large]'     
    return dl**2 * distnorm * gaussian(dl, distmu, distsigma)


def redshift_pdf_given_lumdist_pdf(z, lumdist_pdf, **kwargs):
    '''lumdist_pdf is assumed to be normalized'''
    dl = COSMO.luminosity_distance(z).value
    H_z = COSMO.H(z).value  # H(z) in km/s/Mpc
    chi_z = dl / (1 + z)
    dDL_dz = chi_z + (1 + z) * (SPEED_OF_LIGHT_KMS / H_z)
    return lumdist_pdf(dl, **kwargs) * dDL_dz


def uniform_comoving_prior(z):
    '''Proportional to uniform in comoving volume prior.'''
    z = np.atleast_1d(z)
    chi = COSMO.comoving_distance(z).value         # Mpc
    H_z = COSMO.H(z).value                         # km/s/Mpc
    dchi_dz = SPEED_OF_LIGHT_KMS / H_z             # Mpc
    p = (chi**2 * dchi_dz)
    return p


def z_cut(z, zcut):
    '''Artificial redshift cuts in data analysis should be taken into account.'''
    stepfunc = np.ones_like(z)
    stepfunc[z > zcut] = 0
    return stepfunc


def merger_rate_madau(z, gamma=4.59, k=2.86, zp=2.47):
        """
        Madau rate evolution
        """
        C = 1 + (1 + zp)**(-gamma - k)
        return C * ((1 + z)**gamma) / (1 + ((1 + z) / (1 + zp))**(gamma + k))


def merger_rate_uniform(z):
    return np.ones_like(z)


def merger_rate(z, func, **kwargs):
    return func(z, **kwargs)


def crossmatch_p26(posterior_path, 
                    sky_map,
                    completeness_map,
                    completeness_zedges,
                    completeness_zvals,
                    agn_ra, 
                    agn_dec, 
                    agn_lumdist, 
                    agn_redshift, 
                    skymap_cl, 
                    agn_redshift_err, 
                    assume_perfect_redshift, 
                    s_agn_z_integral_ax, 
                    s_alt_z_integral_ax,
                    gw_zcut,
                    merger_rate_func=merger_rate_uniform, 
                    **merger_rate_kwargs):

    if not assume_perfect_redshift:
        assert np.max(np.diff(s_agn_z_integral_ax)[0]) < np.min(agn_redshift_err), 'LOS zprior resolution is too coarse to capture AGN distribution fully.'

    dz_Sagn = np.diff(np.log10(s_agn_z_integral_ax))[0]
    dz_Salt = np.diff(np.log10(s_alt_z_integral_ax))[0]

    cmap_nside = hp.npix2nside(len(completeness_map))

    def redshift_completeness(z):
        bin_idx = np.digitize(z, completeness_zedges) - 1
        bin_idx[bin_idx == len(completeness_zvals)] = len(completeness_zvals) - 1
        return completeness_zvals[bin_idx.astype(np.int32)]
    
    # zz = np.linspace(0, 1.5, 1000)
    # plt.figure()
    # plt.plot(zz, redshift_completeness(zz))
    # plt.savefig('cz.pdf', bbox_inches='tight')
    # plt.close()
    # sys.exit(1)

    print(f'USING NEW CROSSMATCH, PERFECT REDSHIFT = {assume_perfect_redshift}')

    sky_map = np.flipud(np.sort(sky_map, order="PROBDENSITY"))
    
    # Unpacking skymap
    dP_dA = sky_map["PROBDENSITY"]  # Probdens in 1/sr
    mu = sky_map["DISTMU"]          # Ansatz mean in Mpc
    sigma = sky_map["DISTSIGMA"]    # Ansatz width in Mpc
    norm = sky_map["DISTNORM"]      # Ansatz norm in 1/Mpc

    if np.sum(np.isnan(dP_dA)) == len(dP_dA):
        print('BAD SKYMAP AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        return np.nan, np.nan, np.nan

    # Find the pixel that contains the injection.
    order, ipix = moc.uniq2nest(sky_map["UNIQ"])
    max_order = np.max(order)
    max_nside = ah.level_to_nside(max_order)
    max_ipix = ipix << np.int64(2 * (max_order - order))

    agn_theta = 0.5 * np.pi - agn_dec
    agn_phi = agn_ra
    true_pix = hp.ang2pix(max_nside, agn_theta, agn_phi, nest=True)
    i = np.argsort(max_ipix)
    true_idx = i[np.digitize(true_pix, max_ipix[i]) - 1]  # Indeces that indicate skymap pixels that contain an AGN

    dA = moc.uniq2pixarea(sky_map["UNIQ"])  # Pixel areas in sr
    dP = dP_dA * dA  # Dimensionless probability density in each pixel
    # print('VERIFICATION:', np.sum(dP))
    cumprob = np.cumsum(dP)
    cumprob[cumprob > 1] = 1.  # Correcting floating point error which could cause issues when skymap_cl == 1
    searched_prob = cumprob[true_idx]

    # Getting only relevant AGN and pixels from the skymap
    agn_mask_within_cl = (searched_prob <= skymap_cl)
    gw_idx_with_agn = true_idx[agn_mask_within_cl]

    unique_gw_idx, counts = np.unique(gw_idx_with_agn, return_counts=True)
    nagn_within_cl = np.sum(counts)
    print(f'Found {nagn_within_cl} AGN within {skymap_cl} CL in {len(unique_gw_idx)} pixels')

    # Calculating the evidence for each hypothesis
    print('Calculating S_agn...')
    S_agn_cw = 0
    if assume_perfect_redshift:
        print('PERFECT REDSHIFT NOT YET IMPLEMENTED WITH VARYING MERGER RATE')

        # for _, has_agn in zip(counts, unique_gw_idx):
        #     # print(f'Pixel: {has_agn} contains {count} AGN')
        #     gw_redshift_posterior_in_pix = lambda z: redshift_pdf_given_lumdist_pdf(z, LOS_lumdist_ansatz, distnorm=norm[has_agn], distmu=mu[has_agn], distsigma=sigma[has_agn])

        #     agn_in_pix = (true_idx == has_agn)

        #     # Only evaluate the GW posterior for AGN within 5sigma. If outside this range, the probability will be floored to 0. The AGN is still counted in the normalization.
        #     lumdist_5sig = (agn_lumdist[agn_in_pix] > (mu[has_agn] - 5 * sigma[has_agn])) & (agn_lumdist[agn_in_pix] < (mu[has_agn] + 5 * sigma[has_agn]))
        #     below_zcut = agn_redshift[agn_in_pix] < gw_zcut
        #     if np.sum(lumdist_5sig & below_zcut) == 0:  # The contribution to S_agn is 0 in this pixel, so skip
        #         continue

        #     selected_agn_redshifts = agn_redshift[agn_in_pix][lumdist_5sig & below_zcut]

        #     gw_posterior_at_agn_redshift = gw_redshift_posterior_in_pix(selected_agn_redshifts) #* merger_rate(selected_agn_redshifts, merger_rate_func, **merger_rate_kwargs)
        #     contribution = np.sum( redshift_completeness(selected_agn_redshifts) * gw_posterior_at_agn_redshift * dP_dA[has_agn] / uniform_comoving_prior(selected_agn_redshifts) )  # p_gw(z) * p_gw(Omega), evaluated at AGN position because of delta-function AGN posteriors, sum contributions of all AGN in this pixel
        #     S_agn_cw += contribution

        mp_start = time.time()

        def process_one(args):
            '''
            In this case, it is probably better to mp over many events, maybe that's also better in the case of agn errors xdd, but S_alt calc needs to be not parallel then...
            '''
            has_agn, _ = args

            gw_redshift_posterior_in_pix = lambda z: redshift_pdf_given_lumdist_pdf(z, LOS_lumdist_ansatz, distnorm=norm[has_agn], distmu=mu[has_agn], distsigma=sigma[has_agn])

            agn_in_pix = (true_idx == has_agn)

            # Only evaluate the GW posterior for AGN within 5sigma. If outside this range, the probability will be floored to 0. The AGN is still counted in the normalization.
            lumdist_5sig = (agn_lumdist[agn_in_pix] > (mu[has_agn] - 5 * sigma[has_agn])) & (agn_lumdist[agn_in_pix] < (mu[has_agn] + 5 * sigma[has_agn]))
            below_zcut = agn_redshift[agn_in_pix] < gw_zcut
            if np.sum(lumdist_5sig & below_zcut) == 0:  # The contribution to S_agn is 0 in this pixel, so skip
                return 0

            selected_agn_redshifts = agn_redshift[agn_in_pix][lumdist_5sig & below_zcut]

            gw_posterior_at_agn_redshift = gw_redshift_posterior_in_pix(selected_agn_redshifts) #* merger_rate(selected_agn_redshifts, merger_rate_func, **merger_rate_kwargs)
            contribution = np.sum( redshift_completeness(selected_agn_redshifts) * gw_posterior_at_agn_redshift * dP_dA[has_agn] / uniform_comoving_prior(selected_agn_redshifts) )  # p_gw(z) * p_gw(Omega), evaluated at AGN position because of delta-function AGN posteriors, sum contributions of all AGN in this pixel

            return contribution
        
        tasks = list(zip(unique_gw_idx, counts))
        with ThreadPoolExecutor() as executor:
            contributions = list(executor.map(process_one, tasks))

        S_agn_cw = np.sum(contributions)
        print('Total Sagn time:', time.time() - mp_start)
        
    else:

        ### gw_redshift_posterior_in_pix evaluation and AGN posterior loading are the most expensive operations right now, dominating S_agn computation time ###

        # start = time.time()
        # # post_timer = 0

        # with h5py.File(posterior_path, "r") as f:
        #     dset = f["agn_redshift_posteriors"]
            
        #     for nagn_at_idx, gw_idx in zip(counts, unique_gw_idx):
        #         agn_indices = np.where(true_idx == gw_idx)[0]  # TODO: precompute outside loop
        #         agn_redshift_posterior = dset[agn_indices, :]

        #         phi = agn_phi[agn_indices]
        #         theta = agn_theta[agn_indices]
        #         pix_idx = hp.ang2pix(cmap_nside, theta, phi, nest=True)
        #         completenesses = completeness_map[pix_idx]

        #         # start_post_timer = time.time()
        #         gw_redshift_posterior_in_pix = redshift_pdf_given_lumdist_pdf(s_agn_z_integral_ax, LOS_lumdist_ansatz, distnorm=norm[gw_idx], distmu=mu[gw_idx], distsigma=sigma[gw_idx])
        #         # post_timer += time.time() - start_post_timer

        #         # Population prior should be normalized on z \in [0, infty], but the z_cut makes the integral z \in [0, z_cut], which could be taken into account in the range of S_AGN_Z_INTEGRAL_AX, if you want, but not necessary because of the z_cut() function:
        #         agn_population_prior_unnorm = agn_redshift_posterior * z_cut(s_agn_z_integral_ax, zcut=gw_zcut) * merger_rate(s_agn_z_integral_ax, merger_rate_func, **merger_rate_kwargs) 
        #         agn_population_norms = romb(y=agn_population_prior_unnorm * s_agn_z_integral_ax * np.log(10), dx=dz_Sagn)
        #         # print(agn_population_norms)
                
        #         agn_population_prior = np.sum(completenesses[:,np.newaxis] * agn_population_prior_unnorm / agn_population_norms[:,np.newaxis], axis=0)  # Combine all AGN that talks to the same part of the GW posterior

        #         # Multiply LOS prior with GW posterior and integrate
        #         gw_posterior_times_agn_population_prior = gw_redshift_posterior_in_pix * agn_population_prior * redshift_completeness(s_agn_z_integral_ax)
        #         contribution = dP_dA[gw_idx] * romb(gw_posterior_times_agn_population_prior * s_agn_z_integral_ax * np.log(10), dx=dz_Sagn)  # * nagn_at_idx
                
        #         S_agn_cw += contribution

        # print('SINGLE CORE GOT:', S_agn_cw)
        # print('Total Sagn time:', time.time() - start)


        mp_start = time.time()

        def process_one(args):
            gw_idx, nagn_at_idx = args

            agn_indices = np.where(true_idx == gw_idx)[0]
            
            with h5py.File(posterior_path, "r") as f:
                dset = f["agn_redshift_posteriors"]
                agn_redshift_posterior = dset[agn_indices, :]

            gw_redshift_posterior_in_pix = redshift_pdf_given_lumdist_pdf(
                s_agn_z_integral_ax, LOS_lumdist_ansatz,
                distnorm=norm[gw_idx], distmu=mu[gw_idx], distsigma=sigma[gw_idx]
            )

            # The population prior consists of AGN posteriors, modulated by redshift evolving merger rates
            agn_population_prior_unnorm = (
                agn_redshift_posterior
                * z_cut(s_agn_z_integral_ax, zcut=gw_zcut)
                * merger_rate(s_agn_z_integral_ax, merger_rate_func, **merger_rate_kwargs)
            )
            # Make sure the prior is normalized, which may not be the case due to the redshift evolution modulation
            agn_population_norms = romb(
                y=agn_population_prior_unnorm * s_agn_z_integral_ax * np.log(10),
                dx=dz_Sagn,
            )

            # We can vectorize the contributions of AGN with the same gw_idx
            # TODO: the completeness could differ for AGN with the same gw_idx + sky completeness and redshift completeness should be combined in a 3D map -> redshift_completeness(z, pix)
            agn_population_prior = np.sum(
                redshift_completeness(s_agn_z_integral_ax)  # TODO: make different for each AGN
                * agn_population_prior_unnorm
                / agn_population_norms[:, np.newaxis],
                axis=0,
            )

            gw_posterior_times_agn_population_prior = (
                gw_redshift_posterior_in_pix
                * agn_population_prior
            )
            contribution = dP_dA[gw_idx] * romb(
                gw_posterior_times_agn_population_prior
                * s_agn_z_integral_ax
                * np.log(10),
                dx=dz_Sagn,
            )

            # phi = agn_phi[agn_indices]
            # theta = agn_theta[agn_indices]
            # pix_idx = hp.ang2pix(cmap_nside, theta, phi, nest=True)
            # completeness_at_agn_skypos = completeness_map[pix_idx]

            # contribution = np.sum(dP_dA[gw_idx] * completeness_at_agn_skypos)  # WHEN USING ONLY SKYLOC

            return contribution

        tasks = list(zip(unique_gw_idx, counts))
        with ThreadPoolExecutor() as executor:
            contributions = list(executor.map(process_one, tasks))

        S_agn_cw = np.sum(contributions)
        print('Total Sagn time:', time.time() - mp_start)

        # if count != 1:
        #     print('AYOOOOOOOOOOOOOOOOO')
        #     print(f'Redshifts are {agn_redshift[true_idx == has_agn]}')
        #     print('contribution to S_agn:', contribution)

        #     if np.isnan(S_agn):
        #         print('\n!!! This AGN caused a problem !!!\n')
        
        # plt.figure()
        # z_axis = np.linspace(0, 2, 1000)
        # # plt.plot(z_axis, gw_redshift_posterior_in_pix(z_axis), label='GW posterior', color='blue')
        # plt.plot(z_axis, uniform_comoving_prior(z_axis), label='p_alt', color='red')
        # # plt.plot(z_axis, dVdz_populationprior(z_axis) * gw_redshift_posterior_in_pix(z_axis), label=r'$p(z|d)p_{\rm alt}(z)$', color='teal')
        # plt.vlines(agn_redshift[true_idx == has_agn], 0, 5, label='AGN posterior', color='black')
        # plt.xlabel('Redshift')
        # plt.ylabel('Probability density')
        # # plt.plot(z_axis, agn_redshift_posterior(z_axis), label='AGN')
        # plt.legend()
        # plt.title(f'AGN {count}, pixel {has_agn}')
        # plt.savefig('/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/zpost.pdf', bbox_inches='tight')
        # plt.close()

        # time.sleep(2)


    S_agn_cw *= (4 * np.pi / np.sum(agn_redshift < gw_zcut))
    
    skymap_theta, skymap_phi = moc.uniq2ang(sky_map['UNIQ'])
    pix_idx = hp.ang2pix(cmap_nside, skymap_theta, skymap_phi, nest=True)

    # pixprob_within_cl = (cumprob <= skymap_cl)
    # completenesses = completeness_map[ pix_idx[pixprob_within_cl] ]  # WHEN USING ONLY SKYLOC

    cmap_vals_in_gw_skymap = completeness_map[pix_idx]
    surveyed = (cmap_vals_in_gw_skymap != 0)

    sky_coverage = np.sum(dA[surveyed]) / np.sum(dA)
    S_agn_cw *= sky_coverage

    print('Calculating S_alt...')
    # S_alt = 1  # WHEN USING ONLY SKYLOC
    # S_alt_cw = np.sum(dP * cmap_vals_in_gw_skymap)  # WHEN USING ONLY SKYLOC

    skyprob_nonzero = (dP != 0)
    gw_redshift_posterior_marginalized = lambda z: redshift_pdf_given_lumdist_pdf(z, 
                                                                                    allsky_marginal_lumdist_distribution, 
                                                                                    dP=dP[skyprob_nonzero],
                                                                                    norm=norm[skyprob_nonzero], 
                                                                                    mu=mu[skyprob_nonzero], 
                                                                                    sigma=sigma[skyprob_nonzero])

    gw_redshift_posterior_marginalized_cw = lambda z: redshift_pdf_given_lumdist_pdf(z, 
                                                                                    allsky_marginal_lumdist_distribution, 
                                                                                    dP=dP[surveyed & skyprob_nonzero],
                                                                                    norm=norm[surveyed & skyprob_nonzero], 
                                                                                    mu=mu[surveyed & skyprob_nonzero], 
                                                                                    sigma=sigma[surveyed & skyprob_nonzero])

    # WARNING: THESE LINES HAVE TO BE RETHOUGHT ONCE THE BACKGROUND DISTRIBUTION IS NOT UNIFORM IN COMOVING VOLUME LIKE THE PARAMETER ESTIMATION PRIOR -> use the population distribution in the population prior and divide gw posterior by uniform in comvol later
    alt_redshift_population_prior_rate_weighted = merger_rate(s_alt_z_integral_ax, merger_rate_func, **merger_rate_kwargs) * z_cut(s_alt_z_integral_ax, zcut=gw_zcut)  # Uniform in comvol prior divides out against parameter estimation prior, but has to be present still in the normalization of the population prior.
    alt_redshift_population_prior_rate_weighted /= romb(uniform_comoving_prior(s_alt_z_integral_ax) * alt_redshift_population_prior_rate_weighted * s_alt_z_integral_ax * np.log(10), dx=dz_Salt)

    gw_posterior_times_alt_population_prior = gw_redshift_posterior_marginalized(s_alt_z_integral_ax) * alt_redshift_population_prior_rate_weighted
    gw_posterior_times_cw_alt_population_prior = gw_redshift_posterior_marginalized_cw(s_alt_z_integral_ax) * alt_redshift_population_prior_rate_weighted * redshift_completeness(s_alt_z_integral_ax)  # TODO: only works now with uniform-on-sky redshift completeness
    
    S_alt = romb(y=gw_posterior_times_alt_population_prior * s_alt_z_integral_ax * np.log(10), dx=dz_Salt)
    S_alt_cw = romb(y=gw_posterior_times_cw_alt_population_prior * s_alt_z_integral_ax * np.log(10), dx=dz_Salt)

    S_alt_cw = min(S_alt_cw, S_alt)  # Handle floating point error, causes a NaN in the log-llh if S_alt_cw > S_alt when S_agn_cw = 0

    return S_agn_cw, S_alt_cw, S_alt


'''
Indistinguishable: c=1 and no AGN there, c=0 and many AGN there

So 
if c=1 and no AGN, you have part of the sky that counts in the S_alt_cw integral, while no contribution to S_agn_cw is made (since no AGN is summed over)
if c=0 and no AGN, you have part of the sky that does not contribute to the S_alt_cw integral, while no contribution to S_agn_cw is made (since no AGN is summed over)

If we mask out a part of the sky, this should be c=0, right? So why do I have to multiply S_agn_cw by the sky coverage?
'''


'''
When there are AGN redshift errors, think about how the normalization of the population prior is effectively up to zcut, work it out with pen and paper!
'''