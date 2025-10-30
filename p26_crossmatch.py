import astropy_healpix as ah
import healpy as hp
import numpy as np
from ligo.skymap import moc
from scipy.integrate import romb
from default_globals import *
from redshift_utils import z_cut, merger_rate, uniform_comoving_prior, redshift_pdf_given_lumdist_pdf
from utils import gaussian
from numba import njit, prange
from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import time
# import sys


INV_SQRT2PI = 1.0 / np.sqrt(2.0 * np.pi)


# @njit(parallel=True, fastmath=True)
# def allsky_marginal_lumdist_distribution(dl_array, dP, norm, mu, sigma):
#     M = dl_array.shape[0]
#     N = mu.shape[0]
#     out = np.empty(M)

#     for j in prange(M):                 # parallel over dl_array
#         dl = dl_array[j]
#         s = 0.0
#         for i in range(N):              # fast inner loop
#             diff = (dl - mu[i]) / sigma[i]
#             gauss = np.exp(-0.5 * diff * diff) * INV_SQRT2PI / sigma[i]
#             s += dP[i] * norm[i] * dl * dl * gauss
#         out[j] = s
#     return out

# # import time
def allsky_marginal_lumdist_distribution(dl_array, dP, norm, mu, sigma):
    '''Faster in mp for some reason'''
    # s = time.time()
    dl_array = np.atleast_1d(dl_array)             # shape (M,)
    # Broadcast: (M,1) vs (N,)
    dl = dl_array[:, None]                         # (M,1)
    mu = mu[None, :]                               # (1,N)
    sigma = sigma[None, :]                         # (1,N)
    norm = norm[None, :]
    dP = dP[None, :]

    gauss = np.exp(-0.5 * ((dl - mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))
    result = np.sum(dP * norm * (dl**2) * gauss, axis=1)  # sum over N
    # print(time.time() - s)
    return result


# def allsky_marginal_lumdist_distribution(dl_array, dP, norm, mu, sigma):
#     # s = time.time()
#     dl_array = np.atleast_1d(dl_array)
#     dl = dl_array[:, None]  # (M,1)

#     inv_sigma = 1.0 / sigma
#     coeff = dP * norm * inv_sigma / np.sqrt(2 * np.pi)

#     diff = dl - mu  # (M,N)
#     exp_term = np.exp(-0.5 * (diff * inv_sigma)**2)  # (M,N)

#     # Multiply efficiently, avoiding redundant broadcasting
#     result = (dl_array**2) * (exp_term @ coeff)
#     # print(time.time() - s)
#     return result


def LOS_lumdist_ansatz(dl, distnorm, distmu, distsigma):
    'The ansatz is normalized on dL [0, large]'     
    return dl**2 * distnorm * gaussian(dl, distmu, distsigma)


def crossmatch_p26(agn_posterior_dset, 
                    sky_map,
                    completeness_map,
                    completeness_zedges,
                    completeness_zvals,
                    agn_ra, 
                    agn_dec, 
                    agn_lumdist,  # agn_lumdist is only needed for speed up of s_agn_cw when assume_perfect_redshift = True - is it worth the added complexity?
                    agn_redshift, 
                    skymap_cl, 
                    agn_redshift_err, 
                    assume_perfect_redshift, 
                    s_agn_z_integral_ax, 
                    s_alt_z_integral_ax,
                    gw_zcut,
                    merger_rate_func, 
                    linax,
                    realdata,
                    **merger_rate_kwargs):

    # agn_redshift_err is only needed as argument for this assertion, could remove it.
    if not assume_perfect_redshift:
        maxdiff = np.max(np.diff(s_agn_z_integral_ax))
        thresh = np.min(agn_redshift_err) #/ 10
        assert maxdiff < thresh, f'LOS zprior resolution is too coarse to capture AGN distribution fully: Got {maxdiff} need {thresh}'

    if linax:
        dz_Sagn = np.diff(s_agn_z_integral_ax)[0]
        dz_Salt = np.diff(s_alt_z_integral_ax)[0]
        jacobian_sagn = 1
        jacobian_salt = 1
    else:
        dz_Sagn = np.diff(np.log10(s_agn_z_integral_ax))[0]
        dz_Salt = np.diff(np.log10(s_alt_z_integral_ax))[0]
        jacobian_sagn = s_agn_z_integral_ax * np.log(10)
        jacobian_salt = s_alt_z_integral_ax * np.log(10)

    cmap_nside = hp.npix2nside(len(completeness_map))

    if assume_perfect_redshift or realdata:
        def redshift_completeness(z):
            bin_idx = np.digitize(z, completeness_zedges) - 1
            bin_idx[bin_idx == len(completeness_zvals)] = len(completeness_zvals) - 1
            return completeness_zvals[bin_idx.astype(np.int32)]
    else:
        redshift_completeness = interp1d(s_agn_z_integral_ax, completeness_zvals)

    sky_map = np.flipud(np.sort(sky_map, order="PROBDENSITY"))
    
    # Unpacking skymap
    dP_dA = sky_map["PROBDENSITY"]  # Probdens in 1/sr
    mu = sky_map["DISTMU"]          # Ansatz mean in Mpc
    sigma = sky_map["DISTSIGMA"]    # Ansatz width in Mpc
    norm = sky_map["DISTNORM"]      # Ansatz norm in 1/Mpc

    if np.sum(np.isnan(dP_dA)) > 0:
        print('BAD SKYMAP')
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Find the pixel that contains the injection.
    order, ipix = moc.uniq2nest(sky_map["UNIQ"])
    max_order = np.max(order)
    max_nside = ah.level_to_nside(max_order)
    max_ipix = ipix << np.int64(2 * (max_order - order))

    agn_theta = 0.5 * np.pi - agn_dec
    agn_phi = agn_ra
    agn_pix = hp.ang2pix(max_nside, agn_theta, agn_phi, nest=True)
    i = np.argsort(max_ipix)
    gw_pixidx_at_agn_locs = i[np.digitize(agn_pix, max_ipix[i]) - 1]  # Indeces that indicate skymap pixels that contain an AGN

    dA = moc.uniq2pixarea(sky_map["UNIQ"])  # Pixel areas in sr
    dP = dP_dA * dA  # Dimensionless probability density in each pixel
    cumprob = np.cumsum(dP)
    cumprob[cumprob > 1] = 1.  # Correcting floating point error which could cause issues when skymap_cl == 1
    searched_prob_at_agn_locs = cumprob[gw_pixidx_at_agn_locs]

    # Getting only relevant AGN and pixels from the skymap
    agn_within_cl_mask = (searched_prob_at_agn_locs <= skymap_cl)
    nagn_within_cl = np.sum(agn_within_cl_mask)
    
    if nagn_within_cl == 0:
        # print(f'No AGN found within {skymap_cl} CL')
        S_agn_cw = 0.
        S_agn_cw_binned = 0.

    else:
        gw_pixidx_at_agn_locs_within_cl = gw_pixidx_at_agn_locs[agn_within_cl_mask]

        unique_gw_pixidx_containing_agn = np.unique(gw_pixidx_at_agn_locs_within_cl)  # We only need to consider the GW pixels with catalog support
        # print(f'Found {nagn_within_cl} AGN within {skymap_cl} CL in {len(unique_gw_pixidx_containing_agn)} pixels')

        distnorm_allpix, distmu_allpix, distsigma_allpix = norm[unique_gw_pixidx_containing_agn], mu[unique_gw_pixidx_containing_agn], sigma[unique_gw_pixidx_containing_agn]

        if assume_perfect_redshift:  # Delta-function AGN posteriors make the calculations easier
            agn_redshifts_within_cl = agn_redshift[agn_within_cl_mask]
            agn_lumdists_within_cl = agn_lumdist[agn_within_cl_mask]
            agn_posterior_idx = np.arange(nagn_within_cl)
            
            S_agn_cw = 0
            for i, gw_idx in enumerate(unique_gw_pixidx_containing_agn):
                norm_in_pix, mu_in_pix, sig_in_pix = distnorm_allpix[i], distmu_allpix[i], distsigma_allpix[i]
                gw_redshift_posterior_in_pix = lambda z: redshift_pdf_given_lumdist_pdf(z, LOS_lumdist_ansatz, distnorm=norm_in_pix, distmu=mu_in_pix, distsigma=sig_in_pix)

                agn_posterior_idx_in_pix = agn_posterior_idx[gw_pixidx_at_agn_locs_within_cl == gw_idx]
                agn_redshift_posteriors_in_pix = agn_redshifts_within_cl[agn_posterior_idx_in_pix]
                agn_lumdist_posteriors_in_pix = agn_lumdists_within_cl[agn_posterior_idx_in_pix]

                # Only evaluate the GW posterior for AGN within 5sigma. If outside this range, the probability will be floored to 0. The AGN is still counted in the normalization.
                lumdist_5sig = (agn_lumdist_posteriors_in_pix > (mu_in_pix - 5 * sig_in_pix)) & (agn_lumdist_posteriors_in_pix < (mu_in_pix + 5 * sig_in_pix))
                below_zcut = agn_redshift_posteriors_in_pix < gw_zcut
                selec = lumdist_5sig & below_zcut
                if np.sum(selec) == 0:  # The contribution to S_agn is ~0 in this pixel, so skip
                    continue

                selected_agn_redshifts = agn_redshift_posteriors_in_pix[selec]
                
                # TODO: redshift completeness could be different for different AGN
                # f_c(z) * p(s|z) * p_gw(z) * p_gw(Omega) / pi_PE(z), evaluated at AGN position because of delta-function AGN posteriors, sum contributions of all AGN in this pixel
                S_agn_cw += np.sum( redshift_completeness(selected_agn_redshifts) * merger_rate(selected_agn_redshifts, merger_rate_func, **merger_rate_kwargs) * gw_redshift_posterior_in_pix(selected_agn_redshifts) * dP_dA[gw_idx] / uniform_comoving_prior(selected_agn_redshifts) )
                # S_agn_cw += np.sum( merger_rate(selected_agn_redshifts, merger_rate_func, **merger_rate_kwargs) * gw_redshift_posterior_in_pix(selected_agn_redshifts) * dP_dA[gw_idx] / uniform_comoving_prior(selected_agn_redshifts) )
            
            
            # S_agn_cw_binned = S_agn_cw


        else:  # AGN have z-errors, need to use their full posteriors
            PEprior = uniform_comoving_prior(s_agn_z_integral_ax)
            fc_of_z = redshift_completeness(s_agn_z_integral_ax)  # TODO: make different for each AGN
            zcut = z_cut(s_agn_z_integral_ax, zcut=gw_zcut)
            zrate = merger_rate(s_agn_z_integral_ax, merger_rate_func, **merger_rate_kwargs)

            # Vectorized evaluation of the GW posteriors for all unique relevant pixels - need sufficient RAM to comfortably handle arrays of (npix with agn)*len(z-array) elements
            gw_redshift_posterior_in_allpix = redshift_pdf_given_lumdist_pdf(s_agn_z_integral_ax[:,np.newaxis], LOS_lumdist_ansatz, distnorm=distnorm_allpix, distmu=distmu_allpix, distsigma=distsigma_allpix)
            
            # Loading the AGN posteriors
            agn_redshift_posteriors_in_gw = agn_posterior_dset[agn_within_cl_mask,:]
            agn_posterior_idx = np.arange(nagn_within_cl)

            integrand = np.zeros_like(s_agn_z_integral_ax)
            # LOSzprior = np.zeros_like(s_agn_z_integral_ax)  # For plotting
            for i, gw_idx in enumerate(unique_gw_pixidx_containing_agn):
                gw_redshift_posterior_in_pix = gw_redshift_posterior_in_allpix[:, i].flatten()

                agn_posterior_idx_in_pix = agn_posterior_idx[gw_pixidx_at_agn_locs_within_cl == gw_idx]
                agn_redshift_posteriors_in_pix = agn_redshift_posteriors_in_gw[agn_posterior_idx_in_pix, :]
                
                # The population prior consists of AGN posteriors, modulated by redshift evolving merger rates, normalization is done outside this function (Namely, 06-10-2025: in p26_likelihood.py)
                agn_population_prior_unnorm = np.sum(agn_redshift_posteriors_in_pix * zcut * zrate, axis=0)
                # LOSzprior += agn_population_prior_unnorm
                integrand += dP_dA[gw_idx] * gw_redshift_posterior_in_pix * agn_population_prior_unnorm
            
            S_agn_cw = romb(integrand * fc_of_z / PEprior * jacobian_sagn, dx=dz_Sagn)  # TODO: put fc_of_z inside loop when it can differ over the sky
            
            # plt.figure()
            # plt.plot(s_agn_z_integral_ax, LOSzprior)
            # plt.plot(s_agn_z_integral_ax, gw_redshift_posterior_in_pix)
            # plt.plot(s_agn_z_integral_ax, integrand)
            # plt.show()                
    
    skymap_theta, skymap_phi = moc.uniq2ang(sky_map['UNIQ'])
    pix_idx = hp.ang2pix(cmap_nside, skymap_theta, skymap_phi, nest=True)

    pixprob_within_cl = (cumprob <= skymap_cl)
    # completenesses = completeness_map[ pix_idx[pixprob_within_cl] ]  # WHEN USING ONLY SKYLOC

    cmap_vals_in_gw_skymap = completeness_map[pix_idx]
    surveyed = (cmap_vals_in_gw_skymap != 0)

    sky_coverage = np.sum(dA[surveyed]) / np.sum(dA)
    S_agn_cw *= sky_coverage

    # S_alt = 1  # WHEN USING ONLY SKYLOC
    # S_alt_cw = np.sum(dP * cmap_vals_in_gw_skymap)  # WHEN USING ONLY SKYLOC

    skyprob_nonzero = (dP != 0)
    gw_redshift_posterior_marginalized = lambda z: redshift_pdf_given_lumdist_pdf(z, 
                                                                                    allsky_marginal_lumdist_distribution, 
                                                                                    dP=dP[skyprob_nonzero & pixprob_within_cl],
                                                                                    norm=norm[skyprob_nonzero & pixprob_within_cl], 
                                                                                    mu=mu[skyprob_nonzero & pixprob_within_cl], 
                                                                                    sigma=sigma[skyprob_nonzero & pixprob_within_cl])

    gw_redshift_posterior_marginalized_cw = lambda z: redshift_pdf_given_lumdist_pdf(z, 
                                                                                    allsky_marginal_lumdist_distribution, 
                                                                                    dP=dP[surveyed & skyprob_nonzero & pixprob_within_cl],
                                                                                    norm=norm[surveyed & skyprob_nonzero & pixprob_within_cl], 
                                                                                    mu=mu[surveyed & skyprob_nonzero & pixprob_within_cl], 
                                                                                    sigma=sigma[surveyed & skyprob_nonzero & pixprob_within_cl])

    # WARNING: THESE LINES HAVE TO BE RETHOUGHT ONCE THE BACKGROUND DISTRIBUTION IS NOT UNIFORM IN COMOVING VOLUME LIKE THE PARAMETER ESTIMATION PRIOR -> use the population distribution in the population prior and divide gw posterior by uniform in comvol later
    alt_redshift_population_prior_rate_weighted = merger_rate(s_alt_z_integral_ax, merger_rate_func, **merger_rate_kwargs) * z_cut(s_alt_z_integral_ax, zcut=gw_zcut)  # Uniform in comvol prior divides out against parameter estimation prior, but has to be present still in the normalization of the population prior.
    
    alt_redshift_population_prior_rate_weighted /= romb(uniform_comoving_prior(s_alt_z_integral_ax) * alt_redshift_population_prior_rate_weighted * jacobian_salt, dx=dz_Salt)

    gw_posterior_times_alt_population_prior = gw_redshift_posterior_marginalized(s_alt_z_integral_ax) * alt_redshift_population_prior_rate_weighted

    intermediate = gw_redshift_posterior_marginalized_cw(s_alt_z_integral_ax) * alt_redshift_population_prior_rate_weighted
    gw_posterior_times_cw_alt_population_prior = intermediate * redshift_completeness(s_alt_z_integral_ax)  # TODO: only works now with uniform-on-sky redshift completeness
    
    S_alt = romb(y=gw_posterior_times_alt_population_prior * jacobian_salt, dx=dz_Salt)
    S_alt_cw = romb(y=gw_posterior_times_cw_alt_population_prior * jacobian_salt, dx=dz_Salt)

    S_alt_cw = min(S_alt_cw, S_alt)  # Handle floating point error, otherwise a NaN occurs in the log-llh if S_alt_cw > S_alt when S_agn_cw = 0

    # S_alt_cw_binned = min(S_alt_cw_binned, S_alt)
    S_agn_cw_binned = 1
    S_alt_cw_binned = 1

    return S_agn_cw, S_alt_cw, S_alt, S_agn_cw_binned, S_alt_cw_binned
