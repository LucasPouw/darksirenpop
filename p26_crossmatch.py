import astropy_healpix as ah
import healpy as hp
import numpy as np
from ligo.skymap import moc
from scipy.integrate import romb
from default_globals import *
from redshift_utils import z_cut, merger_rate, uniform_comoving_prior, redshift_pdf_given_lumdist_pdf, time_dilation_correction
from utils import gaussian
from numba import njit, prange
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# from tqdm import tqdm
import time
# import sys

"""
# TODO: redshift completeness could be different for different sky positions outside galactic plane
"""


# INV_SQRT2PI = 1.0 / np.sqrt(2.0 * np.pi)

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
    return dl**2 * distnorm * np.exp(-0.5 * ((dl - distmu) / distsigma)**2) / (np.sqrt(2 * np.pi) * distsigma)  #gaussian(dl, distmu, distsigma)


def angular_sep(ra1,dec1,ra2,dec2):
    """Find the angular separation between two points, (ra1,dec1)
    and (ra2,dec2), in radians."""

    cos_angle = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)
    angle = np.arccos(cos_angle)
    return angle


def ra_dec_from_ipix(nside, ipix, nest=False):
    """RA and dec from HEALPix index"""
    (theta, phi) = hp.pix2ang(nside, ipix, nest=nest)
    return (phi, np.pi/2.-theta)


def identify_samples(idx, ra, dec, nside, nested, minsamps=100):
        """
        Find the samples required

        Parameters
        ----------
        idx : int
            The pixel index
        minsamps : int, optional
            The threshold number of samples to reach per pixel

        Return
        ------
        sel : array of ints
            The indices of posterior samples for pixel idx
        """

        ipix_samples = hp.pixelfunc.ang2pix(nside, np.pi/2 - dec, ra, nest=nested)
        sel = np.where(ipix_samples == idx)[0]
        if len(sel) >= minsamps:
            # print("{} samples fall in pix {}".format(len(sel),idx))
            return sel

        # not enough samples in pixel 'idx', we need to extend the search
        racent, deccent = ra_dec_from_ipix(nside, idx, nest=nested)
        separations = angular_sep(racent, deccent, ra, dec)
        sep = hp.pixelfunc.max_pixrad(nside)/2. # choose initial separation
        step = sep/2. # choose step size for increasing radius

        sel = np.where(separations < sep)[0] # find all the samples within the angular radius sep from the pixel centre
        nsamps = len(sel)
        while nsamps < minsamps:
            sep += step
            sel = np.where(separations < sep)[0]
            nsamps = len(sel)
            if sep > np.pi:
                raise ValueError("Problem with the number of posterior samples.")
        # print('pixel idx {}: angular radius: {} radians, No. samples: {}'.format(idx, sep, len(sel)))

        return sel


def crossmatch_from_samples_p26(posterior_samples,
                                z_integral_ax,
                                agn_posterior_dset,
                                agn_ra, 
                                agn_dec, 
                                minpix=30,
                                skymap_cl=0.999,
                                minsamps=100):

    dec = posterior_samples['mock']['posterior_samples']['dec'][()]
    theta = 0.5 * np.pi - dec  # colatitude [0, pi]
    phi   = posterior_samples['mock']['posterior_samples']['ra'][()]  # longitude [0, 2pi]
    z     = posterior_samples['mock']['posterior_samples']['redshift'][()]
    Nsamps = len(theta)

    ### Pixelize samples ###
    nside = 8
    npix_in_CL = 0
    while npix_in_CL <= minpix:
        nside *= 2
        dA = hp.nside2pixarea(nside)  # sr

        ipix_at_samps = hp.ang2pix(nside, theta, phi, nest=True)
        counts = np.bincount(ipix_at_samps, minlength=hp.nside2npix(nside))
        dP_dA = counts / (Nsamps * dA)  # 1/sr
        dP = dP_dA * dA  # Dimensionless probability density in each pixel

        ind_sorted = np.argsort(-dP)
        cumprob = np.cumsum(dP[ind_sorted])
        cumprob[cumprob > 1] = 1.  # Correcting floating point error which could cause issues when skymap_cl == 1
        lim_ind = np.where(cumprob > skymap_cl)[0][0]

        indices_in_CL = ind_sorted[:lim_ind]
        npix_in_CL = len(indices_in_CL)

        # print(npix_in_CL, 'minpix =', minpix)
        # hp.mollview(dP, nest=True, unit='probability per pixel', )
        # plt.show()
    print(f'Ended with {npix_in_CL} pixels using nside = {nside}.')

    ### Find the pixels that contain AGN ###
    agn_theta = 0.5 * np.pi - agn_dec
    agn_phi = agn_ra
    ipix_at_agn = hp.ang2pix(nside, agn_theta, agn_phi, nest=True)
    mask_agn_in_cl = np.isin(ipix_at_agn, indices_in_CL)  # Selects the AGN within the specified CL localization area
    ipix_at_agn_in_cl = ipix_at_agn[mask_agn_in_cl]
    nagn_in_cl = np.sum(mask_agn_in_cl)
    unique_ipix_with_agn_in_cl = np.unique(ipix_at_agn_in_cl)
    print(f'Found {nagn_in_cl} AGN in {len(unique_ipix_with_agn_in_cl)} pixels within the {skymap_cl * 100}% CL. Given uniform-on-sky AGN distribution, expected {dA * npix_in_CL / (4 * np.pi) * len(agn_ra)}')

    ### AGN-origin population part ###
    total_n_agn = len(agn_ra)
    agn_redshift_posteriors_in_cl = agn_posterior_dset[mask_agn_in_cl,:]  # Load the AGN posteriors
    agn_posterior_idx = np.arange(nagn_in_cl)
    t = 0
    LOSzprior = np.zeros_like(z_integral_ax)
    integrand = np.zeros_like(z_integral_ax)
    px_zOmegaparam = np.zeros((len(unique_ipix_with_agn_in_cl), len(z_integral_ax)))
    for i, ipix in enumerate(unique_ipix_with_agn_in_cl):
        s = time.time()

        # 1. Obtain GW redshift posterior along the LOS
        skyprob = dP[ipix]
        # print(skyprob, 'skyprob')

        # samp_mask = (ipix_at_samps == ipix)
        samp_mask = identify_samples(ipix, ra=phi, dec=dec, nside=nside, nested=True, minsamps=minsamps)  # Search in annuli for samples if the number of samples in the pixel is < minsamps
        z_samps_in_pix = z[samp_mask]

        ### TODO: reweight samples by PEprior and mass population ###

        # Want to evaluate the KDE on as few points as possible to save computation time
        zmin_temp = np.min(z_samps_in_pix)*0.5
        zmax_temp = np.max(z_samps_in_pix)*2.
        z_array_temp = np.linspace(zmin_temp, zmax_temp, 100)  
        los_gw_posterior = gaussian_kde(z_samps_in_pix)  # If gaussian_kde ever crashes, gwcosmo has some fix that I don't understand and don't want to implement immediately

        # Enforce the evaluation of the posterior on the same redshift axis
        los_gw_posterior_interp = interp1d(z_array_temp, los_gw_posterior(z_array_temp), kind='cubic')
        z_mask = (zmin_temp < z_integral_ax) & (z_integral_ax < zmax_temp)
        px_zOmegaparam[i, z_mask] = los_gw_posterior_interp(z_integral_ax[z_mask])

        # if len(z_samps_in_pix) < 200:
        #     plt.figure()
        #     plt.plot(z_array_temp, eval_los_gw_posterior)
        #     plt.plot(z_integral_ax[mask], final_los_posterior)
        #     plt.show()

        # 2. Obtain the LOS redshift prior
        agn_posterior_idx_in_pix = agn_posterior_idx[ipix_at_agn_in_cl == ipix]  # Selects all AGN within the current pixel
        agn_redshift_posteriors_in_pix = agn_redshift_posteriors_in_cl[agn_posterior_idx_in_pix,:]

        # The population prior consists of AGN posteriors, modulated by redshift evolving merger rates (done later)
        sum_of_agn_posteriors = np.sum(agn_redshift_posteriors_in_pix, axis=0)
        integrand += dP_dA[ipix] * px_zOmegaparam[i, :] * sum_of_agn_posteriors

        # plt.plot(np.arange(len(LOSzprior)), sum_of_agn_posteriors)
        # plt.show()
        LOSzprior += sum_of_agn_posteriors * dP_dA[ipix]
        t +=  time.time() - s
    print('total time:', t)
        
    # Normalize
    integrand /= total_n_agn
    
    plt.figure()
    plt.plot(z_integral_ax, LOSzprior)
    plt.show()

    plt.figure()
    plt.plot(z_integral_ax, integrand)
    plt.show()


    # 3. Combine with the rate and integrate



        

    s = time.time()
    allsky_gw_posterior = gaussian_kde(z)
    eval_allsky_gw_posterior = allsky_gw_posterior(z_integral_ax)
    print('allsky t=', time.time() - s)

        # print(np.sum(mask))
        # if np.sum(mask) < 10:
        #     plt.figure()
        #     plt.hist(samps_in_pix, density=True, histtype='step')
        #     plt.plot(z_integral_ax, eval_los_gw_posterior)
        #     plt.xlim(0, 2)
        #     plt.show()
    

    sys.exit(1)
    return


def crossmatch_p26(
                    agn_posterior_dset, 
                    sky_map,
                    completeness_map,
                    redshift_completeness,
                    agn_ra, 
                    agn_dec, 
                    agn_lumdist,  # agn_lumdist is only needed for speed up of s_agn_incat when assume_perfect_redshift = True - is it worth the added complexity?
                    agn_redshift, 
                    skymap_cl, 
                    agn_redshift_err, 
                    assume_perfect_redshift, 
                    z_integral_ax,
                    gw_zcut,
                    merger_rate_func, 
                    linax,
                    correct_time_dilation,
                    background_agn_distribution,
                    **merger_rate_kwargs):

    # agn_redshift_err is only needed as argument for this assertion, could remove it.
    if not assume_perfect_redshift:
        maxdiff = np.max(np.diff(z_integral_ax))
        thresh = np.min(agn_redshift_err) #/ 10
        assert maxdiff < thresh, f'LOS zprior resolution is too coarse to capture AGN distribution fully: Got {maxdiff} need {thresh}'

    if linax:
        dz = np.diff(z_integral_ax)[0]
        jacobian = 1
    else:
        dz = np.diff(np.log10(z_integral_ax))[0]
        jacobian = z_integral_ax * np.log(10)

    sky_map = np.flipud(np.sort(sky_map, order="PROBDENSITY"))
    
    # Unpacking skymap
    dP_dA = sky_map["PROBDENSITY"]  # Probdens in 1/sr
    mu = sky_map["DISTMU"]          # Ansatz mean in Mpc
    sigma = sky_map["DISTSIGMA"]    # Ansatz width in Mpc
    norm = sky_map["DISTNORM"]      # Ansatz norm in 1/Mpc^2
    norm[np.isinf(norm)] = 0        # Infs are observed to happen rarely in low-probability sky regions, this line avoids nans later if using CL->1

    if np.sum(np.isnan(dP_dA)) > 0:
        print('BAD SKYMAP')
        return np.nan, np.nan, np.nan

    # Find the pixels that contain AGN
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

    # Get GW redshift posterior marginalized over the whole sky and sky-completeness-weighted (for now that is just 1 or 0 depending on surveyed sky region)
    skymap_theta, skymap_phi = moc.uniq2ang(sky_map['UNIQ'])
    cmap_nside = hp.npix2nside(len(completeness_map))
    pix_idx = hp.ang2pix(cmap_nside, skymap_theta, skymap_phi, nest=True)
    pixprob_within_cl = (cumprob <= skymap_cl)
    cmap_vals_in_gw_skymap = completeness_map[pix_idx]
    surveyed = (cmap_vals_in_gw_skymap != 0)
    skyprob_nonzero = (dP != 0)

    gw_redshift_posterior_marginalized = lambda z: redshift_pdf_given_lumdist_pdf(z, 
                                                                                    allsky_marginal_lumdist_distribution, 
                                                                                    dP=dP[skyprob_nonzero & pixprob_within_cl],
                                                                                    norm=norm[skyprob_nonzero & pixprob_within_cl], 
                                                                                    mu=mu[skyprob_nonzero & pixprob_within_cl], 
                                                                                    sigma=sigma[skyprob_nonzero & pixprob_within_cl])
    
    # Marginalize the GW posterior over sky position, weighting with sky completeness (currently only 1 for surveyed and 0 for not surveyed): int dOmega p_GW(z, Omega | d) * p(G|z, Omega)
    gw_redshift_posterior_marginalized_cw = lambda z: redshift_pdf_given_lumdist_pdf(z, 
                                                                                    allsky_marginal_lumdist_distribution, 
                                                                                    dP=dP[surveyed & skyprob_nonzero & pixprob_within_cl],
                                                                                    norm=norm[surveyed & skyprob_nonzero & pixprob_within_cl], 
                                                                                    mu=mu[surveyed & skyprob_nonzero & pixprob_within_cl], 
                                                                                    sigma=sigma[surveyed & skyprob_nonzero & pixprob_within_cl])

    
    
    ####################### Population priors and integrals #######################

    # Prepare functions that are used multiple times
    PEprior = uniform_comoving_prior(z_integral_ax)  # 1/4pi sky position PEprior is explicitly added whenever it does not cancel
    # PEprior = redshift_pdf_given_lumdist_pdf(z_integral_ax, lumdist_pdf=lambda dl: dl**2)
    fc_of_z = redshift_completeness(z_integral_ax)  # TODO: make different for each AGN depending on LOS
    zcut = z_cut(z_integral_ax, zcut=gw_zcut)
    zrate = merger_rate(z_integral_ax, merger_rate_func, **merger_rate_kwargs)
    if correct_time_dilation:
        time_dilation = time_dilation_correction(z_integral_ax)
    else: 
        time_dilation = np.ones_like(z_integral_ax)
    p_rate_of_z_alt = time_dilation * zrate * zcut  # Only the alternative hypothesis evolves with SFR!
    p_rate_of_z_agn = time_dilation * zcut  # TODO: add a merger_rate function different for the AGN-origin GW population
    normed_agn_background_dist = background_agn_distribution(z_integral_ax) / romb(background_agn_distribution(z_integral_ax) * jacobian, dx=dz)





    # gw_redshift_posterior_marginalized = lambda z: PEprior.copy()
    # gw_redshift_posterior_marginalized_cw = lambda z: PEprior.copy() * np.sum(dA[surveyed]) / np.sum(dA)




    gw_redshift_posterior_marginalized_evaluated = gw_redshift_posterior_marginalized(z_integral_ax)

    ### Alternative-origin population part ###
    
    # int dz p(z|d_gw)/PEprior(z) * p_pop(z | \conj{A}, \conj{G}): unif. in com.vol.
    background_alt_distribution = uniform_comoving_prior(z_integral_ax)
    alt_redshift_population_prior = background_alt_distribution * p_rate_of_z_alt
    alt_redshift_population_prior /= romb(alt_redshift_population_prior * jacobian, dx=dz)
    S_alt = romb(y=gw_redshift_posterior_marginalized_evaluated / PEprior * alt_redshift_population_prior * jacobian, dx=dz)

    ### AGN-origin population part ###

    total_n_agn = agn_posterior_dset.shape[0]
    
    if (nagn_within_cl == 0) or (np.sum(fc_of_z == 0) == len(np.atleast_1d(fc_of_z))):  # TODO: replace this by actually putting in an empty catalog to avoid precalculating AGN posteriors that are not used
        # print(f'No AGN found within {skymap_cl} CL')
        # S_agn_incat = 0.
        LOSzprior = np.zeros_like(z_integral_ax)  # For plotting
        integrand = np.zeros_like(z_integral_ax)

    else:
        gw_pixidx_at_agn_locs_within_cl = gw_pixidx_at_agn_locs[agn_within_cl_mask]

        unique_gw_pixidx_containing_agn = np.unique(gw_pixidx_at_agn_locs_within_cl)  # We only need to consider the GW pixels with catalog support
        # print(f'Found {nagn_within_cl} AGN within {skymap_cl} CL in {len(unique_gw_pixidx_containing_agn)} pixels')

        distnorm_allpix, distmu_allpix, distsigma_allpix = norm[unique_gw_pixidx_containing_agn], mu[unique_gw_pixidx_containing_agn], sigma[unique_gw_pixidx_containing_agn]

        if assume_perfect_redshift:  # Delta-function AGN posteriors make the calculations easier
            sys.exit('Not up to date')
            # agn_redshifts_within_cl = agn_redshift[agn_within_cl_mask]
            # agn_lumdists_within_cl = agn_lumdist[agn_within_cl_mask]
            # agn_posterior_idx = np.arange(nagn_within_cl)
            
            # S_agn_incat = 0
            # for i, gw_idx in enumerate(unique_gw_pixidx_containing_agn):
            #     norm_in_pix, mu_in_pix, sig_in_pix = distnorm_allpix[i], distmu_allpix[i], distsigma_allpix[i]
            #     gw_redshift_posterior_in_pix = lambda z: redshift_pdf_given_lumdist_pdf(z, LOS_lumdist_ansatz, distnorm=norm_in_pix, distmu=mu_in_pix, distsigma=sig_in_pix)

            #     agn_posterior_idx_in_pix = agn_posterior_idx[gw_pixidx_at_agn_locs_within_cl == gw_idx]
            #     agn_redshift_posteriors_in_pix = agn_redshifts_within_cl[agn_posterior_idx_in_pix]
            #     agn_lumdist_posteriors_in_pix = agn_lumdists_within_cl[agn_posterior_idx_in_pix]

            #     # Only evaluate the GW posterior for AGN within 5sigma. If outside this range, the probability will be floored to 0. The AGN is still counted in the normalization.
            #     lumdist_5sig = (agn_lumdist_posteriors_in_pix > (mu_in_pix - 5 * sig_in_pix)) & (agn_lumdist_posteriors_in_pix < (mu_in_pix + 5 * sig_in_pix))
            #     below_zcut = agn_redshift_posteriors_in_pix < gw_zcut
            #     selec = lumdist_5sig & below_zcut
            #     if np.sum(selec) == 0:  # The contribution to S_agn is ~0 in this pixel, so skip
            #         continue

            #     selected_agn_redshifts = agn_redshift_posteriors_in_pix[selec]
                
            #     # f_c(z) * p(s|z) * p_gw(z) * p_gw(Omega) / pi_PE(z), evaluated at AGN position because of delta-function AGN posteriors, sum contributions of all AGN in this pixel
            #     if correct_time_dilation:
            #         time_dilation_at_agn_redshift = time_dilation_correction(selected_agn_redshifts)
            #     else: 
            #         time_dilation_at_agn_redshift = np.ones_like(selected_agn_redshifts)

            #     S_agn_incat += np.sum( redshift_completeness(selected_agn_redshifts) * time_dilation_at_agn_redshift * gw_redshift_posterior_in_pix(selected_agn_redshifts) * dP_dA[gw_idx] / uniform_comoving_prior(selected_agn_redshifts) )

        else:  # AGN have z-errors, need to use their full posteriors

            # Vectorized evaluation of the GW posteriors for all unique relevant pixels - requires sufficient RAM to comfortably handle arrays of (npix with agn)*len(z-array) elements
            gw_redshift_posterior_in_allpix = redshift_pdf_given_lumdist_pdf(z_integral_ax[:,np.newaxis], LOS_lumdist_ansatz, distnorm=distnorm_allpix, distmu=distmu_allpix, distsigma=distsigma_allpix)
            
            # Loading the AGN posteriors
            agn_redshift_posteriors_in_gw = agn_posterior_dset[agn_within_cl_mask,:]
            agn_posterior_idx = np.arange(nagn_within_cl)

            # Building p_pop(z|A,G) * p_GW(z|d)
            integrand = np.zeros_like(z_integral_ax)  # AGN posteriors weighted by GW sky posterior, to be integrated over redshift
            # LOSzprior = np.zeros_like(z_integral_ax)  # Needed for normalization of population prior
            for i, gw_idx in enumerate(unique_gw_pixidx_containing_agn):
                gw_redshift_posterior_in_pix = gw_redshift_posterior_in_allpix[:, i].flatten()




                # gw_redshift_posterior_in_pix = PEprior.copy()
                # dP_dA[gw_idx] = 1 / (4 * np.pi)





                agn_posterior_idx_in_pix = agn_posterior_idx[gw_pixidx_at_agn_locs_within_cl == gw_idx]
                agn_redshift_posteriors_in_pix = agn_redshift_posteriors_in_gw[agn_posterior_idx_in_pix, :]
                
                # The population prior consists of AGN posteriors, modulated by redshift evolving merger rates (done later)
                sum_of_agn_posteriors = np.sum(agn_redshift_posteriors_in_pix, axis=0)
                # LOSzprior += sum_of_agn_posteriors
                integrand += dP_dA[gw_idx] * gw_redshift_posterior_in_pix * sum_of_agn_posteriors  # TODO: dP_dA or dP?
            # Normalize
            integrand /= total_n_agn
            # LOSzprior /= total_n_agn

            # S_agn_incat = 4 * np.pi * romb(integrand * time_dilation * zcut / PEprior * jacobian, dx=dz)
            
            # plt.figure()
            # plt.plot(z_integral_ax, LOSzprior)
            # plt.plot(z_integral_ax, gw_redshift_posterior_in_pix)
            # plt.plot(z_integral_ax, integrand)
            # plt.show()
    
    average_redshift_completeness = romb(redshift_completeness(z_integral_ax) * normed_agn_background_dist * jacobian, dx=dz)
    sky_coverage = np.sum(dA[surveyed]) / np.sum(dA)
    average_completeness = average_redshift_completeness * sky_coverage

    agn_population_prior = average_redshift_completeness * (np.sum(agn_posterior_dset, axis=0) / total_n_agn) + (1 - fc_of_z) * normed_agn_background_dist
    agn_population_prior_rate_weighted = agn_population_prior * p_rate_of_z_agn
    agn_population_prior_normalization = romb(agn_population_prior_rate_weighted * jacobian, dx=dz)
    # print(agn_population_prior_normalization, 'add this norm')

    # Calculate evidence
    S_agn_incat = romb(integrand * p_rate_of_z_agn / PEprior * jacobian, dx=dz) * 4 * np.pi * average_completeness / agn_population_prior_normalization  # 4pi from PEprior does not cancel
    S_agn_outofcat = romb(y=(gw_redshift_posterior_marginalized_evaluated - fc_of_z * gw_redshift_posterior_marginalized_cw(z_integral_ax)) / PEprior * p_rate_of_z_agn * normed_agn_background_dist * jacobian, dx=dz) / agn_population_prior_normalization

    # print(S_agn_incat + S_agn_outofcat, 'combined')

    return S_agn_incat, S_agn_outofcat, S_alt



    # (1 - f_c(z)) * p_pop(z | A, \conj{G}), integral comes later
    

    # S_agn_outofcat = romb(y=(gw_redshift_posterior_marginalized_evaluated - fc_of_z * gw_redshift_posterior_marginalized_cw(z_integral_ax)) / PEprior * outofcat_agn_redshift_population_prior * jacobian, dx=dz)





    # normed_agn_background_dist = background_agn_distribution(z_integral_ax) / romb(background_agn_distribution(z_integral_ax), dx=dz)

    # average_completeness = romb(redshift_completeness(z_integral_ax) * normed_agn_background_dist, dx=dz)

    # c_incat_old =  fc_of_z * LOSzprior / romb(LOSzprior, dx=dz)

    # print(romb(LOSzprior * time_dilation * zcut, dx=dz))

    # c_incat = average_completeness * LOSzprior / romb(LOSzprior, dx=dz)
    # oneminusc_outcat = (1 - fc_of_z) * normed_agn_background_dist

    # print(romb(c_incat + oneminusc_outcat, dx=dz), 'new')
    # print(romb(c_incat_old + oneminusc_outcat, dx=dz), 'old')

    # plt.figure()
    # plt.plot(z_integral_ax, gw_redshift_posterior_marginalized_evaluated / 10, label='GW', color='black')

    # plt.plot(z_integral_ax, alt_redshift_population_prior, label='ALT', color='red')
    # plt.plot(z_integral_ax, normed_agn_background_dist, label='piAGN', color='steelblue')

    # plt.plot(z_integral_ax, (c_incat + oneminusc_outcat) / romb(c_incat + oneminusc_outcat, dx=dz), label='AGN', color='blue')
    # plt.plot(z_integral_ax, (c_incat_old + oneminusc_outcat) / romb(c_incat_old + oneminusc_outcat, dx=dz), label='AGN old', color='blue', linestyle='dotted')


    # # plt.plot(z_integral_ax, c_incat_old, label='c*incat old')
    # # plt.plot(z_integral_ax, c_incat, label='c*incat')
    # # plt.plot(z_integral_ax, oneminusc_outcat, label='(1-c)*outcat')

    # # plt.plot(z_integral_ax, LOSzprior / romb(LOSzprior * time_dilation * zcut, dx=dz), label='LOSzprior')
    
    # plt.plot(z_integral_ax, fc_of_z, label='c(z)')
    # plt.legend()
    # plt.show()
            
    #########################################################################