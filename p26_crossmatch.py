import astropy_healpix as ah
import healpy as hp
import numpy as np
from ligo.skymap import moc
from scipy.integrate import romb
# from default_globals import *
from redshift_utils import *
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# from tqdm import tqdm
import time
# import sys


TESTING = False


def allsky_marginal_lumdist_distribution(dl_array, dP, norm, mu, sigma):
    dl_array = np.atleast_1d(dl_array)             # shape (M,)

    # Broadcast: (M,1) vs (N,)
    dl = dl_array[:, None]                         # (M,1)
    mu = mu[None, :]                               # (1,N)
    sigma = sigma[None, :]                         # (1,N)
    norm = norm[None, :]
    dP = dP[None, :]

    gauss = np.exp(-0.5 * ((dl - mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))
    result = np.sum(dP * norm * (dl**2) * gauss, axis=1)  # sum over N
    return result


def LOS_lumdist_ansatz(dl, distnorm, distmu, distsigma):
    'The ansatz is normalized on dL [0, large]'    
    return dl**2 * distnorm * np.exp(-0.5 * ((dl - distmu) / distsigma)**2) / (np.sqrt(2 * np.pi) * distsigma)  #gaussian(dl, distmu, distsigma)


def crossmatch_p26(
                    agn_posterior_dset, 
                    sky_map,
                    completeness_map,
                    redshift_completeness,
                    agn_ra, 
                    agn_dec, 
                    agn_lumdist,  # TODO: remove this argument: agn_lumdist is only needed for speed up of s_agn_incat when assume_perfect_redshift = True - is it worth the added complexity?
                    agn_redshift, 
                    skymap_cl, 
                    agn_redshift_err,  # TODO: remove this argument: see comment above assertion (16-03-2026)
                    assume_perfect_redshift, 
                    z_integral_ax,
                    gw_zcut,
                    merger_rate_func, 
                    linax,
                    correct_time_dilation,
                    background_agn_distribution,
                    **merger_rate_kwargs):

    # agn_redshift_err is only needed as argument for this assertion, could remove it. -- 16-03-2026: z_integral_ax is now made automatically to enforce this, so it should never trigger. Can be removed.
    # if not assume_perfect_redshift:
    #     maxdiff = np.max(np.diff(z_integral_ax))
    #     thresh = np.min(agn_redshift_err) #/ 10
    #     assert maxdiff < thresh, f'LOS zprior resolution is too coarse to capture AGN distribution fully: Got {maxdiff} need {thresh}'

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

    # print(np.sum(dP[surveyed & skyprob_nonzero & pixprob_within_cl]), 'amount of prob in surveyed part')
    # plt.figure()
    # plt.plot(z_integral_ax, gw_redshift_posterior_marginalized(z_integral_ax))
    # plt.plot(z_integral_ax, gw_redshift_posterior_marginalized_cw(z_integral_ax))
    # plt.show()
    
    ####################### Population priors and integrals #######################

    # Prepare functions that are used multiple times
    PEprior = uniform_comoving_prior(z_integral_ax)  # 1/4pi sky position PEprior is explicitly added whenever it does not cancel
    # PEprior = redshift_pdf_given_lumdist_pdf(z_integral_ax, lumdist_pdf=lambda dl: dl**2)
    fc_of_z = redshift_completeness(z_integral_ax)
    zcut = z_cut(z_integral_ax, zcut=gw_zcut)
    zrate = merger_rate(z_integral_ax, merger_rate_func, **merger_rate_kwargs)
    if correct_time_dilation:
        time_dilation = time_dilation_correction(z_integral_ax)
    else: 
        time_dilation = np.ones_like(z_integral_ax)
    p_rate_of_z_alt = time_dilation * zrate * zcut  # Only the alternative hypothesis evolves with SFR!
    p_rate_of_z_agn = time_dilation * zcut
    
    normed_agn_background_dist = background_agn_distribution(z_integral_ax) / romb(background_agn_distribution(z_integral_ax) * jacobian, dx=dz)  # 1/4pi cancels with sky position PEprior -- 24-03-2026: Can be put outside loop. Also, I don't think this normalization is necessary, maybe remove later.

    if TESTING:
        gw_redshift_posterior_marginalized = lambda z: PEprior.copy()
        gw_redshift_posterior_marginalized_cw = lambda z: PEprior.copy() * np.sum(dA[surveyed]) / np.sum(dA)

    gw_redshift_posterior_marginalized_evaluated = gw_redshift_posterior_marginalized(z_integral_ax)

    ### Alternative-origin population part ###
    
    # int dz p(z|d_gw)/PEprior(z) * p_pop(z | \conj{A}, \conj{G}): unif. in com.vol.
    background_alt_distribution = uniform_comoving_prior(z_integral_ax)
    alt_redshift_population_prior = background_alt_distribution * p_rate_of_z_alt
    alt_redshift_population_prior /= romb(alt_redshift_population_prior * jacobian, dx=dz)
    S_alt = romb(y=gw_redshift_posterior_marginalized_evaluated / PEprior * alt_redshift_population_prior * jacobian, dx=dz)

    ### AGN-origin population part ###

    average_redshift_completeness = romb(redshift_completeness(z_integral_ax) * normed_agn_background_dist * jacobian, dx=dz)  # -- 24-03-2026: Can be put outside loop.
    sky_coverage = np.sum(dA[surveyed]) / np.sum(dA)
    average_completeness = average_redshift_completeness * sky_coverage

    ### -- 24-03-2026: Can be put outside loop. ###
    if assume_perfect_redshift:
        nagn_norm = np.sum(agn_redshift < gw_zcut)
    else:
        sum_of_all_agn_posteriors = np.sum(agn_posterior_dset, axis=0)
        nagn_norm = romb(sum_of_all_agn_posteriors, dx=dz)

    # Get zprior normalizations, dealing with delta-function AGN posteriors (then assume_perfect_redshift == True) and empty catalogues (then total_n_agn == 0)
    if assume_perfect_redshift:
        if nagn_norm == 0:
            agn_population_prior_normalization = romb((1 - fc_of_z) * normed_agn_background_dist * p_rate_of_z_agn * jacobian, dx=dz)
            
        else:
            # It would be less messy to define this above, oh well.
            if correct_time_dilation:
                p_rate_of_z_agn_func = lambda z: time_dilation_correction(z) * z_cut(z, zcut=gw_zcut)
            else: 
                p_rate_of_z_agn_func = lambda z: np.ones_like(z) * z_cut(z, zcut=gw_zcut)
            
            agn_population_prior_normalization = average_redshift_completeness * np.sum(p_rate_of_z_agn_func(agn_redshift)) / nagn_norm + romb((1 - fc_of_z) * normed_agn_background_dist * p_rate_of_z_agn * jacobian, dx=dz)
    else:
        if nagn_norm == 0:
            agn_population_prior_normalization = romb((1 - fc_of_z) * normed_agn_background_dist * p_rate_of_z_agn * jacobian, dx=dz)
        else:
            # p_rate_of_z_agn imposes a redshift cut in the GW population, up to which the pop. is normalized. Therefore agn_population_prior only has to be evaluated at redshifts up to this cut.
            agn_population_prior = average_redshift_completeness * sum_of_all_agn_posteriors / nagn_norm + (1 - fc_of_z) * normed_agn_background_dist 
            agn_population_prior_rate_weighted = agn_population_prior * p_rate_of_z_agn
            agn_population_prior_normalization = romb(agn_population_prior_rate_weighted * jacobian, dx=dz)
    ##################################################

    # Out-of-catalogue part
    S_agn_outofcat = romb(y=(gw_redshift_posterior_marginalized_evaluated - fc_of_z * gw_redshift_posterior_marginalized_cw(z_integral_ax)) / PEprior * p_rate_of_z_agn * normed_agn_background_dist * jacobian, dx=dz) / agn_population_prior_normalization

    # a = romb(y=gw_redshift_posterior_marginalized_evaluated / PEprior * p_rate_of_z_agn * normed_agn_background_dist * jacobian, dx=dz) / agn_population_prior_normalization
    # b = romb(fc_of_z * gw_redshift_posterior_marginalized_cw(z_integral_ax) / PEprior * p_rate_of_z_agn * normed_agn_background_dist * jacobian, dx=dz) / agn_population_prior_normalization
    # print('should be a:', romb(y=p_rate_of_z_agn * normed_agn_background_dist, dx=dz) / agn_population_prior_normalization)
    # print('should be b:', romb(y=fc_of_z * sky_coverage * p_rate_of_z_agn * normed_agn_background_dist, dx=dz) / agn_population_prior_normalization)
    # print(S_agn_outofcat, a - b, 'different?')
    # print(S_agn_outofcat / S_alt)

    # In-catalogue part
    if nagn_within_cl == 0:
        S_agn_incat = 0
        return S_agn_incat, S_agn_outofcat, S_alt

    gw_pixidx_at_agn_locs_within_cl = gw_pixidx_at_agn_locs[agn_within_cl_mask]
    unique_gw_pixidx_containing_agn = np.unique(gw_pixidx_at_agn_locs_within_cl)  # We only need to consider the GW pixels with catalog support
    distnorm_allpix, distmu_allpix, distsigma_allpix = norm[unique_gw_pixidx_containing_agn], mu[unique_gw_pixidx_containing_agn], sigma[unique_gw_pixidx_containing_agn]
    # print(f'Found {nagn_within_cl} AGN within {skymap_cl} CL in {len(unique_gw_pixidx_containing_agn)} pixels')

    if assume_perfect_redshift:  # Delta-function AGN posteriors make the calculations easier

        PEprior_func = lambda z: uniform_comoving_prior(z)
        agn_redshifts_within_cl = agn_redshift[agn_within_cl_mask]
        # agn_lumdists_within_cl = agn_lumdist[agn_within_cl_mask]
        agn_posterior_idx = np.arange(nagn_within_cl)

        S_agn_incat = 0
        for i, gw_idx in enumerate(unique_gw_pixidx_containing_agn):
            norm_in_pix, mu_in_pix, sig_in_pix = distnorm_allpix[i], distmu_allpix[i], distsigma_allpix[i]
            gw_redshift_posterior_in_pix = lambda z: redshift_pdf_given_lumdist_pdf(z, LOS_lumdist_ansatz, distnorm=norm_in_pix, distmu=mu_in_pix, distsigma=sig_in_pix)
            agn_posterior_idx_in_pix = agn_posterior_idx[gw_pixidx_at_agn_locs_within_cl == gw_idx]
            agn_redshift_posteriors_in_pix = agn_redshifts_within_cl[agn_posterior_idx_in_pix]

            # agn_lumdist_posteriors_in_pix = agn_lumdists_within_cl[agn_posterior_idx_in_pix]
        #     # Only evaluate the GW posterior for AGN within 5sigma. If outside this range, the probability will be floored to 0. The AGN is still counted in the normalization.
        #     lumdist_5sig = (agn_lumdist_posteriors_in_pix > (mu_in_pix - 5 * sig_in_pix)) & (agn_lumdist_posteriors_in_pix < (mu_in_pix + 5 * sig_in_pix))
        #     below_zcut = agn_redshift_posteriors_in_pix < gw_zcut
        #     selec = lumdist_5sig & below_zcut
        #     if np.sum(selec) == 0:  # The contribution to S_agn is ~0 in this pixel, so skip
        #         continue
            selected_agn_redshifts = agn_redshift_posteriors_in_pix  #[selec]

            if TESTING:
                gw_redshift_posterior_in_pix = PEprior_func
                dP_dA[gw_idx] = 1 / (4 * np.pi)
            
            # p(s|z) * p_gw(z) * p_gw(Omega) / pi_PE(z), evaluated at AGN position because of delta-function AGN posteriors, sum contributions of all AGN in this pixel
            S_agn_incat += dP_dA[gw_idx] * np.sum( p_rate_of_z_agn_func(selected_agn_redshifts) * gw_redshift_posterior_in_pix(selected_agn_redshifts) / PEprior_func(selected_agn_redshifts) )

        S_agn_incat *= 4 * np.pi * average_completeness / nagn_norm / agn_population_prior_normalization


    else:  # AGN have z-errors, need to use their full posteriors
        # Vectorized evaluation of the GW posteriors for all unique relevant pixels - requires sufficient RAM to comfortably handle arrays of (npix with agn)*len(z-array) elements
        gw_redshift_posterior_in_allpix = redshift_pdf_given_lumdist_pdf(z_integral_ax[:,np.newaxis], LOS_lumdist_ansatz, distnorm=distnorm_allpix, distmu=distmu_allpix, distsigma=distsigma_allpix)
        
        # Loading the AGN posteriors
        agn_redshift_posteriors_in_cl = agn_posterior_dset[agn_within_cl_mask,:]
        agn_posterior_idx = np.arange(nagn_within_cl)

        # Building p_pop(z|A,G) * p_GW(z|d)
        integrand = np.zeros_like(z_integral_ax)  # AGN posteriors weighted by GW sky posterior, to be integrated over redshift
        LOSzprior = np.zeros_like(z_integral_ax)  # Needed for normalization of population prior
        for i, gw_idx in enumerate(unique_gw_pixidx_containing_agn):
            gw_redshift_posterior_in_pix = gw_redshift_posterior_in_allpix[:, i].flatten()

            if TESTING:
                gw_redshift_posterior_in_pix = PEprior.copy()
                dP_dA[gw_idx] = 1 / (4 * np.pi)

            agn_posterior_idx_in_pix = agn_posterior_idx[gw_pixidx_at_agn_locs_within_cl == gw_idx]
            agn_redshift_posteriors_in_pix = agn_redshift_posteriors_in_cl[agn_posterior_idx_in_pix, :]
            
            # The population prior consists of AGN posteriors, modulated by redshift evolving merger rates (done later)
            sum_of_agn_posteriors = np.sum(agn_redshift_posteriors_in_pix, axis=0)
            LOSzprior += sum_of_agn_posteriors #* dP_dA[gw_idx]
            integrand += dP_dA[gw_idx] * gw_redshift_posterior_in_pix * sum_of_agn_posteriors
        
        # Normalize
        integrand /= nagn_norm
        LOSzprior /= nagn_norm

        # Calculate evidence
        S_agn_incat = romb(integrand * p_rate_of_z_agn / PEprior * jacobian, dx=dz) * 4 * np.pi * average_completeness / agn_population_prior_normalization  # 1/4pi from PEprior does not cancel, since the AGN sky posterior is delta(Omega_i - Omega)

        # plt.figure()
        # plt.plot(z_integral_ax, normed_agn_background_dist * p_rate_of_z_agn / romb(normed_agn_background_dist * p_rate_of_z_agn, dx=dz), label='outcat prior')
        # plt.plot(z_integral_ax, np.sum(agn_posterior_dset, axis=0) * p_rate_of_z_agn / romb(np.sum(agn_posterior_dset, axis=0) * p_rate_of_z_agn, dx=dz), label='incat prior')
        # plt.plot(z_integral_ax, agn_population_prior_rate_weighted / agn_population_prior_normalization, label='combined')
        # plt.legend()
        # plt.show()
    return S_agn_incat, S_agn_outofcat, S_alt


# plt.figure()
# plt.plot(z_integral_ax, LOSzprior)
# plt.plot(z_integral_ax, gw_redshift_posterior_in_pix)
# plt.plot(z_integral_ax, integrand)
# plt.show()

# a = average_redshift_completeness * (np.sum(agn_posterior_dset, axis=0) / total_n_agn) * p_rate_of_z_agn
# b = (1 - fc_of_z) * normed_agn_background_dist * p_rate_of_z_agn
# print(agn_population_prior_normalization, romb(a, dx=dz), romb(b, dx=dz))

# plotthis = (average_redshift_completeness * LOSzprior / romb(LOSzprior, dx=dz) + (1 - fc_of_z) * normed_agn_background_dist) * p_rate_of_z_agn

# print(romb(LOSzprior * p_rate_of_z_agn * average_completeness / agn_population_prior_normalization + (1 - fc_of_z) * p_rate_of_z_agn * normed_agn_background_dist / agn_population_prior_normalization, dx=dz))

# plt.figure()
# plt.plot(z_integral_ax, normed_agn_background_dist * p_rate_of_z_agn / romb(normed_agn_background_dist * p_rate_of_z_agn, dx=dz), label='outcat prior')
# plt.plot(z_integral_ax, np.sum(agn_posterior_dset, axis=0) * p_rate_of_z_agn / romb(np.sum(agn_posterior_dset, axis=0) * p_rate_of_z_agn, dx=dz), label='incat prior')
# plt.plot(z_integral_ax, agn_population_prior_rate_weighted / agn_population_prior_normalization, label='combined')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(z_integral_ax, agn_population_prior_rate_weighted / agn_population_prior_normalization / (normed_agn_background_dist * p_rate_of_z_agn / romb(normed_agn_background_dist * p_rate_of_z_agn, dx=dz)))
# plt.show()

# plt.figure()
# plt.plot(z_integral_ax, plotthis / romb(plotthis, dx=dz) / (normed_agn_background_dist * p_rate_of_z_agn / romb(normed_agn_background_dist * p_rate_of_z_agn, dx=dz)))
# plt.show()


'''
BELOW IS NOT THOROUGHLY TESTED
'''


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


def PEprior_m1d_m2d_dl(dl, m1d, m2d, run):
    r'''
    Detector frame PE prior, unnormalized.
    Currently mass PEpriors are only uniform in detector-frame mass.

    From the GWTC papers, I gather that the luminosity distance prior is \propto dl^2 for the nocosmo samples (before O4) and \propto uniform rate in source frame for O4a
    '''

    if run == None:
        return np.ones_like(dl)
    elif run == 'O4':
        return uniform_source_frame_dl(dl)
    else: 
        return dl**2


def get_weights(redshift, mass_1_source, mass_2_source, PEpriors_detframe, source_frame_mass_prior):
    PEpriors_source_frame = PEpriors_detframe * det2source_jacobian(redshift)  # This is pPE(m1d,m2d,dL) (1+z)^2 |ddL/dz|
    if source_frame_mass_prior == None:
        weights = 1 / PEpriors_source_frame
    else:
        weights = source_frame_mass_prior.joint_prob(mass_1_source, mass_2_source) / PEpriors_source_frame
    return weights




def ignore_weights(weights):

    weights = np.ones(len(weights))
    norm = 0
    return weights, norm

def check_weights(weights):
    """
    Check the weights values to prevent gaussian_kde crash when Neff <= 1,
    where Neff is an internal variable of gaussian_kde
    defined by Neff = sum(weights)^2/sum(weights^2)
    careful, cases with Neff = 1+2e-16 = 1.0000000000000002
    have been seen and give crash: set Neff limit to >= 2
    """
    neff = 0
    if np.isclose(max(weights),0,atol=1e-50):
        weights, norm = ignore_weights(weights)
    else:
        neff = sum(weights)**2/sum(weights**2)
        if neff<2:
            weights, norm = ignore_weights(weights)
        else:
            norm = np.sum(weights)/len(weights)
    return weights, norm, neff

def get_kde(data, weights):
    # deal first with the weights
    weights, norm, neff = check_weights(weights)
    status = True
    if norm != 0:
        try:
            kde = gaussian_kde(data, weights=weights)
        except Exception as e:
            print("Exception:",e)
            anomalies = np.where((weights < 0) | np.isinf(weights) | np.isnan(weights) ) [0]
            print(f"KDE problem! {len(anomalies)} abnormal (negative or inf or NaN) values for the weights (total number: {len(weights)}. Create a default KDE with norm=0.")
            print("norm: {} -> 0, neff: {}".format(norm,neff))
            status = False
            norm = 0
            kde = gaussian_kde(data)
    else:
        kde = gaussian_kde(data)

    return kde, norm, status



def crossmatch_from_samples_p26(posterior_samples,
                                z_integral_ax,
                                agn_posterior_dset,
                                agn_ra, 
                                agn_dec,
                                completeness_map,
                                redshift_completeness,
                                gw_zcut,
                                merger_rate_func,
                                correct_time_dilation,
                                background_agn_distribution,
                                linax,
                                source_frame_mass_prior=None,
                                run=None,
                                minpix=30,
                                skymap_cl=0.999,
                                minsamps=100,
                                **merger_rate_kwargs):


    # # agn_redshift_err is only needed as argument for this assertion, could remove it.
    # if not assume_perfect_redshift:
    #     maxdiff = np.max(np.diff(z_integral_ax))
    #     thresh = np.min(agn_redshift_err) #/ 10
    #     assert maxdiff < thresh, f'LOS zprior resolution is too coarse to capture AGN distribution fully: Got {maxdiff} need {thresh}'
    

    if linax:
        dz = np.diff(z_integral_ax)[0]
        jacobian = 1
    else:
        dz = np.diff(np.log10(z_integral_ax))[0]
        jacobian = z_integral_ax * np.log(10)

    approximant_want = ['C01:Mixed', 'C00:Mixed', 'Mock']
    approximants_available = list(posterior_samples.keys())
    for approx in approximant_want:
        if approx in approximants_available:
            approximant = approx
            break

    dec   = posterior_samples[approximant]['posterior_samples']['dec'][()]
    phi   = posterior_samples[approximant]['posterior_samples']['ra'][()]
    z     = posterior_samples[approximant]['posterior_samples']['redshift'][()]
    m1s   = posterior_samples[approximant]['posterior_samples']['mass_1_source'][()]
    m2s   = posterior_samples[approximant]['posterior_samples']['mass_2_source'][()]
    Nsamps = len(z)
    print(f'Got Nsamps={Nsamps}')

    PEpriors_detframe = PEprior_m1d_m2d_dl(dl=posterior_samples[approximant]['posterior_samples']['luminosity_distance'][()],
                                           m1d=None, 
                                           m2d=None, 
                                           run=run)
    weights = get_weights(z, m1s, m2s, PEpriors_detframe, source_frame_mass_prior)

    ### Pixelize samples ###
    nside = 2  # Minimum nside is 32 to ensure a sufficient resolution of the survey footprint
    npix_in_CL = 0
    while npix_in_CL <= minpix:
        nside *= 2
        dA = hp.nside2pixarea(nside)  # sr

        ipix_at_samps = hp.ang2pix(nside, 0.5 * np.pi - dec, phi, nest=True)
        counts = np.bincount(ipix_at_samps, minlength=hp.nside2npix(nside))
        dP_dA = counts / (Nsamps * dA)  # 1/sr
        dP = dP_dA * dA  # Dimensionless probability density in each pixel

        ind_sorted = np.argsort(-dP)
        cumprob = np.cumsum(dP[ind_sorted])
        cumprob[cumprob > 1] = 1.  # Correcting floating point error which could cause issues when skymap_cl == 1

        if skymap_cl == 1:
            indices_in_CL = ind_sorted
        else:
            lim_ind = np.where(cumprob >= skymap_cl)[0][0]
            indices_in_CL = ind_sorted[:lim_ind]

        npix_in_CL = len(indices_in_CL)
    print(f'Ended with {npix_in_CL} pixels using nside = {nside}.')


    ### Find the pixels that contain AGN ###
    agn_theta = 0.5 * np.pi - agn_dec
    agn_phi = agn_ra
    ipix_at_agn = hp.ang2pix(nside, agn_theta, agn_phi, nest=True)
    mask_agn_in_cl = np.isin(ipix_at_agn, indices_in_CL)  # Selects the AGN within the specified CL localization area
    ipix_at_agn_in_cl = ipix_at_agn[mask_agn_in_cl]
    nagn_in_cl = np.sum(mask_agn_in_cl)
    unique_ipix_with_agn_in_cl = np.unique(ipix_at_agn_in_cl)


    ### Get the sky pixels within the EM survey footprint and calculate the GW posteriors ###
    cmap_at_gw_res = hp.pixelfunc.ud_grade(completeness_map, nside, order_in='NESTED', order_out='NESTED')
    cmap_at_gw_res = np.around(cmap_at_gw_res)  # FIXME: The GW samples in the footprint and the total number of AGN (used to normalize the LOS zprior) may be slightly off due to the jagged edges of this map, but this is expected to be negligible.
    surveyed = (cmap_at_gw_res != 0)

    cl_mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    cl_mask[indices_in_CL] = True
    cl_surveyed_mask = cl_mask & surveyed

    # Calculate the marginalized GW redshift posterior, both within the CL and within the CL+survey footprint (the latter for the term int dOmega p_GW(z, Omega | d) * p(G|z, Omega))
    # TODO: reweight samples by PEprior and mass population !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # s = time.time()
    sel = cl_mask[ipix_at_samps]
    # cl_gw_posterior = gaussian_kde(z[sel], weights=weights[sel])
    cl_gw_posterior, norm, status = get_kde(z[sel], weights=weights[sel])
    eval_cl_gw_posterior = cl_gw_posterior(z_integral_ax)
    # print('within CL: t=', time.time() - s)

    print(norm)

    # s = time.time()
    sel = cl_surveyed_mask[ipix_at_samps]
    # surveyed_cl_gw_posterior = gaussian_kde(z[sel], weights=weights[sel])
    surveyed_cl_gw_posterior, norm, status = get_kde(z[sel], weights=weights[sel])
    eval_surveyed_cl_gw_posterior = surveyed_cl_gw_posterior(z_integral_ax)
    # print('within CL and footprint: t=', time.time() - s)


    ### Prepare some functions for later ###
    PEprior = uniform_comoving_prior(z_integral_ax)
    fc_of_z = redshift_completeness(z_integral_ax)  # TODO: make different for each AGN depending on LOS
    zcut = z_cut(z_integral_ax, zcut=gw_zcut)
    zrate = merger_rate(z_integral_ax, merger_rate_func, **merger_rate_kwargs)
    if correct_time_dilation:
        time_dilation = time_dilation_correction(z_integral_ax)
    else: 
        time_dilation = np.ones_like(z_integral_ax)
    p_rate_of_z_alt = time_dilation * zrate * zcut  # Only the alternative hypothesis evolves with SFR!
    p_rate_of_z_agn = time_dilation * zcut  # TODO: add a merger_rate function different for the AGN-origin GW population
    normed_agn_background_dist = background_agn_distribution(z_integral_ax) / romb(background_agn_distribution(z_integral_ax) * jacobian, dx=dz)  # Only used to model GW population, so it's ok to normalize on this axis (+ it is necessary for e.g. unif-in-comvol prior)


    ### ALT-origin population part ###
    background_alt_distribution = uniform_comoving_prior(z_integral_ax)
    alt_redshift_population_prior = background_alt_distribution * p_rate_of_z_alt
    alt_redshift_population_prior /= romb(alt_redshift_population_prior * jacobian, dx=dz)
    S_alt = romb(y=eval_cl_gw_posterior / PEprior * alt_redshift_population_prior * jacobian, dx=dz)  # int dz p(z|d_gw)/PEprior(z) * p_pop(z | \conj{A}, \conj{G})


    ### AGN-origin population part ###
    total_n_agn = len(agn_ra)
    agn_redshift_posteriors_in_cl = agn_posterior_dset[mask_agn_in_cl,:]  # Load the AGN posteriors
    agn_posterior_idx = np.arange(nagn_in_cl)
    # t = 0
    LOSzprior = np.zeros_like(z_integral_ax)
    integrand = np.zeros_like(z_integral_ax)
    px_zOmegaparam = np.zeros((len(unique_ipix_with_agn_in_cl), len(z_integral_ax)))
    for i, ipix in enumerate(unique_ipix_with_agn_in_cl):
        s = time.time()

        # 1. Obtain GW redshift posterior along the LOS
        samp_mask = identify_samples(ipix, ra=phi, dec=dec, nside=nside, nested=True, minsamps=minsamps)  # Search in annuli for samples if the number of samples in the pixel is < minsamps
        z_samps_in_pix = z[samp_mask]

        ### TODO: reweight samples by PEprior and mass population ###

        # Want to evaluate the KDE on as few points as possible to save computation time
        zmin_temp = np.min(z_samps_in_pix) * 0.5
        zmax_temp = np.max(z_samps_in_pix) * 2.
        z_array_temp = np.linspace(zmin_temp, zmax_temp, 100)  
        # los_gw_posterior = gaussian_kde(z_samps_in_pix, weights=weights[samp_mask])  # If gaussian_kde ever crashes, gwcosmo has some fix that I don't understand and don't want to implement immediately
        los_gw_posterior, norm, status = get_kde(z_samps_in_pix, weights=weights[samp_mask])

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
        sum_of_agn_posteriors = np.sum(agn_redshift_posteriors_in_pix, axis=0)
        integrand += dP_dA[ipix] * px_zOmegaparam[i, :] * sum_of_agn_posteriors
        LOSzprior += sum_of_agn_posteriors * dP_dA[ipix]

    #     t +=  time.time() - s
    # print('total time:', t)
    integrand /= total_n_agn  # Normalize
    
    # plt.figure()
    # plt.plot(z_integral_ax, LOSzprior)
    # plt.show()
    # plt.figure()
    # plt.plot(z_integral_ax, integrand)
    ## plt.plot(z_integral_ax, eval_cl_gw_posterior)
    # plt.show()


    # 3. Combine with the rate and completeness, and integrate
    average_redshift_completeness = romb(redshift_completeness(z_integral_ax) * normed_agn_background_dist * jacobian, dx=dz)
    sky_coverage = np.sum(surveyed) / hp.nside2npix(nside)
    average_completeness = average_redshift_completeness * sky_coverage

    agn_population_prior = average_redshift_completeness * (np.sum(agn_posterior_dset, axis=0) / total_n_agn) + (1 - fc_of_z) * normed_agn_background_dist
    agn_population_prior_rate_weighted = agn_population_prior * p_rate_of_z_agn
    agn_population_prior_normalization = romb(agn_population_prior_rate_weighted * jacobian, dx=dz)

    # Calculate evidence
    S_agn_incat = romb(integrand * p_rate_of_z_agn / PEprior * jacobian, dx=dz) * 4 * np.pi * average_completeness / agn_population_prior_normalization  # 4pi from PEprior does not cancel
    S_agn_outofcat = romb(y=(eval_cl_gw_posterior - fc_of_z * eval_surveyed_cl_gw_posterior) * normed_agn_background_dist * p_rate_of_z_agn / PEprior * jacobian, dx=dz) / agn_population_prior_normalization

    return S_agn_incat, S_agn_outofcat, S_alt