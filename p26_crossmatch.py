import astropy_healpix as ah
import healpy as hp
import numpy as np
from ligo.skymap import distance, moc
import time
from astropy.constants import c
from scipy.integrate import romb
from numba import njit, prange
import h5py
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from default_arguments import DEFAULT_COSMOLOGY as COSMO
import sys
from redshift_utils import z_cut, merger_rate, uniform_comoving_prior, fast_z_at_value
from utils import gaussian
import matplotlib.pyplot as plt
from tqdm import tqdm


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
                    merger_rate_func, 
                    linax,
                    **merger_rate_kwargs):

    if not assume_perfect_redshift:
        maxdiff = np.max(np.diff(s_agn_z_integral_ax))
        thresh = np.min(agn_redshift_err) #/ 10
        assert maxdiff < thresh, f'LOS zprior resolution is too coarse to capture AGN distribution fully: Got {maxdiff} need {thresh}'

    if linax:
        dz_Sagn = np.diff(s_agn_z_integral_ax)[0]
        dz_Salt = np.diff(s_alt_z_integral_ax)[0]
    else:
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

    print(f'PERFECT REDSHIFT = {assume_perfect_redshift}')

    sky_map = np.flipud(np.sort(sky_map, order="PROBDENSITY"))
    
    # Unpacking skymap
    dP_dA = sky_map["PROBDENSITY"]  # Probdens in 1/sr
    mu = sky_map["DISTMU"]          # Ansatz mean in Mpc
    sigma = sky_map["DISTSIGMA"]    # Ansatz width in Mpc
    norm = sky_map["DISTNORM"]      # Ansatz norm in 1/Mpc

    if np.sum(np.isnan(dP_dA)) == len(dP_dA):
        print('BAD SKYMAP')
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
    gw_pixidx_at_agn_locs = i[np.digitize(true_pix, max_ipix[i]) - 1]  # Indeces that indicate skymap pixels that contain an AGN

    dA = moc.uniq2pixarea(sky_map["UNIQ"])  # Pixel areas in sr
    dP = dP_dA * dA  # Dimensionless probability density in each pixel
    # print('VERIFICATION:', np.sum(dP))
    cumprob = np.cumsum(dP)
    cumprob[cumprob > 1] = 1.  # Correcting floating point error which could cause issues when skymap_cl == 1
    searched_prob_at_agn_locs = cumprob[gw_pixidx_at_agn_locs]

    # Getting only relevant AGN and pixels from the skymap
    agn_within_cl_mask = (searched_prob_at_agn_locs <= skymap_cl)
    nagn_within_cl = np.sum(agn_within_cl_mask)
    
    if nagn_within_cl == 0:
        print(f'No AGN found within {skymap_cl} CL')
        S_agn_cw = 0.

    else:
        print('Calculating S_agn...')
        gw_pixidx_at_agn_locs_within_cl = gw_pixidx_at_agn_locs[agn_within_cl_mask]

        unique_gw_pixidx_containing_agn, counts = np.unique(gw_pixidx_at_agn_locs_within_cl, return_counts=True)  # We only need to consider the GW pixels with catalog support
        print(f'Found {nagn_within_cl} AGN within {skymap_cl} CL in {len(unique_gw_pixidx_containing_agn)} pixels')

        if assume_perfect_redshift:  # Delta-function AGN posteriors make the calculations easier
            print('PERFECT REDSHIFT NOT YET IMPLEMENTED WITH VARYING MERGER RATE')  # TODO
            print('!!!!!!!!!!! REMOVE MULTIPROCESSING PLEASE + ONLY CONSIDER AGN WITHIN CL AS DONE IN THE FULL ZPRIOR CASE !!!!!!!!!!!')

            # for _, has_agn in zip(counts, unique_gw_idx):
            #     # print(f'Pixel: {has_agn} contains {count} AGN')
            #     gw_redshift_posterior_in_pix = lambda z: redshift_pdf_given_lumdist_pdf(z, LOS_lumdist_ansatz, distnorm=norm[has_agn], distmu=mu[has_agn], distsigma=sigma[has_agn])

            #     agn_in_pix = (gw_pixidx_at_agn_locs == has_agn)

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

                agn_in_pix = (gw_pixidx_at_agn_locs == has_agn)

                # Only evaluate the GW posterior for AGN within 5sigma. If outside this range, the probability will be floored to 0. The AGN is still counted in the normalization.
                lumdist_5sig = (agn_lumdist[agn_in_pix] > (mu[has_agn] - 5 * sigma[has_agn])) & (agn_lumdist[agn_in_pix] < (mu[has_agn] + 5 * sigma[has_agn]))
                below_zcut = agn_redshift[agn_in_pix] < gw_zcut
                if np.sum(lumdist_5sig & below_zcut) == 0:  # The contribution to S_agn is 0 in this pixel, so skip
                    return 0

                selected_agn_redshifts = agn_redshift[agn_in_pix][lumdist_5sig & below_zcut]

                gw_posterior_at_agn_redshift = gw_redshift_posterior_in_pix(selected_agn_redshifts) #* merger_rate(selected_agn_redshifts, merger_rate_func, **merger_rate_kwargs)
                # TODO: redshift completeness could be different for different AGN
                contribution = np.sum( redshift_completeness(selected_agn_redshifts) * gw_posterior_at_agn_redshift * dP_dA[has_agn] / uniform_comoving_prior(selected_agn_redshifts) )  # f_c(z) * p_gw(z) * p_gw(Omega) / pi_PE(z), evaluated at AGN position because of delta-function AGN posteriors, sum contributions of all AGN in this pixel

                return contribution
            
            tasks = list(zip(unique_gw_pixidx_containing_agn, counts))
            with ThreadPoolExecutor() as executor:
                contributions = list(executor.map(process_one, tasks))

            S_agn_cw = np.sum(contributions)
            print('Total Sagn time:', time.time() - mp_start)
            
        else:  # AGN have z-errors, need to use their full posteriors

            sagn_start = time.time()

            PEprior = uniform_comoving_prior(s_agn_z_integral_ax)
            fc_of_z = redshift_completeness(s_agn_z_integral_ax)  # TODO: make different for each AGN - changes the agn_population_prior
            zcut = z_cut(s_agn_z_integral_ax, zcut=gw_zcut)
            zrate = merger_rate(s_agn_z_integral_ax, merger_rate_func, **merger_rate_kwargs)

            # Vectorized evaluation of the GW posteriors for all unique relevant pixels - need sufficient RAM to comfortably handle arrays of (npix with agn)*len(zarray) elements
            distnorm_allpix, distmu_allpix, distsigma_allpix = norm[unique_gw_pixidx_containing_agn], mu[unique_gw_pixidx_containing_agn], sigma[unique_gw_pixidx_containing_agn]
            gw_redshift_posterior_in_allpix = redshift_pdf_given_lumdist_pdf(s_agn_z_integral_ax[:,np.newaxis], LOS_lumdist_ansatz, distnorm=distnorm_allpix, distmu=distmu_allpix, distsigma=distsigma_allpix)
            
            # Loading the AGN posteriors
            idx_array = np.arange(len(agn_within_cl_mask))[agn_within_cl_mask]
            with h5py.File(posterior_path, "r") as f:
                dset = f["agn_redshift_posteriors"]
                agn_redshift_posteriors = dset[idx_array, :]
            agn_posterior_idx = np.arange(nagn_within_cl)

            integrand = np.zeros_like(s_agn_z_integral_ax)
            for i, gw_idx in enumerate(unique_gw_pixidx_containing_agn):
                gw_redshift_posterior_in_pix = gw_redshift_posterior_in_allpix[:, i].flatten()
                agn_posterior_idx_in_pix = agn_posterior_idx[gw_pixidx_at_agn_locs_within_cl == gw_idx]  # slow?
                agn_redshift_posteriors_in_pix = agn_redshift_posteriors[agn_posterior_idx_in_pix, :]
                
                agn_population_prior_unnorm = np.sum(agn_redshift_posteriors_in_pix * zcut * zrate, axis=0)  # The population prior consists of AGN posteriors, modulated by redshift evolving merger rates, normalization is done outside this function (Namely, 06-10-2025: in p26_likelihood.py)
                integrand += dP_dA[gw_idx] * gw_redshift_posterior_in_pix * agn_population_prior_unnorm * fc_of_z / PEprior
            if linax:
                S_agn_cw = romb(integrand, dx=dz_Sagn)
            else:
                S_agn_cw = romb(integrand * s_agn_z_integral_ax * np.log(10), dx=dz_Sagn)
            print('Total Sagn time:', time.time() - sagn_start)
    
    skymap_theta, skymap_phi = moc.uniq2ang(sky_map['UNIQ'])
    pix_idx = hp.ang2pix(cmap_nside, skymap_theta, skymap_phi, nest=True)

    # pixprob_within_cl = (cumprob <= skymap_cl)  # WHEN USING ONLY SKYLOC
    # completenesses = completeness_map[ pix_idx[pixprob_within_cl] ]  # WHEN USING ONLY SKYLOC

    cmap_vals_in_gw_skymap = completeness_map[pix_idx]
    surveyed = (cmap_vals_in_gw_skymap != 0)

    sky_coverage = np.sum(dA[surveyed]) / np.sum(dA)
    S_agn_cw *= sky_coverage

    print('Calculating S_alt...')
    salttimer = time.time()
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
    
    if linax:
        alt_redshift_population_prior_rate_weighted /= romb(uniform_comoving_prior(s_alt_z_integral_ax) * alt_redshift_population_prior_rate_weighted, dx=dz_Salt)
    else:
        alt_redshift_population_prior_rate_weighted /= romb(uniform_comoving_prior(s_alt_z_integral_ax) * alt_redshift_population_prior_rate_weighted * s_alt_z_integral_ax * np.log(10), dx=dz_Salt)

    gw_posterior_times_alt_population_prior = gw_redshift_posterior_marginalized(s_alt_z_integral_ax) * alt_redshift_population_prior_rate_weighted
    gw_posterior_times_cw_alt_population_prior = gw_redshift_posterior_marginalized_cw(s_alt_z_integral_ax) * alt_redshift_population_prior_rate_weighted * redshift_completeness(s_alt_z_integral_ax)  # TODO: only works now with uniform-on-sky redshift completeness
    
    if linax:
        S_alt = romb(y=gw_posterior_times_alt_population_prior, dx=dz_Salt)
        S_alt_cw = romb(y=gw_posterior_times_cw_alt_population_prior, dx=dz_Salt)
    else:
        S_alt = romb(y=gw_posterior_times_alt_population_prior * s_alt_z_integral_ax * np.log(10), dx=dz_Salt)
        S_alt_cw = romb(y=gw_posterior_times_cw_alt_population_prior * s_alt_z_integral_ax * np.log(10), dx=dz_Salt)

    S_alt_cw = min(S_alt_cw, S_alt)  # Handle floating point error, otherwise a NaN occurs in the log-llh if S_alt_cw > S_alt when S_agn_cw = 0
    print(time.time() - salttimer, 'SALT TOTAL')

    return S_agn_cw, S_alt_cw, S_alt
