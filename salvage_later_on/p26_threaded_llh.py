import numpy as np
import sys
from tqdm import tqdm
from astropy.coordinates import SkyCoord
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import h5py
from default_globals import *
import astropy.units as u
import os
from ligo.skymap.io.fits import read_sky_map
import matplotlib.pyplot as plt
from scipy.integrate import simpson, romb
from scipy import stats
import healpy as hp
import time
from redshift_utils import z_cut, merger_rate_madau, merger_rate_uniform, merger_rate, uniform_comoving_prior, fast_z_at_value
from utils import uniform_shell_sampler
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import astropy_healpix as ah
from ligo.skymap import moc
from redshift_utils import redshift_pdf_given_lumdist_pdf
from utils import gaussian


INV_SQRT2PI = 1.0 / np.sqrt(2.0 * np.pi)

os.environ["OMP_NUM_THREADS"] = "1"

Z_EDGES = np.array([0.0000, 0.1875, 0.3750, 0.5625, 0.7500, 0.9375, 1.1250, 1.3125, 1.5000])

# Quaia completeness (rows = bins, cols = thresholds)
quaia_c_vals = np.array([
                    [0.000, 0.000, 0.229, 0.945, 0.718],
                    [1.000, 1.000, 1.000, 1.000, 0.781],
                    [1.000, 1.000, 1.000, 1.000, 0.408],
                    [1.000, 0.891, 1.000, 0.681, 0.211],
                    [1.000, 1.000, 0.994, 0.429, 0.138],
                    [1.000, 1.000, 0.837, 0.258, 0.085],
                    [0.927, 0.940, 0.576, 0.179, 0.060],
                    [1.000, 1.000, 0.482, 0.155, 0.053],
                ])

threshold_map = {"46.5": 0, "46.0": 1, "45.5": 2, "45.0": 3, "44.5": 4}

FAGN_POSTERIOR_FNAME = 'p26_likelihood_posteriors'
CMAP_PATH = "./completeness_map.fits"
PLOT_CMAP = True
INDICATOR = 'MOCK'
DIRECTORY_ID = 'moc_500'
BATCH = 500
CMAP_NSIDE = 64
N_TRUE_FAGNS = 6
SKYMAP_CL = 0.999
LUM_THRESH = "test"  # '45.5' or 'test'
# COMPLETENESS = 0.7
MASK_GALACTIC_PLANE = False
ADD_NAGN_TO_CAT = int(1e4)
MERGER_RATE_EVOLUTION = merger_rate_uniform
MERGER_RATE_KWARGS = {}

ASSUME_PERFECT_REDSHIFT = False
AGN_ZERROR = 0.01

CALC_LOGLLH_AT_N_POINTS = 1000
LOG_LLH_X_AX = np.linspace(0.0001, 0.9999, CALC_LOGLLH_AT_N_POINTS)

ZMIN = 1e-4
ZMAX = 1.5  # p_rate(z > ZMAX) = 0
AGN_ZMAX = 2
assert AGN_ZMAX >= ZMAX, 'Need AGN at least as deep as GWs can go, otherwise the population prior is not evaluated on the correct axis.'

COMDIST_MIN = COSMO.comoving_distance(ZMIN).value
COMDIST_MAX = COSMO.comoving_distance(ZMAX).value
AGN_COMDIST_MAX = COSMO.comoving_distance(AGN_ZMAX).value

S_AGN_Z_INTEGRAL_AX = np.linspace(ZMIN, ZMAX, int(512)+1)  # Sets the resolution of the redshift prior, should capture all information of AGN posteriors, see Gray et al. 2022, 2023
S_ALT_Z_INTEGRAL_AX = S_AGN_Z_INTEGRAL_AX.copy()
LINAX = True  # If geomspace instead of linspace, make False

npoints = int(2**np.ceil(np.log2(10 * (AGN_ZMAX - ZMIN) / AGN_ZERROR)))  # Enforces at least 10 points within 1 sigma in normalization of AGN posteriors -> 
AGN_POSTERIOR_NORM_AX = np.linspace(ZMIN, AGN_ZMAX, npoints + 1)


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
    return dl**2 * distnorm * gaussian(dl, distmu, distsigma)


def crossmatch(sky_map,
                completeness_map,
                completeness_zedges,
                completeness_zvals,
                skymap_cl, 
                assume_perfect_redshift, 
                s_agn_z_integral_ax, 
                s_alt_z_integral_ax,
                gw_zcut,
                merger_rate_func, 
                linax,
                **merger_rate_kwargs):
    
    agn_ra = AGN_RA
    agn_dec = AGN_DEC
    agn_redshift = AGN_REDSHIFT
    agn_redshift_err = AGN_REDSHIFT_ERR
    agn_lumdist = AGN_LUMDIST
    agn_posterior_dset = AGN_POSTERIOR_DSET

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

    sky_map = np.flipud(np.sort(sky_map, order="PROBDENSITY"))
    
    # Unpacking skymap
    dP_dA = sky_map["PROBDENSITY"]  # Probdens in 1/sr
    mu = sky_map["DISTMU"]          # Ansatz mean in Mpc
    sigma = sky_map["DISTSIGMA"]    # Ansatz width in Mpc
    norm = sky_map["DISTNORM"]      # Ansatz norm in 1/Mpc

    if np.sum(np.isnan(dP_dA)) == len(dP_dA):
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
        S_agn_cw = 0.

    else:
        gw_pixidx_at_agn_locs_within_cl = gw_pixidx_at_agn_locs[agn_within_cl_mask]

        unique_gw_pixidx_containing_agn, counts = np.unique(gw_pixidx_at_agn_locs_within_cl, return_counts=True)  # We only need to consider the GW pixels with catalog support

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
            PEprior = uniform_comoving_prior(s_agn_z_integral_ax)
            fc_of_z = redshift_completeness(s_agn_z_integral_ax)  # TODO: make different for each AGN - changes the agn_population_prior
            zcut = z_cut(s_agn_z_integral_ax, zcut=gw_zcut)
            zrate = merger_rate(s_agn_z_integral_ax, merger_rate_func, **merger_rate_kwargs)

            # Vectorized evaluation of the GW posteriors for all unique relevant pixels - need sufficient RAM to comfortably handle arrays of (npix with agn)*len(zarray) elements
            distnorm_allpix, distmu_allpix, distsigma_allpix = norm[unique_gw_pixidx_containing_agn], mu[unique_gw_pixidx_containing_agn], sigma[unique_gw_pixidx_containing_agn]
            gw_redshift_posterior_in_allpix = redshift_pdf_given_lumdist_pdf(s_agn_z_integral_ax[:,np.newaxis], LOS_lumdist_ansatz, distnorm=distnorm_allpix, distmu=distmu_allpix, distsigma=distsigma_allpix)
            
            # Loading the AGN posteriors
            idx_array = np.arange(len(agn_within_cl_mask))[agn_within_cl_mask]
            agn_redshift_posteriors = agn_posterior_dset[idx_array,:]
            agn_posterior_idx = np.arange(nagn_within_cl)

            integrand = np.zeros_like(s_agn_z_integral_ax)
            for i, gw_idx in enumerate(unique_gw_pixidx_containing_agn):
                gw_redshift_posterior_in_pix = gw_redshift_posterior_in_allpix[:, i].flatten()
                agn_posterior_idx_in_pix = agn_posterior_idx[gw_pixidx_at_agn_locs_within_cl == gw_idx]  # slow?
                agn_redshift_posteriors_in_pix = agn_redshift_posteriors[agn_posterior_idx_in_pix, :]
                
                agn_population_prior_unnorm = np.sum(agn_redshift_posteriors_in_pix * zcut * zrate, axis=0)  # The population prior consists of AGN posteriors, modulated by redshift evolving merger rates, normalization is done outside this function (Namely, 06-10-2025: in p26_likelihood.py)
                integrand += dP_dA[gw_idx] * gw_redshift_posterior_in_pix * agn_population_prior_unnorm * fc_of_z / PEprior

            S_agn_cw = romb(integrand * jacobian_sagn, dx=dz_Sagn)
    
    skymap_theta, skymap_phi = moc.uniq2ang(sky_map['UNIQ'])
    pix_idx = hp.ang2pix(cmap_nside, skymap_theta, skymap_phi, nest=True)

    # pixprob_within_cl = (cumprob <= skymap_cl)  # WHEN USING ONLY SKYLOC
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
    
    alt_redshift_population_prior_rate_weighted /= romb(uniform_comoving_prior(s_alt_z_integral_ax) * alt_redshift_population_prior_rate_weighted * jacobian_salt, dx=dz_Salt)

    gw_posterior_times_alt_population_prior = gw_redshift_posterior_marginalized(s_alt_z_integral_ax) * alt_redshift_population_prior_rate_weighted
    gw_posterior_times_cw_alt_population_prior = gw_redshift_posterior_marginalized_cw(s_alt_z_integral_ax) * alt_redshift_population_prior_rate_weighted * redshift_completeness(s_alt_z_integral_ax)  # TODO: only works now with uniform-on-sky redshift completeness
    
    S_alt = romb(y=gw_posterior_times_alt_population_prior * jacobian_salt, dx=dz_Salt)
    S_alt_cw = romb(y=gw_posterior_times_cw_alt_population_prior * jacobian_salt, dx=dz_Salt)

    S_alt_cw = min(S_alt_cw, S_alt)  # Handle floating point error, otherwise a NaN occurs in the log-llh if S_alt_cw > S_alt when S_agn_cw = 0

    return S_agn_cw, S_alt_cw, S_alt


def compute_and_save_posteriors_hdf5(filename, 
                                     all_agn_z, 
                                     all_agn_z_err, 
                                     agn_posterior_norm_ax,
                                     agn_zmax):
    '''
    AGN redshift posteriors are modelled as truncnorms on [0, inf) with a uniform-in-comoving-volume redshift prior.
    The posteriors are normalized on agn_posterior_norm_ax, which goes up to AGN_ZMAX.
    The posteriors are then evaluated on S_AGN_Z_INTEGRAL_AX, which is what is necessary for the crossmatch.

    TODO: (Maybe in the future calculate each AGN posterior on its own axis from -5sig to 5sig and interpolate to S_AGN_Z_INTEGRAL_AX?)
    '''

    maxdiff = np.max(np.diff(agn_posterior_norm_ax))
    thresh = np.min(all_agn_z_err) / 10
    assert maxdiff < thresh, f'AGN normalization array is too coarse to capture AGN distribution fully. Got {maxdiff:.3e}, need {thresh:.3e}.'

    n_agn = len(all_agn_z)
    n_z = len(S_AGN_Z_INTEGRAL_AX)  # Only need to save the posterior evaluated at this axis
    chunk_size = int(1e6 / n_z)

    # Precompute dx for romb integration
    dx = np.diff(agn_posterior_norm_ax)[0]

    with h5py.File(filename, "w") as f:
        # Prepare dataset for posteriors
        dset = f.create_dataset(
            "agn_redshift_posteriors",
            shape=(n_agn, n_z),
            dtype=np.float64
        )

        for start in tqdm( range(0, n_agn, chunk_size) ):
            end = min(start + chunk_size, n_agn)

            z_chunk = all_agn_z[start:end]
            zerr_chunk = all_agn_z_err[start:end]

            likelihood = lambda z: stats.truncnorm.pdf(
                z,
                a=(ZMIN - z_chunk[:, None]) / zerr_chunk[:, None],
                b=(np.inf - z_chunk[:, None]) / zerr_chunk[:, None],
                loc=z_chunk[:, None],
                scale=zerr_chunk[:, None]
            )
            agn_posteriors_unnorm = lambda z: likelihood(z) * uniform_comoving_prior(z) * z_cut(z, zcut=agn_zmax)
            z_norms = romb(agn_posteriors_unnorm(agn_posterior_norm_ax), dx=dx)

            posteriors = agn_posteriors_unnorm(S_AGN_Z_INTEGRAL_AX) / z_norms[:, None]  # Save the posterior evaluated at the relevant axis

            # Write to HDF5
            dset[start:end, :] = posteriors

            # xx = np.linspace(ZMIN, AGN_ZMAX, int(512*32)+1)
            # znormtest = romb(agn_posteriors_unnorm(xx), dx=np.diff(xx)[0])
            # for i in range(len(znormtest)):
            #     print(znormtest[i] / z_norms[i])

    print(f"All AGN posteriors written to {filename}")


def process_single_gw(gw_idx, trial_idx, fagn_idx):
    
    filename = f"./skymaps_{DIRECTORY_ID}/skymap_{trial_idx}_{fagn_idx}_{gw_idx:05d}.fits.gz"
    skymap = read_sky_map(filename, moc=True)
    print(f'\nLoaded file: {filename}')

    sagn_cw, salt_cw, salt = crossmatch(
        sky_map=skymap,
        completeness_map=COMPLETENESS_MAP,
        completeness_zedges=Z_EDGES,
        completeness_zvals=C_PER_ZBIN,
        skymap_cl=SKYMAP_CL,
        gw_zcut=ZMAX,
        s_agn_z_integral_ax=S_AGN_Z_INTEGRAL_AX, 
        s_alt_z_integral_ax=S_ALT_Z_INTEGRAL_AX,
        assume_perfect_redshift=ASSUME_PERFECT_REDSHIFT,
        merger_rate_func=MERGER_RATE_EVOLUTION,
        linax=LINAX,
        **MERGER_RATE_KWARGS
    )

    return gw_idx, sagn_cw, salt_cw, salt


if __name__ == '__main__':
    if AGN_ZERROR == 0:
        assert ASSUME_PERFECT_REDSHIFT == True, 'Cannot have zero redshift error and not assume a perfect measurement.'

    N_TRIALS = 1
    posteriors = np.zeros((N_TRIALS, CALC_LOGLLH_AT_N_POINTS, N_TRUE_FAGNS))
    for trial_idx in range(N_TRIALS):

        log_llh = np.zeros((len(LOG_LLH_X_AX), N_TRUE_FAGNS))
        for fagn_idx in range(N_TRUE_FAGNS):

            with h5py.File(f'./catalogs_{DIRECTORY_ID}/mockcat_{trial_idx}_{fagn_idx}.hdf5', 'r') as catalog:
                
                AGN_RA = catalog['ra'][()]
                AGN_DEC = catalog['dec'][()]
                agn_rcom = catalog['comoving_distance'][()]

                ### FOR TESTING, ADD UNCORRELATED AGN TO THE CATALOG: S_AGN -> S_ALT SHOULD BE SEEN!! ###
                if ADD_NAGN_TO_CAT:
                    new_rcom, new_theta, new_phi = uniform_shell_sampler(COMDIST_MIN, AGN_COMDIST_MAX, ADD_NAGN_TO_CAT)
                    AGN_RA = np.append(AGN_RA, new_phi)
                    AGN_DEC = np.append(AGN_DEC, np.pi * 0.5 - new_theta)
                    agn_rcom = np.append(agn_rcom, new_rcom)
                #########################################################

                true_agn_redshift = fast_z_at_value(COSMO.comoving_distance, agn_rcom * u.Mpc)

                AGN_REDSHIFT_ERR = np.tile(AGN_ZERROR, len(AGN_RA))
                # df = pd.read_csv("./Quaia_z15.csv")
                # AGN_REDSHIFT_ERR = np.random.choice(df["redshift_quaia_err"], size=len(AGN_RA), replace=False)
                # print(np.min(AGN_REDSHIFT_ERR), np.max(AGN_REDSHIFT_ERR))

                if not AGN_ZERROR:
                    AGN_REDSHIFT = true_agn_redshift
                else:
                    print('Scattering AGN!')
                    AGN_REDSHIFT = stats.truncnorm.rvs(size=len(AGN_RA), 
                                                            a=(ZMIN - true_agn_redshift) / AGN_REDSHIFT_ERR, 
                                                            b=(np.inf - true_agn_redshift) / AGN_REDSHIFT_ERR, 
                                                            loc=true_agn_redshift, 
                                                            scale=AGN_REDSHIFT_ERR)  # Or a Gaussian: np.random.normal(loc=true_agn_redshift, scale=AGN_REDSHIFT_ERR, size=len(AGN_RA))
                
                AGN_LUMDIST = COSMO.luminosity_distance(AGN_REDSHIFT).value
                cat_coord = SkyCoord(AGN_RA * u.rad, AGN_DEC * u.rad, AGN_LUMDIST * u.Mpc)
                b = cat_coord.galactic.b.degree

                ### Making a redshift-incomplete catalog ###
                if LUM_THRESH == 'test':
                    C_PER_ZBIN = np.tile(1, len(Z_EDGES) - 1)
                    redshift_incomplete_mask = np.ones_like(AGN_REDSHIFT, dtype=bool)
                else:
                    C_PER_ZBIN = np.array( quaia_c_vals[:, threshold_map[LUM_THRESH]] )
                    redshift_incomplete_mask = np.zeros_like(AGN_REDSHIFT, dtype=bool)
                    for i, c_in_bin in enumerate(C_PER_ZBIN):
                        z_low = Z_EDGES[i]
                        z_high = Z_EDGES[i + 1]

                        agn_in_bin = np.where((AGN_REDSHIFT > z_low) & (AGN_REDSHIFT < z_high))[0]
                        keep_these = np.random.rand(len(agn_in_bin)) < c_in_bin  # Random realization of true completeness
                        redshift_incomplete_mask[agn_in_bin[keep_these]] = True
                print(C_PER_ZBIN, 'Z-COMPLETENESS')
                #############################################

                if MASK_GALACTIC_PLANE:
                    latitude_mask = np.logical_or(b > 10, b < -10)
                else:
                    latitude_mask = np.ones_like(b, dtype=bool)
                
                incomplete_catalog_mask = (latitude_mask & redshift_incomplete_mask)  # To emulate V25's selection of Quaia, include  & (AGN_REDSHIFT < ZMAX)
                print('KEEPING NAGN =', np.sum(incomplete_catalog_mask))

                AGN_RA = AGN_RA[incomplete_catalog_mask]
                AGN_DEC = AGN_DEC[incomplete_catalog_mask]
                AGN_REDSHIFT = AGN_REDSHIFT[incomplete_catalog_mask]
                AGN_REDSHIFT_ERR = AGN_REDSHIFT_ERR[incomplete_catalog_mask]
                AGN_LUMDIST = AGN_LUMDIST[incomplete_catalog_mask]

                npix = hp.nside2npix(CMAP_NSIDE)
                theta, phi = hp.pix2ang(CMAP_NSIDE, np.arange(npix), nest=True)
                map_coord = SkyCoord(phi * u.rad, (np.pi * 0.5 - theta) * u.rad)
                map_b = map_coord.galactic.b.degree
                outside_galactic_plane_pix = np.logical_or(map_b > 10, map_b < -10)

                cmap_values = np.tile(1, npix)
                if MASK_GALACTIC_PLANE:
                    cmap_values[~outside_galactic_plane_pix] = 0

                hp.write_map(CMAP_PATH, cmap_values, nest=True, dtype=np.float32, overwrite=True)
                COMPLETENESS_MAP = hp.read_map(CMAP_PATH, nest=True)
                
                if PLOT_CMAP:
                    plt.figure()
                    hp.mollview(
                                COMPLETENESS_MAP,
                                nest=True,
                                coord="G",               # plot in Galactic coords
                                title="Mask: 1 outside |b|<=10°, 0 inside",
                                cmap="coolwarm",
                                min=0, max=1
                            )
                    hp.graticule()
                    plt.savefig('cmap.pdf', bbox_inches='tight')
                    plt.close()
            
            ###################### NEW LIKELIHOOD ######################

            posterior_path = f'./agn_posteriors_precompute_{fagn_idx}.hdf5'

            if not ASSUME_PERFECT_REDSHIFT:
                if os.path.exists(posterior_path):
                    os.remove(posterior_path)

                compute_and_save_posteriors_hdf5(posterior_path, 
                                                AGN_REDSHIFT, 
                                                AGN_REDSHIFT_ERR, 
                                                AGN_POSTERIOR_NORM_AX,
                                                AGN_ZMAX)

            # Keep ~few GB in memory as global
            with h5py.File(posterior_path, "r") as f:
                dset = f["agn_redshift_posteriors"]
                AGN_POSTERIOR_DSET = dset[:, :]
            sum_of_posteriors = np.sum(AGN_POSTERIOR_DSET, axis=0)  # Sum of posteriors is required to normalize the in-catalog population prior
            
            if ASSUME_PERFECT_REDSHIFT:
                print('POPULATION NORMALIZATION DOES NOT TAKE MERGER RATE INTO ACCOUNT YET')
                redshift_population_prior_normalization = np.sum(AGN_REDSHIFT < ZMAX)  # TODO: 1/Nagn is the normalization of the population prior, but that should change when the merger rate is z-dependent -> np.sum( merger_rate(agn_redshift) )
            else:
                if LINAX:
                    dz = np.diff(S_AGN_Z_INTEGRAL_AX)[0]
                    redshift_population_prior_normalization = romb(sum_of_posteriors * merger_rate(S_AGN_Z_INTEGRAL_AX, MERGER_RATE_EVOLUTION, **MERGER_RATE_KWARGS) * z_cut(S_AGN_Z_INTEGRAL_AX, zcut=ZMAX), dx=dz)
                else:
                    dz = np.diff(np.log10(S_AGN_Z_INTEGRAL_AX))[0]
                    redshift_population_prior_normalization = romb(sum_of_posteriors * merger_rate(S_AGN_Z_INTEGRAL_AX, MERGER_RATE_EVOLUTION, **MERGER_RATE_KWARGS) * z_cut(S_AGN_Z_INTEGRAL_AX, zcut=ZMAX) * S_AGN_Z_INTEGRAL_AX * np.log(10), dx=dz)
            
            crossmatch_timer = time.time()
            S_agn_cw = np.zeros(BATCH)
            S_alt_cw = np.zeros(BATCH)
            S_alt = np.zeros(BATCH)
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = [
                    executor.submit(
                        process_single_gw, gw_idx, trial_idx, fagn_idx
                    )
                    for gw_idx in range(BATCH)
                ]

                for future in as_completed(futures):
                    gw_idx, sagn_cw, salt_cw, salt = future.result()
                    S_agn_cw[gw_idx] = sagn_cw
                    S_alt_cw[gw_idx] = salt_cw
                    S_alt[gw_idx] = salt
            
            S_agn_cw *= (4 * np.pi / redshift_population_prior_normalization)
            print('Crossmatching took:', time.time() - crossmatch_timer)

            #########################################################

            S_agn_cw = S_agn_cw[~np.isnan(S_agn_cw)]
            S_alt_cw = S_alt_cw[~np.isnan(S_alt_cw)]
            S_alt = S_alt[~np.isnan(S_alt)]

            print(f'\n--- AFTER CROSSMATCHING THERE ARE {len(S_agn_cw)} GWS LEFT ---\n')

            loglike = np.log(SKYMAP_CL * LOG_LLH_X_AX[None,:] * (S_agn_cw[:,None] - S_alt_cw[:,None]) + S_alt[:,None])

            nans = np.where(np.isnan(loglike))
            print('Got NaNs:')
            print((LOG_LLH_X_AX[None,:] * S_agn_cw[:,None])[nans])
            print((LOG_LLH_X_AX[None,:] * S_alt_cw[:,None])[nans])
            
            log_llh[:,fagn_idx] = np.sum(loglike, axis=0)  # sum over all GWs

            posterior = log_llh[:,fagn_idx]
            posterior -= np.max(posterior)
            pdf = np.exp(posterior)
            norm = simpson(y=pdf, x=LOG_LLH_X_AX, axis=0)  # Simpson should be fine...
            pdf = pdf / norm

        posteriors[trial_idx,:,:] = log_llh

    np.save(os.path.join(sys.path[0], FAGN_POSTERIOR_FNAME), posteriors)
