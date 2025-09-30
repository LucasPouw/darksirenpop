import numpy as np
import sys
from tqdm import tqdm
from astropy.coordinates import SkyCoord
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import h5py
from default_arguments import DEFAULT_COSMOLOGY as COSMO
from utils import fast_z_at_value
import astropy.units as u
import os
from ligo.skymap.io.fits import read_sky_map
from p26_crossmatch import crossmatch_p26 as crossmatch
import matplotlib.pyplot as plt
from scipy.integrate import simpson, romb
from scipy import stats
from astropy.constants import c
import healpy as hp
import time

# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
# import json
# import pandas as pd

SPEED_OF_LIGHT_KMS = c.to('km/s').value
os.environ["OMP_NUM_THREADS"] = "1"

z_edges = np.array([0.0000, 0.1875, 0.3750, 0.5625, 0.7500, 0.9375, 1.1250, 1.3125, 1.5000])

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
LOG_LBOL_THRESH = 45.0  # erg/s
CMAP_NSIDE = 64
N_TRUE_FAGNS = 6
SKYMAP_CL = 0.999
LUM_THRESH = "45.5"
ADD_NAGN_TO_CAT = int(1e5)
COMPLETENESS = 0.7

ASSUME_PERFECT_REDSHIFT = True
AGN_ZERROR = 0

CALC_LOGLLH_AT_N_POINTS = 1000
LOG_LLH_X_AX = np.linspace(0.0001, 0.9999, CALC_LOGLLH_AT_N_POINTS)

ZMIN = 1e-4
ZMAX = 1.5  # p_rate(z > ZMAX) = 0
AGN_ZMAX = 1.5

COMDIST_MIN = COSMO.comoving_distance(ZMIN).value
COMDIST_MAX = COSMO.comoving_distance(ZMAX).value
AGN_COMDIST_MAX = COSMO.comoving_distance(AGN_ZMAX).value

AGN_VOLUME = 4 / 3 * np.pi * AGN_COMDIST_MAX**3


### TODO: find optimal axes, maybe also calculate AGN posteriors from -5sig to 5sig and interpolate to AGN_Z_INTEGRAL_AX ###
S_AGN_Z_INTEGRAL_AX = np.geomspace(ZMIN, ZMAX, int(8192/4)+1)  # Sets the resolution of the redshift prior, should capture all information of AGN posteriors, see Gray et al. 2022, 2023
S_ALT_Z_INTEGRAL_AX = np.geomspace(ZMIN, ZMAX, int(8192/4)+1)
AGN_POSTERIOR_NORM_AX = np.geomspace(ZMIN, AGN_ZMAX, int(8192/4)+1)


def sample_spherical_angles(n_samps=1):
    theta = np.arccos(np.random.uniform(size=n_samps, low=-1, high=1))  # Cosine is uniformly distributed between -1 and 1 -> cos between 0 and pi
    phi = 2 * np.pi * np.random.uniform(size=n_samps)  # Draws phi from 0 to 2pi
    return theta, phi


def uniform_shell_sampler(rmin, rmax, n_samps):
    r = ( np.random.uniform(size=n_samps, low=rmin**3, high=rmax**3) )**(1/3)
    theta, phi = sample_spherical_angles(n_samps)
    return r, theta, phi


def uniform_comoving_prior(z):
    '''Proportional to uniform in comoving volume prior.'''
    z = np.atleast_1d(z)
    chi = COSMO.comoving_distance(z).value         # Mpc
    H_z = COSMO.H(z).value                         # km/s/Mpc
    dchi_dz = SPEED_OF_LIGHT_KMS / H_z             # Mpc
    p = (chi**2 * dchi_dz)
    return p


def z_cut(z, zcut=ZMAX):
    '''Artificial redshift cuts in data analysis should be taken into account.'''
    stepfunc = np.ones_like(z)
    stepfunc[z > zcut] = 0
    return stepfunc


def compute_and_save_posteriors_hdf5(filename, 
                                     all_agn_z, 
                                     all_agn_z_err, 
                                     agn_posterior_norm_ax,
                                     agn_zmax):

    n_agn = len(all_agn_z)
    n_z = len(S_AGN_Z_INTEGRAL_AX)
    chunk_size = int(1e6 / n_z)

    # Precompute dx for romb integration
    dx = np.diff(np.log10(agn_posterior_norm_ax))[0]

    with h5py.File(filename, "w") as f:
        # Prepare dataset for posteriors
        dset = f.create_dataset(
            "agn_redshift_posteriors",
            shape=(n_agn, n_z),
            dtype=np.float64
        )

        for start in tqdm( range(0, n_agn, chunk_size) ):
            end = min(start + chunk_size, n_agn)

            # Slice this chunk
            z_chunk = all_agn_z[start:end]
            zerr_chunk = all_agn_z_err[start:end]

            # Likelihood (vectorized over chunk)
            likelihood = lambda z: stats.truncnorm.pdf(
                z,
                a=(ZMIN - z_chunk[:, None]) / zerr_chunk[:, None],
                b=(np.inf - z_chunk[:, None]) / zerr_chunk[:, None],
                loc=z_chunk[:, None],
                scale=zerr_chunk[:, None]
            )

            # Unnormalized posterior
            agn_posteriors_unnorm = lambda z: (
                likelihood(z) * uniform_comoving_prior(z) * z_cut(z, zcut=agn_zmax)
            )

            # Normalization (romb over log10 grid)
            z_norms = romb(
                agn_posteriors_unnorm(agn_posterior_norm_ax) * agn_posterior_norm_ax * np.log(10),
                dx=dx
            )

            # Final posteriors on integration axis
            posteriors = agn_posteriors_unnorm(S_AGN_Z_INTEGRAL_AX) / z_norms[:, None]

            # Write to HDF5
            dset[start:end, :] = posteriors

    print(f"All AGN posteriors written to {filename}")


if AGN_ZERROR == 0:
    assert ASSUME_PERFECT_REDSHIFT == True, 'Cannot have zero redshift error and not assume a perfect measurement.'


N_TRIALS = 1
posteriors = np.zeros((N_TRIALS, CALC_LOGLLH_AT_N_POINTS, N_TRUE_FAGNS))
for trial_idx in range(N_TRIALS):

    log_llh = np.zeros((len(LOG_LLH_X_AX), N_TRUE_FAGNS))
    for fagn_idx in range(N_TRUE_FAGNS):

        # fagn_idx = 1

        with h5py.File(f'./catalogs_{DIRECTORY_ID}/mockcat_{trial_idx}_{fagn_idx}.hdf5', 'r') as catalog:
            
            agn_ra = catalog['ra'][()]
            agn_dec = catalog['dec'][()]
            agn_rcom = catalog['comoving_distance'][()]

            ### FOR TESTING, ADD UNCORRELATED AGN TO THE CATALOG: S_AGN -> S_ALT SHOULD BE SEEN!! ###
            if ADD_NAGN_TO_CAT:
                new_rcom, new_theta, new_phi = uniform_shell_sampler(COMDIST_MIN, AGN_COMDIST_MAX, ADD_NAGN_TO_CAT)
                agn_ra = np.append(agn_ra, new_phi)
                agn_dec = np.append(agn_dec, np.pi * 0.5 - new_theta)
                agn_rcom = np.append(agn_rcom, new_rcom)
            #########################################################

            true_agn_redshift = fast_z_at_value(COSMO.comoving_distance, agn_rcom * u.Mpc)

            agn_redshift_err = np.tile(AGN_ZERROR, len(agn_ra))

            if not AGN_ZERROR:
                obs_agn_redshift = true_agn_redshift
            else:
                print('Scattering AGN!')
                obs_agn_redshift = stats.truncnorm.rvs(size=len(agn_ra), 
                                                        a=(ZMIN - true_agn_redshift) / agn_redshift_err, 
                                                        b=(np.inf - true_agn_redshift) / agn_redshift_err, 
                                                        loc=true_agn_redshift, 
                                                        scale=agn_redshift_err)  # Or a Gaussian: np.random.normal(loc=true_agn_redshift, scale=agn_redshift_err, size=len(agn_ra))
            
            obs_agn_rlum = COSMO.luminosity_distance(obs_agn_redshift).value


            cat_coord = SkyCoord(agn_ra * u.rad, agn_dec * u.rad, obs_agn_rlum * u.Mpc)
            b = cat_coord.galactic.b.degree
            

            ### Making a redshift-incomplete catalog ###
            c_per_zbin = quaia_c_vals[:, threshold_map[LUM_THRESH]]
            # redshift_incomplete_mask = np.zeros_like(obs_agn_redshift, dtype=bool)
            # for i, c in enumerate(c_per_zbin):
            #     z_low = z_edges[i]
            #     z_high = z_edges[i + 1]

            #     agn_in_bin = np.where((obs_agn_redshift > z_low) & (obs_agn_redshift < z_high))[0]
            #     keep_these = np.random.rand(len(agn_in_bin)) < c
            #     redshift_incomplete_mask[agn_in_bin[keep_these]] = True
            
            # plt.figure()
            # plt.hist(obs_agn_redshift[redshift_incomplete_mask], density=True, bins=z_edges)
            # plt.savefig('zhist.pdf', bbox_inches='tight')
            # plt.close()
            #############################################

            # incomplete_catalog_idx = np.ones_like(agn_ra, dtype=bool)
            latitude_mask = np.ones_like(agn_ra, dtype=bool)
            # c_per_zbin = np.tile(COMPLETENESS, len(c_per_zbin))


            # latitude_mask = np.logical_or(b > 10, b < -10)
            incomplete_catalog_idx = np.random.choice(np.arange(len(agn_ra))[latitude_mask], size=int(COMPLETENESS * np.sum(latitude_mask)), replace=False)
            print('KEEPING NAGN =', len(incomplete_catalog_idx))

            clist = []
            for i, _ in enumerate(c_per_zbin):
                z_low = z_edges[i]
                z_high = z_edges[i + 1]

                nagn_in_bin = np.sum((obs_agn_redshift[latitude_mask] > z_low) & (obs_agn_redshift[latitude_mask] < z_high))
                nobs_in_bin = np.sum((obs_agn_redshift[incomplete_catalog_idx] > z_low) & (obs_agn_redshift[incomplete_catalog_idx] < z_high))
                clist.append(nobs_in_bin)
            c_per_zbin = np.array(clist / nagn_in_bin)
            print(c_per_zbin)

            agn_ra = agn_ra[incomplete_catalog_idx]
            agn_dec = agn_dec[incomplete_catalog_idx]
            obs_agn_redshift = obs_agn_redshift[incomplete_catalog_idx]
            agn_redshift_err = agn_redshift_err[incomplete_catalog_idx]
            obs_agn_rlum = obs_agn_rlum[incomplete_catalog_idx]

            npix = hp.nside2npix(CMAP_NSIDE)

            theta, phi = hp.pix2ang(CMAP_NSIDE, np.arange(npix), nest=True)
            map_coord = SkyCoord(phi * u.rad, (np.pi * 0.5 - theta) * u.rad)
            map_b = map_coord.galactic.b.degree
            outside_galactic_plane_pix = np.logical_or(map_b > 10, map_b < -10)

            values = np.tile(1, npix)
            # values[~outside_galactic_plane_pix] = 0

            hp.write_map(CMAP_PATH, values, nest=True, dtype=np.float32, overwrite=True)
            completeness_map = hp.read_map(CMAP_PATH, nest=True)
            
            if PLOT_CMAP:
                plt.figure()
                hp.mollview(
                            completeness_map,
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
                                            obs_agn_redshift, 
                                            agn_redshift_err, 
                                            AGN_POSTERIOR_NORM_AX,
                                            AGN_ZMAX)
        
        S_agn_cw = np.zeros(BATCH)
        S_alt_cw = np.zeros(BATCH)
        S_alt = np.zeros(BATCH)
        # total_counter = 0
        # fromagn_counter = 0
        for gw_idx in range(BATCH):
            filename = f"./skymaps_{DIRECTORY_ID}/skymap_{trial_idx}_{fagn_idx}_{gw_idx:05d}.fits.gz"

            skymap = read_sky_map(filename, moc=True)
            print(f'\nLoaded file: {filename}')

        #     if np.sum(np.isnan( np.array(skymap["PROBDENSITY"])) ) == len(skymap["PROBDENSITY"]):
        #         print('BAD SKYMAP')

        #         total_counter += 1
        #         if int(gw_idx) < int(500 * np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])[fagn_idx]):
        #             fromagn_counter += 1
        
        # print(total_counter, 'total')
        # print(fromagn_counter, 'missed from agn')
                

            # This function calculates the evidence for each hypothesis by integrating in redshift-space
            sagn_cw, salt_cw, salt = crossmatch(posterior_path=posterior_path,
                                                sky_map=skymap,
                                                completeness_map=completeness_map,
                                                completeness_zedges=np.array(z_edges),
                                                completeness_zvals=np.array(c_per_zbin),
                                                agn_ra=agn_ra, 
                                                agn_dec=agn_dec, 
                                                agn_lumdist=obs_agn_rlum, 
                                                agn_redshift=obs_agn_redshift,
                                                agn_redshift_err=agn_redshift_err,
                                                skymap_cl=SKYMAP_CL,
                                                gw_zcut=ZMAX,
                                                s_agn_z_integral_ax=S_AGN_Z_INTEGRAL_AX, 
                                                s_alt_z_integral_ax=S_ALT_Z_INTEGRAL_AX,
                                                assume_perfect_redshift=ASSUME_PERFECT_REDSHIFT)
            
            S_agn_cw[gw_idx] = sagn_cw
            S_alt_cw[gw_idx] = salt_cw
            S_alt[gw_idx] = salt
            print(sagn_cw, salt_cw, salt)

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

        # plt.figure()
        # plt.plot(LOG_LLH_X_AX, pdf)
        # plt.savefig(f'mock_posterior_fagnidx_{fagn_idx}.pdf', bbox_inches='tight')
        # plt.close()

    posteriors[trial_idx,:,:] = log_llh

np.save(os.path.join(sys.path[0], FAGN_POSTERIOR_FNAME), posteriors)
