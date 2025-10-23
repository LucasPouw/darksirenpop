"""
The likelihood of V23 assumes perfect AGN redshift errors for each iteration. Averaging over many iterations of the catalog should then take this into account.
"""

import numpy as np
import sys
from tqdm import tqdm
from astropy.coordinates import SkyCoord
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import h5py
from default_globals import *
from utils import fast_z_at_value
import astropy.units as u
import os
from ligo.skymap.io.fits import read_sky_map
from ligo.skymap.postprocess import crossmatch_v23 as crossmatch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from scipy import stats
import traceback

# import healpy as hp
# import time
# import json
# import matplotlib.pyplot as plt
# from scipy.integrate import simpson, romb
# import pandas as pd


'''
With 500 GWs I get a small downward bias at high f_agn_true at the moment. Don't know why.
'''

FAGN_POSTERIOR_FNAME = 'v23_likelihood_posteriors'
INDICATOR = 'MOCK'
DIRECTORY_ID = 'moc_500'
BATCH = 500

LOG_LBOL_THRESH = 45.0  # erg/s
N_TRUE_FAGNS = 6
SKYMAP_CL = 0.9
AGN_ZERROR = 0

CALC_LOGLLH_AT_N_POINTS = 1000
LOG_LLH_X_AX = np.linspace(0.0001, 0.9999, CALC_LOGLLH_AT_N_POINTS)

ZMIN = 1e-4
ZMAX = 1.5  # p_rate(z > ZMAX) = 0
AGN_ZMAX = 1.5

AGN_COMDIST_MAX = COSMO.comoving_distance(AGN_ZMAX).value
AGN_VOLUME = 4 / 3 * np.pi * AGN_COMDIST_MAX**3


def process_one_gw(gw_idx, trial_idx, fagn_idx, cat_coord):

    filename = f"./skymaps_{DIRECTORY_ID}/skymap_{trial_idx}_{fagn_idx}_{gw_idx:05d}.fits.gz"
    skymap = read_sky_map(filename, moc=True)

    dP_dA = skymap["PROBDENSITY"]
    if np.sum(np.isnan(dP_dA)) == len(dP_dA):
        print('BAD SKYMAP:', filename)
        return gw_idx, np.nan

    result = crossmatch(sky_map=skymap, coordinates=cat_coord, cosmology=True)

    in_90_region = (result.searched_prob_vol <= SKYMAP_CL)
    p90 = np.sum(result.probdensity_vol[in_90_region])

    return gw_idx, p90


N_TRIALS = 1
posteriors = np.zeros((N_TRIALS, CALC_LOGLLH_AT_N_POINTS, N_TRUE_FAGNS))
for trial_idx in range(N_TRIALS):

    log_llh = np.zeros((len(LOG_LLH_X_AX), N_TRUE_FAGNS))
    for fagn_idx in range(N_TRUE_FAGNS):

        with h5py.File(f'./catalogs_{DIRECTORY_ID}/mockcat_{trial_idx}_{fagn_idx}.hdf5', 'r') as catalog:
            agn_ra = catalog['ra'][()]
            agn_dec = catalog['dec'][()]
            agn_rcom = catalog['comoving_distance'][()]
            print('Verify these are in radians:', np.min(agn_ra), np.max(agn_ra), np.min(agn_dec), np.max(agn_dec))

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
            # b = cat_coord.galactic.b.degree
        
        AGN_NUMDENS = len(agn_ra) / AGN_VOLUME
        

        ############################## THREADING ##############################
        S_alt = np.tile(SKYMAP_CL, BATCH)
        S_agn = np.zeros(BATCH)
        with ThreadPoolExecutor() as executor:  # os.cpu_count()
            future_to_index = future_to_index = {executor.submit(process_one_gw, gw_idx, trial_idx, fagn_idx, cat_coord): gw_idx for gw_idx in range(BATCH)}
            
            for future in tqdm(as_completed(future_to_index), total=BATCH):
                try:
                    idx, p90 = future.result()
                    S_agn[idx] = p90

                    if np.isnan(p90):
                        S_alt[idx] = np.nan
                
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error processing trial {future_to_index[future]}: {e}")
        
        S_agn /= AGN_NUMDENS
        #########################################################################

        # S_alt = np.tile(SKYMAP_CL, BATCH)
        # S_agn = np.zeros(BATCH)
        # for gw_idx in range(BATCH):
        #     filename = f"./skymaps_{DIRECTORY_ID}/skymap_{trial_idx}_{fagn_idx}_{gw_idx:05d}.fits.gz"

        #     skymap = read_sky_map(filename, moc=True)
        #     print(f'\nLoaded file: {filename}\n')

        #     dP_dA = skymap["PROBDENSITY"]
        #     if np.sum(np.isnan(dP_dA)) == len(dP_dA):
        #         print('BAD SKYMAP AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        #         S_agn[gw_idx] = np.nan
        #         S_alt[gw_idx] = np.nan
        #         continue  

        #     start_cm = time.time()
        #     result = crossmatch(sky_map=skymap,
        #                         coordinates=cat_coord,
        #                         cosmology=True)
        #     print('Total Crossmatch time:', time.time() - start_cm)
            
        #     in_90_region = (result.searched_prob_vol <= SKYMAP_CL)
        #     n90 = np.sum(in_90_region)
        #     p90 = np.sum(result.probdensity_vol[in_90_region])

        #     AGN_NUMDENS = len(agn_ra) / AGN_VOLUME
        #     S_agn[gw_idx] = p90 / AGN_NUMDENS
        #     print(p90 / AGN_NUMDENS, SKYMAP_CL)
        
        S_agn = S_agn[~np.isnan(S_agn)]
        S_alt = S_alt[~np.isnan(S_alt)]

        loglike = np.log(S_agn[:,None] * SKYMAP_CL * LOG_LLH_X_AX[None,:] + S_alt[:,None] * (1 - SKYMAP_CL * LOG_LLH_X_AX[None,:]))
        log_llh[:,fagn_idx] = np.sum(loglike, axis=0)  # sum over all GWs
    
    posteriors[trial_idx,:,:] = log_llh

np.save(os.path.join(sys.path[0], FAGN_POSTERIOR_FNAME), posteriors)
