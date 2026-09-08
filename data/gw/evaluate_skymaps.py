########################################################################
# Evaluate GW sky maps to obtain sky-marginalized redshift posteriors.
# Generate two files per GW: one contains the z-posterior marginalized 
# over the full sky, and the other is only marginalized outside the
# Galactic plane. The latter is used in the completeness-weighted terms
# in the likelihood calculation.

# Code can be used on both real and mock data
########################################################################

import numpy as np
import glob
from tqdm import tqdm
from numba import njit, prange
import math
import healpy as hp
import os, sys
import json
from pathlib import Path
import h5py

import astropy.units as u
from astropy.coordinates import SkyCoord

from ligo.skymap.io.fits import read_sky_map
from ligo.skymap import moc

from darksirenpop.utilities.redshift_utils import redshift_pdf_given_lumdist_pdf, fast_z_at_value
from darksirenpop.utilities.default_globals import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, required=False)
parser.add_argument('--real-data', action='store_true')
args = parser.parse_args()

ROOT_DIRECTORY = args.root
REAL_DATA = args.real_data

REALDATA_WRITE_DIR = REWEIGHT_GWTC5_SKYMAP_EVALS  # Directory to store all evaluated redshift posteriors from real data

TYPES = ['agn', 'alt']
DIRECTORY_IDS = np.arange(0, 201, 1)  # If using mock data, check that this matches your sample
SKYMAP_CL = 0.999  # We will only integrate over this area to save compute, make sure PDFs normalize to this value!
CMAP_NSIDE = 64  # Completeness map encodes the survey footprint, which weights the GW sky posterior

npix = hp.nside2npix(CMAP_NSIDE)
theta, phi = hp.pix2ang(CMAP_NSIDE, np.arange(npix), nest=True)
map_coord = SkyCoord(phi * u.rad, (np.pi * 0.5 - theta) * u.rad)
map_b = map_coord.galactic.b.degree
outside_galactic_plane_pix = np.logical_or(map_b > 10, map_b < -10)
COMPLETENESS_MAP = np.tile(1., npix)
COMPLETENESS_MAP[~outside_galactic_plane_pix] = 0


@njit(parallel=True, fastmath=True)
def allsky_marginal_lumdist_distribution(dl_array, dP, norm, mu, sigma):
    M = dl_array.shape[0]
    N = mu.shape[0]
    result = np.empty(M)

    # --- Precompute constants ---
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    inv_sigma = 1.0 / sigma
    weight = dP * norm
    coeff = inv_sqrt_2pi * inv_sigma

    for i in prange(M):
        dl = dl_array[i]
        dl2 = dl * dl
        acc = 0.0
        for j in range(N):
            diff = (dl - mu[j]) * inv_sigma[j]
            g = math.exp(-0.5 * diff * diff)
            acc += weight[j] * dl2 * g * coeff[j]
        result[i] = acc
    return result


def evaluate_skymap(filename, completeness_map=COMPLETENESS_MAP, skymap_cl=SKYMAP_CL):
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
        sys.exit('TODO: run function deleting the skymap and its true source')

    dA = moc.uniq2pixarea(skymap_uniq)  # Pixel areas in sr
    dP = dP_dA * dA  # Dimensionless probability density in each pixel
    cumprob = np.cumsum(dP)
    cumprob[cumprob > 1] = 1.  # Correcting floating point error which could cause issues when skymap_cl == 1

    # Get GW redshift posterior marginalized over the whole sky and sky-completeness-weighted (for now that is just 1 or 0 depending on surveyed sky region)
    skymap_theta, skymap_phi = moc.uniq2ang(skymap_uniq)
    cmap_nside = hp.npix2nside(len(completeness_map))
    pix_idx = hp.ang2pix(cmap_nside, skymap_theta, skymap_phi, nest=True)
    pixprob_within_cl = (cumprob <= skymap_cl)
    cmap_vals_in_gw_skymap = completeness_map[pix_idx]
    surveyed = (cmap_vals_in_gw_skymap != 0)
    skyprob_nonzero = (dP != 0)  # Only these pixels have GW posterior support

    selected_gwpix = skyprob_nonzero & pixprob_within_cl

    print(f'Total probability in selected pixels: {np.sum(dP[selected_gwpix]):.5f}. Compare with chosen sky CL: {skymap_cl}')

    if np.isinf(np.median(mu[selected_gwpix])):
        print(filename, 'contains pixels with inf DISTMU. Avoid this by making smoother sky maps.')
        # median = np.median(mu[selected_gwpix])
        # print(median, 'aaa')
        # print(np.sum(np.isinf(mu[selected_gwpix])), len(mu))
        # print(sigma[selected_gwpix][np.isinf(mu[selected_gwpix])])
        # print(norm[selected_gwpix][np.isinf(mu[selected_gwpix])])
        # print(np.sum(dP[selected_gwpix][ np.isinf(mu[selected_gwpix]) ]), 'total dP of those pixels in mu=inf pixels')
        # print(f'IF we decide to remove mu=inf pixels, we are left with total dP={np.sum(dP[selected_gwpix][ ~np.isinf(mu[selected_gwpix]) ])}')
        # sys.exit(1)

    elif np.median(mu[selected_gwpix]) > 0:  # 
        median = fast_z_at_value(COSMO.luminosity_distance, np.median(mu[selected_gwpix]) * u.Mpc)
        error = fast_z_at_value(COSMO.luminosity_distance, np.median(sigma[selected_gwpix]) * u.Mpc)
        eval_ax = np.linspace(median - 10 * error, median + 10 * error, 500)

    else:
        median = 0
        error = fast_z_at_value(COSMO.luminosity_distance, np.median(sigma[selected_gwpix]) * u.Mpc)
        eval_ax = np.linspace(0, 10 * error, 500)

    if np.isnan(median):
        print('BAD SKYMAP, SKIPPING FILE:', filename)
        return None

    gw_redshift_posterior_marginalized_evaluated = redshift_pdf_given_lumdist_pdf(eval_ax, 
                                                                                    allsky_marginal_lumdist_distribution, 
                                                                                    dP=dP[selected_gwpix],
                                                                                    norm=norm[selected_gwpix], 
                                                                                    mu=mu[selected_gwpix], 
                                                                                    sigma=sigma[selected_gwpix])
    
    # Marginalize the GW posterior over sky position, weighting with sky completeness (currently only 1 for surveyed and 0 for not surveyed): int dOmega p_GW(z, Omega | d) * p(G|z, Omega)
    gw_redshift_posterior_marginalized_cw_evaluated = redshift_pdf_given_lumdist_pdf(eval_ax, 
                                                                                    allsky_marginal_lumdist_distribution, 
                                                                                    dP=dP[surveyed & selected_gwpix],
                                                                                    norm=norm[surveyed & selected_gwpix], 
                                                                                    mu=mu[surveyed & selected_gwpix], 
                                                                                    sigma=sigma[surveyed & selected_gwpix])

    if np.sum(dP[selected_gwpix][ ~np.isinf(mu[selected_gwpix]) ]) < 0.998:
        print(f'Warning for event {filename}: dP in pixels with finite mu is {np.sum(dP[selected_gwpix][ ~np.isinf(mu[selected_gwpix]) ])}. Should be 0.999.')

        # gwname = filename.split('/')[-1].split('.')[0]
        # plt.figure()
        # plt.plot(eval_ax, gw_redshift_posterior_marginalized_evaluated, label='post')
        # plt.plot(eval_ax, gw_redshift_posterior_marginalized_cw_evaluated, label='cwpost')
        # plt.title(gwname)
        # plt.xlabel('Redshift')
        # plt.legend()
        # plt.show()

    return eval_ax, gw_redshift_posterior_marginalized_evaluated, gw_redshift_posterior_marginalized_cw_evaluated


def path2json(key, value, json_path):
    output_file = Path(json_path)

    # Load existing data
    if output_file.exists():
        data = json.loads(output_file.read_text())
    else:
        data = {}
    
    data[str(key)] = value  # Append new value
    output_file.write_text(json.dumps(data, indent=2))
    return


### Running the analysis ###

if REAL_DATA:  # TODO: Still saving many separate .npy files for each real GW event. Instead, should change to single .h5 file, just like the mock data analyis is using.

    if not os.path.isdir(REALDATA_WRITE_DIR):
        os.makedirs(REALDATA_WRITE_DIR)

    with open(SKYMAP_JSON_PATH, "r") as f:
        skymaps_dict = json.load(f)

    for key in skymaps_dict.keys():
        # if key != 'GW230702_185453':
        #     continue
        filename = skymaps_dict[key]

        try:
            eval_ax, gw_redshift_posterior_marginalized_evaluated, gw_redshift_posterior_marginalized_cw_evaluated = evaluate_skymap(filename)
        except Exception as e:
            print(f'Error for event {key}: {e}')
            continue

        outfile = f'{REALDATA_WRITE_DIR}zpost_{key}_gpmask_False_skymapcl_{SKYMAP_CL}_cmapnside_{CMAP_NSIDE}.npy'
        cw_outfile = f'{REALDATA_WRITE_DIR}zpost_{key}_gpmask_True_skymapcl_{SKYMAP_CL}_cmapnside_{CMAP_NSIDE}.npy'

        np.save(outfile, np.array([eval_ax, gw_redshift_posterior_marginalized_evaluated]))
        np.save(cw_outfile, np.array([eval_ax, gw_redshift_posterior_marginalized_cw_evaluated]))

        path2json(key, outfile, SKYMAP_EVALS_JSON_PATH)
        path2json(key, cw_outfile, SKYMAP_CW_EVALS_JSON_PATH)

else:
    
    for DIRECTORY_ID in DIRECTORY_IDS:
        print(DIRECTORY_ID)
        output_directory = glob.glob(f'{ROOT_DIRECTORY}/output_run_{DIRECTORY_ID}_*')[0]
        for TYPE in TYPES:

            SKYMAP_DIR = f'{output_directory}/skymaps/{TYPE}/'
            WRITE_DIR = f'{output_directory}/skymaps_evaluated/{TYPE}/'
            if not os.path.isdir(WRITE_DIR):
                os.makedirs(WRITE_DIR)

            gw_fnames = glob.glob(SKYMAP_DIR + 'skymap*.fits.gz')
            false_h5_path = f'{WRITE_DIR}/zpost_gpmask_False_skymapcl_{SKYMAP_CL}_cmapnside_{CMAP_NSIDE}.h5'  # GP mask = False
            true_h5_path = f'{WRITE_DIR}/zpost_gpmask_True_skymapcl_{SKYMAP_CL}_cmapnside_{CMAP_NSIDE}.h5'  # GP mask = True
            with h5py.File(false_h5_path, 'w') as false_h5, \
                h5py.File(true_h5_path, 'w') as true_h5:

                for i, filename in tqdm(enumerate(gw_fnames), total=len(gw_fnames)):

                    gw_id = filename[-13:-8]

                    try:
                        eval_ax, gw_redshift_posterior_marginalized_evaluated, gw_redshift_posterior_marginalized_cw_evaluated = evaluate_skymap(filename)
                    except Exception as e:
                        print(f'Error processing {filename}: {e}')
                        continue

                    # Store
                    gp_false = false_h5.create_group(gw_id)
                    gp_false.create_dataset('eval_ax', data=eval_ax, compression='gzip')
                    gp_false.create_dataset('posterior', data=gw_redshift_posterior_marginalized_evaluated, compression='gzip')

                    gp_true = true_h5.create_group(gw_id)
                    gp_true.create_dataset('eval_ax', data=eval_ax, compression='gzip')
                    gp_true.create_dataset('posterior', data=gw_redshift_posterior_marginalized_cw_evaluated, compression='gzip')
