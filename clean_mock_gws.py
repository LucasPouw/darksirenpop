import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from numba import njit, prange
import math
import healpy as hp
from scipy.interpolate import CubicSpline
import os, sys

import astropy.units as u
from astropy.coordinates import SkyCoord

from ligo.skymap.io.fits import read_sky_map
from ligo.skymap import moc

from redshift_utils import redshift_pdf_given_lumdist_pdf


def allsky_marginal_lumdist_distribution_old(dl_array, dP, norm, mu, sigma):
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


@njit(parallel=True, fastmath=True)
def allsky_marginal_lumdist_distribution(dl_array, dP, norm, mu, sigma):
    M = dl_array.shape[0]
    N = mu.shape[0]

    result = np.empty(M)

    # --- Precompute constants ---
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    # Precompute arrays (1D, cache-friendly)
    inv_sigma = 1.0 / sigma
    weight = dP * norm
    coeff = inv_sqrt_2pi * inv_sigma  # combines normalization

    for i in prange(M):
        dl = dl_array[i]
        dl2 = dl * dl
        acc = 0.0

        for j in range(N):
            # reuse inv_sigma
            diff = (dl - mu[j]) * inv_sigma[j]

            # fully scalar gaussian
            g = math.exp(-0.5 * diff * diff)

            # fused accumulation (minimize temporaries)
            acc += weight[j] * dl2 * g * coeff[j]

        result[i] = acc

    return result


DIRECTORY_ID = 'all'
SKYMAP_DIR = f"./skymaps_{DIRECTORY_ID}/"
SKYMAP_CL = 0.999
CMAP_NSIDE = 64
Z_INTEGRAL_AX = np.linspace(0, 10, 5000)
HIGHRES_Z_AX = np.linspace(0, 1.5, int(1e5))

WRITE_DIR = f"./skymaps_evaluated_{DIRECTORY_ID}/"
if not os.path.isdir(WRITE_DIR):
    os.mkdir(WRITE_DIR)


npix = hp.nside2npix(CMAP_NSIDE)
theta, phi = hp.pix2ang(CMAP_NSIDE, np.arange(npix), nest=True)
map_coord = SkyCoord(phi * u.rad, (np.pi * 0.5 - theta) * u.rad)
map_b = map_coord.galactic.b.degree
outside_galactic_plane_pix = np.logical_or(map_b > 10, map_b < -10)
completeness_map = np.tile(1., npix)
completeness_map[~outside_galactic_plane_pix] = 0



gw_fnames = glob.glob(SKYMAP_DIR + 'skymap*.fits.gz')
for _, filename in tqdm(enumerate(gw_fnames), total=len(gw_fnames)):

    gw_id = filename[-13:-8]

    sky_map = read_sky_map(filename, moc=True)
    sky_map = np.flipud(np.sort(sky_map, order="PROBDENSITY"))
    
    # Unpacking skymap
    dP_dA = sky_map["PROBDENSITY"]  # Probdens in 1/sr
    mu = sky_map["DISTMU"]          # Ansatz mean in Mpc
    sigma = sky_map["DISTSIGMA"]    # Ansatz width in Mpc
    norm = sky_map["DISTNORM"]      # Ansatz norm in 1/Mpc^2
    norm[np.isinf(norm)] = 0        # Infs are observed to happen rarely in low-probability sky regions, this line avoids nans later if using CL->1

    if np.sum(np.isnan(dP_dA)) > 0:
        print('BAD SKYMAP')
        sys.exit('TODO: run function deleting the skymap and its true source')

    dA = moc.uniq2pixarea(sky_map["UNIQ"])  # Pixel areas in sr
    dP = dP_dA * dA  # Dimensionless probability density in each pixel
    cumprob = np.cumsum(dP)
    cumprob[cumprob > 1] = 1.  # Correcting floating point error which could cause issues when skymap_cl == 1

    # Get GW redshift posterior marginalized over the whole sky and sky-completeness-weighted (for now that is just 1 or 0 depending on surveyed sky region)
    skymap_theta, skymap_phi = moc.uniq2ang(sky_map['UNIQ'])
    cmap_nside = hp.npix2nside(len(completeness_map))
    pix_idx = hp.ang2pix(cmap_nside, skymap_theta, skymap_phi, nest=True)
    pixprob_within_cl = (cumprob <= SKYMAP_CL)
    cmap_vals_in_gw_skymap = completeness_map[pix_idx]
    surveyed = (cmap_vals_in_gw_skymap != 0)
    skyprob_nonzero = (dP != 0)

    gw_redshift_posterior_marginalized_evaluated = redshift_pdf_given_lumdist_pdf(Z_INTEGRAL_AX, 
                                                                                    allsky_marginal_lumdist_distribution, 
                                                                                    dP=dP[skyprob_nonzero & pixprob_within_cl],
                                                                                    norm=norm[skyprob_nonzero & pixprob_within_cl], 
                                                                                    mu=mu[skyprob_nonzero & pixprob_within_cl], 
                                                                                    sigma=sigma[skyprob_nonzero & pixprob_within_cl])
    
    # Marginalize the GW posterior over sky position, weighting with sky completeness (currently only 1 for surveyed and 0 for not surveyed): int dOmega p_GW(z, Omega | d) * p(G|z, Omega)
    gw_redshift_posterior_marginalized_cw_evaluated = redshift_pdf_given_lumdist_pdf(Z_INTEGRAL_AX, 
                                                                                    allsky_marginal_lumdist_distribution, 
                                                                                    dP=dP[surveyed & skyprob_nonzero & pixprob_within_cl],
                                                                                    norm=norm[surveyed & skyprob_nonzero & pixprob_within_cl], 
                                                                                    mu=mu[surveyed & skyprob_nonzero & pixprob_within_cl], 
                                                                                    sigma=sigma[surveyed & skyprob_nonzero & pixprob_within_cl])
    
    # np.save(f'{WRITE_DIR}zpost_{gw_id}_gpmask_False_skymapcl_{SKYMAP_CL}_cmapnside_{CMAP_NSIDE}', np.array([Z_INTEGRAL_AX, gw_redshift_posterior_marginalized_evaluated]))
    # np.save(f'{WRITE_DIR}zpost_{gw_id}_gpmask_True_skymapcl_{SKYMAP_CL}_cmapnside_{CMAP_NSIDE}', np.array([Z_INTEGRAL_AX, gw_redshift_posterior_marginalized_cw_evaluated]))





    # gwpost_interp = CubicSpline(Z_INTEGRAL_AX, gw_redshift_posterior_marginalized_evaluated, extrapolate=False)
    # gwpost_cw_interp = CubicSpline(Z_INTEGRAL_AX, gw_redshift_posterior_marginalized_cw_evaluated, extrapolate=False)
    # # gwpost_interp = interp1d(Z_INTEGRAL_AX, gw_redshift_posterior_marginalized_evaluated, kind='linear', bounds_error=False, fill_value=0)  
    # # gwpost_cw_interp = interp1d(Z_INTEGRAL_AX, gw_redshift_posterior_marginalized_cw_evaluated, kind='linear', bounds_error=False, fill_value=0)





    # z_array, post = np.load(f'{WRITE_DIR}zpost_{gw_id}_gpmask_False_skymapcl_{SKYMAP_CL}_cmapnside_{CMAP_NSIDE}.npy')
    # gwpost_interp = CubicSpline(z_array, post, extrapolate=False)

    # z_array, cwpost = np.load(f'{WRITE_DIR}zpost_{gw_id}_gpmask_True_skymapcl_{SKYMAP_CL}_cmapnside_{CMAP_NSIDE}.npy')
    # gwpost_cw_interp = CubicSpline(z_array, cwpost, extrapolate=False)

    # interped_highres = gwpost_interp(HIGHRES_Z_AX)
    # interped_cw_highres = gwpost_cw_interp(HIGHRES_Z_AX)

    # highres = redshift_pdf_given_lumdist_pdf(HIGHRES_Z_AX, 
    #                                                                                 allsky_marginal_lumdist_distribution, 
    #                                                                                 dP=dP[skyprob_nonzero & pixprob_within_cl],
    #                                                                                 norm=norm[skyprob_nonzero & pixprob_within_cl], 
    #                                                                                 mu=mu[skyprob_nonzero & pixprob_within_cl], 
    #                                                                                 sigma=sigma[skyprob_nonzero & pixprob_within_cl])
    
    # highres_cw = redshift_pdf_given_lumdist_pdf(HIGHRES_Z_AX, 
    #                                                                                 allsky_marginal_lumdist_distribution, 
    #                                                                                 dP=dP[surveyed & skyprob_nonzero & pixprob_within_cl],
    #                                                                                 norm=norm[surveyed & skyprob_nonzero & pixprob_within_cl], 
    #                                                                                 mu=mu[surveyed & skyprob_nonzero & pixprob_within_cl], 
    #                                                                                 sigma=sigma[surveyed & skyprob_nonzero & pixprob_within_cl])
    # thresh = 1e-1
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    # ax1.plot(HIGHRES_Z_AX, highres)
    # ax1.plot(HIGHRES_Z_AX, interped_highres)
    # ax2.plot(HIGHRES_Z_AX, highres - interped_highres)
    # ax3.plot(HIGHRES_Z_AX[highres > thresh], (highres - interped_highres)[highres > thresh] / highres[highres > thresh])
    # plt.show()

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    # ax1.plot(HIGHRES_Z_AX, highres_cw)
    # ax1.plot(HIGHRES_Z_AX, interped_cw_highres)
    # ax2.plot(HIGHRES_Z_AX, highres_cw - interped_cw_highres)
    # ax3.plot(HIGHRES_Z_AX[highres_cw > thresh], (highres_cw - interped_cw_highres)[highres_cw > thresh] / highres_cw[highres_cw > thresh])
    # plt.show()
    


### CHECK FOR GOOD SKYMAPS, REMOVE SKYMAPS AND THEIR COORDINATES IF NOT ###

# ID = 'x'
# SKYMAP_DIR = f"/home/lucas/Documents/PhD/skymaps_{ID}/"
# COORD_PATH = f'/home/lucas/Documents/PhD/true_r_theta_phi_{ID}.txt'

# for filename in tqdm(np.array(glob.glob(SKYMAP_DIR + 'skymap_0*'))):

#     skymap = read_sky_map(filename, moc=True)
#     nnans = np.sum(np.isnan(np.array(skymap['PROBDENSITY'])))
#     if nnans > 0:
#         print(nnans)
#         os.system(f'rm -rf {filename}')

# good_skymaps = np.array(glob.glob(SKYMAP_DIR + 'skymap_0*'))
# good_labels = [f[-13:-8] for f in good_skymaps]

# filtered_lines = []
# with open(COORD_PATH, 'r') as file:
#     for line in file:
#         columns = line.strip().split(',')

#         first_column_value = columns[0]
#         if first_column_value in good_labels:
#             filtered_lines.append(line)

# with open(COORD_PATH, 'w') as file:
#     file.writelines(filtered_lines)



# # RENAME TO NEW IDENTIFIER

# # NEW_NAME = 'x'

# # Relabel idx
# for filename in tqdm(np.array(glob.glob(SKYMAP_DIR + 'skymap_0*'))):
#     new_filename = filename[:-13] + NEW_NAME + filename[-12:]

#     if not os.path.exists(new_filename):
#         os.rename(filename, new_filename)
#     else:
#         sys.exit(f'{filename} to {new_filename} cannot be done, already exists! FIXME')

# renamed_lines = []
# with open(COORD_PATH, 'r') as file:
#     for line in file:
#         columns = line.strip().split(',')
#         new_line = NEW_NAME + str(line)[1:]
#         renamed_lines.append(new_line)
# with open(COORD_PATH, 'w') as file:
#     file.writelines(renamed_lines)



# CHECK FOR COORDINATES WITH ID THAT DOES NOT MATCH ANY GW

# SKYMAP_DIR = f"/home/lucas/Documents/PhD/skymaps_all/"
# COORD_PATH = f'/home/lucas/Documents/PhD/true_r_theta_phi_all.txt'

# all_gw_fnames = np.array(glob.glob(SKYMAP_DIR + 'skymap_*'))
# all_gw_IDs = sorted( np.array([f[-13:-8] for f in all_gw_fnames]).astype(int) )

# ALL_TRUE_SOURCES = np.genfromtxt('/home/lucas/Documents/PhD/true_r_theta_phi_all.txt', delimiter=',')
# ALL_TRUE_SOURCES = ALL_TRUE_SOURCES[ALL_TRUE_SOURCES[:, 0].argsort()]
# TRUE_SOURCE_IDENTIFIERS = ALL_TRUE_SOURCES[:,0]

# selected_sources = ALL_TRUE_SOURCES[np.searchsorted(TRUE_SOURCE_IDENTIFIERS, all_gw_IDs)]
# selected_source_ids = selected_sources[:,0]

# print(len(TRUE_SOURCE_IDENTIFIERS), len(selected_source_ids))

# for id in TRUE_SOURCE_IDENTIFIERS:
#     if id not in selected_source_ids:
#         print(id, np.sum(selected_source_ids == id))
#     else:
#         if np.sum(selected_source_ids == id) > 1:
#             print(id, np.sum(selected_source_ids == id))

# for id in all_gw_IDs:
#     if id not in TRUE_SOURCE_IDENTIFIERS:
#         print(id, np.sum(TRUE_SOURCE_IDENTIFIERS == id))
#     else:
#         if np.sum(TRUE_SOURCE_IDENTIFIERS == id) > 1:
#             print(id, np.sum(TRUE_SOURCE_IDENTIFIERS == id))
