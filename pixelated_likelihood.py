import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d
import json

from utils import ra_dec_from_ipix, ipix_from_ra_dec, make_pixdict
import healpy as hp
import h5py
import ast
from tqdm.auto import tqdm

from scipy.integrate import quad
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
import time
import sys

from concurrent.futures import as_completed, ThreadPoolExecutor
from multiprocessing import cpu_count


ZMIN = 1e-10
ZDRAW = 2
ZMAX = ZDRAW
COSMO = FlatLambdaCDM(H0=67.9, Om0=0.3065)

print('ALTERNATIVE HYPOTHESIS IS NORMALIZED UP TO Z=', ZMAX)

def dVdz_unnorm(z, cosmo):
    '''Assuming flat LCDM'''
    Omega_m = cosmo.Om0
    Omega_Lambda = 1 - Omega_m
    E_of_z = np.sqrt((1 + z)**3 * Omega_m + Omega_Lambda)
    com_vol = ((1 + z) * cosmo.angular_diameter_distance(z).value)**2 / E_of_z
    return com_vol

func = lambda z: dVdz_unnorm(z, COSMO)
NORM = quad(func, ZMIN, ZMAX)[0]

def dVdz_prior(z, norm=NORM, cosmo=COSMO):
    return dVdz_unnorm(z, cosmo) / norm



# def get_zprior_single_pix():
#     NotImplemented


# def get_zprior_full_sky():
#     NotImplemented


def get_offset(LOS_catalog):
    diction = eval(LOS_catalog.attrs['opts'])
    return diction["offset"]


def get_array(LOS_catalog, arr_name):
    offset = get_offset(LOS_catalog)

    arr = LOS_catalog[str(arr_name)][:]
    arr = np.exp(arr)
    arr -= offset

    return arr


def get_zprior(LOS_catalog, pixel_index):
    return get_array(LOS_catalog, str(pixel_index))


def unpack_LOS_catalog(LOS_catalog_path):
    LOS_catalog = h5py.File(LOS_catalog_path, 'r')
    temp = LOS_catalog.attrs['opts']
    catalog_opts = ast.literal_eval(temp.decode('utf-8'))
    nside = catalog_opts['nside']
    print(f'Chosen resolution nside: {nside}')
    z_array = LOS_catalog['z_array'][:]
    # zprior_full_sky = get_zprior_full_sky(LOS_catalog)  # TODO: This is needed in the denominator method of the llh class, is for later
    return LOS_catalog, nside, z_array, None


def load_posterior_samples(path, approximant):
    with h5py.File(path, 'r') as file:
        data = file[approximant]
        redshift = data['posterior_samples']['redshift']
        ra = data['posterior_samples']['ra']
        dec = data['posterior_samples']['dec']
        nsamps = len(ra)
    return redshift, ra, dec, nsamps


def single_event_fixpop_3dpos_likelihood(i,
                                         zprior_nside,
                                         posterior_samps_path, 
                                         field,
                                         interp_list,
                                         completeness,
                                         pixdict,
                                         n_mc_samps):

    # TODO: CHECK CONVERGENCE
    # result = []  
    # trials = [int(1e1), int(5e1), int(1e2), int(5e2), int(1e3), int(5e3), int(1e4), int(5e4)]
    # for n_mc_samps in trials:
    redshift, ra, dec, nsamps = load_posterior_samples(posterior_samps_path, approximant=field)

    idx_array = np.random.choice(np.arange(nsamps), size=n_mc_samps)

    mc_ra = ra[idx_array]
    mc_dec = dec[idx_array]
    mc_pix = ipix_from_ra_dec(nside=zprior_nside, ra=mc_ra, dec=mc_dec, nest=True)
    mc_redshift = redshift[idx_array]

    unique_pix = np.unique(mc_pix)  # Loop over the pixels that have been sampled
    cw_p_agn = 0
    cw_p_alt = 0
    for pix in unique_pix:
        z_samp = mc_redshift[mc_pix == pix]                                 # Find all redshift samples in the pixel
        interp = interp_list[pix]                                           # Get the interpolant of the LOS zprior in this pixel
        cmap_pix = pixdict[pix]                                             # Get the coarse resolution completeness map pixel at the location of the high resolution zprior pixel
        cw_p_agn += np.sum(interp(z_samp)) * completeness[cmap_pix]         # Evaluate zprior at sampled redshift
        cw_p_alt += np.sum(dVdz_prior(z_samp)) * completeness[cmap_pix]     # Same with alternative hypothesis due to (1 - cf_agn)p_alt = p_alt - f_agn * c p_alt term

        if np.isnan(cw_p_agn):
            print(f'Got NaNs in p_agn in pixel {pix} and GW {posterior_samps_path}.')
            print(f'zprior: {interp(z_samp)}')

    cw_p_agn /= n_mc_samps
    cw_p_alt /= n_mc_samps
    p_alt = np.sum(dVdz_prior(mc_redshift)) / n_mc_samps  # Also save unweighted value
    return i, cw_p_agn, cw_p_alt, p_alt


def main():

    ###############################################
    post_path = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/jsons/posterior_samples_mock_v7.json'
    with open(post_path) as f:
        posterior_samples_dictionary = json.load(f)

    # LOS_catalog_path = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/LOSzpriors/LOS_redshift_prior_mockcat_NAGN_100000_ZMAX_3_SIGMA_0.01_incomplete_v16_lenzarray_12000_zdraw_2.0_nside_16_pixel_index_None.hdf5'
    LOS_catalog_path = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/LOSzpriors/MOCK_LOS_redshift_prior_lenzarray_12000_zdraw_2.0_zmax_3.0_nside_32_nagn_100000_sigma_0.01_pixel_index_None.hdf5'
    c_map_path = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/maps/completeness_NAGN_100000_ZMAX_3_SIGMA_0.01_v17.fits'
    cmap_nside = 32
    posterior_samples_field = 'mock'
    n_mc_samps = int(1e4)
    ncpu = cpu_count()

    #################################################
    

    N_gws = len(posterior_samples_dictionary.items())

    # Calculate likelihood for each event
    LOS_catalog, nside, z_array, _ = unpack_LOS_catalog(LOS_catalog_path)
    cw_p_agn_arr = np.zeros(N_gws)
    cw_p_alt_arr = np.zeros(N_gws)
    p_alt_arr = np.zeros(N_gws)

    completeness = hp.read_map(c_map_path, nest=True)
    pixdict = make_pixdict(low_nside=cmap_nside, high_nside=nside)

    # Major computation time save: keep interpolated zpriors in memory
    interp_list = []
    print('Interpolating zprior in each pixel...')
    for pix in tqdm(range(hp.nside2npix(nside))):
        zprior = get_zprior(LOS_catalog, pix)
        interpolant = interp1d(z_array, zprior, bounds_error=False, fill_value=0)
        interp_list.append(interpolant)

        if np.isnan(interpolant(1)):
            sys.exit(f'Found NaNs in the zprior of pixel {pix}. Exiting...')

    with ThreadPoolExecutor(max_workers=ncpu) as executor:
        future_to_index = {executor.submit(
                                        single_event_fixpop_3dpos_likelihood, 
                                        i, 
                                        nside,
                                        posterior_samples_dictionary[key], 
                                        posterior_samples_field,
                                        interp_list,
                                        completeness,
                                        pixdict,
                                        n_mc_samps
                                    ): i for i, key in enumerate(posterior_samples_dictionary)
                                    }
        
        for future in tqdm(as_completed(future_to_index), total=N_gws):
            try:
                i, cw_p_agn, cw_p_alt, p_alt = future.result(timeout=60)
                cw_p_agn_arr[i], cw_p_alt_arr[i], p_alt_arr[i] = cw_p_agn, cw_p_alt, p_alt
            except Exception as e:
                print(f"Error processing event {future_to_index[future]}: {e}")
  
    LOS_catalog.close()

    np.save('cweighted_pagn_sky_v17', cw_p_agn_arr)
    np.save('cweighted_palt_sky_v17', cw_p_alt_arr)
    np.save('palt_sky_v17', p_alt_arr)

    return _


if __name__ == '__main__':
    main()

    # for i, (key, _) in tqdm(enumerate(posterior_samples_dictionary.items()), total=len(posterior_samples_dictionary.items())):

    #     posterior_samps_path = posterior_samples_dictionary[key]
    #     field = posterior_samples_field  # TODO: allow for json such that you can do `fields[key]` cf. gwcosmo

    #     cw_p_agn, cw_p_alt, p_alt = single_event_fixpop_3dpos_likelihood(
    #                                                     nside=nside, 
    #                                                     posterior_samps_path=posterior_samps_path, 
    #                                                     field=field,
    #                                                     interp_list=interp_list,
    #                                                     completeness=completeness,
    #                                                     pixdict=pixdict
    #                                                 ) 
        
    #     cw_p_agn_arr[i], cw_p_alt_arr[i], p_alt_arr[i] = cw_p_agn, cw_p_alt, p_alt