import numpy as np
from scipy.interpolate import interp1d
import json
from utils import ra_dec_from_ipix, ipix_from_ra_dec, make_pixdict, make_nice_plots
from default_arguments import *
import healpy as hp
import h5py
import ast
from tqdm.auto import tqdm
from scipy.integrate import quad
import matplotlib.pyplot as plt
import time
import sys
from concurrent.futures import as_completed, ThreadPoolExecutor
from multiprocessing import cpu_count
from scipy.stats import gaussian_kde
from agn_catalogs_and_priors.pixelated_catalog import load_catalog_from_path

make_nice_plots()

ZMIN = 1e-10
ZDRAW = 2
ZMAX = ZDRAW
COSMO = DEFAULT_COSMOLOGY

print('ALTERNATIVE HYPOTHESIS IS NORMALIZED UP TO Z=', ZDRAW)

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


def load_posterior_samples(path, approximant):
    with h5py.File(path, 'r') as file:
        data = file[approximant]
        redshift = data['posterior_samples']['redshift']
        ra = data['posterior_samples']['ra']
        dec = data['posterior_samples']['dec']
        nsamps = len(ra)
    return redshift, ra, dec, nsamps


def single_event_fixpop_no_agn_err_3dpos_likelihood(index,
                                                    catalog_nside,
                                                    posterior_samps_path, 
                                                    field,
                                                    galaxy_catalog,
                                                    completeness,
                                                    pixdict,
                                                    n_mc_samps,
                                                    kde_thresh=100):

    redshift, ra, dec, nsamps = load_posterior_samples(posterior_samps_path, approximant=field)
    post_pix = ipix_from_ra_dec(nside=catalog_nside, ra=ra, dec=dec, nest=True)

    idx_array = np.random.choice(np.arange(nsamps), size=n_mc_samps)

    mc_ra = ra[idx_array]
    mc_dec = dec[idx_array]
    mc_pix = ipix_from_ra_dec(nside=catalog_nside, ra=mc_ra, dec=mc_dec, nest=True)
    mc_redshift = redshift[idx_array]

    # start = time.time()
    unique_pix, counts = np.unique(mc_pix, return_counts=True)  # Loop over the pixels that have been sampled
    cw_p_agn = 0  # int p(z, Omega | d) * f_agn * f_c(Omega) * p_agn(z | Omega) dz dOmega
    cw_p_alt = 0  # int p(z, Omega | d) * f_agn * f_c(Omega) * p_alt(z) dz dOmega
    # print(f'Got {len(unique_pix)} pix: {counts}')
    for j, pix in enumerate(unique_pix):
        cmap_pix = pixdict[pix]
        all_z_in_pix = redshift[post_pix == pix]
        if len(all_z_in_pix) >= kde_thresh:
            subcatalog = galaxy_catalog.select_pixel(catalog_nside, pix, nested=True)  # Load in the galaxy catalogue data from this pixel
            subcatalog = subcatalog[subcatalog['redshift'] < ZDRAW]  # Only consider AGN below z_draw
            if len(subcatalog) != 0:
                redshift_kde = gaussian_kde(all_z_in_pix)
                cw_p_agn += np.sum(redshift_kde(subcatalog['redshift'])) * counts[j] * completeness[cmap_pix] / len(subcatalog)

        mc_z_samp_in_pix = mc_redshift[mc_pix == pix]
        cw_p_alt += np.sum(dVdz_prior(mc_z_samp_in_pix)) * completeness[cmap_pix]

        if np.isnan(cw_p_agn):
            print(f'Got NaNs in p_agn in pixel {pix} and GW {posterior_samps_path}.')

        # print(len(all_z_in_pix))
        # print(len(subcatalog))

        # if len(all_z_in_pix) > 15000:
        #     xmin = 0
        #     xmax = ZDRAW
        #     z = np.linspace(xmin, xmax, 500)
        #     pz_omega_agn = redshift_kde(z)
        #     pz_omega_alt = dVdz_prior(z)
        #     ymin = 0
        #     ymax = max(np.max(pz_omega_agn), np.max(pz_omega_alt)) * 1.1

        #     print(subcatalog['redshift'], 'redhsifts')

        #     plt.figure()
        #     plt.title(f'1k AGN, {hp.nside2npix(catalog_nside)} pixels, ' + r'$\sigma_{d_{L}}=0.2$')
        #     plt.plot(z, pz_omega_agn, label=r'$p(z | \Omega_{j}, d_{\rm GW})$', linewidth=2, color='blue')
        #     plt.plot(z, pz_omega_alt, label=r'$p_{\rm alt}(z)$', linewidth=2, color='red')
        #     plt.vlines(subcatalog['redshift'], ymin, ymax, linestyle='dashed', linewidth=2, color='black', label=r'$p_{\rm agn}(z | \Omega_{j})$', zorder=-1)
        #     plt.xlabel('Redshift')
        #     plt.ylabel('Probability density')
        #     plt.legend(fontsize=16, loc='upper left')
        #     plt.tight_layout()
        #     plt.ylim(ymin, ymax)
        #     plt.xlim(xmin, xmax)
        #     plt.savefig(os.path.join(sys.path[0], f'output/plots/gw_kde_20percent_{hp.nside2npix(catalog_nside)}pix.pdf'), bbox_inches='tight')
        #     plt.show()

        #     # sys.exit('QUITTING')
    # print(len(unique_pix))           

    cw_p_agn /= n_mc_samps
    cw_p_alt /= n_mc_samps
    p_alt = np.sum(dVdz_prior(mc_redshift)) / n_mc_samps  # int p(z, Omega | d) * p_alt(z) dz dOmega
    # print('that took', time.time() - start)
    if cw_p_agn < cw_p_alt:
        print('ALT', cw_p_agn, cw_p_alt, p_alt)
    else:
        print('AGN', cw_p_agn, cw_p_alt, p_alt)
    
    return index, cw_p_agn, cw_p_alt, p_alt


def main():

    ############################################### INPUTS ###############################################
    post_path = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/jsons/dl_20percent_dz_00percent_nagn_1e3.json'  # '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/jsons/posterior_samples_mock_v7.json'
    with open(post_path) as f:
        posterior_samples_dictionary = json.load(f)

    catalog_path = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs/mockcat_NAGN_1000_ZMAX_3_SIGMA_0.hdf5'
    catalog_nside = 16
    c_map_path = None
    outfilename = 'dl_20percent_dz_00percent_nagn_1e3' # f'zero_error_correct_1percent_{hp.nside2npix(catalog_nside)}pix'
    cmap_nside = 1
    posterior_samples_field = 'mock'
    n_mc_samps = int(1e4)
    ncpu = cpu_count()

    #######################################################################################################
    

    N_gws = len(posterior_samples_dictionary.items())
    agn_catalog = load_catalog_from_path(name='MOCK', catalog_path=catalog_path)

    # Calculate likelihood for each event
    cw_p_agn_arr = np.zeros(N_gws)
    cw_p_alt_arr = np.zeros(N_gws)
    p_alt_arr = np.zeros(N_gws)

    if c_map_path is not None:
        cmap = hp.read_map(c_map_path, nest=True)
        completeness = hp.ud_grade(cmap, nside_out=cmap_nside, order_in='NESTED', order_out='NESTED')
    else:
        cmap = None
        completeness = np.ones(hp.nside2npix(catalog_nside))

    pixdict = make_pixdict(low_nside=cmap_nside, high_nside=catalog_nside)

    # FOR TESTING
    # for i, key in enumerate(posterior_samples_dictionary):
    #     single_event_fixpop_no_agn_err_3dpos_likelihood(i, 
    #                                                     catalog_nside,
    #                                                     posterior_samples_dictionary[key], 
    #                                                     posterior_samples_field,
    #                                                     agn_catalog,
    #                                                     completeness,
    #                                                     pixdict,
    #                                                     n_mc_samps)
    #     break

    with ThreadPoolExecutor(max_workers=ncpu) as executor:
        future_to_index = {executor.submit(
                                        single_event_fixpop_no_agn_err_3dpos_likelihood, 
                                        i, 
                                        catalog_nside,
                                        posterior_samples_dictionary[key], 
                                        posterior_samples_field,
                                        agn_catalog,
                                        completeness,
                                        pixdict,
                                        n_mc_samps
                                    ): i for i, key in enumerate(posterior_samples_dictionary)
                                    }
        
        # for future in tqdm(as_completed(future_to_index), total=N_gws):
        for future in as_completed(future_to_index):
            try:
                i, cw_p_agn, cw_p_alt, p_alt = future.result(timeout=60)
                cw_p_agn_arr[i], cw_p_alt_arr[i], p_alt_arr[i] = cw_p_agn, cw_p_alt, p_alt
            except Exception as e:
                print(f"Error processing event {future_to_index[future]}: {e}")

    np.save(os.path.join(sys.path[0], f'cweighted_pagn_sky_{outfilename}'), cw_p_agn_arr)
    np.save(os.path.join(sys.path[0], f'cweighted_palt_sky_{outfilename}'), cw_p_alt_arr)
    np.save(os.path.join(sys.path[0], f'palt_sky_{outfilename}'), p_alt_arr)


if __name__ == '__main__':
    main()
