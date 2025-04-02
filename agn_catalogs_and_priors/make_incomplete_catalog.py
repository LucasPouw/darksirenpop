"""
AGN catalog .hdf5 files already have a uniformly incomplete catalog, 
but we can sample a new incomplete catalog from the truths with this code.
In particular, we want the completeness to vary with sky position and distance.
"""

import numpy as np
from darksirenpop.utils import ra_dec_from_ipix, ipix_from_ra_dec
import healpy as hp
import h5py


# These need to be replaced in the new incomplete catalog
INC_CAT_COLS = ['comoving_distance', 'redshift', 'redshift_error', 'ra', 'dec', 'luminosity_distance', 'detected']


def c_z(redshift):
    '''Completeness as a function of redshift'''
    return np.ones_like(redshift)


def c_omega(pix, nside):
    '''Random completeness'''
    npix = hp.nside2npix(nside)
    return np.random.uniform(low=0, high=1, size=npix)


def make_new_catalog(catalog_path, nside, inccat_path, norm_map_path, c_map_path):
    print('--- MAKING NEW CATALOG ---')

    with h5py.File(catalog_path, 'r') as old_catalog:
        with h5py.File(inccat_path, 'w') as new_catalog:
            
            cols = list(old_catalog)
            for col in cols:
                if col in INC_CAT_COLS:
                    continue

                old_catalog.copy(col, new_catalog)
            
            # SETTING COMPLETENESS PER PIXEL AND SAMPLING AGN BASED ON THAT
            pix = ipix_from_ra_dec(nside, old_catalog['complete_catalog']['ra'][()], old_catalog['complete_catalog']['dec'][()], nest=True)
            completeness = c_omega(pix, nside)  # Completenesses in all pixels
            detected = completeness[pix] > np.random.uniform(low=0, high=1, size=len(pix))
            # hp.fitsfunc.write_map(c_map_path, completeness, nest=True, overwrite=True)  # Save completeness map for use in likelihood


            # detected = 0.5 > np.random.uniform(low=0, high=1, size=len(pix))  # Give each AGN a 50% chance of being detected

            
            new_catalog['detected'] = detected
            for col in INC_CAT_COLS:
                if col == 'detected':
                    continue
                new_catalog[col] = old_catalog['complete_catalog'][col][detected]

    with h5py.File(inccat_path, 'r') as new_catalog:
        print('Making norm map from INCOMPLETE catalog...')
        npix = hp.nside2npix(nside)
        pix_indices = hp.ang2pix(nside, np.pi * 0.5 - new_catalog['dec'][()], new_catalog['ra'][()], nest=True)
        healpix_map = np.zeros(npix)
        np.add.at(healpix_map, pix_indices, 1)  # Count number of AGN in each pixel

        hp.fitsfunc.write_map(norm_map_path, healpix_map, nest=True, overwrite=True)


        below_zdraw = new_catalog['complete_catalog']['redshift_true'][()] < 2
        below_zdraw_pix = hp.ang2pix(nside, np.pi * 0.5 - new_catalog['complete_catalog']['dec'][below_zdraw], new_catalog['complete_catalog']['ra'][below_zdraw], nest=True)
        below_zdraw_map = np.zeros(npix)
        np.add.at(below_zdraw_map, below_zdraw_pix, 1)
        
        below_zdraw_and_detected = below_zdraw & new_catalog['detected']
        detected_below_zdraw_pix = hp.ang2pix(nside, np.pi * 0.5 - new_catalog['complete_catalog']['dec'][below_zdraw_and_detected], new_catalog['complete_catalog']['ra'][below_zdraw_and_detected], nest=True)
        detected_and_below_zdraw_map = np.zeros(npix)
        np.add.at(detected_and_below_zdraw_map, detected_below_zdraw_pix, 1)

        cmap = np.where(below_zdraw_map == 0., 1, detected_and_below_zdraw_map / below_zdraw_map)  # If the true pixel is empty, the pixel is complete.
        hp.fitsfunc.write_map(c_map_path, cmap, nest=True, overwrite=True)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    nside = 1
    pix = np.arange(hp.nside2npix(nside))
    c = c_omega(pix, nside)

    cat_path = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs/mockcat_NAGN_100000_ZMAX_3_SIGMA_0.01.hdf5'
    inccat_path = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs/mockcat_NAGN_100000_ZMAX_3_SIGMA_0.01_incomplete_v24.hdf5'
    norm_map_path = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/maps/mocknorm_NAGN_100000_ZMAX_3_SIGMA_0.01_incomplete_v24.fits'
    c_map_path = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/maps/completeness_NAGN_100000_ZMAX_3_SIGMA_0.01_v24.fits'
    make_new_catalog(cat_path, nside, inccat_path, norm_map_path, c_map_path)

    # plt.figure()
    # hp.mollview(cmap, title="Completeness", coord=["C"])
    # hp.graticule()
    # plt.savefig('completeness2.pdf')
    # plt.show()

    # def make_norm_map(ra, dec, nside):
    #     npix = hp.nside2npix(nside)  # Total number of pixels
    #     pix_indices = hp.ang2pix(nside, np.pi/2 - dec, ra, nest=True)
    #     healpix_map = np.zeros(npix)
    #     np.add.at(healpix_map, pix_indices, 1)  # Count occurrences
    #     return healpix_map
    

    # with h5py.File(inccat_path, 'r') as f:
    #     ra = f['ra'][()]
    #     dec = f['dec'][()]
    #     # z = f['redshift'][()]
    #     # zerr = f['redshift_error'][()]
    
    # hpmap = make_norm_map(ra, dec, nside)

    # plt.figure()
    # hp.mollview(hpmap, title="Incomplete catalog", coord=["C"])
    # hp.graticule()
    # # plt.savefig('incomplete_map.pdf')
    # plt.show()
