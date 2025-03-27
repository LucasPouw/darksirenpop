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
    return np.random.uniform(low=0, high=1, size=npix)[pix]


# Half hemisphere completeness - just gives a bunch of non-informative GWs
# def c_omega(pix, nside):
#     '''Completeness as a function of sky position, given by the pixel index'''
#     npix = hp.nside2npix(nside)
#     completeness = np.zeros_like(pix)
#     completeness[pix < npix // 2] = 1
#     return completeness


def make_new_catalog(catalog_path, nside, norm_map_path, c_map_path):
    print('--- MAKING NEW CATALOG ---')

    with h5py.File(catalog_path, 'r') as old_catalog:
        with h5py.File(catalog_path[:-5] + '_incomplete.hdf5', 'w') as new_catalog:
            
            cols = list(old_catalog)
            for col in cols:
                if col in INC_CAT_COLS:
                    continue

                old_catalog.copy(col, new_catalog)
            
            # SETTING COMPLETENESS PER PIXEL AND SAMPLING AGN BASED ON THAT
            # pix = ipix_from_ra_dec(nside, old_catalog['complete_catalog']['ra'][()], old_catalog['complete_catalog']['dec'][()], nest=True)
            # completeness = c_omega(pix, nside)  # Completenesses in all pixels
            # detected = completeness > np.random.uniform(low=0, high=1, size=len(pix))

            # SETTING COMPLETENESSS FRACTION AND CALCULATING TRUE COMPLETENESS AFTER SAMPLING
            nagn = len(old_catalog['ra'][()])
            detected = np.zeros(nagn, dtype=bool)
            detected[np.random.choice(np.arange(nagn), size=int(0.75 * nagn))] = True  # TODO: hard-coded xdddd
            
            new_catalog['detected'] = detected
            for col in INC_CAT_COLS:
                if col == 'detected':
                    continue
                new_catalog[col] = old_catalog['complete_catalog'][col][detected]

    with h5py.File(catalog_path[:-5] + '_incomplete.hdf5', 'r') as new_catalog:
        print('Making norm map from INCOMPLETE catalog...')
        npix = hp.nside2npix(nside)
        pix_indices = hp.ang2pix(nside, np.pi/2 - new_catalog['dec'][()], new_catalog['ra'][()], nest=True)
        healpix_map = np.zeros(npix)
        np.add.at(healpix_map, pix_indices, 1)  # Count number of AGN in each pixel

        hp.fitsfunc.write_map(norm_map_path, healpix_map, nest=True, overwrite=True)


        # FOR THE COMPLETENESS MAP WE NEED TO DIVIDE THE INCOMPLETE CATALOG NORM MAP BY THE COMPLETE ONE
        pix_complete = hp.ang2pix(nside, np.pi/2 - new_catalog['complete_catalog']['dec'][()], new_catalog['complete_catalog']['ra'][()], nest=True)
        hp_map_complete = np.zeros(npix)
        np.add.at(hp_map_complete, pix_complete, 1)

        completeness = healpix_map / hp_map_complete
        # plt.figure()
        # hp.mollview(completeness, title="Completeness", coord=["C"])
        # hp.graticule()
        # plt.savefig('completeness3.pdf')

        hp.fitsfunc.write_map(c_map_path, completeness, nest=True, overwrite=True)  # Save completeness map for use in likelihood


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    nside = 32
    pix = np.arange(hp.nside2npix(nside))
    c = c_omega(pix, nside)

    cat_path = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs/mockcat_NAGN_100000_ZMAX_3_SIGMA_0.01.hdf5'
    inccat_path = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs/mockcat_NAGN_100000_ZMAX_3_SIGMA_0.01_incomplete.hdf5'
    norm_map_path = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/maps/mocknorm_NAGN_100000_ZMAX_3_SIGMA_0.01_incomplete.fits'
    c_map_path = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/maps/completeness_NAGN_100000_ZMAX_3_SIGMA_0.01.fits'
    # make_new_catalog(cat_path, nside, norm_map_path, c_map_path)

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
