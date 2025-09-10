"""
Module to compute and handle skymaps
Rachel Gray, Ignacio Magana, Archisman Ghosh, Ankan Sur

Lucas Pouw
"""
import numpy as np
import scipy.stats
import healpy as hp
from ligo.skymap.io.fits import read_sky_map
import sys
from astropy.table import Table
import pandas as pd
from tqdm import tqdm
from astropy.coordinates import SkyCoord


# def ra_dec_from_ipix(nside, ipix, nest=False):
#     """RA and dec from HEALPix index"""
#     (theta, phi) = hp.pix2ang(nside, ipix, nest=nest)
#     return (phi, np.pi * 0.5 - theta)


# def ipix_from_ra_dec(nside, ra, dec, nest=False):
#     """HEALPix index from RA and dec"""
#     (theta, phi) = (np.pi * 0.5 - dec, ra)
#     return hp.ang2pix(nside, theta, phi, nest=nest)


# class Skymap:

#     """
#     Read a FITS file and return interpolation kernels on the sky.
#     TODO: Rework to use ligo.skymap
#     """

#     def __init__(self, filename):
#         """
#         Input parameters:
#         - filename : FITS file to load from
#         """
#         # (self.prob, self.distmu, self.distsigma, self.distnorm), metadata = \
#             # read_sky_map(filename, distances=True, moc=False, nest=True)
#         # Setting nest=True in read_sky_map ensures that the return is
#         # in the nested pixel order.

#         m = Table.read(filename, format='fits', memmap=True)  # memmap=True avoids memory filling up, all metadata stuff from read_sky_map() not needed

#         if m.meta['ORDERING'] != 'NESTED':
#             print(f"\nWRONG ORDERING DETECTED IN FILE {filename} BUT TOO LAZY TO FIX PRIOR TO DETECTION: {m.meta['ORDERING']}\n")
#             sys.exit(1)

#         self.prob, self.distmu, self.distsigma, self.distnorm = m['PROB'], m['DISTMU'], m['DISTSIGMA'], m['DISTNORM']

#         self.nested = True

#         self.npix = len(self.prob)
#         self.nside = hp.npix2nside(self.npix)
#         colat, self.ra = hp.pix2ang(self.nside, range(len(self.prob)),
#                                     nest=self.nested)
#         self.dec = np.pi * 0.5 - colat


#     def probability(self, ra, dec, dist):
#         """
#         returns probability density at given ra, dec, dist
#         p(ra,dec) * p(dist | ra,dec )
#         RA, dec : radians
#         dist : Mpc
#         """
#         theta = np.pi * 0.5 - dec
#         # Step 1: find 4 nearest pixels
#         (pixnums, weights) = hp.get_interp_weights(self.nside, theta, ra, nest=self.nested, lonlat=False)
#         # print(pixnums.shape)

#         # print(dist, [self.distmu[i] for i in pixnums])

#         dist_pdfs = [scipy.stats.norm(loc=self.distmu[i], scale=self.distsigma[i]) for i in pixnums]
#         # Step 2: compute p(ra,dec)
#         # p(ra, dec) = sum_i weight_i p(pixel_i)
#         probvals = np.array([self.distnorm[pixel] * dist_pdfs[i].pdf(dist) for i, pixel in enumerate(pixnums)])  # TODO: check if change self.distnorm[i] -> self.distnorm[pixel] is correct
#         skyprob = self.prob[pixnums]
#         p_ra_dec = np.sum(weights * probvals * skyprob)

#         pvals = np.array([self.distnorm[i] * dist_pdfs[i].pdf(dist) for i, pixel in enumerate(pixnums)])  # TODO: check if change self.distnorm[i] -> self.distnorm[pixel] is correct
#         skyp = self.prob[pixnums]
#         old_p = np.sum(weights * pvals * skyp)

#         # print(probvals)
#         # print(skyprob)
#         return p_ra_dec, old_p


#     def skyprob(self, ra, dec):
#         """
#         Return the probability of a given sky location
#         ra, dec: radians
#         """
#         ipix_gal = self.indices(ra, dec)
#         return self.prob[ipix_gal]


#     def indices(self, ra, dec):
#         """
#         Return the index of the skymap pixel that contains the
#         coordinate ra,dec
#         """
#         return hp.ang2pix(self.nside, np.pi * 0.5 - dec, ra, nest=self.nested)


#     def above_percentile(self, thresh, nside):
#         """Returns indices of array within the given threshold
#         credible region."""
#         prob = self.prob
#         pixorder = 'NESTED' if self.nested else 'RING'
#         if nside != self.nside:
#             new_prob = hp.pixelfunc.ud_grade(self.prob, nside,
#                                              order_in=pixorder,
#                                              order_out=pixorder)
#             prob = new_prob / np.sum(new_prob)  # renormalise

#         #  Sort indicies of sky map
#         ind_sorted = np.argsort(-prob)
#         #  Cumulatively sum the sky map
#         cumsum = np.cumsum(prob[ind_sorted])
#         #  Find indicies contained within threshold area
#         lim_ind = np.where(cumsum > thresh)[0][0]
#         return ind_sorted[:lim_ind], prob[ind_sorted[:lim_ind]]


#     def samples_within_region(self, ra, dec, thresh, nside=None):
#         """Returns boolean array of whether galaxies are within
#         the sky map's credible region above the given threshold"""
#         if nside is None:
#             nside = self.nside
#         skymap_ind = self.above_percentile(thresh, nside=nside)[0]
#         samples_ind = hp.ang2pix(nside, np.pi * 0.5 - dec, ra, nest=self.nested)
#         return np.in1d(samples_ind, skymap_ind)


#     def region_with_sample_support(self, ra, dec, thresh, nside=None):
#         """
#         Finds fraction of sky with catalogue support, and corresponding
#         fraction of GW sky probability
#         """
#         if nside is None:
#             nside = self.nside
#         skymap_ind, skymap_prob = self.above_percentile(thresh, nside=nside)
#         samples_ind = hp.ang2pix(nside, np.pi * 0.5 - dec, ra, nest=self.nested)
#         ind = np.in1d(skymap_ind, samples_ind)
#         fraction_of_sky = np.count_nonzero(ind)/hp.nside2npix(nside)
#         GW_prob_in_fraction_of_sky = np.sum(skymap_prob[ind])
#         return fraction_of_sky, GW_prob_in_fraction_of_sky


#     def pixel_split(self, ra, dec, nside):
#         """
#         For a list of galaxy ra and decs, return a dictionary identifying the
#         index of the galaxies in each pixel of a healpy map of resolution nside

#         Parameters
#         ----------
#         ra, dec : (ndarray, ndarray)
#             Coordinates of the sources in radians.

#         nside : int
#             HEALPix nside of the target map

#         Return
#         ------
#         dicts : dictionary
#             dictionary of galaxy indices in each pixel

#         dicts[idx] returns the indices of each galaxy in skymap pixel idx.
#         """

#         # The number of pixels based on the chosen value of nside
#         npix = hp.nside2npix(nside)

#         # convert to theta, phi
#         theta = np.pi * 0.5 - dec
#         phi = ra

#         # convert to HEALPix indices (each galaxy is assigned to a single healpy pixel)
#         indices = hp.ang2pix(nside, theta, phi, nest=self.nested)

#         # sort the indices into ascending order
#         idx_sort = np.argsort(indices)
#         sorted_indices = indices[idx_sort]

#         # idx: the healpy index of each pixel containing a galaxy (arranged in ascending order)
#         # idx_start: the index of 'sorted_indices' corresponding to each new pixel
#         # count: the number of galaxies in each pixel
#         idx, idx_start, count = np.unique(sorted_indices,return_counts=True,return_index=True)

#         # splits indices into arrays - 1 per pixel
#         res = np.split(idx_sort, idx_start[1:])

#         keys = idx
#         values = res
#         dicts = {}
#         for i,key in enumerate(keys):
#             dicts[key] = values[i]

#         return dicts


#     def lineofsight_posterior_dl (self, ra, dec) :
#         """
#         Estimating distance posterior from gw skymap for a given ra nad dec

#         Parameters
#         ----------
#         ra, dec : (float, float)
#             Sky coordinates of the source in radians.

#         Return
#         ------
#         distmin, distmax, distance posterior
#             minimum and maximum distance correspond to 5 sigma interval
#             distance posterior (scipy.stats.norm)

#         """

#         pix_los = ipix_from_ra_dec (self.nside, ra, dec, nest=self.nested)
#         # mu = self.distmu[(self.distmu < np.inf) & (self.distmu > 0)]
#         distmu_los = self.distmu [pix_los]
#         distsigma_los = self.distsigma [pix_los]
#         distmin = distmu_los - 5*distsigma_los
#         distmax = distmu_los + 5*distsigma_los
#         posterior_dl = scipy.stats.norm (distmu_los, distsigma_los)
#         return distmin, distmax, posterior_dl
    


# def make_dataframe(r_lum, ra, dec, z):
    
#     """
#     Assumes:
#     r_lum and r_com in Mpc
#     theta and phi in rad
#     """

#     coords = SkyCoord(ra=ra * u.rad,
#                       dec=dec * u.rad)
    
#     # Rounding minutes and seconds and saving under 1 float
#     RA_ms = np.round(coords.ra.hms[1] + coords.ra.hms[2] / 60, 2)
#     Dec_ms = np.round(np.abs(coords.dec.dms[1] + np.abs(coords.dec.dms[2]) / 60), 2)
    
#     # Rounding hours and days
#     RA_h = coords.ra.hms[0].astype(int)
#     Dec_d = coords.dec.dms[0].astype(int)
    
#     RA_hms = []
#     Dec_dms = []
#     for i in tqdm(range(len(z))):
        
#         RA_hms.append(f'+{RA_h[i]}:{RA_ms[i]}')
        
#         if Dec_d[i] >= 0:  # Handling + and - sign of Dec
#             Dec_dms.append(f'+{Dec_d[i]}:{Dec_ms[i]}')
#         else:
#             Dec_dms.append(f'{Dec_d[i]}:{Dec_ms[i]}')
    
#     df = pd.DataFrame()
    
#     df['ID'] = np.arange(len(z))
#     df['RA'] = RA_hms
#     df['Dec'] = Dec_dms
#     print('IF WRONG RESULT, TRY KPC')
#     df['LumDist'] = np.array(r_lum)  # df['LumDist'] = np.array(r_lum * 1000)  # Convert from Mpc to kpc
#     df['index1'] = np.full(len(z), 1)
#     df['index2'] = np.full(len(z), 1)
#     df['index3'] = np.full(len(z), 1)
#     df['index4'] = np.full(len(z), 1)
    
#     return df

def sample_spherical_angles(n_samps=1):
        theta = np.arccos(np.random.uniform(size=n_samps, low=-1, high=1))  # Cosine is uniformly distributed between -1 and 1 -> cos between 0 and pi
        phi = 2 * np.pi * np.random.uniform(size=n_samps)  # Draws phi from 0 to 2pi
        return theta, phi

def uniform_shell_sampler(rmin, rmax, n_samps):
    r = ( np.random.uniform(size=n_samps, low=rmin**3, high=rmax**3) )**(1/3)
    theta, phi = sample_spherical_angles(n_samps)
    return r, theta, phi


if __name__ == '__main__':

    '''
    TODO:
    DONE 1. Analysis with mock skymaps and ligo.skymap.postprocess.crossmatch blabla, just to see if my skymaps are right
    2. Try to recreate the numbers manually with the lumdist posteriors in each LOS
    3. Do the tests with scattered catalogs and try to compensate
    '''

    import warnings
    warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
    import h5py
    from default_arguments import DEFAULT_COSMOLOGY as COSMO
    from utils import fast_z_at_value
    import astropy.units as u
    import os
    from ligo.skymap.postprocess import crossmatch
    from ligo.skymap.bayestar import rasterize, derasterize
    from ligo.skymap.moc import nest2uniq
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

    invalid_files = [
    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_0_00005.fits.gz",
    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_0_00007.fits.gz",
    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_0_00022.fits.gz",
    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_0_00023.fits.gz",
    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_0_00033.fits.gz",
    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_0_00034.fits.gz",

    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_1_00011.fits.gz",
    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_1_00033.fits.gz",

    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_2_00012.fits.gz",
    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_2_00019.fits.gz",
    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_2_00029.fits.gz",
    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_2_00035.fits.gz",

    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_3_00005.fits.gz",
    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_3_00022.fits.gz",
    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_3_00038.fits.gz",

    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_4_00003.fits.gz",
    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_4_00031.fits.gz",
    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_4_00032.fits.gz",
    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_4_00033.fits.gz",

    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_5_00009.fits.gz",
    "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_5_00031.fits.gz",
]


    os.environ["OMP_NUM_THREADS"] = "1"

    posterior_fname = 'new_ligoskymap_posteriors'

    N_TRUE_FAGNS = 6
    BATCH = 40
    SKYMAP_CL = 0.999

    AGN_ZERROR = 0.1

    CALC_LOGLLH_AT_N_POINTS = 1000
    LOG_LLH_X_AX = np.linspace(0.0001, 0.9999, CALC_LOGLLH_AT_N_POINTS)
    USE_N_AGN_EVENTS = np.arange(0, BATCH + 1, int(BATCH / (N_TRUE_FAGNS - 1)), dtype=np.int32)
    TRUE_FAGNS = USE_N_AGN_EVENTS / BATCH

    ZMIN = 1e-4
    ZMAX = 1.5  # p_rate(z > ZMAX) = 0
    
    COMDIST_MIN = COSMO.comoving_distance(ZMIN).value
    COMDIST_MAX = COSMO.comoving_distance(ZMAX).value  # Maximum comoving distance in Mpc
    
    VOLUME = 4 / 3 * np.pi * COMDIST_MAX**3

    N_TRIALS = 1
    posteriors = np.zeros((N_TRIALS, CALC_LOGLLH_AT_N_POINTS, N_TRUE_FAGNS))
    for trial_idx in range(N_TRIALS):

        log_llh = np.zeros((len(LOG_LLH_X_AX), N_TRUE_FAGNS))
        for fagn_idx in range(N_TRUE_FAGNS):

            # fagn_idx = 5

            with h5py.File(f'/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/catalogs_moc/mockcat_0_{fagn_idx}.hdf5', 'r') as catalog:
                
                agn_ra = catalog['ra'][()]
                agn_dec = catalog['dec'][()]
                agn_rcom = catalog['comoving_distance'][()]

                ### FOR TESTING, ADD UNCORRELATED AGN TO THE CATALOG: S_AGN -> S_ALT SHOULD BE SEEN!! ###
                new_rcom, new_theta, new_phi = uniform_shell_sampler(COMDIST_MIN, COMDIST_MAX, 9900)
                agn_ra = np.append(agn_ra, new_phi)
                agn_dec = np.append(agn_dec, np.pi * 0.5 - new_theta)
                agn_rcom = np.append(agn_rcom, new_rcom)
                #########################################################

                true_agn_redshift = fast_z_at_value(COSMO.comoving_distance, agn_rcom * u.Mpc)

                if not AGN_ZERROR:
                    obs_agn_redshift = true_agn_redshift
                else:
                    obs_agn_redshift = np.random.normal(loc=true_agn_redshift, scale=AGN_ZERROR, size=len(agn_ra))
                
                obs_agn_rlum = COSMO.luminosity_distance(obs_agn_redshift).value
                
                # cat_coord = SkyCoord(agn_ra * u.rad, agn_dec * u.rad, agn_rlum * u.Mpc)

            
            ###################### SINGLE-THREAD ######################

            # S_alt = np.zeros(BATCH)
            # S_agn = np.zeros(BATCH)
            # for gw_idx in range(BATCH):
            #     filename = f"/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_{fagn_idx}_{gw_idx:05d}.fits.gz"

            #     skymap = read_sky_map(filename, moc=True)
            #     print(f'\nLoaded file: {filename}')

            #     # This function calculates the evidence for each hypothesis by integrating in redshift-space
            #     sagn, salt = crossmatch(sky_map=skymap,
            #                             agn_ra=agn_ra, 
            #                             agn_dec=agn_dec, 
            #                             agn_lumdist=obs_agn_rlum, 
            #                             agn_redshift=obs_agn_redshift,
            #                             agn_redshift_err=AGN_ZERROR,
            #                             skymap_cl=SKYMAP_CL)
                
            #     S_agn[gw_idx] = sagn
            #     S_alt[gw_idx] = salt

            #     print(sagn, salt)

            #########################################################


            ###################### THREADING ######################

            def process_gw(gw_idx):
                filename = f"/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_{fagn_idx}_{gw_idx:05d}.fits.gz"
                skymap = read_sky_map(filename, moc=True)
                print(f'\nLoaded file: {filename}')

                sagn, salt = crossmatch(
                    sky_map=skymap,
                    agn_ra=agn_ra, 
                    agn_dec=agn_dec, 
                    agn_lumdist=obs_agn_rlum, 
                    agn_redshift=obs_agn_redshift,
                    agn_redshift_err=AGN_ZERROR,
                    skymap_cl=SKYMAP_CL
                )
                return gw_idx, sagn, salt

            # multithread loop
            S_agn = np.zeros(BATCH)
            S_alt = np.zeros(BATCH)
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_gw, gw_idx) for gw_idx in range(BATCH)]
                for future in as_completed(futures):
                    gw_idx, sagn, salt = future.result()
                    S_agn[gw_idx] = sagn
                    S_alt[gw_idx] = salt
                    print(sagn, salt)
            
            #########################################################

            S_agn = S_agn[~np.isnan(S_agn)]
            S_alt = S_alt[~np.isnan(S_alt)]

            print(f'\n--- AFTER CROSSMATCHING THERE ARE {len(S_agn)} GWS LEFT ---\n')

            loglike = np.log(S_agn[:,None] * SKYMAP_CL * LOG_LLH_X_AX[None,:] + S_alt[:,None] * (1 - SKYMAP_CL * LOG_LLH_X_AX[None,:]))
            log_llh[:,fagn_idx] = np.sum(loglike, axis=0)  # sum over all GWs


            ### For using ligo's own crossmatch version ###

            # S_alt = np.tile(SKYMAP_CL, BATCH)
            # S_agn = np.zeros(BATCH)
            # for gw_idx in range(BATCH):
            #     filename = f"/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/skymaps_moc/skymap_0_{fagn_idx}_{gw_idx:05d}.fits.gz"
                
            #     if filename in invalid_files:
            #          print('Invalid file:', filename)
            #          S_agn[gw_idx] = np.nan
            #          S_alt[gw_idx] = np.nan
            #          continue                     

            #     skymap = read_sky_map(filename, moc=True)
            #     print(f'loaded file {filename}')

            #     result = crossmatch(sky_map=skymap,
            #                         coordinates=cat_coord,
            #                         cosmology=True)
                
            #     in_90_region = (result.searched_prob_vol <= SKYMAP_CL)
                
            #     n90 = np.sum(in_90_region)
            #     p90 = np.sum(result.probdensity_vol[in_90_region])

            #     AGN_NUMDENS = len(agn_ra) / VOLUME

            #     S_agn[gw_idx] = p90 / AGN_NUMDENS
                
            #     print(p90 / AGN_NUMDENS, SKYMAP_CL, 'ligo skymap')
            
            # S_agn = S_agn[~np.isnan(S_agn)]
            # S_alt = S_alt[~np.isnan(S_alt)]

            # loglike = np.log(S_agn[:,None] * SKYMAP_CL * LOG_LLH_X_AX[None,:] + S_alt[:,None] * (1 - SKYMAP_CL * LOG_LLH_X_AX[None,:]))
            # log_llh[:,fagn_idx] = np.sum(loglike, axis=0)  # sum over all GWs

            ###################################################
        
        posteriors[trial_idx,:,:] = log_llh

    np.save(os.path.join(sys.path[0], posterior_fname), posteriors)
        

            # Testing the class above
            # s = Skymap(filename)
            # prob = 0
            # old = 0
            # for agn_idx in range(100):
            #     new_p, old_p = s.probability(agn_ra[agn_idx], agn_dec[agn_idx], agn_rlum[agn_idx])  # TODO: Not sure if vectorization works properly, so not doing that rn
            #     prob += new_p
            #     old += old_p
            # print(gw_idx, prob, old, 'class')

