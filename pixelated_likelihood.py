
# %%
"""
Multi-event likelihood Module
Rachel Gray
"""

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d
import json
# from posterior_samples import *

from utils import ra_dec_from_ipix, ipix_from_ra_dec
import healpy as hp
import bilby
import h5py
import ast
from tqdm import tqdm

from scipy.integrate import quad
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
import time

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



def single_event_fixpop_3dpos_likelihood(nside,
                                         posterior_samps_path, 
                                         field,
                                         interp_list,
                                         n_mc_samps=int(1e4)):

    # result = []
    # trials = [int(1e1), int(5e1), int(1e2), int(5e2), int(1e3), int(5e3), int(1e4), int(5e4)]
    # for n_mc_samps in trials:
    redshift, ra, dec, nsamps = load_posterior_samples(posterior_samps_path, approximant=field)

    idx_array = np.random.choice(np.arange(nsamps), size=n_mc_samps)

    mc_ra = ra[idx_array]
    mc_dec = dec[idx_array]
    mc_pix = ipix_from_ra_dec(nside=nside, ra=mc_ra, dec=mc_dec, nest=True)
    mc_redshift = redshift[idx_array]

    unique_pix = np.unique(mc_pix)  # Loop over the pixels that have been sampled
    p_agn = 0
    for pix in unique_pix:
        z_samp = mc_redshift[mc_pix == pix]     # Find all redshift samples in the pixel
        interp = interp_list[pix]               # Get the interpolant of the LOS zprior in this pixel
        p_agn += np.sum(interp(z_samp))         # Evaluate zprior at sampled redshift
    p_agn /= n_mc_samps

    p_alt = np.sum(dVdz_prior(mc_redshift)) / n_mc_samps

    return p_agn, p_alt


def multiple_event_fixpop_3dpos_likelihood(fagn_array, 
                                           posterior_samples_dictionary,
                                           LOS_catalog_path,
                                           posterior_samples_field='mock'):
    

    N_gws = len(posterior_samples_dictionary.items())

    # Calculate likelihood for each event
    LOS_catalog, nside, z_array, _ = unpack_LOS_catalog(LOS_catalog_path)
    p_agn_arr = np.zeros(N_gws)
    p_alt_arr = np.zeros(N_gws)

    # Major computation time save: keep interpolated zpriors in memory
    interp_list = []
    print('Interpolating zprior in each pixel...')
    for pix in tqdm(range(int(nside**2 * 12))):
        zprior = get_zprior(LOS_catalog, pix)
        interp_list.append(interp1d(z_array, zprior, bounds_error=False, fill_value=0))

    # keys = []
    for i, (key, _) in tqdm(enumerate(posterior_samples_dictionary.items())):
        # print(i, key, 'of', N_gws)
        # if key[1] == '0':
        #     print('skipping', key)
        #     continue

        posterior_samps_path = posterior_samples_dictionary[key]
        field = posterior_samples_field  # TODO: allow for json such that you can do `fields[key]` cf. gwcosmo

        pagn, palt = single_event_fixpop_3dpos_likelihood(
                                                        nside=nside, 
                                                        posterior_samps_path=posterior_samps_path, 
                                                        field=field,
                                                        interp_list=interp_list
                                                    ) 
        
        p_agn_arr[i], p_alt_arr[i] = pagn, palt

        # keys.append(key)  
        # print('breaking')
        # break
  
    LOS_catalog.close()

    # np.save('keys', np.array(keys))
    np.save('pagn_sky_v7', p_agn_arr)
    np.save('palt_sky_v7', p_alt_arr)

    return _

    # # Combine events
    # COMPLETENESS = 1.  # TODO: make AGN selection function
    # log_llh_multiple_events = np.zeros((gw_posterior.shape[0], len(fagn_array)))  # Save the likelihood function for all GWs for all trial f_agns.
    # for i, gw_posterior in enumerate(gw_posteriors):


    #     # If we can get pagn and palt for every GW first, we can vectorize the loop below.
    #     log_llh_single_gw = np.ones_like(fagn_array)
    #     for j, fagn in fagn_array:
    #         pagn = 1.
    #         palt = 1.
    #         in_catalog_prob = fagn * COMPLETENESS * skymap_cl
    #         out_catalog_prob = 1 - in_catalog_prob
    #         log_llh_single_gw[j] = in_catalog_prob * pagn + out_catalog_prob * palt

    # return log_llh_multiple_events


# class PixelatedGalaxyCatalogMultipleEventLikelihood(bilby.Likelihood):

#     """
#     Class for preparing and carrying out the computation of the likelihood on 
#     H0 for a single GW event
#     """
    
#     def __init__(
#                 self, 
#                 posterior_samples_dictionary, 
#                 skymap_dictionary, 
#                 injections, 
#                 LOS_catalog_path,
#                 zrates, 
#                 cosmo, 
#                 mass_priors, 
#                 min_pixels=30, 
#                 sky_cl=0.999, 
#                 posterior_samples_field=None, 
#                 network_snr_threshold=11.
#             ):

#         """
#         Parameters
#         ----------
#         samples : object
#             GW samples
#         skymap : gwcosmo.likelihood.skymap.skymap object
#             provides p(x|Omega) and skymap properties
#         LOS_prior :
#         """
#         super().__init__(parameters={'H0': None, 'gamma':None, 'Madau_k':None, 
#                                      'Madau_zp':None, 'alpha':None, 'delta_m':None, 
#                                      'mu_g':None, 'sigma_g':None, 'lambda_peak':None, 
#                                      'alpha_1':None, 'alpha_2':None, 'b':None, 
#                                      'mminbh':None, 'mmaxbh':None, 'alphans':None, 
#                                      'mminns':None, 'mmaxns':None, 'beta':None, 
#                                      'Xi0':None, 'n':None, 'D':None, 
#                                      'logRc':None, 'nD':None, 'cM':None})

#         # self.zrates = zrates
        
#         #TODO make min_pixels an optional dictionary
#         LOS_catalog, nside, self.z_array, _ = unpack_LOS_catalog(LOS_catalog_path)
        
#         # self.injections = injections
#         # self.injections.Nobs = len(list(posterior_samples_dictionary.keys())) # it's the number of GW events entering the analysis, used for the check Neff >= 4Nobs inside the injection class
#         self.mass_priors = mass_priors
#         # #TODO: add check that the snr threshold is valid for this set of injections
#         # self.injections.update_cut(snr_cut=network_snr_threshold)
#         # self.cosmo = cosmo

#         self.zprior_times_pxOmega_dict = {}
#         self.pixel_indices_dictionary = {}
#         self.samples_dictionary = {}
#         self.samples_indices_dictionary = {}
        
#         self.keys = []
                    
#         for key, _ in posterior_samples_dictionary.items():
#             samples = load_posterior_samples(posterior_samples_dictionary[key], field=posterior_samples_field[key])
#             skymap = Skymap(skymap_dictionary[key])

#             # Skymaps are very high resolution by default, need to degrade to nside of the LOS catalog (i.e. n_high), which is the highest we will ever need
#             low_res_skyprob = hp.pixelfunc.ud_grade(skymap.prob, nside, order_in='NESTED', order_out='NESTED')
#             low_res_skyprob = low_res_skyprob / np.sum(low_res_skyprob)
            
#             # Posterior sample resolution is chosen as to cover the sky area with the lowest number of pixels above the minimum of min_pixels
#             pixelated_samples = make_pixel_px_function(samples, skymap, npixels=min_pixels, thresh=sky_cl)
#             nside_low_res = pixelated_samples.nside
#             if nside_low_res > nside:
#                 raise ValueError(f'Low resolution nside {nside_low_res} is higher than high resolution nside {nside}. Try decreasing min_pixels for event {key}.')

#             # identify which samples will be used to compute p(x|z,H0) for each pixel
#             pixel_indices = pixelated_samples.indices
#             samp_ind = {}
#             for i, pixel_index in enumerate(pixel_indices):
#                 samp_ind[pixel_index] = pixelated_samples.identify_samples(pixel_index, minsamps=100)  # Searches around pixel for more samples if minimum is not reached
                
#             ######################################## I UNDERSTAND UP TO HERE - BUT IS IT ALL NECESSARY? TODO: SEE IF WE SHOULD USE 3D SKYMAPS ########################################
            
#             no_sub_pix_per_pixel = int(4**(np.log2(nside/nside_low_res)))

#             # Get the coordinates of the hi-res pixel centres
#             pixra, pixdec = ra_dec_from_ipix(nside, np.arange(hp.pixelfunc.nside2npix(nside)), nest=True)
#             # compute the low-res index of each of them
#             ipix = ipix_from_ra_dec(nside_low_res, pixra, pixdec, nest=True)

#             print('Loading the redshift prior')
#             zprior_times_pxOmega = np.zeros((len(pixel_indices), len(self.z_array)))
#             for i, pixel_index in enumerate(pixel_indices):
#                 # Find the hi res indices corresponding to the current coarse pixel
#                 hi_res_pixel_indices = np.arange(hp.pixelfunc.nside2npix(nside))[np.where(ipix==pixel_index)[0]]
#                 # load pixels, weight by GW sky area, and combine
#                 for j, hi_res_index in enumerate(hi_res_pixel_indices):
#                     zprior_times_pxOmega[i,:] += get_zprior(LOS_catalog, hi_res_index) * low_res_skyprob[hi_res_index]
#             print(f"Identified {len(pixel_indices) * no_sub_pix_per_pixel} pixels in the galaxy catalogue which correspond to {key}'s {sky_cl*100}% sky area")
                
#             self.zprior_times_pxOmega_dict[key] = zprior_times_pxOmega
#             self.pixel_indices_dictionary[key] = pixel_indices
#             self.samples_dictionary[key] = samples
#             self.samples_indices_dictionary[key] = samp_ind
#             self.keys.append(key)
            
#         LOS_catalog.close()
#         print(self.keys)


#     def log_likelihood_numerator_single_event(self, event_name):

#         pixel_indices = self.pixel_indices_dictionary[event_name]
#         samples = self.samples_dictionary[event_name]
#         samp_ind = self.samples_indices_dictionary[event_name]
#         zprior = self.zprior_times_pxOmega_dict[event_name]
                
#         # set up KDEs for this value of the parameter to be analysed
#         px_zOmegaparam = np.zeros((len(pixel_indices), len(self.z_array)))
#         for i, pixel_index in enumerate(pixel_indices):

#             z_samps, m1_samps, m2_samps = self.reweight_samps.compute_source_frame_samples(samples.distance[samp_ind[pixel_index]], 
#                                                                                            samples.mass_1[samp_ind[pixel_index]], 
#                                                                                            samples.mass_2[samp_ind[pixel_index]])

#             zmin_temp = np.min(z_samps)*0.5
#             zmax_temp = np.max(z_samps)*2.
#             z_array_temp = np.linspace(zmin_temp, zmax_temp, 100)

#             kde, norm = self.reweight_samps.marginalized_redshift_reweight(z_samps, m1_samps, m2_samps)

#             if norm != 0: # px_zOmegaH0 is initialized to 0
#                 px_zOmegaparam_interp = interp1d(z_array_temp, kde(z_array_temp), kind='cubic', bounds_error=False, fill_value=0)
#                 px_zOmegaparam[i,:] = px_zOmegaparam_interp(self.z_array) * norm
        
#         # make p(s|z) have the same shape as p(x|z,Omega,param) and p(z|Omega,s)
#         ps_z_array = np.tile(self.zrates(self.z_array), (len(pixel_indices), 1))
        
#         Inum_vals = np.sum(px_zOmegaparam * zprior * ps_z_array, axis=0)
#         num = simpson(Inum_vals, self.z_array)

#         return np.log(num)
    
        
#     # def log_likelihood_denominator_single_event(self):

#         # z_prior = interp1d(self.z_array,self.zprior_full_sky*self.zrates(self.z_array),bounds_error=False,fill_value=(0,(self.zprior_full_sky*self.zrates(self.z_array))[-1]))
#         # dz=np.diff(self.z_array)
#         # z_prior_norm = np.sum((z_prior(self.z_array)[:-1]+z_prior(self.z_array)[1:])*(dz)/2)
#     #     injections = copy.deepcopy(self.injections) # Nobs is set in self.injection in the init

#     #     # Update the sensitivity estimation with the new model
#     #     injections.update_VT(self.cosmo,self.mass_priors,z_prior,z_prior_norm)
#     #     Neff, Neff_is_ok, var = injections.calculate_Neff()
#     #     if Neff_is_ok: # Neff >= 4*Nobs    
#     #         log_den = np.log(injections.gw_only_selection_effect())
#     #     else:
#     #         print("Not enough Neff ({}) compared to Nobs ({}) for current mass-model {}, z-model {}, zprior_norm {}"
#     #               .format(Neff,injections.Nobs,self.mass_priors,z_prior,z_prior_norm))
#     #         print("mass prior dict: {}, cosmo_prior_dict: {}".format(self.mass_priors_param_dict,self.cosmo_param_dict))
#     #         print("returning infinite denominator")
#     #         print("exit!")
#     #         log_den = np.inf
#     #         #sys.exit()
            
#     #     return log_den, np.log(z_prior_norm)

                       
#     def log_combined_event_likelihood(self):
        
#         # carry norm to apply to numerator as well
#         # den_single, zprior_norm_log = self.log_likelihood_denominator_single_event()
#         # den = den_single*len(self.keys)
        
#         zprior_norm_log = 0  # TODO: verify that this is not needed -Lucas
#         num = 1.
#         for event_name in self.keys:
#             num += self.log_likelihood_numerator_single_event(event_name) - zprior_norm_log

#         return num
#         # return num-den

        
#     def log_likelihood(self):

#         # self.zrates.gamma = self.parameters['gamma']
#         # self.zrates.k = self.parameters['Madau_k']
#         # self.zrates.zp = self.parameters['Madau_zp']
        
#         self.mass_priors_param_dict = {'alpha':self.parameters['alpha'], 'delta_m':self.parameters['delta_m'], 
#                                          'mu_g':self.parameters['mu_g'], 'sigma_g':self.parameters['sigma_g'], 
#                                          'lambda_peak':self.parameters['lambda_peak'],
#                                          'alpha_1':self.parameters['alpha_1'], 
#                                          'alpha_2':self.parameters['alpha_2'], 'b':self.parameters['b'], 
#                                          'mminbh':self.parameters['mminbh'], 'mmaxbh':self.parameters['mmaxbh'], 
#                                          'beta':self.parameters['beta'], 'alphans':self.parameters['alphans'],
#                                          'mminns':self.parameters['mminns'], 'mmaxns':self.parameters['mmaxns']}

#         self.mass_priors.update_parameters(self.mass_priors_param_dict)

#         # self.cosmo_param_dict = {'H0': self.parameters['H0'], 'Xi0': self.parameters['Xi0'], 'n': self.parameters['n'], 'D': self.parameters['D'], 'logRc': self.parameters['logRc'], 'nD': self.parameters['nD'], 'cM': self.parameters['cM']}
#         # self.cosmo.update_parameters(self.cosmo_param_dict)

#         # self.reweight_samps = reweight_posterior_samples(self.cosmo, self.mass_priors)
        
#         return self.log_combined_event_likelihood()
    
        
#     def __call__(self):
#         return np.exp(self.log_likelihood())
    

if __name__ == '__main__':

    post_path = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/inputs/posterior_samples_mock_v7.json'
    with open(post_path) as f:
        posterior_samples_dictionary = json.load(f)

    LOS_catalog_path = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/LOSzpriors/MOCK_LOS_redshift_prior_lenzarray_12000_zdraw_2.0_zmax_3.0_nside_32_nagn_100000_sigma_0.01_pixel_index_None.hdf5'

    multiple_event_fixpop_3dpos_likelihood(fagn_array=None, 
                                           posterior_samples_dictionary=posterior_samples_dictionary,
                                           LOS_catalog_path=LOS_catalog_path,
                                           posterior_samples_field='mock')


    # LOS_catalog, nside, z_array, _ = unpack_LOS_catalog(LOS_catalog_path)
    # arr = get_zprior(LOS_catalog, 10)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(z_array, arr)
    # plt.show()

    # from ligo.skymap.io.fits import read_sky_map
    # filename = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/plots/test.fits'
    # prob, _ = read_sky_map(filename, distances=False, moc=False, nest=True)
    # print(len(prob))
    # prob2 = read_sky_map(filename, distances=False, moc=True, nest=True)
    # print(len(prob2))
    
# %%
