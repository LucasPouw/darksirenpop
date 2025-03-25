"""
LALinference posterior samples class and methods
Ignacio Magana, Ankan Sur
"""

import numpy as np
from scipy.stats import gaussian_kde
import h5py
from utils import ra_dec_from_ipix
import json
import healpy as hp


class load_posterior_samples(object):
    """
    Posterior samples class and methods.

    Parameters
    ----------
    posterior_samples : Path to posterior samples file to be loaded.
    field : Internal field of the json or the h5 file
    """
    
    def __init__(self,posterior_samples,field=None):
        self.posterior_samples = posterior_samples

        self.field=field
        self.load_posterior_samples()


    def load_posterior_samples(self):
        """
        Method to handle different types of posterior samples file formats.
        Currently it supports .dat (LALinference), .hdf5 (GWTC-1),
        .h5 (PESummary) and .hdf (pycbcinference) formats.
        """

        if self.posterior_samples[-3:] == 'dat':
            samples = np.genfromtxt(self.posterior_samples, names = True)
           
            self.distance = np.array([var for var in samples['luminosity_distance']])
            self.ra =  np.array([var for var in samples['ra']])
            self.dec =  np.array([var for var in samples['dec']])
            self.mass_1 =  np.array([var for var in samples['mass_1']])
            self.mass_2 =  np.array([var for var in samples['mass_2']])
            self.nsamples = len(self.distance)

        if self.posterior_samples[-4:] == 'hdf5':
            if self.posterior_samples[-11:] == 'GWTC-1.hdf5':
                if self.posterior_samples[-20:] == 'GW170817_GWTC-1.hdf5':
                    dataset_name = 'IMRPhenomPv2NRT_lowSpin_posterior'
                else:
                    dataset_name = 'IMRPhenomPv2_posterior'
                file = h5py.File(self.posterior_samples, 'r')
                data = file[dataset_name]
                self.distance = data['luminosity_distance_Mpc']
                self.ra = data['right_ascension']
                self.dec = data['declination']
                self.mass_1 = data['m1_detector_frame_Msun']
                self.mass_2 = data['m2_detector_frame_Msun']
                self.nsamples = len(self.distance)
                file.close()

        if self.posterior_samples.endswith('.json'):
            with open(self.posterior_samples) as f:
                data = json.load(f)

            PE_struct=data['posterior_samples'][self.field]

            m1_ind=PE_struct['parameter_names'].index('mass_1')
            m2_ind=PE_struct['parameter_names'].index('mass_2')
            dl_ind=PE_struct['parameter_names'].index('luminosity_distance')
            ra_ind=PE_struct['parameter_names'].index('ra')
            dec_ind=PE_struct['parameter_names'].index('dec')
                        
            nsamp=len(PE_struct['samples'])
            
            self.distance = np.array(PE_struct['samples'])[:,dl_ind].reshape(-1)
            self.ra = np.array(PE_struct['samples'])[:,ra_ind].reshape(-1)
            self.dec = np.array(PE_struct['samples'])[:,dec_ind].reshape(-1)
            self.mass_1 = np.array(PE_struct['samples'])[:,m1_ind].reshape(-1)
            self.mass_2 = np.array(PE_struct['samples'])[:,m2_ind].reshape(-1)
            self.nsamples = len(self.distance)

        ####################### ONLY THIS PART NEEDED FROM THIS CLASS #########################

        if self.posterior_samples[-2:] == 'h5':
            file = h5py.File(self.posterior_samples, 'r')

            if self.field is None:
                approximants = ['mock', 
                                'PublicationSamples','C01:Mixed','C01:PhenomPNRT-HS', 
                                'C01:NRSur7dq4', 'C01:IMRPhenomPv3HM', 'C01:IMRPhenomPv2',
                                'C01:IMRPhenomD', 'C01:IMRPhenomPv2_NRTidal:LowSpin', 
                                'C01:IMRPhenomPv2_NRTidal:HighSpin']
                for approximant in approximants:
                    try:
                        data = file[approximant]
                        print("Using "+approximant+" posterior")
                        break
                    except KeyError:
                        continue
            else:
                data=file[self.field]

            self.distance = data['posterior_samples']['luminosity_distance']
            self.redshift = data['posterior_samples']['redshift']
            self.ra = data['posterior_samples']['ra']
            self.dec = data['posterior_samples']['dec']
            # self.mass_1 = data['posterior_samples']['mass_1']
            # self.mass_2 = data['posterior_samples']['mass_2']
            self.nsamples = len(self.ra)
            file.close()

        ########################################################################################

        if self.posterior_samples[-3:] == 'hdf':
            file = h5py.File(self.posterior_samples, 'r')
            self.distance = file['samples/distance'][:]
            self.ra = file['samples/ra'][:]
            self.dec = file['samples/dec'][:]
            self.mass_1 = file['samples/mass_1'][:]
            self.mass_2 = file['samples/mass_2'][:]
            self.nsamples = len(self.distance)
            file.close()

    def marginalized_sky(self):
        """
        Computes the marginalized sky localization posterior KDE.
        """
        return gaussian_kde(np.vstack((self.ra, self.dec)))


class reweight_posterior_samples(object):
    """
    Posterior samples class and methods.

    Parameters
    ----------
    cosmo : Fast cosmology class
    mass_priors: Fast mass_distributions class
    """
    
    def __init__(self,cosmo,mass_priors):
        self.cosmo = cosmo
        # Prior distribution used in this work
        self.source_frame_mass_prior = mass_priors
    def jacobian_times_prior(self,z):
        """
        (1+z)^2 * ddL/dz * dL^2
        """
        jacobian = np.power(1+z,2)*self.cosmo.ddgw_dz(z)
        dgw = self.cosmo.dgw_z(z)

        return jacobian*(dgw**2)
        
    def compute_source_frame_samples(self, GW_distance, det_mass_1, det_mass_2):
        """
        Posterior samples class and methods.

        Parameters
        ----------
        GW_distance: GW distance samples in Mpc
        det_mass_1, det_mass_2 : detector frame mass samples in Msolar
        H0 : Hubble constant value in kms-1Mpc-1
        """
        redshift = self.cosmo.z_dgw(GW_distance)

        mass_1_source = det_mass_1/(1+redshift)
        mass_2_source = det_mass_2/(1+redshift)
        return redshift, mass_1_source, mass_2_source

    def get_kde(self, data, weights):
        # deal first with the weights
        weights, norm, neff = self.check_weights(weights)
        if norm != 0:
            try:
                kde = gaussian_kde(data, weights=weights)
            except:
                print("KDE problem! create a default KDE with norm=0")
                print("norm: {} -> 0, neff: {}".format(norm,neff))
                norm = 0
                kde = gaussian_kde(data)
        else:
            kde = gaussian_kde(data)
            
        return kde, norm
                    
    def ignore_weights(self, weights):

        weights = np.ones(len(weights))
        norm = 0
        return weights, norm

    def check_weights(self, weights):
        """
        Check the weights values to prevent gaussian_kde crash when Neff <= 1,
        where Neff is an internal variable of gaussian_kde
        defined by Neff = sum(weights)^2/sum(weights^2)
        careful, cases with Neff = 1+2e-16 = 1.0000000000000002
        have been seen and give crash: set Neff limit to >= 2
        """
        neff = 0
        if np.isclose(max(weights),0,atol=1e-50):
            weights, norm = self.ignore_weights(weights)
        else:
            neff = sum(weights)**2/sum(weights**2)
            if neff<2:
                weights, norm = self.ignore_weights(weights)
            else:
                norm = np.sum(weights)/len(weights)
        return weights, norm, neff
    
    def marginalized_redshift_reweight(self, redshift, mass_1_source, mass_2_source):
        """
        Computes the marginalized distance posterior KDE.
        """
        # Re-weight
        weights = self.source_frame_mass_prior.joint_prob(mass_1_source,mass_2_source)/self.jacobian_times_prior(redshift)

        return self.get_kde(redshift,weights)

    def marginalized_redshift(self, redshift):
        """
        Computes the marginalized distance posterior KDE.
        """
        # remove dgw^2 prior and include dz/ddgw jacobian
        weights = 1/(self.cosmo.ddgw_dz(redshift)*self.cosmo.dgw(redshift)**2)
        return self.get_kde(redshift,weights)


class make_pixel_px_function(object):
    """
    Identify the posterior samples which lie within some angular radius
    (depends on skymap pixel size) of the centre of each pixel
    """
    
    def __init__(self, samples, skymap, npixels=30, thresh=0.999):
        """
        Parameters
        ----------
        samples : posterior_samples object
            The GW samples
        skymap : object
            The GW skymap
        npixels : int, optional
            The minimum number of pixels desired to cover given sky area of
            the GW event (default=30)
        thresh : float, optional
            The sky area threshold (default=0.999)
        """
        
        self.skymap = skymap
        self.samples = samples
        nside=1
        indices, prob = skymap.above_percentile(thresh, nside=nside)
    
        while len(indices) < npixels:
            nside = nside*2
            indices, prob = skymap.above_percentile(thresh, nside=nside)
        
        self.nside = nside
        print('{} pixels to cover the {}% sky area (nside={})'.format(len(indices), thresh*100, nside))
        
        dicts = {}
        for i,idx in enumerate(indices):
            dicts[idx] = prob[i]
        self.indices = indices
        self.prob = dicts # dictionary - given a pixel index, returns skymap prob

        
    def identify_samples(self, idx, minsamps=100):
        """
        Find the samples required 
        
        Parameters
        ----------
        idx : int
            The pixel index
        minsamps : int, optional
            The threshold number of samples to reach per pixel
            
        Return
        ------
        sel : array of ints
            The indices of posterior samples for pixel idx
        """
        
        ipix_samples = hp.pixelfunc.ang2pix(self.nside, 0.5*np.pi - self.samples.dec, self.samples.ra, nest=self.skymap.nested)
        sel = np.where(ipix_samples == idx)[0]
        if len(sel) >= minsamps:
            print("{} samples fall in pix {}".format(len(sel), idx))
            return sel

        # not enough samples in pixel 'idx', we need to extend the search
        racent, deccent = ra_dec_from_ipix(self.nside, idx, nest=self.skymap.nested)
        separations = angular_sep(racent, deccent, self.samples.ra, self.samples.dec)
        sep = hp.pixelfunc.max_pixrad(self.nside)*0.5 # choose initial separation
        step = sep*0.5 # choose step size for increasing radius
        
        sel = np.where(separations < sep)[0] # find all the samples within the angular radius sep from the pixel centre
        nsamps = len(sel)
        while nsamps < minsamps:
            sep += step
            sel = np.where(separations < sep)[0]
            nsamps = len(sel)
            if sep > np.pi:
                raise ValueError("Problem with the number of posterior samples.")
        print('pixel idx {}: angular radius: {} radians, No. samples: {}'.format(idx, sep, len(sel)))
         
        return sel
        

def identify_samples_from_posterior(ra_los, dec_los, ra, dec, nsamps=1000):
    """
    Find the angular separation between all posterior samples and a specific
    LOS. Return the indices of the nsamps closest samples, as well as the 
    maxiumum separation of those samples.
    
    Parameters
    ----------
    ra_los : float
        right ascension of the line-of-sight (radians)
    dec_los : float
        declination of the line-of-sight (radians)
    ra : array of floats
        right ascensions of a set of samples (radians)
    dec : array of floats
        declinations of a set of samples (radians)
    nsamps : int, optional
        The number of samples to select (default=1000)
        
    Return
    ------
    index : array of ints
        The indices of of the nsamps samples
    ang_rad_max : float
        The maximum angular radius between the selected samples and the LOS
    """

    separations = angular_sep(ra_los,dec_los,ra,dec)
    sep_argsort = np.argsort(separations)
    index = sep_argsort [:nsamps]
    ang_rad_max = separations [nsamps-1]

    return index, ang_rad_max

        
    
def angular_sep(ra1,dec1,ra2,dec2):
    """Find the angular separation between two points, (ra1,dec1)
    and (ra2,dec2), in radians."""
    
    cos_angle = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)
    angle = np.arccos(cos_angle)
    return angle
