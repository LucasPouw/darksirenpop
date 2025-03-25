"""
Line-of-sight redshift prior creation module

Adapted from LOS_redshift_prior.py in gwcosmo
https://git.ligo.org/lscsoft/gwcosmo/-/blob/master/gwcosmo/prior/LOS_redshift_prior.py
"""

import numpy as np
from scipy.stats import truncnorm
from scipy.interpolate import interp1d
from darksirenpop.utils import ra_dec_from_ipix, ipix_from_ra_dec, get_cachedir
import healpy as hp
from darksirenpop.agn_catalogs_and_priors.pixelated_catalog import GalaxyCatalog, load_catalog_from_path
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import logging
logger = logging.getLogger(__name__)


def dVdz_unnorm(z, cosmo):
    '''Assuming flat LCDM'''
    Omega_m = cosmo.Om0
    Omega_Lambda = 1 - Omega_m
    E_of_z = np.sqrt((1 + z)**3 * Omega_m + Omega_Lambda)
    com_vol = ((1 + z) * cosmo.angular_diameter_distance(z).value)**2 / E_of_z
    return com_vol


def agn_likelihood_unnorm(zobs, z, sigma):
    '''L(zobs|x), where observations are drawn from a normal distribution at the true z'''
    return np.exp(-0.5 * (zobs/z - 1)**2 / sigma**2) / z


def agn_posterior_integrand(y, zobs, sigma, cosmo):
    '''Do a parameter transformation to get stable results: z = e^y'''
    z = np.exp(y)
    jacobian = z
    return agn_likelihood_unnorm(zobs, z, sigma) * dVdz_unnorm(z, cosmo) * jacobian


def normalization(zobs, sigma, cosmo):
    '''Numerically stable for integration bounds x/10 to x*10 with a maximum of zdraw'''
    ymin, ymax = np.log(zobs / 10), np.log(zobs * 10)
    return quad(agn_posterior_integrand, ymin, ymax, args=(zobs, sigma, cosmo))[0]


def compute_norm(zmin, zmax, sigma, npoints, cosmo):
    '''
    AGN redshift posteriors are normalized between zmin and zdraw, but are calculated
    for observed redshifts between zmin and zmax.
    '''

    observations = np.geomspace(zmin, zmax, npoints)
    norm_arr = np.zeros(npoints)

    print('Calculating normalizations...')
    for i, obs in tqdm(enumerate(observations)):
        norm_arr[i] = normalization(obs, sigma, cosmo)

    return observations, norm_arr


def get_norm_interp(zmin, zmax, sigma, npoints, cosmo, cachedir=None):
    '''
    The normalization of the redshift posterior depends on the observed redshift.
    Note that we consider AGN with observed redshift above zdraw, because part of 
    the posterior extends below zdraw.

    This can be precomputed and saved in cache. We then load the file and interpolate
    to the observed AGN redshift.

    zmin:       Lowest redshift an AGN can exist at (take close to 0)
    zmax:       Highest redshift an AGN can exist at (should be higher than zdraw!)
    sigma:      Fractional redshift error
    npoints:    Number of points in the interpolant
    cosmo:      Astropy cosmology for uniform in comoving volume redshift prior
    cachedir:   Path to cache directory
    '''

    if cachedir is None:
        cachedir = get_cachedir()

    filename = f'normalizations_zmin_{zmin}_zmax_{zmax}_sigma_{sigma}_npoints_{npoints}.npz'
    filepath = os.path.join(cachedir, filename)
    
    try:
        normfile = np.load(filepath)
        observations, norm_arr = normfile['x'], normfile['y']

    except Exception as e:
        print(f'Could not load normalizations from cache: {e}')
        observations, norm_arr = compute_norm(zmin, zmax, sigma, npoints, cosmo)

        np.savez(filepath, x=observations, y=norm_arr)  # Store in cache

    return interp1d(observations, norm_arr, kind='linear')


def agn_posterior(x, zobs, sigma, norm, cosmo):
    return agn_likelihood_unnorm(zobs, x, sigma) * dVdz_unnorm(x, cosmo) / norm


class LineOfSightRedshiftPrior:
    """
    Calculate the likelihood of cosmological parameters from one GW event, 
    using the galaxy catalogue method.

    Parameters
    ----------
    pixel_index : Index of the healpy pixel to analyse
    galaxy_catalog : pixelated_catalog.GalaxyCatalog object
        The galaxy catalogue
    nside : int
            The resolution value of nside
    galaxy_norm : string
        Read the galaxy_normalization from a (lower resolution) precomuted galaxy norm map located at this path.
    zmax : float, optional
        The upper redshift limit for integrals (default=10.). Should be well
        beyond the highest redshift reachable by GW data or selection effects.
    min_gals_for_threshold : int, optional
    """

    def __init__(self, 
                 pixel_index:int, 
                 galaxy_catalog:GalaxyCatalog,  # TODO: rework pixelated_catalog.py such that this is true. Only self.populate() (I think) needs to be redone, such that self.data works
                 nside:int,
                 galaxy_norm:str,
                 z_array:np.ndarray,  # TODO: Why return this if it is always the same?
                 zdraw:float,
                 zmin:float=1e-10,
                 zmax:float=3.,
                 sigma:float=0.05,
                 npoints_norm:int=10000,
                 min_gals_for_threshold:int=10,
                 cosmo=FlatLambdaCDM(H0=67.9, Om0=0.3065),
                 cachedir:str=None):
        
        self.pixel_index = pixel_index
        self.nside = nside
        self.z_array = z_array
        self.zmin = zmin
        self.zdraw = zdraw
        self.zmax = zmax
        self.cosmo = cosmo
        subcatalog = galaxy_catalog.select_pixel(nside, pixel_index, nested=True)  # Load in the galaxy catalogue data from this pixel

        m = hp.fitsfunc.read_map(galaxy_norm, nest=True)  # Load precomputed norm map
        pixra, pixdec = ra_dec_from_ipix(nside, pixel_index, nest=True)  # Get the coordinates of the pixel centre
        ipix = ipix_from_ra_dec(hp.pixelfunc.get_nside(m), pixra, pixdec, nest=True)  # Compute the corresponding low-res index
        low_res_galaxy_norm = m[ipix]
        if low_res_galaxy_norm < min_gals_for_threshold:
            self.galaxy_norm = 0
        else:
            self.galaxy_norm = low_res_galaxy_norm / (hp.nside2npix(self.nside) / hp.get_map_size(m))  # Adjust for different resolutions

        self.zs = subcatalog['redshift']
        self.sigma = sigma
        # self.sigmazs = subcatalog['redshift_error']

        self.norminterp = get_norm_interp(zmin=self.zmin, zmax=self.zmax, sigma=self.sigma, npoints=npoints_norm, cosmo=self.cosmo, cachedir=cachedir)

        ngals = len(self.zs)
        print(f"Found {ngals} in fine pixel. Coarse pixel had {low_res_galaxy_norm}. Galaxy norm: {self.galaxy_norm}")
        logger.info(f"Found {ngals} in fine pixel. Coarse pixel had {low_res_galaxy_norm}. Galaxy norm: {self.galaxy_norm}")

    
    def create_redshift_prior(self):
        '''
        The AGN hypothesis has a redshift prior that is a sum over Gaussians. Here we calculate that sum for a single pixel.

        OUTPUT:
        -------
        pz_G (ndarray): 
        z_array (ndarray): redshifts at which the probabilities are evaluated
        '''

        logger.info(f"nside {self.nside} pixel {self.pixel_index}: Start creation of redshift prior")

        pz_G = np.zeros(len(self.z_array))

        if self.galaxy_norm != 0:  # If pixel is not empty

            for i in range(len(self.zs)):
                zobs = self.zs[i]
                if zobs > self.zmax:
                    print('Skipping observation z=', zobs)
                    continue
                norm = self.norminterp(zobs)
                pz_G += agn_posterior(self.z_array, zobs, self.sigma, norm, self.cosmo)

                # low_z_lim, high_z_lim = (0 - self.zs[i]) / self.sigmazs[i], (self.zmax - self.zs[i]) / self.sigmazs[i]  # Set redshift limits so that galaxies can't have negative z - between 0 and zmax
                # pz_G += truncnorm.pdf(self.z_array, low_z_lim, high_z_lim, self.zs[i], self.sigmazs[i])

            # pz_G /= self.galaxy_norm

            # Normalize GW pior between z = 0 and z = zdraw (hopefully good enough for mock data, not using galaxy_norm)
            zprior_interp = interp1d(self.z_array, pz_G)
            pz_G /= quad(zprior_interp, self.zmin, self.zdraw)[0]
            
        return pz_G, self.z_array
    

if __name__ == '__main__':

    ZMIN = 1e-10
    ZDRAW = 2
    ZMAX = 3
    COSMO = FlatLambdaCDM(H0=67.9, Om0=0.3065)

    zarray = np.logspace(-10, np.log10(ZDRAW), 3000)

    from pixelated_catalog import load_catalog_from_path

    CAT_PATH = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs/mockcat_NAGN750000_ZMAX_3.hdf5'
    GalCat = load_catalog_from_path(name='MOCK', catalog_path=CAT_PATH)

    NORM_PATH = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/maps/mocknorm_NAGN750000_ZMAX_3.fits'

    high_nside = 64

    LOS_zprior = LineOfSightRedshiftPrior(pixel_index=1,
                                            galaxy_catalog=GalCat, 
                                            nside=high_nside, 
                                            z_array=zarray,
                                            galaxy_norm=NORM_PATH,
                                            zdraw=ZDRAW,
                                            zmax=ZMAX,
                                            min_gals_for_threshold=10,
                                            cosmo=COSMO)
    p_of_z, _ = LOS_zprior.create_redshift_prior()

    plt.figure()
    plt.plot(zarray, p_of_z)
    plt.xlabel('Redshift')
    plt.ylabel('Probability density')
    plt.savefig('zprior.pdf')
    plt.show()


    # zobs = 2.4
    # sigma = 0.05
    # norminterp = get_norm_interp(zmin=ZMIN, zmax=ZMAX, sigma=sigma)
    # NORM = norminterp(zobs)

    # trunc_low, trunc_high = (ZMIN - zobs) / (sigma * zobs), (ZMAX - zobs) / (sigma * zobs)

    # import time

    # start = time.time()
    # tempax = np.linspace(zobs - 5 * (zobs * sigma), zobs + 5 * (zobs * sigma), 50)
    # trunk = truncnorm.pdf(tempax, trunc_low, trunc_high, zobs, zobs * sigma)
    # interpolate_trunk = interp1d(tempax, trunk, bounds_error=False, fill_value=0)
    # interpolate_trunk(xx)
    # print(f'that took {time.time() - start} s')
    # # print('trunc convoluted', interpolate_trunk(xx))

    # start = time.time()
    # direct = truncnorm.pdf(xx, trunc_low, trunc_high, zobs, zobs * sigma)
    # print(f'that took {time.time() - start} s')
    # # print('trunc direct', truncnorm.pdf(xx, trunc_low, trunc_high, zobs, zobs * sigma))

    # start = time.time()
    # correct = agn_posterior(xx, zobs, sigma)
    # print(f'that took {time.time() - start} s')
    # # print(quad(lambda xx: agn_likelihood(zobs, xx, sigma) * dVdz_prior(xx), ZMIN, 0.01), 'aaa')


    # plt.figure()
    # plt.plot(xx, interpolate_trunk(xx), label='Interp')
    # plt.plot(xx, truncnorm.pdf(xx, trunc_low, trunc_high, zobs, zobs * sigma), label='Truncnorm')
    # plt.plot(xx, agn_posterior(xx, zobs, sigma), label='Correct')
    # plt.legend()
    # # plt.semilogx()
    # plt.xlim(0, zobs + 5 * sigma * zobs)
    # # plt.savefig('interpvstrunc.pdf')
    # plt.show()