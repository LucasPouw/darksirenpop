"""
Line-of-sight redshift prior creation module

Adapted from LOS_redshift_prior.py in gwcosmo
https://git.ligo.org/lscsoft/gwcosmo/-/blob/master/gwcosmo/prior/LOS_redshift_prior.py
"""

import numpy as np
from scipy.stats import truncnorm
from scipy.interpolate import interp1d
from mockgw.utils import ra_dec_from_ipix, ipix_from_ra_dec
import healpy as hp
from pixelated_catalog import GalaxyCatalog

import logging
logger = logging.getLogger(__name__)


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
                 zmax:float=10.,
                 min_gals_for_threshold:int=10):
        
        self.pixel_index = pixel_index
        self.nside = nside
        self.z_array = z_array
        self.zmax = zmax
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
        self.sigmazs = subcatalog['redshift_error']

        ngals = len(self.zs)
        logger.info(f"Found {ngals} in fine pixel. Coarse pixel had {low_res_galaxy_norm}. Galaxy norm: {self.galaxy_norm}")

    
    def create_redshift_prior(self):
        '''
        The AGN hypothesis has a redshift prior that is a sum over Gaussians. Here we calculate that sum for a single pixel.

        OUTPUT:
        -------
        pz_agn (ndarray): 
        z_array (ndarray): redshifts at which the probabilities are evaluated
        '''

        logger.info(f"nside {self.nside} pixel {self.pixel_index}: Start creation of redshift prior")

        ### pG is just f_agn * fc * fobsc. No redshift cuts, color selection or magnitude threshold needed, everything is in the completeness estimates - amazing ###
        ### only pz_G is needed: the actual LOS zprior of the AGN hypothesis. The prior in the ALT hypothesis is 1/V, which we get from the skymaps. ###

        pz_G = np.zeros(len(self.z_array))

        if self.galaxy_norm != 0:  # If pixel is not empty

            for i in range(len(self.zs)):
                agn_zdist_array = np.linspace(self.zs[i] - 5*self.sigmazs[i], self.zs[i] + 5*self.sigmazs[i], 50)  # Redshift array for truncnorm interpolations in redshift uncertainties

                trunk = np.empty_like(agn_zdist_array)

                low_z_lim, high_z_lim = (0 - self.zs[i]) / self.sigmazs[i], (self.zmax - self.zs[i]) / self.sigmazs[i]  # Set redshift limits so that galaxies can't have negative z - between 0 and zmax
                trunk = truncnorm.pdf(agn_zdist_array, low_z_lim, high_z_lim, self.zs[i], self.sigmazs[i])
                interpolate_trunk = interp1d(agn_zdist_array, trunk, bounds_error=False, fill_value=0)
                pz_G += interpolate_trunk(self.z_array)  # We only want values at the pre-determined redshift array

            pz_G /= self.galaxy_norm

        return pz_G, self.z_array
    