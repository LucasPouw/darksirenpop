"""
Module to compute and handle skymaps
Rachel Gray, Ignacio Magana, Archisman Ghosh, Ankan Sur
"""

import numpy as np
import healpy as hp
from ligo.skymap.io.fits import read_sky_map


class Skymap:

    """
    Read a FITS file and return interpolation kernels on the sky.
    """

    def __init__(self, filename):
        """
        Input parameters:
        - filename : FITS file to load from
        """

        try:  # Real skymaps have distance information, which we won't need. Also ignore metadata
            (self.prob, _, _, _), _ = read_sky_map(filename, distances=True, moc=False, nest=True)
        except KeyError:  # Mock skymaps don't have distance information
            self.prob, _ = read_sky_map(filename, distances=False, moc=False, nest=True)
        
        # Setting nest=True in read_sky_map ensures that the return is
        # in the nested pixel order.
        self.nested = True

        self.npix = len(self.prob)
        self.nside = hp.npix2nside(self.npix)
        colat, self.ra = hp.pix2ang(self.nside, range(len(self.prob)),
                                    nest=self.nested)
        self.dec = 0.5 * np.pi - colat


    def skyprob(self, ra, dec):
        """
        Return the probability of a given sky location
        ra, dec: radians
        """
        ipix_gal = self.indices(ra, dec)
        return self.prob[ipix_gal]
    

    def indices(self, ra, dec):
        """
        Return the index of the skymap pixel that contains the
        coordinate ra,dec
        """
        return hp.ang2pix(self.nside, 0.5 * np.pi - dec, ra, nest=self.nested)


    def above_percentile(self, thresh, nside):
        """Returns indices of array within the given threshold
        credible region."""
        prob = self.prob
        pixorder = 'NESTED' if self.nested else 'RING'
        if nside != self.nside:
            new_prob = hp.pixelfunc.ud_grade(self.prob, nside,
                                             order_in=pixorder,
                                             order_out=pixorder)
            prob = new_prob/np.sum(new_prob)  # renormalise

        #  Sort indicies of sky map
        ind_sorted = np.argsort(-prob)
        #  Cumulatively sum the sky map
        cumsum = np.cumsum(prob[ind_sorted])
        #  Find indicies contained within threshold area
        lim_ind = np.where(cumsum > thresh)[0][0]
        return ind_sorted[:lim_ind], prob[ind_sorted[:lim_ind]]


    def samples_within_region(self, ra, dec, thresh, nside=None):
        """Returns boolean array of whether galaxies are within
        the sky map's credible region above the given threshold"""
        if nside is None:
            nside = self.nside
        skymap_ind = self.above_percentile(thresh, nside=nside)[0]
        samples_ind = hp.ang2pix(nside, 0.5 * np.pi - dec, ra, nest=self.nested)
        return np.isin(samples_ind, skymap_ind)


    def region_with_sample_support(self, ra, dec, thresh, nside=None):
        """
        Finds fraction of sky with catalogue support, and corresponding
        fraction of GW sky probability
        """
        if nside is None:
            nside = self.nside
        skymap_ind, skymap_prob = self.above_percentile(thresh, nside=nside)
        samples_ind = hp.ang2pix(nside, 0.5 * np.pi - dec, ra, nest=self.nested)
        ind = np.isin(skymap_ind, samples_ind)
        fraction_of_sky = np.count_nonzero(ind)/hp.nside2npix(nside)
        GW_prob_in_fraction_of_sky = np.sum(skymap_prob[ind])
        return fraction_of_sky, GW_prob_in_fraction_of_sky


    def pixel_split(self, ra, dec, nside):
        """
        For a list of galaxy ra and decs, return a dictionary identifying the
        index of the galaxies in each pixel of a healpy map of resolution nside

        Parameters
        ----------
        ra, dec : (ndarray, ndarray)
            Coordinates of the sources in radians.

        nside : int
            HEALPix nside of the target map

        Return
        ------
        dicts : dictionary
            dictionary of galaxy indices in each pixel

        dicts[idx] returns the indices of each galaxy in skymap pixel idx.
        """

        # The number of pixels based on the chosen value of nside
        # npix = hp.nside2npix(nside)

        # conver to theta, phi
        theta = 0.5 * np.pi - dec
        phi = ra

        # convert to HEALPix indices (each galaxy is assigned to a single healpy pixel)
        indices = hp.ang2pix(nside, theta, phi, nest=self.nested)

        # sort the indices into ascending order
        idx_sort = np.argsort(indices)
        sorted_indices = indices[idx_sort]

        # idx: the healpy index of each pixel containing a galaxy (arranged in ascending order)
        # idx_start: the index of 'sorted_indices' corresponding to each new pixel
        # count: the number of galaxies in each pixel
        idx, idx_start, count = np.unique(sorted_indices, return_counts=True, return_index=True)

        # splits indices into arrays - 1 per pixel
        res = np.split(idx_sort, idx_start[1:])

        keys = idx
        values = res
        dicts = {}
        for i, key in enumerate(keys):
            dicts[key] = values[i]

        return dicts
