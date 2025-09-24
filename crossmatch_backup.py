# #
# # Copyright (C) 2013-2024  Leo Singer
# #
# # This program is free software: you can redistribute it and/or modify
# # it under the terms of the GNU General Public License as published by
# # the Free Software Foundation, either version 3 of the License, or
# # (at your option) any later version.
# #
# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# # GNU General Public License for more details.
# #
# # You should have received a copy of the GNU General Public License
# # along with this program.  If not, see <https://www.gnu.org/licenses/>.
# #
# """Catalog cross matching for HEALPix sky maps."""
# from collections import namedtuple

# import astropy_healpix as ah
# from astropy.coordinates import ICRS, SkyCoord, SphericalRepresentation
# from astropy import units as u
# import healpy as hp
# import numpy as np

# from .. import distance
# from .. import moc
# from .util import interp_greedy_credible_levels

# from .cosmology import dVC_dVL_for_DL

# __all__ = ('crossmatch', 'CrossmatchResult')


# def flood_fill(nside, ipix, m, nest=False):
#     """Stack-based flood fill algorithm in HEALPix coordinates.

#     Based on <http://en.wikipedia.org/w/index.php?title=Flood_fill&oldid=566525693#Alternative_implementations>.
#     """  # noqa: E501
#     # Initialize stack with starting pixel index.
#     stack = [ipix]
#     while stack:
#         # Pop last pixel off of the stack.
#         ipix = stack.pop()
#         # Is this pixel in need of filling?
#         if m[ipix]:
#             # Fill in this pixel.
#             m[ipix] = False
#             # Find the pixels neighbors.
#             neighbors = hp.get_all_neighbours(nside, ipix, nest=nest)
#             # All pixels have up to 8 neighbors. If a pixel has less than 8
#             # neighbors, then some entries of the array are set to -1. We
#             # have to skip those.
#             neighbors = neighbors[neighbors != -1]
#             # Push neighboring pixels onto the stack.
#             stack.extend(neighbors)


# def count_modes(m, nest=False):
#     """Count the number of modes in a binary HEALPix image by repeatedly
#     applying the flood-fill algorithm.

#     WARNING: The input array is clobbered in the process.
#     """
#     npix = len(m)
#     nside = ah.npix_to_nside(npix)
#     for nmodes in range(npix):
#         nonzeroipix = np.flatnonzero(m)
#         if len(nonzeroipix):
#             flood_fill(nside, nonzeroipix[0], m, nest=nest)
#         else:
#             break
#     return nmodes


# def count_modes_moc(uniq, i):
#     n = len(uniq)
#     mask = np.concatenate((np.ones(i + 1, dtype=bool),
#                            np.zeros(n - i - 1, dtype=bool)))
#     sky_map = np.rec.fromarrays((uniq, mask), names=('UNIQ', 'MASK'))
#     sky_map = moc.rasterize(sky_map)['MASK']
#     return count_modes(sky_map, nest=True)


# def cos_angle_distance(theta0, phi0, theta1, phi1):
#     """Cosine of angular separation in radians between two points on the
#     unit sphere.
#     """
#     cos_angle_distance = (
#         np.cos(phi1 - phi0) * np.sin(theta0) * np.sin(theta1) +
#         np.cos(theta0) * np.cos(theta1))
#     return np.clip(cos_angle_distance, -1, 1)


# def angle_distance(theta0, phi0, theta1, phi1):
#     """Angular separation in radians between two points on the unit sphere."""
#     return np.arccos(cos_angle_distance(theta0, phi0, theta1, phi1))


# # Class to hold return value of find_injection method
# CrossmatchResult = namedtuple(
#     'CrossmatchResult',
#     'searched_area searched_prob offset searched_modes contour_areas '
#     'area_probs contour_modes searched_prob_dist contour_dists '
#     'searched_vol searched_prob_vol contour_vols probdensity probdensity_vol')
# """Cross match result as returned by
# :func:`~ligo.skymap.postprocess.crossmatch.crossmatch`.

# Notes
# -----
#  - All probabilities returned are between 0 and 1.
#  - All angles returned are in degrees.
#  - All areas returned are in square degrees.
#  - All distances are luminosity distances in units of Mpc.
#  - All volumes are in units of Mpc³. If :func:`.crossmatch` was run with
#    ``cosmology=False``, then all volumes are Euclidean volumes in luminosity
#    distance. If :func:`.crossmatch` was run with ``cosmology=True``, then all
#    volumes are comoving volumes.

# """
# _same_length_as_coordinates = ''' \
# Same length as the `coordinates` argument passed to \
# :func:`~ligo.skymap.postprocess.crossmatch.crossmatch`.'''
# _same_length_as_contours = ''' \
# of the probabilities specified by the `contour` argument passed to \
# :func:`~ligo.skymap.postprocess.crossmatch.crossmatch`.'''
# _same_length_as_areas = ''' \
# of the areas specified by the `areas` argument passed to
# :func:`~ligo.skymap.postprocess.crossmatch.crossmatch`.'''
# CrossmatchResult.searched_area.__doc__ = '''\
# Area within the 2D credible region containing each target \
# position.''' + _same_length_as_coordinates
# CrossmatchResult.searched_prob.__doc__ = '''\
# Probability within the 2D credible region containing each target \
# position.''' + _same_length_as_coordinates
# CrossmatchResult.offset.__doc__ = '''\
# Angles on the sky between the target positions and the maximum a posteriori \
# position.''' + _same_length_as_coordinates
# CrossmatchResult.searched_modes.__doc__ = '''\
# Number of disconnected regions within the 2D credible regions \
# containing each target position.''' + _same_length_as_coordinates
# CrossmatchResult.contour_areas.__doc__ = '''\
# Area within the 2D credible regions''' + _same_length_as_contours
# CrossmatchResult.area_probs.__doc__ = '''\
# Probability within the 2D credible regions''' + _same_length_as_areas
# CrossmatchResult.contour_modes.__doc__ = '''\
# Number of disconnected regions within the 2D credible \
# regions''' + _same_length_as_contours
# CrossmatchResult.searched_prob_dist.__doc__ = '''\
# Cumulative CDF of distance, marginalized over sky position, at the distance \
# of each of the targets.''' + _same_length_as_coordinates
# CrossmatchResult.contour_dists.__doc__ = '''\
# Distance credible interval, marginalized over sky \
# position,''' + _same_length_as_coordinates
# CrossmatchResult.searched_vol.__doc__ = '''\
# Volume within the 3D credible region containing each target \
# position.''' + _same_length_as_coordinates
# CrossmatchResult.searched_prob_vol.__doc__ = '''\
# Probability within the 3D credible region containing each target \
# position.''' + _same_length_as_coordinates
# CrossmatchResult.contour_vols.__doc__ = '''\
# Volume within the 3D credible regions''' + _same_length_as_contours
# CrossmatchResult.probdensity.__doc__ = '''\
# 2D probability density per steradian at the positions of each of the \
# targets.''' + _same_length_as_coordinates
# CrossmatchResult.probdensity_vol.__doc__ = '''\
# 3D probability density per cubic megaparsec at the positions of each of the \
# targets.''' + _same_length_as_coordinates


# def crossmatch(sky_map, coordinates=None,
#                contours=(), areas=(), modes=False, cosmology=False):
#     """Cross match a sky map with a catalog of points.

#     Given a sky map and the true right ascension and declination (in radians),
#     find the smallest area in deg^2 that would have to be searched to find the
#     source, the smallest posterior mass, and the angular offset in degrees from
#     the true location to the maximum (mode) of the posterior. Optionally, also
#     compute the areas of and numbers of modes within the smallest contours
#     containing a given total probability.

#     Parameters
#     ----------
#     sky_map : :class:`astropy.table.Table`
#         A multiresolution sky map, as returned by
#         :func:`ligo.skymap.io.fits.read_sky_map` called with the keyword
#         argument ``moc=True``.

#     coordinates : :class:`astropy.coordinates.SkyCoord`, optional
#         The catalog of target positions to match against.

#     contours : :class:`tuple`, optional
#         Credible levels between 0 and 1. If this argument is present, then
#         calculate the areas and volumes of the 2D and 3D credible regions that
#         contain these probabilities. For example, for ``contours=(0.5, 0.9)``,
#         then areas and volumes of the 50% and 90% credible regions.

#     areas : :class:`tuple`, optional
#         Credible areas in square degrees. If this argument is present, then
#         calculate the probability contained in the 2D credible levels that have
#         these areas. For example, for ``areas=(20, 100)``, then compute the
#         probability within the smallest credible levels of 20 deg² and 100
#         deg², respectively.

#     modes : :class:`bool`, optional
#         If True, then enable calculation of the number of distinct modes or
#         islands of probability. Note that this option may be computationally
#         expensive.

#     cosmology : :class:`bool`, optional
#         If True, then search space by descending probability density per unit
#         comoving volume. If False, then search space by descending probability
#         per luminosity distance cubed.

#     Returns
#     -------
#     result : :class:`~ligo.skymap.postprocess.crossmatch.CrossmatchResult`

#     Notes
#     -----
#     This function is also be used for injection finding; see
#     :doc:`/tool/ligo_skymap_stats`.

#     Examples
#     --------
#     First, some imports:

#     >>> from astroquery.vizier import VizierClass
#     >>> from astropy.coordinates import SkyCoord
#     >>> from ligo.skymap.io import read_sky_map
#     >>> from ligo.skymap.postprocess import crossmatch

#     Next, retrieve the GLADE catalog using Astroquery and get the coordinates
#     of all its entries:

#     >>> vizier = VizierClass(
#     ...     row_limit=-1,
#     ...     columns=['recno', 'GWGC', '_RAJ2000', '_DEJ2000', 'Dist'])
#     >>> cat, = vizier.get_catalogs('VII/281/glade2')
#     >>> cat.sort('recno')  # sort catalog so that doctest output is stable
#     >>> del cat['recno']
#     >>> coordinates = SkyCoord(cat['_RAJ2000'], cat['_DEJ2000'], cat['Dist'])

#     Load the multiresolution sky map for S190814bv:

#     >>> url = 'https://gracedb.ligo.org/api/superevents/S190814bv/files/bayestar.multiorder.fits'
#     >>> skymap = read_sky_map(url, moc=True)

#     Perform the cross match:

#     >>> result = crossmatch(skymap, coordinates)

#     Using the cross match results, we can list the galaxies within the 90%
#     credible volume:

#     >>> print(cat[result.searched_prob_vol < 0.9])
#          _RAJ2000          _DEJ2000        GWGC            Dist
#            deg               deg                           Mpc
#     ----------------- ----------------- ---------- --------------------
#       9.3396700000000 -19.9342460000000    NGC0171    57.56212553960000
#      20.2009090000000 -31.1146050000000        ---   137.16022925600001
#       8.9144680000000 -20.1252980000000 ESO540-003    49.07809291930000
#      10.6762720000000 -21.7740820000000        ---   276.46938505499998
#      13.5855170000000 -23.5523850000000        ---   138.44550704800000
#      20.6362970000000 -29.9825150000000        ---   160.23313164900000
#                   ...               ...        ...                  ...
#      10.6939000000000 -25.6778300000000        ---   323.59399999999999
#      15.4935000000000 -26.0305000000000        ---   304.78899999999999
#      15.2794000000000 -27.0411000000000        ---   320.62700000000001
#      14.8324000000000 -27.0460000000000        ---   320.62700000000001
#      14.5341000000000 -26.0949000000000        ---   307.61000000000001
#      23.1281000000000 -31.1109200000000        ---   320.62700000000001
#     Length = 1479 rows

#     """  # noqa: E501
#     # Astropy coordinates that are constructed without distance have
#     # a distance field that is unity (dimensionless).
#     if coordinates is None:
#         true_ra = true_dec = true_dist = None
#     else:
#         # Ensure that coordinates are in proper frame and representation
#         coordinates = SkyCoord(coordinates,
#                                representation_type=SphericalRepresentation,
#                                frame=ICRS)
#         true_ra = coordinates.ra.rad
#         true_dec = coordinates.dec.rad
#         if np.any(coordinates.distance != 1):
#             true_dist = coordinates.distance.to_value(u.Mpc)
#         else:
#             true_dist = None

#     contours = np.asarray(contours)

#     # Sort the pixels by descending posterior probability.
#     sky_map = np.flipud(np.sort(sky_map, order='PROBDENSITY'))

#     # Find the pixel that contains the injection.
#     order, ipix = moc.uniq2nest(sky_map['UNIQ'])
#     max_order = np.max(order)
#     max_nside = ah.level_to_nside(max_order)
#     max_ipix = ipix << np.int64(2 * (max_order - order))
#     if true_ra is not None:
#         true_theta = 0.5 * np.pi - true_dec
#         true_phi = true_ra
#         true_pix = hp.ang2pix(max_nside, true_theta, true_phi, nest=True)
#         i = np.argsort(max_ipix)
#         true_idx = i[np.digitize(true_pix, max_ipix[i]) - 1]

#     # Find the angular offset between the mode and true locations.
#     mode_theta, mode_phi = hp.pix2ang(
#         ah.level_to_nside(order[0]), ipix[0], nest=True)
#     if true_ra is None:
#         offset = np.nan
#     else:
#         offset = np.rad2deg(
#             angle_distance(true_theta, true_phi, mode_theta, mode_phi))

#     # Calculate the cumulative area in deg2 and the cumulative probability.
#     dA = moc.uniq2pixarea(sky_map['UNIQ'])
#     dP = sky_map['PROBDENSITY'] * dA
#     prob = np.cumsum(dP)
#     area = np.cumsum(dA) * np.square(180 / np.pi)

#     if true_ra is None:
#         searched_area = searched_prob = probdensity = np.nan
#     else:
#         # Find the smallest area that would have to be searched to find
#         # the true location.
#         searched_area = area[true_idx]

#         # Find the smallest posterior mass that would have to be searched to
#         # find the true location.
#         searched_prob = prob[true_idx]

#         # Find the probability density.
#         probdensity = sky_map['PROBDENSITY'][true_idx]

#     # Find the contours of the given credible levels.
#     contour_idxs = np.digitize(contours, prob) - 1

#     # For each of the given confidence levels, compute the area of the
#     # smallest region containing that probability.
#     contour_areas = interp_greedy_credible_levels(
#         contours, prob, area, right=4*180**2/np.pi).tolist()

#     # For each listed area, find the probability contained within the
#     # smallest credible region of that area.
#     area_probs = interp_greedy_credible_levels(
#         areas, area, prob, right=1).tolist()

#     if modes:
#         if true_ra is None:
#             searched_modes = np.nan
#         else:
#             # Count up the number of modes in each of the given contours.
#             searched_modes = count_modes_moc(sky_map['UNIQ'], true_idx)
#         contour_modes = [
#             count_modes_moc(sky_map['UNIQ'], i) for i in contour_idxs]
#     else:
#         searched_modes = np.nan
#         contour_modes = np.nan

#     # Distance stats now...
#     if 'DISTMU' in sky_map.dtype.names:
#         dP_dA = sky_map['PROBDENSITY']
#         mu = sky_map['DISTMU']
#         sigma = sky_map['DISTSIGMA']
#         norm = sky_map['DISTNORM']

#         # Set up distance grid.
#         n_r = 1000
#         distmean, _ = distance.parameters_to_marginal_moments(dP, mu, sigma)
#         max_r = 6 * distmean
#         if true_dist is not None and np.size(true_dist) != 0 \
#                 and np.max(true_dist) > max_r:
#             max_r = np.max(true_dist)
#         d_r = max_r / n_r

#         # Calculate searched_prob_dist and contour_dists.
#         r = d_r * np.arange(1, n_r)
#         P_r = distance.marginal_cdf(r, dP, mu, sigma, norm)
#         if true_dist is None:
#             searched_prob_dist = np.nan
#         else:
#             searched_prob_dist = interp_greedy_credible_levels(
#                 true_dist, r, P_r, right=1)
#         if len(contours) == 0:
#             contour_dists = []
#         else:
#             lo, hi = interp_greedy_credible_levels(
#                 np.vstack((
#                     0.5 * (1 - contours),
#                     0.5 * (1 + contours)
#                 )), P_r, r, right=np.inf)
#             contour_dists = (hi - lo).tolist()

#         # Calculate volume of each voxel, defined as the region within the
#         # HEALPix pixel and contained within the two centric spherical shells
#         # with radii (r - d_r / 2) and (r + d_r / 2).
#         dV = (np.square(r) + np.square(d_r) / 12) * d_r * dA.reshape(-1, 1)

#         # Calculate probability within each voxel.
#         dP = np.exp(
#             -0.5 * np.square(
#                 (r.reshape(1, -1) - mu.reshape(-1, 1)) / sigma.reshape(-1, 1)
#             )
#         ) * (dP_dA * norm / (sigma * np.sqrt(2 * np.pi))).reshape(-1, 1) * dV
#         dP[np.isnan(dP)] = 0  # Suppress invalid values

#         # Calculate probability density per unit volume.

#         if cosmology:
#             dV *= dVC_dVL_for_DL(r)
#         dP_dV = dP / dV
#         i = np.flipud(np.argsort(dP_dV.ravel()))

#         P_flat = np.cumsum(dP.ravel()[i])
#         V_flat = np.cumsum(dV.ravel()[i])

#         contour_vols = interp_greedy_credible_levels(
#             contours, P_flat, V_flat, right=np.inf).tolist()
#         P = np.empty_like(P_flat)
#         V = np.empty_like(V_flat)
#         P[i] = P_flat
#         V[i] = V_flat
#         P = P.reshape(dP.shape)
#         V = V.reshape(dV.shape)
#         if true_dist is None:
#             searched_vol = searched_prob_vol = probdensity_vol = np.nan
#         else:
#             i_radec = true_idx
#             i_dist = np.digitize(true_dist, r) - 1
#             probdensity_vol = dP_dV[i_radec, i_dist]
#             searched_prob_vol = P[i_radec, i_dist]
#             searched_vol = V[i_radec, i_dist]
#     else:
#         searched_vol = searched_prob_vol = searched_prob_dist \
#             = probdensity_vol = np.nan
#         contour_dists = [np.nan] * len(contours)
#         contour_vols = [np.nan] * len(contours)

#     # Done.
#     return CrossmatchResult(
#         searched_area, searched_prob, offset, searched_modes, contour_areas,
#         area_probs, contour_modes, searched_prob_dist, contour_dists,
#         searched_vol, searched_prob_vol, contour_vols, probdensity,
#         probdensity_vol)


#
# Copyright (C) 2013-2025  Leo Singer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""Catalog cross matching for HEALPix sky maps."""

from collections import namedtuple

import astropy_healpix as ah
import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord, SphericalRepresentation

from .. import distance, moc
from .cosmology import dVC_dVL_for_DL
from .util import interp_greedy_credible_levels

__all__ = ("crossmatch", "CrossmatchResult")


def flood_fill(nside, ipix, m, nest=False):
    """Stack-based flood fill algorithm in HEALPix coordinates.

    Based on <http://en.wikipedia.org/w/index.php?title=Flood_fill&oldid=566525693#Alternative_implementations>.
    """  # noqa: E501
    # Initialize stack with starting pixel index.
    stack = [ipix]
    while stack:
        # Pop last pixel off of the stack.
        ipix = stack.pop()
        # Is this pixel in need of filling?
        if m[ipix]:
            # Fill in this pixel.
            m[ipix] = False
            # Find the pixels neighbors.
            neighbors = hp.get_all_neighbours(nside, ipix, nest=nest)
            # All pixels have up to 8 neighbors. If a pixel has less than 8
            # neighbors, then some entries of the array are set to -1. We
            # have to skip those.
            neighbors = neighbors[neighbors != -1]
            # Push neighboring pixels onto the stack.
            stack.extend(neighbors)


def count_modes(m, nest=False):
    """Count the number of modes in a binary HEALPix image by repeatedly
    applying the flood-fill algorithm.

    WARNING: The input array is clobbered in the process.
    """
    npix = len(m)
    nside = ah.npix_to_nside(npix)
    for nmodes in range(npix):
        nonzeroipix = np.flatnonzero(m)
        if len(nonzeroipix):
            flood_fill(nside, nonzeroipix[0], m, nest=nest)
        else:
            break
    return nmodes


def count_modes_moc(uniq, i):
    n = len(uniq)
    mask = np.concatenate((np.ones(i + 1, dtype=bool), np.zeros(n - i - 1, dtype=bool)))
    sky_map = np.rec.fromarrays((uniq, mask), names=("UNIQ", "MASK"))
    sky_map = moc.rasterize(sky_map)["MASK"]
    return count_modes(sky_map, nest=True)


def cos_angle_distance(theta0, phi0, theta1, phi1):
    """Cosine of angular separation in radians between two points on the
    unit sphere.
    """
    cos_angle_distance = np.cos(phi1 - phi0) * np.sin(theta0) * np.sin(theta1) + np.cos(
        theta0
    ) * np.cos(theta1)
    return np.clip(cos_angle_distance, -1, 1)


def angle_distance(theta0, phi0, theta1, phi1):
    """Angular separation in radians between two points on the unit sphere."""
    return np.arccos(cos_angle_distance(theta0, phi0, theta1, phi1))


# Class to hold return value of find_injection method
CrossmatchResult = namedtuple(
    "CrossmatchResult",
    "searched_prob_vol probdensity_vol",
)
"""Cross match result as returned by
:func:`~ligo.skymap.postprocess.crossmatch.crossmatch`.

Notes
-----
 - All probabilities returned are between 0 and 1.
 - All angles returned are in degrees.
 - All areas returned are in square degrees.
 - All distances are luminosity distances in units of Mpc.
 - All volumes are in units of Mpc³. If :func:`.crossmatch` was run with
   ``cosmology=False``, then all volumes are Euclidean volumes in luminosity
   distance. If :func:`.crossmatch` was run with ``cosmology=True``, then all
   volumes are comoving volumes.

"""
_same_length_as_coordinates = """ \
Same length as the `coordinates` argument passed to \
:func:`~ligo.skymap.postprocess.crossmatch.crossmatch`."""
CrossmatchResult.searched_prob_vol.__doc__ = (
    """\
Probability within the 3D credible region containing each target \
position."""
    + _same_length_as_coordinates
)
CrossmatchResult.probdensity_vol.__doc__ = (
    """\
3D probability density per cubic megaparsec at the positions of each of the \
targets."""
    + _same_length_as_coordinates
)



def crossmatchDONTUSE(
    sky_map, coordinates, cosmology=False
):
    
    ######### 1. LOAD CATALOG COORDINATES #########

    print('USING CLEANED OLD CROSSMATCH')

    # Ensure that coordinates are in proper frame and representation
    coordinates = SkyCoord(coordinates, representation_type=SphericalRepresentation, frame=ICRS)

    true_ra = coordinates.ra.rad
    true_dec = coordinates.dec.rad
    true_dist = coordinates.distance.to_value(u.Mpc)

    ###############################################

    # Sort the pixels by descending posterior probability.
    sky_map = np.flipud(np.sort(sky_map, order="PROBDENSITY"))

    # Find the pixel that contains the injection.
    order, ipix = moc.uniq2nest(sky_map["UNIQ"])
    max_order = np.max(order)
    max_nside = ah.level_to_nside(max_order)
    max_ipix = ipix << np.int64(2 * (max_order - order))

    true_theta = 0.5 * np.pi - true_dec
    true_phi = true_ra
    true_pix = hp.ang2pix(max_nside, true_theta, true_phi, nest=True)
    i = np.argsort(max_ipix)
    true_idx = i[np.digitize(true_pix, max_ipix[i]) - 1]

    # Calculate the cumulative area in deg2 and the cumulative probability.
    dA = moc.uniq2pixarea(sky_map["UNIQ"])
    dP = sky_map["PROBDENSITY"] * dA

    # Distance stats now...
    if "DISTMU" in sky_map.dtype.names:
        dP_dA = sky_map["PROBDENSITY"]
        mu = sky_map["DISTMU"]
        sigma = sky_map["DISTSIGMA"]
        norm = sky_map["DISTNORM"]

        # Set up distance grid.
        n_r = 1000
        distmean, _ = distance.parameters_to_marginal_moments(dP, mu, sigma)
        max_r = 6 * distmean
        if (
            true_dist is not None
            and np.size(true_dist) != 0
            and np.max(true_dist) > max_r
        ):
            max_r = np.max(true_dist)
        d_r = max_r / n_r

        r = d_r * np.arange(1, n_r)

        # Calculate volume of each voxel, defined as the region within the
        # HEALPix pixel and contained within the two centric spherical shells
        # with radii (r - d_r / 2) and (r + d_r / 2).
        dV = (np.square(r) + np.square(d_r) / 12) * d_r * dA.reshape(-1, 1)

        # Calculate probability within each voxel.
        dP = (
            np.exp(
                -0.5
                * np.square(
                    (r.reshape(1, -1) - mu.reshape(-1, 1)) / sigma.reshape(-1, 1)
                )
            )
            * (dP_dA * norm / (sigma * np.sqrt(2 * np.pi))).reshape(-1, 1)
            * dV
        )
        dP[np.isnan(dP)] = 0  # Suppress invalid values

        # Calculate probability density per unit volume.

        if cosmology:
            dV *= dVC_dVL_for_DL(r)
        dP_dV = dP / dV
        i = np.flipud(np.argsort(dP_dV.ravel()))

        P_flat = np.cumsum(dP.ravel()[i])
        V_flat = np.cumsum(dV.ravel()[i])

        P = np.empty_like(P_flat)
        V = np.empty_like(V_flat)
        P[i] = P_flat
        V[i] = V_flat
        P = P.reshape(dP.shape)
        V = V.reshape(dV.shape)
        
        i_radec = true_idx
        i_dist = np.digitize(true_dist, r) - 1
        probdensity_vol = dP_dV[i_radec, i_dist]
        searched_prob_vol = P[i_radec, i_dist]

    # Done.
    return CrossmatchResult(
        searched_prob_vol,
        probdensity_vol
    )

import matplotlib.pyplot as plt
import time
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c
from scipy.integrate import romb, quad, simpson
import sys
from numba import njit, prange
from scipy import stats
import h5py
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


INV_SQRT2PI = 1.0 / np.sqrt(2.0 * np.pi)

@njit(parallel=True, fastmath=True)
def allsky_marginal_lumdist_distribution(dl_array, dP, norm, mu, sigma):
    M = dl_array.shape[0]
    N = mu.shape[0]
    out = np.empty(M)

    for j in prange(M):                 # parallel over dl_array
        dl = dl_array[j]
        s = 0.0
        for i in range(N):              # fast inner loop
            diff = (dl - mu[i]) / sigma[i]
            gauss = np.exp(-0.5 * diff * diff) * INV_SQRT2PI / sigma[i]
            s += dP[i] * norm[i] * dl * dl * gauss
        out[j] = s
    return out


# def allsky_marginal_lumdist_distribution(dl_array, dP, norm, mu, sigma):
#     dl_array = np.atleast_1d(dl_array)             # shape (M,)
#     # Broadcast: (M,1) vs (N,)
#     dl = dl_array[:, None]                         # (M,1)
#     mu = mu[None, :]                               # (1,N)
#     sigma = sigma[None, :]                         # (1,N)
#     norm = norm[None, :]
#     dP = dP[None, :]

#     gauss = np.exp(-0.5 * ((dl - mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))
#     result = np.sum(dP * norm * (dl**2) * gauss, axis=1)  # sum over N
#     return result


SPEED_OF_LIGHT_KMS = c.to('km/s').value
COSMO = FlatLambdaCDM(H0=67.9, Om0=0.3065)


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)


def LOS_lumdist_ansatz(dl, distnorm, distmu, distsigma):
    'The ansatz is normalized on dL [0, large]'     
    return dl**2 * distnorm * gaussian(dl, distmu, distsigma)


def redshift_pdf_given_lumdist_pdf(z, lumdist_pdf, **kwargs):
    '''lumdist_pdf is assumed to be normalized'''
    dl = COSMO.luminosity_distance(z).value
    H_z = COSMO.H(z).value  # H(z) in km/s/Mpc
    chi_z = dl / (1 + z)
    dDL_dz = chi_z + (1 + z) * (SPEED_OF_LIGHT_KMS / H_z)
    return lumdist_pdf(dl, **kwargs) * dDL_dz


def uniform_comoving_prior(z):
    '''Proportional to uniform in comoving volume prior.'''
    z = np.atleast_1d(z)
    chi = COSMO.comoving_distance(z).value         # Mpc
    H_z = COSMO.H(z).value                         # km/s/Mpc
    dchi_dz = SPEED_OF_LIGHT_KMS / H_z             # Mpc
    p = (chi**2 * dchi_dz)
    return p


def z_cut(z, zcut):
    '''Artificial redshift cuts in data analysis should be taken into account.'''
    stepfunc = np.ones_like(z)
    stepfunc[z > zcut] = 0
    return stepfunc


def merger_rate_madau(z, gamma=4.59, k=2.86, zp=2.47):
        """
        Madau rate evolution
        """
        C = 1 + (1 + zp)**(-gamma - k)
        return C * ((1 + z)**gamma) / (1 + ((1 + z) / (1 + zp))**(gamma + k))


def merger_rate_uniform(z):
    return np.ones_like(z)


def merger_rate(z, func, **kwargs):
    return func(z, **kwargs)


def crossmatch(posterior_path, 
               sky_map,
               completeness_map,
               completeness_zedges,
               completeness_zvals,
               agn_ra, 
               agn_dec, 
               agn_lumdist, 
               agn_redshift, 
               skymap_cl, 
               agn_redshift_err, 
               assume_perfect_redshift, 
               s_agn_z_integral_ax, 
               s_alt_z_integral_ax,
               gw_zcut,
               merger_rate_func=merger_rate_uniform, 
               **merger_rate_kwargs):

    if not assume_perfect_redshift:
        assert np.max(np.diff(s_agn_z_integral_ax)[0]) < np.min(agn_redshift_err), 'LOS zprior resolution is too coarse to capture AGN distribution fully.'

    dz_Sagn = np.diff(np.log10(s_agn_z_integral_ax))[0]
    dz_Salt = np.diff(np.log10(s_alt_z_integral_ax))[0]

    cmap_nside = hp.npix2nside(len(completeness_map))

    def redshift_completeness(z):
        bin_idx = np.digitize(z, completeness_zedges) - 1
        bin_idx[bin_idx == len(completeness_zvals)] = len(completeness_zvals) - 1
        return completeness_zvals[bin_idx.astype(np.int32)]
    
    # zz = np.linspace(0, 1.5, 1000)
    # plt.figure()
    # plt.plot(zz, redshift_completeness(zz))
    # plt.savefig('cz.pdf', bbox_inches='tight')
    # plt.close()
    # sys.exit(1)

    print(f'USING NEW CROSSMATCH, PERFECT REDSHIFT = {assume_perfect_redshift}')

    sky_map = np.flipud(np.sort(sky_map, order="PROBDENSITY"))
    
    # Unpacking skymap
    dP_dA = sky_map["PROBDENSITY"]  # Probdens in 1/sr
    mu = sky_map["DISTMU"]          # Ansatz mean in Mpc
    sigma = sky_map["DISTSIGMA"]    # Ansatz width in Mpc
    norm = sky_map["DISTNORM"]      # Ansatz norm in 1/Mpc

    if np.sum(np.isnan(dP_dA)) == len(dP_dA):
        print('BAD SKYMAP AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        return np.nan, np.nan, np.nan

    # Find the pixel that contains the injection.
    order, ipix = moc.uniq2nest(sky_map["UNIQ"])
    max_order = np.max(order)
    max_nside = ah.level_to_nside(max_order)
    max_ipix = ipix << np.int64(2 * (max_order - order))

    agn_theta = 0.5 * np.pi - agn_dec
    agn_phi = agn_ra
    true_pix = hp.ang2pix(max_nside, agn_theta, agn_phi, nest=True)
    i = np.argsort(max_ipix)
    true_idx = i[np.digitize(true_pix, max_ipix[i]) - 1]  # Indeces that indicate skymap pixels that contain an AGN

    dA = moc.uniq2pixarea(sky_map["UNIQ"])  # Pixel areas in sr
    dP = dP_dA * dA  # Dimensionless probability density in each pixel
    # print('VERIFICATION:', np.sum(dP))
    cumprob = np.cumsum(dP)
    cumprob[cumprob > 1] = 1.  # Correcting floating point error which could cause issues when skymap_cl == 1
    searched_prob = cumprob[true_idx]

    # Getting only relevant AGN and pixels from the skymap
    agn_mask_within_cl = (searched_prob <= skymap_cl)
    gw_idx_with_agn = true_idx[agn_mask_within_cl]

    unique_gw_idx, counts = np.unique(gw_idx_with_agn, return_counts=True)
    nagn_within_cl = np.sum(counts)
    print(f'Found {nagn_within_cl} AGN within {skymap_cl} CL in {len(unique_gw_idx)} pixels')

    # Calculating the evidence for each hypothesis
    print('Calculating S_agn...')
    S_agn_cw = 0
    if assume_perfect_redshift:
        print('PERFECT REDSHIFT NOT YET IMPLEMENTED WITH VARYING MERGER RATE AND ZCUTS FOR AGN AND GWS, NOR IS COMPLETENESS THERE')

        for _, has_agn in zip(counts, unique_gw_idx):
            # print(f'Pixel: {has_agn} contains {count} AGN')
            gw_redshift_posterior_in_pix = lambda z: redshift_pdf_given_lumdist_pdf(z, LOS_lumdist_ansatz, distnorm=norm[has_agn], distmu=mu[has_agn], distsigma=sigma[has_agn])

            # Only evaluate the GW posterior for AGN within 5sigma. If outside this range, the probability will be floored to 0. The AGN is still counted in the normalization.
            lumdist_5sig = (agn_lumdist[true_idx == has_agn] > (mu[has_agn] - 5 * sigma[has_agn])) & (agn_lumdist[true_idx == has_agn] < (mu[has_agn] + 5 * sigma[has_agn]))
            if np.sum(lumdist_5sig) == 0:  # The contribution to S_agn is 0 in this pixel, so skip
                # print('NEVERMIND, SKIPPING THIS PIXEL, BUT SHOULD I?????????????????????')
                continue
            # print(len(agn_redshift[true_idx == has_agn][lumdist_5sig]))

            redshift_probdens = gw_redshift_posterior_in_pix(agn_redshift[true_idx == has_agn][lumdist_5sig]) #* merger_rate(agn_redshift[true_idx == has_agn][lumdist_5sig], merger_rate_func, **merger_rate_kwargs)
            contribution = np.sum( redshift_probdens * dP_dA[has_agn] )  # p_gw(z) * p_gw(Omega), evaluated at AGN position because of delta-function AGN posteriors, sum contributions of all AGN in this pixel
            S_agn += contribution  # Will error, should be S_agn_cw, but not yet implemented
    
    else:

        ### gw_redshift_posterior_in_pix evaluation and AGN posterior loading are the most expensive operations right now, dominating S_agn computation time ###

        # start = time.time()
        # # post_timer = 0

        # with h5py.File(posterior_path, "r") as f:
        #     dset = f["agn_redshift_posteriors"]
            
        #     for nagn_at_idx, gw_idx in zip(counts, unique_gw_idx):
        #         agn_indices = np.where(true_idx == gw_idx)[0]  # TODO: precompute outside loop
        #         agn_redshift_posterior = dset[agn_indices, :]

        #         phi = agn_phi[agn_indices]
        #         theta = agn_theta[agn_indices]
        #         pix_idx = hp.ang2pix(cmap_nside, theta, phi, nest=True)
        #         completenesses = completeness_map[pix_idx]

        #         # start_post_timer = time.time()
        #         gw_redshift_posterior_in_pix = redshift_pdf_given_lumdist_pdf(s_agn_z_integral_ax, LOS_lumdist_ansatz, distnorm=norm[gw_idx], distmu=mu[gw_idx], distsigma=sigma[gw_idx])
        #         # post_timer += time.time() - start_post_timer

        #         # Population prior should be normalized on z \in [0, infty], but the z_cut makes the integral z \in [0, z_cut], which could be taken into account in the range of S_AGN_Z_INTEGRAL_AX, if you want, but not necessary because of the z_cut() function:
        #         agn_population_prior_unnorm = agn_redshift_posterior * z_cut(s_agn_z_integral_ax, zcut=gw_zcut) * merger_rate(s_agn_z_integral_ax, merger_rate_func, **merger_rate_kwargs) 
        #         agn_population_norms = romb(y=agn_population_prior_unnorm * s_agn_z_integral_ax * np.log(10), dx=dz_Sagn)
        #         # print(agn_population_norms)
                
        #         agn_population_prior = np.sum(completenesses[:,np.newaxis] * agn_population_prior_unnorm / agn_population_norms[:,np.newaxis], axis=0)  # Combine all AGN that talks to the same part of the GW posterior

        #         # Multiply LOS prior with GW posterior and integrate
        #         gw_posterior_times_agn_population_prior = gw_redshift_posterior_in_pix * agn_population_prior * redshift_completeness(s_agn_z_integral_ax)
        #         contribution = dP_dA[gw_idx] * romb(gw_posterior_times_agn_population_prior * s_agn_z_integral_ax * np.log(10), dx=dz_Sagn)  # * nagn_at_idx
                
        #         S_agn_cw += contribution

        # print('SINGLE CORE GOT:', S_agn_cw)
        # print('Total Sagn time:', time.time() - start)


        mp_start = time.time()

        def process_one(args):
            gw_idx, nagn_at_idx = args

            agn_indices = np.where(true_idx == gw_idx)[0]
            
            with h5py.File(posterior_path, "r") as f:
                dset = f["agn_redshift_posteriors"]
                agn_redshift_posterior = dset[agn_indices, :]

            gw_redshift_posterior_in_pix = redshift_pdf_given_lumdist_pdf(
                s_agn_z_integral_ax, LOS_lumdist_ansatz,
                distnorm=norm[gw_idx], distmu=mu[gw_idx], distsigma=sigma[gw_idx]
            )

            # The population prior consists of AGN posteriors, modulated by redshift evolving merger rates
            agn_population_prior_unnorm = (
                agn_redshift_posterior
                * z_cut(s_agn_z_integral_ax, zcut=gw_zcut)
                * merger_rate(s_agn_z_integral_ax, merger_rate_func, **merger_rate_kwargs)
            )
            # Make sure the prior is normalized, which may not be the case due to the redshift evolution modulation
            agn_population_norms = romb(
                y=agn_population_prior_unnorm * s_agn_z_integral_ax * np.log(10),
                dx=dz_Sagn,
            )

            # We can vectorize the contributions of AGN with the same gw_idx
            # TODO: the completeness could differ for AGN with the same gw_idx + sky completeness and redshift completeness should be combined in a 3D map -> redshift_completeness(z, pix)
            agn_population_prior = np.sum(
                redshift_completeness(s_agn_z_integral_ax)  # TODO: make different for each AGN
                * agn_population_prior_unnorm
                / agn_population_norms[:, np.newaxis],
                axis=0,
            )

            gw_posterior_times_agn_population_prior = (
                gw_redshift_posterior_in_pix
                * agn_population_prior
            )
            contribution = dP_dA[gw_idx] * romb(
                gw_posterior_times_agn_population_prior
                * s_agn_z_integral_ax
                * np.log(10),
                dx=dz_Sagn,
            )

            # phi = agn_phi[agn_indices]
            # theta = agn_theta[agn_indices]
            # pix_idx = hp.ang2pix(cmap_nside, theta, phi, nest=True)
            # completeness_at_agn_skypos = completeness_map[pix_idx]

            # contribution = np.sum(dP_dA[gw_idx] * completeness_at_agn_skypos)  # WHEN USING ONLY SKYLOC

            return contribution


        tasks = list(zip(unique_gw_idx, counts))
        with ThreadPoolExecutor() as executor:
            contributions = list(executor.map(process_one, tasks))

        S_agn_cw = np.sum(contributions)

        print('Total Sagn time:', time.time() - mp_start)

        # if count != 1:
        #     print('AYOOOOOOOOOOOOOOOOO')
        #     print(f'Redshifts are {agn_redshift[true_idx == has_agn]}')
        #     print('contribution to S_agn:', contribution)

        #     if np.isnan(S_agn):
        #         print('\n!!! This AGN caused a problem !!!\n')
        
        # plt.figure()
        # z_axis = np.linspace(0, 2, 1000)
        # # plt.plot(z_axis, gw_redshift_posterior_in_pix(z_axis), label='GW posterior', color='blue')
        # plt.plot(z_axis, uniform_comoving_prior(z_axis), label='p_alt', color='red')
        # # plt.plot(z_axis, dVdz_populationprior(z_axis) * gw_redshift_posterior_in_pix(z_axis), label=r'$p(z|d)p_{\rm alt}(z)$', color='teal')
        # plt.vlines(agn_redshift[true_idx == has_agn], 0, 5, label='AGN posterior', color='black')
        # plt.xlabel('Redshift')
        # plt.ylabel('Probability density')
        # # plt.plot(z_axis, agn_redshift_posterior(z_axis), label='AGN')
        # plt.legend()
        # plt.title(f'AGN {count}, pixel {has_agn}')
        # plt.savefig('/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/zpost.pdf', bbox_inches='tight')
        # plt.close()

        # time.sleep(2)


    S_agn_cw *= (4 * np.pi / len(agn_ra))
    
    skymap_theta, skymap_phi = moc.uniq2ang(sky_map['UNIQ'])
    pix_idx = hp.ang2pix(cmap_nside, skymap_theta, skymap_phi, nest=True)

    # pixprob_within_cl = (cumprob <= skymap_cl)
    # completenesses = completeness_map[ pix_idx[pixprob_within_cl] ]  # WHEN USING ONLY SKYLOC

    cmap_vals_in_gw_skymap = completeness_map[pix_idx]
    surveyed = (cmap_vals_in_gw_skymap != 0)

    sky_coverage = np.sum(dA[surveyed]) / np.sum(dA)
    S_agn_cw *= sky_coverage
    print(f'SKY COVERAGE: {sky_coverage}')

    print('Calculating S_alt...')
    S_alt = 1
    
    # S_alt_cw = np.sum(dP * cmap_vals_in_gw_skymap)  # WHEN USING ONLY SKYLOC


    gw_redshift_posterior_marginalized_cw = lambda z: redshift_pdf_given_lumdist_pdf(z, 
                                                                                    allsky_marginal_lumdist_distribution, 
                                                                                    dP=dP[surveyed],
                                                                                    norm=norm[surveyed], 
                                                                                    mu=mu[surveyed], 
                                                                                    sigma=sigma[surveyed])
    
    alt_redshift_population_prior_rate_weighted = uniform_comoving_prior(s_alt_z_integral_ax) * merger_rate(s_alt_z_integral_ax, merger_rate_func, **merger_rate_kwargs) * z_cut(s_alt_z_integral_ax, zcut=gw_zcut)
    alt_redshift_population_prior_rate_weighted /= romb(alt_redshift_population_prior_rate_weighted * s_alt_z_integral_ax * np.log(10), dx=dz_Salt)

    gw_posterior_times_cw_alt_population_prior = gw_redshift_posterior_marginalized_cw(s_alt_z_integral_ax) * alt_redshift_population_prior_rate_weighted * redshift_completeness(s_alt_z_integral_ax)  # TODO: only works now with uniform-on-sky redshift completeness
    S_alt_cw = romb(y=gw_posterior_times_cw_alt_population_prior * s_alt_z_integral_ax * np.log(10), dx=dz_Salt)

    if S_alt_cw > 1:
        print(S_alt_cw, 'bruh')
        time.sleep(1)

    S_alt_cw = min(S_alt_cw, 1)  # TODO: figure out why this goes wrong sometimes - maybe just numerical error?
        
    return S_agn_cw, S_alt_cw, S_alt
