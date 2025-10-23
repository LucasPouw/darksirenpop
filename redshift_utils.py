import numpy as np
from default_globals import *
from astropy.cosmology import z_at_value


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


def uniform_comoving_prior(z):
    '''Proportional to uniform in comoving volume prior.'''
    z = np.atleast_1d(z)
    chi = COSMO.comoving_distance(z).value         # Mpc
    H_z = COSMO.H(z).value                         # km/s/Mpc
    dchi_dz = SPEED_OF_LIGHT_KMS / H_z             # Mpc
    p = (chi**2 * dchi_dz)
    return p


def fast_z_at_value(function, values, num=100):
    zmin = z_at_value(function, values.min())
    zmax = z_at_value(function, values.max())
    zgrid = np.geomspace(zmin, zmax, num)
    valgrid = function(zgrid)
    zvals = np.interp(values.value, valgrid.value, zgrid)
    return zvals.value


def redshift_pdf_given_lumdist_pdf(z, lumdist_pdf, **kwargs):
    '''lumdist_pdf is assumed to be normalized'''
    dl = COSMO.luminosity_distance(z).value
    H_z = COSMO.H(z).value  # H(z) in km/s/Mpc
    chi_z = dl / (1 + z)
    dDL_dz = chi_z + (1 + z) * (SPEED_OF_LIGHT_KMS / H_z)
    return lumdist_pdf(dl, **kwargs) * dDL_dz
