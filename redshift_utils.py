import numpy as np
from default_globals import *
from astropy.cosmology import z_at_value
import astropy.units as u


def z_cut(z, zcut):
    '''
    Heavyside step function. Equal to 1 below zcut, equal to 0 above zcut.
    '''
    stepfunc = np.ones_like(z)
    stepfunc[z > zcut] = 0
    return stepfunc


def time_dilation_correction(z):
     return 1 / (1 + z)


def merger_rate_madau_lvk(z, gamma=4.59, k=2.86, zp=2.47):
        """
        Madau-Dickinson-like merger rate evolution as measured from GWs by LVK
        """
        return (1 + z)**gamma * (1 + (1 + zp)**(-gamma - k)) / (1 + ((1 + z) / (1 + zp))**(gamma + k))


def merger_rate_madau_dickinson(z, b = 2.7, c = 2.9, d = 5.6):
    """
    Cosmic SFR (no metallicity dependence)
    """
    return (1 + z)**b / (1 + ((1 + z) / c)**d)


def merger_rate_uniform(z):
    return np.ones_like(z)


def merger_rate(z, func, **kwargs):
    return func(z, **kwargs)


def uniform_comoving_prior(z, cosmo=COSMO):
    '''Proportional to uniform in comoving volume prior.'''
    z = np.atleast_1d(z)
    chi = cosmo.comoving_distance(z).value         # Mpc
    H_z = cosmo.H(z).value                         # km/s/Mpc
    dchi_dz = SPEED_OF_LIGHT_KMS / H_z             # Mpc
    p = (chi**2 * dchi_dz)
    return p


def uniform_source_frame(z):
    return uniform_comoving_prior(z) / (1 + z)


def ddl_dz(dl, z, H_z):
    return dl / (1 + z) + (1 + z) * SPEED_OF_LIGHT_KMS / H_z


def uniform_source_frame_dl(dl, cosmo=COSMO):
    """
    Prior proportional to uniform merger rate in comoving volume
    and source-frame time, expressed as a function of luminosity distance.
    """
    dl = np.atleast_1d(dl)
    z = fast_z_at_value(cosmo.luminosity_distance, dl * u.Mpc)
    chi = cosmo.comoving_distance(z).value
    H_z = cosmo.H(z).value
    dchi_dz = SPEED_OF_LIGHT_KMS / H_z
    p = (chi**2 * dchi_dz) / ((1+z) * ddl_dz(dl, z, H_z))
    return p


def fast_z_at_value(function, values, num=100):
    zmin = z_at_value(function, values.min())
    zmax = z_at_value(function, values.max())
    zgrid = np.geomspace(zmin, zmax, num)
    valgrid = function(zgrid)
    zvals = np.interp(values.value, valgrid.value, zgrid)
    return zvals.value


def redshift_pdf_given_lumdist_pdf(z, lumdist_pdf, cosmo=COSMO, **kwargs):
    '''lumdist_pdf is assumed to be normalized'''
    dl = cosmo.luminosity_distance(z).value
    H_z = cosmo.H(z).value  # H(z) in km/s/Mpc
    chi_z = dl / (1 + z)
    dDL_dz = chi_z + (1 + z) * (SPEED_OF_LIGHT_KMS / H_z)
    return lumdist_pdf(dl, **kwargs) * dDL_dz


def comdist_pdf_given_redshift_pdf(dc, redshift_pdf, cosmo=COSMO, **kwargs):
    """redshift_pdf is assumed to be normalized"""
    z = fast_z_at_value(cosmo.comoving_distance, dc * u.Mpc)
    H_z = cosmo.H(z).value  # H(z) in km/s/Mpc
    dz_dDc = H_z / SPEED_OF_LIGHT_KMS
    return redshift_pdf(z, **kwargs) * dz_dDc


def det2source_jacobian(z, cosmo=COSMO):
    """
    (1+z)^2 * ddL/dz
    """
    H_z = cosmo.H(z).value  # H(z) in km/s/Mpc
    dl = cosmo.luminosity_distance(z).value
    return np.power(1 + z, 2) * ddl_dz(dl, z, H_z)


if __name__ == '__main__':

    from utils import uniform_shell_sampler
    import astropy.units as u
    import matplotlib.pyplot as plt
    from scipy.integrate import romb


    r, t, p = uniform_shell_sampler(COSMO.comoving_distance(1e-6).value, COSMO.comoving_distance(10).value, n_samps=int(1e7))
    zsamp = fast_z_at_value(COSMO.comoving_distance, r * u.Mpc)
    weights = 1 / (1 + zsamp)
    zsamp_reweight = np.random.choice(zsamp, size=int(1e6), p=weights / np.sum(weights))
    dlsamp_reweight = COSMO.luminosity_distance(zsamp_reweight).value

    z = np.linspace(1e-6, 10, 1024+1)
    lumdist = np.linspace(COSMO.luminosity_distance(1e-6).value, COSMO.luminosity_distance(10).value, 1024+1)
    
    dlpdf = uniform_source_frame_dl(lumdist) / romb(uniform_source_frame_dl(lumdist), dx=np.diff(lumdist)[0])
    zpdf = uniform_source_frame(z) / romb(uniform_source_frame(z), dx=np.diff(z)[0])

    fig, ax = plt.subplots()
    # ax.plot(lumdist, lumdist**2 / romb(lumdist**2, dx=np.diff(lumdist)[0]), color='orange')
    ax.plot(lumdist, dlpdf, color='blue')
    ax.hist(dlsamp_reweight, density=True, bins=50, histtype='step', color='steelblue')
    # ax2 = ax.twiny()
    # ax2.plot(z, zpdf, color='red')
    # ax2.hist(zsamp_reweight, density=True, bins=50, histtype='step', color='coral')
    plt.show()