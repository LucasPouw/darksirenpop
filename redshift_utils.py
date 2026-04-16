import numpy as np
from default_globals import *

from astropy.cosmology import z_at_value
import astropy.units as u

from scipy.interpolate import CubicSpline


# def fast_z_at_value(function, values, num=1000):
#     zmin = z_at_value(function, values.min())
#     zmax = z_at_value(function, values.max())
#     zgrid = np.geomspace(zmin, zmax, num)
#     valgrid = function(zgrid)
#     return CubicSpline(valgrid.value, zgrid)(values.value)

def fast_z_at_value(function, values, num=100):
    zmin = z_at_value(function, values.min())
    zmax = z_at_value(function, values.max())
    zgrid = np.geomspace(zmin, zmax, num)
    valgrid = function(zgrid)
    zvals = np.interp(values.value, valgrid.value, zgrid)
    return zvals.value


def make_cosmo_interpolators(z_min=0.0, z_max=10.0, n_points=10_000, cosmo=COSMO):  # TODO: call this function at the start of simulation to update the COSMO if necessary
    """Precompute chi(z), dl(z) and H(z) once, return fast interpolators."""
    z_grid = np.linspace(z_min, z_max, n_points)

    chi_grid = cosmo.comoving_distance(z_grid).value
    dl_grid = cosmo.luminosity_distance(z_grid).value
    H_grid  = cosmo.H(z_grid).value
    
    chi_interp = CubicSpline(z_grid, chi_grid)
    dl_interp = CubicSpline(z_grid, dl_grid)
    H_interp  = CubicSpline(z_grid, H_grid)
    return chi_interp, dl_interp, H_interp


_CHI_INTERP, _DL_INTERP, _H_INTERP = make_cosmo_interpolators()


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


def uniform_comoving_prior(z, chi_interp=_CHI_INTERP, H_interp=_H_INTERP, cosmo=COSMO):
    '''Proportional to uniform in comoving volume prior.'''
    z = np.atleast_1d(z)
    return chi_interp(z)**2 * SPEED_OF_LIGHT_KMS / H_interp(z)


def uniform_source_frame(z):
    return uniform_comoving_prior(z) / (1 + z)


def ddl_dz(dl, z, H_z):
    return dl / (1 + z) + (1 + z) * SPEED_OF_LIGHT_KMS / H_z


def uniform_source_frame_dl(dl, chi_interp=_CHI_INTERP, H_interp=_H_INTERP, cosmo=COSMO):
    """
    Prior proportional to uniform merger rate in comoving volume
    and source-frame time, expressed as a function of luminosity distance.
    """
    dl = np.atleast_1d(dl)
    z = fast_z_at_value(cosmo.luminosity_distance, dl * u.Mpc)
    chi = chi_interp(z)
    H_z = H_interp(z)
    dchi_dz = SPEED_OF_LIGHT_KMS / H_z
    p = (chi**2 * dchi_dz) / ((1+z) * ddl_dz(dl, z, H_z))
    return p


def redshift_pdf_given_lumdist_pdf(z, lumdist_pdf, dl_interp=_DL_INTERP, H_interp=_H_INTERP, cosmo=COSMO, **kwargs):
    '''lumdist_pdf is assumed to be normalized. Uses precomputed interpolators.'''
    z = np.asarray(z)
    dl = dl_interp(z)
    H_z = H_interp(z)

    out = lumdist_pdf(dl, **kwargs)
    out *= dl / (1 + z) + (1 + z) * (SPEED_OF_LIGHT_KMS / H_z)  # dDL_dz
    return out


def comdist_pdf_given_redshift_pdf(dc, redshift_pdf, H_interp=_H_INTERP, cosmo=COSMO, **kwargs):
    """redshift_pdf is assumed to be normalized"""
    z = fast_z_at_value(cosmo.comoving_distance, dc * u.Mpc)
    dz_dDc = H_interp(z) / SPEED_OF_LIGHT_KMS  # H(z) in km/s/Mpc
    return redshift_pdf(z, **kwargs) * dz_dDc


def det2source_jacobian(z, dl_interp=_DL_INTERP, H_interp=_H_INTERP, cosmo=COSMO):
    """
    (1+z)^2 * ddL/dz
    """
    H_z = H_interp(z)  # H(z) in km/s/Mpc
    dl = dl_interp(z)
    return np.power(1 + z, 2) * ddl_dz(dl, z, H_z)


def LOS_lumdist_ansatz(dl, distnorm, distmu, distsigma, out=None):
    'The ansatz is normalized on dL [0, large]'

    # Ensure shape
    distnorm = np.asarray(distnorm)
    distmu = np.asarray(distmu)
    distsigma = np.asarray(distsigma)
    dl = np.asarray(dl)

    shape = np.broadcast(dl, distmu, distsigma, distnorm).shape
    out = np.empty(shape, dtype=np.float32)
    
    # Calculate exponential: dl**2 * np.exp(-0.5 * ((dl - distmu) / distsigma)**2)
    np.subtract(dl, distmu, out=out)
    out /= distsigma
    np.square(out, out=out)
    out *= -0.5
    np.exp(out, out=out)
    out *= dl * dl
    
    # Normalization: distnorm / (distsigma * np.sqrt(2 * np.pi))
    out *= distnorm / (distsigma * np.sqrt(2 * np.pi)) 
    
    return out


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    from line_profiler import LineProfiler

 
    

    # def LOS_lumdist_ansatz(dl, distnorm, distmu, distsigma, out=None):
    #     'The ansatz is normalized on dL [0, large]'
        
    #     # Ensure shape
    #     distnorm = np.asarray(distnorm)
    #     distmu = np.asarray(distmu)
    #     distsigma = np.asarray(distsigma)
    #     dl = np.asarray(dl)

    #     shape = np.broadcast(dl, distmu, distsigma, distnorm).shape
    #     out = np.empty(shape, dtype=np.float32)
        
    #     # Calculate exponential: dl**2 * np.exp(-0.5 * ((dl - distmu) / distsigma)**2)
    #     np.subtract(dl, distmu, out=out)
    #     out /= distsigma
    #     np.square(out, out=out)
    #     out *= -0.5
    #     np.exp(out, out=out)
    #     out *= dl * dl

    #     # Normalization: distnorm / (distsigma * np.sqrt(2 * np.pi))
    #     out *= distnorm / (distsigma * np.sqrt(2 * np.pi)) 
        
    #     return out

    N = 10000
    x = np.linspace(100, 1000, N)
    z = np.linspace(1e-6, 10, 16000)[:,np.newaxis]
    dl = _DL_INTERP(z)
    
    lp = LineProfiler()
    lp.add_function(LOS_lumdist_ansatz)
    lp.run('LOS_lumdist_ansatz(dl=dl, distnorm=np.linspace(100, 1000, N), distmu=np.linspace(100, 1000, N), distsigma=np.linspace(100, 1000, N))')
    lp.print_stats()

    

    # def LOS_lumdist_ansatz(dl, distnorm, distmu, distsigma):
    #     return dl**2 * np.exp(-0.5 * ((dl - distmu) / distsigma)**2) * distnorm / (distsigma * np.sqrt(2 * np.pi))

    # for i in range(10):
    #     print(i)
    #     t = time.time()
    #     LOS_lumdist_ansatz(dl=dl, distnorm=np.linspace(100, 1000, N), distmu=np.linspace(100, 1000, N), distsigma=np.linspace(100, 1000, N))
    #     print(time.time() - t)
    #     time.sleep(2)



    # N = 10000
    # z = np.linspace(1e-6, 10, 16000)
    # distnorm, distmu, distsigma = np.linspace(100, 1000, N), np.linspace(100, 1000, N), np.linspace(100, 1000, N)
    

    # t = time.time()
    # pdfbig = redshift_pdf_given_lumdist_pdf(z[:,np.newaxis], LOS_lumdist_ansatz, distnorm=distnorm, distmu=distmu, distsigma=distsigma)
    # print('big:', time.time() - t)

    # # t = time.time()
    # # for _ in range(100):
    # #     pdf = redshift_pdf_given_lumdist_pdf(z, LOS_lumdist_ansatz, distnorm=distnorm[i], distmu=distmu[i], distsigma=distsigma[i])
    # # print('original:', time.time() - t)

    # t = time.time()
    # for i in range(N):
    #     pdffast = redshift_pdf_given_lumdist_pdf(z, LOS_lumdist_ansatz, distnorm=distnorm[i], distmu=distmu[i], distsigma=distsigma[i])
    # print('new:', time.time() - t)

    # plt.figure()
    # plt.plot(z, abs(pdffast - pdf) / pdf)
    # plt.loglog()
    # plt.show()

    # dltrue = COSMO.luminosity_distance(z).value
    # htrue = COSMO.H(z)

    # t = time.time()
    # zfast = fast_z_at_value(COSMO.H, htrue)
    # print('original:', time.time() - t)

    # t = time.time()
    # zfast2 = fast_z_at_value2(COSMO.H, htrue)
    # print('new:', time.time() - t)

    # t = time.time()
    # dltrue = COSMO.luminosity_distance(z).value
    # print('dltrue:', time.time() - t)

    # t = time.time()
    # dlfast = _DL_INTERP(z)
    # print('dlfast:', time.time() - t)

    # dldiff = dltrue - dlfast

    # t = time.time()
    # htrue = COSMO.H(z).value
    # print('htrue:', time.time() - t)

    # t = time.time()
    # hfast = _H_INTERP(z)
    # print('hfast:', time.time() - t)

    # hdiff = htrue - hfast

    # plt.figure()
    # plt.plot(z, dldiff / dltrue)
    # plt.show()

    # plt.figure()
    # plt.plot(z, hdiff / htrue)
    # plt.show()

    # from utils import uniform_shell_sampler
    # import astropy.units as u
    
    # from scipy.integrate import romb


    # r, t, p = uniform_shell_sampler(COSMO.comoving_distance(1e-6).value, COSMO.comoving_distance(10).value, n_samps=int(1e7))
    # zsamp = fast_z_at_value(COSMO.comoving_distance, r * u.Mpc)
    # weights = 1 / (1 + zsamp)
    # zsamp_reweight = np.random.choice(zsamp, size=int(1e6), p=weights / np.sum(weights))
    # dlsamp_reweight = COSMO.luminosity_distance(zsamp_reweight).value

    # z = np.linspace(1e-6, 10, 1024+1)
    # lumdist = np.linspace(COSMO.luminosity_distance(1e-6).value, COSMO.luminosity_distance(10).value, 1024+1)
    
    # dlpdf = uniform_source_frame_dl(lumdist) / romb(uniform_source_frame_dl(lumdist), dx=np.diff(lumdist)[0])
    # zpdf = uniform_source_frame(z) / romb(uniform_source_frame(z), dx=np.diff(z)[0])

    # fig, ax = plt.subplots()
    # # ax.plot(lumdist, lumdist**2 / romb(lumdist**2, dx=np.diff(lumdist)[0]), color='orange')
    # ax.plot(lumdist, dlpdf, color='blue')
    # ax.hist(dlsamp_reweight, density=True, bins=50, histtype='step', color='steelblue')
    # # ax2 = ax.twiny()
    # # ax2.plot(z, zpdf, color='red')
    # # ax2.hist(zsamp_reweight, density=True, bins=50, histtype='step', color='coral')
    # plt.show()