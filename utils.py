#%%
from astropy.cosmology import z_at_value
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import healpy as hp
import os
from scipy.spatial import cKDTree


def make_nice_plots():
    SMALL_SIZE = 10 * 2 
    MEDIUM_SIZE = 12 * 2
    BIGGER_SIZE = 14 * 2

    plt.rc('text', usetex=True)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE, direction='out')
    plt.rc('ytick', labelsize=SMALL_SIZE, direction='out')
    plt.rc('legend', fontsize=SMALL_SIZE)
    mpl.rcParams['axes.titlesize'] = BIGGER_SIZE
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['font.family'] = 'STIXgeneral'

    mpl.rcParams['figure.dpi'] = 100

    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['ytick.minor.visible'] = True
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True

    mpl.rcParams['xtick.major.size'] = 10
    mpl.rcParams['ytick.major.size'] = 10
    mpl.rcParams['xtick.minor.size'] = 4
    mpl.rcParams['ytick.minor.size'] = 4

    mpl.rcParams['xtick.major.width'] = 1.25
    mpl.rcParams['ytick.major.width'] = 1.25
    mpl.rcParams['xtick.minor.width'] = 1
    mpl.rcParams['ytick.minor.width'] = 1


def sample_spherical_angles(n_samps=1):
    theta = np.arccos(np.random.uniform(size=n_samps, low=-1, high=1))  # Cosine is uniformly distributed between -1 and 1 -> cos between 0 and pi
    phi = 2 * np.pi * np.random.uniform(size=n_samps)  # Draws phi from 0 to 2pi
    return theta, phi


def uniform_shell_sampler(rmin, rmax, n_samps):
    r = ( np.random.uniform(size=n_samps, low=rmin**3, high=rmax**3) )**(1/3)
    theta, phi = sample_spherical_angles(n_samps)
    return r, theta, phi


def spherical2cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cartesian2spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1, 1))  # From 0 to pi
    phi = np.arctan2(y, x)  # From -pi to pi
    phi = np.where(phi >= 0, phi, phi + 2*np.pi)  # From 0 to 2pi
    return r, theta, phi


def fast_z_at_value(function, values, num=100):
    zmin = z_at_value(function, values.min())
    zmax = z_at_value(function, values.max())
    zgrid = np.geomspace(zmin, zmax, num)
    valgrid = function(zgrid)
    zvals = np.interp(values.value, valgrid.value, zgrid)
    return zvals.value


# def ckd_tree_kde_evaluation(data, evaluation_points):
#     assert len(data) >= 1e4, 'Too few data points to justify cKDTree. Use gaussian_kde instead.'
#     bandwidth = len(data)**(-1/5) * np.std(data)  # Scott's method (analog?)
#     kdtree = cKDTree(data[:, None])
#     kde_values = np.zeros_like(evaluation_points)
#     for i, point in enumerate(evaluation_points):
#         indices = kdtree.query_ball_point([point], bandwidth)
#         kde_values[i] = len(indices) / (len(data) * bandwidth)
#     return kde_values * 0.5


def histogram_pdf(data, points):
    assert len(data) >= 1e4, 'Too few data points to justify histogram PDF. Use gaussian_kde instead.'
    bandwidth = len(data)**(-1/5) * np.std(data)
    xmin = min(np.min(data), np.min(points))
    xmax = max(np.max(data), np.max(points))
    nbins = int((xmax - xmin) / bandwidth)
    edges = np.linspace(xmin, xmax, nbins+1)
    binned_data, edges = np.histogram(data, bins=edges, density=True)
    idx = np.digitize(points, edges[:-1]) - 1
    return binned_data[idx]


def check_equal(a, b):
    if len(a) != len(b):
        return False
    return sorted(a) == sorted(b)


def ra_dec_from_ipix(nside, ipix, nest=True):
    """RA and dec from HEALPix index"""
    (theta, phi) = hp.pix2ang(nside, ipix, nest=nest)
    return (phi, np.pi/2.-theta)


def ipix_from_ra_dec(nside, ra, dec, nest=True):
    """HEALPix index from RA and dec"""
    (theta, phi) = (np.pi/2.-dec, ra)
    return hp.ang2pix(nside, theta, phi, nest=nest)


def make_pixdict(low_nside, high_nside, nest=True):
    high_pix = np.arange(hp.nside2npix(high_nside))
    high_radec = ra_dec_from_ipix(nside=high_nside, ipix=high_pix, nest=nest)
    low_pix = ipix_from_ra_dec(low_nside, *high_radec, nest=nest)
    return {key: value for key, value in zip(high_pix, low_pix)}


def get_cachedir(cachedir=None):
    if cachedir is None:
        if 'DSP_CACHE' in os.environ.keys():
            cachedir = os.environ['DSP_CACHE']
        else:
            cachedir = os.path.join(os.environ['HOME'], '.cache/darksirenpop')
    os.makedirs(cachedir, exist_ok=True)
    return cachedir


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    import scipy.stats as stats
    from scipy.optimize import root_scalar
    from scipy.special import erf
    import time
    from tqdm import tqdm
    import random

    import vose

    s = int(1e5)
    arr = np.random.normal(0, 1, size=s)
    weights = abs(arr) / np.sum(abs(arr))
    start = time.time()
    for i in tqdm(range(1000)):
        sampler = vose.Sampler(weights)
        idx = sampler.sample(k=s)
        v_arr = arr[idx]
    print('vose', time.time() - start)

    start = time.time()
    for i in tqdm(range(1000)):
        np_arr = np.random.choice(arr, size=s, p=weights)
    print('numpy', time.time() - start)

    plt.figure()
    plt.hist(v_arr, density=True, bins=100, histtype='step')
    plt.hist(np_arr, density=True, bins=100, histtype='step')
    plt.show()



    #%%


    n = int(1e4)
    p = int(1e3)
    data = np.random.normal(size=n)
    points = np.linspace(-5, 5, p)

    start = time.time()
    kde = stats.gaussian_kde(data)(points)
    print('kde', time.time() - start)

    # start = time.time()
    # ckde = ckd_tree_kde_evaluation(data, points)  # Faster and adequate for n=1e4 and p=1e4, never worth it below n=1e3
    # print('ckde', time.time() - start)

    start = time.time()
    hkde = histogram_pdf(data, points)
    print('hkde', time.time() - start)

    plt.figure()
    plt.plot(points, kde, color='blue')
    # plt.plot(points, ckde, color='red')
    plt.plot(points, hkde, color='black')
    plt.show()

#%%

    # cosmo = FlatLambdaCDM(H0=67.9, Om0=0.3065)

    # n_trials = 100
    # time_arr = np.zeros(n_trials)
    # time_arr2 = np.zeros(n_trials)
    # for i in tqdm(range(n_trials)):
    #     rlum_obs = (np.random.uniform(low=0, high=100, size=1))**(1/3)
    #     lumdist_arr = rlum_obs / (1 + 0.1 * np.random.normal(size=4 * int(1e5)))

    #     start = time.time()
    #     vals1 = fast_z_at_value(cosmo.luminosity_distance, lumdist_arr * u.Mpc, num=100000)
    #     time_arr[i] = time.time() - start

    #     start = time.time()
    #     vals2 = fast_z_at_value(cosmo.luminosity_distance, lumdist_arr * u.Mpc, num=100)
    #     time_arr2[i] = time.time() - start
    # print(np.mean(time_arr), np.mean(time_arr2))
    # print(vals1 - vals2)

    # cl = 0.999
    # sig = 500
    # sig2 = 1
    # xyz_true = np.array([[5000, 3000, 1000], [10, 10, 10]])
    # std = np.array([[sig, sig, sig], [sig2, sig2, sig2]])

    # def cl_volume_from_sigma(cl, sigmas):
    #     '''Calculates length of CL% vector when x, y, z are normally distributed with the same sigma'''
    #     radii = np.zeros(len(sigmas))
    #     for i, sigma in enumerate(sigmas):
    #         vector_length_cdf = lambda x: erf(x / (sigma * np.sqrt(2))) - np.sqrt(2) / (sigma * np.sqrt(np.pi)) * x * np.exp(-x**2 / (2 * sigma**2)) - cl
    #         radii[i] = root_scalar(vector_length_cdf, bracket=[0, 100 * sigma], method='bisect').root
    #     return 4 * np.pi * radii**3 / 3

    # xyz_meas = np.random.normal(loc=xyz_true, scale=std, size=(2, 3))
    # print(xyz_meas)
    # volumes = cl_volume_from_sigma(cl, np.array([sig, sig2]))
    # print(volumes)

    # r, theta, phi = cartesian2spherical(x, y, z)

    # z = fast_z_at_value(cosmo.comoving_distance, r * u.Mpc)

    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    # ax[0,0].hist(r, density=True, bins=30)
    # ax[0,1].hist(np.cos(theta), density=True, bins=30)
    # ax[1,0].hist(phi, density=True, bins=30)
    # ax[1,1].hist(z, density=True, bins=30)
    # ax[0,0].set_xlabel('r com [Mpc]')
    # ax[0,1].set_xlabel('theta [rad]')
    # ax[1,0].set_xlabel('phi [rad]')
    # ax[1,1].set_xlabel('z')
    # plt.tight_layout()
    # plt.savefig('test.pdf')
    # plt.show()