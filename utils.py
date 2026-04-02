import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sys

from scipy import stats


def log10addexp10(a, b):
    return np.maximum(a, b) + np.log10(1 + 10**(-abs(a - b)))


def sample_from_distribution(pdf, x_grid, n_samples):
    x_grid = np.asarray(x_grid)

    if not np.all(np.diff(x_grid) > 0):
        raise ValueError("x_grid must be strictly increasing")

    y = np.clip(pdf(x_grid), 0, None)

    dx = np.diff(x_grid)
    dx = np.append(dx, dx[-1])

    cdf = np.cumsum(y * dx)

    if cdf[-1] == 0:
        raise ValueError("PDF integrates to zero over the provided grid.")

    cdf /= cdf[-1]

    u = np.random.rand(n_samples)
    return np.interp(u, cdf, x_grid)


def print_memory_usage(scope=None):
    """
    Print all variables in the given scope and their memory usage in MiB.
    """
    if scope is None:
        scope = globals()  # default to global variables

    total_bytes = 0
    print(f"{'Name':30} {'Type':20} {'MiB':>10}")
    print("-" * 65)

    for name, obj in scope.items():
        try:
            if isinstance(obj, np.ndarray):
                size = obj.nbytes
            else:
                size = sys.getsizeof(obj)
            total_bytes += size
            size_mib = size / (1024**2)
            print(f"{name:30} {type(obj).__name__:20} {size_mib:10.2f}")
        except:
            pass  # some objects may fail getsizeof

    print("-" * 65)
    print(f"{'TOTAL':30} {'':20} {total_bytes / (1024**2):10.2f}")


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)


def truncnorm_pdf_inplace(z, mu, sigma, zmin=0.0, out=None):
    """
    Memory-efficient computation of normalization constants.

    Parameters
    ----------
    mu : (N, 1)
    sigma : (N, 1)
    zmin : float
    zmax : float
    prior_fn : function(z) -> array
    n_norm : int
    out : optional buffer (N, n_norm)

    Returns
    -------
    z_norms : (N,)
    """

    # Ensure shapes
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    z = np.asarray(z)

    if z.ndim == 1:
        N = mu.shape[0]
        M = z.shape[0]

        if out is None:
            out = np.empty((N, M), dtype=np.float64)

        out[:] = z
        out -= mu
        out /= sigma

    elif z.ndim == 2:
        if out is None:
            out = np.empty_like(z)

        out[:] = z
        out -= mu      # (N, M) - (N, 1)
        out /= sigma

    else:
        raise ValueError("z must be 1D or 2D")

    # exp(-0.5 * t^2)
    np.square(out, out=out)
    out *= -0.5
    np.exp(out, out=out)

    # divide by sigma
    out /= sigma

    # normalization
    a = (zmin - mu) / sigma
    Z = 1.0 - stats.norm.cdf(a)

    out /= Z

    return out


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


def get_run(key):
    date = key.split('_')[0][2:]
    y, m, d = int(date[:2]), int(date[2:4]), int(date[4:])
    
    if y < 16:
        run = 'O1'
    elif y < 18:
        run = 'O2'
    elif y < 22:
        run = 'O3'
    else:
        run = 'O4'
    return run


if __name__ == '__main__':

    from redshift_utils import *
    from scipy.interpolate import interp1d
    from scipy.integrate import romb

    AGN_DIST_DIR = './darksirenpop/agn_distribution'
    AGN_ZPRIOR = '46.5_kulkarni'
    ZMAX = 3

    filename = f'{AGN_DIST_DIR}/agn_redshift_pdf_{AGN_ZPRIOR}.npy'
    print(f'Loading AGN redshift distribution from file: {filename}')
    z, n = np.load(filename)
    AGN_DIST = interp1d(z, n, bounds_error=False, fill_value=0)

    Z_DIST_AGN = lambda z: time_dilation_correction(z) * z_cut(z, zcut=ZMAX) * AGN_DIST(z) / romb(time_dilation_correction(z) * AGN_DIST(z), dx=np.diff(z)[0])
    Z_DIST_ALT = lambda z: time_dilation_correction(z) * z_cut(z, zcut=ZMAX) * merger_rate_madau_dickinson(z) * uniform_comoving_prior(z) / romb(time_dilation_correction(z) * z_cut(z, zcut=ZMAX) * merger_rate_madau_dickinson(z) * uniform_comoving_prior(z), dx=np.diff(z)[0])

    zz = np.linspace(0, ZMAX, 1024+1)
    samps = sample_from_distribution(Z_DIST_ALT, zz, n_samples=int(1e7))
    plt.figure()
    plt.hist(samps, density=True, bins=50)
    plt.plot(zz, Z_DIST_ALT(zz))
    plt.show()

