import sys
import numpy as np
from scipy.stats import norm
from scipy.integrate import simpson
import matplotlib as mpl
import matplotlib.pyplot as plt


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

    # Ensure shape
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    z = np.asarray(z)

    shape = np.broadcast(z, mu, sigma).shape
    out = np.empty(shape, dtype=np.float32)

    # Calculate exponential
    out[:] = z
    out -= mu
    out /= sigma
    np.square(out, out=out)
    out *= -0.5
    np.exp(out, out=out)
    out /= sigma

    # Normalization
    a = (zmin - mu) / sigma
    Z = 1.0 - norm.cdf(a)
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


def sigfig_str(x, sig=2):
    '''
    Format number with the requested amount of significant figures.
    '''
    if x == 0:
        decimals = max(sig - 1, 0)
        return f"0.{'0' * decimals}" if decimals > 0 else "0"
    from decimal import Decimal
    d = Decimal(str(x))
    exponent = d.adjusted()  # position of most significant digit
    decimals = sig - 1 - exponent
    if decimals < 0:
        decimals = 0
    return f"{x:.{decimals}f}"


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


def get_pdfs(posteriors, integrate_axis):
    '''
    Go from unnormalized log-posteriors ln(p(f_agn | D)) to normalized posteriors p(f_agn | D).
    Assumes variable ``posteriors`` is a 2D array where axis 0 is the f_agn axis, 
    and axis 1 are posteriors from different runs.
    '''
    posteriors -= np.max(posteriors, axis=0)
    pdf = np.exp(posteriors)
    norms = simpson(y=pdf, x=integrate_axis, axis=0)  # Don't remember why I chose simpson
    pdfs = pdf / norms
    return pdfs


def get_cdfs(posteriors):
    posteriors -= np.max(posteriors, axis=0)
    pdf = np.exp(posteriors)
    cdfs = np.cumsum(pdf, axis=0)
    cdfs /= np.max(cdfs, axis=0)
    return cdfs


def hdi(samples, cred_mass=0.9):
	"""
	Highest Density Interval from posterior samples.

	Parameters
	----------
	samples : array-like
		Posterior samples.
	cred_mass : float
		Desired probability mass (e.g. 0.9 for 90% HDI).

	Returns
	-------
	hdi_low, hdi_high
	"""
	samples = np.asarray(samples)
	samples = np.sort(samples)

	n = len(samples)
	interval_idx = int(np.floor(cred_mass * n))

	if interval_idx < 1:
		raise ValueError("Not enough samples")

	widths = samples[interval_idx:] - samples[:n - interval_idx]
	min_idx = np.argmin(widths)

	return samples[min_idx], samples[min_idx + interval_idx]


if __name__ == '__main__':

    from darksirenpop.utilities.redshift_utils import *
    from scipy.interpolate import interp1d
    from scipy.integrate import romb

    AGN_DIST_DIR = '/home/lucas/Documents/PhD/generated_data/em'
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

