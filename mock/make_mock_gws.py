import numpy as np
import sys, os
from pathlib import Path
from dataclasses import dataclass, field
import h5py
from astropy.table import Table
import shutil
import glob
import time
import argparse

from darksirenpop.utilities.utils import *
from darksirenpop.utilities.redshift_utils import *
from darksirenpop.utilities.default_globals import *

import astropy.units as u

from scipy.integrate import romb
from scipy.special import gammaincinv
from scipy.interpolate import interp1d


@dataclass
class Config:
    # Parser inputs
    RUN_ID: int
    AGNDIST: str
    NGW: int
    OVERWRITE: bool = False
    ZMIN: float = 1e-6
    ZMAX: float = 10
    ZCUT: float = np.inf
    NCPU: int = 1
    NPOSTSAMPS: int = int(5e3)
    FAGN: float | None = None
    MAKE_SKYMAPS: bool = True
    V90_CDF: str = V90_CDF_PATH
    AGN_DIST_DIR: str = AGN_DIST_DIR
    VERBOSE: bool = False

    # Derived quantities
    AGN_DIST: object = field(init=False)
    Z_GRID: np.ndarray = field(init=False)
    ROOT: str = field(init=False)
    OUTPUT_DIR: str = field(init=False)
    ID: str = field(init=False)
    SKYMAP_DIR: str = field(init=False)
    POST_SAMPS_DIR: str = field(init=False)
    TRUE_COORDS_DIR: str = field(init=False)
    SAFE_BASE_DIRECTORY: Path = field(init=False)

    def __post_init__(self):
        if self.FAGN is None:
            self.FAGN = np.random.uniform()

        AGN_ZPRIOR = f'{self.AGNDIST}_kulkarni'
        FILENAME = f'{self.AGN_DIST_DIR}/agn_redshift_pdf_{AGN_ZPRIOR}.npy'

        if self.VERBOSE:
            print(f'Loading AGN redshift distribution from file: {FILENAME}')
        z, n = np.load(FILENAME)
        self.AGN_DIST = interp1d(z, n, bounds_error=False, fill_value=0)
        self.Z_GRID = np.linspace(0, self.ZMAX, 1025)

        self.ROOT = (
            f'mock_gws_agndist_{self.AGNDIST}_ngw_{self.NGW}'
            f'_zmax_{self.ZMAX}_zcut_{self.ZCUT}_LVKvols'
        )
        self.OUTPUT_DIR = f'output_run_{self.RUN_ID}_fagn_{self.FAGN}'
        self.ID = f'{self.ROOT}/{self.OUTPUT_DIR}'
        self.SKYMAP_DIR = f'{self.ID}/skymaps'
        self.POST_SAMPS_DIR = f'{self.ID}/posterior_samples'
        self.TRUE_COORDS_DIR = f'{self.ID}/true_gw_coords'
        self.SAFE_BASE_DIRECTORY = Path(self.ID).resolve()


def parse_config(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--agndist', type=str, required=True, choices=['44.5', '45.5', '46.5'])
    parser.add_argument('--ngw', type=int, required=True)
    parser.add_argument('--zmin', type=float, default=1e-6)
    parser.add_argument('--zmax', type=float, default=10)
    parser.add_argument('--zcut', type=float, default=np.inf)
    parser.add_argument('--ncpu', type=int, default=1)
    parser.add_argument('--npostsamps', type=int, default=int(5e3))
    parser.add_argument('--fagn', type=float, default=None)
    parser.add_argument('--make-skymaps', action='store_true')
    parser.add_argument('--v90-cdf', type=str, default=V90_CDF_PATH)
    parser.add_argument('--agn-dist-dir', type=str, default=AGN_DIST_DIR)
    args = parser.parse_args(argv)

    return Config(
        RUN_ID=args.run_id, OVERWRITE=args.overwrite, AGNDIST=args.agndist, NGW=args.ngw,
        ZMIN=args.zmin, ZMAX=args.zmax, ZCUT=args.zcut, NCPU=args.ncpu,
        NPOSTSAMPS=args.npostsamps, FAGN=args.fagn, MAKE_SKYMAPS=args.make_skymaps,
        V90_CDF=args.v90_cdf, AGN_DIST_DIR=args.agn_dist_dir, VERBOSE=args.verbose
    )


def check_directory(directory, config):
    if os.path.isdir(directory):
        if len(os.listdir(directory)) != 0:
            if config.OVERWRITE:
                directory = Path(directory).resolve()
                if config.SAFE_BASE_DIRECTORY not in directory.parents and directory != config.SAFE_BASE_DIRECTORY:
                    sys.exit(f'Refusing to delete {directory}: not inside {config.SAFE_BASE_DIRECTORY}')
                print(f'Emptying directory: {directory}')
                shutil.rmtree(directory)
                os.makedirs(directory, exist_ok=True)
            else:
                sys.exit(f'{directory} exists and is not empty. Add --overwrite or delete by hand.')
    else:
        os.makedirs(directory, exist_ok=True)


def sample_v90(n, config):
    """Inverse CDF sampling of observed distribution of log10 v90 in cMpc."""
    cdf_vals, x_grid = np.load(config.V90_CDF)
    inv_cdf = interp1d(cdf_vals, x_grid)
    u = np.random.rand(n)
    return 10**inv_cdf(u)


def v90_to_sigma(v90):
    """Going from the 90th percentile radius to sigma using the Maxwell-Boltzmann distribution."""
    r90 = (v90 * 3 / (4 * np.pi))**(1 / 3)
    return r90 / np.sqrt(2 * gammaincinv(3 / 2, 0.9))


def z_dist_agn(z, config):
    numerator = time_dilation_correction(z) * z_cut(z, zcut=config.ZMAX) * config.AGN_DIST(z)
    denominator = romb(
        time_dilation_correction(config.Z_GRID)
        * z_cut(config.Z_GRID, zcut=config.ZMAX)
        * config.AGN_DIST(config.Z_GRID),
        dx=np.diff(config.Z_GRID)[0]
    )
    return numerator / denominator


def z_dist_alt(z, config):
    numerator = (
        time_dilation_correction(z)
        * z_cut(z, zcut=config.ZMAX)
        * merger_rate_madau_dickinson(z)
        * uniform_comoving_prior(z)
    )
    denominator = romb(
        time_dilation_correction(config.Z_GRID)
        * z_cut(config.Z_GRID, zcut=config.ZMAX)
        * merger_rate_madau_dickinson(config.Z_GRID)
        * uniform_comoving_prior(config.Z_GRID),
        dx=np.diff(config.Z_GRID)[0]
    )
    return numerator / denominator


def make_real_gw_positions(n, kind, config):
    if kind == 'agn':
        true_z = sample_from_distribution(lambda z: z_dist_agn(z, config), config.Z_GRID, n_samples=n)
    elif kind == 'alt':
        true_z = sample_from_distribution(lambda z: z_dist_alt(z, config), config.Z_GRID, n_samples=n)
    else:
        raise ValueError(f'Unsupported kind: {kind}')

    true_rcom = COSMO.comoving_distance(true_z).value
    true_theta, true_phi = sample_spherical_angles(n_samps=n)
    true_x, true_y, true_z = spherical2cartesian(true_rcom, true_theta, true_phi)
    return true_x, true_y, true_z, true_rcom, true_theta, true_phi


def make_observed_gw_positions(true_x, true_y, true_z, true_rcom, true_theta, true_phi, config):
    n = len(true_x)
    v90 = sample_v90(n, config)
    sig = v90_to_sigma(v90)

    obs_x = np.random.normal(loc=true_x, scale=sig, size=n)
    obs_y = np.random.normal(loc=true_y, scale=sig, size=n)
    obs_z = np.random.normal(loc=true_z, scale=sig, size=n)
    obs_rcom, _, _ = cartesian2spherical(obs_x, obs_y, obs_z)
    obs_redshift = fast_z_at_value(COSMO.comoving_distance, obs_rcom * u.Mpc)

    sel = obs_redshift < config.ZCUT   # Make hard redshift cut for observation
    if config.VERBOSE:
        print(f'Observed {np.sum(sel)} GWs. Efficiency: {np.sum(sel) / n}')
    return obs_x, obs_y, obs_z, sig, v90, sel


def make_posterior_samples(trial_idx, fagn_idx, obs_x, obs_y, obs_z, sig, kind, config):
    for i in range(len(obs_x)):
        posterior_samples_x = np.random.normal(loc=obs_x[i], scale=sig[i], size=config.NPOSTSAMPS)
        posterior_samples_y = np.random.normal(loc=obs_y[i], scale=sig[i], size=config.NPOSTSAMPS)
        posterior_samples_z = np.random.normal(loc=obs_z[i], scale=sig[i], size=config.NPOSTSAMPS)

        rcom_samples, theta_samples, phi_samples = cartesian2spherical(posterior_samples_x, posterior_samples_y, posterior_samples_z)
        dec_samples = 0.5 * np.pi - theta_samples
        redshift_samples = fast_z_at_value(COSMO.comoving_distance, rcom_samples * u.Mpc)
        rlum_samples = COSMO.luminosity_distance(redshift_samples).value

        samples_table = Table([phi_samples, dec_samples, rlum_samples, rcom_samples, redshift_samples],
                                names=('ra', 'dec', 'luminosity_distance', 'comoving_distance', 'redshift'))

        post_dir = f'{config.POST_SAMPS_DIR}/{kind}'
        os.makedirs(post_dir, exist_ok=True)
        filename = os.path.join(post_dir, f'gw_{trial_idx}_{fagn_idx}_{i:05d}.h5')

        with h5py.File(filename, 'a') as f:
            mock_group = f.require_group('mock')  # Takes place of approximant in real GW data
            mock_group.create_dataset('posterior_samples', data=samples_table)


def save_coordinates(trial_idx, fagn_idx, true_x, true_y, true_z, v90, kind, config):
    post_dir = f'{config.POST_SAMPS_DIR}/{kind}'
    coord_dir = f'{config.TRUE_COORDS_DIR}/{kind}'
    os.makedirs(post_dir, exist_ok=True)
    os.makedirs(coord_dir, exist_ok=True)

    for infile in glob.glob(f'{post_dir}/gw_{trial_idx}_{fagn_idx}_*.h5'):
        gw_idx = infile[-8:-3]
        r, theta, phi = cartesian2spherical(true_x[int(gw_idx)], true_y[int(gw_idx)], true_z[int(gw_idx)])  # Could've just used spherical coordinates directly, oh well.

        with open(f'{coord_dir}/true_r_theta_phi.txt', 'a') as f:
            f.write(f'{gw_idx}, {r}, {theta}, {phi}, {v90[int(gw_idx)]}\n')


def save_coordinates_nondetections(true_x, true_y, true_z, v90, kind, config):
    nondetection_dir = f'{config.TRUE_COORDS_DIR}_nondetections'
    coord_dir = f'{nondetection_dir}/{kind}'
    os.makedirs(coord_dir, exist_ok=True)

    for x, y, z, locvol in zip(true_x, true_y, true_z, v90):
        r, theta, phi = cartesian2spherical(x, y, z)
        with open(f'{coord_dir}/true_r_theta_phi.txt', 'a') as f:
            f.write(f'{r}, {theta}, {phi}, {locvol}\n')


def make_skymaps(trial_idx, fagn_idx, kind, config):
    post_dir = f'{config.POST_SAMPS_DIR}/{kind}'
    sky_dir = f'{config.SKYMAP_DIR}/{kind}'
    os.makedirs(post_dir, exist_ok=True)
    os.makedirs(sky_dir, exist_ok=True)

    for infile in glob.glob(f'{post_dir}/gw_{trial_idx}_{fagn_idx}_*.h5'):
        print(f'Processing: {infile}')
        t = time.time()
        gw_idx = infile[-8:-3]
        outfile = f'skymap_{trial_idx}_{fagn_idx}_{gw_idx}.fits.gz'
        print(f'Output: {sky_dir}/{outfile}')

        os.system(f"ligo-skymap-from-samples {infile} --fitsoutname {outfile} --outdir {sky_dir} --jobs {config.NCPU}")
        print(f'That took {time.time() - t} s.\n')

    os.system(f'rm -rf {sky_dir}/skypost.obj')


def main(config, trial_idx=0, fagn_idx=0):
    check_directory(config.POST_SAMPS_DIR, config)
    check_directory(config.TRUE_COORDS_DIR, config)

    from_agn_mask = np.random.rand(config.NGW) < config.FAGN
    n_from_agn = np.sum(from_agn_mask)
    n_from_alt = np.sum(~from_agn_mask)

    true_x_agn, true_y_agn, true_z_agn, true_rcom_agn, true_theta_agn, true_phi_agn = make_real_gw_positions(n=n_from_agn, kind='agn', config=config)
    true_x_alt, true_y_alt, true_z_alt, true_rcom_alt, true_theta_alt, true_phi_alt = make_real_gw_positions(n=n_from_alt, kind='alt', config=config)

    obs_x_agn, obs_y_agn, obs_z_agn, sig_agn, v90_agn, sel_agn = make_observed_gw_positions(true_x_agn, true_y_agn, true_z_agn, true_rcom_agn, true_theta_agn, true_phi_agn, config)
    obs_x_alt, obs_y_alt, obs_z_alt, sig_alt, v90_alt, sel_alt = make_observed_gw_positions(true_x_alt, true_y_alt, true_z_alt, true_rcom_alt, true_theta_alt, true_phi_alt, config)

    make_posterior_samples(trial_idx, fagn_idx, obs_x_agn[sel_agn], obs_y_agn[sel_agn], obs_z_agn[sel_agn], sig_agn[sel_agn], 'agn', config)
    save_coordinates(trial_idx, fagn_idx, true_x_agn[sel_agn], true_y_agn[sel_agn],true_z_agn[sel_agn], v90_agn[sel_agn], 'agn', config)
    save_coordinates_nondetections(true_x_agn[~sel_agn], true_y_agn[~sel_agn], true_z_agn[~sel_agn], v90_agn[~sel_agn], 'agn', config)

    make_posterior_samples(trial_idx, fagn_idx, obs_x_alt[sel_alt], obs_y_alt[sel_alt],obs_z_alt[sel_alt], sig_alt[sel_alt], 'alt', config)
    save_coordinates(trial_idx, fagn_idx, true_x_alt[sel_alt], true_y_alt[sel_alt],true_z_alt[sel_alt], v90_alt[sel_alt], 'alt', config)
    save_coordinates_nondetections(true_x_alt[~sel_alt], true_y_alt[~sel_alt], true_z_alt[~sel_alt], v90_alt[~sel_alt], 'alt', config)

    if config.MAKE_SKYMAPS:
        check_directory(config.SKYMAP_DIR, config)
        os.environ['OMP_NUM_THREADS'] = '1'  # Important for proper threading when making skymaps
        make_skymaps(trial_idx, fagn_idx, 'agn', config)
        make_skymaps(trial_idx, fagn_idx, 'alt', config)


if __name__ == '__main__':
    config = parse_config()
    main(config, trial_idx=0, fagn_idx=0)
