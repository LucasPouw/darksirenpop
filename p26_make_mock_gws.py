import numpy as np
from utils import uniform_shell_sampler, spherical2cartesian, cartesian2spherical
from default_globals import *
import astropy.units as u
import sys, os
import h5py
from astropy.table import Table
import shutil
import glob
from redshift_utils import fast_z_at_value
from scipy.special import gammaincinv
from scipy.interpolate import interp1d


############## INPUT PARAMETERS ##############
MAKE_SKYMAPS = False 
ID = 'z'
V90_CDF = '/home/lucas/Documents/PhD/darksirenpop/v90_cdf.npy'
SKYMAP_DIR = f'./skymaps_{ID}'
POST_SAMPS_DIR = f'./posterior_samples_{ID}'
NCPU = os.cpu_count()
N_POSTERIOR_SAMPLES = int(1e5)
ZMIN = 1e-6
ZMAX = 1.5  # p_rate(z > ZMAX) = 0
BATCH = int(100)
##############################################


COMDIST_MIN = COSMO.comoving_distance(ZMIN).value
COMDIST_MAX = COSMO.comoving_distance(ZMAX).value  # Maximum comoving distance in Mpc


def check_directory(directory):
    if os.path.isdir(directory):
        if len(os.listdir(directory)) != 0:
            inp = None
            while inp not in ['y', 'yes', 'n', 'no']:
                inp = input(f'Found existing data in output directory: `{directory}`. DELETE existing data? (y/n)')

            if inp in ['y', 'yes']:
                print('Erasing existing data...')
                shutil.rmtree(directory)
                os.mkdir(directory)
            else:
                sys.exit('Not removing data. Please run again with a new output directory.')
    else:
        os.mkdir(directory)


def sample_v90():
    '''Inverse CDF sampling of observed distribution of log10 v90 in cMpc'''
    cdf_vals, x_grid = np.load(V90_CDF)
    inv_cdf = interp1d(cdf_vals, x_grid)
    u = np.random.rand(BATCH)
    return 10**inv_cdf(u)


def v90_to_sigma(v90):
    '''Going from the 90th percentile radius to sigma using the Maxwell-Boltzmann distribution.'''
    r90 = (v90 * 3 / (4 * np.pi))**(1/3)
    return r90 / np.sqrt(2 * gammaincinv(3/2, 0.9))


def make_real_gw_positions():
    true_rcom, true_theta, true_phi = uniform_shell_sampler(COMDIST_MIN, COMDIST_MAX, BATCH)
    true_x, true_y, true_z = spherical2cartesian(true_rcom, true_theta, true_phi)
    return true_x, true_y, true_z


def make_observed_gw_positions(true_x, true_y, true_z):
    v90 = sample_v90()
    sig = v90_to_sigma(v90)
    obs_x, obs_y, obs_z = np.random.normal(loc=true_x, scale=sig, size=BATCH), np.random.normal(loc=true_y, scale=sig, size=BATCH), np.random.normal(loc=true_z, scale=sig, size=BATCH)
    return obs_x, obs_y, obs_z, sig


def make_posterior_samples(trial_idx, fagn_idx, obs_x, obs_y, obs_z, sig):
    '''TODO: apply a prior, right?'''
    for i in range(BATCH):
        posterior_samples_x = np.random.normal(loc=obs_x[i], scale=sig[i], size=N_POSTERIOR_SAMPLES)
        posterior_samples_y = np.random.normal(loc=obs_y[i], scale=sig[i], size=N_POSTERIOR_SAMPLES)
        posterior_samples_z = np.random.normal(loc=obs_z[i], scale=sig[i], size=N_POSTERIOR_SAMPLES)

        rcom_samples, theta_samples, phi_samples = cartesian2spherical(posterior_samples_x, posterior_samples_y, posterior_samples_z)
        dec_samples = 0.5 * np.pi - theta_samples
        redshift_samples = fast_z_at_value(COSMO.comoving_distance, rcom_samples * u.Mpc)
        rlum_samples = COSMO.luminosity_distance(redshift_samples).value

        samples_table = Table([phi_samples, dec_samples, rlum_samples, rcom_samples, redshift_samples], 
                                    names=('ra', 'dec', 'luminosity_distance', 'comoving_distance', 'redshift'))
        filename = os.path.join(POST_SAMPS_DIR, f"gw_{trial_idx}_{fagn_idx}_{i:05d}.h5")

        with h5py.File(filename, "a") as f:
            mock_group = f.require_group("mock")  # Takes place of approximant in real GW data
            mock_group.create_dataset('posterior_samples', data=samples_table)
    return


def save_coordinates(trial_idx, fagn_idx, true_x, true_y, true_z):
    for _, infile in enumerate(glob.glob(f'{POST_SAMPS_DIR}/gw_{trial_idx}_{fagn_idx}_*.h5')):
        gw_idx = infile[-8:-3]
        r, theta, phi = cartesian2spherical(true_x[int(gw_idx)], true_y[int(gw_idx)], true_z[int(gw_idx)])  # Could've just used spherical coordinates directly, oh well.
        with open(f'true_r_theta_phi_{ID}.txt', 'a') as f:
            f.write(f'{gw_idx}, {r}, {theta}, {phi}\n')
    return


def make_skymaps(trial_idx, fagn_idx):
    print('Verifying OMP_NUM_THREADS:')
    os.system("echo $OMP_NUM_THREADS")

    for _, infile in enumerate(glob.glob(f'{POST_SAMPS_DIR}/gw_{trial_idx}_{fagn_idx}_*.h5')):
        print(f'Processing: {infile}')
        gw_idx = infile[-8:-3]
        outfile = f'skymap_{trial_idx}_{fagn_idx}_{gw_idx}.fits.gz'
        print(f'Output: {SKYMAP_DIR}/{outfile}')

        os.system(f"ligo-skymap-from-samples {infile} --fitsoutname {outfile} --outdir {SKYMAP_DIR} --jobs {NCPU}")

    os.system(f"rm -rf {SKYMAP_DIR}/skypost.obj")
    return


def main(trial_idx, fagn_idx):
    check_directory(POST_SAMPS_DIR)
    true_x, true_y, true_z = make_real_gw_positions()
    obs_x, obs_y, obs_z, sig = make_observed_gw_positions(true_x, true_y, true_z)
    make_posterior_samples(trial_idx, fagn_idx, obs_x, obs_y, obs_z, sig)
    save_coordinates(trial_idx, fagn_idx, true_x, true_y, true_z)

    if MAKE_SKYMAPS:
        check_directory(SKYMAP_DIR)
        os.environ["OMP_NUM_THREADS"] = "1"  # Important for proper threading when making skymaps
        make_skymaps(trial_idx, fagn_idx)
    return


if __name__ == '__main__':
    main(trial_idx=0, fagn_idx=0)
