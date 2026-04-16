import numpy as np
import matplotlib.pyplot as plt
import sys, os
from pathlib import Path
import h5py
from astropy.table import Table
import shutil
import glob

from utils import uniform_shell_sampler, spherical2cartesian, cartesian2spherical, sample_from_distribution, sample_spherical_angles
from redshift_utils import fast_z_at_value, merger_rate_madau_dickinson, time_dilation_correction, uniform_comoving_prior, z_cut
from default_globals import *

import astropy.units as u

from scipy.integrate import romb
from scipy.special import gammaincinv
from scipy.interpolate import interp1d

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=int, required=True)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--nonuniform", action="store_true")
parser.add_argument("--batch", type=int, required=True)
args = parser.parse_args()

OUTPUT_DIR = f"output_run_{args.run_id}"
OVERWRITE = args.overwrite
NONUNIFORM = args.nonuniform
BATCH = args.batch  #int(3000)

############## INPUT PARAMETERS ##############
MAKE_SKYMAPS = True 
FIXED_VOL = False  # float or False
V90_CDF = '/home/lucas/Documents/PhD/darksirenpop/v90_cdf_LVK.npy'

NCPU = os.cpu_count() - 2
N_POSTERIOR_SAMPLES = int(5e3)
ZMIN = 1e-6
ZMAX = 10  # p_rate(z > ZMAX) = 0
ZCUT = 1

ID = f'mock_gws_nonuniform_{NONUNIFORM}_batch_{BATCH}_zmax_{ZMAX}_zcut_{ZCUT}_LVKvols/{OUTPUT_DIR}'
SKYMAP_DIR = f'{ID}/skymaps'
POST_SAMPS_DIR = f'{ID}/posterior_samples'
TRUE_COORDS_DIR = f'{ID}/true_gw_coords'

SAFE_BASE_DIRECTORY = Path(ID).resolve()

########## THESE ARGUMENTS ARE USED WHEN NONUNIFORM == TRUE ##########
F_AGN_TRUE = 0.5
AGN_DIST_DIR = './darksirenpop/agn_distribution'
AGN_ZPRIOR = '46.5_kulkarni'

filename = f'{AGN_DIST_DIR}/agn_redshift_pdf_{AGN_ZPRIOR}.npy'
print(f'Loading AGN redshift distribution from file: {filename}')
z, n = np.load(filename)
AGN_DIST = interp1d(z, n, bounds_error=False, fill_value=0)

Z_GRID = np.linspace(0, ZMAX, 1024+1)
Z_DIST_AGN = lambda z: time_dilation_correction(z) * z_cut(z, zcut=ZMAX) * AGN_DIST(z) / romb(time_dilation_correction(z) * z_cut(z, zcut=ZMAX) * AGN_DIST(z), dx=np.diff(z)[0])
Z_DIST_ALT = lambda z: time_dilation_correction(z) * z_cut(z, zcut=ZMAX) * merger_rate_madau_dickinson(z) * uniform_comoving_prior(z) / romb(time_dilation_correction(z) * z_cut(z, zcut=ZMAX) * merger_rate_madau_dickinson(z) * uniform_comoving_prior(z), dx=np.diff(z)[0])

# zz = np.linspace(ZMIN, ZMAX, 1024+1)
# plt.figure()
# plt.plot(zz, Z_DIST_ALT(zz))
# plt.plot(zz, Z_DIST_AGN(zz))
# plt.show()
# sys.exit(1)
#######################################################################


COMDIST_MIN = COSMO.comoving_distance(ZMIN).value
COMDIST_MAX = COSMO.comoving_distance(ZMAX).value  # Maximum comoving distance in Mpc


# def check_directory(directory):
#     if os.path.isdir(directory):
#         if len(os.listdir(directory)) != 0:
#             inp = None
#             while inp not in ['y', 'yes', 'n', 'no']:
#                 inp = input(f'Found existing data in output directory: `{directory}`. DELETE existing data? (y/n)')

#             if inp in ['y', 'yes']:
#                 print('Erasing existing data...')
#                 shutil.rmtree(directory)
#                 os.mkdir(directory)
#             else:
#                 sys.exit('Not removing data. Please run again with a new output directory.')
#     else:
#         os.mkdir(directory)


def check_directory(directory, overwrite=OVERWRITE):
    if os.path.isdir(directory):
        if len(os.listdir(directory)) != 0:
            
            if overwrite:
                
                directory = Path(directory).resolve()
                if SAFE_BASE_DIRECTORY not in directory.parents and directory != SAFE_BASE_DIRECTORY:
                    sys.exit(f"Refusing to delete {directory}: not inside {SAFE_BASE_DIRECTORY}")
                
                else:
                    print(f'Emptying directory: {directory}')
                    shutil.rmtree(directory)
                    os.makedirs(directory, exist_ok=True)

            else:
                sys.exit(f"{directory} exists and is not empty. Add --overwrite or delete by hand.")
    else:
        os.makedirs(directory, exist_ok=True)


def sample_v90(n=BATCH):
    '''Inverse CDF sampling of observed distribution of log10 v90 in cMpc'''
    cdf_vals, x_grid = np.load(V90_CDF)
    inv_cdf = interp1d(cdf_vals, x_grid)
    u = np.random.rand(n)
    return 10**inv_cdf(u)


def v90_to_sigma(v90):
    '''Going from the 90th percentile radius to sigma using the Maxwell-Boltzmann distribution.'''
    r90 = (v90 * 3 / (4 * np.pi))**(1/3)
    return r90 / np.sqrt(2 * gammaincinv(3/2, 0.9))


def make_real_gw_positions(n=BATCH, kind=None):

    if kind == None:
        true_rcom, true_theta, true_phi = uniform_shell_sampler(COMDIST_MIN, COMDIST_MAX, n)
    
    elif kind == 'agn':
        true_z = sample_from_distribution(Z_DIST_AGN, Z_GRID, n_samples=n)
        true_rcom = COSMO.comoving_distance(true_z).value
        true_theta, true_phi = sample_spherical_angles(n_samps=n)
    
    elif kind == 'alt':
        true_z = sample_from_distribution(Z_DIST_ALT, Z_GRID, n_samples=n)
        true_rcom = COSMO.comoving_distance(true_z).value
        true_theta, true_phi = sample_spherical_angles(n_samps=n)

    else:
        sys.exit(f'Function does not support argument value kind == {kind}. Use "alt", "agn" or None.')        

    true_x, true_y, true_z = spherical2cartesian(true_rcom, true_theta, true_phi)
    return true_x, true_y, true_z, true_rcom, true_theta, true_phi


def make_observed_gw_positions(true_x, true_y, true_z, true_rcom, true_theta, true_phi):
    n = len(true_x)

    if FIXED_VOL:
        v90 = np.tile(FIXED_VOL, n)
    else:
        v90 = sample_v90(n)
    sig = v90_to_sigma(v90)
    
    obs_x, obs_y, obs_z = np.random.normal(loc=true_x, scale=sig, size=n), np.random.normal(loc=true_y, scale=sig, size=n), np.random.normal(loc=true_z, scale=sig, size=n)
    
    # Make hard redshift cut for observation
    obs_rcom, _, _ = cartesian2spherical(obs_x, obs_y, obs_z)
    obs_redshift = fast_z_at_value(COSMO.comoving_distance, obs_rcom * u.Mpc)
    sel = obs_redshift < ZCUT
    print(f'Observed {np.sum(sel)} GWs. Efficiency: {np.sum(sel) / n}')

    # true_redshift = fast_z_at_value(COSMO.comoving_distance, true_rcom * u.Mpc)
    # print(np.min(np.log10(v90)))
    # plt.figure()
    # plt.hist(np.log10(v90))
    # # print(np.sum(obs_redshift < ZCUT))
    # # plt.hist(true_redshift, bins=30, histtype='step', linewidth=2)
    # # plt.hist(true_redshift[obs_redshift < ZCUT], bins=30, histtype='step', linewidth=2)
    # plt.show()
    # sys.exit(1)
    return obs_x[sel], obs_y[sel], obs_z[sel], sig[sel], v90[sel]


def make_posterior_samples(trial_idx, fagn_idx, obs_x, obs_y, obs_z, sig, kind=None):
    '''TODO: apply a prior, right?'''
    for i in range(len(obs_x)):
        posterior_samples_x = np.random.normal(loc=obs_x[i], scale=sig[i], size=N_POSTERIOR_SAMPLES)
        posterior_samples_y = np.random.normal(loc=obs_y[i], scale=sig[i], size=N_POSTERIOR_SAMPLES)
        posterior_samples_z = np.random.normal(loc=obs_z[i], scale=sig[i], size=N_POSTERIOR_SAMPLES)

        rcom_samples, theta_samples, phi_samples = cartesian2spherical(posterior_samples_x, posterior_samples_y, posterior_samples_z)
        dec_samples = 0.5 * np.pi - theta_samples
        redshift_samples = fast_z_at_value(COSMO.comoving_distance, rcom_samples * u.Mpc)
        rlum_samples = COSMO.luminosity_distance(redshift_samples).value

        samples_table = Table([phi_samples, dec_samples, rlum_samples, rcom_samples, redshift_samples], 
                                    names=('ra', 'dec', 'luminosity_distance', 'comoving_distance', 'redshift'))
        
        if kind == None:
            filename = os.path.join(POST_SAMPS_DIR, f"gw_{trial_idx}_{fagn_idx}_{i:05d}.h5")
        else:
            os.makedirs(f'{POST_SAMPS_DIR}/{kind}', exist_ok=True)
            filename = os.path.join(f'{POST_SAMPS_DIR}/{kind}', f"gw_{trial_idx}_{fagn_idx}_{i:05d}.h5")

        with h5py.File(filename, "a") as f:
            mock_group = f.require_group("mock")  # Takes place of approximant in real GW data
            mock_group.create_dataset('posterior_samples', data=samples_table)
    return


def save_coordinates(trial_idx, fagn_idx, true_x, true_y, true_z, v90, kind=None):
    if kind == None:
        post_dir = POST_SAMPS_DIR
        coord_dir = TRUE_COORDS_DIR
    else:
        post_dir = f'{POST_SAMPS_DIR}/{kind}'
        coord_dir = f'{TRUE_COORDS_DIR}/{kind}'

        os.makedirs(post_dir, exist_ok=True)
        os.makedirs(coord_dir, exist_ok=True)

    for _, infile in enumerate(glob.glob(f'{post_dir}/gw_{trial_idx}_{fagn_idx}_*.h5')):
        gw_idx = infile[-8:-3]
        r, theta, phi = cartesian2spherical(true_x[int(gw_idx)], true_y[int(gw_idx)], true_z[int(gw_idx)])  # Could've just used spherical coordinates directly, oh well.

        with open(f'{coord_dir}/true_r_theta_phi.txt', 'a') as f:
            f.write(f'{gw_idx}, {r}, {theta}, {phi}, {v90[int(gw_idx)]}\n')
    return


def make_skymaps(trial_idx, fagn_idx, kind=None):
    print('Verifying OMP_NUM_THREADS:')
    os.system("echo $OMP_NUM_THREADS")

    if kind == None:
        post_dir = POST_SAMPS_DIR
        sky_dir = SKYMAP_DIR
    else:
        post_dir = f'{POST_SAMPS_DIR}/{kind}'
        sky_dir = f'{SKYMAP_DIR}/{kind}'

        os.makedirs(post_dir, exist_ok=True)
        os.makedirs(sky_dir, exist_ok=True)

    for _, infile in enumerate(glob.glob(f'{post_dir}/gw_{trial_idx}_{fagn_idx}_*.h5')):
        print(f'Processing: {infile}')
        gw_idx = infile[-8:-3]
        outfile = f'skymap_{trial_idx}_{fagn_idx}_{gw_idx}.fits.gz'
        print(f'Output: {sky_dir}/{outfile}')

        os.system(f"ligo-skymap-from-samples {infile} --fitsoutname {outfile} --outdir {sky_dir} --jobs {NCPU}")

    os.system(f"rm -rf {sky_dir}/skypost.obj")
    return


def main(trial_idx, fagn_idx):
    check_directory(POST_SAMPS_DIR)
    check_directory(TRUE_COORDS_DIR)

    if NONUNIFORM:
    #     Nruns = 500
    #     tot = 0
    #     tagn = 0
    #     talt = 0
    #     for i in range(Nruns):
    #         print(i)
        from_agn_mask = np.random.rand(BATCH) < F_AGN_TRUE
        n_from_agn = np.sum(from_agn_mask)
        n_from_alt = np.sum(~from_agn_mask)

        true_x_agn, true_y_agn, true_z_agn, true_rcom_agn, true_theta_agn, true_phi_agn = make_real_gw_positions(n=n_from_agn, kind='agn')
        true_x_alt, true_y_alt, true_z_alt, true_rcom_alt, true_theta_alt, true_phi_alt = make_real_gw_positions(n=n_from_alt, kind='alt')

        obs_x_agn, obs_y_agn, obs_z_agn, sig_agn, v90_agn = make_observed_gw_positions(true_x_agn, true_y_agn, true_z_agn, true_rcom_agn, true_theta_agn, true_phi_agn)
        obs_x_alt, obs_y_alt, obs_z_alt, sig_alt, v90_alt = make_observed_gw_positions(true_x_alt, true_y_alt, true_z_alt, true_rcom_alt, true_theta_alt, true_phi_alt)
    
            # tot += len(obs_x_agn) + len(obs_x_alt)
            # tagn += len(obs_x_agn) / len(true_x_agn)
            # talt += len(obs_x_alt) / len(true_x_alt)

        #     print(tagn / (i + 1), talt / (i + 1), 'current, i=', i)
        # print(tot / Nruns, 'average observed')
        # print(tagn / Nruns, 'average agn')
        # print(talt / Nruns, 'average alt')
        # sys.exit(1)
        make_posterior_samples(trial_idx, fagn_idx, obs_x_agn, obs_y_agn, obs_z_agn, sig_agn, kind='agn')
        save_coordinates(trial_idx, fagn_idx, true_x_agn, true_y_agn, true_z_agn, v90_agn, kind='agn')

        make_posterior_samples(trial_idx, fagn_idx, obs_x_alt, obs_y_alt, obs_z_alt, sig_alt, kind='alt')
        save_coordinates(trial_idx, fagn_idx, true_x_alt, true_y_alt, true_z_alt, v90_alt, kind='alt')

        if MAKE_SKYMAPS:
            check_directory(SKYMAP_DIR)
            os.environ["OMP_NUM_THREADS"] = "1"  # Important for proper threading when making skymaps
            make_skymaps(trial_idx, fagn_idx, kind='agn')
            make_skymaps(trial_idx, fagn_idx, kind='alt')
    
    else:
        true_x, true_y, true_z, true_rcom, true_theta, true_phi = make_real_gw_positions()

        obs_x, obs_y, obs_z, sig, v90 = make_observed_gw_positions(true_x, true_y, true_z, true_rcom, true_theta, true_phi)
        make_posterior_samples(trial_idx, fagn_idx, obs_x, obs_y, obs_z, sig)
        save_coordinates(trial_idx, fagn_idx, true_x, true_y, true_z, v90)

        if MAKE_SKYMAPS:
            check_directory(SKYMAP_DIR)
            os.environ["OMP_NUM_THREADS"] = "1"  # Important for proper threading when making skymaps
            make_skymaps(trial_idx, fagn_idx)
    return


if __name__ == '__main__':
    main(trial_idx=0, fagn_idx=0)
