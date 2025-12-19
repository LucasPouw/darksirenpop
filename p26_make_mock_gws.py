import numpy as np
from utils import uniform_shell_sampler, make_nice_plots, spherical2cartesian, cartesian2spherical
from tqdm import tqdm
from default_globals import *
import astropy.units as u
from scipy.integrate import quad, romb
from concurrent.futures import as_completed, ThreadPoolExecutor
import traceback
import sys, os
from priors import *
from scipy.stats import vonmises_fisher
from scipy.optimize import root_scalar
from scipy.special import erf
import h5py
from astropy.table import Table
import shutil
import glob
from redshift_utils import fast_z_at_value


########################### INPUT PARAMETERS #################################
MAKE_SKYMAPS = True  # Only implemented in combination with USE_ONLY_3DLOC = True
ID = 'x'
SKYMAP_DIR = f'./skymaps_{ID}'
POST_SAMPS_DIR = f'./posterior_samples_{ID}'
CAT_DIR = f'./catalogs_{ID}'

NCPU = os.cpu_count()

SKY_AREA_LOW = 100
SKY_AREA_HIGH = 10000

SKYMAP_CL = 0.9  # ALWAYS CHECK, MAKE 1 UNLESS YOU WANT TO CHANGE THE LIKELIHOOD FUNCTION

N_POSTERIOR_SAMPLES = int(1e4)

# AGN_ZERROR = float(0.05)  # Absolute error for now, same for all AGN, posterior = Gaussian * dVdz prior / norm TODO: Verify that this likelihood is fine for Quaia
AGN_ZERROR = int(0)

ZMIN = 1e-4
ZMAX = 1.5  # p_rate(z > ZMAX) = 0

COMDIST_MIN = COSMO.comoving_distance(ZMIN).value
COMDIST_MAX = COSMO.comoving_distance(ZMAX).value  # Maximum comoving distance in Mpc

VOLUME = 4 / 3 * np.pi * COMDIST_MAX**3
AGN_NUMDENS = 2e4 / VOLUME
BATCH = int(2e4)
N_TRIALS = 1
MAX_N_FAGNS = 1
CALC_LOGLLH_AT_N_POINTS = 1000

NAGN = int( np.ceil(AGN_NUMDENS * VOLUME) )

# Panicky factor to generate AGN in a larger volume to check boundary issues in USE_ONLY_3DLOC case
AAA = 1
AGN_NUMDENS /= AAA**3
print(f'\nPANICK FACTOR = {AAA}\n')

fname = f'posteriors_SKYMAP_CL_{SKYMAP_CL}_AGNZERROR_{AGN_ZERROR}_NAGN_{NAGN}_BATCH_{BATCH}'

assert NAGN >= BATCH, f'Every AGN-origin GW must come from a unique AGN. Got {NAGN} AGN and {BATCH} AGN-origin GWs.'
print(f'#AGN is rounded from {AGN_NUMDENS * VOLUME} to {NAGN}, giving a number density of {NAGN / VOLUME:.3e}. Target was {AGN_NUMDENS:.3e}.')

N_TRUE_FAGNS = min(BATCH + 1, MAX_N_FAGNS)  # Cannot create more f_agn values than BATCH+1 and don't want to generate more than MAX_N_FAGNS

LOG_LLH_X_AX = np.linspace(0.0001, 0.9999, CALC_LOGLLH_AT_N_POINTS)

if N_TRUE_FAGNS == 1:
    USE_N_AGN_EVENTS = BATCH
else:
    USE_N_AGN_EVENTS = np.arange(0, BATCH + 1, int(BATCH / (N_TRUE_FAGNS - 1)), dtype=np.int32)
TRUE_FAGNS = np.atleast_1d(USE_N_AGN_EVENTS / BATCH)

####################################################################


def sample_sigmas(n_events=BATCH, mean=50, std=50):
    # Estimated by eye, seems to match observations quite well
    return np.abs( np.random.normal(loc=mean, scale=std, size=n_events) )
    

def cl_volume_from_sigma(sigmas):
    '''Calculates length of CL% vector when x, y, z are normally distributed with the same sigma'''
    radii = np.zeros(len(sigmas))
    for i, sigma in enumerate(sigmas):
        vector_length_cdf = lambda x: erf(x / (sigma * np.sqrt(2))) - np.sqrt(2) / (sigma * np.sqrt(np.pi)) * x * np.exp(-x**2 / (2 * sigma**2)) - SKYMAP_CL
        radii[i] = root_scalar(vector_length_cdf, bracket=[0, 100 * sigma], method='bisect').root
    return 4 * np.pi * radii**3 / 3, radii


def get_observed_gw_3dloc_agn_catalog(n_agn_events, fagn_idx, trial_idx):

    # AGN catalog
    true_agn_rcom, true_agn_theta, true_agn_phi = uniform_shell_sampler(COMDIST_MIN, AAA * COMDIST_MAX, NAGN)  # NO AGN NEEDED ABOVE RMAX_GW, p_rate = 0

    if MAKE_SKYMAPS:  # TODO: save only the true AGN positions for testing
        with h5py.File(CAT_DIR + f'/mockcat_{trial_idx}_{fagn_idx}.hdf5', "w") as f:
            f.create_dataset('comoving_distance', data=true_agn_rcom)
            f.create_dataset('ra', data=true_agn_phi)
            f.create_dataset('dec', data=np.pi * 0.5 - true_agn_theta)

    true_agn_x, true_agn_y, true_agn_z = spherical2cartesian(true_agn_rcom, true_agn_theta, true_agn_phi)
    true_agn_redshift = fast_z_at_value(COSMO.comoving_distance, true_agn_rcom * u.Mpc)

    if not AGN_ZERROR:
        obs_agn_redshift = true_agn_redshift
    else:
        obs_agn_redshift = np.random.normal(loc=true_agn_redshift, scale=AGN_ZERROR, size=NAGN)
        # obs_agn_redshift = stats.truncnorm.rvs(size=NAGN, a=(ZMIN - true_agn_redshift) / AGN_ZERROR, b=(np.inf - true_agn_redshift) / AGN_ZERROR, loc=true_agn_redshift, scale=AGN_ZERROR)

    obs_agn_rcom, obs_agn_theta, obs_agn_phi = COSMO.comoving_distance(obs_agn_redshift).value, true_agn_theta, true_agn_phi
    obs_agn_x, obs_agn_y, obs_agn_z = spherical2cartesian(obs_agn_rcom, obs_agn_theta, obs_agn_phi)

    # GWs & likelihood
    agn_possibly_generating_gw = np.arange(NAGN)[true_agn_rcom < COMDIST_MAX]
    agn_generating_gw = np.random.choice(agn_possibly_generating_gw, size=n_agn_events, replace=False)  # Do not generate GWs from the same AGN (see Hitchhiker's guide) - I think it only matters when doing selection effects?
    true_x_gw_agn, true_y_gw_agn, true_z_gw_agn = true_agn_x[agn_generating_gw], true_agn_y[agn_generating_gw], true_agn_z[agn_generating_gw]

    true_rcom_gw_alt, true_theta_gw_alt, true_phi_gw_alt = uniform_shell_sampler(COMDIST_MIN, COMDIST_MAX, BATCH - n_agn_events)
    true_x_gw_alt, true_y_gw_alt, true_z_gw_alt = spherical2cartesian(true_rcom_gw_alt, true_theta_gw_alt, true_phi_gw_alt)

    true_x_gw, true_y_gw, true_z_gw = np.hstack((true_x_gw_agn, true_x_gw_alt)), np.hstack((true_y_gw_agn, true_y_gw_alt)), np.hstack((true_z_gw_agn, true_z_gw_alt))

    locvol_sigmas = sample_sigmas()  # 3D spherical Gaussian with same sigma in each cartesian coordinate
    _, locvol_radius = cl_volume_from_sigma(locvol_sigmas)

    obs_gw_x, obs_gw_y, obs_gw_z = np.random.normal(loc=true_x_gw, scale=locvol_sigmas, size=BATCH), np.random.normal(loc=true_y_gw, scale=locvol_sigmas, size=BATCH), np.random.normal(loc=true_z_gw, scale=locvol_sigmas, size=BATCH)

    return true_x_gw, true_y_gw, true_z_gw, obs_gw_x, obs_gw_y, obs_gw_z, locvol_radius, locvol_sigmas, obs_agn_x, obs_agn_y, obs_agn_z


def make_skymaps_and_catalog(n_agn_events, fagn_idx, trial_idx):
    true_x_gw, true_y_gw, true_z_gw, obs_gw_x, obs_gw_y, obs_gw_z, locvol_radius, locvol_sigmas, obs_agn_x, obs_agn_y, obs_agn_z = get_observed_gw_3dloc_agn_catalog(n_agn_events, fagn_idx, trial_idx)

    # First make posterior samples .hdf5 files
    for i in range(BATCH):
        posterior_samples_x = np.random.normal(loc=obs_gw_x[i], scale=locvol_sigmas[i], size=N_POSTERIOR_SAMPLES)
        posterior_samples_y = np.random.normal(loc=obs_gw_y[i], scale=locvol_sigmas[i], size=N_POSTERIOR_SAMPLES)
        posterior_samples_z = np.random.normal(loc=obs_gw_z[i], scale=locvol_sigmas[i], size=N_POSTERIOR_SAMPLES)

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
    
    # Then process all files into skymaps, this is the slow part
    print('Verifying OMP_NUM_THREADS:')
    os.system("echo $OMP_NUM_THREADS")

    for _, infile in enumerate(glob.glob(f'{POST_SAMPS_DIR}/gw_{trial_idx}_{fagn_idx}_*.h5')):
        print(f'Processing: {infile}')
        gw_idx = infile[-8:-3]

        r, theta, phi = cartesian2spherical(true_x_gw[int(gw_idx)], true_y_gw[int(gw_idx)], true_z_gw[int(gw_idx)])
        with open(f'true_r_theta_phi_{ID}.txt', 'a') as f:
            f.write(f'{gw_idx}, {r}, {theta}, {phi}\n')

        outfile = f'skymap_{trial_idx}_{fagn_idx}_{gw_idx}.fits.gz'
        print(f'Output: {SKYMAP_DIR}/{outfile}')

        os.system(f"ligo-skymap-from-samples {infile} --fitsoutname {outfile} --outdir {SKYMAP_DIR} --jobs {NCPU}")
    
    os.system(f"rm -rf {SKYMAP_DIR}/skypost.obj")
    
    return locvol_radius, obs_agn_x, obs_agn_y, obs_agn_z


def process_skymaps(locvol_radius, obs_agn_x, obs_agn_y, obs_agn_z):
    return np.ones(BATCH), np.ones(BATCH)


####################### ANALYSIS #########################


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


def generate_and_process_universe_realization(fagn, fagn_idx, trial_idx):

    n_agn_events = int(fagn * BATCH)
    locvol_radius, obs_agn_x, obs_agn_y, obs_agn_z = make_skymaps_and_catalog(n_agn_events, fagn_idx, trial_idx)
    return process_skymaps(locvol_radius, obs_agn_x, obs_agn_y, obs_agn_z)


def full_analysis_likelihood_thread(trial_idx):
    
    log_llh = np.zeros((len(LOG_LLH_X_AX), N_TRUE_FAGNS))
    for fagn_idx, fagn_true in enumerate(TRUE_FAGNS):
        # if fagn_idx <= 1:
        #     continue

        # fagn_realized = np.random.binomial(BATCH, fagn_true) / BATCH  # Observed f_agn fluctuates around the true value
        fagn_realized = fagn_true
        print('FAGN = FAGN_TRUE')
        print(f"fagn realized: {fagn_realized}, fagn true: {fagn_true}")
        S_agn, S_alt = generate_and_process_universe_realization(fagn=fagn_realized, fagn_idx=fagn_idx, trial_idx=trial_idx)
        loglike = np.log(S_agn[:,None] * SKYMAP_CL * LOG_LLH_X_AX[None,:] + S_alt[:,None] * (1 - SKYMAP_CL * LOG_LLH_X_AX[None,:]))
        
        log_llh[:,fagn_idx] = np.sum(loglike, axis=0)  # sum over all GWs

    return trial_idx, log_llh


if __name__ == '__main__':

    posteriors = np.zeros((N_TRIALS, CALC_LOGLLH_AT_N_POINTS, N_TRUE_FAGNS))

    if not MAKE_SKYMAPS:  # Then we use multiple cores for multiple Universe realizations
        with ThreadPoolExecutor(max_workers=NCPU) as executor:
            future_to_index = {executor.submit(full_analysis_likelihood_thread, index): index for index in range(N_TRIALS)}
            for future in tqdm(as_completed(future_to_index), total=N_TRIALS):
                try:
                    trial_idx, log_llh = future.result(timeout=20)
                    posteriors[trial_idx,:,:] = log_llh
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error processing trial {future_to_index[future]}: {e}")

    else:  # We will need all the cores for generating skymaps
        check_directory(POST_SAMPS_DIR)
        check_directory(SKYMAP_DIR)
        check_directory(CAT_DIR)
        os.environ["OMP_NUM_THREADS"] = "1"  # Important for proper threading when making skymaps
        
        for trial_idx in range(N_TRIALS):
            _, log_llh = full_analysis_likelihood_thread(trial_idx)
            posteriors[trial_idx,:,:] = log_llh

    np.save(os.path.join(sys.path[0], fname), posteriors)
