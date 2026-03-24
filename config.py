import os
import sys
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass

from redshift_utils import merger_rate_madau_dickinson, merger_rate_uniform, z_cut, uniform_comoving_prior
from astropy.cosmology import Planck15
from scipy.interpolate import interp1d


def positive_redshift_prior(z):
    return z_cut(-z, zcut=0)


class UniformComovingPrior:
    def __init__(self, cosmo):
        self.cosmo = cosmo

    def __call__(self, z):
        return uniform_comoving_prior(z, self.cosmo)
    

'''
required paths that cannot be specified:
self.SKYMAP_DIR = f"./skymaps_
self.SAMPLES_DIR = f"./posterior_samples_
self.ALL_TRUE_SOURCES = np.genfromtxt(f'./true_r_theta_phi_

FIXME: crossmatch does not get the cfg, which means it uses the cosmology from the default_globals file
'''


@dataclass
class Config:
    # ---------------- BASE INPUTS ----------------
    VERBOSE: bool = True
    THREADING: bool = False
    N_WORKERS: int = 16

    REAL_DATA: bool = False
    USE_SKYMAPS: bool = True
    SOURCE_FRAME_MASS_PRIOR: str | None = None

    N_REALIZATIONS: int = 1
    BATCH: int = 150
    TRUE_FAGNS: float = 0.5

    DIRECTORY_ID: str = 'all'
    AGN_DIST_DIR: str = './darksirenpop/agn_distribution'
    CATALOG_PATH: str = "./agn_data/Quaia_z15.csv"
    POST_DIR = '/home/lucas/Documents/PhD/fagn_posteriors'
    PLOT_DIR = './darksirenpop/plots'
    CMAP_PATH: str = "./completeness_map.fits"

    SKYMAP_CL: float = 0.999

    ZMIN: float = 1e-4
    ZMAX: float = 1.5
    AGN_ZMAX: float = 10
    AGN_ZCUT: float = 1.5

    QLF: str | None = None  # None or 'kulkarni' -- shenA and shenB not implemented
    AGN_ZPRIOR: str = 'uniform_comoving_volume'  # Valid: 'positive_redshift', 'uniform_comoving_volume', '44.5', '45.0', '45.5', '46.0', '46.5'
    LUM_THRESH: str = 'zero_upto_cut'  # Valid: '44.5', '45.0', '45.5', '46.0', '46.5' (V25 completeness bins), 'zero' (complete catalog), 'zero_upto_cut' (complete catalog up to a redshift cut), 'inf' (empty catalog)

    MASK_GALACTIC_PLANE: bool = True
    PLOT_CMAP: bool = False

    ADD_NAGN_TO_CAT: int = int(3.5e5)
    ASSUME_PERFECT_REDSHIFT: bool = False
    AGN_ZERROR: str | bool | float = 'quaia'  # 'quaia', 'False', or float

    CORRECT_TIME_DILATION: bool = True
    MERGER_RATE: str = 'madau'

    # ---------------- MAGICS THAT SHOULDN'T NEED TO CHANGE EVER ----------------
    LINAX: bool = True
    CALC_LOGLLH_AT_N_POINTS: int = 1000
    AGN_ZPRIOR_NORM_AX_N_POINTS: int = 1024+1
    CMAP_NSIDE: int = 64
    H0: float = 67.9
    OM0: float = 0.3065

    # ---------------- DERIVED VARIABLES ----------------
    # LOG_LLH_X_AX: np.ndarray | None = None
    # ALL_TRUE_SOURCES: np.ndarray | None = None
    # SKYMAP_DIR: str | None = None
    # SAMPLES_DIR: str | None = None
    # AGN_ZPRIOR_NORM_AX: np.ndarray | None = None
    # REALIZED_FAGNS: np.ndarray | None = None
    # N_TRUE_FAGNS: int | None = None

    # COMDIST_MIN: float | None = None
    # COMDIST_MAX: float | None = None
    # AGN_COMDIST_MAX: float | None = None

    # THRESHOLD_MAP: dict | None = None
    # Z_EDGES: np.ndarray | None = None
    # QUAIA_C_VALS: np.ndarray | None = None

    # FAGN_POSTERIOR_FNAME: str | None = None
    # AGN_ZPRIOR_FUNCTION: callable | None = None
    # Z_INTEGRAL_AX: np.ndarray | None = None

    # ALL_GW_FNAMES: np.ndarray | None = None
    # FILE_TYPE: str | None = None


    def get_z_integral_ax(self, at_least_N=1, npoints_min=512):
        """
        Compute the redshift integral axis based on AGN redshift errors.
        """
        if self.AGN_ZERROR == 'quaia':
            smallest_error = np.min(self.quaia_errors)
        else:
            smallest_error = self.AGN_ZERROR

        if smallest_error == 0:
            return np.linspace(self.ZMIN, self.ZMAX, npoints_min + 1)
        else:
            npoints = int(2**np.ceil(np.log2(at_least_N * (self.ZMAX - self.ZMIN) / smallest_error)))
            print(f'Requiring at least {npoints} points in redshift integral axis to capture all AGN info '
                  f'for smallest error: {smallest_error}.')
            npoints = max(npoints, npoints_min)
            return np.linspace(self.ZMIN, self.ZMAX, npoints + 1)


    def get_agn_zprior(self):
        """
        Return a callable for the AGN redshift prior based on current config.
        """
        
        if self.AGN_ZPRIOR == 'uniform_comoving_volume':
            return UniformComovingPrior(self.COSMO)

        elif str(self.AGN_ZPRIOR[:4]) in ['44.5', '45.0', '45.5', '46.0', '46.5']:
            filename = f'{self.AGN_DIST_DIR}/agn_redshift_pdf_{self.AGN_ZPRIOR}.npy'
            if self.VERBOSE:
                print(f'Loading AGN redshift distribution from file: {filename}')
            z, n = np.load(filename)
            return interp1d(z, n, bounds_error=False, fill_value=0)
        
        elif self.AGN_ZPRIOR == 'positive_redshift':
            return positive_redshift_prior
        
        else:
            sys.exit(f'AGN redshift prior not recognized: {self.AGN_ZPRIOR}. Exiting...')
    

    # ---------------- FINALIZE ----------------
    def finalize(self):
        # ---------------- COSMOLOGY ----------------
        self.COSMO = Planck15.clone(H0=self.H0, Om0=self.OM0)

        # ---------------- LLH GRID ----------------
        self.LOG_LLH_X_AX = np.linspace(0.0001, 0.9999, self.CALC_LOGLLH_AT_N_POINTS)

        # -------- VALIDATION --------
        if not self.AGN_ZERROR and not self.ASSUME_PERFECT_REDSHIFT:
            sys.exit('Stop trying to break my code.')

        if (self.LUM_THRESH in ['44.5', '45.0', '45.5', '46.0', '46.5']) & (self.AGN_ZCUT < 1.3125):
            raise ValueError(
                f"V25 completeness bins require AGN_ZCUT to be higher than V25's highest z-bin, which is 1.3125. Got: {self.AGN_ZCUT}"
            )

        # -------- ENVIRONMENT --------
        if self.THREADING:
            os.environ["OMP_NUM_THREADS"] = "1"

        # -------- TRUE/REALIZED FAGNS --------
        self.TRUE_FAGNS = np.tile(self.TRUE_FAGNS, self.N_REALIZATIONS)
        self.REALIZED_FAGNS = np.random.binomial(self.BATCH, self.TRUE_FAGNS) / self.BATCH
        self.N_TRUE_FAGNS = len(self.TRUE_FAGNS)

        # -------- DISTANCES --------
        self.COMDIST_MIN = self.COSMO.comoving_distance(self.ZMIN).value
        self.COMDIST_MAX = self.COSMO.comoving_distance(self.ZMAX).value
        self.AGN_COMDIST_MAX = self.COSMO.comoving_distance(self.AGN_ZMAX).value

        # -------- QUAIA COMPLETENESS --------
        self.THRESHOLD_MAP = {"46.5": 0, "46.0": 1, "45.5": 2, "45.0": 3, "44.5": 4}
        self.Z_EDGES = np.array([0.0000, 0.1875, 0.3750, 0.5625, 0.7500, 0.9375,
                                 1.1250, 1.3125, self.AGN_ZCUT, self.AGN_ZMAX])
        self.QUAIA_C_VALS = np.array([
            [0.000, 0.000, 0.229, 0.945, 0.718],
            [1.000, 1.000, 1.000, 1.000, 0.781],
            [1.000, 1.000, 1.000, 1.000, 0.408],
            [1.000, 0.891, 1.000, 0.681, 0.211],
            [1.000, 1.000, 0.994, 0.429, 0.138],
            [1.000, 1.000, 0.837, 0.258, 0.085],
            [0.927, 0.940, 0.576, 0.179, 0.060],
            [1.000, 0.482, 0.155, 0.053, 0.053],
            [0., 0., 0., 0., 0.]
        ])

        # -------- AGN REDSHIFT DISTRIBUTION --------
        if self.AGN_ZPRIOR in ['44.5', '45.0', '45.5', '46.0', '46.5']:
            self.AGN_ZPRIOR = f'{self.AGN_ZPRIOR}_{self.QLF}'
        self.AGN_ZPRIOR_NORM_AX = np.linspace(self.ZMIN, self.AGN_ZMAX, self.AGN_ZPRIOR_NORM_AX_N_POINTS)

        # -------- WARNINGS --------
        if self.AGN_ZPRIOR and self.AGN_ZPRIOR[:4] != self.LUM_THRESH:
            print(f'WARNING: You are performing an analysis assuming log10(Lbol) >= {self.LUM_THRESH}, '
                  f'while the AGN redshift posteriors in your catalog may have an inconsistent prior: {self.AGN_ZPRIOR}')

        # -------- AGN ERRORS --------
        if self.AGN_ZERROR == 'quaia':
            self.quaia_errors = pd.read_csv(self.CATALOG_PATH)["redshift_quaia_err"]

        # -------- MERGER RATE --------
        if self.MERGER_RATE == 'madau':
            self.MERGER_RATE_EVOLUTION = merger_rate_madau_dickinson
            self.MERGER_RATE_KWARGS = {}
        elif self.MERGER_RATE == 'uniform':
            self.MERGER_RATE_EVOLUTION = merger_rate_uniform
            self.MERGER_RATE_KWARGS = {}

        # -------- POSTERIOR FILENAME --------
        if self.REAL_DATA:
            self.FAGN_POSTERIOR_FNAME = (
                f'p26_post_realdata_{self.REAL_DATA}_useskymap_{self.USE_SKYMAPS}_rate_{self.MERGER_RATE}'
                f'_timedil_{self.CORRECT_TIME_DILATION}_agnZprior_{self.AGN_ZPRIOR}_Lthresh_{self.LUM_THRESH}'
                f'_perfz_{self.ASSUME_PERFECT_REDSHIFT}_GPmask_{self.MASK_GALACTIC_PLANE}_CL_{self.SKYMAP_CL}'
                f'_gwZmax_{self.ZMAX}_agnZcut_{self.AGN_ZCUT}'
            )
        else:
            self.FAGN_POSTERIOR_FNAME = (
                f'p26_post_realdata_{self.REAL_DATA}_useskymap_{self.USE_SKYMAPS}_rate_{self.MERGER_RATE}'
                f'_timedil_{self.CORRECT_TIME_DILATION}_agnZprior_{self.AGN_ZPRIOR}_Lthresh_{self.LUM_THRESH}'
                f'_perfz_{self.ASSUME_PERFECT_REDSHIFT}_GPmask_{self.MASK_GALACTIC_PLANE}'
                f'_addAGN_{self.ADD_NAGN_TO_CAT}_nreal_{self.N_REALIZATIONS}_batch_{self.BATCH}_CL_{self.SKYMAP_CL}'
                f'_agnZerr_{self.AGN_ZERROR}_gwZmax_{self.ZMAX}_agnZcut_{self.AGN_ZCUT}'
            )

        # -------- AGN REDSHIFT PRIORS & INTEGRAL AXIS --------
        self.AGN_ZPRIOR_FUNCTION = self.get_agn_zprior()
        self.Z_INTEGRAL_AX = self.get_z_integral_ax()

        # -------- GW FILES --------
        self.SKYMAP_DIR = f"./skymaps_{self.DIRECTORY_ID}/"
        self.SAMPLES_DIR = f"./posterior_samples_{self.DIRECTORY_ID}/"

        self.ALL_TRUE_SOURCES = np.genfromtxt(f'./true_r_theta_phi_{self.DIRECTORY_ID}.txt', delimiter=',')
        self.ALL_TRUE_SOURCES = self.ALL_TRUE_SOURCES[self.ALL_TRUE_SOURCES[:,0].argsort()]
        self.TRUE_SOURCE_IDENTIFIERS = self.ALL_TRUE_SOURCES[:,0]

        if self.USE_SKYMAPS:
            self.ALL_GW_FNAMES = np.array(glob.glob(self.SKYMAP_DIR + 'skymap_*'))
            self.FILE_TYPE = 'skymap'
        else:
            self.ALL_GW_FNAMES = np.array(glob.glob(self.SAMPLES_DIR + 'gw_*'))
            self.FILE_TYPE = 'samples'
