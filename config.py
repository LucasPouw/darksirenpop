import os
import sys
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Union
from pathlib import Path

from darksirenpop.utilities.redshift_utils import *
from astropy.cosmology import Planck15
from scipy.interpolate import interp1d, CubicSpline


def positive_redshift_prior(z):
    return z_cut(-z, zcut=0)


class UniformComovingPrior:
    def __init__(self, cosmo):
        self.cosmo = cosmo

    def __call__(self, z):
        return uniform_comoving_prior(z, self.cosmo)
    

@dataclass
class Config:
    # ---------------- BASE INPUTS ----------------
    FLAT_GW_POSTERIORS: bool = False
    VERBOSE: bool = False
    THREADING: bool = False
    N_WORKERS: int = 16
    JOB_ID: int = 0  # Important when running on a cluster to avoid two files with the same filename

    REAL_DATA: bool = False
    USE_SKYMAPS: bool = True
    # SOURCE_FRAME_MASS_PRIOR: str | None = None

    N_REALIZATIONS: int = 1  # Load this many from the provided MOCKDATA_ROOT
    MOCKDATA_ROOT: str = None  # Specify the root directory where all mock data is, generated with all desired properties
    NGW: int = 150

    METADATA_PATH: str = './runs.json'
    AGN_DIST_DIR: str = '/home/lucas/Documents/PhD/generated_data/em'
    CATALOG_PATH: str = "/home/lucas/Documents/PhD/generated_data/em/quaia_zleq3_withlumcorr.csv"
    POST_DIR = './fagn_posteriors'
    PLOT_DIR = './darksirenpop/plots'
    CMAP_PATH: str = "./darksirenpop/mock_analysis"  # TODO 2 Sept 2026: completeness map not used, currently hard-coded the removal of AGN with |b| <= 10
    PDET_PATH: str = "/home/lucas/Documents/PhD/darksirenpop/mock/pdet"

    REAL_POSTERIOR_JSON_DIR: str = '/home/lucas/Documents/PhD/generated_data/jsons'  # Store evaluated terms of the likelihood calculation in this directory
    REAL_SKYMAP_JSON_PATH: str = '/home/lucas/Documents/PhD/generated_data/gw/reweighted-gwtc5/real_skymaps_reweight_gwtc5.json'
    REAL_SAMPLES_JSON_PATH: str = '/home/lucas/Documents/PhD/generated_data/gw/reweighted-gwtc5/real_PEsamples_reweight_gwtc5.json'
    REAL_ZPOSTS_JSON_PATH: str = '/home/lucas/Documents/PhD/generated_data/gw/reweighted-gwtc5/real_skymaps_evaluated.json'
    REAL_CW_ZPOSTS_JSON_PATH: str = '/home/lucas/Documents/PhD/generated_data/gw/reweighted-gwtc5/real_cw_skymaps_evaluated.json'

    SKYMAP_CL: float = 0.999
    ZMIN: float = 1e-6
    ZMAX: float = 10
    ZTHR: float = np.inf
    AGN_ZMAX: float = 10
    AGN_ZCUT: float = 3.0

    QLF: str = 'kulkarni'  # 'kulkarni' -- shenA and shenB not tested, QLF is only used when AGN_ZPRIOR is one of ['44.5', '45.0', '45.5', '46.0', '46.5']
    AGN_ZPRIOR: str = 'uniform_comoving_volume'  # Valid: 'positive_redshift', 'uniform_comoving_volume', '44.5', '45.0', '45.5', '46.0', '46.5'
    LUM_THRESH: str = 'zero_upto_cut'  # Valid: '44.5', '45.0', '45.5', '46.0', '46.5' (V25 completeness bins), 'zero' (complete catalog), 'zero_upto_cut' (complete catalog up to a redshift cut), 'inf' (empty catalog)

    MASK_GALACTIC_PLANE: bool = True
    PLOT_CMAP: bool = False

    ADD_NAGN_TO_CAT: int = int(3.5e5)
    ASSUME_PERFECT_REDSHIFT: bool = False
    AGN_ZERROR: Union[float, bool, str] = 'quaia'  # 'quaia', False or float

    CORRECT_TIME_DILATION: bool = True
    MERGER_RATE: str = 'madau'
    RATE_PARAMETERS: dict = field(default_factory=dict)

    LABEL: str = 'none'

    SNR_THR = 10
    FAR_THR = 1

    # ---------------- MAGIC NUMBERS THAT SHOULDN'T NEED TO CHANGE EVER ----------------
    LINAX: bool = True
    CALC_LOGLLH_AT_N_POINTS: int = 1000
    AGN_ZPRIOR_NORM_AX_N_POINTS: int = 1024+1
    CMAP_NSIDE: int = 64
    H0: float = 67.9
    OM0: float = 0.3065

    # ---------------- DERIVED VARIABLES ----------------
    # LOG_LLH_X_AX: np.ndarray | None = None
    # AGN_ZPRIOR_NORM_AX: np.ndarray | None = None
    # TRUE_FAGNS: np.ndarray | None = None

    # COMDIST_MIN: float | None = None
    # COMDIST_MAX: float | None = None
    # AGN_COMDIST_MAX: float | None = None

    # THRESHOLD_MAP: dict | None = None
    # Z_EDGES: np.ndarray | None = None
    # QUAIA_C_VALS: np.ndarray | None = None

    # FAGN_POSTERIOR_FNAME: str | None = None
    # AGN_ZPRIOR_FUNCTION: callable | None = None
    # Z_INTEGRAL_AX: np.ndarray | None = None


    def get_z_integral_ax(self, at_least_N_in_one_sigma=1, npoints_min=1024):
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
            npoints = int(2**np.ceil(np.log2(at_least_N_in_one_sigma * (self.ZMAX - self.ZMIN) / smallest_error)))
            if self.VERBOSE:
                print(f'Requiring at least {npoints} points in redshift integral axis to capture all AGN info '
                    f'for smallest error: {smallest_error}.')
            npoints = max(npoints, npoints_min)
            if self.VERBOSE:
                print(f'Actual number of points on the axis: {npoints + 1}.\n')
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

        if self.FLAT_GW_POSTERIORS:
            print('Forcing skymap CL to 1 since we test normalizations using flat GW posteriors.')
            self.SKYMAP_CL = 1

        # ---------------- COSMOLOGY ----------------
        self.COSMO = Planck15.clone(H0=self.H0, Om0=self.OM0)

        # ---------------- LLH GRID ----------------
        self.LOG_LLH_X_AX = np.linspace(0.0001, 0.9999, self.CALC_LOGLLH_AT_N_POINTS)

        # -------- VALIDATION --------
        if not self.AGN_ZERROR and not self.ASSUME_PERFECT_REDSHIFT:
            sys.exit('Stop trying to break my code.')

        if (self.LUM_THRESH in ['44.5', '45.0', '45.5', '46.0', '46.5']) & (self.AGN_ZCUT < 1.3125):
            raise ValueError(f"V25 completeness bins require AGN_ZCUT to be higher than V25's highest z-bin, which is 1.3125. Got: {self.AGN_ZCUT}")

        # -------- ENVIRONMENT --------
        if self.THREADING:
            os.environ["OMP_NUM_THREADS"] = "1"

        # -------- TRUE FAGNS --------
        if self.REAL_DATA:
            self.TRUE_FAGNS = np.tile(0.5, self.N_REALIZATIONS)  # Placeholder, not used anywhere
        else:  # Extract true injected f_agn from file name
            fagns = []
            for i in range(self.N_REALIZATIONS):
                try:
                    output_dir = glob.glob(f'{self.MOCKDATA_ROOT}/output_run_{i+1}_*')[0]
                except Exception as e:
                    sys.exit(f'Problem loading file "{self.MOCKDATA_ROOT}/output_run_{i+1}_*". Error message: {e}')
                fagn = output_dir.split('_')[-1]
                fagns.append(fagn)
            self.TRUE_FAGNS = np.array(fagns)

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

        if self.AGN_ZPRIOR and self.AGN_ZPRIOR[:4] != self.LUM_THRESH:
            if self.VERBOSE:
                print(f'WARNING: You are performing an analysis assuming log10(Lbol) >= {self.LUM_THRESH}, '
                    f'but there is also a data-informed completeness available for your chosen AGN redshift prior: {self.AGN_ZPRIOR}\n')

        self.AGN_ZPRIOR_FUNCTION = self.get_agn_zprior()

        # -------- AGN Z-ERRORS --------
        if self.AGN_ZERROR == 'quaia':
            self.quaia_errors = pd.read_csv(self.CATALOG_PATH)["redshift_quaia_err"]

        self.Z_INTEGRAL_AX = self.get_z_integral_ax()

        # -------- MERGER RATE ALT GWS --------
        if self.MERGER_RATE == 'madau':
            self.MERGER_RATE_EVOLUTION = merger_rate_madau_dickinson
            self.MERGER_RATE_KWARGS = self.RATE_PARAMETERS
        elif self.MERGER_RATE == 'madau_lvk_MAP':
            self.MERGER_RATE_EVOLUTION = merger_rate_madau_dickinson
            self.MERGER_RATE_KWARGS = {'b': 3.027636508120894, 'c': 3.1826719462323076, 'd': 7.039509800907652}  # GWTC-5
        elif self.MERGER_RATE == 'lowZ_sfr':
            self.MERGER_RATE_EVOLUTION = merger_rate_lowZ_sfr
            self.MERGER_RATE_KWARGS = {}
        elif self.MERGER_RATE == 'uniform':
            self.MERGER_RATE_EVOLUTION = merger_rate_uniform
            self.MERGER_RATE_KWARGS = {}

        # -------- POSTERIOR FILENAME --------
        self.FAGN_POSTERIOR_FNAME = f'p26_post_realdata_{self.REAL_DATA}_job{self.JOB_ID}'
        if self.REAL_DATA:
            self.FAGN_POSTERIOR_FNAME += f'_{self.LUM_THRESH}'

        # -------- GW SELECTION EFFECTS --------
        if np.isinf(self.ZTHR):
            self.ALPHA_ALT = 1
            self.PDET = np.ones_like(self.Z_INTEGRAL_AX)
        else:
            if self.ZTHR not in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                sys.exit(f'GW detection threshold only has 0.3 and 1.0 implemented. Got: {self.ZTHR}')
            else:
                alpha_alt_dict = {0.3: 0.002918, 
                                  0.4: 0.00693,
                                  0.5: 0.01349,
                                  0.6: 0.0231,
                                  0.7: 0.03618,
                                  0.8: 0.0529,
                                  0.9: 0.0736,
                                  1.0: 0.09788}
                z_arr, pdet = np.load(f'{self.PDET_PATH}/pdet_z_{self.ZTHR}.npy')

                self.ALPHA_ALT = alpha_alt_dict[self.ZTHR] #/ 1.02
                Pdet = CubicSpline(z_arr, pdet, extrapolate=False)
                self.PDET = Pdet

        # -------- GW FILES --------
        if self.REAL_DATA:
            if self.USE_SKYMAPS:
                self.JSON_PATH = self.REAL_SKYMAP_JSON_PATH

            # TODO: add likelihood calculation that directly uses GW samples, and no skymaps. 
            else:  
                raise NotImplementedError('Only analysis of skymaps is fully implemented and tested.')
                # self.JSON_PATH = self.REAL_SAMPLES_JSON_PATH
