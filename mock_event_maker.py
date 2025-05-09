import numpy as np
import pandas as pd
import sys, os
import astropy.units as u
from utils import fast_z_at_value, check_equal, uniform_shell_sampler, spherical2cartesian, cartesian2spherical
from agn_catalogs_and_priors.mock_catalog_maker import MockCatalog
from priors import *
import json
import shutil
from default_arguments import *
from tqdm import tqdm
import h5py
from scipy.stats import vonmises_fisher
from astropy.table import Table
from concurrent.futures import as_completed, ThreadPoolExecutor
from multiprocessing import cpu_count
import traceback
import healpy as hp


class MockEvent:

    def __init__(
                self,
                n_events: int,
                f_agn: float,
                zdraw: float,
                use_extrinsic: bool=False,
                use_intrinsic: bool=False,
                sky_area_low: float=None,
                sky_area_high: float=None,
                lumdist_relerr: float=None,
                mass_relerr: float=None,
                catalog: MockCatalog=None,    # Either a MockCatalog object you just made
                catalog_path: str=None,       # Or a pre-existing one that is stored in hdf5 format
                model_dict: dict=None,
                hyperparam_dict_agn_path: str=None,
                hyperparam_dict_alt_path: str=None,
                n_posterior_samples: int=DEFAULT_N_POSTERIOR_SAMPLES,
                cosmology=DEFAULT_COSMOLOGY,
                outdir: str=DEFAULT_POSTERIOR_OUTDIR,
                ncpu: int=DEFAULT_N_CPU
            ):
        
        """
        model_dict: dict            Contains all population prior models as strings in the format {`agn`: [`agn_model1`, ..., `agn_modelN`], `alt`: [...]},
        catalog: MockCatalog        
        """

        assert use_extrinsic or use_intrinsic, 'Set at least one of `use_extrinsic` and `use_intrinsic` to True.'
        
        if os.path.isdir(outdir):
            if len(os.listdir(outdir)) != 0:
                inp = None
                while inp not in ['y', 'yes', 'n', 'no']:
                    inp = input(f'Found existing data in output directory: `{outdir}`. DELETE existing data? (y/n)')

                if inp in ['y', 'yes']:
                    print('Erasing existing data...')
                    shutil.rmtree(outdir)
                    os.mkdir(outdir)
                else:
                    sys.exit('Not removing data. Please run again with a new output directory.')
        else:
            os.mkdir(outdir)

        self.outdir = outdir
        self.n_events = n_events
        self.f_agn = f_agn
        self.n_agn_events = round(self.f_agn * self.n_events)
        self.n_alt_events = self.n_events - self.n_agn_events
        self.zdraw = zdraw  # Maximum redshift to generate ALT GWs from, currently using doing this uniform in comoving volume
        self.n_posterior_samples = n_posterior_samples
        self.cosmo = cosmology
        self.ncpu = ncpu
        self.extrinsic_flag = use_extrinsic
        self.intrinsic_flag = use_intrinsic
        self.lumdist_relerr = lumdist_relerr
        self.truths = pd.DataFrame()

        if self.extrinsic_flag:
            assert isinstance(catalog, MockCatalog), 'Provided catalog is not MockCatalog instance.'

            if catalog is not None:
                print('Using provided MockCatalog object.')
                self.MockCatalog = catalog
            elif catalog_path is not None:
                print(f'Using pre-existing AGN catalog located at `{catalog_path}`.')
                self.MockCatalog = MockCatalog.from_file(catalog_path)

            cat = catalog.complete_catalog
            self.catalog = cat.loc[cat['redshift_true'] < self.zdraw].reset_index(drop=True)  # AGN from which to generate AGN GWs

            if (self.n_agn_events != 0) and (len(self.catalog) == 0) and (len(cat) != 0):
                sys.exit('\nTried to generate GWs from AGN, but only found AGN outside the GW box. Either provide more AGN or put f_agn = 0.\nExiting...')

            self.sky_area_low = sky_area_low
            self.sky_area_high = sky_area_high

            print('Generating true GW sky positions...')
            self.make_true_extrinsic_parameter_values()  # Adds true values to self.truths
            print('Done.')
        else:
            print('\nNot generating extrinsic parameters.')

        if self.intrinsic_flag:
            assert model_dict is not None and hyperparam_dict_agn_path is not None and hyperparam_dict_alt_path is not None, 'Provide all three of `model_dict`, `hyperparam_dict_agn_path`, `hyperparam_dict_alt_path`.'
            self.model_dict = model_dict
            with open(hyperparam_dict_agn_path, "r") as file:
                self.hyperparam_dict_agn = json.load(file)
            with open(hyperparam_dict_alt_path, "r") as file:
                self.hyperparam_dict_alt = json.load(file)

            self.agn_models, self.alt_models = self._get_models()
            self.mass_relerr = mass_relerr

            print('\nGenerating true GW intrinsic parameter values...')
            self.make_true_intrinsic_parameter_values()  # Adds true values to self.truths
            print('Done.')
        else:
            print('\nNot generating intrinsic parameters.')

        print(f'Confirming: {len(self.truths)} GWs have been sampled')

        print('Writing truths to hdf5...')
        self.write_truth_to_hdf5()
    

    def make_posteriors(self):
        ''' Measures event parameters from their true values. True values stay the same for subsequent calls. '''

        if self.extrinsic_flag:
            print('\nGenerating extrinsic parameter posteriors...')
            self._measure_extrinsic_parameters()
            print('Done.')

        if self.intrinsic_flag:
            print('\nGenerating intrinsic parameter posteriors...')
            self._measure_intrinsic_parameters()
            print('Done.')
    

    def make_true_intrinsic_parameter_values(self):  # From provided population priors

        '''
        Method for adding the true values of intrinsic event parameters to self.truths.

        We assume the injected events be independent of location. So a from-AGN GW can have, e.g., any mass from the population prior, 
        independent of distance or AGN luminosity.
        '''

        # If 3D pos have been made, there exists a column ['from_agn'], which needs to be paired with the proper intrinsic events, 
        # and ['redshift'], which is needed for the measurement errors. TODO: rethink this, because it could cause biases...
        if not self.extrinsic_flag:
            sys.exit('TODO - Implement stuff here')
            # arr = np.zeros(self.n_agn_events + self.n_alt_events, dtype=bool)
            # arr[:self.n_agn_events] = True
            # self.truths['from_agn'] = arr

            # print('No sky locations used. Sampling GW redshifts according to uniform-in-comoving-volume distibution.')
            # r, _, _ = uniform_shell_sampler(0, self.cosmo.comoving_distance(self.zdraw).value, n_samps=self.n_events)
            # self.truths['rcom'] = r
            # self.truths['redshift'] = fast_z_at_value(self.cosmo.comoving_distance, r * u.Mpc)
        
        for Model in self.agn_models:
            self._update_hyperparams(Model, self.hyperparam_dict_agn)
            true_values_all_params = Model.sample(self.n_agn_events)  # Returns the true value of len(Model.param_label) event parameters

            for j, true_values_single_param in enumerate(true_values_all_params):
                key = Model.param_label[j]
                self.truths.loc[self.truths['from_agn'], key] = true_values_single_param

        for Model in self.alt_models:
            self._update_hyperparams(Model, self.hyperparam_dict_alt)
            true_values_all_params = Model.sample(self.n_alt_events)  # Returns the true value of len(Model.param_label) event parameters

            for j, true_values_single_param in enumerate(true_values_all_params):
                key = Model.param_label[j]
                self.truths.loc[~self.truths['from_agn'], key] = true_values_single_param


    def make_true_extrinsic_parameter_values(self) -> None:
        # AGN origin - hosts sampled from AGN catalog
        host_idx = self._select_agn_hosts()
        self.truths['ra'] = self.catalog.loc[host_idx, 'ra_true']
        self.truths['dec'] = self.catalog.loc[host_idx, 'dec_true']
        self.truths['comoving_distance'] = self.catalog.loc[host_idx, 'comoving_distance_true']
        self.truths['luminosity_distance'] = self.catalog.loc[host_idx, 'luminosity_distance_true']
        self.truths['redshift'] = self.catalog.loc[host_idx, 'redshift_true']
        self.truths['from_agn'] = np.ones(self.n_agn_events, dtype=bool)
        self.truths = self.truths.reset_index(drop=True)  # Remove random ordering of indeces caused by sampling of the catalog

        if self.n_alt_events != 0:
            # ALT origin - hosts uniform in comoving volume
            temp_alt_df = pd.DataFrame()
            r, theta, phi = self._sample_alt_coords()
            z = fast_z_at_value(self.cosmo.comoving_distance, r * u.Mpc)
            temp_alt_df['ra'] = phi
            temp_alt_df['dec'] = 0.5*np.pi - theta
            temp_alt_df['comoving_distance'] = r
            temp_alt_df['luminosity_distance'] = self.cosmo.luminosity_distance(z).value
            temp_alt_df['redshift'] = z
            temp_alt_df['from_agn'] = np.zeros(self.n_alt_events, dtype=bool)
            self.truths = pd.concat([self.truths, temp_alt_df], ignore_index=True)  # TODO: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
            del temp_alt_df


    def write_truth_to_hdf5(self):
        for index in tqdm(range(self.n_events)):
            try:
                filename = os.path.join(self.outdir, f"gw_{index:05d}.h5")
                with h5py.File(filename, "a") as f:
                    mock_group = f.require_group("mock")  # Ensure 'mock' group exists
                    truth_group = mock_group.require_group("truths")  # Ensure 'truths' exists
                    for colname, values in self.truths.items():
                        truth_group.create_dataset(colname, data=values.iloc[index], dtype="f8")

            except Exception as e:
                sys.exit(f"Error in event {index}: {e}")


    def _measure_extrinsic_parameters(self) -> None:
        ''' Sky position posteriors are modeled as a 2D VonMises-Fisher distribution '''

        # Get cartesian components of directional unit vector to GW origin
        x_true, y_true, z_true = spherical2cartesian(1, 0.5 * np.pi - self.truths['dec'], self.truths['ra'])
        
        # Compute VonMises-Fisher concentration parameters
        sky_areas_68p = self._sample_68p_sky_area()
        kappas = self._kappa_from_sky_area(sky_areas_68p)
        
        # dtrue = self.truths['luminosity_distance'].to_numpy()
        # dobs = dtrue * (1. + self.lumdist_relerr * np.random.normal(size=self.n_events))  # Observed distances

        print('Analyzing comoving distance for a test')
        dtrue = self.truths['comoving_distance'].to_numpy()
        dobs = np.random.normal(loc=dtrue, scale=np.tile(RCOM_SCALE, self.n_events))  # Fixed error at 10 Mpc


        with ThreadPoolExecutor(max_workers=min(cpu_count(), self.ncpu)) as executor:
            future_to_index = {executor.submit(
                                            self._process_single_event_extrinsic,
                                            i, 
                                            x_true[i], 
                                            y_true[i], 
                                            z_true[i], 
                                            kappas[i], 
                                            dobs[i]
                                        ): i for i in range(self.n_events)
                                        }
            
            for future in tqdm(as_completed(future_to_index), total=self.n_events):
                try:
                    _ = future.result(timeout=60)
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error processing event {future_to_index[future]}: {e}")
                    print('Only implemented the nuclear option: removing file - WATCH SELECTION EFFECTS IF THIS HAPPENS TOO OFTEN')
                    os.remove(os.path.join(self.outdir, f"gw_{future_to_index[future]:05d}.h5"))


    def _process_single_event_extrinsic(self, index, x, y, z, kappa, dobs) -> None:
        """Worker function for threading. Generates posterior samples and writes to HDF5."""

        ###### Sky locations ######
        mu_true = np.array([x, y, z])  # True center
        skymap_center = vonmises_fisher.rvs(mu_true, kappa, size=1)  # Observed center

        # Generate posterior samples
        # skymap_samps = vonmises_fisher.rvs(skymap_center[0], kappa, size=int(1e7))
        # _, theta_skymap, phi_skymap = cartesian2spherical(skymap_samps[:, 0], skymap_samps[:, 1], skymap_samps[:, 2])
        # pix_indices = hp.ang2pix(32, theta_skymap, phi_skymap, nest=True)
        # skymap = np.zeros(hp.nside2npix(32))
        # np.add.at(skymap, pix_indices, 1) 
        # skymap /= 1e7

        samples = vonmises_fisher.rvs(skymap_center[0], kappa, size=self.n_posterior_samples)
        _, theta_samples, phi_samples = cartesian2spherical(samples[:, 0], samples[:, 1], samples[:, 2])

        dec_samples = 0.5 * np.pi - theta_samples


        print('Doing abs error comdist test')
        dtrue_postsamps = np.random.normal(loc=dobs, scale=RCOM_SCALE, size=self.n_posterior_samples)
        lumdist_samples = dtrue_postsamps
        redshift_samples = dtrue_postsamps
        comdist_samples = dtrue_postsamps

        # ##### Distances #####
        # # Importance resampling of distances
        # dtrue_postsamps = dobs / (1 + self.lumdist_relerr * np.random.normal(size=4 * self.n_posterior_samples))
        # neg = dtrue_postsamps < 0
        # if np.sum(neg) != 0:
        #     print(f'Removing {np.sum(neg)} negative luminosity distance samples.')
        # dtrue_postsamps = dtrue_postsamps[~neg]  # WARNING: Negative values are very rare, (currently tested with 20% error, get only 1 negative sample sometimes), so just remove them. But be aware!
        # # print('using different weights for a test')
        # weights = dtrue_postsamps**1 / np.sum(dtrue_postsamps**1)  # Importance weights proportional to d

        # # print('Omit Jacobian for test')
        # lumdist_samples = np.random.choice(dtrue_postsamps, size=2 * self.n_posterior_samples, p=weights)
        
        # # above_thresh = (lumdist_samples > 13975914.444369921)
        # # below_thresh = (lumdist_samples < 4.415205612145357e-05)
        # # n_above_thresh = np.sum(above_thresh)
        # # n_below_thresh = np.sum(below_thresh)
        # # if n_above_thresh != 0:
        # #     print(f'Removing {n_above_thresh} lumdist samples above bracket, namely: {lumdist_samples[above_thresh]}')
        # # if n_below_thresh != 0:
        # #     print(f'Removing {n_below_thresh} lumdist samples below bracket, namely: {lumdist_samples[below_thresh]}')
        # # lumdist_samples = lumdist_samples[~above_thresh & ~below_thresh]

        # # Redshift and comoving distance calculations
        # zsamp = fast_z_at_value(self.cosmo.luminosity_distance, lumdist_samples * u.Mpc)
        # H_z = self.cosmo.H(zsamp).value  # H(z) in km/s/Mpc
        # chi_z = self.cosmo.comoving_distance(zsamp).value  # in Mpc
        # dDL_dz = chi_z + (1 + zsamp) * (3e5 / H_z)  # c = 3e5 km/s, TODO: change to astropy.const
        # z_weights = 1 / dDL_dz
        # z_weights /= np.sum(z_weights)
        # redshift_samples = np.random.choice(zsamp, self.n_posterior_samples, p=z_weights)
        # # redshift_samples = fast_z_at_value(self.cosmo.luminosity_distance, lumdist_samples * u.Mpc)
        # # redshift_samples = lumdist_samples
        # comdist_samples = self.cosmo.comoving_distance(redshift_samples).value  # TODO: check if this is correct

        # Write samples to hdf5
        samples_table = Table([phi_samples, dec_samples, lumdist_samples[:self.n_posterior_samples], comdist_samples, redshift_samples], 
                                    names=('ra', 'dec', 'luminosity_distance', 'comoving_distance', 'redshift'))
        filename = os.path.join(self.outdir, f"gw_{index:05d}.h5")

        with h5py.File(filename, "a") as f:
            mock_group = f.require_group("mock")  # Takes place of approximant in real GW data
            mock_group.create_dataset('posterior_samples', data=samples_table)
            # mock_group.create_dataset('skymap', data=skymap)

        return
    

    def _measure_intrinsic_parameters(self):
        '''
        Currently only implemented mass_1_source
        '''

        mtrue = self.truths['mass_1_source'].to_numpy()
        mobs = mtrue * (1. + self.mass_relerr * np.random.normal(size=self.n_events))  # Observed masses - TODO: not sure if this is correct

        with ThreadPoolExecutor(max_workers=min(cpu_count(), self.ncpu)) as executor:
            future_to_index = {executor.submit(
                                            self._process_single_event_intrinsic,
                                            i, 
                                            mobs[i]
                                        ): i for i in range(self.n_events)
                                        }
            
            for future in tqdm(as_completed(future_to_index), total=self.n_events):
                try:
                    _ = future.result(timeout=60)
                except Exception as e:
                    print(f"Error processing event {future_to_index[future]}: {e}")


    def _process_single_event_intrinsic(self, index, mobs):
        ##### Primary mass #####
        # Importance resampling of masses
        mass_1_postsamps = mobs / (1 + self.mass_relerr * np.random.normal(size=2 * self.n_posterior_samples))
        weights = mass_1_postsamps / np.sum(mass_1_postsamps)  # Importance weights proportional to m
        mass_1_samples = np.random.choice(mass_1_postsamps, size=self.n_posterior_samples, p=weights)

        # Write samples to hdf5
        intrinsic_samples = Table([mass_1_samples], names=('mass_1_source'))
        filename = os.path.join(self.outdir, f"gw_{index:05d}.h5")

        with h5py.File(filename, "a") as f:
            mock_group = f.require_group("mock")  # Takes place of approximant in real GW data

            if 'posterior_samples' in mock_group:
                posterior_table = Table.read(mock_group['posterior_samples'])
                posterior_table['mass_1_source'] = intrinsic_samples['mass_1_source']
                
                # Overwrite dataset with updated table
                del mock_group['posterior_samples']  # Remove old dataset
                mock_group.create_dataset('posterior_samples', data=posterior_table)
            else:
                mock_group.create_dataset('posterior_samples', data=intrinsic_samples)


    def _select_agn_hosts(self):
        host_idx = np.random.choice(np.arange(len(self.catalog)), self.n_agn_events)
        return host_idx
    

    def _sample_alt_coords(self):
        r, theta, phi = uniform_shell_sampler(0, self.cosmo.comoving_distance(self.zdraw).value, n_samps=self.n_alt_events)
        return r, theta, phi
    

    def _sample_68p_sky_area(self):
        ''' 68% CL sky localization area uniformly sampled between low and high in deg^2 '''
        sky_area = np.random.uniform(low=self.sky_area_low, high=self.sky_area_high, size=self.n_events) * (np.pi / 180)**2  # Deg^2 to sr
        return sky_area


    def _get_models(self):

        agn_params2use = []
        agn_models = []
        for model in self.model_dict['agn']:
            Model = self._get_model(model, hypothesis='AGN')
            agn_params2use += Model.param_label
            agn_models.append(Model)

        alt_params2use = []
        alt_models = []
        for model in self.model_dict['alt']:
            Model = self._get_model(model, hypothesis='ALT')
            alt_params2use += Model.param_label
            alt_models.append(Model)

        # Check if inputs are allowed
        assert check_equal(agn_params2use, alt_params2use), f'AGN and ALT models must compare the same event parameters. Got {agn_params2use} and {alt_params2use}'
        self.intrinsic_param_keys = agn_params2use
        print(f"\nGot the population priors on {agn_params2use}, modelled as `{self.model_dict['agn']}` for AGN and `{self.model_dict['alt']}` for ALT.")     
        return agn_models, alt_models


    @staticmethod
    def _get_model(model_str, hypothesis):
        # match model_str:
        #     case 'BBH-powerlaw':
        #         return BBH_powerlaw()
        #     case 'BBH-powerlaw-gaussian':
        #         return BBH_powerlaw_gaussian()
        #     case 'BBH-broken-powerlaw':
        #         return BBH_broken_powerlaw()
        #     case 'PrimaryMass-gaussian':
        #         return PrimaryMass_gaussian()
        #     case 'PrimaryMass-powerlaw-gaussian':
        #         return PrimaryMass_powerlaw_gaussian()
        #     case _:
        #         sys.exit(f'Unknown {hypothesis} model: {model_str}')

        if model_str == 'BBH-powerlaw':
            return BBH_powerlaw()
        elif model_str == 'BBH-powerlaw-gaussian':
            return BBH_powerlaw_gaussian()
        elif model_str == 'BBH-broken-powerlaw':
            return BBH_broken_powerlaw()
        elif model_str == 'PrimaryMass-gaussian':
            return PrimaryMass_gaussian()
        elif model_str == 'PrimaryMass-powerlaw-gaussian':
            return PrimaryMass_powerlaw_gaussian()
        else:
            sys.exit(f'Unknown {hypothesis} model: {model_str}')



    @staticmethod
    def _update_hyperparams(Model, hyperparam_dict):
        ''' Update the hyperparameters in a given event parameter model. 
        This is needed to independently update the different event parameter models for both hypotheses. 

        TODO: when real data is used, this function shouldn't be called and ['value'] can be a list encoding the bilby prior. Remember this such that this doesn't break.
        '''

        # Expand this dictionary if you add models with new hyperparameters to priors.py
        hyperparams = {
                    'alpha':hyperparam_dict['alpha']['value'], 'delta_m':hyperparam_dict['delta_m']['value'], 
                    'mu_g':hyperparam_dict['mu_g']['value'], 'sigma_g':hyperparam_dict['sigma_g']['value'], 
                    'lambda_peak':hyperparam_dict['lambda_peak']['value'], 'alpha_1':hyperparam_dict['alpha_1']['value'], 
                    'alpha_2':hyperparam_dict['alpha_2']['value'], 'b':hyperparam_dict['b']['value'], 
                    'mminbh':hyperparam_dict['mminbh']['value'], 'mmaxbh':hyperparam_dict['mmaxbh']['value'], 
                    'beta':hyperparam_dict['beta']['value'], 'alphans':hyperparam_dict['alphans']['value'],
                    'mminns':hyperparam_dict['mminns']['value'], 'mmaxns':hyperparam_dict['mmaxns']['value']
                }
        Model.update_parameters(hyperparams)


    @staticmethod
    def _kappa_from_sky_area(sky_area):
        ''' Factor 1.14 empirically seems to work for 68% CL - only approximate needed anyway '''
        kappa = 2 * np.pi * 1.14 / sky_area
        return kappa


if __name__ == '__main__':
    pass
