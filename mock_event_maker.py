import numpy as np
import pandas as pd
import sys, os
import astropy.units as u
from utils import fast_z_at_value, make_nice_plots, check_equal, uniform_shell_sampler, spherical2cartesian, cartesian2spherical
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


class MockEvent:

    def __init__(
                self,
                n_events: int,
                f_agn: float,
                zdraw: float,
                use_extrinsic: bool = True,
                use_intrinsic: bool = True,
                sky_area_low: float=DEFAULT_SKY_AREA_LOW,
                sky_area_high: float=DEFAULT_SKY_AREA_HIGH,
                lumdist_relerr: float=DEFAULT_LUMDIST_RELERR,
                catalog: MockCatalog = None,    # Either a MockCatalog object you just made
                catalog_path: str = None,       # Or a pre-existing one that is stored in hdf5 format
                model_dict: dict = None,
                hyperparam_dict_agn: dict = None,
                hyperparam_dict_alt: dict = None,
                n_posterior_samples: int = DEFAULT_N_POSTERIOR_SAMPLES,
                cosmology = DEFAULT_COSMOLOGY,
                outdir: str=DEFAULT_POSTERIOR_OUTDIR,
                ncpu: int=DEFAULT_N_CPU
            ):
        
        """
        OUTDATED DOCSTRING
        
        If you don't want skymaps, call MockEvent.without_skymaps()
        If you want skymaps while f_agn = 0, call MockEvent.with_skymaps_without_agn()
        In both these cases, provide zdraw: the maximum redshift to generate GWs from

        model_dict: dict            Contains all population prior models as strings in the format {`agn`: [`agn_model1`, ..., `agn_modelN`], `alt`: [...]},
        catalog: MockCatalog        
        """
        
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
            assert model_dict is not None and hyperparam_dict_agn is not None and hyperparam_dict_alt is not None, 'Provide all three of `model_dict`, `param_dict_agn`, `param_dict_alt`.'
            self.model_dict = model_dict
            self.hyperparam_dict_agn = hyperparam_dict_agn
            self.hyperparam_dict_alt = hyperparam_dict_alt

            self.agn_models, self.alt_models = self._get_models()

            print('\nGenerating true GW intrinsic parameter values...')
            self.make_true_intrinsic_parameter_values()  # Adds true values to self.truths
            print('Done.')
        else:
            print('\nNot generating extrinsic parameters.')

        print(f'Confirming: {len(self.truths)} GWs have been sampled')

        print('Writing truths to hdf5...')
        self.write_truth_to_hdf5()


    # TODO: rework for new mock catalog class
    # @classmethod
    # def without_skymaps(
    #                 cls,
    #                 n_events: int,
    #                 f_agn: float,
    #                 zdraw: float,
    #                 model_dict: dict,
    #                 hyperparam_dict_agn: dict,
    #                 hyperparam_dict_alt: dict,
    #                 n_posterior_samples: int = DEFAULT_N_POSTERIOR_SAMPLES,
    #                 cosmology = DEFAULT_COSMOLOGY,
    #                 outdir: str=DEFAULT_POSTERIOR_OUTDIR,
    #                 ncpu: int=DEFAULT_N_CPU
    #             ):
        
    #     ''' Make MockEvent instances with this method if you do not want any skymaps. '''

    #     # TODO: make `low` and `high` arguments in class methods and skymap/events classes

    #     # Empty catalog instance, since the max GW distance (zdraw) is still necessary: the measurement error depends on this
    #     Catalog = MockCatalog(n_agn=0, max_redshift=zdraw, gw_box_radius=zdraw)

    #     obj = cls(
    #             n_events=n_events,
    #             f_agn=f_agn,
    #             catalog=Catalog,
    #             catalog_path=None,
    #             use_skymaps=False,
    #             model_dict=model_dict,
    #             hyperparam_dict_agn=hyperparam_dict_agn,
    #             hyperparam_dict_alt=hyperparam_dict_alt,
    #             n_posterior_samples=n_posterior_samples,
    #             cosmology=cosmology,
    #             outdir=outdir,
    #             ncpu=ncpu
    #         )
    #     return obj
    

    # @classmethod
    # def with_skymaps_without_agn(
    #                             cls,
    #                             n_events: int,
    #                             zdraw: float,
    #                             model_dict: dict = None,
    #                             hyperparam_dict_agn: dict = None,
    #                             hyperparam_dict_alt: dict = None,
    #                             n_posterior_samples: int = DEFAULT_N_POSTERIOR_SAMPLES,
    #                             cosmology = DEFAULT_COSMOLOGY,
    #                             outdir: str=DEFAULT_POSTERIOR_OUTDIR,
    #                             ncpu: int=DEFAULT_N_CPU
    #                         ):
        
    #     ''' Let's you generate f_agn = 0 dataset without specifying an AGN catalog in the case you still want to use skymaps. '''

    #     # TODO: make `low` and `high` arguments in class methods and skymap/events classes
        
    #     # Empty catalog instance, since the max GW distance (gw_box_radius) is still necessary: the measurement error depends on this
    #     Catalog = MockCatalog(n_agn=0, max_redshift=zdraw, gw_box_radius=zdraw)

    #     obj = cls(
    #             n_events=n_events,
    #             f_agn=0.,
    #             catalog=Catalog,
    #             catalog_path=None,
    #             use_skymaps=True,
    #             model_dict=model_dict,
    #             hyperparam_dict_agn=hyperparam_dict_agn,
    #             hyperparam_dict_alt=hyperparam_dict_alt,
    #             n_posterior_samples=n_posterior_samples,
    #             cosmology=cosmology,
    #             outdir=outdir,
    #             ncpu=ncpu
    #         )
    #     return obj
    

    def make_posteriors(self):
        ''' Measures event parameters from their true values. True values stay the same for subsequent calls. '''

        if self.extrinsic_flag:
            print('\nGenerating extrinsic parameter posteriors...')
            self._measure_extrinsic_parameters()
            print('Done.')

        if self.model_dict is not None:
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
            sys.exit('TODO Implement stuff here')
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

        dtrue = self.truths['luminosity_distance'].to_numpy()
        dobs = dtrue * (1. + self.lumdist_relerr * np.random.normal(size=self.n_events))  # Observed distances


        with ThreadPoolExecutor(max_workers=min(cpu_count(), self.ncpu)) as executor:
            future_to_index = {executor.submit(
                                            self._process_single_event_extrinsic,
                                            i, 
                                            x_true[i], 
                                            y_true[i], 
                                            z_true[i], 
                                            kappas[i], 
                                            dobs[i], 
                                            self.lumdist_relerr
                                        ): i for i in range(self.n_events)
                                        }
            
            for future in tqdm(as_completed(future_to_index), total=self.n_events):
                try:
                    _ = future.result(timeout=60)
                except Exception as e:
                    print(f"Error processing event {future_to_index[future]}: {e}")


    def _process_single_event_extrinsic(self, index, x, y, z, kappa, dobs, sigma) -> None:
        """Worker function for threading. Generates posterior samples and writes to HDF5."""

        ###### Sky locations ######
        mu_true = np.array([x, y, z])  # True center
        skymap_center = vonmises_fisher.rvs(mu_true, kappa, size=1)  # Observed center

        # Generate posterior samples
        samples = vonmises_fisher.rvs(skymap_center[0], kappa, size=self.n_posterior_samples)
        _, theta_samples, phi_samples = cartesian2spherical(samples[:, 0], samples[:, 1], samples[:, 2])

        dec_samples = 0.5 * np.pi - theta_samples

        ##### Distances #####
        # Importance resampling of distances
        dtpostsamps = dobs / (1 + sigma * np.random.normal(size=2 * self.n_posterior_samples))
        weights = dtpostsamps / np.sum(dtpostsamps)  # Importance weights proportional to d
        lumdist_samples = np.random.choice(dtpostsamps, size=self.n_posterior_samples, p=weights)

        # Redshift and comoving distance calculations
        redshift_samples = fast_z_at_value(self.cosmo.luminosity_distance, lumdist_samples * u.Mpc)
        comdist_samples = self.cosmo.comoving_distance(redshift_samples).value

        # Write samples to hdf5
        samples_table = Table([phi_samples, dec_samples, lumdist_samples, comdist_samples, redshift_samples], 
                                    names=('ra', 'dec', 'luminosity_distance', 'comoving_distance', 'redshift'))
        filename = os.path.join(self.outdir, f"gw_{index:05d}.h5")

        with h5py.File(filename, "a") as f:
            mock_group = f.require_group("mock")  # Takes place of approximant in real GW data
            mock_group.create_dataset('posterior_samples', data=samples_table)

        return None


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
    










    

    def _measure_intrinsic_parameters(self):
        '''
        TODO: rethink this, because it could cause biases...

        We assume our measurements to be Gaussian. The width should be dependent on redshift. 
        Either sample and reorder errors (done for skymaps) or use closed formula (currently done here for intrinsics).
        
        TODO: make truncnorm posteriors and make the measurement_errors dict an argument of the class -> [min_value, max_value, min_error, max_error]
        OR DO WE ALREADY HAVE MMIN AND MMAX FROM HYPERPARAM DICT? POG: mminbh, mmaxbh
        however, the true values of mminbh and mmaxbh may be unknown, so the posteriors would extend past these...bruhhh
        '''

        # param: [min_error, max_error], such that total error = min_error * (1 + z), capped at max_error
        measurement_errors={'mass_1_source': [1, 15],
                            'mass_2_source': [1, 20]}
        n_intrinsic_params = len(self.intrinsic_param_keys)
        
        # Unpack measurement_errors dictionary
        min_errs, max_errs = np.zeros(n_intrinsic_params), np.zeros(n_intrinsic_params)
        for i, param in enumerate(self.intrinsic_param_keys):  # All intrinsic parameters previously found must have a value in measurement_errors dict
            min_errs[i], max_errs[i] = measurement_errors[param]

        errors = min_errs[np.newaxis, :] * (1 + self.truths['redshift'].to_numpy()[:, np.newaxis])
        errors = np.where(errors > max_errs, max_errs, errors)
        means = np.random.normal(loc=self.truths[self.intrinsic_param_keys].to_numpy(), scale=errors, size=(self.n_events, n_intrinsic_params))  # Posterior mean should not coincide with true value
        posteriors = np.random.normal(loc=means[:, :, np.newaxis], scale=errors[:, :, np.newaxis], size=(self.n_events, n_intrinsic_params, self.n_posterior_samples))

        # Pass posteriors to DataFrame
        for i, param in enumerate(self.intrinsic_param_keys):
            self.posteriors[param] = list(posteriors[:, i, :])


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
        match model_str:
            case 'BBH-powerlaw':
                return BBH_powerlaw()
            case 'BBH-powerlaw-gaussian':
                return BBH_powerlaw_gaussian()
            case 'BBH-broken-powerlaw':
                return BBH_broken_powerlaw()
            case 'PrimaryMass-gaussian':
                return PrimaryMass_gaussian()
            case 'PrimaryMass-powerlaw-gaussian':
                return PrimaryMass_powerlaw_gaussian()
            case _:
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

    make_nice_plots()

    # N_TOT = 100
    ZDRAW = 2
    N_EVENTS = int(100)
    F_AGN = 0.5

    ##### FAGN = 0 EVENTS WITH SKYMAPS #####
    # events = MockEvent.with_skymaps_without_agn(n_events=N_EVENTS,
    #                                             zdraw=GW_BOX_SIZE)
    
    # events.make_posteriors()
    # print(events.posteriors)

    ########################################  

    # The model dictionary should contain choices of implemented models. These models have parameters for which we pass another dictionary
    # TODO: think about what happens when we want to do the chi_eff vs q correlation here (mass ratios are used in the mass models already)
    model_dict = {
                'agn': ['PrimaryMass-gaussian'], 
                'alt': ['PrimaryMass-gaussian']
            }
    
    # These dicts should contain a selection of allowed population parameters: only parameters that are implemented in models are allowed
    try:
        with open("hyperparam_agn.json", "r") as file:
            hyperparam_dict_agn = json.load(file)
        with open("hyperparam_alt.json", "r") as file:
            hyperparam_dict_alt = json.load(file)
    except FileNotFoundError:
        with open("/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/inputs/hyperparam_agn.json", "r") as file:
            hyperparam_dict_agn = json.load(file)
        with open("/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/inputs/hyperparam_alt.json", "r") as file:
            hyperparam_dict_alt = json.load(file)


    ##### EVENTS WITH EMPTY CATALOG #####
    # events = MockEvent.without_skymaps(n_events=N_EVENTS,
    #                                    f_agn=F_AGN,
    #                                    zdraw=GW_BOX_SIZE,
    #                                    model_dict=model_dict,
    #                                    hyperparam_dict_agn=hyperparam_dict_agn,
    #                                    hyperparam_dict_alt=hyperparam_dict_alt,
    #                                    )
    # events.make_posteriors()
    # print(events.posteriors)
    #####################################
    

    # Catalog = MockCatalog(n_agn=N_TOT,
    #                         max_redshift=GRID_SIZE,
    #                         gw_box_radius=GW_BOX_SIZE,
    #                         completeness=1)

    Catalog = MockCatalog.from_file(file_path='/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs/mockcat_NAGN750000_ZMAX_3.hdf5')
    GWEvents = MockEvent(
                    n_events=N_EVENTS,
                    f_agn=F_AGN,
                    zdraw=ZDRAW,
                    catalog=Catalog,
                    ncpu=1,
                    n_posterior_samples=int(5e3)
                )

    # GWEvents = MockEvent(
    #                     model_dict=model_dict,
    #                     n_events=N_EVENTS,
    #                     f_agn=F_AGN,
    #                     use_skymaps=True,
    #                     hyperparam_dict_agn=hyperparam_dict_agn,
    #                     hyperparam_dict_alt=hyperparam_dict_alt,
    #                     catalog=Catalog, 
    #                 )

    GWEvents.make_posteriors()
    # print(GWEvents.posteriors)

    # GWEvents.write_3D_location_samples_to_hdf5(outdir='./output/mocksamples/without_lumdist')

    ############# PLOTTING POSITION POSTERIORS #############

    # for i in range(10):
    #     ra_post = np.array(GWEvents.posteriors['ra'].iloc[i])
    #     dec_post = np.array(GWEvents.posteriors['dec'].iloc[i])
    #     z_post = np.array(GWEvents.posteriors['redshift'].iloc[i])

    #     ra_truth = GWEvents.truths['ra'].iloc[i]
    #     dec_truth = GWEvents.truths['dec'].iloc[i]
    #     z_truth = GWEvents.truths['redshift'].iloc[i]

    #     fig, ax = plt.subplots(ncols=3, figsize=(24, 6))
    #     ax[0].scatter(ra_post, dec_post, color='black', marker='.', alpha=0.4)
    #     ax[0].scatter(ra_truth, dec_truth, color='red', marker='o', zorder=5)
    #     ax[0].set_xlabel('RA')
    #     ax[0].set_ylabel('Dec')

    #     ax[1].scatter(ra_post, z_post, color='black', marker='.', alpha=0.4)
    #     ax[1].scatter(ra_truth, z_truth, color='red', marker='o', zorder=5)
    #     ax[1].set_xlabel('RA')
    #     ax[1].set_ylabel('z')

    #     ax[2].scatter(z_post, dec_post, color='black', marker='.', alpha=0.4)
    #     ax[2].scatter(z_truth, dec_truth, color='red', marker='o', zorder=5)
    #     ax[2].set_xlabel('z')
    #     ax[2].set_ylabel('Dec')
    #     plt.show()

    ############# INTINSIC PARAMETERS #############

    agn_colors = np.array(['red', 'blue', 'black'])
    alt_colors = np.array(['coral', 'skyblue', 'grey'])

    agn_mass1 = GWEvents.truths['mass_1_source'].loc[GWEvents.truths['from_agn']]
    alt_mass1 = GWEvents.truths['mass_1_source'].loc[~GWEvents.truths['from_agn']]
    # agn_mass2 = GWEvents.truths['mass_2_source'].loc[GWEvents.truths['from_agn']]
    # alt_mass2 = GWEvents.truths['mass_2_source'].loc[~GWEvents.truths['from_agn']]

    agn_mass1_post = GWEvents.posteriors['mass_1_source'].loc[GWEvents.truths['from_agn']]
    alt_mass1_post = GWEvents.posteriors['mass_1_source'].loc[~GWEvents.truths['from_agn']]
    # agn_mass2_post = GWEvents.posteriors['mass_2_source'].loc[GWEvents.truths['from_agn']]
    # alt_mass2_post = GWEvents.posteriors['mass_2_source'].loc[~GWEvents.truths['from_agn']]

    ### PLOTTING POPULATION PRIORS ###

    # plt.figure(figsize=(8,6))
    # plt.hist(agn_mass1, density=True, bins=30, histtype='step', color='blue', linewidth=3, label='AGN m1')
    # plt.hist(alt_mass1, density=True, bins=30, histtype='step', color='red', linewidth=3, label='ALT m1')
    # plt.hist(agn_mass2, density=True, bins=30, histtype='step', color='skyblue', linewidth=3, label='AGN m2')
    # plt.hist(alt_mass2, density=True, bins=30, histtype='step', color='coral', linewidth=3, label='ALT m2')
    # plt.legend()
    # plt.show()

    ### PLOTTING INTRINSIC PARAM POSTERIORS ###

    # plt.figure(figsize=(8,6))
    # h, _, _ = plt.hist(agn_mass1_post, density=True, bins=30, histtype='step', color=agn_colors[:GWEvents.n_agn_events], linewidth=3, label='Measured AGN')
    # h2, _, _, = plt.hist(alt_mass1_post, density=True, bins=30, histtype='step', color=alt_colors[:GWEvents.n_alt_events], linewidth=3, label='Measured ALT')
    
    # ymin = 0
    # ymax = 1.1 * max(max(h.flatten()), max(h2.flatten()))
    # plt.vlines(agn_mass1, ymin, ymax, color=agn_colors[:GWEvents.n_agn_events], linestyle='dashed', linewidth=3, label='True AGN')
    # plt.vlines(alt_mass1, ymin, ymax, color=alt_colors[:GWEvents.n_alt_events], linestyle='dashed', linewidth=3, label='True ALT')   
    # plt.xlabel('log Primary mass')
    # plt.ylabel('Probability density')
    # plt.ylim(ymin, ymax)
    # plt.legend()
    # plt.show()
    

#%%