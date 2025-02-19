#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import sys
import astropy.units as u
from utils import fast_z_at_value, make_nice_plots, check_equal, uniform_shell_sampler
from mock_catalog_maker import MockCatalog
from mock_skymap_maker import MockSkymap
from priors import *
import json


class MockEvent(MockSkymap):

    def __init__(
                self,
                n_events: int,
                f_agn: float,
                catalog: MockCatalog,
                use_skymaps: bool,
                model_dict: dict = None,
                hyperparam_dict_agn: dict = None,
                hyperparam_dict_alt: dict = None,
                skymap_cl: float = 1.,
                n_posterior_samples: int = 1000,
                cosmology = FlatLambdaCDM(H0=67.9, Om0=0.3065)
            ):
        
        """
        If you don't want skymaps, call MockEvent.without_skymaps()
        If you want skymaps while f_agn = 0, call MockEvent.with_skymaps_without_agn()
        In both these cases, provide zmax: the maximum redshift to generate GWs from

        model_dict: dict            Contains all population prior models as strings in the format {`agn`: [`agn_model1`, ..., `agn_modelN`], `alt`: [...]},
        catalog: MockCatalog        
        """
        
        # Inherit all properties and methods from MockSkymap
        super().__init__(
                        n_events,
                        f_agn,
                        catalog,
                        skymap_cl,
                        n_posterior_samples,
                        cosmology
                    )
        

        self.skymaps_flag = use_skymaps
        self.properties = pd.DataFrame()
        self.posteriors = pd.DataFrame()  # Call get_posteriors() method to make posteriors, call again to make new posteriors from the same true values in self.properties

        if self.skymaps_flag:
            print('Generating true GW sky positions...')
            self.make_skymaps()  # Adds true values to self.properties
            print(f'Confirming: {len(self.properties)} Skymaps have been generated with CL = {self.skymap_cl}')
        else:
            print('\nNot generating skymaps.')

        self.model_dict = model_dict
        self.hyperparam_dict_agn = hyperparam_dict_agn
        self.hyperparam_dict_alt = hyperparam_dict_alt
        if self.model_dict is not None or self.hyperparam_dict_agn is not None or self.hyperparam_dict_alt is not None:
            assert self.model_dict is not None and self.hyperparam_dict_agn is not None and self.hyperparam_dict_alt is not None, 'Either provide all three of `model_dict`, `param_dict_agn`, `param_dict_alt` or none.'
            self.agn_models, self.alt_models = self._get_models()
        else:
            self.agn_models, self.alt_models = None, None

        if self.model_dict is not None:
            print('\nGenerating true GW intrinsic parameter values...')
            self.inject_events()  # Adds true values to self.properties
            print('Done.')

    
    @classmethod
    def without_skymaps(
                    cls,
                    n_events: int,
                    f_agn: float,
                    zmax: float,
                    model_dict: dict,
                    hyperparam_dict_agn: dict,
                    hyperparam_dict_alt: dict,
                    n_posterior_samples: int = 1000,
                    cosmology = FlatLambdaCDM(H0=67.9, Om0=0.3065)
                ):
        
        ''' Make MockEvent instances with this method if you do not want any skymaps. '''

        # Empty catalog instance, since the max GW distance (gw_box_radius) is still necessary since the measurement error depends on this
        Catalog = MockCatalog(n_agn=0, max_redshift=zmax, gw_box_radius=zmax)

        obj = cls(
                n_events=n_events,
                f_agn=f_agn,
                catalog=Catalog,
                use_skymaps=False,
                model_dict=model_dict,
                hyperparam_dict_agn=hyperparam_dict_agn,
                hyperparam_dict_alt=hyperparam_dict_alt,
                skymap_cl=1.,
                n_posterior_samples=n_posterior_samples,
                cosmology=cosmology
            )
        return obj
    

    @classmethod
    def with_skymaps_without_agn(
                                cls,
                                n_events: int,
                                zmax: float,
                                skymap_cl: float,
                                model_dict: dict = None,
                                hyperparam_dict_agn: dict = None,
                                hyperparam_dict_alt: dict = None,
                                n_posterior_samples: int = 1000,
                                cosmology = FlatLambdaCDM(H0=67.9, Om0=0.3065)
                            ):
        
        ''' Let's you generate f_agn = 0 dataset without specifying an AGN catalog in the case you still want to use skymaps. '''
        
        # Empty catalog instance, since the max GW distance (gw_box_radius) is still necessary since the measurement error depends on this
        Catalog = MockCatalog(n_agn=0, max_redshift=zmax, gw_box_radius=zmax)

        obj = cls(
                n_events=n_events,
                f_agn=0.,
                catalog=Catalog,
                use_skymaps=True,
                model_dict=model_dict,
                hyperparam_dict_agn=hyperparam_dict_agn,
                hyperparam_dict_alt=hyperparam_dict_alt,
                skymap_cl=skymap_cl,
                n_posterior_samples=n_posterior_samples,
                cosmology=cosmology
            )
        return obj
    

    def get_posteriors(self):
        ''' Measures quantities from true values. True values stay the same for subsequent calls. '''

        if self.skymaps_flag:
            print('\nGenerating sky localization...')
            self.get_skymap_posteriors()
            print('Done.')

        if self.model_dict is not None:
            print('\nGenerating intrinsic parameter posteriors...')
            self._measure_events()
            print('Done.')

        return self.posteriors
    

    def inject_events(self):  # From provided population priors

        '''
        Method for adding the true values of intrinsic event parameters to self.properties.

        We assume the injected events be independent of location. So a from-AGN GW can have, e.g., any mass from the population prior, 
        independent of distance of AGN luminosity.
        '''

        # If skymaps have been made, there exists a column ['from_agn'], which needs to be paired with the proper intrinsic events, 
        # and ['redshift'], which is needed for the measurement errors.
        if not self.skymaps_flag:
            arr = np.zeros(self.n_agn_events + self.n_alt_events, dtype=bool)
            arr[:self.n_agn_events] = True
            self.properties['from_agn'] = arr

            print('No sky locations used. Sampling GW redshifts according to uniform-in-comoving-volume distibution.')
            r, _, _ = uniform_shell_sampler(0, self.cosmo.comoving_distance(self.z_max).value, n_samps=self.n_events)
            self.properties['r'] = r
            self.properties['redshift'] = fast_z_at_value(self.cosmo.comoving_distance, r * u.Mpc)
        
        for Model in self.agn_models:
            self._update_hyperparams(Model, self.hyperparam_dict_agn)
            true_values_all_params = Model.sample(self.n_agn_events)  # Returns the true value of len(Model.param_label) event parameters

            for j, true_values_single_param in enumerate(true_values_all_params):
                key = Model.param_label[j]
                self.properties.loc[self.properties['from_agn'], key] = true_values_single_param

        for Model in self.alt_models:
            self._update_hyperparams(Model, self.hyperparam_dict_alt)
            true_values_all_params = Model.sample(self.n_alt_events)  # Returns the true value of len(Model.param_label) event parameters

            for j, true_values_single_param in enumerate(true_values_all_params):
                key = Model.param_label[j]
                self.properties.loc[~self.properties['from_agn'], key] = true_values_single_param
    

    def _measure_events(self):
        '''
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

        errors = min_errs[np.newaxis, :] * (1 + self.properties['redshift'].to_numpy()[:, np.newaxis])
        errors = np.where(errors > max_errs, max_errs, errors)
        means = np.random.normal(loc=self.properties[self.intrinsic_param_keys].to_numpy(), scale=errors, size=(self.n_events, n_intrinsic_params))  # Posterior mean should not coincide with true value
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


if __name__ == '__main__':

    make_nice_plots()

    N_TOT = 1000
    GRID_SIZE = 5  # Radius of the whole grid in redshift
    GW_BOX_SIZE = 2  # Radius of the GW box in redshift
    N_EVENTS = 6
    F_AGN = 0.5
    CL = 0.999

    ##### FAGN = 0 EVENTS WITH SKYMAPS #####
    # events = MockEvent.with_skymaps_without_agn(n_events=N_EVENTS,
    #                                             zmax=GW_BOX_SIZE,
    #                                             skymap_cl=CL)
    
    # events.get_posteriors()
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
        with open("/net/vdesk/data2/pouw/MRP/mockdata_analysis/mockgw/hyperparam_agn.json", "r") as file:
            hyperparam_dict_agn = json.load(file)
        with open("/net/vdesk/data2/pouw/MRP/mockdata_analysis/mockgw/hyperparam_alt.json", "r") as file:
            hyperparam_dict_alt = json.load(file)


    ##### EVENTS WITH EMPTY CATALOG #####
    # events = MockEvent.without_skymaps(n_events=N_EVENTS,
    #                                    f_agn=F_AGN,
    #                                    zmax=GW_BOX_SIZE,
    #                                    model_dict=model_dict,
    #                                    hyperparam_dict_agn=hyperparam_dict_agn,
    #                                    hyperparam_dict_alt=hyperparam_dict_alt,
    #                                    )
    # events.get_posteriors()
    # print(events.posteriors)
    #####################################
    

    Catalog = MockCatalog(n_agn=N_TOT,
                            max_redshift=GRID_SIZE,
                            gw_box_radius=GW_BOX_SIZE,
                            completeness=1)

    # GWEvents = MockEvent(
    #                     n_events=N_EVENTS,
    #                     f_agn=F_AGN,
    #                     catalog=Catalog, 
    #                     skymap_cl=CL
    #                 )

    GWEvents = MockEvent(
                        model_dict=model_dict,
                        n_events=N_EVENTS,
                        f_agn=F_AGN,
                        use_skymaps=True,
                        hyperparam_dict_agn=hyperparam_dict_agn,
                        hyperparam_dict_alt=hyperparam_dict_alt,
                        catalog=Catalog, 
                        skymap_cl=CL
                    )

    GWEvents.get_posteriors()
    # print(GWEvents.posteriors)

    ############# PLOTTING #############

    agn_colors = np.array(['red', 'blue', 'black'])
    alt_colors = np.array(['coral', 'skyblue', 'grey'])

    agn_mass1 = GWEvents.properties['mass_1_source'].loc[GWEvents.properties['from_agn']]
    alt_mass1 = GWEvents.properties['mass_1_source'].loc[~GWEvents.properties['from_agn']]
    # agn_mass2 = GWEvents.properties['mass_2_source'].loc[GWEvents.properties['from_agn']]
    # alt_mass2 = GWEvents.properties['mass_2_source'].loc[~GWEvents.properties['from_agn']]

    agn_mass1_post = GWEvents.posteriors['mass_1_source'].loc[GWEvents.properties['from_agn']]
    alt_mass1_post = GWEvents.posteriors['mass_1_source'].loc[~GWEvents.properties['from_agn']]
    # agn_mass2_post = GWEvents.posteriors['mass_2_source'].loc[GWEvents.properties['from_agn']]
    # alt_mass2_post = GWEvents.posteriors['mass_2_source'].loc[~GWEvents.properties['from_agn']]

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