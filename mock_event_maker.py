#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import sys
import astropy.units as u
from utils import fast_z_at_value, make_nice_plots, check_equal
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
        In both these cases: provide zmax: the maximum redshift to generate GWs from



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
        self.posteriors = None  # Call get_posteriors() method to make posteriors, call again to make new posteriors from the same true values in self.properties

        if self.skymaps_flag:
            self.make_skymaps()  # Adds true values to self.properties
            print(f'\nConfirm: {len(self.properties)} Skymaps have been generated.')
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
            self.inject_events()  # Adds true values to self.properties

    
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


    @staticmethod
    def _update_hyperparams(Model, hyperparam_dict):
        ''' Update the hyperparameters in a given event parameter model. 
        This is needed to independently update the different event parameter models for both hypotheses. 

        TODO: when real data is used, this function shouldn't be called and ['value'] can be a list encoding the bilby prior.
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
    

    def inject_events(self):  # From provided population priors

        '''
        Method for adding the true values of intrinsic event parameters to self.properties.

        We assume the injected events be independent of location. So a from-AGN GW can have, e.g., any mass from the population prior, 
        independent of distance of AGN luminosity.
        '''

        # If skymaps have been made, there exists a column ['from_agn'], which needs to be paired with the proper intrinsic events.  
        if 'from_agn' not in self.properties.columns:
            arr = np.zeros(self.n_agn_events + self.n_alt_events, dtype=bool)
            arr[:self.n_agn_events] = True
            self.properties['from_agn'] = arr
        
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
        ''' We assume our measurements to be Gaussian. The width should be dependent on redshift. '''

        # TODO: Should I make a different function for each parameter? Or how can I otherwise generalize the measurement error?
        return
        

    def get_posteriors(self):
        ''' Measures quantities from true values. True values stay the same for subsequent calls. '''

        if self.skymaps_flag:
            print('\nGenerating sky localization...')
            self.get_skymap_posteriors()

        if self.model_dict is not None:
            print('\nGenerating intrinsic parameter posteriors...')
            self._measure_events()

        return self.posteriors


if __name__ == '__main__':

    make_nice_plots()

    N_TOT = 1
    GRID_SIZE = 40  # Radius of the whole grid in redshift
    GW_BOX_SIZE = 30  # Radius of the GW box in redshift
    N_EVENTS = 10000
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
                'agn': ['BBH-powerlaw-gaussian'], 
                'alt': ['BBH-powerlaw-gaussian']
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
                        hyperparam_dict_agn=hyperparam_dict_agn,
                        hyperparam_dict_alt=hyperparam_dict_alt,
                        catalog=Catalog, 
                        skymap_cl=CL
                    )

    GWEvents.get_posteriors()
    # print(GWEvents.posteriors)

    agn_mass1 = GWEvents.properties['mass_1'].loc[GWEvents.properties['from_agn']]
    alt_mass1 = GWEvents.properties['mass_1'].loc[~GWEvents.properties['from_agn']]
    agn_mass2 = GWEvents.properties['mass_2'].loc[GWEvents.properties['from_agn']]
    alt_mass2 = GWEvents.properties['mass_2'].loc[~GWEvents.properties['from_agn']]

    plt.figure(figsize=(8,6))
    plt.hist(agn_mass1, density=True, bins=30, histtype='step', color='blue', linewidth=3, label='AGN m1')
    plt.hist(alt_mass1, density=True, bins=30, histtype='step', color='red', linewidth=3, label='ALT m1')
    plt.hist(agn_mass2, density=True, bins=30, histtype='step', color='skyblue', linewidth=3, label='AGN m2')
    plt.hist(alt_mass2, density=True, bins=30, histtype='step', color='coral', linewidth=3, label='ALT m2')
    plt.legend()
    plt.show()
    

#%%