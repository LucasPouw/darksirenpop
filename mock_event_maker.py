import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import sys
import astropy.units as u
from utils import fast_z_at_value, make_nice_plots
from mock_catalog_maker import MockCatalog
from mock_skymap_maker import MockSkymap
from priors import *


def _get_model(model_str, hypothesis):

        if model_str == 'BBH-powerlaw':
            model = BBH_powerlaw()
        elif model_str == 'BBH-powerlaw-gaussian':
            model = BBH_powerlaw_gaussian()
        elif model_str == 'BBH-broken-powerlaw':
            model = BBH_broken_powerlaw()
        elif model_str == 'PrimaryMass_gaussian':
            model = PrimaryMass_gaussian()
        else:
            sys.exit(f'Unrecognized {hypothesis} model: {model_str}')

        return model


class MockEvent(MockSkymap):

    def __init__(self,
                 n_events: int,
                 f_agn: float,
                 catalog: MockCatalog,
                 model_dict: dict = None,
                 param_dict_agn: dict = None,
                 param_dict_alt: dict = None,
                 skymap_cl: float = 1,
                 cosmology = FlatLambdaCDM(H0=67.9, Om0=0.3065)):
        
        """
        model_dict: dict            Contains all population prior models as strings in the format {`parameter`: [`agn_model`, `alt_model`]},
                                    The `parameter` key is purely for bookkeeping.
        catalog: MockCatalog        If you don't want skymaps, give an empty catalog (use n_agn = 0). If you want skymaps while f_agn = 0,
                                    still provide an AGN catalog with at least 1 AGN and the desired gw_box_radius!
        """
        
        super().__init__(n_events,
                         f_agn,
                         catalog,
                         skymap_cl,
                         cosmology)  # Inherit all properties and methods from MockSkymap
        
        assert param_dict_agn is not None or catalog is not None, 'Please provide either an AGN catalog, parameter dictionaries or both.'

        if len(self.properties) != 0:
            print(f'Confirm: {len(self.properties)} Skymaps have been generated.')
            self.skymaps_flag = True
        else:
            print('Confirm: No skymaps have been generated.')
            self.skymaps_flag = False

        self.models_agn = []
        self.models_alt = []
        if model_dict is not None or param_dict_agn is not None or param_dict_alt is not None:  # If one is provided, all should be provided.
            assert model_dict is not None and param_dict_agn is not None and param_dict_alt is not None, 'Either provide all three of `model_dict`, `param_dict_agn`, `param_dict_alt` or none.'
            self.model_dict = model_dict
            self.param_dict_agn = param_dict_agn
            self.param_dict_alt = param_dict_alt
            self._get_models()

        for i, model in enumerate(self.models_agn):
            pass # TODO: sample distributions -> generate posteriors

        
    def _get_models(self):

        for model_str in self.model_dict.keys():

            model_agn_str, model_alt_str = self.model_dict[model_str]
            model_agn = _get_model(model_agn_str, hypothesis='AGN')
            model_alt = _get_model(model_alt_str, hypothesis='ALT')

            agn_label, alt_label = model_agn.param_label, model_alt.param_label
            assert agn_label == alt_label, f'AGN and ALT models must compare the same parameters. Got {agn_label} and {alt_label}'

            print(f"\nGot the population prior on `{model_str}`, \
modelled as `{model_agn_str}` for AGN and `{model_alt_str}` for ALT. This gives a distribution on `{model_agn.param_label}`.")

            self.models_agn.append(model_agn)
            self.models_alt.append(model_alt)
                
        return
    

    def _inject_events(self):  # From population models
        return
    

    def _select_injections(self):
        return
    

    def _measure_events(self):
        return
        

    def get_posteriors(self):
        if self.skymaps_flag:
            self.get_skymap_posteriors()
        # Get other posteriors using these if necessary
        return


if __name__ == '__main__':

    make_nice_plots()

    N_TOT = 0
    GRID_SIZE = 40  # Radius of the whole grid in redshift
    GW_BOX_SIZE = 30  # Radius of the GW box in redshift
    
    Catalog = MockCatalog(n_agn=N_TOT,
                            max_redshift=GRID_SIZE,
                            gw_box_radius=GW_BOX_SIZE,
                            completeness=1)

    n_events = 100
    f_agn = 0.5
    cl = 0.999

    # The model dictionary should contain choices of implemented models. These models have parameters for which we pass another dictionary
    # TODO: think about what happens when we want to do the chi_eff vs q correlation here (mass ratios are used in the mass models already)
    model_dict = {'mass': ['PrimaryMass_gaussian', 'PrimaryMass_gaussian']}
    
    # These dicts should contain a selection of allowed population parameters: only parameters that are implemented in models are allowed
    param_dict_agn = {
            'mminbh': 6.,
            'mu_g': 15.,
            'sigma_g': 1.
        }
    
    param_dict_alt = {
            'mminbh': 6.,
            'mu_g': 50.,
            'sigma_g': 5.
        }

    
    GWEvents = MockEvent(
                        model_dict=model_dict,
                        n_events=n_events,
                        f_agn=f_agn,
                        param_dict_agn=param_dict_agn,
                        param_dict_alt=param_dict_alt,
                        catalog=Catalog, 
                        skymap_cl=cl
                    )

    GWEvents.get_posteriors()
