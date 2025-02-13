import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import sys
import astropy.units as u
from utils import uniform_shell_sampler, spherical2cartesian, cartesian2spherical, fast_z_at_value, make_nice_plots
from scipy.optimize import root_scalar
from scipy.special import erf
from mock_catalog_maker import MockCatalog
from mock_skymap_maker import MockSkymap


class MockEvents(MockSkymap):

    def __init__(self,
                 
                 something_else,

                 n_events: int = None,
                 f_agn: float = None,
                 catalog: MockCatalog = None,  
                 z_max: float = None,
                 skymap_cl: float = None,
                 cosmology = FlatLambdaCDM(H0=67.9, Om0=0.3065)):
        
        super().__init__(n_events,
                         f_agn,
                         catalog, 
                         z_max,
                         skymap_cl,
                         cosmology)  # Inherit all properties and methods from MockSkymap
        
        self.something_else = something_else
        

    def get_posteriors(self):
        self.get_skymap_posteriors()
        # Get other posteriors using these
        return


if __name__ == '__main__':

    make_nice_plots()

    N_TOT = 10000
    GRID_SIZE = 40  # Radius of the whole grid in redshift
    GW_BOX_SIZE = 30  # Radius of the GW box in redshift
    
    Catalog = MockCatalog(n_agn=N_TOT,
                            grid_radius=GRID_SIZE,
                            gw_box_radius=GW_BOX_SIZE,
                            completeness=1)
    cat = Catalog.complete_catalog

    n_events = 100
    f_agn = 0.5
    cl = 0.999
    GWEvents = MockEvents(something_else=1,
                          n_events=n_events,
                          f_agn=f_agn,
                          catalog=cat.loc[cat['in_gw_box'] == True],  
                          z_max=GW_BOX_SIZE,
                          skymap_cl=cl)

    print(GWEvents.posteriors)
    GWEvents.get_posteriors()
    print(GWEvents.posteriors)
