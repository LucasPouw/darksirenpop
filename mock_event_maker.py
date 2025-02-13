import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import sys
import astropy.units as u
from utils import fast_z_at_value, make_nice_plots
from mock_catalog_maker import MockCatalog
from mock_skymap_maker import MockSkymap


class MockEvents(MockSkymap):

    def __init__(self,
                 
                 param_dict: dict,

                 n_events: int,
                 f_agn: float,
                 catalog: MockCatalog,  
                 z_max: float,
                 skymap_cl: float,
                 cosmology = FlatLambdaCDM(H0=67.9, Om0=0.3065)):
        
        super().__init__(n_events,
                         f_agn,
                         catalog, 
                         z_max,
                         skymap_cl,
                         cosmology)  # Inherit all properties and methods from MockSkymap
        
        self.param_dict = param_dict
        

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
                            max_redshift=GRID_SIZE,
                            gw_box_radius=GW_BOX_SIZE,
                            completeness=1)

    n_events = 100
    f_agn = 0.5
    cl = 0.999
    GWEvents = MockEvents(param_dict=None,
                          n_events=n_events,
                          f_agn=f_agn,
                          catalog=Catalog,  
                          z_max=GW_BOX_SIZE,
                          skymap_cl=cl)

    print(GWEvents.posteriors)
    GWEvents.get_posteriors()
    print(GWEvents.posteriors)
