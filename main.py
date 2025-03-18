from mock_event_maker import MockEvent
from agn_catalogs_and_priors.mock_catalog_maker import MockCatalog
from pixelated_likelihood import *


def main():
    N_EVENTS = int(1e4)
    F_AGN = 0.5
    CL = 0.999
    Catalog = MockCatalog.from_file(file_path='/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs/mockcat_NAGN750000_ZMAX5_GWZMAX2.hdf5')
    GWEvents = MockEvent(
                    n_events=N_EVENTS,
                    f_agn=F_AGN,
                    catalog=Catalog, 
                    skymap_cl=CL,  # Unused
                    ncpu=1,  # Not doing mp
                    n_posterior_samples=int(5e3)  # More is just not happening for the time being
                )
    GWEvents.make_posteriors()
    return

if __name__ == '__main__':
    main()