from mock_event_maker import MockEvent
from agn_catalogs_and_priors.mock_catalog_maker import MockCatalog
from pixelated_likelihood import *
import os, sys


def main():
    N_EVENTS = int(10000)
    F_AGN = 0.5
    ZDRAW = 2

    Catalog = MockCatalog.from_file(file_path='/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs/mockcat_NAGN100000_ZMAX_3_SIGMA_0.01.hdf5')
    GWEvents = MockEvent(
                    n_events=N_EVENTS,
                    f_agn=F_AGN,
                    zdraw=ZDRAW,
                    sky_area_low=160,
                    sky_area_high=161,
                    catalog=Catalog,
                    outdir=os.path.join(sys.path[0], "output/mock_posteriors_v6"),
                    ncpu=1,  # Not doing mp
                    n_posterior_samples=int(5e4)
                )
    GWEvents.make_posteriors()
    return

if __name__ == '__main__':
    main()