from mock_event_maker import MockEvent
from agn_catalogs_and_priors.mock_catalog_maker import MockCatalog
from pixelated_likelihood import *
from default_arguments import *

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
    with open("/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/jsons/hyperparam_agn.json", "r") as file:
        hyperparam_dict_agn = json.load(file)
    with open("/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/jsons/hyperparam_alt.json", "r") as file:
        hyperparam_dict_alt = json.load(file)


def main():
    N_EVENTS = int(1e3)
    F_AGN = 0.5
    ZDRAW = 2

    Catalog = MockCatalog.from_file(file_path='/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs/mockcat_NAGN_100000_ZMAX_3_SIGMA_0.01.hdf5')
    GWEvents = MockEvent(
                    n_events=N_EVENTS,
                    f_agn=F_AGN,
                    zdraw=ZDRAW,
                    use_extrinsic=True,
                    use_intrinsic=True,
                    sky_area_low=10,
                    sky_area_high=20,
                    lumdist_relerr=0.01,
                    catalog=Catalog,
                    model_dict=model_dict,
                    hyperparam_dict_agn=hyperparam_dict_agn,
                    hyperparam_dict_alt=hyperparam_dict_alt,
                    outdir=DEFAULT_POSTERIOR_OUTDIR,  # os.path.join(sys.path[0], "output/mock_posteriors_v7"),
                    ncpu=4,
                    n_posterior_samples=int(5e4)
                )
    GWEvents.make_posteriors()
    return

if __name__ == '__main__':
    main()