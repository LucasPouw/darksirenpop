from mock_event_maker import MockEvent
from agn_catalogs_and_priors.mock_catalog_maker import MockCatalog
from default_globals import *  # Here are some default arguments that I'm using in multiple files
import os
from jsons.make_jsons import make_mock_data_jsons


def make_agn_catalog(n_agn, max_redshift, relative_redshift_error, completeness, catalog_dir):
    fname = f'NAGN_{n_agn}_ZMAX_{max_redshift}_SIGMA_{relative_redshift_error}'
    cat_path = os.path.join(catalog_dir, f'mockcat_{fname}.hdf5')

    print(f'\n------ GENERATING AGN CATALOG AT {cat_path} ------\n')

    Catalog = MockCatalog(n_agn=n_agn, 
                        max_redshift=max_redshift,
                        redshift_error=relative_redshift_error,
                        completeness=completeness)
    Catalog.write_to_hdf5(cat_path)
    return cat_path


def make_gws_from_catalog(catalog_path, 
                          nevents, 
                          fagn,
                          zdraw, 
                          use_extrinsic,
                          use_intrinsic, 
                          sky_area_low, 
                          sky_area_high, 
                          lumdist_relerr, 
                          mass_relerr, 
                          model_dict, 
                          hyperparam_dict_agn_path,
                          hyperparam_dict_alt_path,
                          gw_outdir,
                          ncpu,
                          npostsamps):
    
    print(f'\n------ GENERATING GW POSTERIORS AT {gw_outdir} ------\n')

    Catalog = MockCatalog.from_file(file_path=catalog_path)
    GWEvents = MockEvent(
                    n_events=nevents,
                    f_agn=fagn,
                    zdraw=zdraw,
                    use_extrinsic=use_extrinsic,
                    use_intrinsic=use_intrinsic,
                    sky_area_low=sky_area_low,
                    sky_area_high=sky_area_high,
                    lumdist_relerr=lumdist_relerr,
                    mass_relerr=mass_relerr,
                    catalog=Catalog,
                    model_dict=model_dict,
                    hyperparam_dict_agn_path=hyperparam_dict_agn_path,
                    hyperparam_dict_alt_path=hyperparam_dict_alt_path,
                    outdir=gw_outdir,
                    ncpu=ncpu,
                    n_posterior_samples=npostsamps
                )
    GWEvents.make_posteriors()
    return


def main():

    ################# ARGUMENTS FOR AGN CATALOG #################
    n_agn = int(1e4)                # Number of AGN in the catalog
    max_redshift = 0.1                # Maximum true AGN redshift
    relative_redshift_error = 0     # Relative redshift error (sigma), such that z_obs = z_true * (1 + sigma * N(0,1))
    completeness = 1                # AGN catalog completeness
    catalog_dir = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs'  # Directory to save catalog as .hdf5 file
    
    # catalog_path = make_agn_catalog(n_agn, max_redshift, relative_redshift_error, completeness, catalog_dir)  # If you want a pre-made catalog, change to cat_path = \your\path\catalog.hdf5
    catalog_path = '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs/mockcat_NAGN_10000_ZMAX_0.1_SIGMA_0.hdf5'
    #############################################################


    ################# ARGUMENTS FOR GW POSTERIORS #################
    nevents = int(1e4)      # Number of GW events to simulate
    fagn = 0.5              # Fraction of GW events from AGN
    zdraw = max_redshift               # Maximum true GW redshift
    lumdist_relerr = 0.01    # Relative luminosity distance error
    sky_area_low = 100       # (Approximate) minimum 68% CL sky area in deg^2
    sky_area_high = 200      # (Approximate) maximum 68% CL sky area in deg^2
    gw_outdir = f'/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/dl_{lumdist_relerr * 100}percent_dz_{relative_redshift_error * 100}percent_nagn_{n_agn}_dsky_{sky_area_low}_{sky_area_high}'  # DEFAULT_POSTERIOR_OUTDIR  # os.path.join(sys.path[0], "output/mock_posteriors_v7"),
    npostsamps = DEFAULT_N_POSTERIOR_SAMPLES  # Default is 50k samples per GW
    ncpu = 4    # Number of threads for writing posteriors
    json_path = f'/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/jsons/dl_{lumdist_relerr * 100}percent_dz_{relative_redshift_error * 100}percent_nagn_{n_agn}_dsky_{sky_area_low}_{sky_area_high}.json'  # .json file is used to locate GW posteriors by name, e.g., {'gw_00013': /path/to/gw_00013.hdf5'}

    use_extrinsic = True
    use_intrinsic = False  # Don't change this, intrinsic parameter stuff doesn't work yet
    if not use_intrinsic:
        model_dict, hyperparam_dict_agn_path, hyperparam_dict_alt_path, mass_relerr = None, None, None, None
    else:
        model_dict = {
                    'agn': ['PrimaryMass-gaussian'], 
                    'alt': ['PrimaryMass-gaussian']
                }
        # These dicts should contain a selection of allowed population parameters: only parameters that are implemented in models are allowed
        hyperparam_dict_agn_path = "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/jsons/hyperparam_agn.json"
        hyperparam_dict_alt_path = "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/jsons/hyperparam_alt.json"    
    
    make_gws_from_catalog(catalog_path, 
                          nevents, 
                          fagn,
                          zdraw, 
                          use_extrinsic,
                          use_intrinsic, 
                          sky_area_low, 
                          sky_area_high, 
                          lumdist_relerr, 
                          mass_relerr, 
                          model_dict, 
                          hyperparam_dict_agn_path,
                          hyperparam_dict_alt_path,
                          gw_outdir,
                          ncpu,
                          npostsamps)
    ###################################################################

    json_path = make_mock_data_jsons(posterior_samples_path=gw_outdir, outpath=json_path)
    print(f'\n------ POSTERIOR DICTIONARY LOCATED AT {json_path} ------\n')


if __name__ == '__main__':
    main()

