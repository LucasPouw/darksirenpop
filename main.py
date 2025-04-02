from mock_event_maker import MockEvent
from agn_catalogs_and_priors.mock_catalog_maker import MockCatalog
from default_arguments import *


def main():
    nevents = int(1e3)
    fagn = 0.5
    zdraw = 2
    outdir = DEFAULT_POSTERIOR_OUTDIR  # os.path.join(sys.path[0], "output/mock_posteriors_v7"),
    npostsamps = DEFAULT_N_POSTERIOR_SAMPLES
    ncpu = 4

    model_dict = {
                'agn': ['PrimaryMass-gaussian'], 
                'alt': ['PrimaryMass-gaussian']
            }
    
    # These dicts should contain a selection of allowed population parameters: only parameters that are implemented in models are allowed
    hyperparam_dict_agn_path = "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/jsons/hyperparam_agn.json"
    hyperparam_dict_alt_path = "/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/jsons/hyperparam_alt.json"

    Catalog = MockCatalog.from_file(file_path='/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs/mockcat_NAGN_100000_ZMAX_3_SIGMA_0.01.hdf5')
    GWEvents = MockEvent(
                    n_events=nevents,
                    f_agn=fagn,
                    zdraw=zdraw,
                    use_extrinsic=True,
                    use_intrinsic=True,
                    sky_area_low=10,
                    sky_area_high=20,
                    lumdist_relerr=0.01,
                    mass_relerr=0.01,
                    catalog=Catalog,
                    model_dict=model_dict,
                    hyperparam_dict_agn_path=hyperparam_dict_agn_path,
                    hyperparam_dict_alt_path=hyperparam_dict_alt_path,
                    outdir=outdir,
                    ncpu=ncpu,
                    n_posterior_samples=npostsamps
                )
    GWEvents.make_posteriors()
    return

if __name__ == '__main__':
    main()

    
#     make_nice_plots()

#     # N_TOT = 100
#     ZDRAW = 2
#     N_EVENTS = int(100)
#     F_AGN = 0.5

#     ##### FAGN = 0 EVENTS WITH SKYMAPS #####
#     # events = MockEvent.with_skymaps_without_agn(n_events=N_EVENTS,
#     #                                             zdraw=GW_BOX_SIZE)
    
#     # events.make_posteriors()
#     # print(events.posteriors)

#     ########################################  

#     # The model dictionary should contain choices of implemented models. These models have parameters for which we pass another dictionary
#     # TODO: think about what happens when we want to do the chi_eff vs q correlation here (mass ratios are used in the mass models already)
#     model_dict = {
#                 'agn': ['PrimaryMass-gaussian'], 
#                 'alt': ['PrimaryMass-gaussian']
#             }
    
#     # These dicts should contain a selection of allowed population parameters: only parameters that are implemented in models are allowed
#     try:
#         with open("hyperparam_agn.json", "r") as file:
#             hyperparam_dict_agn = json.load(file)
#         with open("hyperparam_alt.json", "r") as file:
#             hyperparam_dict_alt = json.load(file)
#     except FileNotFoundError:
#         with open("/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/inputs/hyperparam_agn.json", "r") as file:
#             hyperparam_dict_agn = json.load(file)
#         with open("/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/inputs/hyperparam_alt.json", "r") as file:
#             hyperparam_dict_alt = json.load(file)


#     ##### EVENTS WITH EMPTY CATALOG #####
#     # events = MockEvent.without_skymaps(n_events=N_EVENTS,
#     #                                    f_agn=F_AGN,
#     #                                    zdraw=GW_BOX_SIZE,
#     #                                    model_dict=model_dict,
#     #                                    hyperparam_dict_agn=hyperparam_dict_agn,
#     #                                    hyperparam_dict_alt=hyperparam_dict_alt,
#     #                                    )
#     # events.make_posteriors()
#     # print(events.posteriors)
#     #####################################
    

#     # Catalog = MockCatalog(n_agn=N_TOT,
#     #                         max_redshift=GRID_SIZE,
#     #                         gw_box_radius=GW_BOX_SIZE,
#     #                         completeness=1)

#     Catalog = MockCatalog.from_file(file_path='/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs/mockcat_NAGN750000_ZMAX_3.hdf5')
#     GWEvents = MockEvent(
#                     n_events=N_EVENTS,
#                     f_agn=F_AGN,
#                     zdraw=ZDRAW,
#                     catalog=Catalog,
#                     ncpu=1,
#                     n_posterior_samples=int(5e3)
#                 )

#     # GWEvents = MockEvent(
#     #                     model_dict=model_dict,
#     #                     n_events=N_EVENTS,
#     #                     f_agn=F_AGN,
#     #                     use_skymaps=True,
#     #                     hyperparam_dict_agn=hyperparam_dict_agn,
#     #                     hyperparam_dict_alt=hyperparam_dict_alt,
#     #                     catalog=Catalog, 
#     #                 )

#     GWEvents.make_posteriors()
#     # print(GWEvents.posteriors)

#     # GWEvents.write_3D_location_samples_to_hdf5(outdir='./output/mocksamples/without_lumdist')

#     ############# PLOTTING POSITION POSTERIORS #############

#     # for i in range(10):
#     #     ra_post = np.array(GWEvents.posteriors['ra'].iloc[i])
#     #     dec_post = np.array(GWEvents.posteriors['dec'].iloc[i])
#     #     z_post = np.array(GWEvents.posteriors['redshift'].iloc[i])

#     #     ra_truth = GWEvents.truths['ra'].iloc[i]
#     #     dec_truth = GWEvents.truths['dec'].iloc[i]
#     #     z_truth = GWEvents.truths['redshift'].iloc[i]

#     #     fig, ax = plt.subplots(ncols=3, figsize=(24, 6))
#     #     ax[0].scatter(ra_post, dec_post, color='black', marker='.', alpha=0.4)
#     #     ax[0].scatter(ra_truth, dec_truth, color='red', marker='o', zorder=5)
#     #     ax[0].set_xlabel('RA')
#     #     ax[0].set_ylabel('Dec')

#     #     ax[1].scatter(ra_post, z_post, color='black', marker='.', alpha=0.4)
#     #     ax[1].scatter(ra_truth, z_truth, color='red', marker='o', zorder=5)
#     #     ax[1].set_xlabel('RA')
#     #     ax[1].set_ylabel('z')

#     #     ax[2].scatter(z_post, dec_post, color='black', marker='.', alpha=0.4)
#     #     ax[2].scatter(z_truth, dec_truth, color='red', marker='o', zorder=5)
#     #     ax[2].set_xlabel('z')
#     #     ax[2].set_ylabel('Dec')
#     #     plt.show()

#     ############# INTINSIC PARAMETERS #############

#     agn_colors = np.array(['red', 'blue', 'black'])
#     alt_colors = np.array(['coral', 'skyblue', 'grey'])

#     agn_mass1 = GWEvents.truths['mass_1_source'].loc[GWEvents.truths['from_agn']]
#     alt_mass1 = GWEvents.truths['mass_1_source'].loc[~GWEvents.truths['from_agn']]
#     # agn_mass2 = GWEvents.truths['mass_2_source'].loc[GWEvents.truths['from_agn']]
#     # alt_mass2 = GWEvents.truths['mass_2_source'].loc[~GWEvents.truths['from_agn']]

#     agn_mass1_post = GWEvents.posteriors['mass_1_source'].loc[GWEvents.truths['from_agn']]
#     alt_mass1_post = GWEvents.posteriors['mass_1_source'].loc[~GWEvents.truths['from_agn']]
#     # agn_mass2_post = GWEvents.posteriors['mass_2_source'].loc[GWEvents.truths['from_agn']]
#     # alt_mass2_post = GWEvents.posteriors['mass_2_source'].loc[~GWEvents.truths['from_agn']]

#     ### PLOTTING POPULATION PRIORS ###

#     # plt.figure(figsize=(8,6))
#     # plt.hist(agn_mass1, density=True, bins=30, histtype='step', color='blue', linewidth=3, label='AGN m1')
#     # plt.hist(alt_mass1, density=True, bins=30, histtype='step', color='red', linewidth=3, label='ALT m1')
#     # plt.hist(agn_mass2, density=True, bins=30, histtype='step', color='skyblue', linewidth=3, label='AGN m2')
#     # plt.hist(alt_mass2, density=True, bins=30, histtype='step', color='coral', linewidth=3, label='ALT m2')
#     # plt.legend()
#     # plt.show()

#     ### PLOTTING INTRINSIC PARAM POSTERIORS ###

#     # plt.figure(figsize=(8,6))
#     # h, _, _ = plt.hist(agn_mass1_post, density=True, bins=30, histtype='step', color=agn_colors[:GWEvents.n_agn_events], linewidth=3, label='Measured AGN')
#     # h2, _, _, = plt.hist(alt_mass1_post, density=True, bins=30, histtype='step', color=alt_colors[:GWEvents.n_alt_events], linewidth=3, label='Measured ALT')
    
#     # ymin = 0
#     # ymax = 1.1 * max(max(h.flatten()), max(h2.flatten()))
#     # plt.vlines(agn_mass1, ymin, ymax, color=agn_colors[:GWEvents.n_agn_events], linestyle='dashed', linewidth=3, label='True AGN')
#     # plt.vlines(alt_mass1, ymin, ymax, color=alt_colors[:GWEvents.n_alt_events], linestyle='dashed', linewidth=3, label='True ALT')   
#     # plt.xlabel('log Primary mass')
#     # plt.ylabel('Probability density')
#     # plt.ylim(ymin, ymax)
#     # plt.legend()
#     # plt.show()
