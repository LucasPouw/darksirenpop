#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from utils import make_nice_plots, fast_z_at_value, spherical2cartesian
from mock_skymap_maker import MockSkymap
from mock_catalog_maker import MockCatalog
from mock_event_maker import MockEvent
import sys
import astropy.units as u
from scipy.stats import norm
from tqdm import tqdm
from scipy.spatial import KDTree


colors = ['black', 'blue', 'red']


def gaussian_3d(xyz, mus, sigmas):
    x, y, z = xyz
    mu_x, mu_y, mu_z = mus
    sig_x, sig_y, sig_z = sigmas
    return norm.pdf(x, loc=mu_x, scale=sig_x) * norm.pdf(y, loc=mu_y, scale=sig_y) * norm.pdf(z, loc=mu_z, scale=sig_z)


def kdtree_in_skymap(points, centers, radii):
    tree = KDTree(points)
    indices = tree.query_ball_point(centers, radii)
    return indices


def get_num_dens(MockSkymaps, MockCatalog):
    ''' Assuming uniform AGN number density. TODO: Change to weighted/pixelated. '''
    cosmo = MockCatalog.cosmo
    uniform_agn_numdens = len(MockCatalog.incomplete_catalog) / cosmo.comoving_volume(MockCatalog.max_redshift).value

    # print(uniform_agn_numdens)

    MockSkymaps.properties['agn_numdens'] = np.tile(uniform_agn_numdens, MockSkymaps.n_events)  # All maps have the same AGN number density for now


def _crossmatch_skymaps(MockSkymaps, MockCatalog, true_agn_pos=False):

    '''MockSkymaps can be either MockSkymap or MockEvent object'''

    incomplete_cat = MockCatalog.incomplete_catalog

    skymap_centers_numpy = MockSkymaps.properties[['x_meas_center', 'y_meas_center', 'z_meas_center']].to_numpy()
    
    if true_agn_pos:  # Neglect redshift errors and use true AGN positions
        incomplete_cat_cartesians_numpy = incomplete_cat[['x', 'y', 'z']].to_numpy()
    else:
        incomplete_cat_cartesians_numpy = incomplete_cat[['x_meas', 'y_meas', 'z_meas']].to_numpy()

    # Calculate a mask for selecting the AGN that lie within the skymap
    # diff = incomplete_cat_cartesians_numpy[:, np.newaxis, :] - skymap_centers_numpy
    # in_skymap_slow = np.sum(diff**2, axis=2) < MockSkymaps.properties['loc_rad'].to_numpy()**2  # Shape (n_agn, n_gw)
    # del diff

    # Calculate lists of AGN indeces that are located in the GW localization volume
    in_skymap = kdtree_in_skymap(incomplete_cat_cartesians_numpy, skymap_centers_numpy, MockSkymaps.properties['loc_rad'].to_numpy())  # Works for spherical mock volumes, will be challenging applying to real data...

    ### The crossmatching is vectorized by flattening the array of idx lists (in_skymap) and repeating the skymap Gaussian mean/std for each AGN ###
    flat_in_skymap = np.concatenate(in_skymap)
    agn_in_all_maps = incomplete_cat.iloc[flat_in_skymap]

    # Create a repeated mean and std array for each point (since lists are of different sizes)
    n_agn_per_skymap = [len(idx) for idx in in_skymap] 
    repeated_means_x = np.repeat(MockSkymaps.properties['x_meas_center'], n_agn_per_skymap)
    repeated_means_y = np.repeat(MockSkymaps.properties['y_meas_center'], n_agn_per_skymap)
    repeated_means_z = np.repeat(MockSkymaps.properties['z_meas_center'], n_agn_per_skymap)
    repeated_stds = np.repeat(MockSkymaps.properties['sigma'], n_agn_per_skymap)

    # Compute normal distribution values at these points with the correct mean/std
    if true_agn_pos:
        x_probs = norm.pdf(agn_in_all_maps['x'], loc=repeated_means_x, scale=repeated_stds)
        y_probs = norm.pdf(agn_in_all_maps['y'], loc=repeated_means_y, scale=repeated_stds)
        z_probs = norm.pdf(agn_in_all_maps['z'], loc=repeated_means_z, scale=repeated_stds)
    else:
        x_probs = norm.pdf(agn_in_all_maps['x_meas'], loc=repeated_means_x, scale=repeated_stds)
        y_probs = norm.pdf(agn_in_all_maps['y_meas'], loc=repeated_means_y, scale=repeated_stds)
        z_probs = norm.pdf(agn_in_all_maps['z_meas'], loc=repeated_means_z, scale=repeated_stds)

    # Split back into lists corresponding to original queries
    split_back_x = np.split(x_probs, np.cumsum(n_agn_per_skymap)[:-1])
    split_back_y = np.split(y_probs, np.cumsum(n_agn_per_skymap)[:-1])
    split_back_z = np.split(z_probs, np.cumsum(n_agn_per_skymap)[:-1])

    total_gw_prob = np.zeros(MockSkymaps.n_events)
    for i in range(len(split_back_x)):  # Not vectorized since the split back lists have arrays of different sizes as elements

        if n_agn_per_skymap[i] == 0:
            total_gw_prob[i] = 0
            continue

        # print(f"Expected {(MockSkymaps.properties['agn_numdens'].iloc[i] * MockSkymaps.properties['loc_vol'].iloc[i])} got {n_agn_per_skymap[i]}")
        total_gw_prob[i] = np.sum(split_back_x[i] * split_back_y[i] * split_back_z[i]) / (MockSkymaps.properties['agn_numdens'].iloc[i] * MockSkymaps.properties['loc_vol'].iloc[i])

    return total_gw_prob


def _crossmatch_intrinsic_params():
    # crossmatch_skymaps(MockSkymaps, MockCatalog, true_agn_pos=False)
    # do_other_parameters
    return


def crossmatch(MockEvents, MockCatalog, use_intrinsic_params=False, true_agn_pos=False):

    try:
        if MockEvents.posteriors == None:
            MockEvents.get_posteriors()
    except ValueError:  # In this case the posteriors already exist and we don't want to overwrite them
        pass

    incomplete_cat = MockCatalog.incomplete_catalog

    if len(incomplete_cat) == 0:
        print('\nWARNING: AGN catalog is empty. Not using extrinsic parameters in this analysis!\n')
        p_agn_sky = np.ones(MockEvents.n_events)
        p_alt_sky = np.ones(MockEvents.n_events)
    else:   
        p_agn_sky = _crossmatch_skymaps(MockEvents, MockCatalog, true_agn_pos=False)
        p_alt_sky = MockEvents.skymap_cl / MockEvents.properties['loc_vol']
    
    if use_intrinsic_params:
        p_agn_intrinsic, p_alt_intrinsic = _crossmatch_intrinsic_params()
    else:
        p_agn_intrinsic = np.ones(MockEvents.n_events)
        p_alt_intrinsic = np.ones(MockEvents.n_events)

    return p_agn_sky * p_agn_intrinsic, p_alt_sky * p_alt_intrinsic


if __name__ == '__main__':

    make_nice_plots()

    N_TOT = 1
    GRID_SIZE = 40  # Radius of the whole grid in redshift
    GW_BOX_SIZE = 30  # Radius of the GW box in redshift
    
    Catalog = MockCatalog(n_agn=N_TOT,
                            max_redshift=GRID_SIZE,
                            gw_box_radius=GW_BOX_SIZE,
                            completeness=1)

    n_events = 10000
    f_agn = 0.5
    cl = 0.999
    GWEvents = MockEvent(
                        n_events=n_events,
                        f_agn=f_agn,
                        catalog=Catalog,
                        skymap_cl=cl
                    )
    
    get_num_dens(GWEvents, Catalog)  # TODO: Changes in principle after each measuring of the catalog, right? Check this.

    n_catalog_resamps = 100
    skyprobs = np.zeros(n_events)
    for i in tqdm(range(n_catalog_resamps)):
        Catalog.measure_redshift()  # TODO: Does not work in empty-cat case, but shouldn't be called then anyway
        skyprobs += crossmatch(MockCatalog=Catalog, MockEvents=GWEvents, use_intrinsic_params=False, true_agn_pos=False)[0]
    skyprobs /= n_catalog_resamps

    # print(np.sum(skyprobs == 0), 'samped')

    # skyprobs_truepos = crossmatch_skymaps(MockCatalog=Catalog, MockSkymaps=SkyMaps, true_agn_pos=True)
    # print(np.sum(skyprobs_truepos == 0), 'truepos')

    p_alt = cl / GWEvents.properties['loc_vol']
    from_agn = GWEvents.properties['from_agn']

    fig, ax = plt.subplots(figsize=(10, 10))
    xmin = -12
    xmax = 6
    xx = np.logspace(xmin, xmax, 100)
    ax.plot(xx, xx, linestyle='dashed', color='black', zorder=6)
    ax.scatter(p_alt[from_agn], skyprobs[from_agn], color='blue', alpha=0.3, marker='.', zorder=5, label='From AGN')
    ax.scatter(p_alt[~from_agn], skyprobs[~from_agn], color='red', alpha=0.3, marker='.', zorder=5, label='From ALT')
    # ax.scatter(p_alt, skyprobs_truepos, color='red', alpha=0.3, marker='.', zorder=4)
    ax.set_xlabel(r'$p_{\rm alt}$')
    ax.set_ylabel(r'$p_{\rm agn}$')
    ax.set_xlim(10**xmin, 10**xmax)
    ax.set_ylim(10**xmin, 10**xmax)
    ax.loglog()
    ax.legend()
    ax.grid()
    plt.show()

    #%%

    # print(skyprobs, 0.999 / GWEvents.properties['loc_vol'])









    # # cat['r'] = 10000
    # # cat['theta'] = 0
    # # cat['phi'] = 0

    # # x,y,z = spherical2cartesian(cat['r'], cat['theta'], cat['phi'])
    # # cat['x'] = x
    # # cat['y'] = y
    # # cat['z'] = z

    # # cat['redshift'] = fast_z_at_value(Catalog.cosmo.comoving_distance, cat['r'].iloc[0] * u.Mpc)

    # plt.figure(figsize=(8,6))
    # # for j, zerror in enumerate([10, 0.01]):
    # cls = np.zeros(100)
    # for i in tqdm(range(100)):
    #     # cat['redshift_error'] = zerror
    #     # cat['redshift_meas'] = np.random.normal(loc=cat['redshift'], scale=cat['redshift_error'])
    #     # cat['r_meas'] = Catalog.cosmo.comoving_distance(cat['redshift_meas'].iloc[0])

    #     # # print(cat['r_meas'], 'MEASURED')
    #     # # print(cat['r'], 'REAL')

    #     # x,y,z = spherical2cartesian(cat['r_meas'], cat['theta'], cat['phi'])
    #     # cat['x_meas'] = x
    #     # cat['y_meas'] = y
    #     # cat['z_meas'] = z
    #     # cat['in_gw_box'] = True

    #     Catalog.measure_redshift()

        # n_maps = 1000
        # f_agn = 1
        # cl = 0.999
        # GWEvents = MockSkymap(n_maps=n_maps,
        #                         f_agn=f_agn,
        #                         catalog=cat.loc[cat['in_gw_box'] == True],  
        #                         z_max=GW_BOX_SIZE,
        #                         CL=cl)
        
    #     # print(GWEvents.properties['loc_rad'])
        
    #     cls[i] = crossmatch_skymaps(GWEvents, Catalog)
    # plt.hist(cls, bins=10, density=True, histtype='step', linewidth=5)
    # plt.legend()
    # plt.xlabel('Fraction of GWs with host AGN in 99.9% CL')
    # plt.ylabel('Probability density')
    # plt.savefig('compare_error_effect_realsitu.pdf', bbox_inches='tight')
    # plt.show()
    
    # # n_iter = 500
    # # gw_probs = np.zeros((n_maps, n_iter))
    # # for i in tqdm(range(n_iter)):
    # #     Catalog.measure_redshift()
    # #     gw_probs[:, i] = crossmatch_skymaps(GWEvents, Catalog)

    # # plt.figure()
    # # for i in range(3):
    # #     plt.hist(gw_probs[i,:], bins=30, density=True, histtype='step', color=colors[i])
    # # plt.savefig('p90.pdf')
    # # plt.show()

# %%
