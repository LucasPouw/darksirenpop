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


class MockSkymap():

    def __init__(
        self,
        n_events: int,
        f_agn: float,
        catalog: MockCatalog,
        skymap_cl: float,
        n_posterior_samples: int = 1000, # TODO: make n_posterior_samples variable per GW event or resample GW posteriors to minimum of ensemble
        cosmology = FlatLambdaCDM(H0=67.9, Om0=0.3065)
    ):

        """
        Building skymap dataframe. 
        
        TODO: documentation
        """

        assert isinstance(catalog, MockCatalog), 'Provided catalog is not MockCatalog instance.'

        self.n_events = n_events
        self.f_agn = f_agn
        self.n_agn_events = round(self.f_agn * self.n_events)
        self.n_alt_events = self.n_events - self.n_agn_events
        self.skymap_cl = skymap_cl
        self.n_posterior_samples = n_posterior_samples
        self.cosmo = cosmology
        
        self.MockCatalog = catalog  # Used to access completeness in likelihood, inheritance is not necessary
        cat = catalog.complete_catalog
        self.catalog = cat.loc[cat['in_gw_box'] == True]  # From which to generate AGN GWs
        self.z_max = catalog.gw_box_radius  # Maximum redshift to generate ALT GWs from, currently using doing this uniform in comoving volume

        if (self.n_agn_events != 0) and (len(self.catalog) == 0) and (len(cat) != 0):
            sys.exit('\nTried to generate GWs from AGN, but only found AGN outside the GW box. Either provide more AGN or put f_agn = 0.\nExiting...')
        
        # columns=['r', 'theta', 'phi',
        #         'x', 'y', 'z',
        #         'redshift', 
        #         'r_meas_center', 'theta_meas_center', 'phi_meas_center',
        #         'x_meas_center', 'y_meas_center', 'z_meas_center',
        #         'redshift_meas_center',
        #         'sigma', 'loc_vol', 'loc_rad',
        #         'from_agn']
        self.properties = pd.DataFrame()
        if len(cat) != 0:
            print('\nFound AGN in GW box. Generating skymaps...')
            self.make_skymaps()
        else:
            print('\nEmpty AGN catalog provided. Not generating skymaps.')

        self.posteriors = None  # Call get_skymap_posteriors() method to make posteriors, call again to make new posteriors from the same true values in self.properties

    
    def select_agn_hosts(self):
        host_idx = np.random.choice(np.arange(len(self.catalog)), self.n_agn_events)
        return host_idx
    

    def sample_alt_coords(self):
        r, theta, phi = uniform_shell_sampler(0, self.cosmo.comoving_distance(self.z_max).value, n_samps=self.n_alt_events)
        x, y, z = spherical2cartesian(r, theta, phi)
        return x, y, z, r, theta, phi
    

    def make_true_skymap_locations(self):
        # AGN origin - sampled from catalog
        host_idx = self.select_agn_hosts()
        self.properties['x'] = self.catalog.loc[host_idx, 'x']
        self.properties['y'] = self.catalog.loc[host_idx, 'y']
        self.properties['z'] = self.catalog.loc[host_idx, 'z']
        self.properties['r'] = self.catalog.loc[host_idx, 'r']
        self.properties['theta'] = self.catalog.loc[host_idx, 'theta']
        self.properties['phi'] = self.catalog.loc[host_idx, 'phi']
        self.properties['redshift'] = self.catalog.loc[host_idx, 'redshift']
        self.properties['from_agn'] = np.ones(self.n_agn_events, dtype=bool)

        self.properties = self.properties.reset_index(drop=True)  # Remove random ordering of indeces caused by sampling of the catalog

        if self.n_alt_events != 0:
            # ALT origin - uniform in comoving volume
            temp_alt_df = pd.DataFrame()
            x, y, z, r, theta, phi = self.sample_alt_coords()
            temp_alt_df['x'] = x
            temp_alt_df['y'] = y
            temp_alt_df['z'] = z
            temp_alt_df['r'] = r
            temp_alt_df['theta'] = theta
            temp_alt_df['phi'] = phi
            temp_alt_df['redshift'] = fast_z_at_value(self.cosmo.comoving_distance, r * u.Mpc)
            temp_alt_df['from_agn'] = np.zeros(self.n_alt_events, dtype=bool)

            self.properties = pd.concat([self.properties, temp_alt_df], ignore_index=True)  # TODO: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.

            del temp_alt_df


    def sample_sigmas(self, mean=50, std=100):
        # TODO: Make a better distribution
        return np.abs( np.random.normal(loc=mean, scale=std, size=self.n_events) )
    

    def cl_volume_from_sigma(self, sigmas):
        '''Calculates length of CL% vector when x, y, z are normally distributed with the same sigma'''
        radii = np.zeros(len(sigmas))
        for i, sigma in enumerate(sigmas):
            vector_length_cdf = lambda x: erf(x / (sigma * np.sqrt(2))) - np.sqrt(2) / (sigma * np.sqrt(np.pi)) * x * np.exp(-x**2 / (2 * sigma**2)) - self.skymap_cl
            radii[i] = root_scalar(vector_length_cdf, bracket=[0, 100 * sigma], method='bisect').root
        return 4 * np.pi * radii**3 / 3, radii

    
    def make_skymap_posteriors(self, sigmas):
        xyz_true = self.properties[['x', 'y', 'z']].to_numpy()
        std = np.tile(sigmas[:, np.newaxis], 3)  # Same sigma for x, y and z
        xyz_meas = np.random.normal(loc=xyz_true, scale=std, size=(self.n_events, 3))

        xyz_posteriors = np.random.normal(loc=xyz_meas[:, np.newaxis, :], scale=std[:, np.newaxis, :], size=(self.n_events, self.n_posterior_samples, 3))
        x, y, z = xyz_posteriors[:, :, 0], xyz_posteriors[:, :, 1], xyz_posteriors[:, :, 2]
        r, theta, phi = cartesian2spherical(x, y, z)
        redshift = fast_z_at_value(self.cosmo.comoving_distance, r * u.Mpc)

        self.posteriors = pd.DataFrame({'x': list(x),
                                        'y': list(y),
                                        'z': list(z),
                                        'r': list(r),
                                        'theta': list(theta),
                                        'phi': list(phi),
                                        'redshift': list(redshift)})
        
        self.properties['x_meas_center'] = np.median(x, axis=1)
        self.properties['y_meas_center'] = np.median(y, axis=1)
        self.properties['z_meas_center'] = np.median(z, axis=1)
        self.properties['r_meas_center'] = np.median(r, axis=1)
        self.properties['theta_meas_center'] = np.median(theta, axis=1)
        self.properties['phi_meas_center'] = np.median(phi, axis=1)
        self.properties['redshift_meas_center'] = np.median(redshift, axis=1)


    def make_skymaps(self):
        self.make_true_skymap_locations()
        sigmas = self.sample_sigmas()  # Independent of host
        self.properties['sigma'] = sigmas
        self.properties['loc_vol'], self.properties['loc_rad'] = self.cl_volume_from_sigma(sigmas)
    

    def get_skymap_posteriors(self):
        self.make_skymap_posteriors(self.properties['sigma'].to_numpy())
        return self.posteriors
    

    # def sigma_from_cl_volume(self, volumes):
    #     '''CURRENTLY UNTESTED'''
    #     radii = (3 * volumes / (4 * np.pi))**(1/3)
    #     sigmas = np.zeros(len(radii))
    #     for i, radius in enumerate(radii):
    #         vector_length_cdf = lambda x: erf(radius / (x * np.sqrt(2))) - np.sqrt(2) / (x * np.sqrt(np.pi)) * x * np.exp(-radius**2 / (2 * x**2)) - self.skymap_cl
    #         sigmas[i] = root_scalar(vector_length_cdf, bracket=[0, 100 * radius], method='bisect').root
    #     return sigmas


if __name__ == '__main__':

    make_nice_plots()

    N_TOT = 10000
    GRID_SIZE = 10  # Radius of the whole grid in redshift
    GW_BOX_SIZE = 2  # Radius of the GW box in redshift
    
    Catalog = MockCatalog(n_agn=N_TOT,
                            max_redshift=GRID_SIZE,
                            gw_box_radius=GW_BOX_SIZE,
                            completeness=1)

    n_events = 3
    f_agn = 0.5
    skymap_cl = 0.999
    SkyMaps = MockSkymap(n_events=n_events,
                        f_agn=f_agn,
                        catalog=Catalog,
                        skymap_cl=skymap_cl)
    
    colors = ['red', 'blue', 'black']
    posteriors = ['x', 'y', 'z', 'r', 'theta', 'phi', 'redshift']
    labels = [r'$x$ [Mpc]', r'$y$ [Mpc]', r'$z$ [Mpc]', r'$r_{\rm com}$ [Mpc]', r'$\theta$ [rad]', r'$\phi$ [rad]', 'redshift']

    # plt.figure(figsize=(8,6))
    # plt.hist(np.log10(SkyMaps.properties['loc_vol']), density=True, bins=30)
    # plt.xlabel(r'$\log_{10}\left( 99.9\% \, \mathrm{CL \, localization \, volume \, [Mpc^{3}]} \right)$')
    # plt.ylabel('Probability density')
    # # plt.savefig('locvol.pdf')
    # plt.show()

    # plt.figure(figsize=(8,6))
    # plt.ecdf(np.log10(SkyMaps.properties['loc_vol']))
    # plt.xlabel(r'$\log_{10}\left( 99.9\% \, \mathrm{CL \, localization \, volume \, [Mpc^{3}]} \right)$')
    # plt.ylabel('CDF')
    # # plt.savefig('locvolcdf.pdf')
    # plt.show()


    # fig = plt.figure(figsize=(24, 18))

    # posteriors = SkyMaps.get_skymap_posteriors()

    # for i, key in enumerate(posteriors):
    #     ax = fig.add_subplot(3, 3, i+1)
    #     ax.set_xlabel(labels[i])
    #     ax.set_ylabel('Probability density')
        
    #     max_height = 0
    #     for j in range(n_events):
    #         heights, _, _ = ax.hist(posteriors[key].iloc[j], density=True, bins=50, histtype='step', linewidth=5, color=colors[j])
    #         if np.max(heights) > max_height: 
    #             max_height = np.max(heights)
    #     for j in range(n_events):
    #         ax.vlines(SkyMaps.properties[key].iloc[j], 0, 1.1 * max_height, color=colors[j], linestyle='dotted', linewidth=5)

    #     ax.set_ylim(0, 1.1 * max_height)
    
    # plt.tight_layout()
    # # plt.savefig('skymap_proof.pdf')
    # plt.show()
