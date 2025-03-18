import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import astropy.units as u
from utils import uniform_shell_sampler, fast_z_at_value, make_nice_plots, spherical2cartesian, cartesian2spherical
from agn_catalogs_and_priors.mock_catalog_maker import MockCatalog
from scipy.stats import vonmises_fisher
from astropy.table import Table
import h5py
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class MockSkymap():

    def __init__(
        self,
        n_events: int,
        f_agn: float,
        catalog: MockCatalog,
        catalog_path: str,
        skymap_cl: float,  # TODO: Currently unused
        n_posterior_samples: int,
        cosmology,
        outdir: str,
        ncpu: int
    ):

        """
        Building skymap dataframe. Instances of MockSkymap could be buggy, this is not rigorously tested. Preferably access through MockEvent.
        
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
        self.outdir = outdir
        self.ncpu = ncpu

        if catalog is not None:
            print('Using provided MockCatalog object.')
            self.MockCatalog = catalog
        elif catalog_path is not None:
            print(f'Using pre-existing AGN catalog located at `{catalog_path}`.')
            self.MockCatalog = MockCatalog.from_file(catalog_path)

        cat = catalog.complete_catalog
        self.catalog = cat.loc[cat['can_host_gw'] == True]  # From which to generate AGN GWs
        self.max_gw_redshift = catalog.gw_box_radius  # Maximum redshift to generate ALT GWs from, currently using doing this uniform in comoving volume

        if (self.n_agn_events != 0) and (len(self.catalog) == 0) and (len(cat) != 0):
            sys.exit('\nTried to generate GWs from AGN, but only found AGN outside the GW box. Either provide more AGN or put f_agn = 0.\nExiting...')
    

    def make_true_3D_locations(self) -> None:
        # AGN origin - hosts sampled from AGN catalog
        host_idx = self._select_agn_hosts()
        self.truths['ra'] = self.catalog.loc[host_idx, 'ra_true']
        self.truths['dec'] = self.catalog.loc[host_idx, 'dec_true']
        self.truths['rcom'] = self.catalog.loc[host_idx, 'rcom_true']
        self.truths['rlum'] = self.catalog.loc[host_idx, 'rlum_true']
        self.truths['redshift'] = self.catalog.loc[host_idx, 'redshift_true']
        self.truths['from_agn'] = np.ones(self.n_agn_events, dtype=bool)
        self.truths = self.truths.reset_index(drop=True)  # Remove random ordering of indeces caused by sampling of the catalog

        if self.n_alt_events != 0:
            # ALT origin - hosts uniform in comoving volume
            temp_alt_df = pd.DataFrame()
            r, theta, phi = self._sample_alt_coords()
            z = fast_z_at_value(self.cosmo.comoving_distance, r * u.Mpc)
            temp_alt_df['ra'] = phi
            temp_alt_df['dec'] = 0.5*np.pi - theta
            temp_alt_df['rcom'] = r
            temp_alt_df['rlum'] = self.cosmo.luminosity_distance(z).value
            temp_alt_df['redshift'] = z
            temp_alt_df['from_agn'] = np.zeros(self.n_alt_events, dtype=bool)
            self.truths = pd.concat([self.truths, temp_alt_df], ignore_index=True)  # TODO: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
            del temp_alt_df

        # Save to HDF5
        print('Writing truths to hdf5...')
        for index in tqdm(range(self.n_events)):
            try:
                filename = os.path.join(self.outdir, f"gw_{index:05d}.h5")
                with h5py.File(filename, "a") as f:
                    mock_group = f.require_group("mock")  # Ensure 'mock' group exists
                    truth_group = mock_group.require_group("truths")  # Ensure 'truths' exists

                    for column in ['ra', 'dec', 'rcom', 'rlum', 'redshift', 'from_agn']:
                        truth_group.create_dataset(column, data=self.truths[column].iloc[index], dtype="f8")

            except Exception as e:
                sys.exit(f"Error in event {index}: {e}")


    def make_3D_location_posteriors(self, low:int=100, high:int=10000) -> None:
        ''' Sky position posteriors are modeled as a 2D VonMises-Fisher distribution '''

        # Get cartesian components of directional unit vector to GW origin
        x_true, y_true, z_true = spherical2cartesian(1, 0.5 * np.pi - self.truths['dec'], self.truths['ra'])
        
        # Compute VonMises-Fisher concentration parameters
        sky_areas_68p = self._sample_68p_sky_area(low, high)
        kappas = self._kappa_from_sky_area(sky_areas_68p)

        dtrue = self.truths['rlum'].to_numpy()
        sigma = 0.1  # 10% error on luminosity distance
        dobs = dtrue * (1. + sigma * np.random.normal(size=self.n_events))  # Observed distances

        args = [(i, x_true[i], y_true[i], z_true[i], kappas[i], dobs[i], sigma) for i in range(self.n_events)]
        for arg in tqdm(args):
            _ = self._process_single_event_3dloc(arg)

        # TODO: WHY IS THIS SO SLOW???
        # with Pool(processes=min(cpu_count(), self.ncpu), maxtasksperchild=10) as pool:
        #     _ = list( tqdm(pool.imap(self._process_single_event_3dloc, args), total=self.n_events) )


    def _process_single_event_3dloc(self, args) -> None:
        """Worker function for multiprocessing. Generates posterior samples and writes to HDF5."""

        index, x, y, z, kappa, dobs, sigma = args

        try:
            ###### Sky locations ######
            mu_true = np.array([x, y, z])  # True center
            skymap_center = vonmises_fisher.rvs(mu_true, kappa, size=1)  # Observed center

            # Generate posterior samples
            samples = vonmises_fisher.rvs(skymap_center[0], kappa, size=self.n_posterior_samples)
            _, theta_samples, phi_samples = cartesian2spherical(samples[:, 0], samples[:, 1], samples[:, 2])

            dec_samples = 0.5 * np.pi - theta_samples

            ##### Distances #####
            # Importance resampling of distances
            dtpostsamps = dobs / (1 + sigma * np.random.normal(size=2 * self.n_posterior_samples))
            weights = dtpostsamps / np.sum(dtpostsamps)  # Importance weights proportional to d
            lumdist_samples = np.random.choice(dtpostsamps, size=self.n_posterior_samples, p=weights)

            # Redshift and comoving distance calculations
            redshift_samples = fast_z_at_value(self.cosmo.luminosity_distance, lumdist_samples * u.Mpc)
            comdist_samples = self.cosmo.comoving_distance(redshift_samples).value

            # Write samples to hdf5
            samples_table = Table([phi_samples, dec_samples, lumdist_samples, comdist_samples, redshift_samples], 
                                      names=('ra', 'dec', 'rlum', 'rcom', 'redshift'))
            filename = os.path.join(self.outdir, f"gw_{index:05d}.h5")

            with h5py.File(filename, "a") as f:
                mock_group = f.require_group("mock")  # Takes place of approximant in real GW data
                mock_group.create_dataset('posterior_samples', data=samples_table)

            return None

        except Exception as e:
            sys.exit(f"Error in event {index}: {e}")


    def _select_agn_hosts(self):
        host_idx = np.random.choice(np.arange(len(self.catalog)), self.n_agn_events)
        return host_idx
    

    def _sample_alt_coords(self):
        r, theta, phi = uniform_shell_sampler(0, self.cosmo.comoving_distance(self.max_gw_redshift).value, n_samps=self.n_alt_events)
        return r, theta, phi
    

    def _sample_68p_sky_area(self, low=100, high=10000):
        ''' 68% CL sky localization area uniformly sampled between low and high in deg^2 '''
        sky_area = np.random.uniform(low=low, high=high, size=self.n_events) * (np.pi / 180)**2  # Deg^2 to sr
        return sky_area
    

    @staticmethod
    def _kappa_from_sky_area(sky_area):
        ''' Factor 1.14 empirically seems to work for 68% CL - only approximate needed anyway '''
        kappa = 2 * np.pi * 1.14 / sky_area
        return kappa


if __name__ == '__main__':

    make_nice_plots()

    N_TOT = 1000
    GRID_SIZE = 10  # Radius of the whole grid in redshift
    GW_BOX_SIZE = 2  # Radius of the GW box in redshift
    
    Catalog = MockCatalog(n_agn=N_TOT,
                            max_redshift=GRID_SIZE,
                            gw_box_radius=GW_BOX_SIZE,
                            completeness=1)

    n_events = 5
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
