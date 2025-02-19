import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import sys
import astropy.units as u
from utils import uniform_shell_sampler, spherical2cartesian, fast_z_at_value, make_nice_plots
from scipy.stats import truncnorm


class MockCatalog():
    
    def __init__(
        self,
        max_redshift: float,
        gw_box_radius: float,
        n_bins: int = None,
        n_agn: int = None,
        n_agn_per_shell: list = None,
        shell_radii: list = None,  # TODO: use this to make shell_radii variable, fix some degeneracy with n_bins and n_agn_per_shell
        completeness: float = 1.,
        cosmology = FlatLambdaCDM(H0=67.9, Om0=0.3065)
    ):
        """
        :param max_redshift: redshift of outer boundary of simulated universe
        :param gw_box_radius: redshift within which GWs can be generated
        :param n_bins: number of redshift bins
        :param n_agn_per_shell: number of AGN in each redshift bin
        :param completeness: completeness of the AGN catalog used in crossmatching

        TODO: finish documentation here
        TODO: add functionality to make healpix map from catalog
        TODO: improve method naming with _ in front and stuff
        """
        
        self.max_redshift = max_redshift
        self.gw_box_radius = gw_box_radius
        self.completeness = completeness
        self.cosmo = cosmology
        
        if n_bins == None:
            if n_agn == None:
                sys.exit('No redshift bins specified, therefore assuming a uniform catalog. Please specify the number of AGN in the catalog.')
            else:
                print('No redshift bins specified, therefore assuming a uniform catalog.')
                self.n_agn = n_agn
                self.n_bins = 2  # We treat the GW box and the outer edge as 2 bins
                
                fraction_in_gw_box = (self.cosmo.comoving_distance(self.gw_box_radius) / self.cosmo.comoving_distance(self.max_redshift))**3
                self.n_agn_per_shell = np.around([fraction_in_gw_box * self.n_agn, (1 - fraction_in_gw_box) * self.n_agn]).astype(int)
                self.uniform_flag = True
        else:
            self.n_bins = n_bins
            self.n_agn = np.sum(n_agn_per_shell)
            self.n_agn_per_shell = n_agn_per_shell
            self.uniform_flag = False

        if self.uniform_flag:
            shell_radii_z = np.array([self.gw_box_radius, self.max_redshift])
            self.shell_radii = self.cosmo.comoving_distance(shell_radii_z).value     
        else:
            shell_radii_z = np.linspace(0, self.max_redshift, self.n_bins + 1)[1:]  # TODO: distribute redshift bins such that the enclosed volume is constant
            self.shell_radii = self.cosmo.comoving_distance(shell_radii_z).value

        self.complete_catalog = pd.DataFrame(columns=['r', 'theta', 'phi', 
                                                      'x', 'y', 'z', 
                                                      'redshift', 'redshift_error', 'redshift_meas', 
                                                      'r_meas', 'theta_meas', 'phi_meas', 
                                                      'x_meas', 'y_meas', 'z_meas', 
                                                      'detected', 'in_gw_box'])
        if self.n_agn != 0:
            self.make_complete_catalog()
        else:
            print('Using empty AGN catalog.')

        self.incomplete_catalog = self.complete_catalog.loc[self.complete_catalog['detected'] == True]
    

    def get_true_redshift_and_error(self, comoving_distance):
        redshift_true = fast_z_at_value(self.cosmo.comoving_distance, comoving_distance)
        redshift_error = 0.01 * (1 + redshift_true)**3  # Inspired by https://arxiv.org/pdf/2212.08694 and catches dynamic range in Quaia, though not the actual distribution
        redshift_error[redshift_error > 1] = 1  # TODO: maybe look at relative errors?
        return redshift_true, redshift_error
    

    def measure_redshift(self):
        # redshift_meas = np.random.normal(loc=self.complete_catalog['redshift'], scale=self.complete_catalog['redshift_error'])
        z_low, z_high = 0, self.max_redshift
        a, b = (z_low - self.complete_catalog['redshift']) * self.complete_catalog['redshift_error']**(-1), (z_high - self.complete_catalog['redshift']) * self.complete_catalog['redshift_error']**(-1)
        redshift_meas = truncnorm.rvs(a, b, loc=self.complete_catalog['redshift'], scale=self.complete_catalog['redshift_error'])

        self.complete_catalog['redshift_meas'] = redshift_meas
        self.complete_catalog['r_meas'] = self.cosmo.comoving_distance(redshift_meas).value
        self.complete_catalog['theta_meas'] = self.complete_catalog['theta']  # We assume theta and phi to be perfectly determined
        self.complete_catalog['phi_meas'] = self.complete_catalog['phi']

        x_meas, y_meas, z_meas = spherical2cartesian(self.complete_catalog['r_meas'], self.complete_catalog['theta_meas'], self.complete_catalog['phi_meas'])

        self.complete_catalog['x_meas'] = x_meas
        self.complete_catalog['y_meas'] = y_meas
        self.complete_catalog['z_meas'] = z_meas
        
    
    def distribute_agn_shells(self):
        
        complete_r = np.array([])
        complete_theta = np.array([])
        complete_phi = np.array([])
        for i in range(self.n_bins):
            
            if i == 0:
                r, theta, phi = uniform_shell_sampler(0, self.shell_radii[0], self.n_agn_per_shell[i])
                print(f'Distributing {len(theta)} sources in the bin r = {[0, self.shell_radii[0]]}.')
            else:
                r, theta, phi = uniform_shell_sampler(self.shell_radii[i - 1], self.shell_radii[i], self.n_agn_per_shell[i])
                print(f'Distributing {len(theta)} sources in the bin r = {[self.shell_radii[i - 1], self.shell_radii[i]]}.')

            complete_r = np.append(complete_r, r)
            complete_theta = np.append(complete_theta, theta)
            complete_phi = np.append(complete_phi, phi)
            
        in_gw_box = complete_r < self.cosmo.comoving_distance(self.gw_box_radius).value

        return complete_r, complete_theta, complete_phi, in_gw_box

    
    def sample_incomplete_shell_catalog(self):

        catalog_indeces = np.arange(self.n_agn)
        indeces_in_incomplete_catalog = np.array([])
        
        for i in range(self.n_bins):  # We sample a certain fraction of AGN from each shell according to self.completeness TODO: make completeness a list to get varying completeness with distance
            
            # Make a mask to select the spherical coordinates of the AGN in the current shell.
            # We make use of the fact that the AGN coordinates are ordered from the innermost shell outward
            in_this_shell = np.zeros( self.n_agn, dtype=bool )
            
            if i == 0:
                in_this_shell[:self.n_agn_per_shell[i]] = 1
            else:
                in_this_shell[np.sum(self.n_agn_per_shell[:i]):np.sum(self.n_agn_per_shell[:i]) + self.n_agn_per_shell[i]] = 1

            catalog_in_shell = catalog_indeces[in_this_shell]
            
            # Choose which AGN stay
            incomplete_idx = np.random.choice(np.arange(self.n_agn_per_shell[i]),
                                              int(round(self.completeness * self.n_agn_per_shell[i])),
                                              replace=False)
            
            indeces_in_incomplete_catalog = np.append(indeces_in_incomplete_catalog, catalog_in_shell[incomplete_idx])

        agn_is_detected = np.isin(catalog_indeces, indeces_in_incomplete_catalog)
        return agn_is_detected
    

    def make_complete_catalog(self):
        r, theta, phi, in_gw_box = self.distribute_agn_shells()

        self.complete_catalog['r'] = r
        self.complete_catalog['theta'] = theta
        self.complete_catalog['phi'] = phi

        print('Spherical coordinates added to catalog')

        x, y, z = spherical2cartesian(r, theta, phi)

        self.complete_catalog['x'] = x
        self.complete_catalog['y'] = y
        self.complete_catalog['z'] = z

        print('Cartesian coordinates added to catalog')

        redshift_true, redshift_error = self.get_true_redshift_and_error(r * u.Mpc)

        self.complete_catalog['redshift'] = redshift_true
        self.complete_catalog['redshift_error'] = redshift_error

        print('Redshifts and errors added to catalog')

        self.measure_redshift()  # Adds redshift_meas column to self.complete_catalog

        print('Measured redshifts added to catalog')

        agn_is_detected = self.sample_incomplete_shell_catalog()

        self.complete_catalog['detected'] = agn_is_detected
        self.complete_catalog['in_gw_box'] = in_gw_box

        print('Detected AGN added to catalog')
    

    def plot_catalogs(self, save=None):  # TODO: probably delete this because projection isnt really useful?

        fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection': 'polar'})
        for i, shell_radius in enumerate(self.shell_radii):
            if i == 0:  # Fixing legend
                ax.plot(np.linspace(0, 2 * np.pi, 1000), 
                        np.tile(shell_radius, 1000), 
                        linestyle='dashed', 
                        color='r',
                        label='Redshift bins',
                        alpha=0.3)
            else:
                ax.plot(np.linspace(0, 2 * np.pi, 1000), 
                        np.tile(shell_radius, 1000), 
                        linestyle='dashed', 
                        color='r',
                        alpha=0.3)
        ax.plot(np.linspace(0, 2 * np.pi, 1000), 
                    np.tile(self.cosmo.comoving_distance(self.gw_box_radius).value, 1000), 
                    linestyle='dashed', 
                    color='black',
                    label='GW box')
        ax.scatter(self.complete_catalog['phi'], 
                    self.complete_catalog['r'], 
                    marker='.', 
                    s=1, 
                    color='black',
                    label='Complete AGN catalog')
        ax.scatter(self.complete_catalog.loc[self.complete_catalog['detected'] == True, 'phi'],
                    self.complete_catalog.loc[self.complete_catalog['detected'] == True, 'r'], 
                    marker='.',
                    s=1,
                    color='orange',
                    label='Incomplete AGN catalog')
        ax.set_rmax(self.cosmo.comoving_distance(self.max_redshift).value)
        ax.grid(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title('Radius is comoving distance')
        ax.legend(loc='upper center', 
                    bbox_to_anchor=(0.5, -0.05),
                    fancybox=True, 
                    shadow=True, 
                    ncol=1)
        if save:
            plt.savefig(f'{save}.pdf', bbox_inches='tight')
        plt.show()
    

if __name__ == '__main__':

    make_nice_plots()
    
    N_TOT = 10000
    GRID_SIZE = 5  # Radius of the whole grid in redshift
    GW_BOX_SIZE = 1  # Radius of the GW box in redshift
    
    ShaiHulud = MockCatalog(n_agn=N_TOT,
                            max_redshift=GRID_SIZE,
                            gw_box_radius=GW_BOX_SIZE,
                            completeness=0.2)  # Shai Hulud is the Maker of the Deep Desert...and also this catalog

    ### TESTING COORDINATE DISTRIBUTIONS ###
    catalog = ShaiHulud.complete_catalog
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(catalog['r']**3, bins=30, density=True)
    ax1.set_xlabel(r'$r_{\rm com}^{3}$ [Mpc]')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(np.cos(catalog['theta']), bins=30, density=True)
    ax2.set_xlabel(r'$\cos(\theta)$')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(catalog['phi'], bins=30, density=True)
    ax3.set_xlabel(r'$\phi$ [rad]')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(catalog['redshift'], bins=30, density=True)
    ax4.set_xlabel('Redshift')

    plt.tight_layout()
    # plt.savefig('catalog_proof.pdf')
    plt.show()


    ### TESTING REDSHFIT RESAMPLING FUNCTIONALITY ###

    # def gauss(x, x0, sigma): 
    #     return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - x0)**2 / (2 * sigma**2)) 

    # complete = ShaiHulud.complete_catalog

    # plt.figure(figsize=(8,6))

    # plot_x = np.linspace(complete['redshift'] - 5 * complete['redshift_error'], complete['redshift'] + 5 * complete['redshift_error'], 1000)
    # plt.plot(plot_x, gauss(plot_x, complete['redshift'].iloc[0], complete['redshift_error'].iloc[0]))
    # plt.plot(plot_x, gauss(plot_x, complete['redshift'].iloc[1], complete['redshift_error'].iloc[1]))
    # plt.plot(plot_x, gauss(plot_x, complete['redshift'].iloc[2], complete['redshift_error'].iloc[2]))

    # N = 1000
    # z_arr = np.zeros((N, 3))
    # for i in range(N):
    #     ShaiHulud.measure_redshift()
    #     z_arr[i, 0] = ShaiHulud.complete_catalog['redshift_meas'].iloc[0]
    #     z_arr[i, 1] = ShaiHulud.complete_catalog['redshift_meas'].iloc[1]
    #     z_arr[i, 2] = ShaiHulud.complete_catalog['redshift_meas'].iloc[2]

    # plt.hist(z_arr[:,0], density=True, bins=30)
    # plt.hist(z_arr[:,1], density=True, bins=30)
    # plt.hist(z_arr[:,2], density=True, bins=30)
    # plt.xlabel('Measured z')
    # plt.ylabel('Probability density')
    # plt.show()


    ### TESTING SPHERICAL TO CARTESIAN ###

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter(complete['x'], complete['y'], complete['z'], marker='.')
    # plt.show()
