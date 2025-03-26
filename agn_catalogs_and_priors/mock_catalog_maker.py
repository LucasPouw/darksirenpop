import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from darksirenpop.utils import uniform_shell_sampler, fast_z_at_value, make_nice_plots
import h5py
import healpy as hp


COLUMNS = ['comoving_distance_true', 'luminosity_distance_true', 'ra_true', 'dec_true', 'redshift_true', 
           'comoving_distance', 'luminosity_distance', 'ra', 'dec', 'redshift', 'redshift_error', 'detected']


class MockCatalog:
    
    def __init__(
        self,
        n_agn: int,
        max_redshift: float,
        redshift_error: float=0.05,
        completeness: float = 1.,
        uniform: bool=True,  # In case we ever want to do something non-uniform
        cosmology = FlatLambdaCDM(H0=67.9, Om0=0.3065)
    ):
        """
        max_redshift: redshift of outer boundary of simulated universe
        completeness: completeness of the AGN catalog
        redshift_error: RELATIVE redshift error

        TODO: finish documentation
        """
        
        self.n_agn = n_agn
        self.max_redshift = max_redshift
        self.redshift_error = redshift_error
        self.completeness = completeness
        self.cosmo = cosmology
        self.uniform_flag = uniform

        self.max_rcom = self.cosmo.comoving_distance(self.max_redshift).value  # In Mpc

        self.complete_catalog = pd.DataFrame(columns=COLUMNS)
        if self.n_agn != 0:
            self._make_complete_catalog()
        else:
            print('Using empty AGN catalog.')

        self.incomplete_catalog = self.complete_catalog.loc[self.complete_catalog['detected'] == True]


    @classmethod
    def from_real_catalog(cls):
        NotImplemented


    @classmethod
    def from_file(
            cls, 
            file_path: str
        ):

        """Loads an instance from an HDF5 file"""
        with h5py.File(file_path, "r") as f:
            obj = cls.__new__(cls)

            # Get catalogs
            obj.complete_catalog = pd.DataFrame(columns=COLUMNS)
            for col in COLUMNS:
                obj.complete_catalog[col] = f['complete_catalog'][col][()]
            obj.incomplete_catalog = obj.complete_catalog.loc[obj.complete_catalog['detected'] == True]

            # Get other attributes
            for attr, value in f.attrs.items():
                setattr(obj, attr, value)

            # Get cosmology
            print('WARNING: Assuming FlatLambdaCDM cosmology!')  # TODO: handle cosmology better than this crap
            cosmo_group = f["cosmology"]
            H0 = cosmo_group.attrs['H0']
            Om0 = cosmo_group.attrs['Om0']
            Tcmb0 = cosmo_group.attrs['Tcmb0']
            Neff = cosmo_group.attrs['Neff']
            m_nu = cosmo_group.attrs['m_nu']
            Ob0 = cosmo_group.attrs['Ob0']
            if m_nu == 'None':
                m_nu = None
            if Ob0 == 'None':
                Ob0 = None
            obj.cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, Tcmb0=Tcmb0, Neff=Neff, m_nu=m_nu, Ob0=Ob0)

        return obj


    def write_to_hdf5(self, output_path:str):
        print('\nWriting to HDF5\n')
        cat = self.complete_catalog
        incat = self.incomplete_catalog
        with h5py.File(output_path, "w") as f:

            for col in COLUMNS:
                data = incat[col].to_numpy()
                f.create_dataset(col, data=data)

            # Save the complete catalog 1 layer deeper such that the rest of the code defaults to the incomplete catalog
            complete_cat_group = f.create_group("complete_catalog")
            for col in COLUMNS:
                data = cat[col].to_numpy()
                complete_cat_group.create_dataset(col, data=data)

            # Save attributes
            for attr, value in vars(self).items():
                if attr in ['complete_catalog', 'incomplete_catalog', 'cosmo']:
                    continue
                f.attrs[attr] = value

            # Save cosmological parameters
            print('WARNING: Assuming FlatLambdaCDM cosmology!')  # TODO: handle cosmology better than this crap
            cosmo_group = f.create_group("cosmology")
            cosmo_group.attrs['H0'] = self.cosmo.H0.value
            cosmo_group.attrs['Om0'] = self.cosmo.Om0
            cosmo_group.attrs['Tcmb0'] = self.cosmo.Tcmb0.value
            cosmo_group.attrs['Neff'] = self.cosmo.Neff
            if self.cosmo.m_nu is not None:
                cosmo_group.attrs['m_nu'] = self.cosmo.m_nu.value.tolist()
            else:
                cosmo_group.attrs['m_nu'] = 'None'
            if self.cosmo.Ob0 is not None:
                cosmo_group.attrs['Ob0'] = self.cosmo.Ob0
            else:
                cosmo_group.attrs['Ob0'] = 'None'

    
    def make_norm_map(self, nside, filepath=None):
        print('Making norm map from INCOMPLETE catalog...')
        npix = hp.nside2npix(nside)
        pix_indices = hp.ang2pix(nside, np.pi/2 - self.incomplete_catalog['dec'], self.incomplete_catalog['ra'], nest=True)
        healpix_map = np.zeros(npix)
        np.add.at(healpix_map, pix_indices, 1)  # Count number of AGN in each pixel

        if filepath is not None:
            hp.fitsfunc.write_map(filepath, healpix_map, nest=True, overwrite=True)

        return healpix_map


    def _get_true_and_observed_redshift(self, comoving_distance:float):
        redshift_true = fast_z_at_value(self.cosmo.comoving_distance, comoving_distance)
        redshift_obs = redshift_true * (1. + self.redshift_error * np.random.normal(size=self.n_agn))
        return redshift_true, redshift_obs
        
    
    def _distribute_agn(self):
        if self.uniform_flag:
            complete_r, complete_theta, complete_phi = uniform_shell_sampler(0, self.max_rcom, self.n_agn)
        else:
            sys.exit('Currently only uniform in comoving volume AGN  distribution supported.')
        return complete_r, complete_theta, complete_phi

    
    def _sample_incomplete_catalog(self, selection_function=None):
        '''Currently, all AGN are equally likely to be detected. TODO: make redshift dependent cf. Quaia'''
        complete_indeces = np.arange(self.n_agn)
        if selection_function is None:
            incomplete_idx = np.random.choice(complete_indeces,
                                              int(round(self.completeness * self.n_agn)),
                                              replace=False)

        agn_is_detected = np.isin(complete_indeces, incomplete_idx)
        return agn_is_detected
    

    def _make_complete_catalog(self):
        r_true, theta_true, phi_true = self._distribute_agn()
        self.complete_catalog['comoving_distance_true'] = r_true
        self.complete_catalog['ra_true'] = phi_true
        self.complete_catalog['dec_true'] = 0.5*np.pi - theta_true  # From colatitude [0, pi] to dec [-pi/2, pi/2]

        redshift_true, redshift_obs = self._get_true_and_observed_redshift(r_true * u.Mpc)
        self.complete_catalog['redshift_true'] = redshift_true
        self.complete_catalog['luminosity_distance_true'] = self.cosmo.luminosity_distance(redshift_true).value
        print('Truths added to catalog')

        self.complete_catalog['redshift'] = redshift_obs
        self.complete_catalog['comoving_distance'] = self.cosmo.comoving_distance(redshift_obs).value
        self.complete_catalog['luminosity_distance'] = self.cosmo.luminosity_distance(redshift_obs).value
        # We assume RA and Dec to be perfectly measured
        self.complete_catalog['ra'] = phi_true
        self.complete_catalog['dec'] = 0.5*np.pi - theta_true
        print('Measurements added to catalog')

        self.complete_catalog['redshift_error'] = np.tile(self.redshift_error, self.n_agn)

        agn_is_detected = self._sample_incomplete_catalog()
        self.complete_catalog['detected'] = agn_is_detected
        print('Detections added to catalog')
    

if __name__ == '__main__':
    import os

    make_nice_plots()
    
    N_TOT = 100000
    MAX_REDSHIFT = 3  # Max AGN true redshift
    N_COARSE = 32
    SIGMA = 0.01

    ShaiHulud = MockCatalog(n_agn=N_TOT, 
                            max_redshift=MAX_REDSHIFT,
                            redshift_error=SIGMA)  # Shai Hulud is the Maker of the Deep Desert...and also this catalog

    fname = f'NAGN_{N_TOT}_ZMAX_{MAX_REDSHIFT}_SIGMA_{SIGMA}'
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
    output_path = os.path.join(script_dir, "../output")
    output_path = os.path.abspath(output_path)  # Normalize path to resolve '..'
    
    ShaiHulud.write_to_hdf5(output_path + f'/catalogs/mockcat_{fname}.hdf5')
    _ = ShaiHulud.make_norm_map(nside=N_COARSE, filepath=output_path + f'/maps/mocknorm_{fname}.fits')


    ### TESTING COORDINATE DISTRIBUTIONS ###
    catalog = ShaiHulud.complete_catalog
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(catalog['comoving_distance']**3, bins=30, density=True)
    ax1.set_xlabel(r'$r_{\rm com}^{3}$ [Mpc]')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(np.cos(0.5*np.pi - catalog['dec']), bins=30, density=True)
    ax2.set_xlabel(r'$\cos(\delta)$')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(catalog['ra'], bins=30, density=True)
    ax3.set_xlabel(r'RA [rad]')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(catalog['redshift'], bins=30, density=True)
    ax4.set_xlabel('Redshift')

    plt.tight_layout()
    plt.savefig('catalog_proof.pdf')
    plt.show()

