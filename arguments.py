"""
Module for holding all gwcosmo command line arguments
DRY: Don't Repeat Yourself

Rachel Gray
"""

import argparse

class argument:
    def __init__(self, long_flag=None, default=None, metavar=None, type=None, help=None):
        self.long_flag = long_flag
        self.optional_arguments = {
            'default': default,
            'type': type,
            'metavar': metavar,
            'help': help,
        }

def create_parser(*config):
    parser = argparse.ArgumentParser()

    # Each new argument requires only one new line, DRY!
    arguments_dictionary = {
        "--method": argument(None, default=None,
            help="Analysis method, choose from 'gridded' or 'sampling' (required)"),
        "--posterior_samples": argument(None, default=None,
            help="Path to LALinference posterior samples file in format (.dat or hdf5)"),
        "--posterior_samples_field": argument(None, default=None,
            help="Internal field of the posterior samples file, e.g. h5 or json field"),
        "--skymap": argument(None, default=None,
            help="Path to LALinference 3D skymap file in format (.fits or fits.gz)"),
        
        "--reweight_posterior_samples": argument(None, default='True',
            help="Reweight posterior samples with the same priors used to calculate the selection effects (default=True)."),
        "--zmax": argument(None, default=10.0, type=float,
            help="Upper redshift limit for integrals (default=10)"),
        "--zmin": argument(None, default=1e-10, type=float,
            help="Lower redshift limit for integrals (default=1e-10)"),
        "--zdraw": argument(None, default=2, type=float,
            help="Highest possible true redshift of GW events (default=2)."),
        
        "--H0": argument(None, default=67.9, type=float,
            help="Hubble constant value in cosmology (default=67.9)"),
        "--Omega_m": argument(None, default=0.3065, type=float,
            help="Omega matter value in cosmology (default=0.3065)"),
        
        "--nside": argument(None, default=32, type=int,
            help="healpix nside choice for constructing the line-of-sight redshift prior (default=32)"),
        "--coarse_nside": argument(None, default=32, type=int,
            help="coarse healpix nside choice for building the mth and galaxy_norm maps (default=32)"),
        "--maps_path": argument(None, default=None, type=str,
            help="path to store or look for mth and mth maps (default=None)"),
        "--sky_area": argument(None, default=0.999, type=float,
            help="contour boundary for galaxy catalogue method (default=0.999)"),
        "--pixel_index": argument(None, default=None, type=int,
            help="index of the healpix pixel to analyse"),
        "--min_pixels": argument(None, default=30, type=int,
            help="minimum number of pixels desired to cover sky area of event (for use with pixel method only), (default=30)"),
        "--outputfile": argument(None, default='Posterior',
            help="Name of output file (default='Posterior')"),
        "--seed": argument(None, default=None, type=int, 
            help="Random seed (default=None)"),
        "--catalog_name": argument(None, default=None, metavar="NAME",
            help="""Specify a galaxy catalog by name. Known catalogs are: MOCK, QUAIA"""),

        "--catalog_path": argument(None, default=None, type=str,
            help="path to look for AGN catalog (default=None)"),
        
        "--min_gals_for_threshold": argument(None, default=10, type=int,
            help="The minimum number of galaxies in a coarse pixel for it not to be considered empty (default=10)."),
        "--LOS_catalog": argument(None, default=None,
            help="""The pre-computed line-of-sight redshift prior in hdf5 format"""),
        "--cpus": argument(None, default=1, type=int,
            help="Number of cpus asked for each run (default=1)"),
        "--ram": argument(None, default=1000, type=int,
            help="RAM asked for each run (default=1000 MB)"),
        "--run_on_ligo_cluster": argument(None, default='False', type=str,
            help="Set to true if running on a LIGO cluster (default=False)"),
        "--parameter_dict": argument(None, default=None,
            help="dictionary of parameter set up"),
        "--plot": argument(None, default='True',
            help="Create plots (default=True)"),
        "--sampler": argument(None,
            help="Choice of sampler (choose from Dynesty, Emcee)"),
        "--nwalkers": argument(None, default=100,
            help="Number of walkers for MCMC (default=100)"),
        "--walks": argument(None, default=10,
            help="Minimum number of steps before proposing new live point for nested sampling (10-20 dimensions) (default=10)"),
        "--npool": argument(None, default=2,
            help="Number of pools to use with nested sampling (default=2)"),
        "--nsteps": argument(None, default=1000,
            help="Number of steps for MCMC (default=1000)"),
        "--nlive": argument(None, default=1000,
            help="Number of live points for nested sampling"),
        "--dlogz": argument(None, default=0.1,
            help="Stopping criterion for nested sampling"),
        "--injections_path": argument(None, default=None,type=str,
            help="Path to the injetions file."),
        "--mass_model": argument(None, default=None,type=str,
            help="Mass model. Choose from 'BBH-powerlaw', 'BBH-broken-powerlaw', 'BBH-powerlaw-gaussian', 'NSBH-powerlaw', 'NSBH-broken-powerlaw', 'NSBH-powerlaw-gaussian' or 'BNS'."),
        "--run": argument(None, default=1, type=int,
            help="The number of the run."),
        "--detectors": argument(None, default='HLV', type=str,
            help="The detectors to be cosnidered for the injections (default='HLV')."),
        "--asd_path": argument(None, default='None', type=str,
            help="The path to the asd files' folder. They should be named as detector_run_strain.txt like H1_O1_strain.txt."),
        "--psd": argument(None, default='None', type=str,
            help="Fix to a certain psd: O1, O2, O3 or O4. Default is None which will marginalize over detector sensitivities."),
        "--priors_file": argument(None, default='None', type=str,
            help="Set the path in case of a custom prior file (bilby format). If not the code will use the default priors."),
        "--Nsamps": argument(None, default=100, type=int,
            help="Number of detected events."),
        "--days_of_O1": argument(None, default=129, type=float,
            help="Total days of O1 run (default=129)."),
        "--days_of_O2": argument(None, default=268, type=float,
            help="Total days of O2 run (default=268)."),
        "--days_of_O3": argument(None, default=330, type=float,
            help="Total days of O3 run (default=330)."),
        "--days_of_O4": argument(None, default=330, type=float,
            help="Total days of O4 run (default=330)."),
        "--O4sensitivity": argument(None, default='low', type=str,
            help="Optimistic (high) or pessimistic (low) sensitivity for O4 (default='low')."),
        "--duty_factor_O4_H1": argument(None, default=0.75, type=float,
            help="The duty factor of H1 for the O4 run (default=0.75)."),
        "--duty_factor_O4_L1": argument(None, default=0.75, type=float,
            help="The duty factor of L1 for the O4 run (default=0.75)."),
        "--duty_factor_O4_V1": argument(None, default=0.75, type=float,
            help="The duty factor of V1 for the O4 run (default=0.75)."),
        "--duty_factor_O3_H1": argument(None, default=0.7459, type=float,
            help="The duty factor of H1 for the O3 run (default=0.7459)."),
        "--duty_factor_O3_L1": argument(None, default=0.7698, type=float,
            help="The duty factor of L1 for the O3 run (default=0.7698)."),
        "--duty_factor_O3_V1": argument(None, default=0.7597, type=float,
            help="The duty factor of V1 for the O3 run (default=0.7597)."),
        "--duty_factor_O2_H1": argument(None, default=0.653, type=float,
            help="The duty factor of H1 for the O2 run (default=0.653)."),
        "--duty_factor_O2_L1": argument(None, default=0.618, type=float,
            help="The duty factor of L1 for the O2 run (default=0.618)."),
        "--duty_factor_O2_V1": argument(None, default=0.0777, type=float,
            help="The duty factor of V1 for the O2 run (default=0.0777)."),
        "--duty_factor_O1_H1": argument(None, default=0.646, type=float,
            help="The duty factor of H1 for the O1 run (default=0.646)."),
        "--duty_factor_O1_L1": argument(None, default=0.574, type=float,
            help="The duty factor of L1 for the O1 run (default=0.574)."),
        "--frame": argument(None, default='detectors_frame', type=str,
            help="The frame of the injections(source or detectors) (default is 'detectors_frame')."),
        "--num_threads": argument(None, default=1, type=int,
            help="Number of threads (default is 1)."),            
        "--snr": argument(None, default=9, type=float,
            help="The SNR threshold to claim detection of events (default is 9)."),
        "--fmin": argument(None, default=20, type=float,
            help="The minimum frequency of the waveforms in Hz (default is 20)."),
        "--sampling_frequency": argument(None, default=4096, type=int,
            help="The sampling frequency of the waveforms in Hz (default is 4096)."),
        "--approximant": argument(None, default='IMRPhenomPv2', type=str,
            help="The waveform approximant to be used for the waveform calculation (default is 'IMRPhenomPv2')."),
        "--output_dir": argument(None, default='./injection_files', type=str,
            help="The name of the output folder (default is './injection_files')."),
        "--offset": argument(None, default=1., type=float,
            help="Offset added to redshift priors for saving (default is 1.)"),
        "--tmp_to_dict": argument(None, default=None, type=str,
            help="for injections: converts a tmp file (list format) into a dict format"),
        "--tmp_to_stdout": argument(None, default=None, type=str,
            help="for injections: dump a tmp file (list format) to stdout for control"),
        "--Tobs": argument(None, default=None, 
            help="The total observational time in years."),    
        "--combine": argument(None, default='False', type=str,
            help="Set to True if combining different injections files to one."),
        "--output_combine": argument(None, default=None,
            help="The name of the output file."),
        "--path_combine": argument(None, default='./injection_files', type=str,
            help="The name of the files' folder in case of combining (default is './injection_files'."),
        "--merge_tmpfile_list": argument(None, default=None, type=str,
            help="Combine all files listed in the filename into a unique dict file for injections."),
        "--dLmax_depends_on_m1": argument(None, default=1, type=int,
            help="Uses a max value of luminosity distance depending on m1 (defaut=1) ONLY for SNR in {9, 10, 11, 12}!!!!"),
        
        "--disk": argument(None, default=5000, type=int,
            help="Disk asked for each run (default=5000 MB)"),
        "--search_tag": argument(None, default='ligo.dev.o4.cbc.hubble.gwcosmo', type=str,
            help="Search tag for the runs -- used in LIGO clusters (default=ligo.dev.o4.cbc.hubble.gwcosmo)"),
        "--nruns":argument(None, default=1, type=int,
            help="The total number of runs (default is 1)."),        
            
        "--gravity_model":argument(None, default='GR', type=str,
            help="The gravity model to analyse. Choose between 'GR' and 'Xi0_n' (default='GR')."),

        "--counterpart_dictionary": argument(None, default=None,
            help="Counterpart information for multiple bright sirens"),
        "--post_los": argument(None, default='True', type=None,
            help="posterior samples are conditioned on line-of-sight (default is True); can handle bool and dictionary"),
        "--nsamps": argument(None, default=1000, type=int,
            help="number of samples from posterior, not conditioned over line-of-sight (default is 1000)"),
        "--skymap_prior_distance": argument(None, default="dlSquare", type=str,
            help="Distance prior used when generating the GW skymap. Choose from 'dlSquare', 'Uniform' or 'UniformComoving' (default dlSquare)"), 
        "--skymap_H0": argument(None, default=67.9, type=float,
            help="H0 when distance prior of skymap is uniform in comoving volume (default is 67.9)"), 
        "--skymap_Omega_m": argument(None, default=0.3065, type=float,
            help="Omega_m when distance prior of skymap is uniform in comoving volume (default is 0.3065)"),

        "--agn_mass_model": argument(None, default=None,type=str,
            help="Mass model for the host-in-AGNcatalog-hypothesis. Choose from 'BBH-powerlaw', 'BBH-broken-powerlaw', 'BBH-powerlaw-gaussian', 'NSBH-powerlaw', 'NSBH-broken-powerlaw', 'NSBH-powerlaw-gaussian' or 'BNS'."),
        "--alt_mass_model": argument(None, default=None,type=str,
            help="Mass model for the host-not-in-AGNcatalog-hypothesis. Choose from 'BBH-powerlaw', 'BBH-broken-powerlaw', 'BBH-powerlaw-gaussian', 'NSBH-powerlaw', 'NSBH-broken-powerlaw', 'NSBH-powerlaw-gaussian' or 'BNS'.")
    }

    for arg in config:
        config_instance = arguments_dictionary[arg]

        positional_arguments = [arg]
        if config_instance.long_flag:
            positional_arguments.append(config_instance.long_flag)

        parser.add_argument(*positional_arguments, **config_instance.optional_arguments)

    return parser

#### Full list of command line arguments (for ease of defining config in command line scripts)  TODO: update -Lucas
"""
"--method", "--posterior_samples", "--posterior_samples_field", "--skymap", "--counterpart_ra", "--counterpart_dec", "--counterpart_z", "--counterpart_sigmaz", "--counterpart_v", "--counterpart_sigmav", "--redshift_evolution", "--Kcorrections", "--reweight_posterior_samples", "--zmax", "--galaxy_weighting", "--assume_complete_catalog", "--zcut", "--mth", "--schech_alpha", "--schech_Mstar", "--schech_Mmin", "--schech_Mmax", "--H0", "--Omega_m", "--w0", "--wa", "--nside", "--coarse_nside", "--maps_path", "--sky_area", "--pixel_index", "--min_pixels", "--outputfile", "--seed", "--catalog", "--catalog_band", "--min_gals_for_threshold", "--LOS_catalog", "--cpus", "--ram", "--run_on_ligo_cluster", "--parameter_dict", "--plot", "--sampler", "--nwalkers", "--walks", "--npool", "--nsteps", "--nlive", "--dlogz", "--injections_path", "--mass_model", "--run", "--detectors", "--asd_path", "--psd", "--priors_file", "--Nsamps", "--days_of_O1", "--days_of_O2", "--days_of_O3", "--days_of_O4", "--O4sensitivity", "--duty_factor_O4_H1", "--duty_factor_O4_L1", "--duty_factor_O4_V1", "--duty_factor_O3_H1", "--duty_factor_O3_L1", "--duty_factor_O3_V1", "--duty_factor_O2_H1", "--duty_factor_O2_L1", "--duty_factor_O2_V1", "--duty_factor_O1_H1", "--duty_factor_O1_L1", "--frame", "--num_threads", "--snr", "--fmin", "--sampling_frequency", "--approximant", "--output_dir", "--offset", "--tmp_to_dict", "--tmp_to_stdout", "--Tobs", "--combine", "--output_combine", "--path_combine", "--merge_tmpfile_list", "--dLmax_depends_on_m1", "--disk", "--search_tag", "--nruns", "--gravity_model", "--counterpart_dictionary", "--post_los", "--nsamps", "--skymap_prior_distance", "--skymap_H0", "--skymap_Omega_m"
"""
