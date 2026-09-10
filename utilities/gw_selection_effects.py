import h5py
import numpy as np
import pandas as pd
import gc

from scipy.integrate import romb
from scipy.interpolate import interp1d
from scipy.stats import truncnorm

from popsummary import PopulationResult

from darksirenpop.utilities.default_globals import *
from darksirenpop.utilities.redshift_utils import *
from darksirenpop.utilities.priors import BBH_broken_powerlaw_multi_peak_gaussian_m1m2


hyperposterior_path = LVK_HYPERPOSTERIOR_PATH
injections_path = INJECTIONS_PATH

###################################################
# DEFINE THE REFERENCE POPULATION IN MASS AND SPIN
###################################################

result = PopulationResult(fname=hyperposterior_path)
# rate = result.get_hyperparameter_samples(hyperparameters=['rate']).squeeze()  # 1/Gpc^3 1/year

# MAP values of the hyperparameters
df = pd.DataFrame(result.get_hyperparameter_samples(), columns=result.get_metadata("hyperparameters"))
hyperparams = df.iloc[(df.log_likelihood + df.log_prior).idxmax()]
del result
del df
gc.collect()

# Relevant hyperparameters
mass_hyperparam = ['alpha_1', 'alpha_2', 'break_mass', 'mpp_1', 'sigpp_1', 'mpp_2', 'sigpp_2', 'mlow_1', 'delta_m_1', 'lam_0', 'lam_1', 'mmax',
                    'beta', 'mlow_2', 'delta_m_2']
spin_hyperparams = ['mu_chi', 'sigma_chi', 'mu_spin', 'sigma_spin', 'xi_spin']
z_hyperparams = ['z_peak', 'gamma', 'kappa']
all_hyperparams = z_hyperparams + spin_hyperparams + mass_hyperparam


# # Redshift hyperparams
# kappa = hyperparams.loc['kappa']
# gamma = hyperparams.loc['gamma']
# z_peak = hyperparams.loc['z_peak']
# print(gamma, 1+z_peak, kappa)

# MAP values
alpha_1 = hyperparams.loc['alpha_1']
alpha_2 = hyperparams.loc['alpha_2']
break_mass = hyperparams.loc['break_mass']
mpp_1 = hyperparams.loc['mpp_1']
sigpp_1 = hyperparams.loc['sigpp_1']
mpp_2 = hyperparams.loc['mpp_2']
sigpp_2 = hyperparams.loc['sigpp_2']
mlow_1 = hyperparams.loc['mlow_1']
delta_m_1 = hyperparams.loc['delta_m_1']
lam_0 = hyperparams.loc['lam_0']
lam_1 = hyperparams.loc['lam_1']
mmax = hyperparams.loc['mmax']
beta = hyperparams.loc['beta']
mlow_2 = hyperparams.loc['mlow_2']
delta_m_2 = hyperparams.loc['delta_m_2']

b = (break_mass - mlow_1) / (mmax - mlow_1)
lambda_g = 1 - lam_0
lambda_g_0 = lam_1 / lambda_g

masspop_MAP = BBH_broken_powerlaw_multi_peak_gaussian_m1m2(alpha_1=alpha_1,
                                                        alpha_2=alpha_2,
                                                        b=b,   # Calc from break mass: break_point = mminbh + b * (mmaxbh - mminbh)
                                                        beta=beta,
                                                        mminbh=mlow_1,
                                                        mmaxbh=mmax,
                                                        lambda_g=lambda_g,  # Fraction in both peaks
                                                        lambda_g_0=lambda_g_0,  # Fraction in lower peak
                                                        mu_g_0=mpp_1,
                                                        sigma_g_0=sigpp_1,
                                                        mu_g_1=mpp_2,
                                                        sigma_g_1=sigpp_2,
                                                        delta_m=delta_m_1,
                                                        mlow_2=mlow_2,
                                                        delta_m_2=delta_m_2)




def truncated_normal_pdf(x, mu, sigma, low, high):
    """
    PDF of a truncated normal.
    """
    a = (low - mu) / sigma
    b = (high - mu) / sigma
    return truncnorm.pdf(x, a=a, b=b, loc=mu, scale=sigma)


def spin_magnitude_pdf(chi1, chi2, mu_chi, sigma_chi):
    """
    Eq. (B15)
    """
    p1 = truncated_normal_pdf(chi1, mu_chi, sigma_chi, 0.0, 1.0)
    p2 = truncated_normal_pdf(chi2, mu_chi, sigma_chi, 0.0, 1.0)
    return p1 * p2


def spin_tilt_pdf(cost1, cost2, mu_spin, sigma_spin, xi_spin):
    """
    Eq. (B16)
    """
    aligned = (
        truncated_normal_pdf(cost1, mu_spin, sigma_spin, -1.0, 1.0)
        * truncated_normal_pdf(cost2, mu_spin, sigma_spin, -1.0, 1.0)
    )

    isotropic = 0.25

    return xi_spin * aligned + (1.0 - xi_spin) * isotropic


def spin_azimuthal_pdf(phi1, phi2):
    """
    Uniform azimuthal angles:
        phi1, phi2 ~ Uniform(0, 2π)
    """
    return np.ones_like(phi1) / (2 * np.pi) ** 2


def log_joint_spin_pdf(
    phi1,
    phi2,
    chi1,
    chi2,
    cost1,
    cost2,
    mu_chi,
    sigma_chi,
    mu_spin,
    sigma_spin,
    xi_spin,
):
    """
    Joint spin distribution
    p(chi1, chi2, cos1, cos2)
    """
    return (
        np.log(spin_magnitude_pdf(chi1, chi2, mu_chi, sigma_chi))
        + np.log(spin_tilt_pdf(cost1, cost2, mu_spin, sigma_spin, xi_spin) * np.sin(np.arccos(cost1)) * np.sin(np.arccos(cost2)))  # jacobian!
        + np.log(spin_azimuthal_pdf(phi1, phi2))
    )

mu_chi = hyperparams.loc['mu_chi']
sigma_chi = hyperparams.loc['sigma_chi']
mu_spin = hyperparams.loc['mu_spin']
sigma_spin = hyperparams.loc['sigma_spin']
xi_spin = hyperparams.loc['xi_spin']

spinpop_MAP_joint_prob_func = lambda s1phi, s2phi, s1r, s2r, cost1, cost2: log_joint_spin_pdf(
                                                                                        phi1=s1phi, 
                                                                                        phi2=s2phi,
                                                                                        chi1=s1r,
                                                                                        chi2=s2r,
                                                                                        cost1=cost1,
                                                                                        cost2=cost2,
                                                                                        mu_chi=mu_chi,
                                                                                        sigma_chi=sigma_chi,
                                                                                        mu_spin=mu_spin,
                                                                                        sigma_spin=sigma_spin,
                                                                                        xi_spin=xi_spin,
                                                                                    )


###################################################
# SETUP THE REDSHIFT POPULATION MODELS
###################################################

def get_normed_zpop(zpop_unnorm, norm_ax):
    return lambda z: zpop_unnorm(z) / romb(zpop_unnorm(norm_ax), dx=np.diff(norm_ax)[0])


def get_agn_outofcat_pop(lum, agn_dist_dir, zmax):
    filename = f'{agn_dist_dir}/agn_redshift_pdf_{lum}_kulkarni.npy'
    z, n = np.load(filename)
    agndist = interp1d(z, n, bounds_error=False, fill_value=0)
    zpop_unnorm = lambda z: time_dilation_correction(z) * z_cut(z, zcut=zmax) * agndist(z)
    zpop_norm = get_normed_zpop(zpop_unnorm, norm_ax=np.linspace(0, zmax, 1024+1))
    return zpop_norm


def get_alt_pop(zmax=10, rate_model='madau', rate_parameters={}):
    if rate_model == 'madau':
        merger_rate_model = merger_rate_madau_dickinson
        redshift_model_kwargs = rate_parameters
    elif rate_model == 'madau_lvk_MAP':
        merger_rate_model = merger_rate_madau_dickinson
        redshift_model_kwargs = {'b': 3.027636508120894, 'c': 3.1826719462323076, 'd': 7.039509800907652}  # GWTC-5
    elif rate_model == 'lowZ_sfr':
        merger_rate_model = merger_rate_lowZ_sfr
        redshift_model_kwargs = rate_parameters
    
    zpop_unnorm = lambda z: uniform_comoving_prior(z) * time_dilation_correction(z) * merger_rate(z, merger_rate_model, **redshift_model_kwargs) * z_cut(z, zcut=zmax)
    zpop_norm = get_normed_zpop(zpop_unnorm, norm_ax=np.linspace(0, zmax, 1024+1))
    return zpop_norm


def ln_ppop(m1, m2, z, spins, zpop, alt_rate_model, agn_dist_dir, zmax, joint_mass_model, alt_rate_parameters):

    ### Redshift ###
    if callable(zpop):
        z_pop = zpop
    elif zpop == 'alt':
        z_pop = get_alt_pop(zmax=zmax, rate_model=alt_rate_model, rate_parameters=alt_rate_parameters)
    elif zpop.split('_')[0] == 'emptycat':
        z_pop = get_agn_outofcat_pop(lum=zpop.split('_')[1], agn_dist_dir=agn_dist_dir, zmax=zmax)
    else:  # zpop is the filename
        z_temp, p_temp = np.load(zpop)
        zpop_unnorm = interp1d(z_temp, p_temp, bounds_error=False, fill_value=0)
        z_pop = get_normed_zpop(zpop_unnorm, norm_ax=np.linspace(0, zmax, 1024+1))

    ### Masses ###
    m_pop = lambda primary, secondary: joint_mass_model.joint_prob(primary, secondary).detach().cpu().numpy()

    ### Spins ###
    # TODO: add something like + np.log(spinpop_MAP_joint_prob_func(*spins))
    return np.log(z_pop(z)) + np.log(m_pop(m1, m2))


###################################################
# CALCULATE ALPHA WITH INJECTION CAMPAIGN
###################################################

with h5py.File(injections_path, 'r') as obj:
    total_generated = obj.attrs['total_generated']
    total_analysis_time = obj.attrs['total_analysis_time']

    snr_inject = obj['events']['semianalytic_observed_phase_maximized_snr_net'][:]
    far_inject = np.min([obj['events']['%s_far'%search][:] for search in obj.attrs['searches']], axis=0)

    weights = obj['events']['weights'][:]

    mass1_source_inject = obj['events']['mass1_source'][:]
    mass2_source_inject = obj['events']['mass2_source'][:]

    redshift_inject = obj['events']['redshift'][:]

    spin1r = obj['events']['spin1_magnitude'][:]
    spin1theta = obj['events']['spin1_polar_angle'][:]
    spin1phi = obj['events']['spin1_azimuthal_angle'][:]
    spin2r = obj['events']['spin2_magnitude'][:]
    spin2theta = obj['events']['spin2_polar_angle'][:]
    spin2phi = obj['events']['spin2_azimuthal_angle'][:]

    spins_inject = (spin1r, spin1theta, spin1phi, spin2r, spin2theta, spin2phi)

    lnprob = obj['events']['lnpdraw_mass1_source_mass2_source_redshift_spin1_magnitude_spin1_polar_angle_spin1_azimuthal_angle_spin2_magnitude_spin2_polar_angle_spin2_azimuthal_angle'][:]


def get_alpha_alt(snr_thr, far_thr, alt_rate_model, agn_dist_dir, joint_mass_model=masspop_MAP, alt_rate_parameters={}, zmax=10):
    sel = ((snr_inject > snr_thr) | (far_inject < far_thr))
    lnp_alt = ln_ppop(m1=mass1_source_inject, 
                      m2=mass2_source_inject, 
                      z=redshift_inject, 
                      spins=spins_inject, 
                      agn_dist_dir=agn_dist_dir,
                      zpop='alt', 
                      alt_rate_model=alt_rate_model, 
                      joint_mass_model=joint_mass_model, 
                      alt_rate_parameters=alt_rate_parameters, 
                      zmax=zmax)
    alpha_alt = np.sum( weights[sel] * np.exp(lnp_alt[sel] - lnprob[sel]) ) / total_generated
    return alpha_alt


def get_alpha_agn(snr_thr, far_thr, agn_zpop, alt_rate_model, agn_dist_dir, joint_mass_model=masspop_MAP, alt_rate_parameters={}, zmax=10):
    sel = ((snr_inject > snr_thr) | (far_inject < far_thr))
    lnp_agn = ln_ppop(m1=mass1_source_inject, 
                      m2=mass2_source_inject, 
                      z=redshift_inject, 
                      spins=spins_inject, 
                      agn_dist_dir=agn_dist_dir,
                      zpop=agn_zpop, 
                      alt_rate_model=alt_rate_model, 
                      joint_mass_model=joint_mass_model, 
                      alt_rate_parameters=alt_rate_parameters,
                      zmax=zmax)
    alpha_agn = np.sum( weights[sel] * np.exp(lnp_agn[sel] - lnprob[sel]) ) / total_generated
    return alpha_agn


def alpha(fagn, snr_thr, far_thr, agn_zpop, alt_rate_model, agn_dist_dir, joint_mass_model=masspop_MAP, alt_rate_parameters={}, zmax=10):
    alpha_alt = get_alpha_alt(snr_thr=snr_thr, 
                              far_thr=far_thr, 
                              alt_rate_model=alt_rate_model, 
                              agn_dist_dir=agn_dist_dir, 
                              joint_mass_model=joint_mass_model, 
                              alt_rate_parameters=alt_rate_parameters, 
                              zmax=zmax)
    alpha_agn = get_alpha_agn(snr_thr=snr_thr, 
                              far_thr=far_thr, 
                              agn_zpop=agn_zpop, 
                              alt_rate_model=alt_rate_model, 
                              agn_dist_dir=agn_dist_dir, 
                              joint_mass_model=joint_mass_model, 
                              alt_rate_parameters=alt_rate_parameters,
                              zmax=zmax)
    return alpha_alt, alpha_agn, fagn * alpha_agn + (1 - fagn) * alpha_alt
