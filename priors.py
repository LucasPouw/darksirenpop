"""
Population priors
Ignacio Magana, Rachel Gray, Sergio Vallejo-Peña, Antonio Enea Romano
Lucas Pouw

From gwcosmo by Rachel Gray et al.: https://git.ligo.org/lscsoft/gwcosmo
"""
from __future__ import absolute_import
import numpy as np
from scipy.interpolate import interp1d
import custom_math_priors as _cmp


class PrimaryPrior(object):
    """
    Parent class with common methods for managing the priors on the source frame primary mass.
    The prior is :math:`p(m_1)`
    """

    def __init__(self):
        self.param_label = ['mass_1']  # Used by MockEvent class

    def update_parameters(self, param_dict):
        """
        Update values of mass priors parameters.
        For more details about functions/parameter definitions see 2111.03604
        """
        #Minimum mass of the PL component of the black hole mass distribution
        if 'mminbh' in param_dict:
            self.mminbh = param_dict['mminbh']
        #Maximum mass of the PL component of the black hole mass distribution
        if 'mmaxbh' in param_dict:
            self.mmaxbh = param_dict['mmaxbh']
        #Spectral index for the PL of the primary mass distribution
        if 'alpha' in param_dict:
            self.alpha = param_dict['alpha']
        #Mean of the Gaussian component in the primary mass distribution
        if 'mu_g' in param_dict:
            self.mu_g = param_dict['mu_g']
        #Width of the Gaussian component in the primary mass distribution
        if 'sigma_g' in param_dict:
            self.sigma_g = param_dict['sigma_g']
        #Fraction of the model in the Gaussian component
        if 'lambda_peak' in param_dict:
            self.lambda_peak = param_dict['lambda_peak']
        #PL slope of the primary mass distribution for masses below mbreak
        if 'alpha_1' in param_dict:
            self.alpha_1 = param_dict['alpha_1']
        #PL slope for the primary mass distribution for masses above mbreak
        if 'alpha_2' in param_dict:
            self.alpha_2 = param_dict['alpha_2']
        #The fraction of the way between mminbh and mmaxbh at which the primary mass distribution breaks
        if 'b' in param_dict:
            self.b = param_dict['b']
        #Range of mass tapering on the lower end of the mass distribution
        if 'delta_m' in param_dict:
            self.delta_m = param_dict['delta_m']
        self.update_mass_priors()

    def joint_prob(self, ms1):
        """
        This method returns the probability :math:`p(m_1)`
        Name is joint_prob for consistency with the rest of the code

        Parameters
        ----------
        ms1: np.array(matrix)
            mass one in solar masses
        ms2: dict
            mass two in solar masses
        """

        to_ret = self.mdis['mass_1'].prob(ms1)
        return to_ret
    
    def log_joint_prob(self, ms1):
        
        to_ret = np.log(self.joint_prob(ms1))
        to_ret[np.isnan(to_ret)] = -np.inf

        return to_ret

    def sample(self, Nsample):
        """
        *Not used in O4, due to the use of injections instead of Pdet*
        This method samples from the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        Nsample: int
            Number of samples you want
        """

        vals_m1 = np.random.rand(Nsample)
        m1_trials = np.logspace(np.log10(self.mdis['mass_1'].minimum), np.log10(self.mdis['mass_1'].maximum), 10000)
        cdf_m1_trials = self.mdis['mass_1'].cdf(m1_trials)
        m1_trials = np.log10(m1_trials)

        # At very high or very low values of x, the CDF evaluates to 0 or 1 for all x. We only want a 0 or 1 once.
        _, indxm1 = np.unique(cdf_m1_trials, return_index=True)
        interpo_icdf_m1 = interp1d(cdf_m1_trials[indxm1], m1_trials[indxm1], bounds_error=False, fill_value=(m1_trials[0], m1_trials[-1]))
        mass_1_samples = 10**interpo_icdf_m1(vals_m1)
        return (mass_1_samples,)


class PrimaryMass_gaussian(PrimaryPrior):

    def __init__(self, mminbh=6.0, mu_g=32.27, sigma_g=3.88):
        super().__init__()

        self.update_parameters(param_dict={'mminbh':mminbh, 'mu_g':mu_g, 'sigma_g':sigma_g})               
                
    def update_mass_priors(self):

        '''These methods are only needed for calculating joint_prob, which should subsequently used in the likelihood'''

        # self.mmax = self.mmaxbh #Maximum value of m1, used in injections.Injections.update_VT (m_prior.mmax)
        # self.mmin = self.mminbh #Minimum value of m2, used in self.joint_prob and in injections.Injections.update_VT (m_prior.mmin) 

        self.mdis={'mass_1':_cmp.Truncated_Gaussian_math(mu=self.mu_g, sigma=self.sigma_g, min_g=self.mminbh, max_g=self.mu_g + 5 * self.sigma_g)}  # TODO: implement smoothing?



class m_priors(object):
    """
    Parent class with common methods for managing the priors on source frame masses.
    The prior is factorized as :math:`p(m_1,m_2) \\propto p(m_1)p(m_2|m_1)`
    """

    def __init__(self):
        self.param_label = ['mass_1', 'mass_2']  # Used by MockEvent class

    def update_parameters(self, param_dict):
        """
        Update values of mass priors parameters.
        For more details about functions/parameter definitions see 2111.03604
        """
        #Minimum mass of the PL component of the black hole mass distribution
        if 'mminbh' in param_dict:
            self.mminbh = param_dict['mminbh']
        #Maximum mass of the PL component of the black hole mass distribution
        if 'mmaxbh' in param_dict:
            self.mmaxbh = param_dict['mmaxbh']
        #Spectral index for the PL of the primary mass distribution
        if 'alpha' in param_dict:
            self.alpha = param_dict['alpha']
        #Mean of the Gaussian component in the primary mass distribution
        if 'mu_g' in param_dict:
            self.mu_g = param_dict['mu_g']
        #Width of the Gaussian component in the primary mass distribution
        if 'sigma_g' in param_dict:
            self.sigma_g = param_dict['sigma_g']
        #Fraction of the model in the Gaussian component
        if 'lambda_peak' in param_dict:
            self.lambda_peak = param_dict['lambda_peak']
        #PL slope of the primary mass distribution for masses below mbreak
        if 'alpha_1' in param_dict:
            self.alpha_1 = param_dict['alpha_1']
        #PL slope for the primary mass distribution for masses above mbreak
        if 'alpha_2' in param_dict:
            self.alpha_2 = param_dict['alpha_2']
        #The fraction of the way between mminbh and mmaxbh at which the primary mass distribution breaks
        if 'b' in param_dict:
            self.b = param_dict['b']
        #Range of mass tapering on the lower end of the mass distribution
        if 'delta_m' in param_dict:
            self.delta_m = param_dict['delta_m']
        #Spectral index for the PL of the mass ratio distribution
        if 'beta' in param_dict:
            self.beta = param_dict['beta']
        self.update_mass_priors()

    def joint_prob(self, ms1, ms2):
        """
        This method returns the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        ms1: np.array(matrix)
            mass one in solar masses
        ms2: dict
            mass two in solar masses
        """

        to_ret = self.mdis['mass_1'].prob(ms1) * self.mdis['mass_2'].conditioned_prob(ms2, self.mmin * np.ones_like(ms1), np.minimum(ms1, self.mmax2))
        return to_ret
    
    def log_joint_prob(self, ms1, ms2):
        
        to_ret = np.log(self.joint_prob(ms1, ms2))
        to_ret[np.isnan(to_ret)] = -np.inf

        return to_ret

    def sample(self, Nsample):
        """
        *Not used in O4, due to the use of injections instead of Pdet*
        This method samples from the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        Nsample: int
            Number of samples you want
        """

        vals_m1 = np.random.rand(Nsample)
        vals_m2 = np.random.rand(Nsample)

        m1_trials = np.logspace(np.log10(self.mdis['mass_1'].minimum), np.log10(self.mdis['mass_1'].maximum), 10000)
        m2_trials = np.logspace(np.log10(self.mdis['mass_2'].minimum), np.log10(self.mdis['mass_2'].maximum), 10000)

        cdf_m1_trials = self.mdis['mass_1'].cdf(m1_trials)
        cdf_m2_trials = self.mdis['mass_2'].cdf(m2_trials)

        m1_trials = np.log10(m1_trials)
        m2_trials = np.log10(m2_trials)

        # At very high or very low values of x, the CDF evaluates to 0 or 1 for all x. We only want a 0 or 1 once.
        _,indxm1 = np.unique(cdf_m1_trials,return_index=True)
        _,indxm2 = np.unique(cdf_m2_trials,return_index=True)

        interpo_icdf_m1 = interp1d(cdf_m1_trials[indxm1], m1_trials[indxm1], bounds_error=False, fill_value=(m1_trials[0], m1_trials[-1]))
        interpo_icdf_m2 = interp1d(cdf_m2_trials[indxm2], m2_trials[indxm2], bounds_error=False, fill_value=(m2_trials[0], m2_trials[-1]))

        mass_1_samples = 10**interpo_icdf_m1(vals_m1)
        mass_2_samples = 10**interpo_icdf_m2(vals_m2 * self.mdis['mass_2'].cdf(mass_1_samples))

        return mass_1_samples, mass_2_samples
    

class BBH_powerlaw(m_priors):
    """
    Child class for BBH power law distribution.
    
    Parameters
    -------------
    mminbh: Minimum mass of the PL component of the black hole mass distribution
    mmaxbh: Maximum mass of the PL component of the black hole mass distribution
    alpha: Spectral index for the PL of the primary mass distribution
    beta: Spectral index for the PL of the mass ratio distribution

    The default values of the parameters are set to the corresponding median values in the uniform priors reported in 2111.03604

    ************
    NOTE: The spectral indices passed to PowerLaw_math are alpha=-self.alpha, and alpha=self.beta, according to eqs. A8,A10 in 2111.03604
    ************
    
    The method m_priors.update_parameters is used  in the constructor to initialize the objects
    """
    def __init__(self, mminbh=6.0, mmaxbh=125.0, alpha=6.75, beta=4.0):
        super().__init__()

        self.update_parameters(param_dict={'alpha':alpha, 'beta':beta, 'mminbh':mminbh, 'mmaxbh':mmaxbh})               
              
    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects. 
        It sets the maximum value of the primary mass distribution mmax to mmaxbh, 
        the minimum value of the secondary mass distribution mmin to mminbh, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxbh.
        It's called by update_paratemters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''

        self.mmax = self.mmaxbh #Maximum value of m1, used in injections.Injections.update_VT (m_prior.mmax)
        self.mmin = self.mminbh #Minimum value of m2, used in self.joint_prob and in injections.Injections.update_VT (m_prior.mmin) 
        self.mmax2 = self.mmaxbh #Maximum value of m2, used in self.joint_prob

        self.mdis={'mass_1':_cmp.PowerLaw_math(alpha=-self.alpha, min_pl=self.mminbh, max_pl=self.mmaxbh),
                     'mass_2':_cmp.PowerLaw_math(alpha=self.beta, min_pl=self.mminbh, max_pl=self.mmaxbh)}


class BBH_powerlaw_gaussian(m_priors):
    """
    Child class for BBH power law gaussian distribution.
    
    Parameters
    -------------
    mminbh: Minimum mass of the PL component of the black hole mass distribution
    mmaxbh: Maximum mass of the PL component of the black hole mass distribution
    alpha: Spectral index for the PL of the primary mass distribution    
    mu_g: Mean of the Gaussian component in the primary mass distribution
    sigma_g: Width of the Gaussian component in the primary mass distribution
    lambda_peak: Fraction of the model in the Gaussian component
    delta_m: Range of mass tapering on the lower end of the mass distribution
    beta: Spectral index for the PL of the mass ratio distribution

    The default values of the parameters are set to the corresponding values reported in section 4.2 (page 23) in 2111.03604

    ************
    NOTE: The spectral indices passed to PowerLawGaussian_math, and PowerLaw_math, are alpha=-self.alpha, and alpha=self.beta, according to eqs. A8,A11 in 2111.03604
    *************   

    The method m_priors.update_parameters is used in the constructor to initialize the objects.
    """
    def __init__(self, mminbh=4.98, mmaxbh=112.5, alpha=3.78, mu_g=32.27, sigma_g=3.88, lambda_peak=0.03, delta_m=4.8, beta=0.81):
        super().__init__()

        self.update_parameters(param_dict={'alpha':alpha, 'beta':beta, 'mminbh':mminbh, 'mmaxbh':mmaxbh, 'mu_g':mu_g, 'sigma_g':sigma_g, 'lambda_peak':lambda_peak, 'delta_m':delta_m})

    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to self.mdis['mass_1'].maximum, 
        the minimum value of the secondary mass distribution mmin to mminbh, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxbh.
        It's called by update_paratemters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''
                       
        self.m1pr = _cmp.PowerLawGaussian_math(alpha=-self.alpha, min_pl=self.mminbh, max_pl=self.mmaxbh, lambda_g=self.lambda_peak,
                                               mean_g=self.mu_g, sigma_g=self.sigma_g, min_g=self.mminbh, max_g=self.mu_g + 5 * self.sigma_g)

        # The max of the secondary mass is adapted to the primary mass maximum which is desided by the Gaussian and PL
        self.m2pr = _cmp.PowerLaw_math(alpha=self.beta, min_pl=self.mminbh, max_pl=np.max([self.mu_g + 5 * self.sigma_g, self.mmaxbh]))

        self.mdis={'mass_1': _cmp.SmoothedProb(origin_prob=self.m1pr, bottom=self.mminbh, bottom_smooth=self.delta_m),
                      'mass_2':_cmp.SmoothedProb(origin_prob=self.m2pr, bottom=self.mminbh, bottom_smooth=self.delta_m)}
       
        # TO DO Add a check on the mu_g - 5 sigma of the gaussian to not overlap with mmin, print a warning
        #if (mu_g - 5*sigma_g)<=mmin:
        #print('Warning, your mean (minuse 5 sigma) of the gaussian component is too close to the minimum mass')

        self.mmax = self.mdis['mass_1'].maximum 
        self.mmin = self.mminbh  
        self.mmax2 = self.mmaxbh


class BBH_broken_powerlaw(m_priors):
    """
    Child class for BBH broken power law distribution.

    Parameters
    -------------
    mminbh: Minimum mass of the PL component of the black hole mass distribution
    mmaxbh: Maximum mass of the PL component of the black hole mass distribution
    alpha_1: PL slope of the primary mass distribution for masses below mbreak 
    alpha_2: PL slope for the primary mass distribution for masses above mbreak 
    b: The fraction of the way between mminbh and mmaxbh at which the primary mass distribution breaks
    delta_m: Range of mass tapering on the lower end of the mass distribution
    beta: Spectral index for the PL of the mass ratio distribution

    The default values of the parameters are set to the corresponding median values in the uniform priors reported in 2111.03604
    
    ************
    NOTE: The spectral indices passed to BrokenPowerLaw_math, and PowerLaw_math, are alpha_1=-self.alpha_1, alpha_2=-self.alpha_2, and alpha=self.beta, according to eqs. A8,A12 in 2111.03604
    ************

    The method m_priors.update_parameters is used in the constructor to initialize the objects.
    """
    def __init__(self, mminbh=26, mmaxbh=125, alpha_1=6.75, alpha_2=6.75, b=0.5, delta_m=5, beta=4):
        super().__init__()

        self.update_parameters(param_dict={'alpha_1':alpha_1, 'alpha_2':alpha_2, 'beta':beta, 'mminbh':mminbh, 'mmaxbh':mmaxbh, 'b':b, 'delta_m':delta_m})

    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects. 
        It sets the maximum value of the primary mass distribution mmax to mmaxbh, 
        the minimum value of the secondary mass distribution mmin to mminbh, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxbh.
        It's called by update_paratemters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''

        self.mmax = self.mmaxbh 
        self.mmin = self.mminbh  
        self.mmax2 = self.mmaxbh
                
        self.m1pr = _cmp.BrokenPowerLaw_math(alpha_1=-self.alpha_1, alpha_2=-self.alpha_2, min_pl=self.mminbh, max_pl=self.mmaxbh, b=self.b)
        self.m2pr = _cmp.PowerLaw_math(alpha=self.beta, min_pl=self.mminbh, max_pl=self.mmaxbh)

        self.mdis={'mass_1': _cmp.SmoothedProb(origin_prob=self.m1pr, bottom=self.mminbh, bottom_smooth=self.delta_m),
                      'mass_2':_cmp.SmoothedProb(origin_prob=self.m2pr, bottom=self.mminbh, bottom_smooth=self.delta_m)}
