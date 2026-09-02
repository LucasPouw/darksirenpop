"""
Priors
Ignacio Magana, Rachel Gray, Sergio Vallejo-Peña, Antonio Enea Romano 
"""


from __future__ import absolute_import

import numpy as np

# from scipy.interpolate import interp1d
# import bilby 
# from gwcosmo.utilities.mass_prior_utilities import peaks_sampling_constraint, peaks_grid_constraint

import darksirenpop.utilities.custom_math_priors as _cmp

from typing import Optional, List

import torch
torch.set_num_threads(1)  # to avoid core-hogging

@torch.jit.script
def torch_interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, fill_value: Optional[List[float]] = None):
    lower_fill: float = 0.0
    upper_fill: float = 0.0
    if fill_value is not None:
        lower_fill = fill_value[0]
        upper_fill = fill_value[1]

    idx = torch.searchsorted(xp, x, side='right')  # where our sample points lie on the xp grid
    idx = torch.clamp(idx, 1, len(xp) - 1)  # Ensure indices are within bounds

    # Use gather to get the values at the indices
    x1 = xp.gather(0, idx)
    x0 = xp.gather(0, idx - 1)
    y1 = fp.gather(0, idx)
    y0 = fp.gather(0, idx - 1)

    interps = (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)
    interps = torch.where(x > xp[-1], torch.full_like(x, upper_fill), interps)
    interps = torch.where(x < xp[0], torch.full_like(x, lower_fill), interps)
    return interps

def pH0(H0, prior='log'):
    """
    Returns p(H0)
    The prior probability of H0

    Parameters
    ----------
    H0 : float or array_like
        Hubble constant value(s) in kms-1Mpc-1
    prior : str, optional
        The choice of prior (default='log')
        if 'log' uses uniform in log prior
        if 'uniform' uses uniform prior

    Returns
    -------
    float or array_like
        p(H0)
    """
    if prior == 'uniform':
        return np.ones(len(H0))
    if prior == 'log':
        return 1./H0
    
def pairing_func(m1, m2, beta, device="cpu"):
    """
    Returns simple pairing function (Torch compatible)

    Parameters
    ----------
    m1, m2 : torch.Tensor or array_like
        primary and secondary mass(es)
    beta: float
        index of mass ratio powerlaw
    """

    m1 = torch.as_tensor(m1,device=device)
    m2 = torch.as_tensor(m2,device=device)
    q = m2 / m1
    toret = q**beta
    toret = torch.where(q>1, torch.zeros_like(toret), toret)  # Ensure 0.0 is a float tensor

    return toret

def pairing_func_broken(m1, m2, beta1, beta2, mbreak, device ="cpu"):
    """
    Returns broken pairing function (Torch compatible)

    Parameters
    ---------
     m1, m2 : torch.Tensor or array_like
        primary and secondary mass(es)
    beta1: float
        index of mass ratio powerlaw before mbreak
    beta2: float
        index of mass ratio powerlaw after mbreak
    mbreak: float
        secondary mass value at which powerlaw index changes

    """
    m1 = torch.as_tensor(m1,device=device)
    m2 = torch.as_tensor(m2,device=device)
    q = m2 / m1
    t_before = q ** beta1
    t_after = q ** beta2
    toret = torch.where(m2<mbreak, t_before, t_after)
    toret = torch.where(q>1, torch.zeros_like(toret), toret)  # Ensure 0.0 is a float tensor

    return toret

class distance_distribution(object):
    def __init__(self, name):
        self.name = name

        if self.name == 'BBH-powerlaw':
            dist = PriorDict(conversion_function=constrain_m1m2)
            dist['luminosity_distance'] = PowerLaw(alpha=2, minimum=1, maximum=15000)

        if self.name == 'BNS':
            dist = PriorDict(conversion_function=constrain_m1m2)
            dist['luminosity_distance'] = PowerLaw(alpha=2, minimum=1, maximum=1000)

        if self.name == 'NSBH':
            dist = PriorDict(conversion_function=constrain_m1m2)
            dist['luminosity_distance'] = PowerLaw(alpha=2, minimum=1, maximum=1000)

        if self.name == 'BBH-constant':
            dist = PriorDict()
            dist['luminosity_distance'] = PowerLaw(alpha=2, minimum=1, maximum=15000)

        self.dist = dist

    def sample(self, N_samples):
        samples = self.dist.sample(N_samples)
        return samples['luminosity_distance']

    def prob(self, samples):
        return self.dist['luminosity_distance'].prob(samples)

class m_priors(object):
    """
    Parent class with common methods for managing the priors on source frame masses.
    The prior is factorized as :math:`p(m_1,m_2) \\propto p(m_1)p(m_2|m_1)`
    """

    def __init__(self, device):
        self.device = device
        self.dtype = torch.get_default_dtype()

    def update_parameters(self,param_dict):
        """
        Method to dynamically determine attributes in a mass prior class and use these.
        """
        for key, value in param_dict.items():
            setattr(self, key, value)
        self.update_mass_priors()

    def set_device(self, device):
        self.mdis["mass_1"].set_device(device)
        self.mdis["mass_2"].set_device(device)
        self.device = device

    def set_dtype(self, dtype):
        self.mdis["mass_1"].set_dtype(dtype)
        self.mdis["mass_2"].set_dtype(dtype)
        self.dtype = dtype

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

        tensor_ms1 = torch.as_tensor(ms1, device=self.device, dtype=self.dtype)
        tensor_ms2 = torch.as_tensor(ms2, device=self.device, dtype=self.dtype)

        to_ret = self.mdis['mass_1'].prob(tensor_ms1)*self.mdis['mass_2'].conditioned_prob(tensor_ms2,self.mmin*torch.ones_like(tensor_ms1, device=self.device, dtype=self.dtype),torch.minimum(tensor_ms1,torch.as_tensor(self.mmax2, device=self.device, dtype=self.dtype)))
        
        return to_ret
    
    def log_joint_prob(self,ms1, ms2):
        
        to_ret = torch.log(self.joint_prob(ms1, ms2))
        to_ret[torch.isnan(to_ret)] = -torch.inf

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

        vals_m1 = torch.rand(Nsample, device=self.device, dtype=self.dtype)
        vals_m2 = torch.rand(Nsample, device=self.device, dtype=self.dtype)

        m1_min_tensor = torch.tensor(self.mdis['mass_1'].minimum, device=self.device, dtype=self.dtype)
        m1_max_tensor = torch.tensor(self.mdis['mass_1'].maximum, device=self.device, dtype=self.dtype)
        m2_min_tensor = torch.tensor(self.mdis['mass_2'].minimum, device=self.device, dtype=self.dtype)
        m2_max_tensor = torch.tensor(self.mdis['mass_2'].maximum, device=self.device, dtype=self.dtype)

        m1_trials = torch.logspace(torch.log10(m1_min_tensor), torch.log10(m1_max_tensor), 10000, device=self.device)
        m2_trials = torch.logspace(torch.log10(m2_min_tensor), torch.log10(m2_max_tensor), 10000, device=self.device)

        cdf_m1_trials = self.mdis['mass_1'].cdf(m1_trials)
        cdf_m2_trials = self.mdis['mass_2'].cdf(m2_trials)

        m1_trials = torch.log10(m1_trials)
        m2_trials = torch.log10(m2_trials)

        indxm1 = torch.where(torch.diff(cdf_m1_trials) != 0)[0][[0,-1]]
        indxm2 = torch.where(torch.diff(cdf_m2_trials) != 0)[0][[0,-1]]

        mass_1_samples = 10**torch_interp(vals_m1, cdf_m1_trials[indxm1[0]:indxm1[1]+2], m1_trials[indxm1[0]:indxm1[1]+2], fill_value=[m1_trials[0].item(), m1_trials[-1].item()])
        mass_2_samples = 10**torch_interp(vals_m2*self.mdis['mass_2'].cdf(mass_1_samples), cdf_m2_trials[indxm2[0]:indxm2[1]+2], m2_trials[indxm2[0]:indxm2[1]+2], fill_value=[m2_trials[0].item(), m2_trials[-1].item()])

        return mass_1_samples, mass_2_samples
    
    @staticmethod
    def grid_constraint_call(*args):
        pass 

    @staticmethod
    def sampling_constraint_call(prior_dict):
        return prior_dict

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
    def __init__(self,mminbh=6.0,mmaxbh=125.0,alpha=6.75,beta=4.0, device="cpu"):
        super().__init__(device=device)

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

        self.mdis={'mass_1':_cmp.PowerLaw_math(alpha=-self.alpha,min_pl=self.mminbh,max_pl=self.mmaxbh, device=self.device),
                     'mass_2':_cmp.PowerLaw_math(alpha=self.beta,min_pl=self.mminbh,max_pl=self.mmaxbh, device=self.device)}
        

class NSBH_powerlaw(m_priors):
    """
    Child class for NS-BH power law distribution.
    
    Parameters
    -------------
    mminbh: Minimum mass of the PL component of the black hole mass distribution
    mmaxbh: Maximum mass of the PL component of the black hole mass distribution
    alpha: Spectral index for the PL of the primary mass distribution
    mminns: Minimum mass of the neutron star distribution
    mmaxns: Maximum mass of the neutron star distribution
    alphans: Spectral index for the PL of the neutron star mass distribution

    The default values of the black hole mass distribution parameters are set to the corresponding median values in the uniform priors reported in 2111.03604
    The default values of the neutron star mass distribution parameters are set to the corresponding values reported in section 4.2 (page 23) in 2111.03604

    ************
    NOTE: The spectral indices passed to PowerLaw_math are alpha=-self.alpha, and alpha=-self.alphans, according to eq. A10 in 2111.03604
    *************   

    The method m_priors.update_parameters is used in the constructor to initialize the objects
    """
    def __init__(self,mminbh=6.0,mmaxbh=125.0,alpha=6.75,mminns=1.0,mmaxns=3.0,alphans=0.0, device="cpu"):
        super().__init__(device=device)

        self.update_parameters(param_dict={'alpha':alpha, 'mminbh':mminbh, 'mmaxbh':mmaxbh, 'alphans':alphans, 'mminns':mminns, 'mmaxns':mmaxns})
        
    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to mmaxbh, 
        the minimum value of the secondary mass distribution mmin to mminns, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxns.
        It's called by update_paratemters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''

        self.mmax=self.mmaxbh
        self.mmin=self.mminns        
        self.mmax2=self.mmaxns

        self.mdis={'mass_1':_cmp.PowerLaw_math(alpha=-self.alpha,min_pl=self.mminbh,max_pl=self.mmaxbh, device=self.device),
                     'mass_2':_cmp.PowerLaw_math(alpha=-self.alphans,min_pl=self.mminns,max_pl=self.mmaxns, device=self.device)}
   
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
    lambda_g: Fraction of the model in the Gaussian component
    delta_m: Range of mass tapering on the lower end of the mass distribution
    beta: Spectral index for the PL of the mass ratio distribution

    The default values of the parameters are set to the corresponding values reported in section 4.2 (page 23) in 2111.03604

    ************
    NOTE: The spectral indices passed to PowerLawGaussian_math, and PowerLaw_math, are alpha=-self.alpha, and alpha=self.beta, according to eqs. A8,A11 in 2111.03604
    *************   

    The method m_priors.update_parameters is used in the constructor to initialize the objects.
    """
    def __init__(self,mminbh=4.98,mmaxbh=112.5,alpha=3.78,mu_g=32.27,sigma_g=3.88,lambda_g=0.03,delta_m=4.8,beta=0.81, device="cpu"):
        super().__init__(device=device)
        
        self.update_parameters(param_dict={'alpha':alpha, 'beta':beta, 'mminbh':mminbh, 'mmaxbh':mmaxbh, 'mu_g':mu_g, 'sigma_g':sigma_g, 'lambda_g':lambda_g, 'delta_m':delta_m})

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
                       
        self.m1pr = _cmp.PowerLawGaussian_math(alpha=-self.alpha,min_pl=self.mminbh,max_pl=self.mmaxbh,lambda_g=self.lambda_g
                    ,mu_g=self.mu_g,sigma_g=self.sigma_g,min_g=self.mminbh,max_g=self.mu_g+5*self.sigma_g, device=self.device)

        # The max of the secondary mass is adapted to the primary mass maximum which is decided by the Gaussian and PL
        self.m2pr = _cmp.PowerLaw_math(alpha=self.beta,min_pl=self.mminbh,max_pl=np.max([self.mu_g+5*self.sigma_g,self.mmaxbh]), device=self.device)
        self.mdis={'mass_1': _cmp.SmoothedProb(origin_prob=self.m1pr,high_pass_min=self.mminbh,high_pass_smooth=self.delta_m, device=self.device),
                      'mass_2':_cmp.SmoothedProb(origin_prob=self.m2pr,high_pass_min=self.mminbh,high_pass_smooth=self.delta_m, device=self.device)}
       
        # TO DO Add a check on the mu_g - 5 sigma of the gaussian to not overlap with mmin, print a warning
        #if (mu_g - 5*sigma_g)<=mmin:
        #print('Warning, your mean (minuse 5 sigma) of the gaussian component is too close to the minimum mass')

        self.mmax = self.mdis['mass_1'].maximum 
        self.mmin = self.mminbh  
        self.mmax2 = self.mmaxbh

class NSBH_powerlaw_gaussian(m_priors):
    """
    Child class for NS-BH power law gaussian distribution.
    
    Parameters
    -------------
    mminbh: Minimum mass of the PL component of the black hole mass distribution
    mmaxbh: Maximum mass of the PL component of the black hole mass distribution
    alpha: Spectral index for the PL of the primary mass distribution    
    mu_g: Mean of the Gaussian component in the primary mass distribution
    sigma_g: Width of the Gaussian component in the primary mass distribution
    lambda_g: Fraction of the model in the Gaussian component    
    delta_m: Range of mass tapering on the lower end of the mass distribution
    mminns: Minimum mass of the neutron star distribution
    mmaxns: Maximum mass of the neutron star distribution
    alphans: Spectral index for the PL of the neutron star mass distribution

    The default values of the parameters are set to the corresponding values reported in section 4.2 (page 23) in 2111.03604

    ************
    NOTE: The spectral indices passed to PowerLawGaussian_math, and PowerLaw_math, are alpha=-self.alpha, and alpha=-self.alphans, according to eqs. A10,A11 in 2111.03604
    *************
        
    The method m_priors.update_parameters is used in the constructor to initialize the objects.
    """
    def __init__(self,mminbh=4.98,mmaxbh=112.5,alpha=3.78,mu_g=32.27,sigma_g=3.88,lambda_g=0.03,delta_m=4.8,mminns=1.0,mmaxns=3.0,alphans=0.0, device="cpu"):
        super().__init__(device=device)

        self.update_parameters(param_dict={'alpha':alpha, 'mminbh':mminbh, 'mmaxbh':mmaxbh, 'mu_g':mu_g, 'sigma_g':sigma_g, 'lambda_g':lambda_g, 'delta_m':delta_m, 'alphans':alphans, 'mminns':mminns, 'mmaxns':mmaxns})

    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to self.mdis['mass_1'].maximum, 
        the minimum value of the secondary mass distribution mmin to mminns, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxns.
        It's called by update_paratemters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''
                
        self.m1pr = _cmp.PowerLawGaussian_math(alpha=-self.alpha,min_pl=self.mminbh,max_pl=self.mmaxbh,lambda_g=self.lambda_g
                    ,mu_g=self.mu_g,sigma_g=self.sigma_g,min_g=self.mminbh,max_g=self.mu_g+5*self.sigma_g, device=self.device)

        # The max of the secondary mass is adapted to the primary mass maximum which is decided by the Gaussian and PL
        self.m2pr = _cmp.PowerLaw_math(alpha=-self.alphans,min_pl=self.mminns,max_pl=self.mmaxns, device=self.device)

        self.mdis={'mass_1': _cmp.SmoothedProb(origin_prob=self.m1pr,high_pass_min=self.mminbh,high_pass_smooth=self.delta_m, device=self.device),
                      'mass_2':self.m2pr}

        self.mmax = self.mdis['mass_1'].maximum 
        self.mmin = self.mminns  
        self.mmax2 = self.mmaxns

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
    def __init__(self,mminbh=26,mmaxbh=125,alpha_1=6.75,alpha_2=6.75,b=0.5,delta_m=5,beta=4, device="cpu"):
        super().__init__(device=device)

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
                
        self.m1pr = _cmp.BrokenPowerLaw_math(alpha_1=-self.alpha_1,alpha_2=-self.alpha_2,min_pl=self.mminbh,max_pl=self.mmaxbh,b=self.b, device=self.device)
        self.m2pr = _cmp.PowerLaw_math(alpha=self.beta,min_pl=self.mminbh,max_pl=self.mmaxbh, device=self.device)

        self.mdis={'mass_1': _cmp.SmoothedProb(origin_prob=self.m1pr,high_pass_min=self.mminbh,high_pass_smooth=self.delta_m, device=self.device),
                      'mass_2':_cmp.SmoothedProb(origin_prob=self.m2pr,high_pass_min=self.mminbh,high_pass_smooth=self.delta_m, device=self.device)}

class NSBH_broken_powerlaw(m_priors):
    """
    Child class for NS-BH broken power law distribution.
    
    Parameters
    -------------
    mminbh: Minimum mass of the PL component of the black hole mass distribution
    mmaxbh: Maximum mass of the black hole mass distribution
    alpha_1: PL slope of the primary mass distribution for masses below mbreak 
    alpha_2: PL slope for the primary mass distribution for masses above mbreak 
    b: The fraction of the way between mminbh and mmaxbh at which the primary mass distribution breaks
    delta_m: Range of mass tapering on the lower end of the mass distribution
    mminns: Minimum mass of the neutron star distribution
    mmaxns: Maximum mass of the neutron star distribution
    alphans: Spectral index for the PL of the neutron star mass distribution

    The default values of the black hole mass distribution parameters are set to the corresponding median values in the uniform priors reported in 2111.03604
    The default values of the neutron star mass distribution parameters are set to the corresponding values reported in section 4.2 (page 23) in 2111.03604

    ************
    NOTE: The spectral indices passed to BrokenPowerLaw_math, and PowerLaw_math, are alpha_1=-self.alpha_1, alpha_2=-self.alpha_2, and alpha=-self.alphans, according to eqs. A10,A12 in 2111.03604
    ************
    
    The method m_priors.update_parameters is used in the constructor to initialize the objects.
    """
    def __init__(self,mminbh=26,mmaxbh=125,alpha_1=6.75,alpha_2=6.75,b=0.5,delta_m=5,mminns=1.0,mmaxns=3.0,alphans=0.0,device="cpu"):
        super().__init__(device=device)

        self.update_parameters(param_dict={'alpha_1':alpha_1, 'alpha_2':alpha_2, 'mminbh':mminbh, 'mmaxbh':mmaxbh, 'b':b, 'delta_m':delta_m, 'alphans':alphans, 'mminns':mminns, 'mmaxns':mmaxns})

    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to mmaxbh, 
        the minimum value of the secondary mass distribution mmin to mminns, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxns.
        It's called by update_paratemters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''
        
        self.mmax=self.mmaxbh
        self.mmin=self.mminns        
        self.mmax2=self.mmaxns
                
        self.m1pr = _cmp.BrokenPowerLaw_math(alpha_1=-self.alpha_1,alpha_2=-self.alpha_2,min_pl=self.mminbh,max_pl=self.mmaxbh,b=self.b, device=self.device)
        self.m2pr = _cmp.PowerLaw_math(alpha=-self.alphans,min_pl=self.mminns,max_pl=self.mmaxns, device=self.device)

        self.mdis={'mass_1': _cmp.SmoothedProb(origin_prob=self.m1pr,high_pass_min=self.mminbh,high_pass_smooth=self.delta_m, device=self.device),
                      'mass_2':self.m2pr}
             

class BBH_multi_peak_gaussian(m_priors):
    """
    Child class for BBH with powerlaw component and two gaussian peaks.

    Parameters
    -------------
    mminbh: Minimum mass of the PL component of the black hole mass distribution
    mmaxbh: Maximum mass of the PL component of the black hole mass distribution
    alpha: Spectral index for the PL of the primary mass distribution    
    mu_g_0: Mean of the lower mass Gaussian component in the primary mass distribution
    sigma_g_0: Width of the lower mass Gaussian component in the primary mass distribution
    mu_g_1: Mean of the higher mass Gaussian component in the primary mass distribution
    sigma_g_1: Width of the higher mass Gaussian component in the primary mass distribution
    lambda_g: Fraction of the model in the Gaussian component
    lambda_g_0: Fraction of the Gaussian component in the lower mass peak
    delta_m: Range of mass tapering on the lower end of the mass distribution
    beta: Spectral index for the PL of the mass ratio distribution

    ************
    NOTE: The spectral indices passed to PowerLawDoubleGaussian_math, and PowerLaw_math, are alpha=-self.alpha, and alpha=self.beta, according to eqs. A8,A11 in 2111.03604
    ************* 
    
    The method m_priors.update_parameters is used in the constructor to initialize the objects.
    """
    def __init__(self,alpha=3.78,beta=0.8,mminbh=4.98,mmaxbh=112.5,lambda_g=0.03,lambda_g_0= 0.5,mu_g_0=10.5,sigma_g_0=3.88,mu_g_1=32.27,sigma_g_1=5,delta_m=5,device="cpu"):
        super().__init__(device=device)

        self.update_parameters(param_dict={'alpha':alpha,'beta':beta,'mminbh':mminbh,'mmaxbh':mmaxbh,'lambda_g':lambda_g,'lambda_g_0':lambda_g_0,'mu_g_0':mu_g_0,'sigma_g_0':sigma_g_0,'mu_g_1':mu_g_1,'sigma_g_1':sigma_g_1,'delta_m':delta_m})

    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to self.mdis['mass_1'].maximum, 
        the minimum value of the secondary mass distribution mmin to mminbh, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxbh.
        It's called by update_parameters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''

        self.m1pr =_cmp.PowerLawDoubleGaussian_math(alpha=-self.alpha,min_pl=self.mminbh,max_pl=self.mmaxbh,lambda_g=self.lambda_g,lambda_g_0=self.lambda_g_0,
                                                    mu_g_0=self.mu_g_0,sigma_g_0=self.sigma_g_0,mu_g_1=self.mu_g_1,sigma_g_1=self.sigma_g_1,min_g=self.mminbh,max_g=self.mu_g_1+5*self.sigma_g_1,
                                                    device=self.device)
        

        self.m2pr =_cmp.PowerLaw_math(alpha=self.beta,min_pl=self.mminbh,max_pl=np.max([self.mu_g_0+5*self.sigma_g_0,self.mmaxbh]),device=self.device)

        self.mdis={'mass_1': _cmp.SmoothedProb(origin_prob=self.m1pr,high_pass_min=self.mminbh,high_pass_smooth=self.delta_m,device=self.device),
                      'mass_2':_cmp.SmoothedProb(origin_prob=self.m2pr,high_pass_min=self.mminbh,high_pass_smooth=self.delta_m,device=self.device)}

        self.mmax = self.mdis['mass_1'].maximum
        self.mmin = self.mminbh
        self.mmax2 = self.mmaxbh

    @staticmethod
    def grid_constraint_call(constraint_grid, values, parameter_grid):
        new_grid = peaks_grid_constraint(constraint_grid, values, parameter_grid)
        return new_grid

    @staticmethod
    def sampling_constraint_call(prior_dict):
        new_dict = peaks_sampling_constraint(prior_dict)
        return new_dict


class NSBH_multi_peak_gaussian(m_priors):
    """
    Child class for NS-BH with powerlaw component and two gaussian peaks.

    Parameters
    -------------
    mminbh: Minimum mass of the PL component of the black hole mass distribution
    mmaxbh: Maximum mass of the PL component of the black hole mass distribution
    alpha: Spectral index for the PL of the primary mass distribution    
    mu_g_0: Mean of the lower mass Gaussian component in the primary mass distribution
    sigma_g_0: Width of the lower mass Gaussian component in the primary mass distribution
    mu_g_1: Mean of the higher mass Gaussian component in the primary mass distribution
    sigma_g_1: Width of the higher mass Gaussian component in the primary mass distribution
    lambda_g: Fraction of the model in the Gaussian component
    lambda_g_0: Fraction of the Gaussian component in the lower mass peak
    delta_m: Range of mass tapering on the lower end of the mass distribution
    mminns: Minimum mass of the neutron star distribution
    mmaxns: Maximum mass of the neutron star distribution
    alphans: Spectral index for the PL of the neutron star mass distribution

    ************
    NOTE: The spectral indices passed to PowerLawDoubleGaussian_math, and PowerLaw_math, are alpha=-self.alpha, and alpha=-self.alphans, according to eqs. A10,A11 in 2111.03604
    *************
    
    The method m_priors.update_parameters is used in the constructor to initialize the objects.
    """

    def __init__(self,alpha=3.78,mminbh=4.98,mmaxbh=112.5,lambda_g=0.03,lambda_g_0= 0.5,mu_g_0=10.5,sigma_g_0=3.88,mu_g_1=32.27,sigma_g_1=5,delta_m=5,mminns=1.0,mmaxns=3.0,alphans=0,device="cpu"):
        super().__init__(device=device)

        self.update_parameters(param_dict={'alpha':alpha,'mminbh':mminbh,'mmaxbh':mmaxbh,'lambda_g':lambda_g,'lambda_g_0':lambda_g_0,'mu_g_0':mu_g_0,'sigma_g_0':sigma_g_0,'mu_g_1':mu_g_1,'sigma_g_1':sigma_g_1,'delta_m':delta_m,'mminns':mminns,'mmaxns':mmaxns,'alphans':alphans})

    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to self.mdis['mass_1'].maximum, 
        the minimum value of the secondary mass distribution mmin to mminns, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxns.
        It's called by update_parameters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''


        self.m1pr =_cmp.PowerLawDoubleGaussian_math(alpha=-self.alpha,min_pl=self.mminbh,max_pl=self.mmaxbh,lambda_g=self.lambda_g,lambda_g_0=self.lambda_g_0,
                                                    mu_g_0=self.mu_g_0,sigma_g_0=self.sigma_g_0,mu_g_1=self.mu_g_1,sigma_g_1=self.sigma_g_1,min_g=self.mminbh,max_g=self.mu_g_1+5*self.sigma_g_1,device=self.device)
        

        self.m2pr = _cmp.PowerLaw_math(alpha=-self.alphans,min_pl=self.mminns,max_pl=self.mmaxns,device=self.device)

        self.mdis={'mass_1': _cmp.SmoothedProb(origin_prob=self.m1pr,high_pass_min=self.mminbh,high_pass_smooth=self.delta_m, device=self.device),
                      'mass_2': self.m2pr}

        self.mmax = self.mdis['mass_1'].maximum
        self.mmin = self.mminns
        self.mmax2 = self.mmaxns

    @staticmethod
    def grid_constraint_call(constraint_grid, values, parameter_grid):
        new_grid = peaks_grid_constraint(constraint_grid, values, parameter_grid)
        return new_grid
    
    @staticmethod
    def sampling_constraint_call(prior_dict):
        new_dict = peaks_sampling_constraint(prior_dict)
        return new_dict


class BBH_broken_powerlaw_multi_peak_gaussian_m1m2(m_priors):

    def __init__(self,
                 alpha_1=6.75,
                 alpha_2=6.75,
                 b=0.5,
                 beta=0.8,
                 mminbh=4.98,
                 mmaxbh=112.5,
                 lambda_g=0.03,
                 lambda_g_0= 0.5,
                 mu_g_0=10.5,
                 sigma_g_0=3.88,
                 mu_g_1=32.27,
                 sigma_g_1=5,
                 delta_m=5,
                 mlow_2=4.98,
                 delta_m_2=5,
                 device="cpu"):
        super().__init__(device=device)

        self.update_parameters(param_dict={'alpha_1':alpha_1,
                                           'alpha_2':alpha_2,
                                           'b':b,
                                           'beta':beta,
                                           'mminbh':mminbh,
                                           'mmaxbh':mmaxbh,
                                           'lambda_g':lambda_g,
                                           'lambda_g_0':lambda_g_0,
                                           'mu_g_0':mu_g_0,
                                           'sigma_g_0':sigma_g_0,
                                           'mu_g_1':mu_g_1,
                                           'sigma_g_1':sigma_g_1,
                                           'delta_m':delta_m,
                                           'mlow_2':mlow_2,
                                           'delta_m_2':delta_m_2})
    
    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to self.mdis['mass_1'].maximum, 
        the minimum value of the secondary mass distribution mmin to mminns, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxns.
        It's called by update_parameters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''

        self.break_point = self.mminbh+self.b*(self.mmaxbh-self.mminbh)
        self.m1pr =_cmp.BrokenPowerLawDoubleGaussian_math(min_pl=self.mminbh,max_pl=self.mmaxbh,lambda_g=self.lambda_g,lambda_g_0=self.lambda_g_0,
                                                    mu_g_0=self.mu_g_0,sigma_g_0=self.sigma_g_0,mu_g_1=self.mu_g_1,
                                                    sigma_g_1=self.sigma_g_1,min_g=self.mminbh,max_g=self.mu_g_1+5*self.sigma_g_1,
                                                    alpha_1=-self.alpha_1,alpha_2=-self.alpha_2,break_point=self.break_point,device=self.device)
        

        self.m2pr = _cmp.PowerLaw_math(alpha=self.beta,min_pl=self.mminbh,max_pl=np.max([self.mu_g_0+5*self.sigma_g_0,self.mmaxbh]),device=self.device)

        self.mdis={'mass_1': _cmp.SmoothedProb(origin_prob=self.m1pr,high_pass_min=self.mminbh,high_pass_smooth=self.delta_m,device=self.device),
                   'mass_2': _cmp.SmoothedProb(origin_prob=self.m2pr,high_pass_min=self.mlow_2,high_pass_smooth=self.delta_m_2,device=self.device)}

        self.mmax = self.mdis['mass_1'].maximum
        self.mmin = self.mminbh
        self.mmax2 = self.mmaxbh

    @staticmethod
    def grid_constraint_call(constraint_grid, values, parameter_grid):
        new_grid = peaks_grid_constraint(constraint_grid, values, parameter_grid)
        return new_grid
    
    @staticmethod
    def sampling_constraint_call(prior_dict):
        new_dict = peaks_sampling_constraint(prior_dict)
        return new_dict


class BBH_broken_powerlaw_multi_peak_gaussian(m_priors):

    def __init__(self,alpha_1=6.75,alpha_2=6.75,b=0.5,beta=0.8,mminbh=4.98,mmaxbh=112.5,lambda_g=0.03,lambda_g_0= 0.5,mu_g_0=10.5,sigma_g_0=3.88,mu_g_1=32.27,sigma_g_1=5,delta_m=5,device="cpu"):
        super().__init__(device=device)

        self.update_parameters(param_dict={'alpha_1':alpha_1,'alpha_2':alpha_2,'b':b,'beta':beta,'mminbh':mminbh,'mmaxbh':mmaxbh,
                                           'lambda_g':lambda_g,'lambda_g_0':lambda_g_0,'mu_g_0':mu_g_0,'sigma_g_0':sigma_g_0,'mu_g_1':mu_g_1,'sigma_g_1':sigma_g_1,'delta_m':delta_m})
    
    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to self.mdis['mass_1'].maximum, 
        the minimum value of the secondary mass distribution mmin to mminns, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxns.
        It's called by update_parameters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''

        self.break_point = self.mminbh+self.b*(self.mmaxbh-self.mminbh)
        self.m1pr =_cmp.BrokenPowerLawDoubleGaussian_math(min_pl=self.mminbh,max_pl=self.mmaxbh,lambda_g=self.lambda_g,lambda_g_0=self.lambda_g_0,
                                                    mu_g_0=self.mu_g_0,sigma_g_0=self.sigma_g_0,mu_g_1=self.mu_g_1,
                                                    sigma_g_1=self.sigma_g_1,min_g=self.mminbh,max_g=self.mu_g_1+5*self.sigma_g_1,
                                                    alpha_1=-self.alpha_1,alpha_2=-self.alpha_2,break_point=self.break_point,device=self.device)
        

        self.m2pr = _cmp.PowerLaw_math(alpha=self.beta,min_pl=self.mminbh,max_pl=np.max([self.mu_g_0+5*self.sigma_g_0,self.mmaxbh]),device=self.device)

        self.mdis={'mass_1': _cmp.SmoothedProb(origin_prob=self.m1pr,high_pass_min=self.mminbh,high_pass_smooth=self.delta_m,device=self.device),
                      'mass_2': _cmp.SmoothedProb(origin_prob=self.m2pr,high_pass_min=self.mminbh,high_pass_smooth=self.delta_m,device=self.device)}

        self.mmax = self.mdis['mass_1'].maximum
        self.mmin = self.mminbh
        self.mmax2 = self.mmaxbh

    @staticmethod
    def grid_constraint_call(constraint_grid, values, parameter_grid):
        new_grid = peaks_grid_constraint(constraint_grid, values, parameter_grid)
        return new_grid
    
    @staticmethod
    def sampling_constraint_call(prior_dict):
        new_dict = peaks_sampling_constraint(prior_dict)
        return new_dict
    
class NSBH_broken_powerlaw_multi_peak_gaussian(m_priors):

    def __init__(self,alpha_1=6.75,alpha_2=6.75,b=0.5,mminbh=4.98,mmaxbh=112.5,lambda_g=0.03,lambda_g_0= 0.5,mu_g_0=10.5,sigma_g_0=3.88,mu_g_1=32.27,sigma_g_1=5,delta_m=5,mminns=1,mmaxns=5,alphans=0,device="cpu"):
        super().__init__(device=device)

        self.update_parameters(param_dict={'alpha_1':alpha_1,'alpha_2':alpha_2,'b':b,'mminbh':mminbh,'mmaxbh':mmaxbh,
                                           'lambda_g':lambda_g,'lambda_g_0':lambda_g_0,'mu_g_0':mu_g_0,'sigma_g_0':sigma_g_0,'mu_g_1':mu_g_1,'sigma_g_1':sigma_g_1,'delta_m':delta_m,
                                           'mminns':mminns,'mmaxns':mmaxns,'alphans':alphans})
    
    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to self.mdis['mass_1'].maximum, 
        the minimum value of the secondary mass distribution mmin to mminns, 
        and the maximum value of the secondary mass distribution mmax2 to mmaxns.
        It's called by update_parameters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''
        self.break_point = self.mminbh+self.b*(self.mmaxbh-self.mminbh)

        self.m1pr =_cmp.BrokenPowerLawDoubleGaussian_math(min_pl=self.mminbh,max_pl=self.mmaxbh,lambda_g=self.lambda_g,lambda_g_0=self.lambda_g_0,
                                                    mu_g_0=self.mu_g_0,sigma_g_0=self.sigma_g_0,mu_g_1=self.mu_g_1,
                                                    sigma_g_1=self.sigma_g_1,min_g=self.mminbh,max_g=self.mu_g_1+5*self.sigma_g_1,
                                                    alpha_1=-self.alpha_1,alpha_2=-self.alpha_2,break_point=self.break_point,device=self.device)
        

        self.m2pr = _cmp.PowerLaw_math(alpha=-self.alphans,min_pl=self.mminns,max_pl=self.mmaxns,device=self.device)

        self.mdis={'mass_1': _cmp.SmoothedProb(origin_prob=self.m1pr,high_pass_min=self.mminbh,high_pass_smooth=self.delta_m,device=self.device),
                      'mass_2': self.m2pr}

        self.mmax = self.mdis['mass_1'].maximum
        self.mmin = self.mminns
        self.mmax2 = self.mmaxns

    @staticmethod
    def grid_constraint_call(constraint_grid, values, parameter_grid):
        new_grid = peaks_grid_constraint(constraint_grid, values, parameter_grid)
        return new_grid
    
    @staticmethod
    def sampling_constraint_call(prior_dict):
        new_dict = peaks_sampling_constraint(prior_dict)
        return new_dict
    


class BNS(m_priors):
    """
    Child class for BNS distribution.
    
    Parameters
    -----------
    mminns: Minimum mass of the neutron star distribution
    mmaxns: Maximum mass of the neutron star distribution
    alphans: Spectral index for the PL of the neutron star mass distribution

    The default values of the parameters are set to the corresponding values reported in section 4.2 (page 23) in 2111.03604

    ************
    NOTE: The spectral index passed to PowerLaw_math is alpha=-self.alphans according to eq. A10 in 2111.03604
    ************
    
    The method m_priors.update_parameters is used in the constructor to initialize the objects.
    """
    def __init__(self,mminns=1.0,mmaxns=3.0,alphans=0.0, device="cpu"):
        super().__init__(device=device)

        self.update_parameters(param_dict={'alphans':alphans, 'mminns':mminns, 'mmaxns':mmaxns})
                    
    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to mmaxns, 
        and the minimum value of the secondary mass distribution mmin to mminns.
        It's called by update_paratemters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax and mmin definitions depend on the mass prior model.          
        '''

        self.mmax = self.mmaxns
        self.mmin = self.mminns        

        self.mdis={'mass_1':_cmp.PowerLaw_math(alpha=-self.alphans,min_pl=self.mminns,max_pl=self.mmaxns, device=self.device),
                  'mass_2':_cmp.PowerLaw_math(alpha=-self.alphans,min_pl=self.mminns,max_pl=self.mmaxns, device=self.device)}
        
    
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

        to_ret =self.mdis['mass_1'].prob(torch.as_tensor(ms1))*self.mdis['mass_2'].prob(torch.as_tensor(ms2))

        return to_ret
    

    def sample(self, Nsample):
        """
        This method samples from the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        Nsample: int
            Number of samples you want
        """

        vals_m1 = torch.rand(Nsample, device=self.device)
        vals_m2 = torch.rand(Nsample, device=self.device)

        m1_min_tensor = torch.tensor(self.mdis['mass_1'].minimum, device=self.device)
        m1_max_tensor = torch.tensor(self.mdis['mass_1'].maximum, device=self.device)
        m2_min_tensor = torch.tensor(self.mdis['mass_2'].minimum, device=self.device)
        m2_max_tensor = torch.tensor(self.mdis['mass_2'].maximum, device=self.device)

        m1_trials = torch.logspace(torch.log10(m1_min_tensor), torch.log10(m1_max_tensor), 10000, device=self.device)
        m2_trials = torch.logspace(torch.log10(m2_min_tensor), torch.log10(m2_max_tensor), 10000, device=self.device)

        cdf_m1_trials = self.mdis['mass_1'].cdf(m1_trials)
        cdf_m2_trials = self.mdis['mass_2'].cdf(m2_trials)

        m1_trials = torch.log10(m1_trials)
        m2_trials = torch.log10(m2_trials)

        indxm1 = torch.where(torch.diff(cdf_m1_trials) != 0)[0][[0,-1]]
        indxm2 = torch.where(torch.diff(cdf_m2_trials) != 0)[0][[0,-1]]

        mass_1_samples = 10**torch_interp(vals_m1, cdf_m1_trials[indxm1[0]:indxm1[1]+2], m1_trials[indxm1[0]:indxm1[1]+2], fill_value=[m1_trials[0].item(), m1_trials[-1].item()])
        mass_2_samples = 10**torch_interp(vals_m2, cdf_m2_trials[indxm2[0]:indxm2[1]+2], m2_trials[indxm2[0]:indxm2[1]+2], fill_value=[m2_trials[0].item(), m2_trials[-1].item()])

        indx = torch.where(mass_2_samples>mass_1_samples)[0]
        mass_1_samples[indx],mass_2_samples[indx] = mass_2_samples[indx],mass_1_samples[indx]
        
        return mass_1_samples, mass_2_samples
  
class multipopulation_broken_powerlaw_multi_peak_gaussian(m_priors):
    """
    Child class which builds a multipopulation model covering whole CBC range. This has two identical distribtuions in m1 and m2,
    with both being BPL+2G+Dip with smoothing.

    This cannot be directly specified as a mass model in the command line.

    Parameters
    ----------
    alpha_1 : float
        Slope of the power law for masses below the break point.
    alpha_2 : float
        Slope of the power law for masses above the break point.
    mmin : float
        Minimum mass of the distribution.
    mmax : float
        Maximum mass of the distribution.
    lambda_g : float
        Scaling parameter for the overall Gaussian component.
    lambda_g_0 : float
        Scaling parameter for the low mass Gaussian peak.
    mu_g_0 : float
        Mean of the low mass Gaussian peak.
    sigma_g_0 : float
        Standard deviation of the low mass Gaussian peak.
    mu_g_1 : float
        Mean of the high mass Gaussian peak.
    sigma_g_1 : float
        Standard deviation of the high mass Gaussian peak.
    delta_m_low_pass : float
        Smoothing parameter for the high mass end of distribution.
    delta_m_high_pass : float
        Smoothing parameter for the low mass end of distribution.
    A : float
        Amplitude of the notch feature.
    notch_left : float
        Left boundary of the notch region.
    notch_right : float
        Right boundary of the notch region.
    notch_smooth_left : float
        Smoothing factor on the left side of the notch.
    notch_smooth_right : float
        Smoothing factor on the right side of the notch.
    """

    def __init__(self, alpha_1=1.0, alpha_2=3.0, mmin=1.0, mmax=112.5, lambda_g=0.3,
                  lambda_g_0=0.4, mu_g_0=15.0, sigma_g_0=3.88, mu_g_1=32.27, sigma_g_1=3.88, 
                  delta_m_low_pass=1, delta_m_high_pass=0.1, A=0, notch_left=3.0, notch_right=5.0, 
                  notch_smooth_left=1, notch_smooth_right=0.5,device="cpu"):
        super().__init__(device=device)

        self.update_parameters(param_dict={'alpha_1':alpha_1,'alpha_2':alpha_2,'mmin':mmin,'mmax':mmax,'lambda_g':lambda_g,'lambda_g_0':lambda_g_0,'mu_g_0':mu_g_0,'sigma_g_0':sigma_g_0,
                                           'mu_g_1':mu_g_1,'sigma_g_1':sigma_g_1,'delta_m_low_pass':delta_m_low_pass,'delta_m_high_pass':delta_m_high_pass,
                                           'A':A,'notch_left':notch_left,'notch_right':notch_right,'notch_smooth_left':notch_smooth_left,'notch_smooth_right':notch_smooth_right})
    
    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to self.mdis['mass_1'].maximum,
        It's called by update_parameters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''

        self.break_point = 0.5*(self.notch_left+self.notch_right+self.notch_smooth_left-self.notch_smooth_right)


        self.m1pr =_cmp.BrokenPowerLawDoubleGaussian_math(min_pl=self.mmin,max_pl=self.mmax,lambda_g=self.lambda_g,lambda_g_0=self.lambda_g_0,
                                                    mu_g_0=self.mu_g_0,sigma_g_0=self.sigma_g_0,mu_g_1=self.mu_g_1,
                                                    sigma_g_1=self.sigma_g_1,min_g=self.mmin,max_g=self.mu_g_1+5*self.sigma_g_1,
                                                    alpha_1=-self.alpha_1,alpha_2=-self.alpha_2,break_point=self.break_point,device=self.device)
        self.m2pr = _cmp.BrokenPowerLawDoubleGaussian_math(min_pl=self.mmin,max_pl=self.mmax,lambda_g=self.lambda_g,lambda_g_0=self.lambda_g_0,
                                                    mu_g_0=self.mu_g_0,sigma_g_0=self.sigma_g_0,mu_g_1=self.mu_g_1,
                                                    sigma_g_1=self.sigma_g_1,min_g=self.mmin,max_g=self.mu_g_1+5*self.sigma_g_1,
                                                    alpha_1=-self.alpha_1,alpha_2=-self.alpha_2,break_point=self.break_point,device=self.device)

        self.mdis={'mass_1': _cmp.SmoothedDipProb(origin_prob=self.m1pr,right_smooth=self.delta_m_high_pass,
                                                  left_smooth=self.delta_m_low_pass,A=self.A,notch_lower=self.notch_left,
                                                  notch_lower_smooth=self.notch_smooth_left,notch_upper=self.notch_right,notch_upper_smooth=self.notch_smooth_right,device=self.device),
                      'mass_2': _cmp.SmoothedDipProb(origin_prob=self.m2pr,right_smooth=self.delta_m_high_pass,
                                                  left_smooth=self.delta_m_low_pass,A=self.A,notch_lower=self.notch_left,
                                                  notch_lower_smooth=self.notch_smooth_left,notch_upper=self.notch_right,notch_upper_smooth=self.notch_smooth_right,device=self.device)}
        
        self.mmax = self.mdis['mass_1'].maximum
        self.mmin = self.mmin
        self.mmax2 = self.mmax

        # generate samples for normalisation 
        self.m1_samp, self.m2_samp = self.base_sample(10000)
        
    @staticmethod
    def grid_constraint_call(constraint_grid, values, parameter_grid):
        new_grid = peaks_grid_constraint(constraint_grid, values, parameter_grid)
        return new_grid
    
    @staticmethod
    def sampling_constraint_call(prior_dict):
        new_dict = peaks_sampling_constraint(prior_dict)
        return new_dict

    def base_sample(self, Nsample):
        '''
        Samples from the probability distribution. Assumes m1 and m2 are independent for use with pairing function.
        
        Parameters
        ----------
        Nsample: int
            Number of samples to generate
        
        Returns
        -------
        Samples: array_like
    
        '''
    
        # Create linspaces as torch tensors on the correct device
        sarray_1 = torch.linspace(self.mdis['mass_1'].minimum, self.mdis['mass_1'].maximum, 10000, device=self.device, dtype=self.dtype)
        sarray_2 = torch.linspace(self.mdis['mass_2'].minimum, self.mdis['mass_2'].maximum, 10000, device=self.device, dtype=self.dtype)
    
        # Evaluate CDFs using torch
        cdfeval_1 = self.mdis['mass_1'].cdf(sarray_1)
        cdfeval_2 = self.mdis['mass_2'].cdf(sarray_2)
    
        # Generate random CDF values as torch tensors on the correct device
        randomcdf_1 = torch.rand(Nsample, device=self.device, dtype=self.dtype)
        randomcdf_2 = torch.rand(Nsample, device=self.device, dtype=self.dtype)
    
        # Use torch_interp for interpolation
        samples_1 = torch_interp(randomcdf_1, cdfeval_1, sarray_1)
        samples_2 = torch_interp(randomcdf_2, cdfeval_2, sarray_2)

        return samples_1, samples_2
            
class multipopulation_pairing_func(multipopulation_broken_powerlaw_multi_peak_gaussian):
    """
    Child class of multipopulation_broken_powerlaw_multi_peak_gaussian which also inherits from m_priors.
    
    Adds a simple powerlaw pairing function to the distribution, with index beta

    Parameters
    ----------
    beta: float
        powerlaw index for pairing function
    """

    def __init__(self, alpha_1=1.0, alpha_2=3.0, beta=0.81, mmin=1.0, mmax=112.5, lambda_g=0.3,
                  lambda_g_0=0.4, mu_g_0=15.0, sigma_g_0=3.88, mu_g_1=32.27, sigma_g_1=3.88, 
                  delta_m_low_pass=1, delta_m_high_pass=0.1, A=0, notch_left=3.0, notch_right=5.0, 
                  notch_smooth_left=1, notch_smooth_right=0.5,device="cpu"):
        super().__init__(alpha_1, alpha_2, mmin, mmax, lambda_g, lambda_g_0, mu_g_0, sigma_g_0, 
                         mu_g_1, sigma_g_1, delta_m_low_pass, delta_m_high_pass, A, notch_left, notch_right, notch_smooth_left, notch_smooth_right, device)
        
        self.update_parameters(param_dict={'alpha_1':alpha_1,'alpha_2':alpha_2,'beta':beta,'mmin':mmin,'mmax':mmax,'lambda_g':lambda_g,'lambda_g_0':lambda_g_0,'mu_g_0':mu_g_0,'sigma_g_0':sigma_g_0,
                                           'mu_g_1':mu_g_1,'sigma_g_1':sigma_g_1,'delta_m_low_pass':delta_m_low_pass,'delta_m_high_pass':delta_m_high_pass,
                                           'A':A,'notch_left':notch_left,'notch_right':notch_right,'notch_smooth_left':notch_smooth_left,'notch_smooth_right':notch_smooth_right})


    def pairing_wrapper(self,ms1,ms2):
        """
        Wrapper function for pairing function
        """
        return pairing_func(ms1, ms2, self.beta, device=self.device)

    def joint_prob(self, ms1, ms2):
        """
        Calculates the joint probability of the masses according to the pairing function. Replaces parent joint probability method.
        Parent joint probability:
        :math: `p(m_1,m_2) = p(m_1)p(m_2|m_1)
        Pairing function joint probability:
        :math: `p(m_1,m_2) = p(m_1)p(m_2)Q(m_1,m_2)`
        Where Q(m_1,m_2) denotes the pairing function.

        Parameters
        ----------
        ms1 : array_like
            Primary mass samples.
        ms2 : array_like
            Secondary mass samples.

        Returns
        -------
        array_like
            Joint probability of the masses.
        """
        self.paired_dist = _cmp.PairingFunc(self.mdis,self.pairing_wrapper, self.m1_samp,self.m2_samp, device = self.device)

        return self.paired_dist.prob(ms1,ms2)

    def sample(self, Nsample):
        """
        Samples from the probability distribution, including the pairing function.
        """
        self.paired_dist = _cmp.PairingFunc(self.mdis,self.pairing_wrapper, self.m1_samp,self.m2_samp, device = self.device)

        return self.paired_dist.sample(Nsample)
    

class multipopulation_pairing_func_broken(multipopulation_broken_powerlaw_multi_peak_gaussian):
    """
    Child class of multipopulation_broken_powerlaw_multi_peak_gaussian which also inherits from m_priors.
    
    Adds a broken powerlaw pairing function to the distribution, with indices beta_1 and beta_2

    Parameters
    ----------
    beta_1: float
        powerlaw index for pairing function below break point
    beta_2: float
        powerlaw index for pairing function above break point
    """

    def __init__(self, alpha_1=1.0, alpha_2=3.0, beta_1=0.81, beta_2=1.11, mmin=1.0, mmax=112.5, lambda_g=0.3,
                  lambda_g_0=0.4, mu_g_0=15.0, sigma_g_0=3.88, mu_g_1=32.27, sigma_g_1=3.88, 
                  delta_m_low_pass=1., delta_m_high_pass=0.1, A=0., notch_left=3.0, notch_right=5.0, 
                  notch_smooth_left=1., notch_smooth_right=0.5,device="cpu"):
        super().__init__(alpha_1, alpha_2, mmin, mmax, lambda_g, lambda_g_0, mu_g_0, sigma_g_0, 
                         mu_g_1, sigma_g_1, delta_m_low_pass, delta_m_high_pass, A, notch_left, notch_right, notch_smooth_left, notch_smooth_right,device)
        
        self.update_parameters(param_dict={'alpha_1':alpha_1,'alpha_2':alpha_2,'beta_1':beta_1,'beta_2':beta_2,'mmin':mmin,'mmax':mmax,'lambda_g':lambda_g,'lambda_g_0':lambda_g_0,'mu_g_0':mu_g_0,'sigma_g_0':sigma_g_0,
                                           'mu_g_1':mu_g_1,'sigma_g_1':sigma_g_1,'delta_m_low_pass':delta_m_low_pass,'delta_m_high_pass':delta_m_high_pass,
                                           'A':A,'notch_left':notch_left,'notch_right':notch_right,'notch_smooth_left':notch_smooth_left,'notch_smooth_right':notch_smooth_right})

    def pairing_wrapper(self,ms1,ms2):
        """
        Wrapper function for pairing function
        """
        return pairing_func_broken(ms1, ms2, self.beta_1, self.beta_2, self.break_point, device = self.device)
    
    def joint_prob(self, ms1, ms2):
       
        """
        Calculates the joint probability of the masses according to the pairing function. Replaces parent joint probability method.
        Parent joint probability:
        :math: `p(m_1,m_2) = p(m_1)p(m_2|m_1)
        Pairing function joint probability:
        :math: `p(m_1,m_2) = p(m_1)p(m_2)Q(m_1,m_2)`
        Where Q(m_1,m_2) denotes the pairing function.

        Parameters
        ----------
        ms1 : array_like
            Primary mass samples.
        ms2 : array_like
            Secondary mass samples.

        Returns
        -------
        array_like
            Joint probability of the masses.
        """
        self.paired_dist = _cmp.PairingFunc(self.mdis,self.pairing_wrapper, self.m1_samp,self.m2_samp, device = self.device)

        return self.paired_dist.prob(ms1,ms2)

    def sample(self, Nsample):
        """
        Samples from the probability distribution, including the pairing function.
        """
        self.paired_dist = _cmp.PairingFunc(self.mdis,self.pairing_wrapper, self.m1_samp,self.m2_samp, device = self.device)

        return self.paired_dist.sample(Nsample)

class multipopulation_broken_powerlaw_triple_peak_gaussian(m_priors):
    """
    Child class which builds a multipopulation model covering whole CBC range. This has two identical distribtuions in m1 and m2,
    with both being BPL+3G+Dip with smoothing.

    This cannot be directly specified as a mass model in the command line.

    Parameters
    ----------
    alpha_1 : float
        Slope of the power law for masses below the break point.
    alpha_2 : float
        Slope of the power law for masses above the break point.
    mmin : float
        Minimum mass of the distribution.
    mmax : float
        Maximum mass of the distribution.
    lambda_g : float
        Scaling parameter for the overall Gaussian component.
    lambda_g_0 : float
        Scaling parameter for the 1st Gaussian peak.
    mu_g_0 : float
        Mean of the 1st Gaussian peak.
    sigma_g_0 : float
        Standard deviation of the 1st Gaussian peak.
    lambda_g_1 : float
        Scaling parameter for the 2nd Gaussian peak.    
    mu_g_1 : float
        Mean of the 2nd Gaussian peak.
    sigma_g_1 : float
        Standard deviation of the 2nd Gaussian peak.
    mu_g_2 : float
        Mean of the 3rd Gaussian peak.
    sigma_g_2 : float
        Standard deviation of the 3rd Gaussian peak.    
    delta_m_low_pass : float
        Smoothing parameter for the high mass end of distribution.
    delta_m_high_pass : float
        Smoothing parameter for the low mass end of distribution.
    A : float
        Amplitude of the notch feature.
    notch_left : float
        Left boundary of the notch region.
    notch_right : float
        Right boundary of the notch region.
    notch_smooth_left : float
        Smoothing factor on the left side of the notch.
    notch_smooth_right : float
        Smoothing factor on the right side of the notch.
    """

    def __init__(self, alpha_1=1.0, alpha_2=3.0, mmin=1.0, mmax=112.5, lambda_g=0.3,
                  lambda_g_0=0.4, mu_g_0=15.0, sigma_g_0=3.88, lambda_g_1 = 0.5, mu_g_1=32.27, sigma_g_1=3.88, mu_g_2=60.0,sigma_g_2=5.0,
                  delta_m_low_pass=1, delta_m_high_pass=0.1, A=0, notch_left=3.0, notch_right=5.0, 
                  notch_smooth_left=1, notch_smooth_right=0.5,device="cpu"):
        super().__init__(device=device)
        self.update_parameters(param_dict={'alpha_1':alpha_1,'alpha_2':alpha_2,'mmin':mmin,'mmax':mmax,'lambda_g':lambda_g,'lambda_g_0':lambda_g_0,'mu_g_0':mu_g_0,'sigma_g_0':sigma_g_0,
                                           'lambda_g_1':lambda_g_1, 'mu_g_1':mu_g_1,'sigma_g_1':sigma_g_1,'mu_g_2':mu_g_2,'sigma_g_2':sigma_g_2,'delta_m_low_pass':delta_m_low_pass,'delta_m_high_pass':delta_m_high_pass,
                                           'A':A,'notch_left':notch_left,'notch_right':notch_right,'notch_smooth_left':notch_smooth_left,'notch_smooth_right':notch_smooth_right})
    def update_mass_priors(self):
        ''' 
        This method creates a dictionary of mass distributions objects.         
        It sets the maximum value of the primary mass distribution mmax to self.mdis['mass_1'].maximum, 
        It's called by update_parameters everytime the mass priors parameters are changed.
        Every mass priors model has a different implementation because the distributions are different,
        and mmax, mmin and mmax2 definitions depend on the mass prior model.          
        '''

        self.break_point = 0.5*(self.notch_left+self.notch_right+self.notch_smooth_left-self.notch_smooth_right)


        self.m1pr =_cmp.BrokenPowerLawTripleGaussian_math(min_pl=self.mmin,max_pl=self.mmax,lambda_g=self.lambda_g,lambda_g_0=self.lambda_g_0,
                                                    mu_g_0=self.mu_g_0,sigma_g_0=self.sigma_g_0,lambda_g_1=self.lambda_g_1,mu_g_1=self.mu_g_1,
                                                    sigma_g_1=self.sigma_g_1,mu_g_2=self.mu_g_2,sigma_g_2=self.sigma_g_2,min_g=self.mmin,max_g=self.mu_g_2+5*self.sigma_g_2,
                                                    alpha_1=-self.alpha_1,alpha_2=-self.alpha_2,break_point=self.break_point,device=self.device)
        self.m2pr = _cmp.BrokenPowerLawTripleGaussian_math(min_pl=self.mmin,max_pl=self.mmax,lambda_g=self.lambda_g,lambda_g_0=self.lambda_g_0,
                                                    mu_g_0=self.mu_g_0,sigma_g_0=self.sigma_g_0,lambda_g_1=self.lambda_g_1,mu_g_1=self.mu_g_1,
                                                    sigma_g_1=self.sigma_g_1,mu_g_2=self.mu_g_2,sigma_g_2=self.sigma_g_2,min_g=self.mmin,max_g=self.mu_g_2+5*self.sigma_g_2,
                                                    alpha_1=-self.alpha_1,alpha_2=-self.alpha_2,break_point=self.break_point,device=self.device)

        self.mdis={'mass_1': _cmp.SmoothedDipProb(origin_prob=self.m1pr,right_smooth=self.delta_m_high_pass,
                                                  left_smooth=self.delta_m_low_pass,A=self.A,notch_lower=self.notch_left,
                                                  notch_lower_smooth=self.notch_smooth_left,notch_upper=self.notch_right,notch_upper_smooth=self.notch_smooth_right,device=self.device),
                      'mass_2': _cmp.SmoothedDipProb(origin_prob=self.m2pr,right_smooth=self.delta_m_high_pass,
                                                  left_smooth=self.delta_m_low_pass,A=self.A,notch_lower=self.notch_left,
                                                  notch_lower_smooth=self.notch_smooth_left,notch_upper=self.notch_right,notch_upper_smooth=self.notch_smooth_right,device=self.device)}
        
        self.mmax = self.mdis['mass_1'].maximum
        self.mmin = self.mmin
        self.mmax2 = self.mmax

        # generate samples for normalisation 
        self.m1_samp, self.m2_samp = self.base_sample(10000)
        
    @staticmethod
    def grid_constraint_call(constraint_grid, values, parameter_grid):
        new_grid = peaks_grid_constraint(constraint_grid, values, parameter_grid)
        return new_grid
    
    @staticmethod
    def sampling_constraint_call(prior_dict):
        new_dict = peaks_sampling_constraint(prior_dict)
        return new_dict

    def base_sample(self, Nsample):
        '''
        Samples from the probability distribution. Assumes m1 and m2 are independent for use with pairing function.
        
        Parameters
        ----------
        Nsample: int
            Number of samples to generate
        
        Returns
        -------
        Samples: array_like
        '''
        # Create linspaces as torch tensors on the correct device
        sarray_1 = torch.linspace(self.mdis['mass_1'].minimum, self.mdis['mass_1'].maximum, 10000, device=self.device)
        sarray_2 = torch.linspace(self.mdis['mass_1'].minimum, self.mdis['mass_1'].maximum, 10000, device=self.device)
    
        # Evaluate CDFs using torch
        cdfeval_1 = self.mdis['mass_1'].cdf(sarray_1)
        cdfeval_2 = self.mdis['mass_2'].cdf(sarray_2)
    
        # Generate random CDF values as torch tensors on the correct device
        randomcdf_1 = torch.rand(Nsample, device=self.device)
        randomcdf_2 = torch.rand(Nsample, device=self.device)
    
        # Use torch_interp for interpolation
        samples_1 = torch_interp(randomcdf_1, cdfeval_1, sarray_1)
        samples_2 = torch_interp(randomcdf_2, cdfeval_2, sarray_2)

        return samples_1, samples_2

class multipopulation_pairing_func_triple_peak(multipopulation_broken_powerlaw_triple_peak_gaussian):
    """
    Child class of multipopulation_broken_powerlaw_triple_peak_gaussian which also inherits from m_priors.
    
    Adds a simple powerlaw pairing function to the distribution, with index beta

    Parameters
    ----------
    beta_1: float
        powerlaw index 1 for pairing function
    beta_2: float
        powerlaw index 2 for pairing function
    """

    def __init__(self, alpha_1=1.0, alpha_2=3.0, beta = 0.81, mmin=1.0, mmax=112.5, lambda_g=0.3,
                  lambda_g_0=0.4, mu_g_0=15.0, sigma_g_0=3.88, lambda_g_1 = 0.5, mu_g_1=32.27, sigma_g_1=3.88, mu_g_2=60.0,sigma_g_2=5.0,
                  delta_m_low_pass=1, delta_m_high_pass=0.1, A=0, notch_left=3.0, notch_right=5.0, 
                  notch_smooth_left=1, notch_smooth_right=0.5, device = "cpu"):
        super().__init__(alpha_1, alpha_2, mmin, mmax, lambda_g, lambda_g_0, mu_g_0, sigma_g_0, lambda_g_1,
                         mu_g_1, sigma_g_1, mu_g_2, sigma_g_2, delta_m_low_pass, delta_m_high_pass, A, notch_left, notch_right, notch_smooth_left, notch_smooth_right, device)
        
        self.update_parameters(param_dict={'alpha_1':alpha_1,'alpha_2':alpha_2,'beta':beta,'mmin':mmin,'mmax':mmax,'lambda_g':lambda_g,'lambda_g_0':lambda_g_0,'mu_g_0':mu_g_0,'sigma_g_0':sigma_g_0,
                                           'lambda_g_1':lambda_g_1, 'mu_g_1':mu_g_1,'sigma_g_1':sigma_g_1,'mu_g_2':mu_g_2,'sigma_g_2':sigma_g_2,'delta_m_low_pass':delta_m_low_pass,'delta_m_high_pass':delta_m_high_pass,
                                           'A':A,'notch_left':notch_left,'notch_right':notch_right,'notch_smooth_left':notch_smooth_left,'notch_smooth_right':notch_smooth_right})

    def pairing_wrapper(self,ms1,ms2):
        """
        Wrapper function for pairing function
        """
        return pairing_func(ms1, ms2, self.beta)

    def joint_prob(self, ms1, ms2):
        """
        Calculates the joint probability of the masses according to the pairing function. Replaces parent joint probability method.
        Parent joint probability:
        :math: `p(m_1,m_2) = p(m_1)p(m_2|m_1)
        Pairing function joint probability:
        :math: `p(m_1,m_2) = p(m_1)p(m_2)Q(m_1,m_2)`
        Where Q(m_1,m_2) denotes the pairing function.

        Parameters
        ----------
        ms1 : array_like
            Primary mass samples.
        ms2 : array_like
            Secondary mass samples.

        Returns
        -------
        array_like
            Joint probability of the masses.
        """
        self.paired_dist = _cmp.PairingFunc(self.mdis,self.pairing_wrapper, self.m1_samp,self.m2_samp, device = self.device)

        return self.paired_dist.prob(ms1,ms2)

    def sample(self, Nsample):
        """
        Samples from the probability distribution, including the pairing function.
        """
        self.paired_dist = _cmp.PairingFunc(self.mdis,self.pairing_wrapper, self.m1_samp,self.m2_samp, device = self.device)

        return self.paired_dist.sample(Nsample)
    
class multipopulation_pairing_func_broken_triple_peak(multipopulation_broken_powerlaw_triple_peak_gaussian):
    """
    Child class of multipopulation_broken_powerlaw_multi_peak_gaussian which also inherits from m_priors.
    
    Adds a broken powerlaw pairing function to the distribution, with indices beta_1 and beta_2

    Parameters
    ----------
    beta_1: float
        powerlaw index for pairing function below break point
    beta_2: float
        powerlaw index for pairing function above break point
    """

    def __init__(self, alpha_1=1.0, alpha_2=3.0, beta_1= 0.81, beta_2= 0.81, mmin=1.0, mmax=112.5, lambda_g=0.3,
                  lambda_g_0=0.4, mu_g_0=15.0, sigma_g_0=3.88, lambda_g_1 = 0.5, mu_g_1=32.27, sigma_g_1=3.88, mu_g_2=60.0,sigma_g_2=5.0,
                  delta_m_low_pass=1, delta_m_high_pass=0.1, A=0, notch_left=3.0, notch_right=5.0, 
                  notch_smooth_left=1, notch_smooth_right=0.5, device = "cpu"):
        super().__init__(alpha_1, alpha_2, mmin, mmax, lambda_g, lambda_g_0, mu_g_0, sigma_g_0, lambda_g_1,
                         mu_g_1, sigma_g_1, mu_g_2, sigma_g_2, delta_m_low_pass, delta_m_high_pass, A, notch_left, notch_right, notch_smooth_left, notch_smooth_right, device)
        
        self.update_parameters(param_dict={'alpha_1':alpha_1,'alpha_2':alpha_2,'beta_1':beta_1,'beta_2':beta_2,'mmin':mmin,'mmax':mmax,'lambda_g':lambda_g,'lambda_g_0':lambda_g_0,'mu_g_0':mu_g_0,'sigma_g_0':sigma_g_0,
                                           'lambda_g_1':lambda_g_1, 'mu_g_1':mu_g_1,'sigma_g_1':sigma_g_1,'mu_g_2':mu_g_2,'sigma_g_2':sigma_g_2,'delta_m_low_pass':delta_m_low_pass,'delta_m_high_pass':delta_m_high_pass,
                                           'A':A,'notch_left':notch_left,'notch_right':notch_right,'notch_smooth_left':notch_smooth_left,'notch_smooth_right':notch_smooth_right})

    def pairing_wrapper(self,ms1,ms2):
        """
        Wrapper function for pairing function
        """
        return pairing_func_broken(ms1, ms2, self.beta_1, self.beta_2, self.break_point, device = self.device)
    
    def joint_prob(self, ms1, ms2):
       
        """
        Calculates the joint probability of the masses according to the pairing function. Replaces parent joint probability method.
        Parent joint probability:
        :math: `p(m_1,m_2) = p(m_1)p(m_2|m_1)
        Pairing function joint probability:
        :math: `p(m_1,m_2) = p(m_1)p(m_2)Q(m_1,m_2)`
        Where Q(m_1,m_2) denotes the pairing function.

        Parameters
        ----------
        ms1 : array_like
            Primary mass samples.
        ms2 : array_like
            Secondary mass samples.

        Returns
        -------
        array_like
            Joint probability of the masses.
        """
        self.paired_dist = _cmp.PairingFunc(self.mdis,self.pairing_wrapper, self.m1_samp,self.m2_samp, device = self.device)

        return self.paired_dist.prob(ms1,ms2)

    def sample(self, Nsample):
        """
        Samples from the probability distribution, including the pairing function.
        """
        self.paired_dist = _cmp.PairingFunc(self.mdis,self.pairing_wrapper, self.m1_samp,self.m2_samp, device = self.device)

        return self.paired_dist.sample(Nsample)

