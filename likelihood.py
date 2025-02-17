#%%

import numpy as np
import matplotlib.pyplot as plt
import bilby
from mock_event_maker import MockEvent
import pandas as pd
from mock_catalog_maker import MockCatalog
from utils import make_nice_plots


"""
To estimate f_agn, we can calculate the likelihood at several values, given a realization of measurements (=pdf) and from this find CL

That is, draw from large reservoir of skymaps according to f_agn (and maybe also selection function???)
calculate likelihood for f_agn_trials = np.linspace(0, 1, 1000) (or less and interpolate).

Better would be to have e.g. a bilby routine which could be more easily expanded to more population parameters
In that case, the trials are done by the walkers.

The crossmatch of event parameters must be done inside a sampler when inferring population params, 
because the p_agn and p_alt change according to the population params. This is not the case for f_agn, which is the easiest to compute.
=> Crossmatching with catalog can be precomputed (ask Lorenzo again what he meant by the joint prob p(z, Omega, M) )
"""


def fagn_fixedpop(fagn, total_prob_agn: np.ndarray, total_prob_alt: np.ndarray, events: MockEvent):
    """
    Likelihood for estimating ONLY f_agn, while all other population parameters remain fixed.
    """
    
    try:
        total_prob_agn = np.array(total_prob_agn)[:, np.newaxis]
        total_prob_alt = np.array(total_prob_alt)[:, np.newaxis]
    except IndexError:  # If only one GW is used
        total_prob_agn = np.array([total_prob_agn])[:, np.newaxis]
        total_prob_alt = np.array([total_prob_alt])[:, np.newaxis]

    fagn_times_fc_fcl_fobsc = fagn * events.skymap_cl * events.MockCatalog.completeness  # TODO: obscuration
    log_detection_efficiency = 0  # TODO: gw selection effects
    # print(np.log(fagn_times_fc_fcl_fobsc * total_prob_agn[:, np.newaxis] + (1 - fagn_times_fc_fcl_fobsc) * total_prob_alt[:, np.newaxis]))
    return np.sum( np.log(fagn_times_fc_fcl_fobsc * total_prob_agn + (1 - fagn_times_fc_fcl_fobsc) * total_prob_alt), axis=0 ) - log_detection_efficiency**len(total_prob_agn)


def _get_population_param_priors(parameter_dict):
    priors = {}

    for key in parameter_dict:
        if isinstance(parameter_dict[key]['value'], list):
            if "label" in parameter_dict[key]:
                label = parameter_dict[key]['label']
            else:
                label = key
            if parameter_dict[key]['prior'].lower() == 'uniform':
                priors[key] = bilby.core.prior.Uniform(parameter_dict[key]['value'][0], parameter_dict[key]['value'][1], name=key, latex_label=label)
            elif parameter_dict[key]['prior'].lower() == 'gaussian':
                priors[key] = bilby.core.prior.Gaussian(parameter_dict[key]['value'][0], parameter_dict[key]['value'][1], name=key, latex_label=label)
            else:
                raise ValueError(f"Unrecognised prior settings for parameter {key} (specify 'Uniform' or 'Gaussian'.")
        else:
            priors[key] = float(parameter_dict[key]['value'])

    # return priors_agn, priors_alt  # TODO: use the hypothesis key in the parameter dict to distinguish the priors and add _agn or _alt to parameter names and make this consistent with the labels explicitly written in bilby likelihood (or do + _agn and + _alt in bilby as well, which should then be correct)


# class PopulationLikelihood(bilby.likelihood):  # TODO: Could be that we can recycle large parts of gwcosmo here if we also do pixelation

#     def __init__(self, data, skymap_cl=1, completeness=1):  # TODO: maybe add param_dict? gwcosmo does posterior_samples_dictionary and skymap_dictionary
#         """

#         Keep skymap_cl = 1 if you don't want to use sky location as a parameter. 
#         In this case, data['skyprob_agn'] and data['skyprob_alt'] should be equal to 1. TODO: again check how to do missing parameters correctly

#         Parameters
#         ----------
#         data: array_like
#             The data to analyse
#         """
#         super().__init__(parameters={"fagn": None})  # TODO: add all possible params currently implemented
#         self.data = data
#         self.skymap_cl = skymap_cl  # TODO: Probably just pass a MockEvent type and from its param_dict property infer everything we need, including if skymaps are even necessary
#         self.completeness = completeness


#     def log_likelihood(self):
#         """TODO: test + include other params + include completeness"""
#         fagn_times_cl = self.parameters["fagn"] * self.skymap_cl * self.completeness
#         llh = np.log(fagn_times_cl * self.data["skyprob_agn"] + (1 - fagn_times_cl) * self.data["skyprob_alt"])
#         return np.sum(llh)

#         # mu = self.parameters["mu"]
#         # sigma = self.parameters["sigma"]
#         # res = self.data - mu
#         # return -0.5 * (
#         #     np.sum((res / sigma) ** 2) + self.N * np.log(2 * np.pi * sigma**2)
#         # )


if __name__ == '__main__':
    make_nice_plots()

    N_TOT = 0
    GRID_SIZE = 40  # Radius of the whole grid in redshift
    GW_BOX_SIZE = 30  # Radius of the GW box in redshift
    
    Catalog = MockCatalog(n_agn=N_TOT,
                            max_redshift=GRID_SIZE,
                            gw_box_radius=GW_BOX_SIZE,
                            completeness=1)

    n_events = 100
    f_agn = 0.5
    cl = 0.999

    # The model dictionary should contain choices of implemented models. These models have parameters for which we pass another dictionary
    # TODO: think about what happens when we want to do the chi_eff vs q correlation here (mass ratios are used in the mass models already)
    model_dict = {'mass': ['PrimaryMass_gaussian', 'PrimaryMass_gaussian']}
    
    # These dicts should contain a selection of allowed population parameters: only parameters that are implemented in models are allowed
    param_dict_agn = {
            'mminbh': 6.,
            'mu_g': 15.,
            'sigma_g': 1.
        }
    
    param_dict_alt = {
            'mminbh': 6.,
            'mu_g': 50.,
            'sigma_g': 5.
        }

    
    GWEvents = MockEvent(
                        model_dict=model_dict,
                        n_events=n_events,
                        f_agn=f_agn,
                        param_dict_agn=param_dict_agn,
                        param_dict_alt=param_dict_alt,
                        catalog=Catalog, 
                        skymap_cl=cl
                    )

    GWEvents.get_posteriors()

    pmax = 1000
    pmin = 0.01
    N = 10
    f_agn_arr = np.linspace(0.01, 1., 100)
    fake_agn_prob = np.array([1.])
    fake_alt_prob = np.array([0.])
    # fake_agn_prob = np.tile(pmin, N)
    # fake_agn_prob[:N//2] = pmax
    # fake_alt_prob = np.tile(pmax, N)
    # fake_alt_prob[:N//2] = pmin

    log_llh = fagn_fixedpop(f_agn_arr, fake_agn_prob, fake_alt_prob, GWEvents)
    
    plt.figure(figsize=(8,6))
    plt.plot(f_agn_arr, log_llh)
    plt.show()

    #%%

    # f_agn_prior = {
    #             'prior': 'Uniform',
    #             'value': [0., 1.],
    #             'label': r'$f_{\rm agn}$',
    #             'description': 'Fraction of GW events coming from AGN'
    #         }
    
    # # These dicts should contain a selection of allowed population parameters: only parameters that are implemented in models are allowed
    # param_dict_agn = {
    #         'mminbh': {
    #             'prior': 'Uniform',
    #             'value': 6.,
    #             'label': r'$M_{\rm min, agn}$',
    #             'description': None
    #         },
    #         'mu_g': {
    #             'prior': 'Uniform',
    #             'value': [15., 20.],
    #             'label': r'$\mu_{\rm agn}$',
    #             'description': None
    #         },
    #         'sigma_g': {
    #             'prior': 'Uniform',
    #             'value': 1.,
    #             'label': r'$\sigma_{\rm agn}$',
    #             'description': None
    #         }
    #     }
    
    # param_dict_alt = {
    #         'mminbh': {
    #             'prior': 'Uniform',
    #             'value': 6.,
    #             'label': r'$M_{\rm min, alt}$',
    #             'description': None
    #         },
    #         'mu_g': {
    #             'prior': 'Uniform',
    #             'value': [40., 100.],
    #             'label': r'$\mu_{\rm alt}$',
    #             'description': None
    #         },
    #         'sigma_g': {
    #             'prior': 'Uniform',
    #             'value': 5.,
    #             'label': r'$\sigma_{\rm alt}$',
    #             'description': None
    #         }
    #     }