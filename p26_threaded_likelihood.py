from p26_control_room import *
import glob
from utils import uniform_shell_sampler
from ligo.skymap.io.fits import read_sky_map
from p26_crossmatch import crossmatch_p26 as crossmatch
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


all_gw_fnames = np.array(glob.glob(SKYMAP_DIR + 'skymap_*'))
gw_fnames_per_realization = np.random.choice(all_gw_fnames, size=(BATCH, N_TRUE_FAGNS), replace=False)  # Only unique GWs for a single data set

all_true_sources = np.genfromtxt('/home/lucas/Documents/PhD/true_r_theta_phi_all.txt', delimiter=',')
all_true_sources = all_true_sources[all_true_sources[:, 0].argsort()]
true_source_identifiers = all_true_sources[:,0]


def process_one_fagn(fagn_idx, fagn_true):
    # s = time.time()
    gw_fnames = gw_fnames_per_realization[:,fagn_idx]

    ### Get true source coordinates for GWs from AGN to put in the AGN catalog ###

    gw_fnames_from_agn = gw_fnames[:round(fagn_true * BATCH)]
    gw_identifiers = sorted(np.array([f[-13:-8] for f in gw_fnames_from_agn]).astype(int))
    true_sources = all_true_sources[np.searchsorted(true_source_identifiers, gw_identifiers)]
    agn_ra, agn_dec, agn_rcom = true_sources[:,3], 0.5 * np.pi - true_sources[:,2], true_sources[:,1]

    ### Complete catalog to preserve uniform in comoving volume distribution ###
    n2complete = int(round(len(agn_ra) * ( (AGN_COMDIST_MAX / COMDIST_MAX)**3 - 1)))
    new_rcom, new_theta, new_phi = uniform_shell_sampler(COMDIST_MIN, AGN_COMDIST_MAX, n2complete)
    agn_ra = np.append(agn_ra, new_phi)
    agn_dec = np.append(agn_dec, np.pi * 0.5 - new_theta)
    agn_rcom = np.append(agn_rcom, new_rcom)
    if VERBOSE:
        print(f'Adding {n2complete} AGN to get a uniform catalog.')
    ############################################################################

    if ADD_NAGN_TO_CAT > n2complete:  # Add uncorrelated AGN as background
        if VERBOSE:
            print(f'Adding {ADD_NAGN_TO_CAT - n2complete} more AGN to get to the requested number of AGN.')
        new_rcom, new_theta, new_phi = uniform_shell_sampler(COMDIST_MIN, AGN_COMDIST_MAX, ADD_NAGN_TO_CAT - n2complete)
        agn_ra = np.append(agn_ra, new_phi)
        agn_dec = np.append(agn_dec, np.pi * 0.5 - new_theta)
        agn_rcom = np.append(agn_rcom, new_rcom)
    if len(agn_rcom) == 0:
        obs_agn_redshift, agn_redshift_err = np.empty_like(agn_rcom), np.empty_like(agn_rcom)
    else:
        obs_agn_redshift, agn_redshift_err = get_observed_redshift_from_rcom(agn_rcom)
    obs_agn_rlum = COSMO.luminosity_distance(obs_agn_redshift).value

    ### Make an incomplete AGN catalog from these coordinates ###

    if not ASSUME_PERFECT_REDSHIFT:
        # Measure completeness in the surveyed sky area
        if MASK_GALACTIC_PLANE:
            latitude_mask, _ = make_latitude_selection(agn_ra, agn_dec, obs_agn_rlum)
            _, _, sum_of_posteriors_complete = get_agn_posteriors_and_zprior_normalization(fagn_idx, obs_agn_redshift[latitude_mask], agn_redshift_err[latitude_mask], label='COMPLETE')
        else:
            _, _, sum_of_posteriors_complete = get_agn_posteriors_and_zprior_normalization(fagn_idx, obs_agn_redshift, agn_redshift_err, label='COMPLETE')

    incomplete_catalog_mask, c_per_zbin, completeness_map = make_incomplete_catalog(agn_ra, agn_dec, obs_agn_rlum, obs_agn_redshift)
    agn_ra = agn_ra[incomplete_catalog_mask]
    agn_dec = agn_dec[incomplete_catalog_mask]
    obs_agn_redshift = obs_agn_redshift[incomplete_catalog_mask]
    agn_redshift_err = agn_redshift_err[incomplete_catalog_mask]
    obs_agn_rlum = obs_agn_rlum[incomplete_catalog_mask]

    agn_posterior_dset, redshift_population_prior_normalization, sum_of_posteriors_incomplete = get_agn_posteriors_and_zprior_normalization(fagn_idx, obs_agn_redshift, agn_redshift_err, label='INCOMPLETE')
    
    if not ASSUME_PERFECT_REDSHIFT:    
        redshift_agn_selection_function = sum_of_posteriors_incomplete / sum_of_posteriors_complete
        redshift_completeness = interp1d(Z_INTEGRAL_AX, redshift_agn_selection_function, bounds_error=False, fill_value=0)
    else:
        def redshift_completeness(z, completeness_zvals=c_per_zbin):
            bin_idx = np.digitize(z, Z_EDGES) - 1
            bin_idx[bin_idx == len(completeness_zvals)] = len(completeness_zvals) - 1
            return completeness_zvals[bin_idx.astype(np.int32)]

    ### Calculate the integrals in the likelihood ###
    S_agn_incat = np.zeros(BATCH)
    S_agn_outofcat = np.zeros(BATCH)
    S_alt = np.zeros(BATCH)
    for gw_idx, filename in enumerate(gw_fnames):
        skymap = read_sky_map(filename, moc=True)
        
        sagn_incat, sagn_outofcat, salt = crossmatch(agn_posterior_dset=agn_posterior_dset,              # AGN data (needed when using AGN z-errors)
                                                    sky_map=skymap,                                     # GW data
                                                    completeness_map=completeness_map,                  # For getting the surveyed sky-area
                                                    redshift_completeness=redshift_completeness,        # Callable: redshift selection function
                                                    agn_ra=agn_ra,                                      # AGN data (needed when neglecting AGN z-errors)
                                                    agn_dec=agn_dec,                                    # AGN data (needed when neglecting AGN z-errors)
                                                    agn_lumdist=obs_agn_rlum,                           # AGN data (needed when neglecting AGN z-errors)
                                                    agn_redshift=obs_agn_redshift,                      # AGN data (needed when neglecting AGN z-errors)
                                                    agn_redshift_err=agn_redshift_err,                  # AGN data (needed when neglecting AGN z-errors)
                                                    skymap_cl=SKYMAP_CL,                                # Only analyze AGN within this CL, only for code speed-up
                                                    gw_zcut=ZMAX,                                       # GWs are not generated above ZMAX
                                                    z_integral_ax=Z_INTEGRAL_AX,                        # Integrating the likelihood in redshift space                                                                
                                                    assume_perfect_redshift=ASSUME_PERFECT_REDSHIFT,    # Integrating delta functions is handled differently
                                                    background_agn_distribution=AGN_ZPRIOR_FUNCTION,
                                                    merger_rate_func=MERGER_RATE_EVOLUTION,             # Merger rate can evolve
                                                    linax=LINAX,                                        # Integration can be done in linspace or in geomspace
                                                    **MERGER_RATE_KWARGS)                               # kwargs for  merger rate function

        if len(agn_ra) != 0:
            sagn_incat *= (4 * np.pi / redshift_population_prior_normalization)  # 4pi comes from uniform-on-sky parameter estimation prior and divide by the normalization of the redshift population prior: int dz Sum(p_red(z|z_obs)) p_rate(z) p_cut(z)

        S_agn_incat[gw_idx] = sagn_incat
        S_agn_outofcat[gw_idx] = sagn_outofcat
        S_alt[gw_idx] = salt
    
    ### Evaluate the likelihood ###

    S_agn_incat = S_agn_incat[~np.isnan(S_agn_incat)]
    S_agn_outofcat = S_agn_outofcat[~np.isnan(S_agn_outofcat)]
    S_alt = S_alt[~np.isnan(S_alt)]

    loglike = np.log(SKYMAP_CL * LOG_LLH_X_AX[None,:] * (S_agn_incat[:,None] + S_agn_outofcat[:,None] - S_alt[:,None]) + S_alt[:,None])
    
    nans = np.isnan(loglike)
    if np.sum(nans) != 0:
        print('Got NaNs:')
        print(S_agn_incat[:,None][nans])
        print(S_agn_outofcat[:,None][nans])
        print(S_alt[:,None][nans])
    # print(f'\n{fagn_idx} Done in {time.time() - s}\n')

    return fagn_idx, np.sum(loglike, axis=0)


log_llh = np.zeros((len(LOG_LLH_X_AX), N_TRUE_FAGNS))
with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:  # 1 proc, 11.5s/it - 8 proc, 14.5s/it - 16 proc, 25s/it - 32 proc, 58s/it
    futures = [executor.submit(process_one_fagn, fagn_idx, fagn_true) for fagn_idx, fagn_true in enumerate(REALIZED_FAGNS)]
    for future in tqdm(as_completed(futures)):
        fagn_idx, llh = future.result()
        log_llh[:,fagn_idx] = llh

fname = os.path.join(sys.path[0], FAGN_POSTERIOR_FNAME)
np.save(fname, log_llh)
print(f'Posterior is located at: {fname}')
