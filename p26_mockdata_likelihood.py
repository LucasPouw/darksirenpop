from p26_control_room import *
import glob
from utils import uniform_shell_sampler, sample_spherical_angles
from ligo.skymap.io.fits import read_sky_map
from p26_crossmatch import crossmatch_p26 as crossmatch
from p26_crossmatch import crossmatch_from_samples_p26
import sys
from scipy.integrate import simpson, romb
from concurrent.futures import ProcessPoolExecutor, as_completed


ALL_TRUE_SOURCES = ALL_TRUE_SOURCES[ALL_TRUE_SOURCES[:, 0].argsort()]
TRUE_SOURCE_IDENTIFIERS = ALL_TRUE_SOURCES[:,0]


if USE_SKYMAPS:
    all_gw_fnames = np.array(glob.glob(SKYMAP_DIR + 'skymap_*'))
    FILE_TYPE = 'skymap'
else:
    all_gw_fnames = np.array(glob.glob(SAMPLES_DIR + 'gw_*'))
    FILE_TYPE = 'samples'


def get_id_from_fname(fname):
    file_type = fname.split('/')[-1].split('_')[-4]
    if file_type == 'gw':
        return fname[-8:-3]
    elif file_type == 'skymap':
        return fname[-13:-8]
    else:
        sys.exit(f'Extracted the following file type from file name and did not recognize: {file_type}')
    return 


def get_fnames(ids, file_type):
    '''file_type either skymap or samples'''
    if file_type == 'samples':
        label = 'gw'
        dir = SAMPLES_DIR
    elif file_type == 'skymap':
        label = 'skymap'
        dir = SKYMAP_DIR
    else:
        sys.exit(f'Do not recognize file type: {file_type}. Choose between str(skymap) or str(samples).')

    fnames = []
    for id in ids:
        s = f'{dir}{label}_0_0_{id:05d}.fits.gz'
        fnames.append(s)
    return np.array(fnames)


def get_gw_fnames_resampled(fagn_realized):
    '''
    Currently assuming ALT GW hosts are always distributed uniform in comoving volume, but rate evolution can be set.

    Warning: Resampling of a finite amount of GW data will cause biases when analyzing many data realizations due to duplicate GWs
    '''
    
    agn_rcom = ALL_TRUE_SOURCES[:,1]
    agn_z = fast_z_at_value(COSMO.comoving_distance, agn_rcom * u.Mpc)

    # Make target GW-from-AGN population, which is normalized on Z_INTEGRAL_AX
    norm = romb(AGN_ZPRIOR_FUNCTION(Z_INTEGRAL_AX), dx=np.diff(Z_INTEGRAL_AX)[0])
    target_population = lambda z: AGN_ZPRIOR_FUNCTION(z) / norm

    weights = target_population(agn_z) / uniform_comoving_prior(agn_z)  # Divide out the distribution of the mock data
    if CORRECT_TIME_DILATION:
        weights *= 1 / (1 + agn_z)
    from_agn_population = np.random.choice(np.arange(len(agn_z)), p=weights / np.sum(weights), size=round(fagn_realized * BATCH))

    need_these = TRUE_SOURCE_IDENTIFIERS[from_agn_population].astype(int)
    gw_fnames_from_agn = get_fnames(need_these, file_type=FILE_TYPE)

    # GW-from-ALT population
    if not CORRECT_TIME_DILATION and (MERGER_RATE == 'uniform'):  # Target population is equal to mock data population
        gw_fnames_from_alt = np.random.choice(all_gw_fnames, size=BATCH - gw_fnames_from_agn.shape[0], replace=False)
    else:
        weights = merger_rate(agn_z, MERGER_RATE_EVOLUTION, **MERGER_RATE_KWARGS)
        if CORRECT_TIME_DILATION:
            weights *= 1 / (1 + agn_z)
        from_alt_population = np.random.choice(np.arange(len(agn_z)), p=weights / np.sum(weights), size=BATCH - gw_fnames_from_agn.shape[0])
        
        need_these = TRUE_SOURCE_IDENTIFIERS[from_alt_population].astype(int)
        gw_fnames_from_alt = get_fnames(need_these, file_type=FILE_TYPE)
    
    gw_fnames = np.append(gw_fnames_from_agn, gw_fnames_from_alt)

    return gw_fnames, gw_fnames_from_agn


def fill_catalog_to_complete(agn_ra, agn_dec, agn_rcom):
    '''To preserve overall distribution, need to add AGN above COMDIST_MAX, where the GW-hosting AGN are.'''
    if AGN_ZPRIOR == 'uniform_comoving_volume':
        n2complete = int(round(len(agn_ra) * ( (AGN_COMDIST_MAX / COMDIST_MAX)**3 - 1)))
        new_rcom, new_theta, new_phi = uniform_shell_sampler(COMDIST_MAX, AGN_COMDIST_MAX, n2complete)

    else:
        rcom_integrate_ax = np.linspace(COMDIST_MIN, AGN_COMDIST_MAX, 1024*4+1)
        total = romb(comdist_pdf_given_redshift_pdf(rcom_integrate_ax, AGN_ZPRIOR_FUNCTION), dx=np.diff(rcom_integrate_ax)[0])
        rcom_integrate_ax = np.linspace(COMDIST_MIN, COMDIST_MAX, 1024*4+1)
        current = romb(comdist_pdf_given_redshift_pdf(rcom_integrate_ax, AGN_ZPRIOR_FUNCTION), dx=np.diff(rcom_integrate_ax)[0])

        n2complete = int(round(len(agn_ra) * ( total / current - 1)))
        if n2complete != 0:

            new_theta, new_phi = sample_spherical_angles(n2complete)

            norm = romb(AGN_ZPRIOR_FUNCTION(AGN_ZPRIOR_NORM_AX) * (1 - z_cut(AGN_ZPRIOR_NORM_AX, zcut=ZMAX)), dx=np.diff(AGN_ZPRIOR_NORM_AX)[0])
            target_population = lambda z: AGN_ZPRIOR_FUNCTION(z) * (1 - z_cut(z, zcut=ZMAX)) / norm
            cdf = np.cumsum(target_population(AGN_ZPRIOR_NORM_AX))
            cdf /= cdf[-1]
            unif = np.random.rand(n2complete)
            new_z = np.interp(unif, cdf, AGN_ZPRIOR_NORM_AX)
            new_rcom = COSMO.comoving_distance(new_z).value
        else:
            new_phi, new_theta, new_rcom = np.empty(1), np.empty(1), np.empty(1)

    if VERBOSE:
        print(f'Adding {n2complete} AGN above GW zmax ({ZMAX}) to get a catalog with distribution: {AGN_ZPRIOR}.')

    agn_ra = np.append(agn_ra, new_phi)
    agn_dec = np.append(agn_dec, np.pi * 0.5 - new_theta)
    agn_rcom = np.append(agn_rcom, new_rcom)

    return agn_ra, agn_dec, agn_rcom, n2complete


def add_agn_to_catalog(agn_ra, agn_dec, agn_rcom, nsamps):
    '''As background noise'''

    if AGN_ZPRIOR == 'uniform_comoving_volume':
        new_rcom, new_theta, new_phi = uniform_shell_sampler(COMDIST_MIN, AGN_COMDIST_MAX, nsamps)
    
    else:
        new_theta, new_phi = sample_spherical_angles(nsamps)

        norm = romb(AGN_ZPRIOR_FUNCTION(AGN_ZPRIOR_NORM_AX), dx=np.diff(AGN_ZPRIOR_NORM_AX)[0])
        target_population = lambda z: AGN_ZPRIOR_FUNCTION(z) / norm
        cdf = np.cumsum(target_population(AGN_ZPRIOR_NORM_AX))
        cdf /= cdf[-1]
        unif = np.random.rand(nsamps)
        new_z = np.interp(unif, cdf, AGN_ZPRIOR_NORM_AX)
        new_rcom = COSMO.comoving_distance(new_z).value

    agn_ra = np.append(agn_ra, new_phi)
    agn_dec = np.append(agn_dec, np.pi * 0.5 - new_theta)
    agn_rcom = np.append(agn_rcom, new_rcom)

    return agn_ra, agn_dec, agn_rcom


def process_one_fagn(fagn_idx, fagn_realized):
    print(f'\nRealization {fagn_idx + 1}/{N_TRUE_FAGNS}: fagn = {fagn_realized}')
    # Give every process a unique seed -- TODO: save the seeds somewhere
    seed = np.random.SeedSequence().generate_state(1)[0]
    np.random.seed(seed)

    ### Get true source coordinates for GWs from AGN to put in the AGN catalog ###
    gw_fnames, gw_fnames_from_agn = get_gw_fnames_resampled(fagn_realized)

    gw_identifiers = sorted(np.array([get_id_from_fname(f) for f in gw_fnames_from_agn]).astype(int))
    true_sources = ALL_TRUE_SOURCES[np.searchsorted(TRUE_SOURCE_IDENTIFIERS, gw_identifiers)]
    agn_ra, agn_dec, agn_rcom = true_sources[:,3], 0.5 * np.pi - true_sources[:,2], true_sources[:,1]
    
    ### Complete catalog to preserve uniform in comoving volume distribution ###
    agn_ra_complete, agn_dec_complete, agn_rcom_complete, n2complete = fill_catalog_to_complete(agn_ra, agn_dec, agn_rcom)
    ############################################################################

    # plt.figure()
    # plt.hist(fast_z_at_value(COSMO.comoving_distance, agn_rcom_complete * u.Mpc), density=True, bins=100)
    # plt.plot(AGN_ZPRIOR_NORM_AX, AGN_ZPRIOR_FUNCTION(AGN_ZPRIOR_NORM_AX))
    # plt.show()
    # sys.exit(1)
    
    if ADD_NAGN_TO_CAT > n2complete:  # Add uncorrelated AGN as background
        if VERBOSE:
            print(f'Adding {ADD_NAGN_TO_CAT - n2complete} more AGN to get to the requested number of AGN.')

        agn_ra_complete, agn_dec_complete, agn_rcom_complete = add_agn_to_catalog(agn_ra_complete, agn_dec_complete, agn_rcom_complete, ADD_NAGN_TO_CAT - n2complete)

    if len(agn_rcom_complete) == 0:
        obs_agn_redshift_complete, agn_redshift_err_complete = np.empty_like(agn_rcom_complete), np.empty_like(agn_rcom_complete)
        obs_agn_rlum_complete = np.empty_like(agn_rcom_complete)
    else:
        obs_agn_redshift_complete, agn_redshift_err_complete = get_observed_redshift_from_rcom(agn_rcom_complete)
        obs_agn_rlum_complete = COSMO.luminosity_distance(obs_agn_redshift_complete).value

    ### Make an incomplete AGN catalog from these coordinates ###
    incomplete_catalog_mask, z_selection_function, completeness_map = make_incomplete_catalog(agn_ra_complete, agn_dec_complete, obs_agn_rlum_complete, obs_agn_redshift_complete)
    agn_ra = agn_ra_complete[incomplete_catalog_mask]
    agn_dec = agn_dec_complete[incomplete_catalog_mask]
    obs_agn_redshift = obs_agn_redshift_complete[incomplete_catalog_mask]
    agn_redshift_err = agn_redshift_err_complete[incomplete_catalog_mask]
    obs_agn_rlum = obs_agn_rlum_complete[incomplete_catalog_mask]

    agn_posterior_dset, sum_of_posteriors_incomplete = get_agn_posteriors(fagn_idx, obs_agn_redshift, agn_redshift_err, label='INCOMPLETE')
    
    ### Characterize the redshift-completeness ###
    if ASSUME_PERFECT_REDSHIFT or LUM_THRESH == 'inf':
        redshift_completeness = z_selection_function

    else:  # Measure the selection function from the data realization
        latitude_mask, _ = make_latitude_selection(agn_ra_complete, agn_dec_complete, obs_agn_rlum_complete)  # Measure completeness in the surveyed sky area
        expected_distribution = np.sum(latitude_mask) * AGN_ZPRIOR_FUNCTION(Z_INTEGRAL_AX) / romb(AGN_ZPRIOR_FUNCTION(AGN_ZPRIOR_NORM_AX), dx=np.diff(AGN_ZPRIOR_NORM_AX)[0])
        no_zero = (expected_distribution != 0)

        redshift_agn_selection_function = np.zeros_like(expected_distribution)
        redshift_agn_selection_function[no_zero] = sum_of_posteriors_incomplete[no_zero] / expected_distribution[no_zero]
        redshift_agn_selection_function[redshift_agn_selection_function > 1] = 1
        redshift_completeness = interp1d(Z_INTEGRAL_AX, redshift_agn_selection_function, bounds_error=False, fill_value=0)

    # print(np.sum(fast_z_at_value(COSMO.comoving_distance, agn_rcom_complete * u.Mpc) < 0.4))
    # print(np.sum(fast_z_at_value(COSMO.comoving_distance, agn_rcom_complete * u.Mpc) > 0.4))
    # zzz = fast_z_at_value(COSMO.comoving_distance, agn_rcom_complete * u.Mpc)
    # print(np.min(agn_rcom_complete), np.max(agn_rcom_complete))
    # print(np.min(zzz), np.max(zzz))

    # print(np.sum(obs_agn_redshift < 0.4))
    # print(np.sum(obs_agn_redshift > 0.4))
    # plt.figure()
    # plt.plot(Z_INTEGRAL_AX, redshift_completeness(Z_INTEGRAL_AX), label='Estimated')
    # plt.plot(Z_INTEGRAL_AX, z_selection_function(Z_INTEGRAL_AX), label='Input')
    # plt.xlabel('Redshift')
    # plt.ylabel('Completeness')
    # plt.legend()
    # plt.show()
    # sys.exit(1)

    ### Calculate the integrals in the likelihood ###
    S_agn_incat = np.zeros(BATCH)
    S_agn_outofcat = np.zeros(BATCH)
    S_alt = np.zeros(BATCH)

    S_agn_incat_dict = {}
    S_agn_outofcat_dict = {}
    S_alt_dict = {}
    from_agn_dict = {}
    # for gw_idx, filename in tqdm(enumerate(gw_fnames)):
    for gw_idx, filename in enumerate(gw_fnames):
        if VERBOSE:
                print(f'({gw_idx+1}/{len(gw_fnames)})')

        if USE_SKYMAPS:
            skymap = read_sky_map(filename, moc=True)
            sagn_incat, sagn_outofcat, salt = crossmatch(
                                                        agn_posterior_dset=agn_posterior_dset,              # AGN data (needed when using AGN z-errors)
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
                                                        correct_time_dilation=CORRECT_TIME_DILATION,
                                                        **MERGER_RATE_KWARGS)                               # kwargs for  merger rate function
        
        else:
            if VERBOSE:
                print(f'Using GW posterior samples!')
            with h5py.File(filename, 'r') as posterior_samples:
                sagn_incat, sagn_outofcat, salt = crossmatch_from_samples_p26(posterior_samples=posterior_samples, 
                                                                              z_integral_ax=Z_INTEGRAL_AX,
                                                                              agn_posterior_dset=agn_posterior_dset,
                                                                              agn_ra=agn_ra,
                                                                              agn_dec=agn_dec,
                                                                              completeness_map=completeness_map,
                                                                              redshift_completeness=redshift_completeness,
                                                                              gw_zcut=ZMAX,
                                                                              merger_rate_func=MERGER_RATE_EVOLUTION,
                                                                              correct_time_dilation=CORRECT_TIME_DILATION,
                                                                              background_agn_distribution=AGN_ZPRIOR_FUNCTION,
                                                                              linax=LINAX,
                                                                              minpix=30,
                                                                              skymap_cl=SKYMAP_CL,
                                                                              minsamps=100,
                                                                              **MERGER_RATE_KWARGS)
 
        S_agn_incat[gw_idx] = sagn_incat
        S_agn_outofcat[gw_idx] = sagn_outofcat
        S_alt[gw_idx] = salt

        key = get_id_from_fname(filename)
        S_agn_incat_dict[key] = sagn_incat
        S_agn_outofcat_dict[key] = sagn_outofcat
        S_alt_dict[key] = salt

        if int(key) in gw_identifiers:
            from_agn_dict[key] = True
        else:
            from_agn_dict[key] = False

        if VERBOSE:
            print(f'S_alt: {salt}, S_incat: {sagn_incat}, S_outcat: {sagn_outofcat}, S_agn: {sagn_incat + sagn_outofcat}, negative values: {np.sum((LOG_LLH_X_AX * (sagn_incat + sagn_outofcat - salt) + salt) < 0)}\n')
    
    ### Evaluate the likelihood ###
    S_agn_incat = S_agn_incat[~np.isnan(S_agn_incat)]
    S_agn_outofcat = S_agn_outofcat[~np.isnan(S_agn_outofcat)]
    S_alt = S_alt[~np.isnan(S_alt)]

    loglike = np.log(SKYMAP_CL * LOG_LLH_X_AX[None,:] * (S_agn_incat[:,None] + S_agn_outofcat[:,None] - S_alt[:,None]) + S_alt[:,None])

    nans = np.isnan(loglike)
    if np.sum(nans) != 0:
        print('Got NaNs:')
        arr = np.ones_like(LOG_LLH_X_AX)
        print((arr[None,:] * S_agn_incat[:,None])[nans])
        print((arr[None,:] * S_agn_outofcat[:,None])[nans])
        print((arr[None,:] * S_alt[:,None])[nans])
    
    # np.save(f'{POST_DIR}/s_agn_incat_dict_{fagn_idx}_{FAGN_POSTERIOR_FNAME}.npy', S_agn_incat_dict)
    # np.save(f'{POST_DIR}/s_agn_outofcat_dict_{fagn_idx}_{FAGN_POSTERIOR_FNAME}.npy', S_agn_outofcat_dict)
    # np.save(f'{POST_DIR}/s_alt_dict_{fagn_idx}_{FAGN_POSTERIOR_FNAME}.npy', S_alt_dict)
    # np.save(f'{POST_DIR}/from_agn_dict_{fagn_idx}_{FAGN_POSTERIOR_FNAME}.npy', from_agn_dict)
    return fagn_idx, np.sum(loglike, axis=0)  # Sum over all GWs


if __name__ == '__main__':

    log_llh = np.zeros((len(LOG_LLH_X_AX), N_TRUE_FAGNS))
    if THREADING:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:  # 1 proc, 11.5s/it - 8 proc, 14.5s/it - 16 proc, 25s/it - 32 proc, 58s/it
            futures = [executor.submit(process_one_fagn, fagn_idx, fagn_true) for fagn_idx, fagn_true in enumerate(REALIZED_FAGNS)]
            for future in tqdm(as_completed(futures)):
                fagn_idx, llh = future.result()
                log_llh[:,fagn_idx] = llh
    else:
        for fagn_idx, fagn_realized in enumerate(REALIZED_FAGNS):
            fagn_idx, llh = process_one_fagn(fagn_idx, fagn_realized)
            log_llh[:,fagn_idx] = llh

            if VERBOSE:
                print('Done.')
        
            plt.figure()
            posterior = log_llh[:,fagn_idx]
            posterior -= np.max(posterior)
            pdf = np.exp(posterior)
            norm = simpson(y=pdf, x=LOG_LLH_X_AX, axis=0)
            pdf = pdf / norm
            plt.plot(LOG_LLH_X_AX, pdf)
            plt.vlines(TRUE_FAGNS[0], 0, np.max(pdf))
            plt.show()
            sys.exit('Exiting...')

    fname = f'{POST_DIR}/{FAGN_POSTERIOR_FNAME}'
    np.save(fname, log_llh)
    print(f'Posterior is located at: {fname}.npy')
