from p26_control_room import *
import glob
from utils import uniform_shell_sampler
from ligo.skymap.io.fits import read_sky_map
from p26_crossmatch import crossmatch_p26 as crossmatch
import sys


all_gw_fnames = np.array(glob.glob(SKYMAP_DIR + 'skymap_*'))
# gw_fnames_per_realization = np.random.choice(all_gw_fnames, size=(BATCH, N_TRUE_FAGNS), replace=False)  # Only unique GWs for a single data set



gw_fnames_per_realization = []
keys = []
for i, file in enumerate(glob.glob('/home/lucas/Documents/PhD/mockstats/' + 'gw*.dat')):
    keys.append(file[-9:-4])
for gw_fnames in all_gw_fnames:
    key = gw_fnames[-13:-8]
    if key in keys:
        gw_fnames_per_realization.append(gw_fnames)
gw_fnames_per_realization = np.atleast_2d(gw_fnames_per_realization).T
print(gw_fnames_per_realization.shape)    



all_true_sources = np.genfromtxt('/home/lucas/Documents/PhD/true_r_theta_phi_all.txt', delimiter=',')
all_true_sources = all_true_sources[all_true_sources[:, 0].argsort()]
true_source_identifiers = all_true_sources[:,0]

log_llh = np.zeros((len(LOG_LLH_X_AX), N_TRUE_FAGNS))
# log_llh_bin = np.zeros((len(LOG_LLH_X_AX), N_TRUE_FAGNS))
for fagn_idx, fagn_true in enumerate(REALIZED_FAGNS):
    print(f'\nRealization {fagn_idx + 1}/{N_TRUE_FAGNS}: fagn = {fagn_true}')

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
        c_per_zbin = redshift_agn_selection_function
    
    # plt.figure()
    # plt.plot(S_AGN_Z_INTEGRAL_AX, c_per_zbin)
    # plt.show()

    # from utils import make_nice_plots
    # make_nice_plots()
    # plt.figure(figsize=(8,6))
    # # h, _, _ = plt.hist(obs_agn_redshift, bins=Z_EDGES, density=True, histtype='step', linewidth=5, label=r'$\langle z_{\rm obs} \rangle$')
    # # plt.hist(obs_agn_redshift, bins=10, histtype='step', linewidth=3)
    # # plt.hist(obs_agn_redshift[mask], bins=Z_EDGES, density=True, histtype='step', linewidth=3)
    # plt.plot(S_AGN_Z_INTEGRAL_AX, sum_of_posteriors_complete / romb(sum_of_posteriors_complete, dx=np.diff(S_AGN_Z_INTEGRAL_AX)[0]), linewidth=3, color='red', label=r'$\sum p(z|z_{\rm complete})$')
    # # plt.plot(S_AGN_Z_INTEGRAL_AX, sum_of_posteriors_incomplete, linewidth=3, color='indigo', label=r'$\sum p(z|z_{\rm observed})$')
    # plt.plot(S_AGN_Z_INTEGRAL_AX, uniform_comoving_prior(S_AGN_Z_INTEGRAL_AX) / romb(uniform_comoving_prior(S_AGN_Z_INTEGRAL_AX), dx=np.diff(S_AGN_Z_INTEGRAL_AX)[0]))
    # plt.plot(S_AGN_Z_INTEGRAL_AX, redshift_agn_selection_function, linewidth=3, color='cyan', label=r'$f_{\rm c}(z)$')
    # plt.vlines([ZMAX, AGN_ZCUT, AGN_ZMAX], ymin=0, ymax=1, color='black', linestyle='dashed', label='Edges')
    # plt.legend()
    # plt.xlabel('Redshift')
    # plt.ylabel('Completeness')
    # plt.grid()
    # plt.show()
    # sys.exit(1)

    ### Calculate the integrals in the likelihood ###
    
    S_agn_cw = np.zeros(BATCH)
    S_alt_cw = np.zeros(BATCH)
    S_alt = np.zeros(BATCH)



    S_agn_cw_dict = {}
    S_alt_cw_dict = {}
    S_alt_dict = {}
    from_agn_dict = {}


    # S_agn_cw_bin = np.zeros(BATCH)
    # S_alt_cw_bin = np.zeros(BATCH)
    # for gw_idx in range(BATCH):
    for gw_idx, filename in tqdm(enumerate(gw_fnames)):
        skymap = read_sky_map(filename, moc=True)
        # print(f'\nLoaded file {gw_idx+1}/{len(gw_fnames)}: {filename}')
        
        sagn_cw, salt_cw, salt, sagn_bin, salt_bin = crossmatch(agn_posterior_dset=agn_posterior_dset,              # AGN data (needed when using AGN z-errors)
                                                                sky_map=skymap,                                     # GW data
                                                                completeness_map=completeness_map,                  # For getting the surveyed sky-area
                                                                completeness_zedges=Z_EDGES,                        # Redshift selection function
                                                                completeness_zvals=c_per_zbin,                      # Redshift selection function
                                                                agn_ra=agn_ra,                                      # AGN data (needed when neglecting AGN z-errors)
                                                                agn_dec=agn_dec,                                    # AGN data (needed when neglecting AGN z-errors)
                                                                agn_lumdist=obs_agn_rlum,                           # AGN data (needed when neglecting AGN z-errors)
                                                                agn_redshift=obs_agn_redshift,                      # AGN data (needed when neglecting AGN z-errors)
                                                                agn_redshift_err=agn_redshift_err,                  # AGN data (needed when neglecting AGN z-errors)
                                                                skymap_cl=SKYMAP_CL,                                # Only analyze AGN within this CL, only for code speed-up
                                                                gw_zcut=ZMAX,                                       # GWs are not generated above ZMAX
                                                                s_agn_z_integral_ax=S_AGN_Z_INTEGRAL_AX,            # Integrating the likelihood in redshift space    
                                                                s_alt_z_integral_ax=S_ALT_Z_INTEGRAL_AX,            # Integrating the likelihood in redshift space    
                                                                assume_perfect_redshift=ASSUME_PERFECT_REDSHIFT,    # Integrating delta functions is handled differently
                                                                merger_rate_func=MERGER_RATE_EVOLUTION,             # Merger rate can evolve
                                                                linax=LINAX,                                        # Integration can be done in linspace or in geomspace
                                                                realdata=False,
                                                                **MERGER_RATE_KWARGS)                               # kwargs for  merger rate function
        
        if len(agn_ra) != 0:
            sagn_cw *= (4 * np.pi / redshift_population_prior_normalization)  # 4pi comes from uniform-on-sky parameter estimation prior and divide by the normalization of the redshift population prior: int dz Sum(p_red(z|z_obs)) p_rate(z) p_cut(z)
        # sagn_bin *= (4 * np.pi / redshift_population_prior_normalization)

        S_agn_cw[gw_idx] = sagn_cw
        S_alt_cw[gw_idx] = salt_cw
        S_alt[gw_idx] = salt


        key = filename[-13:-8]
        S_agn_cw_dict[key] = sagn_cw
        S_alt_cw_dict[key] = salt_cw
        S_alt_dict[key] = salt

        if int(key) in gw_identifiers:
            from_agn_dict[key] = True
        else:
            from_agn_dict[key] = False


        # S_agn_cw_bin[gw_idx] = sagn_bin
        # S_alt_cw_bin[gw_idx] = salt_bin
        # print(sagn_cw, salt_cw, salt)
    
    ### Evaluate the likelihood ###

    S_agn_cw = S_agn_cw[~np.isnan(S_agn_cw)]
    S_alt_cw = S_alt_cw[~np.isnan(S_alt_cw)]
    S_alt = S_alt[~np.isnan(S_alt)]

    loglike = np.log(SKYMAP_CL * LOG_LLH_X_AX[None,:] * (S_agn_cw[:,None] - S_alt_cw[:,None]) + S_alt[:,None])
    
    nans = np.isnan(loglike)
    if np.sum(nans) != 0:
        print('Got NaNs:')
        print((LOG_LLH_X_AX[None,:] * S_agn_cw[:,None])[nans])
        print((LOG_LLH_X_AX[None,:] * S_alt_cw[:,None])[nans])
        print((LOG_LLH_X_AX[None,:] * S_alt[:,None])[nans])

    # S_agn_cw_bin = S_agn_cw_bin[~np.isnan(S_agn_cw_bin)]
    # S_alt_cw_bin = S_alt_cw_bin[~np.isnan(S_alt_cw_bin)]
    # loglike_bin = np.log(SKYMAP_CL * LOG_LLH_X_AX[None,:] * (S_agn_cw_bin[:,None] - S_alt_cw_bin[:,None]) + S_alt[:,None])
    # nans = np.isnan(loglike_bin)
    # if np.sum(nans) != 0:
    #     print('Got NaNs:')
    #     print((LOG_LLH_X_AX[None,:] * S_agn_cw_bin[:,None])[nans])
    #     print((LOG_LLH_X_AX[None,:] * S_alt_cw_bin[:,None])[nans])
    #     sys.exit(1)
    # log_llh_bin[:,fagn_idx] = np.sum(loglike_bin, axis=0)  # sum over all GWs
    
    log_llh[:,fagn_idx] = np.sum(loglike, axis=0)  # Sum over all GWs
    if VERBOSE:
        print('Done.')

np.save(os.path.join(sys.path[0], FAGN_POSTERIOR_FNAME), log_llh)
# np.save(os.path.join(sys.path[0], FAGN_POSTERIOR_FNAME + '_binned.npy'), log_llh_bin)




np.save(f's_agn_cw_dict_{FAGN_POSTERIOR_FNAME}.npy', S_agn_cw_dict)
np.save(f's_alt_cw_dict_{FAGN_POSTERIOR_FNAME}.npy', S_alt_cw_dict)
np.save(f's_alt_dict_{FAGN_POSTERIOR_FNAME}.npy', S_alt_dict)
np.save(f'from_agn_dict_{FAGN_POSTERIOR_FNAME}.npy', from_agn_dict)