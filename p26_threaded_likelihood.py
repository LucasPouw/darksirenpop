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
    ############################################################################

    if ADD_NAGN_TO_CAT:  # Add uncorrelated AGN as background
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

    ### Calculate the integrals in the likelihood ###
    
    S_agn_cw = np.zeros(BATCH)
    S_alt_cw = np.zeros(BATCH)
    S_alt = np.zeros(BATCH)

    # S_agn_cw_bin = np.zeros(BATCH)
    # S_alt_cw_bin = np.zeros(BATCH)
    # for gw_idx in range(BATCH):
    for gw_idx, filename in enumerate(gw_fnames):
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
                                                                **MERGER_RATE_KWARGS)                               # kwargs for  merger rate function

        if len(agn_ra) != 0:
            sagn_cw *= (4 * np.pi / redshift_population_prior_normalization)  # 4pi comes from uniform-on-sky parameter estimation prior and divide by the normalization of the redshift population prior: int dz Sum(p_red(z|z_obs)) p_rate(z) p_cut(z)
        # sagn_bin *= (4 * np.pi / redshift_population_prior_normalization)

        S_agn_cw[gw_idx] = sagn_cw
        S_alt_cw[gw_idx] = salt_cw
        S_alt[gw_idx] = salt

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
    # print(f'\n{fagn_idx} Done in {time.time() - s}\n')

    return fagn_idx, np.sum(loglike, axis=0)


log_llh = np.zeros((len(LOG_LLH_X_AX), N_TRUE_FAGNS))
# log_llh_bin = np.zeros((len(LOG_LLH_X_AX), N_TRUE_FAGNS))
with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:  # 1 proc, 11.5s/it - 8 proc, 14.5s/it - 16 proc, 25s/it - 32 proc, 58s/it
    futures = [executor.submit(process_one_fagn, fagn_idx, fagn_true) for fagn_idx, fagn_true in enumerate(REALIZED_FAGNS)]
    for future in tqdm(as_completed(futures)):
        fagn_idx, llh = future.result()
        log_llh[:,fagn_idx] = llh

np.save(os.path.join(sys.path[0], FAGN_POSTERIOR_FNAME), log_llh)
# np.save(os.path.join(sys.path[0], FAGN_POSTERIOR_FNAME + '_binned.npy'), log_llh_bin)
