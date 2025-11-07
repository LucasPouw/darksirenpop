from p26_control_room import *
import glob
from ligo.skymap.io.fits import read_sky_map
from p26_crossmatch import crossmatch_p26 as crossmatch
import sys
import json
from scipy.integrate import simpson


CATALOG_PATH = "/home/lucas/Documents/PhD/Quaia_z15.csv"
SKYMAP_JSON_PATH = '/home/lucas/Documents/PhD/gw_data/real_skymaps.json'

fagn_idx = 0
with open(SKYMAP_JSON_PATH, "r") as f:
    gw_path_dict = json.load(f)
gw_keys = list(gw_path_dict.keys())

### Load Quaia ###

df = pd.read_csv(CATALOG_PATH)
cols = ["redshift_quaia", "redshift_quaia_err", "ra", "dec", "b", "loglbol_corr"]
data = df[cols]
b              = data["b"].to_numpy()
loglbol_corr   = data["loglbol_corr"].to_numpy()
outside_galactic_plane = np.logical_or((b > 10), (b < -10))
above_lbol_thresh = loglbol_corr >= float(LUM_THRESH)

b                  = b[outside_galactic_plane & above_lbol_thresh]
loglbol_corr       = loglbol_corr[outside_galactic_plane & above_lbol_thresh]
agn_redshift       = data["redshift_quaia"].to_numpy()[outside_galactic_plane & above_lbol_thresh]
agn_redshift_err   = data["redshift_quaia_err"].to_numpy()[outside_galactic_plane & above_lbol_thresh]
agn_ra             = np.deg2rad( data["ra"].to_numpy()[outside_galactic_plane & above_lbol_thresh] )
agn_dec            = np.deg2rad( data["dec"].to_numpy()[outside_galactic_plane & above_lbol_thresh] )
agn_rlum           = COSMO.luminosity_distance(agn_redshift).value

agn_posterior_dset, redshift_population_prior_normalization, sum_of_posteriors_incomplete = get_agn_posteriors_and_zprior_normalization(fagn_idx, agn_redshift, agn_redshift_err, label=LUM_THRESH, replace_old_file=False)
_, c_per_zbin, completeness_map = make_incomplete_catalog(agn_ra, agn_dec, agn_rlum, agn_redshift)  # Quaia is already redshift incomplete, but convenient to get completeness maps this way

### Calculate the integrals in the likelihood ###

S_agn_cw_dict = {}
S_alt_cw_dict = {}
S_alt_dict = {}
log_llh = np.zeros((len(LOG_LLH_X_AX), 1))
for gw_idx, key in enumerate(gw_keys):
    filename = gw_path_dict[key]
    skymap = read_sky_map(filename, moc=True)
    
    sagn_cw, salt_cw, salt, _, _ = crossmatch(agn_posterior_dset=agn_posterior_dset,              # AGN data (needed when using AGN z-errors)
                                            sky_map=skymap,                                     # GW data
                                            completeness_map=completeness_map,                  # For getting the surveyed sky-area
                                            completeness_zedges=Z_EDGES,                        # Redshift selection function
                                            completeness_zvals=c_per_zbin,                      # Redshift selection function
                                            agn_ra=agn_ra,                                      # AGN data (needed when neglecting AGN z-errors)
                                            agn_dec=agn_dec,                                    # AGN data (needed when neglecting AGN z-errors)
                                            agn_lumdist=agn_rlum,                               # AGN data (needed when neglecting AGN z-errors)
                                            agn_redshift=agn_redshift,                          # AGN data (needed when neglecting AGN z-errors)
                                            agn_redshift_err=agn_redshift_err,                  # AGN data (needed when neglecting AGN z-errors)
                                            skymap_cl=SKYMAP_CL,                                # Only analyze AGN within this CL, only for code speed-up
                                            gw_zcut=ZMAX,                                       # GWs are not generated above ZMAX
                                            s_agn_z_integral_ax=S_AGN_Z_INTEGRAL_AX,            # Integrating the likelihood in redshift space    
                                            s_alt_z_integral_ax=S_ALT_Z_INTEGRAL_AX,            # Integrating the likelihood in redshift space    
                                            assume_perfect_redshift=ASSUME_PERFECT_REDSHIFT,    # Integrating delta functions is handled differently
                                            merger_rate_func=MERGER_RATE_EVOLUTION,             # Merger rate can evolve
                                            linax=LINAX,                                        # Integration can be done in linspace or in geomspace
                                            realdata=True,
                                            **MERGER_RATE_KWARGS)                               # kwargs for  merger rate function
    
    if len(agn_ra) != 0:
        sagn_cw *= (4 * np.pi / redshift_population_prior_normalization)  # 4pi comes from uniform-on-sky parameter estimation prior and divide by the normalization of the redshift population prior: int dz Sum(p_red(z|z_obs)) p_rate(z) p_cut(z)

    S_agn_cw_dict[key] = sagn_cw
    S_alt_cw_dict[key] = salt_cw
    S_alt_dict[key] = salt

    if VERBOSE:
        print(f"\n({gw_idx+1}/{len(gw_keys)}) {key}: CW S_agn={sagn_cw}, CW S_alt={salt_cw}\n")
        if sagn_cw > salt_cw:
            print('!!! HIGHER AGN PROB !!!')

np.save(f's_agn_cw_dict_{FAGN_POSTERIOR_FNAME}.npy', S_agn_cw_dict)
np.save(f's_alt_cw_dict_{FAGN_POSTERIOR_FNAME}.npy', S_alt_cw_dict)
np.save(f's_alt_dict_{FAGN_POSTERIOR_FNAME}.npy', S_alt_dict)

S_agn_cw = np.array([S_agn_cw_dict[key] for key in gw_keys])
S_alt_cw = np.array([S_alt_cw_dict[key] for key in gw_keys])
S_alt = np.array([S_alt_dict[key] for key in gw_keys])

### Evaluate the likelihood ###

loglike = np.log(SKYMAP_CL * LOG_LLH_X_AX[None,:] * (S_agn_cw[:,None] - S_alt_cw[:,None]) + S_alt[:,None])
log_llh[:,fagn_idx] = np.sum(loglike, axis=0)  # Sum over all GWs
if VERBOSE:
    print('Done.')

np.save(os.path.join(sys.path[0], FAGN_POSTERIOR_FNAME), log_llh)

### Plot posterior ###

posterior = log_llh
posterior -= np.max(posterior)
pdf = np.exp(posterior)
norm = simpson(y=pdf, x=LOG_LLH_X_AX, axis=0)  # Simpson should be fine...
pdf = pdf / norm
plt.figure()
plt.plot(LOG_LLH_X_AX, pdf)
plt.savefig(f'real_posterior_lumthresh_{LUM_THRESH}_perfectz_{ASSUME_PERFECT_REDSHIFT}.pdf', bbox_inches='tight')
plt.show()
