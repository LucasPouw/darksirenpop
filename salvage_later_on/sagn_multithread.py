
            sagn_start = time.time()
            def process_one(args):
                '''
                AGN posterior loading is the most expensive operation right now, dominating S_agn computation time
                '''
                mp_start = time.time()

                gw_idx, _ = args
                current_pix = (unique_gw_idx == gw_idx)
                
                gw_redshift_posterior_in_pix = gw_redshift_posterior_in_allpix[:, current_pix].flatten()

                # gwmu, gwsig = distmu_allpix[current_pix], distsigma_allpix[current_pix]
                # agn_indices = np.array(index_map[gw_idx])  # Vectorize this function over all AGN with the same GW sky probdens
                # ### But first check the distance between the AGN posterior and the GW posterior, skip unnecessary contributions - NOTE: When pixelating the sky, grouping AGN together, this procedure cannot be used anymore. ###
                # speedup = time.time()
                # if gwmu < 1e-4:
                #     gwmu_redshiftspace = 1e-4  # Just put it to a valid value, it is only for some speed up anyway
                # else:
                #     gwmu_redshiftspace = fast_z_at_value(COSMO.luminosity_distance, gwmu * u.Mpc)
                # gwsig_redshiftspace = gwsig / (COSMO.comoving_distance(gwmu_redshiftspace).value + SPEED_OF_LIGHT_KMS * (1 + gwmu_redshiftspace) / COSMO.H(gwmu_redshiftspace).value)
                # distance_between_posteriors = abs(gwmu_redshiftspace - agn_redshift[agn_indices])
                # maxdist = (5 * agn_redshift_err[agn_indices] + 5 * gwsig_redshiftspace)
                # agn_close_to_gw_mask = distance_between_posteriors < maxdist
                # agn_indices = agn_indices[agn_close_to_gw_mask]
                # if len(agn_indices) == 0:
                #     return 0., 0., 0.
                # speeduptotal = time.time() - speedup
                ##########################################
                
                agn_redshift_posterior = agn_redshift_posteriors[gw_to_rows[gw_idx], :]

                # The population prior consists of AGN posteriors, modulated by redshift evolving merger rates, normalization is done outside this function (Namely, 06-10-2025: in p26_likelihood.py)
                agn_population_prior_unnorm = np.sum(agn_redshift_posterior * zcut * zrate, axis=0)

                # TODO: the completeness could differ for AGN with the same gw_idx + sky completeness and redshift completeness should be combined in a 3D map -> def redshift_completeness(z, pix)
                # TODO: Sum all integrands and do a single integral at the end
                integral = time.time()
                if linax:
                    contribution = dP_dA[gw_idx] * romb(gw_redshift_posterior_in_pix * agn_population_prior_unnorm * fc_of_z / PEprior, dx=dz_Sagn)
                else:
                    contribution = dP_dA[gw_idx] * romb(gw_redshift_posterior_in_pix * agn_population_prior_unnorm * fc_of_z / PEprior * s_agn_z_integral_ax * np.log(10), dx=dz_Sagn)
                inttimer = time.time() - integral

                ###### DIAGNOSTIC PLOTTING ######
                # if len(agn_indices) > 1:
                #     plt.figure()
                #     plt.plot(s_agn_z_integral_ax, 1e10*gw_redshift_posterior_in_pix * agn_population_prior_unnorm * redshift_completeness(s_agn_z_integral_ax) / uniform_comoving_prior(s_agn_z_integral_ax), label='To int')
                #     plt.plot(s_agn_z_integral_ax, gw_redshift_posterior_in_pix, label=r'$p_{\rm GW}$')
                #     plt.plot(s_agn_z_integral_ax, agn_population_prior_unnorm, label=r'$p_{\rm pop}$')
                #     plt.plot(s_agn_z_integral_ax, redshift_completeness(s_agn_z_integral_ax), label=r'$f_{\rm c}(z)$')
                    
                #     plt.hlines(np.tile(np.max(gw_redshift_posterior_in_pix), len(agn_indices)), gwmu_redshiftspace + 5 * gwsig_redshiftspace, gwmu_redshiftspace - 5 * gwsig_redshiftspace, linestyle='dotted', color='black',zorder=10)
                #     plt.hlines(np.tile(np.max(gw_redshift_posterior_in_pix), len(agn_indices)), agn_redshift[agn_indices] + 5 * agn_redshift_err[agn_indices], agn_redshift[agn_indices] - 5 * agn_redshift_err[agn_indices], color='red')

                #     plt.xlabel(r'$z$')
                #     # plt.semilogy()
                #     plt.legend()

                #     if np.sum(agn_close_to_gw_mask) == 0:
                #         plt.title(f'KEEPING NONE SINCE {maxdist} < {distance_between_posteriors}')
                #     elif np.sum(agn_close_to_gw_mask) != len(agn_indices):
                #         plt.title(f'KEEPING SOME SINCE {maxdist} ~ {distance_between_posteriors}')
                #     else:
                #         plt.title(f'KEEPING ALL SINCE {maxdist} > {distance_between_posteriors}')
                #     plt.show()
                ####################################

                ### WHEN USING ONLY SKYLOC ###
                # phi = agn_phi[agn_indices]
                # theta = agn_theta[agn_indices]
                # pix_idx = hp.ang2pix(cmap_nside, theta, phi, nest=True)
                # completeness_at_agn_skypos = completeness_map[pix_idx]
                # contribution = np.sum(dP_dA[gw_idx] * completeness_at_agn_skypos)  
                ##############################
                total = time.time() - mp_start
                return contribution, total, inttimer
        
            tasks = list(zip(unique_gw_idx, counts))
            with ThreadPoolExecutor() as executor:
                result = np.array(list(executor.map(process_one, tasks)))
                contributions, total, ints = result[:,0], result[:,1], result[:,2]

            # contributions = []
            # for task in tasks:
            #     contributions.append(process_one(task))

            S_agn_cw = np.sum(contributions)
            totaltime = time.time() - sagn_start
            print('MULTICORE Total Sagn time:', totaltime + dicts_time)
            print(S_agn_cw_singlecore, S_agn_cw, S_agn_cw / S_agn_cw_singlecore, 'ARE THESE THE SAME?????')
