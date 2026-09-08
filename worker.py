import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from darksirenpop.likelihood import process_one_fagn


def run_worker(cfg):
    """
    Run the posterior computation over all data realizations.
    Supports multithreading.
    """
    
    log_llh = np.zeros((len(cfg.LOG_LLH_X_AX), cfg.N_REALIZATIONS))

    if cfg.THREADING:
        with ProcessPoolExecutor(max_workers=cfg.N_WORKERS) as executor:
            futures = [executor.submit(process_one_fagn, fagn_idx, cfg) for fagn_idx in range(cfg.N_REALIZATIONS)]
            for future in tqdm(as_completed(futures)):
                fagn_idx, llh = future.result()
                log_llh[:, fagn_idx] = llh
    else:
        for fagn_idx in range(cfg.N_REALIZATIONS):
            fagn_idx, llh = process_one_fagn(fagn_idx, cfg)
            log_llh[:, fagn_idx] = llh

            # if cfg.VERBOSE:
            #     print('Done.')

            #     plt.figure(figsize=(8,6))
            #     posterior = log_llh[:, fagn_idx]
            #     posterior -= np.max(posterior)
            #     pdf = np.exp(posterior)
            #     norm = simpson(y=pdf, x=cfg.LOG_LLH_X_AX, axis=0)
            #     pdf = pdf / norm
            #     plt.plot(cfg.LOG_LLH_X_AX, pdf)
            #     plt.vlines(cfg.TRUE_FAGNS[fagn_idx], 0, np.max(pdf), linestyle='dashed', color='black')
            #     plt.xlim(0, 1)
            #     plt.xlabel(r'$f_{\rm agn}$')
            #     plt.ylabel('Probability density')
            #     plt.show()

    return log_llh
