import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import sys
from process_mock import process_one_fagn


def run_worker(cfg):
    """
    Run the posterior computation over all realizations of f_agn.
    Supports multithreading. If verbose output and not threading, 
    a plot is made of a single f_agn posterior and the code exits.
    """
    
    log_llh = np.zeros((len(cfg.LOG_LLH_X_AX), cfg.N_TRUE_FAGNS))

    if cfg.THREADING:
        with ProcessPoolExecutor(max_workers=cfg.N_WORKERS) as executor:
            futures = [
                executor.submit(process_one_fagn, fagn_idx, fagn_realized, cfg)
                for fagn_idx, fagn_realized in enumerate(cfg.REALIZED_FAGNS)
            ]
            for future in tqdm(as_completed(futures)):
                fagn_idx, llh = future.result()
                log_llh[:, fagn_idx] = llh
    else:
        for fagn_idx, fagn_realized in enumerate(cfg.REALIZED_FAGNS):
            fagn_idx, llh = process_one_fagn(fagn_idx, fagn_realized, cfg)
            log_llh[:, fagn_idx] = llh

            if cfg.VERBOSE:
                print('Done.')

                plt.figure()
                posterior = log_llh[:, fagn_idx]
                posterior -= np.max(posterior)
                pdf = np.exp(posterior)
                norm = simpson(y=pdf, x=cfg.LOG_LLH_X_AX, axis=0)
                pdf = pdf / norm
                plt.plot(cfg.LOG_LLH_X_AX, pdf)
                plt.vlines(cfg.TRUE_FAGNS[0], 0, np.max(pdf))
                plt.show()
                sys.exit('Exiting...')

    return log_llh
