#!/usr/bin/env python3

"""
This script computes the line-of-sight redshift prior for a given galaxy catalogue.

Rachel Gray
Freija Beirnaert
Lucas Pouw
"""

import numpy as np
import healpy as hp
import h5py
from LOS_zprior import LineOfSightRedshiftPrior
import multiprocessing as mp
import threading
import sys, os
import logging
from pixelated_catalog import load_catalog_from_path
from darksirenpop.arguments import create_parser
from tqdm import tqdm
from darksirenpop.default_arguments import *
from LOS_zprior import get_norm_interp

handler_out = logging.StreamHandler(stream=sys.stdout)
handler_err = logging.StreamHandler(stream=sys.stderr)
handler_err.setLevel(logging.ERROR)
logging.basicConfig(handlers=[handler_out, handler_err], level = logging.INFO)
logger = logging.getLogger(__name__)

POISON_PILL = None


def _denom_offset_log(array, denom, offset):
    array /= denom
    array += offset
    return np.log(array)


def handle_results(h5file: h5py.File, queue: mp.Queue, npix=1, denom=1., offset=1.):
    """
    Writes back the results and accumulates them if necessary. If npix differs from 1, the combined_pixels will be evaluated.
    """
    logger.info("Writer thread started.")
    combined_pixels = 0

    while True:
        result = queue.get()
        if result == POISON_PILL:
            logger.info("Writer thread being stopped.")
            if npix != 1 and not isinstance(combined_pixels, int):
                combined_pixels = _denom_offset_log(combined_pixels, denom * npix, offset)
                logger.info("Writing back combined pixels.")
                h5file.create_dataset("combined_pixels", (len(combined_pixels),), dtype='f', data=combined_pixels)
            return

        (p_of_z, _, pixel_index) = result

        if npix != 1:
            combined_pixels += p_of_z
        p_of_z = _denom_offset_log(p_of_z, denom, offset)
        logger.info(f"pixel {pixel_index}: writing back results.")
        h5file.create_dataset(
            f"{pixel_index}", (len(p_of_z),), dtype='f', data=p_of_z)


def LOS_mp_thread(in_queue, out_queue, catalog, nside, galaxy_norm, zarray, zmax, min_gals_for_threshold, zmin, zdraw, sigma, cosmo):
    """
    Handles the multi process threading for multiple pixels.
    """
    while True:
        pixel_index = in_queue.get()
        if pixel_index is None:
            break
        try:
            LOS_zprior = LineOfSightRedshiftPrior(
                                                pixel_index=pixel_index, 
                                                galaxy_catalog=catalog,
                                                nside=nside,
                                                galaxy_norm=galaxy_norm,
                                                z_array=zarray,
                                                zmax=zmax,
                                                min_gals_for_threshold=min_gals_for_threshold,
                                                zdraw=zdraw,
                                                zmin=zmin,
                                                sigma=sigma,
                                                cosmo=cosmo
                                            )
            (p_of_z, z_array) = LOS_zprior.create_redshift_prior()
            res = (p_of_z, z_array, LOS_zprior.pixel_index)
            out_queue.put(res)
        except Exception:
            logger.exception(f"During calculation of pixel {pixel_index} the following error occurred:")


def main():
    parser = create_parser("--zmax",
                           "--zmin",
                           "--zdraw",
                           "--sigma",
                           "--nside", 
                           "--coarse_nside",
                           "--catalog_name",
                           "--catalog_path",
                           "--maps_path",
                           "--min_gals_for_threshold", 
                           "--pixel_index", 
                           "--num_threads", 
                           "--offset")
    opts = parser.parse_args()
    logger.info(opts)

    zmax = float(opts.zmax)
    zmin = float(opts.zmin)
    zdraw = float(opts.zdraw)
    nside = int(opts.nside)
    sigma = float(opts.sigma)

    # TODO: make this not hard-coded
    cosmo = DEFAULT_COSMOLOGY
    zarray = np.logspace(-10, np.log10(zdraw), 12000)
    # zarray = np.linspace(0, zmax, 1000)  

    #############################################################
    ########################## MAPS #############################
    #############################################################

    # coarse_nside = opts.coarse_nside  
    # if opts.pixel_index is None:
    #     coarse_pixel_index = None
    # else:
    #     coarse_ra, coarse_dec = ra_dec_from_ipix(nside, opts.pixel_index, True)
    #     coarse_pixel_index = ipix_from_ra_dec(coarse_nside, coarse_ra, coarse_dec, True)
   
    # if opts.maps_path == None:
    #     #maps_path = os.path.abspath(create_norm_map.__file__).strip(r'create_norm_map.py').strip(r'/')
    #     maps_path = os.path.abspath(create_norm_map.__file__).rstrip(r'/create_norm_map.py')
    # else:
    #     maps_path = (opts.maps_path).rstrip("/")

    # catalog = load_catalog_from_opts(opts)
    catalog = load_catalog_from_path(name=opts.catalog_name, catalog_path=opts.catalog_path)
    catalog.clean_cache(mtime=np.inf)  # Clean cache
    catalog.select_pixel(nside=nside, pixel_index=0)  # Put correct indexing file in cache, otherwise multiprocessing breaks and I'm not going to fix that -Lucas
    _ = get_norm_interp(zmin=zmin, zmax=zmax, sigma=sigma, npoints=10000, cosmo=cosmo, cachedir=None)  # Same thing, but this time it's my own fault -Lucas
    
    # # Try to use all sky map if it exists
    # norm_map_path = f"{maps_path}/norm_map_{opts.catalog}_nside{coarse_nside}_pixel_indexNone_zmax{str(zmax).replace('.', ',')}.fits"
    # if not os.path.exists(norm_map_path):
    #     # Try tu use map for coarse_pixel_index if it exists
    #     norm_map_path = f"{maps_path}/norm_map_{opts.catalog}_nside{coarse_nside}_pixel_index{coarse_pixel_index}_zmax{str(zmax).replace('.', ',')}.fits"
    #     if not os.path.exists(norm_map_path):
    #         # Create map for coarse_pixel_index, which can be None
    #         create_norm_map.create_norm_map(norm_map_path, catalog, coarse_nside, coarse_pixel_index, zmax)
    galaxy_norm = opts.maps_path
    logger.info(f"norm map path: {galaxy_norm}")

    ### ERRORS IF NSIDE < COURSE_NSIDE FROM PRECOMPUTED MAP ### TODO
    m = hp.read_map(galaxy_norm)
    assert np.sqrt(len(m) / 12) <= nside, "Coarse nside of norm map is larger than nside of high resolution map."

    #############################################################
    ##################### MAIN FUNCTIONS ########################
    #############################################################

    logger.info(f"npixels_tot =  {hp.nside2npix(nside)}")
    if opts.pixel_index is None:
        npix = hp.nside2npix(nside)
        pixel_indices = np.arange(0, npix)
    else:
        npix = 1
        pixel_indices = np.array([int(opts.pixel_index)])

    offset = opts.offset  # TODO: What is the offset for? -Lucas
    denom = 1.

    catname = opts.catalog_path.split('/')[-1][:-5]
    fname = f'/LOSzpriors/LOS_redshift_prior_{catname}_lenzarray_{len(zarray)}_zdraw_{zdraw}_nside_{nside}_pixel_index_{opts.pixel_index}.hdf5'
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
    output_path = os.path.join(script_dir, "../output")
    output_path = os.path.abspath(output_path)  # Normalize path to resolve '..'

    f1 = h5py.File(output_path + fname, "w" )
    f1.create_dataset("z_array", (len(zarray),), dtype='f', data=zarray)

    # Create queues for multiprocessing message passing
    mgr = mp.Manager()
    task_queue = mgr.Queue()
    results_queue = mgr.Queue()

    logger.info("Starting worker threads")
    logger.info(npix)
    n_threads = min(npix, opts.num_threads)
    logger.info("Starting writer thread.")
    writer_thread = threading.Thread(target=handle_results, args=(f1, results_queue), kwargs={"npix": npix, "denom": denom, "offset": offset})
    writer_thread.start()
    
    with mp.Pool(n_threads) as p:
        # Create the mp threads
        logger.info(f"Launching {n_threads} worker threads")
        args=[(task_queue, 
               results_queue, 
               catalog, 
               nside, 
               galaxy_norm, 
               zarray, 
               zmax, 
               opts.min_gals_for_threshold, 
               zmin, 
               zdraw, 
               sigma,
               cosmo) for _ in range(n_threads)]

        p.starmap_async(LOS_mp_thread, args)
        for pixel_index in pixel_indices: task_queue.put(pixel_index)

        logger.info(f"Awaiting end of all redshift_prior calculations.")
        for _ in range(opts.num_threads): task_queue.put(POISON_PILL)

        p.close()
        p.join()

    logger.info("Stopping writer thread")
    results_queue.put(POISON_PILL)

    # Wait untill everything is written back
    writer_thread.join()
    logger.info(f"Writer thread stopped.")

    opts_string = np.bytes_(vars(opts))
    f1.attrs["opts"] = opts_string
    logger.info(f"opts_string: {opts_string}")

    #############################################################
    ###################### CHECK OUTPUT #########################
    #############################################################
    logger.info(f"Checking output file")
    print('NOPE xoxo Lucas')
    
    # keys = f1.keys()
    # npix_out = len([key for key in keys if key.isdigit()])
    # if npix_out != npix:
    #     logger.warning(f"Number of pixels in output file is {npix_out} which doesn't correspond to expected {npix}")

    # arr_names = ["z_array"]
    # if npix != 1:
    #     arr_names += ["combined_pixels"]
    # arr_names += list(pixel_indices)

    # for name in tqdm(arr_names):
    #     try:
    #         arr = np.array(list(f1[str(name)]))
    #         if np.isnan(arr).any():
    #             logger.warning(f"{name} contains nan values")
    #         if np.isinf(arr).any():
    #             logger.warning(f"{name} contains inf values")
    #     except Warning:
    #         logger.warning(f"Output file doesn't contain {name}")
    f1.close()
    
    print('Done')

if __name__ == "__main__":
    main()
