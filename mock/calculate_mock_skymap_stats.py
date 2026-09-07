import os
import numpy as np
import glob
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"

ROOT_DIRECTORY = '/home/lucas/Documents/PhD/generated_data/mock_gws/mock_gws_agndist_46.5_ngw_3000_zmax_10_zcut_1.0_LVKvols'
TYPES = ['agn', 'alt']
DIRECTORY_IDS = np.arange(1, 201, 1)

for DIRECTORY_ID in DIRECTORY_IDS:
    for TYPE in TYPES:
        print(DIRECTORY_ID, TYPE)

        SKYMAP_DIR = f'{ROOT_DIRECTORY}/output_run_{DIRECTORY_ID}/skymaps/{TYPE}/'
        WRITE_DIR = f'{ROOT_DIRECTORY}/output_run_{DIRECTORY_ID}/skymap_stats/{TYPE}/'
        if not os.path.isdir(WRITE_DIR):
            os.makedirs(WRITE_DIR)

        gw_fnames = glob.glob(SKYMAP_DIR + 'skymap*.fits.gz')
        for i, filename in tqdm(enumerate(gw_fnames), total=len(gw_fnames)):
            id = filename[-13:-8]
            os.system(f'ligo-skymap-stats {filename} -p 90 --cosmology --output {f"{WRITE_DIR}{id}.dat"}')

