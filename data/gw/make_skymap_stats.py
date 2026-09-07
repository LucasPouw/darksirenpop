import glob
import os
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"  # Important for proper threading when making skymaps

SKYMAP_DIR = '/home/lucas/Documents/PhD/generated_data/gw/reweighted-gwtc5/skymaps'
WRITE_DIR = '/home/lucas/Documents/PhD/generated_data/gw/reweighted-gwtc5/skymap_stats'

for infile in tqdm( glob.glob(SKYMAP_DIR + '/*') ):
    gwname = infile.split('/')[-1].split('.')[0]
    outfile = f'{gwname}.fits.gz'
    os.system(f'ligo-skymap-stats {infile} -p 90 --cosmology --output {f"{WRITE_DIR}/{gwname}.dat"}')
