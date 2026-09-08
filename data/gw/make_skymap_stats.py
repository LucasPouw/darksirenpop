import glob
import os
from tqdm import tqdm
from darksirenpop.utilities.default_globals import *

os.environ["OMP_NUM_THREADS"] = "1"  # Important for proper threading when making skymaps

skymap_dir = REWEIGHT_GWTC5_SKYMAPS
write_dir = REWEIGHT_GWTC5_SKYMAP_STATS

for infile in tqdm( glob.glob(skymap_dir + '/*') ):
    gwname = infile.split('/')[-1].split('.')[0]
    outfile = f'{gwname}.fits.gz'
    os.system(f'ligo-skymap-stats {infile} -p 90 --cosmology --output {f"{write_dir}/{gwname}.dat"}')
