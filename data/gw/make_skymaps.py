import glob
import os
from darksirenpop.utilities.default_globals import *

os.environ["OMP_NUM_THREADS"] = "1"  # Important for proper threading when making skymaps

post_samps_dir = REWEIGHT_GWTC5_SAMPLES
skymap_dir = REWEIGHT_GWTC5_SKYMAPS

for infile in glob.glob(post_samps_dir + '/*'):
    gwname = infile.split('/')[-1].split('.')[0]
    outfile = f'{gwname}.fits.gz'
    os.system(f"ligo-skymap-from-samples {infile} --fitsoutname {outfile} --outdir {skymap_dir} --jobs {30}")
