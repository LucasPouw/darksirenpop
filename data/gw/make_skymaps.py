import glob
import os

os.environ["OMP_NUM_THREADS"] = "1"  # Important for proper threading when making skymaps

POST_SAMPS_DIR = '/home/lucas/Documents/PhD/generated_data/gw/reweighted-gwtc5/samples'
SKYMAP_DIR = '/home/lucas/Documents/PhD/generated_data/gw/reweighted-gwtc5/skymaps'

for infile in glob.glob(POST_SAMPS_DIR + '/*'):
    gwname = infile.split('/')[-1].split('.')[0]
    outfile = f'{gwname}.fits.gz'
    os.system(f"ligo-skymap-from-samples {infile} --fitsoutname {outfile} --outdir {SKYMAP_DIR} --jobs {30}")
