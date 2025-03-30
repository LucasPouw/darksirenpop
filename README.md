# darksirenpop
This is code contains parts of `gwcosmo`, repurposed for distinguishing two populations using more GW event parameters than just position (cf. Veronesi et al.). 
I also added functionality for mock GW data generation.

### Making AGN catalog
Run `mock_catalog_maker.py`. Does not work yet in the command line.
Automatically saves the catalog as `.hdf5` in `./output/catalogs/`.

Also makes a normalization map for the LOS zprior and stores it as
`.fits` in `./output/maps/`.

### Making Line-of-sight redshift prior
Run `compute_zprior.py` from the command line.

For example,

`python3 compute_zprior.py --zmax 3 --zdraw 2 --zmin 1e-10 --nside 32 --catalog_name MOCK --catalog_path '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs/mockcat_NAGN_100000_ZMAX_3_SIGMA_0.01_incomplete.hdf5' --maps_path '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/maps/mocknorm_NAGN_100000_ZMAX_3_SIGMA_0.01_incomplete.fits' --min_gals_for_threshold 1 --num_threads 6`

<!-- python3 compute_zprior.py --zmax 3 --nside 16 --catalog_name MOCK --catalog_path '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs/mockcat_NAGN_100000_ZMAX_3_SIGMA_0.01_incomplete_v13.hdf5' --maps_path '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/maps/mocknorm_NAGN_100000_ZMAX_3_SIGMA_0.01_incomplete_v13.fits' --min_gals_for_threshold 1 --num_threads 6 --zdraw 2 --zmin 1e-10 --sigma 0.01 -->

### Generating mock GW data
Run `mock_event_maker.py`. This will generate `.h5` files with two layers. 
The top layer is the group called `mock`, which is analogous to the approximant group in real GW data.
This group contains the dataset `posterior_samples`, which is an `astopy.table.Table` object with columns such as `ra`, `dec` and `rlum`.

<!-- ### Calculating skymaps
The confidence sky areas are needed as well as the pixelated sky probabilities. This is easiest by making mock skymaps using
`ligo-skymap-from-samples` from `ligo.skymap`. I made a wrapper for this function called `make_skymaps.sh`. This allows automatic generations
of skymaps from all posterior sample files in a directory.

Arguments:
`--indir`
`--outdir`
`--jobs`
`--skip`

For example,

`bash make_skymaps.sh --indir /your/path/output/h5dir --outdir /your/path/output/fitsdir --jobs 4`

With 1000 samples and 6 jobs, I get 5s/map.
Doing 50000 samples and 6 jobs, 20-25 s/map.
bash make_skymaps.sh --indir /net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/mock_posterior_samples --outdir /net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/mock_skymaps --jobs 6 --skip 4700 -->