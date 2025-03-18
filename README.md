# darksirenpop
This is mostly just `gwcosmo` stripped for parts and applied to distinguishing two populations using more GW event parameters than just position and mass. 
I also added functionality for mock GW data generation.

### Making AGN catalog
Run `mock_catalog_maker.py`. Does not work yet in the command line.
Automatically saves the catalog as `.hdf5` in `./output/catalogs/`.

Also makes a normalization map for the LOS zprior and stores it as
`.fits` in `./output/maps/`

### Making Line-of-sight redshift prior
Run `compute_zprior.py` from the command line.

For example,

`python3 compute_zprior.py --zmax 10 --nside 64 --catalog_name MOCK --catalog_path /your/path/output/catalogs/mockcat_NAGN750000_ZMAX5_GWZMAX2.hdf5 --maps_path /your/path/output/maps/mocknorm_NAGN750000_ZMAX5_GWZMAX2.fits --min_gals_for_threshold 10 --num_threads 6`

### Generating mock GW data
Run `mock_event_maker.py`. This will generate `.h5` files with two layers. 
The top layer is the group called `mock`, which is analogous to the approximant group in real GW data.
This group contains the dataset `posterior_samples`, which is an `astopy.table.Table` object with columns such as `ra`, `dec` and `rlum`.

### Calculating sky areas
The confidence sky areas are needed. Currently, I only know how to do this by first calling
`ligo-skymap-from-samples` and then `ligo-skymap-stats`. I made a wrapper for these functions called
`make_skymaps.sh`.

For example,

`bash make_skymaps.sh --indir /your/path/output/h5dir --outdir /your/path/output/fitsdir --jobs 4`

With 1000 samples and 6 jobs, I get 5s/map.
Doing 5000 samples and 6 jobs, 20-25 s/map.
<!-- bash make_skymaps.sh --indir /net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/mock_posterior_samples --outdir /net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/mock_skymaps --jobs 6 -->