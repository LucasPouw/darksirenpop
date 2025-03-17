# darksirenpop
This is mostly just `gwcosmo` stripped for parts and applied to distinguishing two populations using more GW event parameters than just position and mass. 
I also added functionality for mock GW data generation.

### Making AGN catalog
Run `mock_catalog_maker.py`. Does not work yet in the command line.
Automatically saves the catalog as `.hdf5` in `./output/catalogs/`.

Also makes a normalization map for the LOS zprior and stores it as
`.fits` in `./output/maps/`

### Making Line-of-sight redshift prior
Run `LOS_zprior.py` from the command line.

e.g.,

`python3 compute_zprior.py --zmax 10 --nside 64 --catalog_name MOCK --catalog_path /your/path/output/catalogs/mockcat_NAGN750000_ZMAX5_GWZMAX2.hdf5 --maps_path /your/path/output/maps/mocknorm_NAGN750000_ZMAX5_GWZMAX2.fits --min_gals_for_threshold 10 --num_threads 6`
