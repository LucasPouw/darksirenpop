# darksirenpop
This is code contains parts of `gwcosmo`, repurposed for distinguishing two populations using more GW event parameters than just position (cf. Veronesi et al.). 
I also added functionality for mock GW data generation.


### Mock data generation
All mock data generation can be done through `main.py`. I have not implemented a parser yet, so please manually change the input variables
in the function `main()`. This function then calls the class in `mock_catalog_maker.py` and saves an AGN catalog in `.hdf5` format at your given path.
If you want to generate GWs from a previous catalog, just change line `cat_path = make_agn_catalog(...)` to `cat_path = \your\path\catalog.hdf5`.
But make sure the arguments above match your input catalog!

Then, GW posteriors are generated using the class in `mock_event_maker.py`. This will generate `.h5` files with two layers. 
The top layer is the group called `mock`, which is analogous to the approximant group in real GW data.
This group contains the dataset `posterior_samples`, which is an `astopy.table.Table` object with columns such as `ra`, `dec` and `luminosity_distance`.
If the provided output directory already contains files, the user is asked
if these files may be deleted. Type 'y' or 'n' in the terminal and press `enter`.

Finally, a `.json` file is generated, which is used to locate GW posteriors by name, e.g., `{'gw_00013': /path/to/gw_00013.hdf5'}`. The path to this
file is needed in the likelihood calculations.


### Likelihood estimation
Just look at `zero_agn_error_likelihood.py` for now, don't bother with `pixelated_likelihood.py`, because that requires a LOS zprior and that's
not necessary for now. Again, only provide the inputs indicated in the function `main()`. Note that the `outfilename` variable will be used in
the `checkresults.ipynb` as well.

This code calculates the integral in the likelihood. Specifically,

$
\begin{equation}
\int p(\theta | d_{\rm GW}) \left[f_{\rm agn} \cdot f_{\rm c}(\Omega) \cdot p_{\rm agn}(z | Omega) + (1 - f_{\rm agn} \cdot f_{\rm c}(\Omega)) \cdot p_{\rm alt}(z) \right]
\end{equation}
$

which is split into three integrals:

$
\begin{equation}
\int p(\theta | d_{\rm GW}) \left[f_{\rm agn} \cdot f_{\rm c}(\Omega) \cdot p_{\rm agn}(z | Omega) \right]
\end{equation}

\begin{equation}
\int p(\theta | d_{\rm GW}) \left[p_{\rm alt}(z) \right]
\end{equation}

\begin{equation}
\int p(\theta | d_{\rm GW}) \left[f_{\rm agn} \cdot f_{\rm c}(\Omega) \cdot p_{\rm alt}(z) \right]
\end{equation}
$

These are saved as three seperate `.npy` files used in the next notebook.

### Checking results
Use `checkresults.ipynb` to make the diagnostic plots I have been showing. You only have to provide the path to the `.json` file you previously
generated and the `outfilename` variable used in the likelihood calculation.


<!-- ### Making AGN catalog
Run `mock_catalog_maker.py`. Does not work yet in the command line.
Automatically saves the catalog as `.hdf5` in `./output/catalogs/`.

Also makes a normalization map for the LOS zprior and stores it as
`.fits` in `./output/maps/`.

### Making Line-of-sight redshift prior
Run `compute_zprior.py` from the command line.

For example,

`python3 compute_zprior.py --zmax 3 --zdraw 2 --zmin 1e-10 --nside 32 --catalog_name MOCK --catalog_path '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs/mockcat_NAGN_100000_ZMAX_3_SIGMA_0.01_incomplete.hdf5' --maps_path '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/maps/mocknorm_NAGN_100000_ZMAX_3_SIGMA_0.01_incomplete.fits' --min_gals_for_threshold 1 --num_threads 6` -->

<!-- python3 compute_zprior.py --zmax 3 --nside 32 --catalog_name MOCK --catalog_path '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/catalogs/mockcat_NAGN_100000_ZMAX_3_SIGMA_0.01_incomplete_v22.hdf5' --maps_path '/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/maps/mocknorm_NAGN_100000_ZMAX_3_SIGMA_0.01_incomplete_v22.fits' --min_gals_for_threshold 1 --num_threads 6 --zdraw 2 --zmin 1e-10 --sigma 0.01 -->

<!-- ### Generating mock GW data
Run `main.py`. This will generate `.h5` files with two layers. 
The top layer is the group called `mock`, which is analogous to the approximant group in real GW data.
This group contains the dataset `posterior_samples`, which is an `astopy.table.Table` object with columns such as `ra`, `dec` and `luminosity_distance`. -->

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