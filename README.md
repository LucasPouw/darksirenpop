# darksirenpop

Measures the fraction of binary black hole (BBH) mergers that happen in active galactic nuclei (AGN). It cross-matches gravitational-wave (GW) sky maps with an AGN catalogue.

Paper: [arXiv:2609.17665](https://doi.org/10.48550/arXiv.2609.17665)

The code infers $f_{\rm AGN}$, the fraction of GW events that come from a chosen AGN subpopulation. The rest come from an "alternative" population whose merger rate follows a Madau–Dickinson star-formation history. The likelihood accounts for:

- the GW 3D localisation (sky map plus distance ansatz),
- AGN redshift uncertainties,
- catalogue incompleteness as a function of redshift, plus a Galactic-plane mask,
- GW selection effects for both populations.

It runs on simulated (mock) data and on real data. For real data, the paper uses 256 GWTC-5.0 BBH events and the Quaia quasar catalogue up to $z = 3$, split into unobscured AGN subpopulations with $L_{\rm bol} \geq 10^{44.5}$, $10^{45.5}$ and $10^{46.5}\,\mathrm{erg\,s^{-1}}$.

**Notation.** The code and the paper name some quantities differently:

| Paper | Code |
|---|---|
| $Z_A(d_i)$, the AGN-origin evidence | `S_agn_incat + S_agn_outcat` |
| $Z_{\bar A}(d_i)$, the alternative-origin evidence | `S_alt` |
| $\alpha(\Lambda, f_{\rm agn})$, the detection efficiency | `alpha_agn`, `alpha_alt` |
| $\gamma$, $1 + z_{\rm peak}$, $\kappa$ (Madau–Dickinson) | `b`, `c`, `d` |
| $z^{\rm GW}_{\rm thresh}$, the mock GW detection threshold | `--zcut` in `make_mock_gws.py`, `ZTHR` in `run.py` |
| $z^{\rm EM}_{\rm cut}$, the AGN catalogue redshift cut | `AGN_ZCUT` |
| $P^{\rm EM}_{\rm det}(z,\Omega)$, the catalogue completeness | `LUM_THRESH` together with `MASK_GALACTIC_PLANE` |
| $\pi_{\rm agn}(z)$, the AGN redshift distribution | `AGN_ZPRIOR` |

> **Status:** this is research code tied to one machine's directory layout. It does not run out of the box. You must edit the paths in [`utilities/default_globals.py`](utilities/default_globals.py) and provide external data that is not in this repository (see [Data](#data)).

---

## Contents

1. [Installation](#installation)
2. [Data](#data)
3. [Quick start](#quick-start)
4. [Configuration (`Config`)](#configuration-config)
5. [Mock-data pipeline](#mock-data-pipeline)
6. [Real-data pipeline](#real-data-pipeline)
7. [Rate hyperparameter sampling](#rate-hyperparameter-sampling)
8. [Outputs](#outputs)
9. [The likelihood](#the-likelihood)
10. [Repository layout](#repository-layout)
11. [Known issues](#known-issues)
12. [Citation and licence](#citation-and-licence)

---

## Installation

The repository root is itself the `darksirenpop` package; [`pyproject.toml`](pyproject.toml) maps it as such. Install it in editable mode, which also installs the dependencies:

```bash
git clone https://github.com/LucasPouw/darksirenpop.git
cd darksirenpop

python -m venv .venv && source .venv/bin/activate   # Python >= 3.12
pip install -e .                  # or: pip install -e ".[notebooks]" for the notebooks
```

The dependencies include `torch`, because `run.py` imports [`utilities/priors.py`](utilities/priors.py) through `gw_selection_effects`. On a machine without a GPU, you can avoid the large CUDA download by installing the CPU build first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

To reproduce the author's environment exactly, install the full `pip freeze` in [`requirements.txt`](requirements.txt) (Python 3.13) instead. Run it from the repository root, because it contains `-e .`:

```bash
pip install -r requirements.txt
```

The scripts can be run from any directory. The commands below use `$DSP` for the repository path:

```bash
export DSP=/path/to/darksirenpop
```

Check that the install works:

```bash
python -c "from darksirenpop.config import Config; print(Config())"
```

---

## Data

All paths are defined in [`utilities/default_globals.py`](utilities/default_globals.py), and that is the only file you should need to edit. It assumes four top-level directories:

| Variable             | Contents                                                                              |
|----------------------|---------------------------------------------------------------------------------------|
| `REPOSITORY_DIR`     | this repository                                                                       |
| `GENERATED_DATA_DIR` | everything the pipeline produces: AGN distributions, mock GWs, JSONs, posteriors      |
| `GW_DATA_DIR`        | LVK PE samples, population (popsummary) results, injection set, GWOSC event table     |
| `EM_DATA_DIR`        | raw AGN catalogues (Quaia FITS, SDSS DR16Q)                                           |

Inputs that are **not** in the repository:

| File (variable)                                   | Used for                                       | Where it comes from                                   |
|---------------------------------------------------|------------------------------------------------|-------------------------------------------------------|
| `agn_redshift_pdf_<L>_kulkarni.npy` (`AGN_DIST_DIR`) | AGN redshift prior, mock AGN population     | [`data/em/qlf.ipynb`](data/em/qlf.ipynb)              |
| `quaia_zleq<zcut>_<L>_kulkarni.hdf5`, `completeness_zleq<zcut>_<L>_kulkarni.npy` | real-data AGN catalogue and its completeness | [`data/em/qlf.ipynb`](data/em/qlf.ipynb) |
| `QUAIA_PATH` (`quaia_zleq3_withlumcorr.csv`)      | Quaia redshift errors (`AGN_ZERROR='quaia'`)   | [`data/em/luminosity_calibration.ipynb`](data/em/luminosity_calibration.ipynb) |
| `V90_CDF_PATH`                                    | mock GW localisation volumes                   | CDF of the GWTC-4.0 $V_{90\%}$ distribution; the script that produces it is not in this repository |
| GWTC-2.1 to GWTC-5.0 PE samples (`GWTC2P1` … `GWTC5P0`) | real-data GW posteriors                  | LVK Zenodo releases (links in Sect. 2.1 of the paper) |
| `LVK_HYPERPOSTERIOR_PATH`                         | real-data selection effects (BBH mass/spin model) | LVK GWTC-5 population release                      |
| `INJECTIONS_PATH`                                 | real-data selection effects                    | Zenodo record 19500052 (`zenodo_get 19500052`)        |
| `GWOSC_SUMMARY`                                   | SNR/FAR of events                              | CSV download from gwosc.org                           |
| `SKYMAP_JSON_PATH`                                | `{event_name: skymap_path}` for the real events | not generated by code in this repository (the PE-sample JSONs are made by [`data/gw/PE_samples.ipynb`](data/gw/PE_samples.ipynb)) |

The mock GW detection probabilities, [`mock/pdet/pdet_z_<zthr>.npy`](mock/pdet/), *are* included in the repository. [`mock/pdet/calc_mock_pdet.ipynb`](mock/pdet/calc_mock_pdet.ipynb) generates them.

Note: [`utilities/gw_selection_effects.py`](utilities/gw_selection_effects.py) reads the hyperposterior and injection files **when it is imported**. `likelihood.py` imports that module, so `run.py` fails without those files, **even for mock-only runs**.

---

## Quick start

A minimal mock analysis has three steps.

```bash
# Work in your mock-data directory (MOCK_DATA_DIR in default_globals.py), not inside the repository.
cd /path/to/generated_data/mock_gws

# 1. Inject 3000 GWs (random f_AGN). Keep those with observed z < 1.0 (about 160 on average)
#    and make their sky maps. Output is written relative to the current directory.
python $DSP/mock/make_mock_gws.py --run_id 1 --agndist 46.5 --ngw 3000 \
    --zcut 1.0 --npostsamps 1000 --ncpu 4 --make-skymaps

# 2. Turn every sky map into sky-marginalised redshift posteriors.
python $DSP/data/gw/evaluate_skymaps.py \
    --root mock_gws_agndist_46.5_ngw_3000_zmax_10_zcut_1.0_LVKvols

# 3. Infer f_AGN. --zthr must equal the --zcut used in step 1.
python $DSP/run.py \
    --mockdata_root mock_gws_agndist_46.5_ngw_3000_zmax_10_zcut_1.0_LVKvols \
    --n_realizations 1 --agn_zprior 46.5 --lum_thresh zero_upto_cut \
    --agn_zcut 1.0 --agn_zerror 0.01 --zthr 1.0 --add_nagn_to_cat 10000 --verbose True
```

Step 3 writes `p26_post_realdata_False_job0_<timestamp>.npz` to `FAGN_POST_DIR`. It also appends the configuration to `JSON_DIR/metadata_default.json`.

---

## Configuration (`Config`)

[`config.py`](config.py) defines a dataclass `Config`. [`run.py`](run.py) turns **every field into a lower-case CLI flag**, for example `AGN_ZCUT` becomes `--agn_zcut 1.0`. Booleans take `True`/`False`. Dicts take a JSON string, for example `--rate_parameters '{"b": 3.3, "c": 2.55, "d": 6.1}'`. `Config.finalize()` then derives the cosmology, grids, priors and selection function.

The fields you will most often change:

| Field | Default | Meaning |
|---|---|---|
| `REAL_DATA` | `False` | real GWTC events + Quaia, or mock data |
| `MOCKDATA_ROOT` | `None` | mock root directory (required for mock runs) |
| `N_REALIZATIONS` | `1` | number of mock realisations to analyse (`output_run_1_*` … `output_run_N_*`). Use `1` for real data. |
| `THREADING`, `N_WORKERS` | `False`, `16` | run realisations in parallel with `ProcessPoolExecutor` |
| `AGN_ZPRIOR` | `'uniform_comoving_volume'` | AGN redshift distribution: `'uniform_comoving_volume'`, `'positive_redshift'`, or a $\log_{10} L_{\rm bol}$ threshold `'44.5'`, `'45.0'`, `'45.5'`, `'46.0'`, `'46.5'`. A threshold loads `agn_redshift_pdf_<L>_<QLF>.npy`. |
| `LUM_THRESH` | `'zero_upto_cut'` | catalogue completeness: `'zero'` (complete), `'zero_upto_cut'` (complete up to `AGN_ZCUT`), `'inf'` (empty catalogue), or a threshold `'44.5'`…`'46.5'`. A threshold means different things for the two data types. **Mock data:** AGN are thinned per redshift bin to the Veronesi et al. (2025) completeness. **Real data:** the Quaia subcatalogue above that luminosity is loaded, with its QLF-based completeness (Sect. 2.2.3 of the paper). |
| `AGN_ZCUT` | `3.0` | redshift above which the catalogue is empty. Must be ≥ 1.3125 when `LUM_THRESH` is a numeric threshold. |
| `AGN_ZERROR` | `'quaia'` | AGN redshift error: a float (Gaussian σ), `'quaia'` (drawn from the Quaia errors), or `False` (must be combined with `ASSUME_PERFECT_REDSHIFT=True`) |
| `ASSUME_PERFECT_REDSHIFT` | `False` | delta-function AGN redshift posteriors |
| `ADD_NAGN_TO_CAT` | `350000` | mock only: number of AGN in the mock catalogue. Hosts are added until this number is reached. |
| `MASK_GALACTIC_PLANE` | `True` | drop AGN with Galactic latitude $\|b\| \le 10^\circ$ |
| `MERGER_RATE` | `'madau'` | alternative-population rate evolution: `'madau'` (needs `RATE_PARAMETERS`), `'madau_lvk_MAP'` (GWTC-5 MAP), `'lowZ_sfr'`, `'uniform'` |
| `RATE_PARAMETERS` | `{}` | `{"b", "c", "d"}` for `'madau'` (see [below](#rate-hyperparameter-sampling)) |
| `ZTHR` | `inf` | mock GW detection threshold $z^{\rm GW}_{\rm thresh}$: loads `mock/pdet/pdet_z_<ZTHR>.npy` for `ZTHR` ∈ {0.3, 0.4, …, 1.0}. **Must equal the `--zcut` the mock data were generated with.** `inf` means no selection effects. Ignored for real data. |
| `SKYMAP_CL` | `0.999` | only sky-map pixels inside this credible level are integrated |
| `ZMIN`, `ZMAX` | `1e-6`, `10` | redshift integration range |
| `METADATA_PATH` | `JSON_DIR/metadata_default.json` | where the run configuration is logged |
| `LABEL` | `'none'` | tag added to the real-data per-event JSON filename |
| `JOB_ID` | `0` | added to the output filename. Use it to avoid filename collisions on a cluster. |

The fixed constants are cosmology ($H_0 = 67.9$, $\Omega_m = 0.3065$, Planck15 otherwise), the $f_{\rm AGN}$ grid (1000 points in $[10^{-4}, 1-10^{-4}]$) and the redshift integration grid (1025 points, which suits Romberg integration).

Two settings are not supported:

- `USE_SKYMAPS=False` raises `NotImplementedError`. Only the sky-map likelihood is implemented.
- `CMAP_PATH` is unused. The Galactic-plane cut is hard-coded.

---

## Mock-data pipeline

### 1. Simulate GWs: [`mock/make_mock_gws.py`](mock/make_mock_gws.py)

```bash
python $DSP/mock/make_mock_gws.py --run_id <i> --agndist {44.5,45.5,46.5} --ngw <N> \
    [--zcut Z] [--fagn F] [--npostsamps 5000] [--ncpu 1] [--make-skymaps] [--overwrite] [--verbose]
```

`--ngw` is the **total number of injected** GWs ($N^{\rm tot}_{\rm inj}$), not the number detected. This follows Sect. 4.1.1 of the paper. For each injected GW, the script:

1. assigns an AGN origin with probability `F` and an alternative origin otherwise. `F` is drawn from U(0, 1) if `--fagn` is not given;
2. samples the true redshift from $\pi_{\rm agn}(z)/(1+z)$ for AGN-origin GWs, or from the alternative-population prior (fiducial rate $\gamma, 1+z_{\rm peak}, \kappa = 3.3, 2.55, 6.1$);
3. samples an isotropic sky position;
4. draws a $V_{90\%}$ from `V90_CDF_PATH` and offsets the observed position by a spherical Gaussian in comoving coordinates, with $\sigma = (3V_{90\%}/4\pi)^{1/3} / \sqrt{2P^{-1}(3/2, 0.9)}$;
5. keeps the GW as **detected** only if its *observed* redshift is below `--zcut` ($z^{\rm GW}_{\rm thresh}$);
6. writes posterior samples (RA, Dec, $d_L$, $d_C$, $z$) for detected GWs, drawn from the Gaussian likelihood, so the PE prior is uniform in comoving volume. With `--make-skymaps` it also writes FITS sky maps via `ligo-skymap-from-samples`.

The paper's fiducial setup is `--ngw 100000 --zcut 0.3`, which gives about 160 detections on average, similar to GWTC-4.0. Table 1 of the paper lists `--ngw` for the other thresholds.

**Sky maps are required by the inference, so always pass `--make-skymaps`.** The output goes to the *current working directory*:

```
mock_gws_agndist_<L>_ngw_<N>_zmax_<ZMAX>_zcut_<ZCUT>_LVKvols/
└── output_run_<i>_fagn_<F>/
    ├── posterior_samples/{agn,alt}/
    ├── skymaps/{agn,alt}/
    └── true_gw_coords/{agn,alt}/
```

The true $f_{\rm AGN}$ is read back from the directory name, so **do not rename these directories**. [`bash_scripts/sweep_gw_generation.sh`](bash_scripts/sweep_gw_generation.sh) loops over `run_id` to make many realisations.

### 2. Optional housekeeping

- [`mock/combine_posterior_samples.py`](mock/combine_posterior_samples.py) merges the per-event HDF5 files into `samples_{agn,alt}.h5` and deletes the originals.
- [`mock/calculate_mock_skymap_stats.py`](mock/calculate_mock_skymap_stats.py) runs `ligo-skymap-stats` on each sky map.

Both scripts have a **hard-coded `ROOT`** at the top. Edit it before running.

### 3. Evaluate the sky maps: [`data/gw/evaluate_skymaps.py`](data/gw/evaluate_skymaps.py)

```bash
python $DSP/data/gw/evaluate_skymaps.py --root <MOCK_ROOT>
```

For every GW, the script integrates the sky map over the sky to get $p(z \mid d_{\rm GW})$ on the redshift grid. It does this twice: over the full sky, and over the sky outside the Galactic plane. The second version is used in the completeness-weighted terms. Results go to `output_run_*/skymaps_evaluated/{agn,alt}/zpost_gpmask_{False,True}_skymapcl_0.999_cmapnside_64.h5`.

### 4. Infer $f_{\rm AGN}$: [`run.py`](run.py)

```bash
python $DSP/run.py --mockdata_root <MOCK_ROOT> --n_realizations <R> --threading True --n_workers 4 ...
```

For each realisation, the script builds a mock AGN catalogue. It contains the true hosts of all AGN-origin GWs, detected or not, plus uncorrelated AGN drawn from $\pi_{\rm agn}(z)$, up to `ADD_NAGN_TO_CAT` sources in total. The paper uses $3.5\times10^5$, the QLF-integrated number up to $z=10$. The script then draws observed redshifts with errors `AGN_ZERROR` and applies the completeness and the Galactic mask. Finally, it evaluates the likelihood on the $f_{\rm AGN}$ grid.

The settings below match the paper's validation run (Sect. 4.1 of the paper; the P–P plot in Fig. 4). They use 200 realisations generated with `--agndist 46.5 --ngw 100000 --zcut 0.3`:

```bash
python $DSP/run.py --mockdata_root <MOCK_ROOT> --n_realizations 200 --threading True \
    --agn_zprior 46.5 --lum_thresh 46.5 --agn_zcut 1.5 --agn_zerror quaia \
    --mask_galactic_plane True --zthr 0.3 --add_nagn_to_cat 350000
```

The catalogue-quality study (Fig. 6 of the paper) uses `--lum_thresh zero_upto_cut`, a fixed float `--agn_zerror`, and `--zthr 1.0`, with sweeps over `--agn_zerror` and `--agn_zcut`. [`bash_scripts/grid_sweep.sh`](bash_scripts/grid_sweep.sh) and [`bash_scripts/rerun_missing.sh`](bash_scripts/rerun_missing.sh) run this sweep. For the empty- and perfect-catalogue limits (Fig. 5 of the paper), use `--lum_thresh inf`, or `--lum_thresh zero --assume_perfect_redshift True --agn_zerror False`.

---

## Real-data pipeline

0. **Select and reweight the events** with [`data/gw/PE_samples.ipynb`](data/gw/PE_samples.ipynb). Following Sect. 2.1 of the paper, the notebook:
   - selects BBH events with SNR > 10 (O1–O2) or FAR < 1/yr (O3–O4), which gives 256 events;
   - uses the `IMRPhenomXPHM` samples, or `IMRPhenomXPHM-SpinTaylor` for O4;
   - importance-resamples them to a PE prior that is uniform in comoving volume, with the LVK MAP `Broken Powerlaw + 2 Peaks` mass population;
   - writes the samples to `REWEIGHT_GWTC5_SAMPLES` and their paths to JSON.
1. **Prepare sky maps.**
   - [`data/gw/make_skymaps.py`](data/gw/make_skymaps.py) runs `ligo-skymap-from-samples` on everything in `REWEIGHT_GWTC5_SAMPLES` and writes to `REWEIGHT_GWTC5_SKYMAPS`.
   - [`data/gw/make_skymap_stats.py`](data/gw/make_skymap_stats.py) is optional and produces 90% areas and volumes.
2. **Write `SKYMAP_JSON_PATH`.** This is a JSON file mapping `event_name → skymap path`. No script in this repository creates it.
3. **Evaluate the sky maps:**
   ```bash
   python $DSP/data/gw/evaluate_skymaps.py --real-data
   ```
   The script writes one `.npy` per event to `REWEIGHT_GWTC5_SKYMAP_EVALS` and records the paths in `SKYMAP_EVALS_JSON_PATH` and `SKYMAP_CW_EVALS_JSON_PATH`.
4. **Prepare the AGN catalogue.** Run [`data/em/qlf.ipynb`](data/em/qlf.ipynb) to produce `quaia_zleq<AGN_ZCUT>_<L>_kulkarni.hdf5`, which holds positions and precomputed redshift posteriors, and the matching completeness `.npy` in `AGN_DIST_DIR`.
5. **Run the inference:**
   ```bash
   python $DSP/run.py --real_data True --n_realizations 1 \
       --agn_zprior 46.5 --lum_thresh 46.5 --agn_zcut 3.0 --agn_zerror quaia \
       --merger_rate madau --rate_parameters '{"b": 3.3, "c": 2.55, "d": 6.1}' \
       --label review --verbose True
   ```
   Use `--lum_thresh inf` for the "no catalogue" comparison. [`bash_scripts/realdata_sweep.sh`](bash_scripts/realdata_sweep.sh) loops over AGN luminosity thresholds 44.5, 45.5 and 46.5, each with and without the catalogue.

Real-data selection effects ($\alpha_{\rm AGN}$, $\alpha_{\rm alt}$) are computed from the GWTC-5.0 search-sensitivity injections, with the same cuts as the event selection (`SNR_THR=10`, `FAR_THR=1`).

The fiducial run in the paper (Figs. 7 and 8) uses `--merger_rate madau --rate_parameters '{"b": 3.3, "c": 2.55, "d": 6.1}'`, the Strolger et al. (2020) fit. It is run for each `L` in {44.5, 45.5, 46.5} with `--agn_zprior L`, and with both `--lum_thresh L` (with Quaia) and `--lum_thresh inf` (GW-only). This is exactly what `realdata_sweep.sh` does.

---

## Rate hyperparameter sampling

[`hypersampler/hypersampler.py`](hypersampler/hypersampler.py) samples $(b, c, d, f_{\rm AGN})$ jointly with `emcee`. The alternative-population merger rate is

$$
\psi(z) = \frac{(1+z)^{b}}{1 + \left(\frac{1+z}{c}\right)^{d}} ,
$$

where $b = \gamma$ is the low-$z$ slope, $c = 1 + z_{\rm peak}$, and $d = \kappa$ is the high-$z$ slope (Sect. 6.2 and Table 2 of the paper). The sampler reuses the per-event evidences that `run.py` stored in step 5 of the real-data pipeline and recomputes the alternative population's terms at each step. It reads exactly `JSON_DIR/realdata_posterior_madau_review.json`, so **run the real-data sweep with `--label review` first**, or edit `label` in `main()`.

```bash
python $DSP/hypersampler/hypersampler.py --ncpu 8 --prior gaussian --lthresh 46.5 \
    --nwalkers 100 --nsteps 10000
```

| `--prior` (paper name)  | $b=\gamma$ | $c=1+z_{\rm peak}$ | $d=\kappa$ | $f_{\rm AGN}$ |
|-------------------------|-----|-----|-----|---------------|
| `uniform` ("LVK")       | U(−10, 10) | U(1, 3.5) | U(0, 10) | U(0, 1) |
| `gaussian` ("Gaussian") | N(3.3, 0.2) | N(2.55, 0.09) | N(6.1, 0.2) | U(0, 1) |

The paper's third case, "Fiducial", fixes $b, c, d$. It is just `run.py` with `--merger_rate madau`, so the hypersampler is not needed for it.

Chains are written to `GENERATED_DATA_DIR/alt_origin_hyperparameters/sampler_<prior>_priors_withcat_<L>.h5`, and a run resumes from that file if it exists. Analyse the chains with [`hypersampler/cornerplot_hypersamples.ipynb`](hypersampler/cornerplot_hypersamples.ipynb) and [`hypersampler/rate_posteriors.ipynb`](hypersampler/rate_posteriors.ipynb).

---

## Outputs

| Output | Location | Contents |
|---|---|---|
| $f_{\rm AGN}$ log-likelihood | `POST_DIR/p26_post_realdata_<REAL_DATA>_job<JOB_ID>[_<LUM_THRESH>]_<timestamp>.npz` | `log_likelihood`, an array of shape `(1000, N_REALIZATIONS)` on the grid `np.linspace(1e-4, 1-1e-4, 1000)`, unnormalised and with selection effects already included; `true_fagns`, the injected values (mock) or a placeholder 0.5 (real data) |
| Run metadata | `METADATA_PATH` | JSON list. Each entry is `{filename, timestamp, config}`. Use it to find which settings produced a given `.npz`. |
| Per-event evidences (real data only) | `JSON_DIR/realdata_posterior[_nocat]_<MERGER_RATE>[_<LABEL>].json` | `{AGN_ZPRIOR: {event: {S_agn_incat, S_agn_outcat, S_alt, alpha_agn, alpha_alt}}}`, used for per-event analysis and by the hypersampler |

To turn a result into a posterior with a flat prior:

```python
import numpy as np
d = np.load(fname)
x = np.linspace(1e-4, 1 - 1e-4, 1000)
logL = d["log_likelihood"][:, 0]
post = np.exp(logL - logL.max())
post /= np.trapezoid(post, x)
```

[`plotting_scripts/`](plotting_scripts/) contains the notebooks that made the paper figures.

---

## The likelihood

For each GW event $i$, [`likelihood.py`](likelihood.py) computes three evidences from its sky-marginalised redshift posterior:

- $S^{\rm in}_{{\rm AGN},i}$: the host is an AGN **in** the catalogue. This is a sum over catalogue AGN, weighted by the sky-map density at each AGN position and integrated over each AGN's redshift posterior.
- $S^{\rm out}_{{\rm AGN},i}$: the host is an AGN **missing** from the catalogue. This uses the AGN redshift distribution weighted by catalogue incompleteness, $\pi_{\rm agn}(z)\,[1 - P^{\rm EM}_{\rm det}(z,\Omega)]$, including the Galactic plane.
- $S_{{\rm alt},i}$: the host belongs to the alternative population, with redshift prior $\propto \psi(z)\,\frac{dV_c}{dz}\,\frac{1}{1+z}$.

Both AGN terms carry a time-dilation factor $1/(1+z)$, which is controlled by `CORRECT_TIME_DILATION`. Together they form $Z_A$ (Eq. 13 of the paper, with the prior of Eq. 10), and $S_{\rm alt}$ is $Z_{\bar A}$ (prior of Eq. 11). The GW likelihood is the sky-map posterior divided by a PE prior that is uniform in comoving volume. Only the 99.9% credible sky area (`SKYMAP_CL`) is integrated.

The log-likelihood for $N$ detected events is

$$
\ln \mathcal{L}(f_{\rm AGN}) = \sum_{i=1}^{N} \ln\!\Big[f_{\rm AGN}\big(S^{\rm in}_{{\rm AGN},i} + S^{\rm out}_{{\rm AGN},i}\big) + (1-f_{\rm AGN})\,S_{{\rm alt},i}\Big] \;-\; N \ln\!\big[f_{\rm AGN}\,\alpha_{\rm AGN} + (1-f_{\rm AGN})\,\alpha_{\rm alt}\big],
$$

where $\alpha_{\rm AGN}$ and $\alpha_{\rm alt}$ are the detection efficiencies of the two populations. This is Eq. 14 of the paper, with $\alpha(\Lambda, f_{\rm agn}) = f_{\rm agn}\alpha_{\rm AGN} + (1-f_{\rm agn})\alpha_{\rm alt}$. Appendix A of the paper gives the derivation.

Code map:

| Function | Role |
|---|---|
| `process_one_fagn` | full calculation for one realisation (entry point used by [`worker.py`](worker.py)) |
| `make_mock_agn_catalog` / `load_quaia` | build or load the AGN catalogue and its redshift posteriors |
| `prepare_functions` | redshift priors, completeness, Jacobians |
| `calculate_normalizations` | AGN prior normalisation, $\alpha_{\rm AGN}$, $\alpha_{\rm alt}$ |
| `calculate_evidence` | the three $S$ terms for one GW |

---

## Repository layout

```
run.py, worker.py      CLI entry point; serial/parallel loop over realisations
config.py              Config dataclass and finalize()
likelihood.py          the likelihood (see above)
utilities/
  default_globals.py   ALL paths — edit this
  redshift_utils.py    cosmology interpolators, merger-rate models, d_L→z Jacobians
  mockdata_utils.py    mock AGN catalogue, AGN redshift posteriors, completeness
  gw_selection_effects.py  real-data α from LVK injections + population model
  qlf_utils.py         quasar luminosity functions (Kulkarni et al.)
  priors.py, custom_math_priors.py  BBH population priors
  utils.py             sampling, coordinates, plotting helpers
mock/                  mock GW generation, pdet tables, validation notebooks
data/em/               QLF / Quaia luminosity calibration notebooks
data/gw/               real sky-map creation and sky-map → p(z) evaluation
hypersampler/          emcee sampler over (b, c, d, f_AGN) and its notebooks
bash_scripts/          sweeps used for the paper (author-specific paths)
plotting_scripts/      figure notebooks
plots/                 paper figures
```

---

## Known issues

- **Hard-coded paths.** Paths appear in `default_globals.py`, in every script under `bash_scripts/`, in `combine_posterior_samples.py`, `calculate_mock_skymap_stats.py`, and in the `test` branch of `hypersampler.py`.
- **`bash_scripts/rerun_missing.sh` passes `--outfile`.** That is not a `Config` field, so `run.py` rejects it. Use `--metadata_path` instead.
- **`mock/calculate_mock_skymap_stats.py` globs `output_run_<i>/`.** The generator actually writes `output_run_<i>_fagn_<F>/`.
- **Random seeds are not saved.** Each mock realisation draws a fresh seed.
- **Memory.** AGN redshift posteriors are stored as an (N_AGN × 1025) array. With the default `ADD_NAGN_TO_CAT = 3.5e5` this is about 3 GB per worker, so choose `N_WORKERS` with that in mind.
- **No automated tests.** Validation was done in the notebooks under `mock/`.

---

## Citation and licence

If you use this code, please cite the paper ([arXiv:2609.17665](https://doi.org/10.48550/arXiv.2609.17665)). Please also cite the datasets it relies on: Quaia, the LVK GWTC catalogues and population results, and the LVK injection set.

The repository has no licence file yet. Until one is added, contact the author before reusing the code.
