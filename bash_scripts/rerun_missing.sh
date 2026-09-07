#!/bin/bash

# ---------------------------------------------------------
# Find missing parameter combinations and rerun only those
# ---------------------------------------------------------

missing_runs=$(python <<'PYTHON'
import json
import numpy as np

# -------------------------------------------------
# Full expected parameter grid
# -------------------------------------------------

zcuts = np.linspace(0, 2, 50)
agn_errs = np.geomspace(0.001, 1, 50)[::-1]

expected = {
    (agn_err, zcut)
    for agn_err in agn_errs
    for zcut in zcuts
}

# -------------------------------------------------
# Load completed runs
# -------------------------------------------------

with open("combined_46p5_1p0.json", "r") as f:
    configs = json.load(f)

completed = {
    (
        cfg['config']["AGN_ZERROR"],
        cfg['config']["AGN_ZCUT"],
    )
    for cfg in configs
}

# -------------------------------------------------
# Compute missing runs
# -------------------------------------------------

missing = sorted(expected - completed)

# Print as space-separated pairs
for agn_err, zcut in missing:
    print(f"{agn_err} {zcut}")

PYTHON
)

# ---------------------------------------------------------
# Run missing jobs
# ---------------------------------------------------------

while read -r agn_err zcut; do

    # skip empty lines
    [ -z "$agn_err" ] && continue

    echo "Running missing job:"
    echo "  agn_zerror = $agn_err"
    echo "  agn_zcut   = $zcut"

    ~/Documents/PhD/.venv/bin/python darksirenpop/run.py \
        --threading True \
        --n_workers 2 \
        --verbose False \
        --mockdata_root mock_gws_agndist_46.5_ngw_3000_zmax_10_zcut_1.0_LVKvols \
        --n_realizations 50 \
        --agn_zprior 46.5 \
        --lum_thresh zero_upto_cut \
        --agn_zcut "$zcut" \
        --assume_perfect_redshift False \
        --agn_zerror "$agn_err" \
        --mask_galactic_plane True \
        --zmin 0.000001 \
        --zmax 10 \
        --zthr 1.0 \
        --add_nagn_to_cat 350000 \
        --outfile A_final_46p5_1p0_grid.json.json

done <<< "$missing_runs"
