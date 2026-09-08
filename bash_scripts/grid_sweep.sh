#!/bin/bash

#######
# Completeness vs agn err plot
#######

zcuts=$(python -c "import numpy as np; print(' '.join(map(str, np.linspace(0, 2, 2)[::-1])))")
agn_errs=$(python -c "import numpy as np; print(' '.join(map(str, np.geomspace(0.001, 1, 1))))")

for zcut in $zcuts; do
    for agn_err in $agn_errs; do
        echo $zcut
        echo $agn_err

        ~/Documents/PhD/.venv/bin/python darksirenpop/run.py \
        --threading True \
        --n_workers 1 \
        --verbose True \
        --mockdata_root /home/lucas/Documents/PhD/generated_data/mock_gws/mock_gws_agndist_46.5_ngw_3000_zmax_10_zcut_1.0_LVKvols \
        --n_realizations 1 \
        --agn_zprior 46.5 \
        --lum_thresh zero_upto_cut \
        --agn_zcut $zcut \
        --assume_perfect_redshift False \
        --agn_zerror $agn_err \
        --mask_galactic_plane True \
        --zmin 0.000001 \
        --zmax 10 \
        --zthr 1.0 \
        --add_nagn_to_cat 10000 \
        --metadata_path /home/lucas/Documents/PhD/generated_data/jsons/metadata_test.json \
        --label testing
    done
done
