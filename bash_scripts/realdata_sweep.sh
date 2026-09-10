#!/bin/bash

#######
# REAL DATA WITH AND WITHOUT CATALOG
# Fiducial: '{"b": 3.3, "c": 2.55, "d": 6.1}'
# MF (Madau Fragos): '{"b": 2.6, "c": 3.2, "d": 6.2}'
#######

for rate in 'madau'; do
    echo $rate

    for agn_prior_lum in '46.5' '45.5' '44.5'; do
        echo $agn_prior_lum

        for catalog_lum_thresh in 'inf' $agn_prior_lum; do
            
            ~/Documents/PhD/.venv/bin/python darksirenpop/run.py \
                --label 'testing3' \
                --rate_parameters '{"b": 3.3, "c": 2.55, "d": 6.1}' \
                --catalog_path '/home/lucas/Documents/PhD/generated_data/em/quaia_zleq3_withlumcorr.csv' \
                --merger_rate $rate \
                --threading False \
                --n_workers 1 \
                --verbose True \
                --n_realizations 1 \
                --agn_zprior $agn_prior_lum \
                --lum_thresh $catalog_lum_thresh \
                --agn_zerror quaia \
                --mask_galactic_plane True \
                --assume_perfect_redshift False \
                --agn_zcut 3.0 \
                --zmin 0.000001 \
                --zmax 10 \
                --metadata_path /home/lucas/Documents/PhD/generated_data/jsons/metadata_test.json \
                --real_data True
        done
    done
done
