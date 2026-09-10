#!/bin/bash

for i in {1..200}
do
    ~/Documents/PhD/.venv/bin/python darksirenpop/mock/make_mock_gws.py \
    --run_id $i \
    --agndist 46.5 \
    --ngw 100000 \
    --zcut 0.3
done
