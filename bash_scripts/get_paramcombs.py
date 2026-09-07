import json
import numpy as np

# -----------------------------------------
# Full parameter grid
# -----------------------------------------

zcuts = np.linspace(0, 2, 50)
agn_errs = np.geomspace(0.001, 1, 50)[20:][::-1]

expected = {
    (round(float(agn_err), 12), round(float(zcut), 12))
    for agn_err in agn_errs
    for zcut in zcuts
}

# -----------------------------------------
# Completed runs from JSON
# -----------------------------------------

with open("A_final_46p5_0p3_grid.json", "r") as f:
    configs = json.load(f)

completed = {
    (
        round(float(cfg['config']["AGN_ZERROR"]), 12),
        round(float(cfg['config']["AGN_ZCUT"]), 12),
    )
    for cfg in configs
}

# -----------------------------------------
# Missing combinations
# -----------------------------------------

missing = sorted(expected - completed)

print(f"{len(missing)} missing runs\n")

for agn_err, zcut in missing:
    print(f"agn_zerror={agn_err}, agn_zcut={zcut}")