import os
import glob
import h5py
from tqdm import tqdm
import shutil

ROOT = "/home/lucas/Documents/PhD/mock_gws_agndist_46.5_ngw_100000_zmax_10_zcut_0.3_LVKvols"
TYPES = ["agn", "alt"]

output_dirs = glob.glob(f"{ROOT}/output_run_*")

for output_dir in output_dirs:

    print(f"\nProcessing {output_dir}")

    base_ps_dir = f"{output_dir}/posterior_samples"

    success = True

    for TYPE in TYPES:

        input_pattern = f"{base_ps_dir}/{TYPE}/*.h5"
        input_files = glob.glob(input_pattern)

        if not input_files:
            print(f"  No files found for {TYPE}")
            continue

        output_file = f"{base_ps_dir}/samples_{TYPE}.h5"

        print(f"  Writing {output_file}")

        try:
            with h5py.File(output_file, "w") as fout:

                for infile in tqdm(input_files):

                    gw_id = os.path.splitext(
                        os.path.basename(infile)
                    )[0]

                    with h5py.File(infile, "r") as fin:

                        gw_group = fout.create_group(gw_id)

                        for key in fin.keys():
                            fin.copy(key, gw_group)

        except Exception as e:
            print(f"  ERROR writing {output_file}: {e}")
            success = False
            break

    # --------------------------------------------------
    # Delete old folders only if everything succeeded
    # --------------------------------------------------

    if success:
        for TYPE in TYPES:

            old_dir = f"{base_ps_dir}/{TYPE}"

            if os.path.isdir(old_dir):
                print(f"  Removing {old_dir}")
                shutil.rmtree(old_dir)

    else:
        print("  Skipping deletion due to errors")