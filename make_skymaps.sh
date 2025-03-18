#!/bin/bash

### 100% GPT-powered bash script ###

# Function to display the usage
usage() {
    echo "Usage: $0 --indir <input_directory> --outdir <output_directory> --jobs <num_jobs> [additional arguments for ligo-skymap-from-samples]"
    exit 1
}

# Initialize variables
JOBS=1  # Default number of jobs if not provided
EXTRA_ARGS=()  # To collect additional arguments

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --indir)
            INDIR="$2"
            shift 2
            ;;
        --outdir)
            OUTDIR="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")  # Collect any additional arguments
            shift
            ;;
    esac
done

# Check if both input-dir and outdir were provided
if [[ -z "$INDIR" || -z "$OUTDIR" ]]; then
    usage
fi

# Check if the input directory exists
if [[ ! -d "$INDIR" ]]; then
    echo "Error: Directory '$INDIR' does not exist."
    exit 1
fi

# Check if the output directory exists, and create it if it does not
if [[ ! -d "$OUTDIR" ]]; then
    echo "Output directory '$OUTDIR' does not exist. Creating it..."
    mkdir -p "$OUTDIR"
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to create output directory '$OUTDIR'."
        exit 1
    fi
fi

echo "Setting environment variable OMP_NUM_THREADS=1"
export OMP_NUM_THREADS=1

# Counter for naming output files
counter=0

# Loop through all .h5 files in the input directory
for file in "$INDIR"/*.h5; do
    if [[ -f "$file" ]]; then
        # Generate the fits output name with an incrementing number
        FITS_NAME="skymap_${counter}.fits"

        echo "Processing: $file"
        echo "Output: $OUTDIR/$FITS_NAME"

        # Run the command with dynamic --fitsoutname and adjusted --outdir
        ligo-skymap-from-samples "$file" --fitsoutname "$FITS_NAME" --outdir "$OUTDIR" --jobs "$JOBS" --disable-distance-map "${EXTRA_ARGS[@]}"

        # Increment counter for the next file
        ((counter++))
    fi
done

# Remove skypost.obj if it exists in the output directory
SKYPOST_FILE="$OUTDIR/skypost.obj"
if [[ -f "$SKYPOST_FILE" ]]; then
    echo "Removing $SKYPOST_FILE"
    rm "$SKYPOST_FILE"
fi

echo 'Done.'

# Move to the output directory's parent
cd "$OUTDIR"
cd ..

# Create 'stats' directory if it does not exist
if [[ ! -d "stats" ]]; then
    echo "Output directory 'stats' does not exist. Creating it..."
    mkdir -p "stats"
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to create output directory 'stats'."
        exit 1
    fi
fi

cd stats

# Get the last part of the input directory name
outfilename=$(basename "$INDIR")

# Run ligo-skymap-stats with --jobs
ligo-skymap-stats "$OUTDIR"/*.fits --output "${outfilename}_stats.txt" --contour 68 90 99.9 --jobs "$JOBS"
