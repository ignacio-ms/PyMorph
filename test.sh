#!/bin/bash

# Define source and destination directories
SOURCE_DIR="/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/LAB/Ignacio/ATLAS/myocardium/FeaturesNormalized/Nuclei"
DEST_DIR="/home/imarcoss/ht_morphogenesis/to_miguel/myocardium/FeaturesNormalized/Nuclei"

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Iterate over each subfolder in the source directory
find "$SOURCE_DIR" -mindepth 2 -type d -name "grid" | while read -r GRID_DIR; do
    # Get the relative path of the current grid directory
    RELATIVE_PATH="${GRID_DIR#$SOURCE_DIR/}"

    # Define the destination path for the current grid directory
    TARGET_PATH="$DEST_DIR/${RELATIVE_PATH%/grid}"

    # Create the target directory if it doesn't exist
    mkdir -p "$TARGET_PATH"

    # Copy all .png files from the grid directory to the target directory
    find "$GRID_DIR" -maxdepth 1 -type f -name "*.png" -exec cp {} "$TARGET_PATH/" \;

    echo "Copied .png files from $GRID_DIR to $TARGET_PATH"
done

echo "Copy operation completed."
