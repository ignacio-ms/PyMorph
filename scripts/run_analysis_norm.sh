#!/bin/bash

#while getopts 'f:' flag; do
#  case "${flag}" in
#    f) feature="${OPTARG}" ;;
#    *) error "Unexpected option ${flag}" ;;
#  esac
#done

source /home/imarcoss/mambaforge/etc/profile.d/conda.sh
conda activate py310ml

features=(
  "columnarity" #"sphericity" "perpendicularity" "MeshVolume"
)

# Run analysis for membrane features
for feature in "${features[@]}"
do
    python meshes/run_normalize_atlas.py -f $feature -l "Membrane" -t "myocardium" -v 1
done
