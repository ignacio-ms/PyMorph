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
  "cell_division"
)

groups=("Gr1" "Gr2" "Gr3" "Gr4" "Gr5" "Gr6" "Gr7" "Gr8")


# Run analysis for membrane features
for feature in "${features[@]}"
do
  echo "Running analysis for feature: $feature"
  for group in "${groups[@]}"
  do
    python meshes/run_atlas.py -g $group -f $feature -l "Nuclei" -t "myocardium" -v 1
  done
  python meshes/utils/visualize_atlas.py -g $group -f $feature -l "Nuclei" -t "myocardium" -v 1
  python meshes/run_normalize_atlas.py -f $feature -l "Nuclei" -t "myocardium" -v 1
done
