#!/bin/bash

#while getopts 'f:' flag; do
#  case "${flag}" in
#    f) feature="${OPTARG}" ;;
#    *) error "Unexpected option ${flag}" ;;
#  esac
#done

source /home/imarcoss/mambaforge/etc/profile.d/conda.sh
conda activate py310ml


membrane_features=(
  "MeshVolume" "columnarity" "perpendicularity" "sphericity"
  "solidity" "Elongation" "Flatness"
  "LeastAxisLength" "MajorAxisLength" "Maximum2DDiameterColumn"
  "Maximum2DDiameterRow" "Maximum2DDiameterSlice" "Maximum3DDiameter"
  "MinorAxisLength" "SurfaceArea" "axis_major_length" "axis_minor_length"
  "SurfaceVolumeRatio" "VoxelVolume"
)

#  "sphericities" "Maximum3DDiameter" "MinorAxisLength" "Sphericity"
#  "Elongation" "Flatness" "LeastAxisLength" "MajorAxisLength"
#  "Maximum2DDiameterColumn" "Maximum2DDiameterRow" "Maximum2DDiameterSlice"
nuclei_features=(
  "VoxelVolume" "10Percentile" "90Percentile" "Energy"
  "Entropy" "InterquartileRange" "Kurtosis" "Maximum" "MeanAbsoluteDeviation"
  "Mean" "Median" "Minimum" "Range" "RobustMeanAbsoluteDeviation"
  "RootMeanSquared" "Skewness" "TotalEnergy" "Uniformity" "Variance"
  "perpendicularity" "sphericity" "columnarity" "MeshVolume" # "cell_division"
)


groups=("Gr1" "Gr2" "Gr3" "Gr4" "Gr5" "Gr6" "Gr7" "Gr8" "Gr9")


# Run analysis for membrane features
for feature in "${membrane_features[@]}"
do
  echo "Running analysis for feature: $feature"
  for group in "${groups[@]}"
  do
    python meshes/run_atlas.py -g $group -f $feature -l "Membrane" -t "myocardium" -v 1
  done
  python meshes/utils/visualize_analysis.py -f $feature -l "Membrane" -t "myocardium" -v 1
  python meshes/run_normalize_atlas.py -f $feature -l "Membrane" -t "myocardium" -v 1
done

# Run analysis for nuclei features
for feature in "${nuclei_features[@]}"
do
  echo "Running analysis for feature: $feature"
  for group in "${groups[@]}"
  do
    python meshes/run_atlas.py -g $group -f $feature -l "Nuclei" -t "myocardium" -v 1
  done
  python meshes/utils/visualize_analysis.py -f $feature -l "Nuclei" -t "myocardium" -v 1
  python meshes/run_normalize_atlas.py -f $feature -l "Nuclei" -t "myocardium" -v 1
done
