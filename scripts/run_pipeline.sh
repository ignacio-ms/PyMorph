#!/bin/bash

source /home/imarcoss/mambaforge/etc/profile.d/conda.sh

#specimens=(
#  "0308_E2" "0401_E2" "0502_E1"
#  "0403_E2" "0517_E4" "0521_E3"
#  "0520_E5" "0520_E2" "0209_E2" "0516_E3"
#  "0806_E5" "0521_E4" "0516_E2"
#  "0523_E1" "0806_E3" "0806_E4"
#  "0806_E1" "0517_E1"
#  "0504_E1" "0806_E6" "0518_E3"
#  "0503_E1" "0209_E1"
#  "0503_E2"
#  "0521_E1"
#  "0401_E3"
#)

specimens=(
  "0308_E4" "0404_E2"
  "0516_E4"
  "0516_E1" "0518_E1" "0520_E1"
  "0517_E2" "0401_E1"
)


for specimen in "${specimens[@]}"
do
  echo "Processing specimen $specimen"

  conda activate plant-seg
  python membrane_segmentation/run_plantseg.py -v 1 -s $specimen

  conda deactivate
#  conda activate py310ml
#
#  python feature_extraction/run_extractor.py -s $specimen -l 'myocardium' -t 'Membrane' -v 1
#  python meshes/run_mesh_reconstruction.py -s $specimen  -t 'myocardium' -l 'Membrane' -v 1
#  python meshes/run_extractor_complex.py -s $specimen  -l 'Membrane' -t 'myocardium' -v 1 -m 1
#  python filtering/run_filter_tissue.py -s $specimen -t 'myocardium' -l 'Membrane' -v 1
done
