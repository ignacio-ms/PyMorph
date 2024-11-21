#!/bin/bash

source /home/imarcoss/mambaforge/etc/profile.d/conda.sh
conda activate py310ml

#"0806_E5" "0504_E1" "0208_E2" "0521_E4" "0521_E2"
#"0516_E2" "0503_E1" "0208_E3"

specimens=(
    "0209_E1" "0520_E2"
    "0209_E2" "0308_E3" "0503_E2" "0516_E3"
    "0521_E1" "0523_E1" "0806_E3" "0806_E4" "0806_E6"
    "0401_E3" "0123_E1" "0122_E1" "0518_E3" "0521_E3"
    "0308_E2" "0401_E1" "0401_E2" "0502_E1" "0517_E2" "0806_E1" "0806_E2"
    "0308_E4" "0403_E2" "0404_E2" "0516_E5" "0517_E4"
    "0402_E1" "0402_E2" "0516_E4" "0517_E1" "0518_E2"
    "0119_E1" "0404_E1" "0515_E2" "0516_E1" "0518_E1" "0520_E1" "0520_E5"
    "0515_E1" "0520_E4"
    "0515_E3"
)

#for specimen in "${specimens[@]}"
#do
#  python feature_extraction/run_extractor.py -s $specimen -l 'myocardium' -t 'Nuclei' -v 1
#  python feature_extraction/run_extractor.py -s $specimen -l 'splanchnic' -t 'Nuclei' -v 1
#
#  python feature_extraction/run_extractor.py -s $specimen -l 'myocardium' -t 'Nuclei' -v 1 -n 1
#  python feature_extraction/run_extractor.py -s $specimen -l 'splanchnic' -t 'Nuclei' -v 1 -n 1
#
#  python meshes/run_mesh.py -s $specimen -t 'myocardium' -l 'Nuclei' -v 1
#  python meshes/run_mesh.py -s $specimen -t 'splanchnic' -l 'Nuclei' -v 1
#
##  python filtering/run_filter_tissue.py -s $specimen -t 'myocardium' -l 'Nuclei' -v 1
##  python filtering/run_filter_tissue.py -s $specimen -t 'splanchnic' -l 'Nuclei' -v 1
#
##  python meshes/run_extractor_complex.py -s $specimen -l 'Nuclei' -t 'myocardium' -v 1 -m 1
##  python meshes/run_extractor_complex.py -s $specimen -l 'Nuclei' -t 'splanchnic' -v 1 -m 1
#done

for specimen in "${specimens[@]}"
do
  python cell_division/run_cell_division.py -s $specimen -t 'myocardium' -v 1
  python cell_division/run_cell_division.py -s $specimen -t 'splanchnic' -v 1
done
