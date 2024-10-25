#!/bin/bash

source /home/imarcoss/mambaforge/etc/profile.d/conda.sh

#conda activate py310ml
#python filtering/run_filter_tissue.py -t 'myocardium' -l 'Membrane' -v 1
#python filtering/run_filter_tissue.py -t 'splanchnic' -l 'Membrane' -v 1

specimens=("0504_E1" "0122_E1" "0123_E1" "0402_E2")

for specimen in "${specimens[@]}"
do
  conda activate plant-seg
  python membrane_segmentation/run_plantseg.py -v 1 -s $specimen
  conda deactivate

  conda activate py310ml
  python feature_extraction/run_extractor.py -l 'myocardium' -t 'Membrane' -v 1 -s $specimen
  python feature_extraction/run_extractor.py -l 'splanchnic' -t 'Membrane' -v 1 -s $specimen

  python meshes/run_mesh.py -t 'myocardium' -l 'Membrane' -v 1 -s $specimen
  python meshes/run_mesh.py -t 'splanchnic' -l 'Membrane' -v 1 -s $specimen

  python meshes/run_extractor_complex.py -l 'Membrane' -t 'myocardium' -v 1 -m 1 -s $specimen
#  python meshes/run_extractor_complex.py -l 'splanchnic' -t 'myocardium' -v 1 -m 1 -s $specimen
  conda deactivate
done


conda activate py310ml
python feature_extraction/run_extractor.py -l 'myocardium' -t 'Nuclei' -v 1 -s $specimen
python feature_extraction/run_extractor.py -l 'splanchnic' -t 'Nuclei' -v 1 -s $specimen

python meshes/run_mesh.py -t 'myocardium' -l 'Nuclei' -v 1 -s $specimen
python meshes/run_mesh.py -t 'splanchnic' -l 'Nuclei' -v 1 -s $specimen

conda deactivate
