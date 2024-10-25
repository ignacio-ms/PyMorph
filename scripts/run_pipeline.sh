#!/bin/bash

source /home/imarcoss/mambaforge/etc/profile.d/conda.sh

#conda activate plant-seg
#python membrane_segmentation/run_plantseg.py -v 1
#conda deactivate

#conda activate cellpose
#python nuclei_segmentation/run_cellpose.py -m 'nuclei' -v 1
#conda deactivate

conda activate py310ml
python feature_extraction/run_extractor.py -l 'myocardium' -t 'Nuclei' -v 1
python feature_extraction/run_extractor.py -l 'splanchnic' -t 'Nuclei' -v 1

#python feature_extraction/run_extractor.py -l 'myocardium' -t 'Membrane' -v 1
#python feature_extraction/run_extractor.py -l 'splanchnic' -t 'Membrane' -v 1

python meshes/run_mesh.py -t 'myocardium' -l 'Nuclei' -v 1
python meshes/run_mesh.py -t 'splanchnic' -l 'Nuclei' -v 1

#python meshes/run_mesh.py -t 'myocardium' -l 'Membrane' -v 1
#python meshes/run_mesh.py -t 'splanchnic' -l 'Membrane' -v 1

#python meshes/run_extractor_complex.py -l 'Membrane' -t 'myocardium' -v 1 -m 1
#python meshes/run_extractor_complex.py -l 'Membrane' -t 'splanchnic' -v 1 -m 1

#python meshes/run_extractor_complex.py -l 'Nuclei' -t 'myocardium' -v 1 -m 1
#python meshes/run_extractor_complex.py -l 'Nuclei' -t 'splanchnic' -v 1 -m 1
