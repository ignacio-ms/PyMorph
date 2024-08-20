#!/bin/bash

conda init
conda activate plant-seg

python membrane_segmentation/run_plantseg.py -v 1

conda deactivate
conda activate cellpose

python nuclei_segmentation/run_cellpose.py -m 'nuclei' -n True -e True -d 17 -v 1
conda deactivate

conda activate py310ml
python filtering/run_filter_tissue.py -t 'myocardium' -l 'Membrane' -v 1
python filtering/run_filter_tissue.py -t 'splanchnic' -l 'Membrane' -v 1

python filtering/run_filter_tissue.py -t 'myocardium' -l 'Nuclei' -v 1
python filtering/run_filter_tissue.py -t 'splanchnic' -l 'Nuclei' -v 1


python feature_extraction/run_extractor.py -l 'myocardium' -t 'Membrane' -v 1
python feature_extraction/run_extractor.py -l 'splanchnic' -t 'Membrane' -v 1

python feature_extraction/run_extractor.py -l 'myocardium' -t 'Nuclei' -v 1
python feature_extraction/run_extractor.py -l 'splanchnic' -t 'Nuclei' -v 1
