#!/bin/bash

source /home/imarcoss/mambaforge/etc/profile.d/conda.sh
conda activate py310ml

python filtering/run_smoothing.py
python feature_extraction/run_mesh.py

python feature_extraction/run_extractor.py -l 'myocardium' -t 'Nuclei' -v 1
python feature_extraction/run_extractor.py -l 'splanchnic' -t 'Nuclei' -v 1