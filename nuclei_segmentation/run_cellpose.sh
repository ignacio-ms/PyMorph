#!/bin/bash

# Default variables
#data_path="heart_tube/data/"
#project="MT"
#accounting="LAB_MT"
#job_name="Cellpose hyper-tunning crop"
index=$((SGE_TASK_ID - 1))

# Take arguments
while getopts 'p:i:s:g:m:n:e:d:c:v:' flag; do
  case "${flag}" in
    p) data_path="${OPTARG}" ;;
    i) image_path="${OPTARG}" ;;
    v) verbose="${OPTARG}" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

conda activate py310ml
cd /home/imarcoss/heart_tube/

# Create list of models to test and run cellpose
models=("nuclei", "cyto3", "cyto2_cp3", "livecell_cp3", "tissuenet_cp3", "deepbacs_cp3")
model=${models[$index]}

# Run cellpose
python nuclei_segmentation/run_cellpose.py -p $data_path -i $image_path -m $model -v $verbose

# Script call:
# echo "bash nuclei_segmentation/run_cellpose.sh -p 'heart_tube/data/' -i 'RawImages/Nuclei/20190806_E6_DAPI_decon_0.5.nii.gz' -v 1 | qsub -P MT -A "LAB_MT" -t 1-6 -l h_vmem=128G -N Cellpose hyper-tunning crop -o logs/ -e errors/
