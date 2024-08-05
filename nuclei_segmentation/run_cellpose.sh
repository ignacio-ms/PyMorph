#!/bin/bash

# Default variables
#data_path="heart_tube/data/"
#project="MT"
#accounting="LAB_MT"
#job_name="Cellpose hyper-tunning crop"
index=$((SGE_TASK_ID - 1))

# Take arguments
while getopts 'p:i:v:' flag; do
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
models=("nuclei" "cyto3" "cyto2_cp3" "livecell_cp3" "tissuenet_cp3" "deepbacs_cp3")
model=${models[$index]}

# Run cellpose
python ht_morphogenesis/nuclei_segmentation/run_cellpose.py -p $data_path -i $image_path -m $model -v $verbose
