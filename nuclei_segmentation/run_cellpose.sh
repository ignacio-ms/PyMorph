#!/bin/bash

# Default variables
data_path="heart_tube/data/"
project="MT"
accounting="LAB_MT"
job_name="Cellpose hyper-tunning crop"

# Take arguments
while getopts 'p:i:s:g:m:n:e:d:c:v:' flag; do
  case "${flag}" in
    p) data_path="${OPTARG}" ;;
    i) image_path="${OPTARG}" ;;
    s) specimen="${OPTARG}" ;;
    g) group="${OPTARG}" ;;
    m) model="${OPTARG}" ;;
    n) normalize="${OPTARG}" ;;
    e) equalize="${OPTARG}" ;;
    d) diameter="${OPTARG}" ;;
    c) channels="${OPTARG}" ;;
    v) verbose="${OPTARG}" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

conda activate py310ml
cd /home/imarcoss/heart_tube/

# Create list of models to test and run cellpose
models=("nuclei", "cyto3", "cyto2_cp3", "livecell_cp3", "tissuenet_cp3", "deepbacs_cp3")

for model in "${models[@]}"
do
    echo "python nuclei_segmentation/run_cellpose.py -p $data_path -i $image_path -m $model -v 1" | qsub -P $project -A $accounting -N $job_name -l h_vmem=128G -t 1-6
done
