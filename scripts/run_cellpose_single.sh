#!/bin/bash


# Take arguments
while getopts 'p:v:e:' flag; do
  case "${flag}" in
    p) data_path="${OPTARG}" ;;
    v) verbose="${OPTARG}" ;;
    e) specimen="${OPTARG}" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

# Loop over the specimens based on the array job index
singularity exec -e -B /data_lab_MT/Ignacio/ht_morphogenesis:/app/ -B $data_path:/data/ --nv /data_lab_MT/Ignacio/ht_morphogenesis_latest.sif python /app/nuclei_segmentation/run_cellpose.py -p /data/ -s $specimen -m 'nuclei' -d 17 -v $verbose
#singularity clean cache all
singularity cache clean -f