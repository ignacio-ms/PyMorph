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

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data_lab_MT/Ignacio/ht_morphogenesis/meshes/surface_map/INTEL/libs
# Loop over the specimens based on the array job index
singularity exec -e -B /data_lab_MT/Ignacio/ht_morphogenesis:/app/ -B $data_path:/data/ --nv /data_lab_MT/Ignacio/ht_morphogenesis_latest.sif python /app/meshes/run_surface_map.py -p /data/ -s $specimen -v $verbose
#singularity clean cache all
singularity cache clean -f

