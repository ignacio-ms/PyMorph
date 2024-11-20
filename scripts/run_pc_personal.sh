#!/bin/bash

# Take arguments
while getopts 'p:v:s:' flag; do
  case "${flag}" in
    p) data_path="${OPTARG}" ;;
    v) verbose="${OPTARG}" ;;
    s) specimen="${OPTARG}" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

singularity exec -e -B //wsl.localhost/Ubuntu/home/txete/ht_morphogenesis:/repo/ -B $data_path:/data/ //wsl.localhost/Ubuntu/home/txete/ht_morphogenesis/containers/mapping_cluster.sif python3 /repo/meshes/run_surface_map_cluster.py -p /data/ -s $specimen -v $verbose
singularity cache clean -f

