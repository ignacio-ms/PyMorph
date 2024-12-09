#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/txete/ht_morphogenesis/meshes/surface_map/INTEL/libs

specimens=(
    "0119_E1" "0516_E1" "0516_E5"
    "0517_E2" "0518_E1" "0518_E2"
    "0520_E1"
)

printf "%s\n" "${specimens[@]}" | parallel -j 2 --progress --joblog joblog.txt python3 meshes/run_surface_map_cluster.py -s {} -p /home/txete/data/cluster/ -v 1


