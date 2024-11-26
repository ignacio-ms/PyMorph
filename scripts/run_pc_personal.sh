#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/txete/ht_morphogenesis/meshes/surface_map/INTEL/libs

#"0119_E1"
#"0404_E1"
#"0516_E1"
#"0518_E1"
#"0520_E1"
#"0517_E2"
#"0516_E5"
#"0518_E2"

specimens=(
    "0520_E4"
    "0515_E1"
)

printf "%s\n" "${specimens[@]}" | parallel -j 2 --progress --joblog joblog.txt python3 meshes/run_surface_map_cluster.py -s {} -p /home/txete/data/cluster/ -v 1


