#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/txete/ht_morphogenesis/meshes/surface_map/INTEL/libs


printf "%s\n" "${specimens[@]}" | parallel -j 2 --progress --joblog joblog.txt python3 meshes/run_surface_map_cluster.py -s {} -p /home/txete/data/cluster/ -v 1


