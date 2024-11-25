#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/txete/ht_morphogenesis/meshes/surface_map/INTEL/libs

specimens=(
    "0119_E1"
    "0521_E1"
    "0404_E1"
    "0523_E1"
    "0515_E2"
    "0806_E3"
    "0516_E1"
    "0806_E4"
    "0518_E1"
    "0520_E1"
    "0520_E5"
    "0806_E6"
    "0209_E1"
    "0208_E3"
    "0308_E3"
    "0503_E2"
    "0520_E1"
    "0520_E4"
    "0123_E1"
    "0515_E1"
    "0122_E1"
    "0517_E2"
    "0806_E2"
    "0516_E5"
    "0517_E4"
    "0402_E1"
    "0518_E2"
)

printf "%s\n" "${specimens[@]}" | parallel -j 8 --progress --joblog joblog.txt python meshes/run_surface_map_cluster -s {} -p /home/txete/data/cluster -v 1
