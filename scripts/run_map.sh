#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data_lab_MT/Ignacio/ht_morphogenesis/meshes/surface_map/INTEL/libs

specimens=(
    "0503_E1" "0209_E1" "0520_E2" "0208_E3"
    "0308_E3" "0503_E2" "0516_E3"
    "0521_E1" "0523_E1" "0806_E3" "0806_E4" "0806_E6"
    "0401_E3" "0123_E1" "0122_E1" "0518_E3" "0521_E3"
    "0308_E2" "0401_E1" "0401_E2" "0502_E1" "0517_E2" "0806_E1" "0806_E2"
    "0308_E4" "0403_E2" "0404_E2" "0516_E5" "0517_E4"
    "0402_E1" "0402_E2" "0516_E4" "0517_E1" "0518_E2"
    "0119_E1" "0404_E1" "0515_E2" "0516_E1" "0518_E1" "0520_E1" "0520_E5"
    "0515_E1" "0520_E4"
    "0515_E3"
)

printf "%s\n" "${specimens[@]}" | parallel -j 8 --progress --joblog joblog.txt python meshes/run_surface_map.py -s {} -v 1
