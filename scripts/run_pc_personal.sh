##!/bin/bash
#
## Take arguments
#while getopts 'p:v:s:' flag; do
#  case "${flag}" in
#    p) data_path="${OPTARG}" ;;
#    *) error "Unexpected option ${flag}" ;;
#  esac
#done
#
#specimens=(
#    "0503_E1" "0209_E1" "0520_E2" "0208_E3"
#    "0308_E3" "0503_E2" "0516_E3"
#    "0521_E1" "0523_E1" "0806_E3" "0806_E4" "0806_E6"
#    "0401_E3" "0123_E1" "0122_E1" "0518_E3" "0521_E3"
#    "0308_E2" "0401_E1" "0401_E2" "0502_E1" "0517_E2" "0806_E1" "0806_E2"
#    "0308_E4" "0403_E2" "0404_E2" "0516_E5" "0517_E4"
#    "0402_E1" "0402_E2" "0516_E4" "0517_E1" "0518_E2"
#    "0404_E1" "0515_E2" "0516_E1" "0518_E1" "0520_E5"
#
#    "0515_E3"
#)
#
#
#singularity exec -e -B //wsl.localhost/Ubuntu/home/txete/ht_morphogenesis:/repo/ -B $data_path:/data/ //wsl.localhost/Ubuntu/home/txete/ht_morphogenesis/containers/mapping_cluster.sif python3 /repo/meshes/run_surface_map_cluster.py -p /data/ -s $specimen -v $verbose
#singularity cache clean -f

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/txete/ht_morphogenesis/meshes/surface_map/INTEL/libs

specimens=(
    "0401_E3"
    "0401_E1"
    "0404_E2"
    "0517_E1"
    "0520_E1"
    "0520_E4"
    "0123_E1"
    "0308_E2"
    "0403_E2"
    "0402_E2"
    "0119_E1"
    "0515_E1"
    "0122_E1" "0518_E3" "0521_E3"
    "0401_E2" "0502_E1" "0517_E2" "0806_E1" "0806_E2"
    "0308_E4" "0516_E5" "0517_E4"
    "0402_E1" "0516_E4" "0518_E2"
)

printf "%s\n" "${specimens[@]}" | parallel -j 10 --progress --joblog joblog.txt python meshes/run_surface_map.py -s {} -p /home/txete/data/cluster -v 1
