#!/bin/bash

#index=$((SGE_TASK_ID - 1))

# Take arguments
while getopts 'p:i:v:' flag; do
  case "${flag}" in
    p) data_path="${OPTARG}" ;;
    i) image_path="${OPTARG}" ;;
#    e) embryo="${OPTARG}" ;;
    v) verbose="${OPTARG}" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

# Create list of models to test and run cellpose
#models=("nuclei" "cyto3" "cyto2_cp3" "livecell_cp3" "tissuenet_cp3" "deepbacs_cp3")
#model=${models[$index]}

#-----Manual-----#
#for model in "${models[@]}"
#do
#    # Run cellpose
#    python ht_morphogenesis/nuclei_segmentation/run_cellpose.py -p $data_path -i $image_path -m $model -v $verbose
#done

#-----ArrayJob-----#
## Run cellpose
#python ht_morphogenesis/nuclei_segmentation/run_cellpose.py -p $data_path -i $image_path -m $model -v $verbose

#-----Singularity-----#
singularity exec -e -B /data_lab_MT/Ignacio/ht_morphogenesis:/app/ -B $data_path:/data/ --nv /data_lab_MT/Ignacio/ht_morphogenesis_latest.sif python /app/nuclei_segmentation/run_cellpose.py -p /data/ -i $image_path -m 'nuclei' -n True -e True -v $verbose


#-----Usage example-----#
# echo "bash /data_lab_MT/Ignacio/ht_morphogenesis/nuclei_segmentation/run_cellpose.sh -p '/data_lab_MT/Ignacio/data/data/' -i 'Auxiliary/CellposeClusterTest/RawImages/Nuclei/20190806_E6_DAPI_decon_0.5.nii.gz' -v 1" | qsub -P MT -l h_vmem=120G -N "Cellpose_singularity" -e err/ -o out/
