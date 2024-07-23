#!/bin/bash

#SBATCH -J conversion
#SBATCH -p high
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH -o %N.%J.out # STDOUT
#SBATCH -e %N.%j.err # STDERR

set -e


source ~/anaconda3/bin/activate "";
conda activate cellpose;
cd /homedtic/dvarela/pretrained/pytunet3D
python convert_nii_h5.py