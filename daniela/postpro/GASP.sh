#!/bin/bash

#SBATCH -J Pytorch_3dUNET_post
#SBATCH -p high
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G
#SBATCH -o %N.%J.out # STDOUT
#SBATCH -e %N.%j.err # STDERR

set -e


source ~/anaconda3/bin/activate "";
conda activate GASP;
cd /homedtic/dvarela/postpro
python run_gasp.py