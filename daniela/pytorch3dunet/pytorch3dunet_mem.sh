#!/bin/bash

#SBATCH -J Pytorch_3dUNET
#SBATCH -p high
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16G
#SBATCH -o %N.%J.out # STDOUT
#SBATCH -e %N.%j.err # STDERR

set -e

ml CUDA/11.4.3
ml GLib/2.69.1-GCCcore-10.2.0
ml torchvision/0.9.1-GCCcore-10.2.0-PyTorch-1.10.0-CUDA-11.4.3

nvidia-smi
nvcc --version

ps -ef | grep slurm
uname -a >> /homedtic/dvarela/pytorch.txt

source ~/anaconda3/bin/activate "";
conda activate pytorch3dunet;
cd /homedtic/dvarela/pretrained/pytunet3D
predict3dunet --config my_test_config_mem.yml