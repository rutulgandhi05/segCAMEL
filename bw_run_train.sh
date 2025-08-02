#!/bin/bash

#SBATCH -J Train
#SBATCH -o train_output.txt
#SBATCH -e train_error.txt
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -p gpu_a100_il
#SBATCH --gres=gpu:2
#SBATCH --mail-type ALL
#SBATCH --mail-user rutul.gandhi@uni-ulm.de

export HERCULES_DATASET=$(ws_find hercules_dataset)
source venv/bin/activate
module load devel/cuda/12.8
CUDA_LAUNCH_BLOCKING=1 python -m scripts.train


