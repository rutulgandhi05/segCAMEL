#!/bin/bash

#SBATCH -J Train
#SBATCH -o train_output.txt
#SBATCH -e train_error.txt
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00 
#SBATCH -p gpu_a100_short
#SBATCH --gres=gpu:2
#SBATCH --mail-type ALL
#SBATCH --mail-user rutul.gandhi@uni-ulm.de

source venv/bin/activate
module load devel/cuda/12.8
python -m scripts.train


