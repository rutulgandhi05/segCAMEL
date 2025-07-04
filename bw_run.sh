#!/bin/bash

#SBATCH -J thechosenone
#SBATCH -o output.txt
#SBATCH -e error.txt
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00 
#SBATCH -p dev_gpu_a100_il
#SBATCH --gres=gpu:2
#SBATCH --mem=64000
#SBATCH --mail-type ALL
#SBATCH --mail-user rutul.gandhi@uni-ulm.de



python -m scripts.feature_extractor


