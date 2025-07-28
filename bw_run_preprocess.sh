#!/bin/bash

#SBATCH -J Preprocess
#SBATCH -o preprocess_output.txt
#SBATCH -e preprocess_error.txt
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:60:00 
#SBATCH -p gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --mem=64000
#SBATCH --mail-type ALL
#SBATCH --mail-user rutul.gandhi@uni-ulm.de

source venv/bin/activate
python -m scripts.preprocess


