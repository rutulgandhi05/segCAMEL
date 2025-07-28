#!/bin/bash

#SBATCH -J Preprocess
#SBATCH -o preprocess_output.txt
#SBATCH -e preprocess_error.txt
#SBATCH -c 8
#SBATCH -N 4    
#SBATCH -n 2
#SBATCH -t 00:30:00 
#SBATCH -p gpu_a100_short
#SBATCH --gres=gpu:4
#SBATCH --mem=94000
#SBATCH --mail-type ALL
#SBATCH --mail-user rutul.gandhi@uni-ulm.de

source venv/bin/activate
python -m scripts.preprocess


