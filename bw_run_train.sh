#!/bin/bash

#SBATCH -J SegCAMEL
#SBATCH -o train_output.txt
#SBATCH -e train_error.txt
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:55:00 
#SBATCH -p gpu_a100_il
#SBATCH --gres=gpu:2
#SBATCH --mem=64000
#SBATCH --mail-type ALL
#SBATCH --mail-user rutul.gandhi@uni-ulm.de

source venv/bin/activate
python -m scripts.train


