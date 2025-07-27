#!/bin/bash

#SBATCH -J SegCAMEL
#SBATCH -o hercules_output.txt
#SBATCH -e hercules_error.txt
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00 
#SBATCH -p gpu_a100_short

#SBATCH --mem=64000
#SBATCH --mail-type ALL
#SBATCH --mail-user rutul.gandhi@uni-ulm.de

source venv/bin/activate
python -m hercules.dataset_download.gdrive_extract


