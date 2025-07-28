#!/bin/bash

#SBATCH -J HerculesDownload2
#SBATCH -o hercules_output2.txt
#SBATCH -e hercules_error2.txt
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
python -m hercules.dataset_download.gdrive_extract


