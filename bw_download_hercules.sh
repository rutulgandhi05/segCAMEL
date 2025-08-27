#!/bin/bash

#SBATCH -J HerculesDownload
#SBATCH -o hercules_output.txt
#SBATCH -e hercules_error.txt
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 15:00:00 
#SBATCH -p cpu_il
#SBATCH --mail-type ALL
#SBATCH --mail-user rutul.gandhi@uni-ulm.de

export HERCULES_DATASET=$(ws_find hercules_dataset_complete)
export TMP_HERCULES_DATASET=$TMPDIR/hercules_dataset
source venv/bin/activate
module load devel/cuda/12.8
python -m hercules.dataset_download.gdrive_extract

cp -r --verbose $TMP_HERCULES_DATASET/* $HERCULES_DATASET/
