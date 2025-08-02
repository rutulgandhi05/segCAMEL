#!/bin/bash

#SBATCH -J Preprocess
#SBATCH -o preprocess_output.txt
#SBATCH -e preprocess_error.txt
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -p gpu_h100_il
#SBATCH --gres=gpu:2
#SBATCH --mail-type ALL
#SBATCH --mail-user rutul.gandhi@uni-ulm.de

source venv/bin/activate
module load devel/cuda/12.8

export HERCULES_DATASET=$(ws_find hercules_dataset)

python -m scripts.preprocess

rsync -av $TMPDIR/processed_data $HERCULES_DATASET/processed_data-${SLURM_JOB_ID}/
tar -cvzf $HERCULES_DATASET/processed_data-${SLURM_JOB_ID}.tgz  $HERCULES_DATASET/processed_data-${SLURM_JOB_ID}/
