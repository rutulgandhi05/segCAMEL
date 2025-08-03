#!/bin/bash

#SBATCH -J Train
#SBATCH -o train_output.txt
#SBATCH -e train_error.txt
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:30:00
#SBATCH -p gpu_h100_il
#SBATCH --gres=gpu:2
#SBATCH --mail-type ALL
#SBATCH --mail-user rutul.gandhi@uni-ulm.de

source venv/bin/activate
module load devel/cuda/12.8


export HERCULES_DATASET=$(ws_find hercules_dataset)
rsync -av $HERCULES_DATASET/processed_data/ $TMPDIR/processed_data/
CUDA_LAUNCH_BLOCKING=1 python -m scripts.train
rsync -av $TMPDIR/checkpoints $HERCULES_DATASET/checkpoints/
