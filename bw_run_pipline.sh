#!/bin/bash

#SBATCH -J Preprocess
#SBATCH -o preprocess_output.txt
#SBATCH -e preprocess_error.txt
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
export PREPROCESS_OUTPUT_DIR=$TMPDIR/segcamel/processed_data
export TRAIN_CHECKPOINT=$TMPDIR/segcamel/checkpoints/ckpt_her_m1dl1dsc1d.pth

python -m scripts.preprocess
python -m scripts.train

cp --verbose $TRAIN_CHECKPOINT $HERCULES_DATASET/checkpoints/