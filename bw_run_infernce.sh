#!/bin/bash

#SBATCH -J Inference
#SBATCH -o inference_output.txt
#SBATCH -e inference_error.txt
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -p gpu_a100_short
#SBATCH --gres=gpu:2
#SBATCH --mail-type ALL
#SBATCH --mail-user rutul.gandhi@uni-ulm.de

source venv/bin/activate
module load devel/cuda/12.8


export HERCULES_DATASET=$(ws_find hercules_dataset)
export PREPROCESS_OUTPUT_DIR=$TMPDIR/segcamel/processed_data
export TRAIN_CHECKPOINTS=$TMPDIR/segcamel/checkpoints
export INFERENCE_OUTPUT_DIR=$TMPDIR/segcamel/inference_output

mkdir -p $TRAIN_CHECKPOINTS
mkdir -p $INFERENCE_OUTPUT_DIR
cp --verbose $HERCULES_DATASET/22082025_0443_segcamel_train_with_vel_md1_ld1_sd1/checkpoints/best_model.pth $TRAIN_CHECKPOINTS/
 
python -m scripts.preprocess
python -m scripts.inference

mkdir -p $HERCULES_DATASET/22082025_0443_segcamel_train_with_vel_md1_ld1_sd1/inference_output_bm_sd3
cp -r --verbose $INFERENCE_OUTPUT_DIR/ $HERCULES_DATASET/22082025_0443_segcamel_train_with_vel_md1_ld1_sd1/inference_output_bm_sd3