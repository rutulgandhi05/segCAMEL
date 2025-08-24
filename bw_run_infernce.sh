#!/bin/bash

#SBATCH -J Inference
#SBATCH -o inference_output.txt
#SBATCH -e inference_error.txt
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -p gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mail-type ALL
#SBATCH --mail-user rutul.gandhi@uni-ulm.de

source venv/bin/activate
module load devel/cuda/12.8


export HERCULES_DATASET=$(ws_find hercules_dataset)
export PREPROCESS_OUTPUT_DIR=$TMPDIR/segcamel/processed_data
export TRAIN_CHECKPOINTS=$TMPDIR/segcamel/checkpoints
export INFERENCE_OUTPUT_DIR=$TMPDIR/segcamel/inference_output
export PIPELINE_MODE="inference"
export SEGMENTATION_OUT_DIR=$TMPDIR/segcamel/unsup_outputs

mkdir -p $TRAIN_CHECKPOINTS
mkdir -p $INFERENCE_OUTPUT_DIR
cp --verbose $HERCULES_DATASET/23082025_0235_segcamel_train_with_vel_md1_ld1_sd1/checkpoints/best_model.pth $TRAIN_CHECKPOINTS/

python -m scripts.preprocess
python -m scripts.segmentation_runner

cp --verbose -r $SEGMENTATION_OUT_DIR/ $HERCULES_DATASET/23082025_0235_segcamel_train_with_vel_md1_ld1_sd1/