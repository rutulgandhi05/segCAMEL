#!/bin/bash

#SBATCH -J Inference
#SBATCH -o inference_output.txt
#SBATCH -e inference_error.txt
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:00:00
#SBATCH -p gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mail-type ALL
#SBATCH --mail-user rutul.gandhi@uni-ulm.de

source venv/bin/activate
module load devel/cuda/12.8

export HERCULES_DATASET=$(ws_find hercules_dataset_complete)
export TMP_HERCULES_DATASET=$TMPDIR/segcamel/hercules_dataset
mkdir -p $TMP_HERCULES_DATASET
echo "[INFO] Copying dataset to $TMP_HERCULES_DATASET"
cp -r --verbose $HERCULES_DATASET/* $TMP_HERCULES_DATASET/
echo "[INFO] Dataset copied."

export PREPROCESS_OUTPUT_DIR=$TMPDIR/segcamel/processed_data
export TRAIN_CHECKPOINTS=$TMPDIR/segcamel/checkpoints
export INFERENCE_OUTPUT_DIR=$TMPDIR/segcamel/$(date +"%d%m%Y_%H%M")_inference_output
export SEGMENTATION_OUT_DIR=$TMPDIR/segcamel/$(date +"%d%m%Y_%H%M")_unsup_outputs
export PIPELINE_MODE="inference"

mkdir -p $TRAIN_CHECKPOINTS
mkdir -p $INFERENCE_OUTPUT_DIR
cp --verbose $HERCULES_PROCESSED/28082025_1348_segcamel_train_output/checkpoints/best_model.pth $TRAIN_CHECKPOINTS/

python -m scripts.preprocess
python -m scripts.segment_once

cp --verbose -r $INFERENCE_OUTPUT_DIR/ $HERCULES_PROCESSED/28082025_1348_segcamel_train_output/
cp --verbose -r $SEGMENTATION_OUT_DIR/ $HERCULES_PROCESSED/28082025_1348_segcamel_train_output/