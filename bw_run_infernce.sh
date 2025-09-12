#!/bin/bash

#SBATCH -J Inference
#SBATCH -o inference_output.txt
#SBATCH -e inference_error.txt
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 20:00:00
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

export HERCULES_PROCESSED=$(ws_find hercules_preprocessed)
export PREPROCESS_OUTPUT_DIR=$TMPDIR/segcamel/processed_data
export TRAIN_CHECKPOINTS=$TMPDIR/segcamel/checkpoints
export INFERENCE_OUTPUT_DIR=$TMPDIR/segcamel/$(date +"%d%m%Y_%H%M")_inference_output_river_island_01_Day
export SEGMENTATION_OUT_DIR=$TMPDIR/segcamel/$(date +"%d%m%Y_%H%M")_unsup_outputs_river_island_01_Day
export PIPELINE_MODE="inference"

export PREPROCESS_FOLDERS="street_01_Day"

mkdir -p $TRAIN_CHECKPOINTS
mkdir -p $INFERENCE_OUTPUT_DIR
cp --verbose $HERCULES_PROCESSED/09092025_0900_segcamel_train_output_epoch_50/checkpoints/best_model.pth $TRAIN_CHECKPOINTS/

python -m scripts.preprocess
python -m scripts.segment_once

cp --verbose -r $INFERENCE_OUTPUT_DIR/ $HERCULES_PROCESSED/09092025_0900_segcamel_train_output_epoch_50/
cp --verbose -r $SEGMENTATION_OUT_DIR/ $HERCULES_PROCESSED/09092025_0900_segcamel_train_output_epoch_50/