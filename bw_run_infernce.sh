#!/bin/bash

#SBATCH -J Inference
#SBATCH -o logs/inference_output_ri.txt
#SBATCH -e logs/inference_error_ri.txt
#SBATCH -c 32
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 30:00:00
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
cp -r  $HERCULES_DATASET/* $TMP_HERCULES_DATASET/
ls $TMP_HERCULES_DATASET
echo "[INFO] Dataset copied."

export HERCULES_PROCESSED=$(ws_find hercules_preprocessed)
export PREPROCESS_OUTPUT_DIR=$TMPDIR/segcamel/processed_data
export TRAIN_CHECKPOINTS=$TMPDIR/segcamel/checkpoints
export PREPROCESS_FOLDERS="street_01_Day"
export FEAT_MODE="ri"  # "rvi", "rv", "none", etc.
export INFERENCE_OUTPUT_DIR=$TMPDIR/segcamel/$(date +"%d%m%Y_%H%M")_inference_output_$PREPROCESS_FOLDERS_$FEAT_MODE
export SEGMENTATION_OUT_DIR=$TMPDIR/segcamel/$(date +"%d%m%Y_%H%M")_unsup_outputs_$PREPROCESS_FOLDERS_$FEAT_MODE
export PIPELINE_MODE="inference"

mkdir -p $TRAIN_CHECKPOINTS
mkdir -p $INFERENCE_OUTPUT_DIR
cp --verbose $HERCULES_PROCESSED/11092025_1205_segcamel_train_output_epoch_50_$FEAT_MODE/checkpoints/best_model.pth $TRAIN_CHECKPOINTS/
export TRAIN_CHECKPOINT_PTH=$TRAIN_CHECKPOINTS/best_model.pth

python -m scripts.preprocess
python -m scripts.segment_once

cp -r $INFERENCE_OUTPUT_DIR/ $HERCULES_PROCESSED/11092025_1205_segcamel_train_output_epoch_50_$FEAT_MODE/
cp -r $SEGMENTATION_OUT_DIR/ $HERCULES_PROCESSED/11092025_1205_segcamel_train_output_epoch_50_$FEAT_MODE/

echo "[INFO] Job completed successfully."