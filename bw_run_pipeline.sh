#!/bin/bash

#SBATCH -J Pipeline
#SBATCH -o pipeline_output.txt
#SBATCH -e pipeline_error.txt
#SBATCH -c 32
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 30:00:00
#SBATCH -p gpu_a100_il
#SBATCH --gres=gpu:2
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
export PIPELINE_MODE="train"
export PREPROCESS_FOLDERS="library_01_Day, mountain_01_Day, parking_lot_01_Day, river_island_01_Day, sports_complex_01_Day, stream_01_Day"

mkdir -p $PREPROCESS_OUTPUT_DIR
mkdir -p $TRAIN_CHECKPOINTS

echo "[INFO] Starting preprocessing..."
python -m scripts.preprocess
echo "[INFO] Preprocessing finished."

echo "[INFO] Starting training..."
python -m scripts.train_segmentation

export HERCULES_PROCESSED=$(ws_find hercules_preprocessed)
export RESULT_DIR=$HERCULES_PROCESSED/$(date +"%d%m%Y_%H%M")_segcamel_train_output_epoch_50

echo "[INFO] Copying checkpoints to $RESULT_DIR"
mkdir -p $RESULT_DIR
cp -r --verbose $TRAIN_CHECKPOINTS/ $RESULT_DIR/    
echo "[INFO] Job completed successfully."