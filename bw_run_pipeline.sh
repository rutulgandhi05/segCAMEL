#!/bin/bash

#SBATCH -J Pipeline
#SBATCH -o logs/pipeline_output_rvi.txt
#SBATCH -e logs/pipeline_error_rvi.txt
#SBATCH -c 32
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -p gpu_a100_il
#SBATCH --gres=gpu:2
#SBATCH --mail-type ALL
#SBATCH --mail-user rutul.gandhi@uni-ulm.de

source venv/bin/activate
module load devel/cuda/12.8

export HERCULES_DATASET=$(ws_find hercules_dataset_complete)
export TMP_HERCULES_DATASET=$TMPDIR/segcamel/hercules_dataset
mkdir -p $TMP_HERCULES_DATASET

export PREPROCESS_FOLDERS="mountain_01_Day library_01_Day sports_complex_01_Day stream_01_Day"
for dir in $PREPROCESS_FOLDERS; do
  echo "[INFO] Copying $dir to $TMP_HERCULES_DATASET"
  cp -r $HERCULES_DATASET/"$dir" $TMP_HERCULES_DATASET/
done
ls $TMP_HERCULES_DATASET
echo "[INFO] Dataset copied."

export PREPROCESS_OUTPUT_DIR=$TMPDIR/segcamel/processed_data
export TRAIN_CHECKPOINTS=$TMPDIR/segcamel/checkpoints

export FEAT_MODE="rvi"  # "rvi", "rv", "none", etc.
export HERCULES_PROCESSED=$(ws_find hercules_preprocessed)
export RESULT_DIR=$HERCULES_PROCESSED/$(date +"%d%m%Y_%H%M")_segcamel_train_output_epoch_50_$FEAT_MODE

mkdir -p $PREPROCESS_OUTPUT_DIR
mkdir -p $TRAIN_CHECKPOINTS
mkdir -p $RESULT_DIR

echo "[INFO] Starting preprocessing..."
python -m scripts.preprocess
echo "[INFO] Preprocessing finished."

echo "[INFO] Starting training..."
python -m scripts.train_segmentation

echo "[INFO] Copying checkpoints to $RESULT_DIR"
cp -r $TRAIN_CHECKPOINTS/ $RESULT_DIR/    
echo "[INFO] Job completed successfully."