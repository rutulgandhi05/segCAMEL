#!/bin/bash

#SBATCH -J Pipeline
#SBATCH -o pipeline_output.txt
#SBATCH -e pipeline_error.txt
#SBATCH -c 32
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -p gpu_a100_il
#SBATCH --gres=gpu:2
#SBATCH --mail-type ALL
#SBATCH --mail-user rutul.gandhi@uni-ulm.de

source venv/bin/activate
module load devel/cuda/12.8

export HERCULES_DATASET=$(ws_find hercules_dataset)
export PREPROCESS_OUTPUT_DIR=$TMPDIR/segcamel/processed_data
export TRAIN_CHECKPOINTS=$TMPDIR/segcamel/checkpoints

mkdir -p $PREPROCESS_OUTPUT_DIR
mkdir -p $TRAIN_CHECKPOINTS

echo "[INFO] Starting preprocessing..."
python -m scripts.preprocess
echo "[INFO] Preprocessing finished."

echo "[INFO] Starting training..."
python -m scripts.train_segmentation


export RESULT_DIR=$HERCULES_DATASET/$(date +"%d%m%Y_%H%M")_segcamel_train_with_vel_md1_ld1_sd1

echo "[INFO] Copying checkpoints to $RESULT_DIR"
mkdir -p $RESULT_DIR
cp -r --verbose $TRAIN_CHECKPOINTS/ $RESULT_DIR/    
echo "[INFO] Job completed successfully."