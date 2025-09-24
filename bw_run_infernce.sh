#!/bin/bash

#SBATCH -J Inference
#SBATCH -o logs/inference_output_rvi_%j.txt
#SBATCH -e logs/inference_error_rvi_%j.txt
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

HERCULES_DATASET=$(ws_find hercules_dataset_complete)
HERCULES_PROCESSED=$(ws_find hercules_preprocessed)
TS="$(date +'%Y%m%d_%H%M')"

MODEL="best_model.pth"
TRAIN_FOLDER_STAMP="22092025_0509_segcamel_train_output_epoch_50"
export FEAT_MODE="rvi"  # "rvi", "rv", "none", etc.
export PREPROCESS_FOLDERS="river_island_01_Day"

export TMP_ROOT="$TMPDIR/segcamel"
export TMP_HERCULES_DATASET="$TMP_ROOT/hercules_dataset"
export PREPROCESS_OUTPUT_DIR="$TMP_ROOT/processed_data"
export TRAIN_CHECKPOINTS="$TMP_ROOT/checkpoints"

mkdir -p "$TMP_HERCULES_DATASET" "$TRAIN_CHECKPOINTS"

echo "[INFO] Copying $PREPROCESS_FOLDERS to $TMP_HERCULES_DATASET"
cp -r  "${HERCULES_DATASET}/${PREPROCESS_FOLDERS}" "$TMP_HERCULES_DATASET/"
ls "$TMP_HERCULES_DATASET"
echo "[INFO] Dataset copied."

echo "[INFO] Copying checkpoint to $TRAIN_CHECKPOINTS"
cp --verbose "${HERCULES_PROCESSED}/${TRAIN_FOLDER_STAMP}_${FEAT_MODE}/checkpoints/${MODEL}" "$TRAIN_CHECKPOINTS/"
echo "[INFO] Checkpoint copied."
export TRAIN_CHECKPOINT_PTH="$TRAIN_CHECKPOINTS/${MODEL}"

export INFERENCE_OUTPUT_DIR="$TMP_ROOT/${TS}_inference_output_${PREPROCESS_FOLDERS}_${FEAT_MODE}"
export SEGMENTATION_OUT_DIR="$TMP_ROOT/${TS}_unsup_outputs_${PREPROCESS_FOLDERS}_${FEAT_MODE}"
mkdir -p "$INFERENCE_OUTPUT_DIR" "$SEGMENTATION_OUT_DIR"

rsync -a --progress "$HERCULES_PROCESSED/${TRAIN_FOLDER_STAMP}_${FEAT_MODE}/20250924_0137_inference_output_river_island_01_Day_rvi/" "$INFERENCE_OUTPUT_DIR/"

CFG="$TMP_ROOT/cfg.json"
cat > "$CFG" <<'JSON'
{
  "mode": "vmf",
  "K": 10,

  "inference_batch": 4,
  "inference_workers": 12,
  "inference_limit": 2000,

  "feature_cfg": {
    "use_range": true,   "range_scale": 70.0,
    "use_height": true,  "height_scale": 2.5,
    "use_speed": true,   "speed_scale": 10.0,
    "speed_deadzone_per_m": 0.02,
    "speed_deadzone_min": 0.18,
    "speed_deadzone_max": 0.80,
    "use_speed_signed": false
  },

  "max_passes": 3,
  "sample_per_frame": 50000,
  "dist_edges": [0.0, 15.0, 30.0, 60.0, 120.0],
  "dist_ratios": [0.35, 0.30, 0.20, 0.15],

  "smooth_iters": 1,
  "neighbor_range": 1,
  "min_component": 80,
  "range_gate_m": 0.8,

  "posterior_tau": 0.13,
  "tau_edges": [0.0, 15.0, 30.0, 60.0],
  "tau_map":   [0.12, 0.14, 0.16, 0.18],

  "metrics": {
    "sample_n": 200000,
    "speed_tau_policy": "value",
    "speed_tau_list": [0.3, 0.5, 1.0]
  }
}
JSON

export CONFIG_JSON="$CFG"

# python -m scripts.preprocess
python -m scripts.segment_once

#cp -r "$INFERENCE_OUTPUT_DIR/" "$HERCULES_PROCESSED/${TRAIN_FOLDER_STAMP}_${FEAT_MODE}/"
cp -r "$SEGMENTATION_OUT_DIR/" "$HERCULES_PROCESSED/${TRAIN_FOLDER_STAMP}_${FEAT_MODE}/"
cp -r "$CFG" "$HERCULES_PROCESSED/${TRAIN_FOLDER_STAMP}_${FEAT_MODE}/"
echo "[INFO] Job completed successfully."