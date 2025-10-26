# segCAMEL

>## Description:
### Project: Unsupervised Segmentation of FMCW LiDAR Point-clouds via Features Distilled from 2D Images

>### Abstract
>"Unsupervised segmentation of outdoor LiDAR point clouds remains difficult because geometric cues alone provide limited semantics and large-scale 3D annotations are scarce. At the same time, frequencymodulated continuous-wave (FMCW) LiDAR introduces Doppler velocity signals, but their potential for segmentation has been largely overlooked. Prior research has explored unsupervised learning from LiDAR or fusion with images, yet most pipelines ignore FMCW’s unique motion cues and treat scans as static. In this work, we introduce a cross-modal approach that projects features from a frozen2D vision foundation model (vfm) onto camera-visible points, distills them into a transformer-based 3D backbone with a cosine regression loss, and clusters the resulting embeddings using a von Mises–Fisher (vMF) mixture. Our framework requires no manual labels, operates directly on LiDAR at inference, and can optionally incorporate perpoint radial velocity. By transferring semantic priors from images into 3D and probing the role of Doppler signals, this study highlights cross-modal distillation as a strong path toward label-free 3D segmentation and points to tighter integration of FMCW velocity as an important direction forward."


> ## Requiremnets
Python 3.12.1

Cuda 12.8

### Setup

1.  Clone git repo
```
git clone https://github.com/rutulgandhi05/segCAMEL.git
```
2. Create vitrual env
```
cd segCAMEL
python -m venv venv
pip install -r requirements.txt
```

> ## Run pipeline

1. Set enviornment variables as below:
```
# path to dataset root
export TMP_HERCULES_DATASET=path/to/hecules_dataset   

# path to store preprocessed data
export PREPROCESS_OUTPUT_DIR=path/to/preprocessed_data/ 

# path to store checkpoints 
export TRAIN_CHECKPOINTS=path/to/checkpoints               

# change to different feature input mode r = reflectivity, v = velocity, i = intesity
export FEAT_MODE="rvi"                                    

# folders to train
export PREPROCESS_FOLDERS="mountain_01_Day library_01_Day sports_complex_01_Day stream_01_Day"

```

2. Run preprocess
```
# from root of the repo run 
python -m scripts.preprocess
```

3. Run train
```
# using same enviornment variables run 
python -m scripts.train_segmentation
```

> ## Inference

1. Create config.json in repo root and copy paste below code into the file.
```
{
  "mode": "vmf",
  "K": 12,

  "inference_batch": 4,
  "inference_workers": 12,
  "inference_limit": 2000,

  # change these values for different settings  
  "feature_cfg": {
    "use_range": true,   "range_scale": 70.0,
    "use_height": true,  "height_scale": 2.5,
    "use_speed": true,   "speed_scale": 10.0,
    "speed_deadzone_per_m": 0.01,
    "speed_deadzone_min": 0.1,
    "speed_deadzone_max": 0.80,
    "use_speed_signed": true
  },

  "max_passes": 2,
  "sample_per_frame": 50000,
  "dist_edges": [0.0, 15.0, 30.0, 60.0, 120.0],
  "dist_ratios": [0.35, 0.30, 0.20, 0.15],

  "smooth_iters": 0,
  "neighbor_range": 1,
  "min_component": 100,
  "range_gate_m": 0.5,

  "posterior_tau": 0.15,
  "tau_edges": [0.0, 15.0, 30.0, 60.0],
  "tau_map":   [0.10, 0.12, 0.14, 0.16],

  "metrics": {
    "sample_n": 200000,
    "speed_tau_policy": "value",
    "speed_tau_list": [0.3, 0.5, 1.0]
  }
}
```

2. Set enviornment variables as below: 
```
# path to store inference outputs
export INFERENCE_OUTPUT_DIR=path/to/inference_out

# path to store segmentation outputs
export SEGMENTATION_OUT_DIR=path/to segmentation_out

# path to store preprocessed data for inference data
export PREPROCESS_OUTPUT_DIR=path/to/preprocessed_data/ 

# path to checkpoints from train output
export TRAIN_CHECKPOINTS=path/to/checkpoints   

# config path
export CONFIG_JSON=config.json

# match training 
export FEAT_MODE="rvi"  
```

3. Run preprocess
```
# from root of the repo run 
python -m scripts.preprocess
```

4. Run inference
```
# using same enviornment variables run 
python -m scripts.segment_once
```

> ## Download HeRCULES dataset
1. Request HeRCULES dataset
    1. Go to the dataset download form: https://sites.google.com/view/herculesdataset/download?authuser=0
    2. Fill the form and click submit
    3. Wait for email with the google drive links to dataset folders.

2. Setup dataset links
    1. Go to hercules\dataset_download\gdrive_extract.py
    2. Replace the links in the links dict under main function, with the links recieved via email.

3. Set enviornment variable
```
# path to store hercules_dataset
export TMP_HERCULES_DATASET=path/to/hercules_dataset
```
4. Run download
```
python -m hercules.dataset_download.gdrive_extract
```
> NOTE: If prompted, sign into google with email that was used to request the dataset.

## Evaluation
To evaluate and read metrices.csv generated after inference with different feature mode you can edit and run scripts/analyze_metrices.py

## Visualization
To viusalize segmentation results edit the path variable and run scripts/unsup_seg_viz.py.

Use following keybinds for visuliztion window

- N = next_frame
- P = prev_frame
-  ] = inc_ps
-  [ = dec_ps
-  R = reset_v
-  G = toggle_mode  labels <-> range
-  S = cycle_speed  off -> |v| -> signed -> fused
-  M = cycle_label_policy id <-> by_speed

τ controls 
- = = tau_inc
- \+ = tau_inc
- \- = tau_dec
- _ = tau_dec
- K = tau_inc         # backups
- J = tau_dec
- V = toggle_fade
- C = cycle_palette
- Q = quit_v