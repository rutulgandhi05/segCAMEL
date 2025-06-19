# scripts/build_dataset.py

import torch
import random
import logging
import numpy as np


from PIL import Image
from pathlib import Path
from scripts.dataset import load_hercules_dataset_folder
from scripts.project_2d_to_3d import LabelProjector, project_points_to_dino_patches
from scripts.twod_feature_extractor import DINOv2FeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

IGNORE_FOLDERS = ["Mountain_01_Day", "Parking_lot_02_Day"]
DATA_ROOT_FOLDER  = Path("data/hercules/")
RAW_FOLDERS   = iter([p for p in DATA_ROOT_FOLDER.iterdir() if p.is_dir() and p.name in IGNORE_FOLDERS])
RAW_FOLDER    = next(RAW_FOLDERS, None)
GRID_SIZE    = 0.05
VAL_SPLIT    = 0.2
TIME_THRESH  = 0.05 

OUT_FOLDER   = DATA_ROOT_FOLDER / "processed"
OUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUT_TRAIN    = OUT_FOLDER / "train"
OUT_TRAIN.mkdir(parents=True, exist_ok=True)
OUT_VAL      = OUT_FOLDER / "val"
OUT_VAL.mkdir(parents=True, exist_ok=True)

def get_closest_images(lidar_ts, camera_ts, thresh):
    delta = np.abs(camera_ts - lidar_ts)
    min_delta = np.min(delta)
    close_idxs = np.where(delta == min_delta)[0]
    if min_delta > thresh:
        return []
    return close_idxs.tolist()

def extract_timestamp(p: Path):
    return float(p.stem) * 1e-9


point_clouds = []
left_images = []
intrinsic = None
extrinsic = None

while RAW_FOLDER and RAW_FOLDER.is_dir():
    
    logging.info(f"Processing folder: {RAW_FOLDER}")

    data = load_hercules_dataset_folder(RAW_FOLDER, return_all_fields=True)
    if data is None:
        logging.error(f"Failed to load data from {RAW_FOLDER}. Check the folder structure and files.")

    point_clouds.extend(data["point_clouds"])               # list of (xyz, fields) tuples
    left_images.extend(data["stereo_left_images"])          # list of Path
    intrinsic     = data["stereo_left_intrinsics"] if intrinsic != data["stereo_left_intrinsics"] else intrinsic    # (3×3)
    extrinsic     = data["lidar_to_stereo_left_extrinsic"] if extrinsic != data["lidar_to_stereo_left_extrinsic"] else extrinsic  # (4×4)

logging.info(f"Loaded {len(point_clouds)} point clouds and {len(left_images)} left images.")

camera_timestamps = np.array([extract_timestamp(p) for p in left_images])

projector = LabelProjector(
    intrinsic=intrinsic,
    extrinsic=extrinsic,
    image_shape=tuple(Image.open(left_images[0]).convert("L").size[::-1])
)
extractor = DINOv2FeatureExtractor(model_name="vit_small_patch14_dinov2.lvd142m", device="cuda")

num_scans = len(point_clouds)
split_idx = int((1.0 - VAL_SPLIT) * num_scans)


for idx, ((xyz, fields), lidar_path) in enumerate(zip(point_clouds, data["point_cloud_paths"])):
    intensity = fields.get('intensity', np.zeros_like(fields['x']))
    velocity = fields.get('velocity', np.zeros_like(fields['x']))
    original_features = np.stack([intensity, velocity], axis=1)  # (N, 2)

    lidar_ts = extract_timestamp(lidar_path)
    img_idxs = get_closest_images(lidar_ts, camera_timestamps, TIME_THRESH)
    if not img_idxs:
        logging.warning(f"No camera image found within {TIME_THRESH}s for LiDAR scan {lidar_path}. Skipping.")
        continue

    N = xyz.shape[0]
    dino_feat_dim = None
    point_dino_feat = None

    all_img_dino_feats = []
    all_img_valid_mask = []

    for cam_idx in img_idxs:
        this_img_path = left_images[cam_idx]
        dino_dense = extractor.extract_features(Image.open(this_img_path).convert("RGB"))
        Hf = Wf = int(np.sqrt(dino_dense.shape[0]))
        dino_feat_dim = dino_dense.shape[1]

        # Project points into image (in resized 518x518 grid!)
        uv, valid = projector.project_points(xyz)
        img_shape = (518, 518)
        dino_shape = (Hf, Wf)
        patch_idx = project_points_to_dino_patches(uv, valid, img_shape, dino_shape)

        this_img_feats = np.zeros((N, dino_feat_dim), dtype=np.float32)
        # Assign features for valid points
        valid_mask = patch_idx >= 0
        this_img_feats[valid_mask] = dino_dense[patch_idx[valid_mask]]
        all_img_dino_feats.append(this_img_feats)
        all_img_valid_mask.append(valid_mask)

    # For each point: randomly choose a visible image (if multiple)
    point_dino_feat = np.zeros((N, dino_feat_dim), dtype=np.float32)
    for i in range(N):
        visible_imgs = [j for j, mask in enumerate(all_img_valid_mask) if mask[i]]
        if visible_imgs:
            chosen = random.choice(visible_imgs)
            point_dino_feat[i] = all_img_dino_feats[chosen][i]
        else:
            pass  # leave as zeros

    # Save everything
    sample = {
        "coord": torch.from_numpy(xyz).float(),
        "feat": torch.from_numpy(original_features).float(),
        "dino_feat": torch.from_numpy(point_dino_feat).float(),
        "grid_size": GRID_SIZE
    }
    out_dir = OUT_TRAIN if idx < split_idx else OUT_VAL
    out_path = out_dir / f"sample_{idx:04d}.pth"
    torch.save(sample, out_path)
    print(f"Saved {out_path}")

print("Dataset build complete!")
