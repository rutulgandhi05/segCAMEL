# scripts/build_dataset.py

from pathlib import Path
import torch
import numpy as np
from PIL import Image
import random

from scripts.dataset import load_hercules_dataset_folder
from scripts.project_2d_to_3d import LabelProjector, project_points_to_dino_patches
from scripts.twod_feature_extractor import DINOv2FeatureExtractor
from scripts.mask_generator import MaskGenerator

RAW_FOLDER   = Path("data/hercules/Mountain_01_Day/")
OUT_FOLDER   = Path("data/hercules/processed/Mountain_01_Day/")
OUT_TRAIN    = OUT_FOLDER / "train"
OUT_VAL      = OUT_FOLDER / "val"
GRID_SIZE    = 0.05
VAL_SPLIT    = 0.2
TIME_THRESH  = 0.05  # 50 ms

OUT_TRAIN.mkdir(parents=True, exist_ok=True)
OUT_VAL.mkdir(parents=True, exist_ok=True)

# Load raw data
data = load_hercules_dataset_folder(RAW_FOLDER, return_all_fields=True)
point_clouds  = data["point_clouds"]                # list of (xyz, fields) tuples
left_images   = data["stereo_left_images"]          # list of Path
intrinsic     = data["stereo_left_intrinsics"]      # (3×3)
extrinsic     = data["lidar_to_stereo_left_extrinsic"]  # (4×4)

print(f"Loaded {len(point_clouds)} point clouds and {len(left_images)} left images.")

def extract_timestamp(p: Path):
    return float(p.stem) * 1e-9

camera_timestamps = np.array([extract_timestamp(p) for p in left_images])

projector = LabelProjector(
    intrinsic=intrinsic,
    extrinsic=extrinsic,
    image_shape=tuple(Image.open(left_images[0]).convert("L").size[::-1])
)
extractor = DINOv2FeatureExtractor(model_name="vit_small_patch14_dinov2.lvd142m", device="cuda")
mask_generator = MaskGenerator(num_clusters=50)

num_scans = len(point_clouds)
split_idx = int((1.0 - VAL_SPLIT) * num_scans)

def get_closest_images(lidar_ts, camera_ts, thresh):
    delta = np.abs(camera_ts - lidar_ts)
    min_delta = np.min(delta)
    close_idxs = np.where(delta == min_delta)[0]
    if min_delta > thresh:
        return []
    return close_idxs.tolist()

for idx, ((xyz, fields), lidar_path) in enumerate(zip(point_clouds, data["point_cloud_paths"])):
    intensity = fields.get('intensity', np.zeros_like(fields['x']))
    velocity = fields.get('velocity', np.zeros_like(fields['x']))
    original_features = np.stack([intensity, velocity], axis=1)  # (N, 2)

    lidar_ts = extract_timestamp(lidar_path)
    img_idxs = get_closest_images(lidar_ts, camera_timestamps, TIME_THRESH)
    if not img_idxs:
        print(f"Warning: No camera image found within {TIME_THRESH}s for LiDAR scan {lidar_path}. Skipping.")
        continue

    N = xyz.shape[0]
    dino_feat_dim = None
    point_dino_feat = None
    point_labels = None

    all_img_dino_feats = []
    all_img_valid_mask = []
    all_img_labels = []

    for cam_idx in img_idxs:
        this_img_path = left_images[cam_idx]
        dino_dense = extractor.extract_features(Image.open(this_img_path).convert("RGB"))
        Hf = Wf = int(np.sqrt(dino_dense.shape[0]))
        dino_feat_dim = dino_dense.shape[1]
        mask = mask_generator.generate(dino_dense, (Hf, Wf))
        mask_flat = mask.reshape(-1)

        # Project points into image (in resized 518x518 grid!)
        uv, valid = projector.project_points(xyz)
        img_shape = (518, 518)
        dino_shape = (Hf, Wf)
        patch_idx = project_points_to_dino_patches(uv, valid, img_shape, dino_shape)

        this_img_feats = np.zeros((N, dino_feat_dim), dtype=np.float32)
        this_img_labels = -1 * np.ones(N, dtype=np.int32)
        # Assign features and labels for valid points
        valid_mask = patch_idx >= 0
        this_img_feats[valid_mask] = dino_dense[patch_idx[valid_mask]]
        this_img_labels[valid_mask] = mask_flat[patch_idx[valid_mask]]
        all_img_dino_feats.append(this_img_feats)
        all_img_valid_mask.append(valid_mask)
        all_img_labels.append(this_img_labels)

    # For each point: randomly choose a visible image (if multiple)
    point_dino_feat = np.zeros((N, dino_feat_dim), dtype=np.float32)
    point_labels = -1 * np.ones(N, dtype=np.int32)
    for i in range(N):
        visible_imgs = [j for j, mask in enumerate(all_img_valid_mask) if mask[i]]
        if visible_imgs:
            chosen = random.choice(visible_imgs)
            point_dino_feat[i] = all_img_dino_feats[chosen][i]
            point_labels[i] = all_img_labels[chosen][i]
        else:
            pass  # leave as zeros

    # Save everything
    sample = {
        "coord": torch.from_numpy(xyz).float(),
        "feat": torch.from_numpy(original_features).float(),
        "dino_feat": torch.from_numpy(point_dino_feat).float(),
        "pseudo_label": torch.from_numpy(point_labels).long(),
        "grid_size": GRID_SIZE
    }
    out_dir = OUT_TRAIN if idx < split_idx else OUT_VAL
    out_path = out_dir / f"sample_{idx:04d}.pth"
    torch.save(sample, out_path)
    print(f"Saved {out_path}")

print("Dataset build complete!")
