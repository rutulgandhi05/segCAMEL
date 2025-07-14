import os
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from scripts.feature_extractor import Extractor
from scripts.dust3r_wrapper import Dust3RWrapper
from scripts.dataloader import ScantinelDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from PIL import Image
from utils.misc import setup_logger

logger = setup_logger("preprocess")

def lcm(a, b):
    import math
    return abs(a * b) // math.gcd(a, b)

def preprocess_and_save(
    root_dir,
    save_dir,
    grid_size=0.05,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = ScantinelDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # -- Load DINO extractor and DUSt3R wrapper, get patch sizes --
    extractor = Extractor()
    dust3r = Dust3RWrapper(device=device)

    dino_patch = extractor.model.patch_size
    dust3r_patch = dust3r.model.patch_embed.patch_size[0] \
        if hasattr(dust3r.model, "patch_embed") else 16  # fallback

    # Use LCM for max compatibility (e.g., 112 for 14 and 16)
    target_patch = lcm(dino_patch, dust3r_patch)
    # Pick a reasonable crop size that's a multiple of LCM (and not huge for GPU)
    # E.g., 448x336, 560x448, etc.
    target_w = 448  # Change as needed (must be multiple of target_patch)
    target_h = 336

    logger.info(f"DINO patch: {dino_patch}, DUSt3R patch: {dust3r_patch}, target crop: {target_w}x{target_h}")

    for idx, batch in enumerate(tqdm(dataloader, desc="Processing frames")):
        image_tensor = batch["image_tensor"][0]
        pil_img = ToPILImage()(image_tensor).convert("RGB")

        # -- Resize/crop image to common, compatible shape --
        pil_img_resized = pil_img.resize((target_w, target_h), Image.BILINEAR)
        pil_img_resized = pil_img_resized.crop((0, 0, target_w, target_h))

        # === Step 1: Extract DINO features on resized image ===
        features = extractor.extract_dino_features(image=pil_img_resized, filename=f"frame_{idx:05d}")
        dino_feat_tensor = features["features"].flat().tensor.squeeze(0).cpu().numpy()  # [num_patches, dim]
        feature_map_size = features["feature_map_size"]  # (w, h) grid

        logger.info(f"[Frame {idx}] DINO features: {dino_feat_tensor.shape}, Patch grid: {feature_map_size}")

        # === Step 2: Predict DUSt3R 3D pointmap (per-pixel) ===
        pts3d_dense = dust3r.predict_pointmap(pil_img_resized)  # [H*W, 3]
        # If output is 2x (stereo), only keep first image
        if pts3d_dense.shape[0] > target_w * target_h:
            pts3d_dense = pts3d_dense[:target_w * target_h]
        pts3d_dense = pts3d_dense.reshape(target_h, target_w, 3)

        # === Step 3: Downsample DUSt3R pointmap to DINO patch centers ===
        grid_w = target_w // dino_patch
        grid_h = target_h // dino_patch
        pts3d_patches = []
        for i in range(grid_h):
            for j in range(grid_w):
                y = i * dino_patch + dino_patch // 2
                x = j * dino_patch + dino_patch // 2
                y = min(y, target_h - 1)
                x = min(x, target_w - 1)
                pts3d_patches.append(pts3d_dense[y, x])
        pts3d_patches = np.stack(pts3d_patches)  # [num_patches, 3]

        # === Step 4: Check alignment ===
        if pts3d_patches.shape[0] != dino_feat_tensor.shape[0]:
            logger.info(f"[Frame {idx}] Patch mismatch: DUSt3R={pts3d_patches.shape[0]}, DINO={dino_feat_tensor.shape[0]}")
            min_len = min(pts3d_patches.shape[0], dino_feat_tensor.shape[0])
            pts3d_patches = pts3d_patches[:min_len]
            dino_feat_tensor = dino_feat_tensor[:min_len]

        # === Step 5: Assign features to LiDAR points using k-NN ===
        pointcloud = batch["pointcloud"][0].numpy()
        lidar_xyz = pointcloud[:, :3]           # [N, 3]
        lidar_feats = pointcloud[:, 3:]         # [N, 3] (keep, ignore for now)

        knn = NearestNeighbors(n_neighbors=1).fit(pts3d_patches)
        _, indices = knn.kneighbors(lidar_xyz)  # [N, 1]
        assigned_feats = dino_feat_tensor[indices.squeeze(1)]  # [N, feat_dim]

        # === Step 6: Save ===
        save_data = {
            "coord": torch.tensor(lidar_xyz, dtype=torch.float32),
            "feat": torch.tensor(lidar_feats, dtype=torch.float32),
            "dino_feat": torch.tensor(assigned_feats, dtype=torch.float32),
            "grid_size": grid_size
        }

        save_path = save_dir / f"frame_{idx:05d}.pth"
        torch.save(save_data, save_path)
        logger.info(f"[Saved] {save_path}")

if __name__ == "__main__":
    ROOT_DIR = "data/scantinel/250612_RG_dynamic_test_drive/IN003_MUL_SEN_0.2.0.post184+g6da4bed"
    SAVE_DIR = "data/scantinel/250612_RG_dynamic_test_drive/IN003_processed_pth"
    preprocess_and_save(ROOT_DIR, SAVE_DIR)
