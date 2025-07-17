import os
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from scripts.feature_extractor import Extractor
from scripts.dust3r_wrapper import Dust3RWrapper
from scripts.dataloader import ScantinelDataset, HerculesDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from PIL import Image
from utils.misc import setup_logger

logger = setup_logger("preprocess")

def lcm(a, b):
    import math
    return abs(a * b) // math.gcd(a, b)

def preprocess_and_save_scantinel(
    root_dir,
    save_dir,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    debug_save_dir = save_dir / "debug"
    debug_save_dir.mkdir(exist_ok=True)

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
    target_w = 560  # Should be multiple of target_patch
    target_h = 448

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
        lidar_feats = pointcloud[:, 3:]         # [N, 3] or as many LiDAR features as you have

        # Swap X and Y for X-forward convention (if this is your convention)
        lidar_xyz_swapped = lidar_xyz[:, [1, 0, 2]]

        knn = NearestNeighbors(n_neighbors=1).fit(pts3d_patches)
        dists, indices = knn.kneighbors(lidar_xyz_swapped)  # [N, 1], [N, 1]
        assigned_feats = dino_feat_tensor[indices.squeeze(1)]  # [N, feat_dim]

        # === DEBUG: Save all intermediates for this frame ===
        # 1. Images
        pil_img.save(debug_save_dir / f"orig_frame_{idx:05d}.png")
        pil_img_resized.save(debug_save_dir / f"resized_frame_{idx:05d}.png")
        # 2. DUSt3R full dense and patch center pointmap
        np.save(debug_save_dir / f"pts3d_dense_{idx:05d}.npy", pts3d_dense)
        np.save(debug_save_dir / f"pts3d_patches_{idx:05d}.npy", pts3d_patches)
        # 3. DINO feature tensor for patches + feature map size
        np.save(debug_save_dir / f"dino_feat_tensor_{idx:05d}.npy", dino_feat_tensor)
        np.save(debug_save_dir / f"feature_map_size_{idx:05d}.npy", np.array(feature_map_size))
        # 4. k-NN assignment indices and distances
        np.save(debug_save_dir / f"indices_{idx:05d}.npy", indices.squeeze(1))
        np.save(debug_save_dir / f"dists_{idx:05d}.npy", dists.squeeze(1))
        # 5. LiDAR
        np.save(debug_save_dir / f"lidar_xyz_{idx:05d}.npy", lidar_xyz)
        np.save(debug_save_dir / f"lidar_xyz_swapped_{idx:05d}.npy", lidar_xyz_swapped)
        np.save(debug_save_dir / f"lidar_feats_{idx:05d}.npy", lidar_feats)

        # === Step 6: Save main training data ===
        # Compute grid size for point cloud (mean spacing)
        if lidar_xyz_swapped.shape[0] > 1:
            diffs = np.diff(np.sort(lidar_xyz_swapped, axis=0), axis=0)
            mean_spacing = np.mean(np.abs(diffs), axis=0)
            grid_size_manual = float(np.mean(mean_spacing))
        else:
            grid_size_manual = 0.05  # fallback if only one point

        save_data = {
            "coord": torch.tensor(lidar_xyz_swapped, dtype=torch.float32),
            "feat": torch.tensor(lidar_feats, dtype=torch.float32),
            "dino_feat": torch.tensor(assigned_feats, dtype=torch.float32),
            "grid_size": torch.tensor(grid_size_manual, dtype=torch.float32)
        }
        save_path = save_dir / f"frame_{idx:05d}.pth"
        torch.save(save_data, save_path)
        logger.info(f"[Saved] {save_path}")

def preprocess_and_save_hercules(
    root_dir,
    save_dir,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    from scripts.dataloader import HerculesDataset  # Make sure this import is present
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    debug_save_dir = save_dir / "debug"
    debug_save_dir.mkdir(exist_ok=True)

    dataset = HerculesDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    extractor = Extractor()
    dust3r = Dust3RWrapper(device=device)

    dino_patch = extractor.model.patch_size
    dust3r_patch = dust3r.model.patch_embed.patch_size[0] if hasattr(dust3r.model, "patch_embed") else 16

    target_patch = lcm(dino_patch, dust3r_patch)
    target_w = 560
    target_h = 448

    logger.info(f"DINO patch: {dino_patch}, DUSt3R patch: {dust3r_patch}, target crop: {target_w}x{target_h}")

    for idx, batch in enumerate(tqdm(dataloader[:50], desc="Processing frames")):
        # Use left camera for main processing
        image_tensor = batch["image_tensor"][0]
        pil_img = ToPILImage()(image_tensor).convert("RGB")

        pil_img_resized = pil_img.resize((target_w, target_h), Image.BILINEAR)
        pil_img_resized = pil_img_resized.crop((0, 0, target_w, target_h))

        features = extractor.extract_dino_features(image=pil_img_resized, filename=f"frame_{idx:05d}")
        dino_feat_tensor = features["features"].flat().tensor.squeeze(0).cpu().numpy()
        feature_map_size = features["feature_map_size"]

        logger.info(f"[Frame {idx}] DINO features: {dino_feat_tensor.shape}, Patch grid: {feature_map_size}")

        pts3d_dense = dust3r.predict_pointmap(pil_img_resized)
        if pts3d_dense.shape[0] > target_w * target_h:
            pts3d_dense = pts3d_dense[:target_w * target_h]
        pts3d_dense = pts3d_dense.reshape(target_h, target_w, 3)

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
        pts3d_patches = np.stack(pts3d_patches)

        if pts3d_patches.shape[0] != dino_feat_tensor.shape[0]:
            logger.info(f"[Frame {idx}] Patch mismatch: DUSt3R={pts3d_patches.shape[0]}, DINO={dino_feat_tensor.shape[0]}")
            min_len = min(pts3d_patches.shape[0], dino_feat_tensor.shape[0])
            pts3d_patches = pts3d_patches[:min_len]
            dino_feat_tensor = dino_feat_tensor[:min_len]

        # Pointcloud handling for Hercules
        pointcloud = batch["pointcloud"][0].numpy()
        lidar_xyz = pointcloud[:, :3]
        lidar_feats = pointcloud[:, 3:]

        # If Hercules uses a different convention, modify here (swap axes if needed)
        lidar_xyz_swapped = lidar_xyz  # Don't swap axes unless you have evidence it's needed!

        knn = NearestNeighbors(n_neighbors=1).fit(pts3d_patches)
        dists, indices = knn.kneighbors(lidar_xyz_swapped)
        assigned_feats = dino_feat_tensor[indices.squeeze(1)]

        # === DEBUG: Save all intermediates for this frame ===
        pil_img.save(debug_save_dir / f"orig_frame_{idx:05d}.png")
        pil_img_resized.save(debug_save_dir / f"resized_frame_{idx:05d}.png")
        np.save(debug_save_dir / f"pts3d_dense_{idx:05d}.npy", pts3d_dense)
        np.save(debug_save_dir / f"pts3d_patches_{idx:05d}.npy", pts3d_patches)
        np.save(debug_save_dir / f"dino_feat_tensor_{idx:05d}.npy", dino_feat_tensor)
        np.save(debug_save_dir / f"feature_map_size_{idx:05d}.npy", np.array(feature_map_size))
        np.save(debug_save_dir / f"indices_{idx:05d}.npy", indices.squeeze(1))
        np.save(debug_save_dir / f"dists_{idx:05d}.npy", dists.squeeze(1))
        np.save(debug_save_dir / f"lidar_xyz_{idx:05d}.npy", lidar_xyz)
        np.save(debug_save_dir / f"lidar_feats_{idx:05d}.npy", lidar_feats)

        # Compute grid size for point cloud (mean spacing)
        if lidar_xyz_swapped.shape[0] > 1:
            diffs = np.diff(np.sort(lidar_xyz_swapped, axis=0), axis=0)
            mean_spacing = np.mean(np.abs(diffs), axis=0)
            grid_size_manual = float(np.mean(mean_spacing))
        else:
            grid_size_manual = 0.05

        save_data = {
            "coord": torch.tensor(lidar_xyz_swapped, dtype=torch.float32),
            "feat": torch.tensor(lidar_feats, dtype=torch.float32),
            "dino_feat": torch.tensor(assigned_feats, dtype=torch.float32),
            "grid_size": torch.tensor(grid_size_manual, dtype=torch.float32)
        }
        save_path = save_dir / f"frame_{idx:05d}.pth"
        torch.save(save_data, save_path)
        logger.info(f"[Saved] {save_path}")


if __name__ == "__main__":
    """ SCANTINEL_ROOT_DIR = "data/scantinel/250612_RG_dynamic_test_drive/IN003_MUL_SEN_0.2.0.post184+g6da4bed"
    SCANTINEL_SAVE_DIR = "data/scantinel/250612_RG_dynamic_test_drive/IN003_processed_pth"
    preprocess_and_save_scantinel(SCANTINEL_ROOT_DIR, SCANTINEL_SAVE_DIR) """

    HERCULES_ROOT_DIR = "data/hercules/Mountain_01_Day/"
    HERCULES_SAVE_DIR = "data/hercules/Mountain_01_Day/processed_data"
    preprocess_and_save_hercules(HERCULES_ROOT_DIR, HERCULES_SAVE_DIR)
