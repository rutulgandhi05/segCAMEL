import os
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from scripts.feature_extractor import Extractor
from scripts.dataloader import ScantinelDataset, HerculesDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from PIL import Image
from utils.misc import setup_logger
from utils.misc import scale_intrinsics
from scripts.project_2d_to_3d import LidarToImageProjector

logger = setup_logger("preprocess")


def preprocess_and_save_hercules(
    root_dir,
    save_dir,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = HerculesDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    extractor = Extractor()
    print(f"Using DINO model: {extractor.dino_model}")

    for idx, batch in enumerate(tqdm(dataloader, desc="Processing frames")):

        image_tensor = batch["image_tensor"][0]
        pil_img = ToPILImage()(image_tensor).convert("RGB")
    
        features = extractor.extract_dino_features(image=pil_img, filename=f"frame_{idx:05d}")
        pil_img.close()  # Close the PIL image to free resources
        dino_feat_tensor = features["features"].flat().tensor
        input_size = features["input_size"]
        feature_map_size = features["feature_map_size"]

        # ======= Pointcloud Handling =======
        pointcloud = batch["pointcloud"]  # shape should be [N, C] already
        if isinstance(pointcloud, torch.Tensor):
            pointcloud = pointcloud.squeeze().numpy()
        if pointcloud.ndim == 3 and pointcloud.shape[0] == 1:
            pointcloud = pointcloud.squeeze(0)  # now shape [N, C]

        lidar_xyz = pointcloud[:, :3]
        lidar_feats = pointcloud[:, 3:] if pointcloud.shape[1] > 3 else np.zeros((lidar_xyz.shape[0], 1), dtype=np.float32)

        # ======= Intrinsics Handling =======
        intrinsics = batch["intrinsics"]
        intrinsics = scale_intrinsics(intrinsics, pil_img.size, input_size)
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.squeeze().numpy()
        elif isinstance(intrinsics, np.ndarray):
            intrinsics = intrinsics.squeeze()
        
        extrinsics = batch["extrinsics"]
        if isinstance(extrinsics, torch.Tensor):
            extrinsics = extrinsics.squeeze().numpy()
        elif isinstance(extrinsics, np.ndarray):
            extrinsics = extrinsics.squeeze()

        #print(type(intrinsics), type(extrinsics), type(dino_feat_tensor), type(features["input_size"]))
        
        # ======= Projector & Feature Assignment =======
        projector = LidarToImageProjector(
            intrinsic=intrinsics,
            extrinsic=extrinsics,
            image_size=input_size,
            feature_map_size=feature_map_size,
            patch_features=dino_feat_tensor
        )
        assigned_feats, mask = projector.assign_features(lidar_xyz=lidar_xyz)
        assigned_feats = torch.tensor(assigned_feats, dtype=torch.float32)

        # ======= Grid Size Calculation =======
        if lidar_xyz.shape[0] > 1:
            diffs = np.diff(np.sort(lidar_xyz, axis=0), axis=0)
            mean_spacing = np.mean(np.abs(diffs), axis=0)
            grid_size_manual = float(np.mean(mean_spacing))
        else:
            grid_size_manual = 0.05

        # ======= Save Processed Data =======
        save_data = {
            "coord": torch.tensor(lidar_xyz, dtype=torch.float32),
            "feat": torch.tensor(lidar_feats, dtype=torch.float32),
            "dino_feat": assigned_feats,
            "grid_size": torch.tensor(grid_size_manual, dtype=torch.float32)
        }
        save_path = save_dir / f"frame_{idx:05d}.pth"
        torch.save(save_data, save_path)
        print(f"[Saved] {save_path}")


if __name__ == "__main__":
    import time

    start_time = time.time()
    print("Starting preprocessing...")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    HERCULES_ROOT_DIR = "data/hercules/Mountain_01_Day/"
    HERCULES_SAVE_DIR = "data/hercules/Mountain_01_Day/processed_data"
    preprocess_and_save_hercules(HERCULES_ROOT_DIR, HERCULES_SAVE_DIR)
    end_time = time.time()
    print("Preprocessing completed.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")