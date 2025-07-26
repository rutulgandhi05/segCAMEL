from tkinter import Image
import numpy as np
import torch
from typing import Tuple

class LidarToImageProjector:
    def __init__(
        self,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        image_size: Tuple[int, int],
        feature_map_size: Tuple[int, int],
        patch_features: torch.Tensor,
    ):
        self.K = intrinsic  # (3x3)
        self.T = extrinsic  # (4x4) LiDAR-to-Camera
        self.image_size = image_size  # (width, height)
        self.feature_map_size = feature_map_size  # (w_patches, h_patches)
        self.features = patch_features.squeeze(0).cpu().numpy()  # [N_patches, D]
        self.patch_w = image_size[0] / feature_map_size[0]
        self.patch_h = image_size[1] / feature_map_size[1]

    def project_points(self, lidar_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = lidar_xyz.shape[0]
        lidar_homo = np.hstack([lidar_xyz, np.ones((N, 1))])  # (N, 4)
        cam_xyz = (self.T @ lidar_homo.T).T[:, :3]  # (N, 3)

        valid_mask = cam_xyz[:, 2] > 0  # only points in front of the camera
        cam_pts = cam_xyz[valid_mask]

        pixel_coords = (self.K @ cam_pts.T).T  # (N_valid, 3)
        uvs = pixel_coords[:, :2] / pixel_coords[:, 2:3]

        return uvs, valid_mask

    def assign_features(self, lidar_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        uvs, valid_mask = self.project_points(lidar_xyz)
        N, D = lidar_xyz.shape[0], self.features.shape[1]
        point_feats = np.zeros((N, D), dtype=np.float32)

        u, v = uvs[:, 0], uvs[:, 1]
        i = np.clip((v / self.patch_h).astype(int), 0, self.feature_map_size[1] - 1)
        j = np.clip((u / self.patch_w).astype(int), 0, self.feature_map_size[0] - 1)
        patch_idx = i * self.feature_map_size[0] + j

        point_feats[valid_mask] = self.features[patch_idx]
        return point_feats, valid_mask.astype(np.uint8)

# Example usage
def main():
    # Dummy example, replace with real data
    from scripts.dataset import load_hercules_dataset_folder
    from scripts.feature_extractor import Extractor
    from PIL import Image
    from utils.visualization import visualize_pca_colored_pointcloud
    from utils.misc import scale_intrinsics
    from pathlib import Path

    data = load_hercules_dataset_folder(Path("data/hercules/Mountain_01_Day"), return_all_fields=False)
    sample = data[50]

    left_image_path = sample["right_image"]
    image = Image.open(left_image_path).convert("RGB")
    intrinsic = sample["stereo_right_intrinsics"]
    print(f"Image size: {image.size}, Intrinsics: {type(intrinsic)}")
    extrinsic = sample["lidar_to_stereo_right_extrinsic"]

    extractor = Extractor()
    res = extractor.extract_dino_features(image=image, filename="sample_image")
    feature_map_size = res['feature_map_size']
    patch_feats = res['features'].flat().tensor
    input_size = res['input_size']

    lidar_points = sample["pointcloud"]

    intrinsic = scale_intrinsics(intrinsic, image.size, input_size)
    print(f"Scaled Intrinsics: {intrinsic}")
    projector = LidarToImageProjector(
        intrinsic, extrinsic, input_size, feature_map_size, patch_feats
    )
    feats, mask = projector.assign_features(lidar_points)
    visualize_pca_colored_pointcloud(
        lidar_points, feats, mask
    )
    #np.savez("data/hercules/Mountain_01_Day/lidar_with_dino_features.npz", xyz=lidar_points, dino_feats=feats, visible=mask)

if __name__ == "__main__":
    main()
