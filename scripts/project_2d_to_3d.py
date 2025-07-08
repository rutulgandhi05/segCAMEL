import numpy as np
from typing import Tuple
import torch
class Projector:
    def __init__(self,
                 intrinsic: np.ndarray,      # 3×3 (or 3×4) camera intrinsic matrix
                 extrinsic: np.ndarray,      # 4×4 transform from LiDAR to camera frame
                 image_shape: Tuple[int,int]  # (H, W)
                 ):
        """
        Initialize with camera calibration.
        """
        self.K = intrinsic
        self.T = extrinsic
        self.H, self.W = image_shape
    

    def project_points(self, points):
        """
        points: (N, 3) 3D world points
        K: (3, 3) intrinsic matrix
        T: (4, 4) extrinsic (world-to-camera) matrix
        Returns: (N, 2) pixel coordinates, (N,) depth values in camera frame
        """
        N = points.shape[0]
        # Convert to homogeneous (N, 4)
        points_h = torch.cat([points, torch.ones((N, 1), device=points.device)], dim=1)  # (N, 4)
        # World to camera (T)
        points_cam = (self.T @ points_h.T).T  # (N, 4)
        # Drop last coordinate (homogeneous divide comes later)
        xyz_cam = points_cam[:, :3]  # (N, 3)
        # Project with K
        uvw = (self.K @ xyz_cam.T).T  # (N, 3)
        # Pixel coordinates
        u = uvw[:, 0] / uvw[:, 2]
        v = uvw[:, 1] / uvw[:, 2]
        z = uvw[:, 2]  # Depth in camera frame
        return torch.stack([u, v], dim=1), z

    def get_visibility_masks(self, pixel_coords, depth, image_shape):
        """
        pixel_coords: (N, 2) - u, v
        depth: (N,)
        image_shape: (H, W)
        Returns: (N,) boolean tensor
        """
        H, W = image_shape
        u, v = pixel_coords[:, 0], pixel_coords[:, 1]
        visible = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depth > 0)
        return visible

    def assign_patch_features(self, pixel_coords, visible, dino_feature_map, patch_size):
        """
        pixel_coords: (N, 2) - u, v
        visible: (N,) boolean mask
        dino_feature_map: (H_p, W_p, D)
        patch_size: int
        Returns: (N, D) tensor with DINO features (zero if not visible)
        """
        H_p, W_p, D = dino_feature_map.shape
        N = pixel_coords.shape[0]
        # Patch indices for visible points
        patch_u = torch.floor(pixel_coords[visible, 0] / patch_size).long().clamp(0, W_p-1)
        patch_v = torch.floor(pixel_coords[visible, 1] / patch_size).long().clamp(0, H_p-1)
        # Assign features
        features = torch.zeros((N, D), device=dino_feature_map.device)
        features[visible] = dino_feature_map[patch_v, patch_u]  # index shape (num_visible, D)
        return features


    def map_points_to_dino_features(self, points, dino_feature_map, patch_size, image_shape):
        """
        Map 3D points to DINO features in the image.
        points: (N, 3) 3D world points
        dino_feature_map: (H_p, W_p, D) DINO feature map
        patch_size: int - size of each patch in the feature map
        image_shape: (H, W) - shape of the image (height, width)
        Returns: (N, D) tensor with DINO features, (N,) boolean mask for visibility
        """
        # 1. Project 3D points to 2D
        pixel_coords, depth = self.project_points(points, self.K, self.T)
        # 2. Check which are visible
        visible = self.get_visibility_mask(pixel_coords, depth, image_shape)
        # 3. Assign DINO features
        features = self.assign_patch_features(pixel_coords, visible, dino_feature_map, patch_size)
        return features, visible
    

def scale_intrinsics(K, original_size, new_size):
    """
    Scales the intrinsic matrix K from original image size to new image size.

    Args:
        K (np.ndarray or torch.Tensor): (3, 3) camera intrinsic matrix.
        original_size (tuple): (width, height) of the original image.
        new_size (tuple): (width, height) of the new image.

    Returns:
        K_scaled: scaled (3, 3) intrinsic matrix, same type as input.
    """
    orig_w, orig_h = original_size
    new_w, new_h = new_size

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    if isinstance(K, np.ndarray):
        K_scaled = K.copy()
    else:
        K_scaled = K.clone()

    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy

    return K_scaled


if __name__ == "__main__":
    # Intrinsics and extrinsics as numpy arrays
    K = np.array([[1000, 0, 720],
                  [0, 1000, 540],
                  [0, 0, 1]], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)  # identity (world = camera for test)
    orig_size = (1440, 1080)         # (width, height) original
    new_size = (1428, 1078)          # after crop to patch multiple
    # Scale K for cropping/resizing if needed
    K_scaled = scale_intrinsics(K, orig_size, new_size)
    # Assume points (N, 3)
    N = 1000
    points = torch.randn((N, 3))
    # Fake DINOv2 feature map (H_p, W_p, D)
    H_p, W_p, D = 77, 102, 384
    dino_feature_map = torch.randn((H_p, W_p, D))
    patch_size = 14
    projector = Projector(K_scaled, T, (new_size[1], new_size[0]))
    features, visible = projector.map_points_to_dino_features(points, dino_feature_map, patch_size)
    print(features.shape, visible.sum().item())