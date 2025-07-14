import numpy as np


class Projector:
    def __init__(self, patch_size=14):
        self.patch_size = patch_size
        self.input_image_size = None         # (H, W) of the resized image input to DINO
        self.feature_map_size = None         # (H_patch, W_patch) feature grid
        self.K = None                        # Intrinsic matrix

    def set_camera_intrinsics(self, K):
        self.K = K

    def project_points(self, points):
        """
        Projects 3D LiDAR points to 2D pixel coordinates and computes patch IDs.
        """
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        z = np.where(z == 0, 1e-6, z)  # avoid divide-by-zero
        u = (fx * x / z) + cx
        v = (fy * y / z) + cy
        pixel_coords = np.stack([u, v], axis=1)
        depth = z

        h_img, w_img = self.input_image_size
        h_feat, w_feat = self.feature_map_size

        # Normalize pixel coordinates to patch indices
        x_idx = (pixel_coords[:, 0] / w_img * w_feat).astype(int)
        y_idx = (pixel_coords[:, 1] / h_img * h_feat).astype(int)
        patch_ids = y_idx * w_feat + x_idx

        return pixel_coords, patch_ids, depth

    def get_visibility_masks(self, pixel_coords, depth, image_shape, depth_thresh=0.1):
        """
        Returns a boolean mask for whether the 3D points project inside the image bounds.
        """
        h, w = image_shape
        valid = (
            (pixel_coords[:, 0] >= 0) &
            (pixel_coords[:, 1] >= 0) &
            (pixel_coords[:, 0] < w) &
            (pixel_coords[:, 1] < h) &
            (depth > 0)
        )
        return valid

    def assign_patch_features(self, points, feature_map):
        """
        Assigns a DINO feature vector to each 3D LiDAR point based on projected patch ID.
        """
        pixel_coords, patch_ids, depth = self.project_points(points)
        visible = self.get_visibility_masks(pixel_coords, depth, self.input_image_size)

        dino_feats = np.zeros((len(points), feature_map.shape[-1]), dtype=np.float32)
        valid_patch_ids = patch_ids[visible]
        valid_patch_ids = np.clip(valid_patch_ids, 0, feature_map.shape[0] - 1)
        dino_feats[visible] = feature_map[valid_patch_ids]

        return dino_feats

