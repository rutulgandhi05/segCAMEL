import torch

class LidarToImageProjector:
    def __init__(self, intrinsic, extrinsic, image_size, feature_map_size, patch_features):
        self.K = intrinsic
        self.T = extrinsic
        self.image_size = image_size
        self.feature_map_size = feature_map_size
        self.features = patch_features.squeeze(0) if patch_features.dim() == 3 else patch_features
        self.patch_w = image_size[0] / feature_map_size[0]
        self.patch_h = image_size[1] / feature_map_size[1]

    def project_points(self, lidar_xyz):
        N = lidar_xyz.shape[0]
        ones = torch.ones((N, 1), device=lidar_xyz.device, dtype=lidar_xyz.dtype)
        lidar_homo = torch.cat([lidar_xyz, ones], dim=1)  # [N,4]
        cam_xyz = (self.T @ lidar_homo.T).T[:, :3]
        valid_mask = cam_xyz[:, 2] > 0
        cam_pts = cam_xyz[valid_mask]
        pixel_coords = (self.K @ cam_pts.T).T
        uvs = pixel_coords[:, :2] / pixel_coords[:, 2:3]
        return uvs, valid_mask

    def assign_features(self, lidar_xyz):
        device = lidar_xyz.device
        uvs, valid_mask = self.project_points(lidar_xyz)
        N, D = lidar_xyz.shape[0], self.features.shape[1]
        point_feats = torch.zeros((N, D), dtype=torch.float32, device=device)
        u, v = uvs[:, 0], uvs[:, 1]
        i = torch.clamp((v / self.patch_h).long(), 0, self.feature_map_size[1] - 1)
        j = torch.clamp((u / self.patch_w).long(), 0, self.feature_map_size[0] - 1)
        patch_idx = i * self.feature_map_size[0] + j
        feats = self.features.view(-1, D).to(device)  # Flatten patch grid
        point_feats[valid_mask] = feats[patch_idx]
        return point_feats, valid_mask
