import torch

class LidarToImageProjector:
    def __init__(self, intrinsic, extrinsic, image_size, feature_map_size, patch_features):
        """
        intrinsic: (3,3) torch.float32
        extrinsic: (4,4) torch.float32  (camera_T_lidar; transforms LiDAR -> camera)
        image_size: (W, H) in pixels after TransformFactory resize
        feature_map_size: (Wf, Hf) token lattice size from TransformFactory
        patch_features: (Hf*Wf, D) or (Hf, Wf, D) features (torch.float32/float16)
        """
        self.K = intrinsic
        self.T = extrinsic
        self.W, self.H = image_size
        self.Wf, self.Hf = feature_map_size
        self.features = patch_features
        if self.features.dim() == 2:
            self.features = self.features.view(self.Hf, self.Wf, -1).contiguous()
        self.D = self.features.shape[-1]

        # pixel size of a token cell (assumes uniform grid)
        self.patch_w = float(self.W) / float(self.Wf)
        self.patch_h = float(self.H) / float(self.Hf)

    def project_points(self, lidar_xyz):
        """
        lidar_xyz: (N,3) in LiDAR frame
        Returns:
          uvs: (N,2) pixel coords in the resized image
          in_front: (N,) bool (Z_cam > 0)
          depth: (N,) float (Z_cam)
        """
        N = lidar_xyz.shape[0]
        ones = torch.ones((N, 1), device=lidar_xyz.device, dtype=lidar_xyz.dtype)
        lidar_homo = torch.cat([lidar_xyz, ones], dim=1)  # [N,4]
        cam_xyz = (self.T @ lidar_homo.T).T[:, :3]        # [N,3]
        depth = cam_xyz[:, 2]
        in_front = depth > 0
        cam_pts = cam_xyz[in_front]
        pix = (self.K @ cam_pts.T).T
        uvs = torch.empty((N, 2), device=lidar_xyz.device, dtype=lidar_xyz.dtype)
        uvs[in_front] = pix[:, :2] / pix[:, 2:3]
        # for points not in_front, uvs are left uninitialized; always mask them
        return uvs, in_front, depth

    def _sample_bilinear(self, u, v):
        """
        Bilinear sampling on token lattice. u,v are pixel coords (valid in-bounds).
        Returns features (M,D) for the selected subset.
        """
        # continuous token coords
        jf = u / self.patch_w
        if_ = v / self.patch_h

        j0 = torch.floor(jf).clamp(0, self.Wf - 1).long()
        i0 = torch.floor(if_).clamp(0, self.Hf - 1).long()
        j1 = (j0 + 1).clamp(0, self.Wf - 1)
        i1 = (i0 + 1).clamp(0, self.Hf - 1)

        wj1 = (jf - j0.to(jf.dtype)).clamp(0, 1)
        wi1 = (if_ - i0.to(if_.dtype)).clamp(0, 1)
        wj0 = 1.0 - wj1
        wi0 = 1.0 - wi1

        # gather 4 corners
        F = self.features  # (Hf, Wf, D)
        f00 = F[i0, j0]    # top-left
        f01 = F[i0, j1]    # top-right
        f10 = F[i1, j0]    # bottom-left
        f11 = F[i1, j1]    # bottom-right

        w00 = (wi0 * wj0).unsqueeze(-1)
        w01 = (wi0 * wj1).unsqueeze(-1)
        w10 = (wi1 * wj0).unsqueeze(-1)
        w11 = (wi1 * wj1).unsqueeze(-1)

        return f00 * w00 + f01 * w01 + f10 * w10 + f11 * w11  # (M,D)

    def assign_features(self, lidar_xyz, *, bilinear: bool = False, occlusion_eps: float = 0.05):
        """
        Returns:
          point_feats: (N,D) float32
          valid_mask: (N,) bool  (true if Z>0 and within image bounds)
        Args:
          bilinear: if True, use bilinear sampling on token lattice; else nearest token
          occlusion_eps: if >0, keep only front-most points within each token cell
        """
        device = lidar_xyz.device
        N = lidar_xyz.shape[0]
        point_feats = torch.zeros((N, self.D), dtype=torch.float32, device=device)

        uvs, in_front, depth = self.project_points(lidar_xyz)
        u, v = uvs[:, 0], uvs[:, 1]
        in_img = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        valid = in_front & in_img

        # Optional: depth-aware occlusion filtering per token cell
        keep = valid.clone()
        if occlusion_eps > 0:
            # token index (nearest) for grouping
            j = torch.clamp((u / self.patch_w).long(), 0, self.Wf - 1)
            i = torch.clamp((v / self.patch_h).long(), 0, self.Hf - 1)
            tok = i * self.Wf + j
            # compute min depth per token
            depth_valid = depth.clone()
            depth_valid[~valid] = float("inf")
            # scatter reduce (PyTorch 2.0+). If unavailable, do CPU numpy fallback.
            min_depth = torch.full((self.Hf * self.Wf,), float("inf"), device=device, dtype=depth.dtype)
            min_depth = min_depth.scatter_reduce(0, tok[valid], depth[valid], reduce='amin', include_self=True)
            # keep only those near the min
            keep = valid & (depth <= (min_depth[tok] + occlusion_eps))

        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            if bilinear:
                feats_sel = self._sample_bilinear(u[idx], v[idx])
            else:
                # nearest token
                j = torch.clamp((u[idx] / self.patch_w).long(), 0, self.Wf - 1)
                i = torch.clamp((v[idx] / self.patch_h).long(), 0, self.Hf - 1)
                feats_sel = self.features[i, j]

            point_feats[idx] = feats_sel.to(point_feats.dtype)

        return point_feats, keep  # keep is the true visibility mask (in-front & in-image [+ occlusion])
