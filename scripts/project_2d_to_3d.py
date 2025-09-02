import math
from typing import Tuple, Optional, Union

import torch
import numpy as np


def _as_4x4(T: torch.Tensor) -> torch.Tensor:
    """Ensure a 4x4 homogeneous transform (float32, device preserved)."""
    T = T.to(torch.float32)
    if T.dim() == 2 and T.shape == (4, 4):
        return T
    if T.dim() == 2 and T.shape == (3, 4):
        out = torch.eye(4, dtype=torch.float32, device=T.device)
        out[:3, :4] = T
        return out
    raise ValueError(f"Extrinsic must be (4,4) or (3,4); got {tuple(T.shape)}")


def _K_params(K: torch.Tensor):
    """Extract fx, fy, cx, cy from a 3x3 intrinsics matrix."""
    K = K.to(torch.float32)
    if K.dim() != 2 or K.shape != (3, 3):
        raise ValueError(f"Intrinsic must be 3x3; got {tuple(K.shape)}")
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


class LidarToImageProjector:
    """
    Projects LiDAR XYZ into camera image & token lattice and assigns 2D patch features to 3D points.

    Fixes included:
      - Safe UV init (uvs=-1 for behind-camera points) to avoid bogus in-image hits.
      - One-time auto-orientation: choose T vs inv(T) by which yields more (in-image & z>0).
      - Token z-buffer via scatter_reduce_('amin') with a NumPy fallback.
      - Fallback feature sampler that ignores occlusion (for preprocess fallback path).
    """

    def __init__(
        self,
        intrinsic: torch.Tensor,            # (3,3)
        extrinsic: torch.Tensor,            # (4,4) or (3,4); nominal LiDAR->Camera
        image_size: Tuple[int, int],        # (W, H) after any resize done for DINO
        feature_map_size: Tuple[int, int],  # (Wf, Hf) DINO token grid
        patch_features: torch.Tensor,       # (Hf*Wf, D) OR (Hf, Wf, D)
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = intrinsic.device
        self.device = device

        # Intrinsics
        self.K = intrinsic.to(device=device, dtype=torch.float32)
        self.fx, self.fy, self.cx, self.cy = _K_params(self.K)

        # Extrinsics (store both directions; pick one after probing)
        self.T_nom = _as_4x4(extrinsic.to(device=device))
        self.T_inv = torch.linalg.inv(self.T_nom)
        self._checked_orientation = False
        self._use_inv = False  # False => use T_nom; True => use T_inv

        # Image & token sizes
        W, H = image_size
        self.img_w = int(W)
        self.img_h = int(H)

        Wf, Hf = feature_map_size
        self.Wf = int(Wf)
        self.Hf = int(Hf)

        # Token cell size in pixels
        self.patch_w = float(self.img_w) / max(1.0, float(self.Wf))
        self.patch_h = float(self.img_h) / max(1.0, float(self.Hf))

        # Features as (Hf, Wf, D) float32
        if patch_features.dim() == 3 and patch_features.shape[0] == self.Hf and patch_features.shape[1] == self.Wf:
            feats = patch_features
        elif patch_features.dim() == 2 and patch_features.shape[0] == self.Hf * self.Wf:
            feats = patch_features.view(self.Hf, self.Wf, -1)
        else:
            raise ValueError(
                f"patch_features must be (Hf,Wf,D) or (Hf*Wf,D); got {tuple(patch_features.shape)} "
                f"with Hf={self.Hf},Wf={self.Wf}"
            )
        self.features = feats.to(device=self.device, dtype=torch.float32).contiguous()
        self.D = int(self.features.shape[-1])

        # Small epsilon to avoid div by zero
        self._eps = 1e-8

    # --------------------- core transforms ---------------------
    def _transform(self, xyz: torch.Tensor, use_inv: bool) -> torch.Tensor:
        """Apply chosen extrinsic: LiDAR->Camera = (R|t) * [x;y;z;1]."""
        T = self.T_inv if use_inv else self.T_nom
        N = xyz.shape[0]
        ones = torch.ones((N, 1), device=xyz.device, dtype=xyz.dtype)
        homog = torch.cat([xyz, ones], dim=1)   # (N,4)
        cam = (homog @ T.T)[:, :3]              # (N,3)
        return cam

    def _project_xy(self, Xc: torch.Tensor):
        """Project camera-frame 3D points to pixel coords (float)."""
        z = Xc[:, 2]
        x = Xc[:, 0]
        y = Xc[:, 1]
        # avoid div by zero; keep sign for z>0 test
        z_safe = torch.where(z.abs() > self._eps, z, torch.sign(z) * self._eps + (z == 0) * self._eps)
        u = self.fx * (x / z_safe) + self.cx
        v = self.fy * (y / z_safe) + self.cy
        return u, v, z

    def _auto_choose_orientation(self, xyz: torch.Tensor):
        """Pick T vs T^-1 by which yields more (in-image & z>0) for a subset (first frame)."""
        if self._checked_orientation:
            return
        with torch.no_grad():
            subset = xyz
            if subset.shape[0] > 5000:
                step = max(1, subset.shape[0] // 5000)
                subset = subset[::step]

            # Try nominal
            Xc_nom = self._transform(subset, use_inv=False)
            u_nom, v_nom, z_nom = self._project_xy(Xc_nom)
            in_nom = (u_nom >= 0) & (u_nom < self.img_w) & (v_nom >= 0) & (v_nom < self.img_h) & (z_nom > 0)
            score_nom = int(in_nom.sum().item())

            # Try inverse
            Xc_inv = self._transform(subset, use_inv=True)
            u_inv, v_inv, z_inv = self._project_xy(Xc_inv)
            in_inv = (u_inv >= 0) & (u_inv < self.img_w) & (v_inv >= 0) & (v_inv < self.img_h) & (z_inv > 0)
            score_inv = int(in_inv.sum().item())

            self._use_inv = score_inv > score_nom
            self._checked_orientation = True
            # Optional debug:
            # print(f"[Projector] orientation = {'T_inv' if self._use_inv else 'T_nom'} "
            #       f"(scores: nom={score_nom}, inv={score_inv})")

    # --------------------- public API ---------------------
    @torch.no_grad()
    def project_points(self, lidar_xyz: torch.Tensor):
        """
        Returns:
          uvs: (N,2) pixel coords (float). Behind-camera points are set to (-1, -1).
          z_cam: (N,) depth in camera Z (can be negative).
          valid_mask: (N,) True if in-image & z>0 (NO occlusion applied here).
        """
        xyz = lidar_xyz.to(device=self.device, dtype=torch.float32)

        # Choose orientation on first call
        if not self._checked_orientation:
            self._auto_choose_orientation(xyz)

        Xc = self._transform(xyz, use_inv=self._use_inv)
        u, v, z = self._project_xy(Xc)

        # Safe UV init: mark all as invalid by default
        uvs = torch.full((xyz.shape[0], 2), -1.0, dtype=torch.float32, device=self.device)
        in_front = z > 0
        if in_front.any():
            uvs[in_front, 0] = u[in_front]
            uvs[in_front, 1] = v[in_front]

        in_img = (uvs[:, 0] >= 0) & (uvs[:, 0] < self.img_w) & (uvs[:, 1] >= 0) & (uvs[:, 1] < self.img_h)
        valid = in_front & in_img

        return uvs, z, valid

    # --------------------- feature assignment ---------------------
    def _token_indices_from_uv(self, u: torch.Tensor, v: torch.Tensor):
        """Map pixel coords (float) to nearest-token integer indices (i,j) on (Hf,Wf)."""
        j = torch.clamp((u / self.patch_w).floor().to(torch.int64), 0, self.Wf - 1)  # x-index (cols)
        i = torch.clamp((v / self.patch_h).floor().to(torch.int64), 0, self.Hf - 1)  # y-index (rows)
        return i, j

    def _sample_bilinear(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
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

    @torch.no_grad()
    def sample_features_simple(
        self,
        lidar_xyz: torch.Tensor,
        *,
        bilinear: bool = False,
    ):
        """
        Assign features ignoring occlusion (used as a fallback).
        Returns:
          point_feats: (N,D) float32
          mask_img_zpos: (N,) bool (in-image & z>0)
        """
        N = lidar_xyz.shape[0]
        device = lidar_xyz.device
        point_feats = torch.zeros((N, self.D), dtype=torch.float32, device=device)

        uvs, z_cam, valid = self.project_points(lidar_xyz)
        if not valid.any():
            return point_feats, valid  # all zeros

        idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
        u = uvs[idx, 0]; v = uvs[idx, 1]

        if bilinear:
            feats_sel = self._sample_bilinear(u, v)
        else:
            i, j = self._token_indices_from_uv(u, v)
            feats_sel = self.features[i, j]

        point_feats[idx] = feats_sel.to(point_feats.dtype)
        return point_feats, valid

    @torch.no_grad()
    def assign_features(
        self,
        lidar_xyz: torch.Tensor,
        *,
        bilinear: bool = False,
        occlusion_eps: Union[float, torch.Tensor] = 0.05,
    ):
        """
        Assign one token feature per 3D point with z-buffer occlusion at token resolution.

        Returns:
          point_feats: (N,D) float32
          vis_mask:    (N,) bool (True = kept)
        """
        device = lidar_xyz.device
        N = lidar_xyz.shape[0]
        point_feats = torch.zeros((N, self.D), dtype=torch.float32, device=device)

        # Project & get basic validity (in-image & z>0)
        uvs, z_cam, valid = self.project_points(lidar_xyz)
        if not valid.any():
            return point_feats, valid  # nothing to keep

        # Map to tokens
        u = uvs[:, 0]; v = uvs[:, 1]
        i_all, j_all = self._token_indices_from_uv(u, v)
        tok_all = i_all * self.Wf + j_all

        # Restrict to valid points when building z-buffer
        tok_v = tok_all[valid]
        z_v = z_cam[valid]

        # Build per-token min-depth z-buffer
        zbuf = torch.full((self.Hf * self.Wf,), float("inf"), device=device, dtype=torch.float32)
        try:
            # In-place scatter reduce (torch >= 2.0)
            zbuf.scatter_reduce_(0, tok_v, z_v, reduce="amin", include_self=True)
        except Exception:
            # CPU/NumPy fallback
            tok_cpu = tok_v.detach().cpu().numpy()
            z_cpu = z_v.detach().cpu().numpy().astype(np.float32)
            arr = np.full((self.Hf * self.Wf,), np.inf, dtype=np.float32)
            np.minimum.at(arr, tok_cpu, z_cpu)
            zbuf = torch.from_numpy(arr).to(device=device, dtype=torch.float32)

        # Per-point epsilon (scalar or vector)
        if torch.is_tensor(occlusion_eps):
            eps_all = occlusion_eps.to(device=device, dtype=torch.float32).view(-1)
            if eps_all.numel() != N:
                raise ValueError(f"occlusion_eps tensor must be shape (N,), got {tuple(occlusion_eps.shape)}")
            eps_v = eps_all[valid]
        else:
            eps_v = torch.full((int(valid.sum().item()),), float(occlusion_eps), device=device, dtype=torch.float32)

        # Keep those near the token's minimum depth
        keep_v = z_v <= (zbuf.index_select(0, tok_v) + eps_v)

        # Build full vis mask
        vis_mask = torch.zeros((N,), dtype=torch.bool, device=device)
        vis_mask[valid] = keep_v

        # Assign features to kept points
        if vis_mask.any().item():
            idx = torch.nonzero(vis_mask, as_tuple=False).squeeze(1)
            u_keep = u[idx]; v_keep = v[idx]
            if bilinear:
                feats_sel = self._sample_bilinear(u_keep, v_keep)
            else:
                i_k, j_k = self._token_indices_from_uv(u_keep, v_keep)
                feats_sel = self.features[i_k, j_k]
            point_feats[idx] = feats_sel.to(point_feats.dtype)

        return point_feats, vis_mask
