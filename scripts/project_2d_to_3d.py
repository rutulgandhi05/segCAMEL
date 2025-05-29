import numpy as np
from typing import Tuple

class LabelProjector:
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

    def project_points(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project Nx3 LiDAR points into pixel coordinates.
        Returns:
          - uv: (N,2) float array of (u, v) locations
          - valid: (N,) boolean mask of points with positive depth
        """
        # to homogeneous
        pts_h = np.hstack([points, np.ones((points.shape[0],1), dtype=points.dtype)])
        # transform to camera frame
        cam_pts = pts_h @ self.T.T            # (N,4)
        valid = cam_pts[:,2] > 0
        # perspective project
        uv_h = cam_pts[:,:3] @ self.K.T       # (N,3)
        uv = uv_h[:, :2] / uv_h[:, 2:3]
        return uv, valid

    def get_point_labels(self,
                         points: np.ndarray,
                         mask: np.ndarray   # H×W array with class IDs or 0/1
                         ) -> np.ndarray:
        """
        For each point, project to image, sample the mask, and return per-point labels.
        Invalid or out-of-bounds points get label = -1.
        """
        uv, valid = self.project_points(points)
        labels = -1 * np.ones(points.shape[0], dtype=np.int32)

        # round to nearest pixel
        u = np.round(uv[:,0]).astype(int)
        v = np.round(uv[:,1]).astype(int)

        # check bounds & depth
        in_bounds = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H) & valid
        labels[in_bounds] = mask[v[in_bounds], u[in_bounds]]

        return labels


def project_points_to_dino_patches(uv, valid, img_shape, dino_shape):
    """
    Map pixel coordinates (uv) in image space (HxW) to DINO patch grid (Hf x Wf).
    Returns patch indices for valid points, else -1.
    """
    N = uv.shape[0]
    H, W = img_shape
    Hf, Wf = dino_shape

    patch_u = np.clip((uv[:,0] / W * Wf).astype(int), 0, Wf-1)
    patch_v = np.clip((uv[:,1] / H * Hf).astype(int), 0, Hf-1)
    patch_idx = patch_v * Wf + patch_u

    patch_idx[~valid] = -1   # set to -1 for invalid projections

    return patch_idx  # shape (N,)
