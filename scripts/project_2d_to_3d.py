import numpy as np
import cv2

class LabelProjector:
    """
    Projects 2D mask labels onto 3D point clouds using camera intrinsics and extrinsics.
    Designed to be dataset-agnostic: pass calibration and mapping as arguments.
    """
    def __init__(self, intrinsic, extrinsic=None, image_shape=None):
        """
        Args:
            intrinsic: (3, 3) numpy array, camera intrinsic matrix
            extrinsic: (4, 4) numpy array, camera extrinsic matrix (world-to-camera or lidar-to-camera)
            image_shape: (H, W) tuple, shape of the 2D mask/image
        """
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.image_shape = image_shape

    def project_points(self, points_3d):
        """
        Projects 3D points to 2D image plane.
        Args:
            points_3d: (N, 3) numpy array
        Returns:
            points_2d: (N, 2) numpy array of pixel coordinates
        """
        N = points_3d.shape[0]
        points_h = np.concatenate([points_3d, np.ones((N, 1))], axis=1)  # (N, 4)
        if self.extrinsic is not None:
            points_cam = (self.extrinsic @ points_h.T).T[:, :3]
        else:
            points_cam = points_3d
        # Project to image
        points_proj = (self.intrinsic @ points_cam.T).T  # (N, 3)
        points_proj = points_proj[:, :2] / points_proj[:, 2:3]
        return points_proj

    def get_point_labels(self, points_3d, mask_2d, default_label=255):
        """
        Assigns a 2D mask label to each 3D point by projection.
        Args:
            points_3d: (N, 3) numpy array
            mask_2d: (H, W) numpy array of int labels
            default_label: label to assign if point projects outside image
        Returns:
            labels: (N,) numpy array of int labels
        """
        points_2d = self.project_points(points_3d)
        H, W = mask_2d.shape if self.image_shape is None else self.image_shape
        labels = np.full(points_3d.shape[0], default_label, dtype=mask_2d.dtype)
        for i, (u, v) in enumerate(points_2d):
            u_int, v_int = int(round(u)), int(round(v))
            if 0 <= v_int < H and 0 <= u_int < W:
                labels[i] = mask_2d[v_int, u_int]
        return labels
