import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

try:
    import open3d as o3d
except ImportError:
    o3d = None

from sklearn.decomposition import PCA

class PointCloudVisualizer:
    @staticmethod
    def plot_pointcloud(points, colors=None, title="Point Cloud", show=True, save_path=None):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        if colors is not None:
            colors = np.asarray(colors)
            if len(colors.shape) == 1:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1, cmap="tab20")
            elif len(colors.shape) == 2 and colors.shape[1] == 3:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
            else:
                raise ValueError("Colors must be [N] (labels) or [N,3] (RGB)")
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")
        if show:
            plt.show()
        plt.close(fig)

    @staticmethod
    def visualize_open3d(points, colors=None):
        if o3d is None:
            raise ImportError("Open3D not installed. Install with 'pip install open3d'")
        points = np.asarray(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            colors = np.asarray(colors)
            if len(colors.shape) == 2 and colors.shape[1] == 3:
                if colors.max() > 1.0:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
            else:
                raise ValueError("Colors must be [N,3] (RGB)")
        o3d.visualization.draw_geometries([pcd])




def visualize_pca_colored_pointcloud(xyz: np.ndarray, features: np.ndarray, mask: np.ndarray):
    # Reduce feature dimension to 3D for RGB coloring
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(features)

    # Normalize to 0-1 for coloring
    colors = (reduced - reduced.min(axis=0)) / (reduced.max(axis=0) - reduced.min(axis=0) + 1e-6)
    colors[mask == 0] = [0.5, 0.5, 0.5]  # grey out invisible points

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([pcd])



def visualize_pca_colored_inference(coord: np.ndarray, features: np.ndarray):
    # PCA to 3D for RGB color
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(features)

    # Normalize for visualization
    colors = (reduced - reduced.min(axis=0)) / (reduced.max(axis=0) - reduced.min(axis=0) + 1e-6)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd])