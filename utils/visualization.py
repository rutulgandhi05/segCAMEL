# visualizer.py

import numpy as np
import matplotlib.pyplot as plt

""" try:
    import open3d as o3d
except ImportError:
    o3d = None  """ # Open3D is optional

o3d = None
class PointCloudVisualizer:
    def __init__(self):
        pass

    @staticmethod
    def plot_pointcloud(
        points, colors=None, title="Point Cloud", show=True, save_path=None, cmap="tab20"
    ):
        """
        points: [N, 3] numpy array
        colors: [N], [N, 3], or None (labels or RGB)
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        # If coloring by label
        if colors is not None:
            colors = np.asarray(colors)
            if len(colors.shape) == 1:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1, cmap=cmap)
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
    def visualize_open3d(points, colors=None, cmap="tab20"):
        """
        points: [N, 3] numpy
        colors: [N] (labels) or [N,3] (RGB)
        """
        if o3d is None:
            raise ImportError("Open3D is not installed. Please install with 'pip install open3d'")
        points = np.asarray(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            colors = np.asarray(colors)
            if len(colors.shape) == 1:
                # Map label to color using matplotlib colormap
                color_map = plt.get_cmap(cmap)
                # Normalize labels to [0,1] for colormap
                normalized_labels = (colors % 20) / 20
                colors_rgb = color_map(normalized_labels)[:, :3]
                pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
            elif len(colors.shape) == 2 and colors.shape[1] == 3:
                # Already RGB
                if colors.max() > 1.0:
                    colors = colors / 255.0  # Scale if needed
                pcd.colors = o3d.utility.Vector3dVector(colors)
            else:
                raise ValueError("Colors must be [N] (labels) or [N,3] (RGB)")
        o3d.visualization.draw_geometries([pcd])

    def show_gt_pred(
        self, coords, gt_labels=None, pred_labels=None, dino_feat=None, save_prefix=None, use_open3d=False
    ):
        """
        Visualize ground truth and prediction for a point cloud.
        coords: [N,3] numpy
        gt_labels: [N] numpy, optional
        pred_labels: [N] numpy, optional
        dino_feat: [N] or [N, dino_channels] numpy, optional (visualizes norm or first channel)
        save_prefix: filename prefix if saving
        use_open3d: whether to use Open3D instead of Matplotlib
        """
        if gt_labels is not None:
            title = "Ground Truth Labels"
            path = f"{save_prefix}_gt.png" if save_prefix else None
            if use_open3d:
                self.visualize_open3d(coords, gt_labels)
            else:
                self.plot_pointcloud(coords, gt_labels, title, save_path=path)
        if pred_labels is not None:
            title = "Predicted Labels"
            path = f"{save_prefix}_pred.png" if save_prefix else None
            if use_open3d:
                self.visualize_open3d(coords, pred_labels)
            else:
                self.plot_pointcloud(coords, pred_labels, title, save_path=path)
        if dino_feat is not None:
            if len(dino_feat.shape) == 2:
                # Visualize the first channel and norm
                title = "DINO Feature 0"
                path = f"{save_prefix}_dino0.png" if save_prefix else None
                self.plot_pointcloud(coords, dino_feat[:, 0], title, save_path=path)
                dino_norm = np.linalg.norm(dino_feat, axis=1)
                title = "DINO Feature Norm"
                path = f"{save_prefix}_dinonorm.png" if save_prefix else None
                self.plot_pointcloud(coords, dino_norm, title, save_path=path)
            else:
                # Visualize directly
                title = "DINO Feature"
                path = f"{save_prefix}_dino.png" if save_prefix else None
                self.plot_pointcloud(coords, dino_feat, title, save_path=path)

    def show_rgb(self, coords, rgb, title="RGB Point Cloud", save_path=None, use_open3d=False):
        if use_open3d:
            self.visualize_open3d(coords, rgb)
        else:
            self.plot_pointcloud(coords, rgb, title, save_path=save_path)

# --------------- Example Usage ---------------

if __name__ == "__main__":
    import torch

    # Simulate one batch for demo
    N = 4096
    coords = np.random.rand(N, 3) * 10
    gt_labels = np.random.randint(0, 5, size=N)
    pred_labels = np.random.randint(0, 5, size=N)
    dino_feat = np.random.randn(N, 16)

    visualizer = PointCloudVisualizer()
    # Show GT and pred with Matplotlib
    visualizer.show_gt_pred(coords, gt_labels, pred_labels, dino_feat)

    # To use with a real batch from your loader (example):
    # coords = data_dict["coord"].cpu().numpy()
    # labels = labels.cpu().numpy()
    # preds = model(data_dict).argmax(dim=1).cpu().numpy()
    # dino_feat = data_dict["dino_feat"].cpu().numpy()
    # visualizer.show_gt_pred(coords, labels, preds, dino_feat)
