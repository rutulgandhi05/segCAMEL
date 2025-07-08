# visualizer.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

try:
    import open3d as o3d
except ImportError:
    o3d = None  # Open3D is optional

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


def create_video_from_frames(
    tmpdir: str, output_path: str, framerate: float = 30
) -> None:
    """Create video from frame images using ffmpeg."""
    output_path = Path(output_path)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        
        import cv2

        images = sorted(Path(tmpdir).glob("*.jpg"))
        frame = cv2.imread(str(images[0]))
        height, width, layers = frame.shape

        # Video writer to create .avi file
        video = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'DIVX'), framerate, (width, height))
        # Appending images to video
        for image in images:
            video.write(cv2.imread(str(image)))

        # Release the video file
        video.release()
        cv2.destroyAllWindows()
        print("Video generated successfully!")

        print(f"Saved visualization to {output_path}")
    
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred while creating video: {e}"
        ) from e
    

def visualize_3d_to_2d_projection(
    image: np.ndarray,          # shape (H, W, 3), RGB, uint8 or float
    points_3d: torch.Tensor,    # (N, 3), world coordinates
    projector,                  # Your Projector instance (with scaled K)
    marker_color='r',           # color for 2D points
    marker_size=2
):
    """
    Projects 3D points onto the image and visualizes the result.
    Only visible points are shown.
    image: numpy array of shape (H, W, 3) or (H, W) for grayscale
    points_3d: torch.Tensor of shape (N, 3) with 3D coordinates in world space
    projector: instance of your Projector class with project_points and get_visibility_mask methods
    marker_color: color for the projected points in the image
    marker_size: size of the markers in the image

    Returns:
        None: Displays the image with projected points.
        
    """
    # Project 3D points to 2D
    pixel_coords, depth = projector.project_points(points_3d)
    visible = projector.get_visibility_mask(pixel_coords, depth)
    pixel_coords_vis = pixel_coords[visible].cpu().numpy()

    # Prepare image
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().numpy()
        if img_np.shape[0] == 3:  # (3, H, W)
            img_np = np.transpose(img_np, (1, 2, 0))
        img_vis = img_np
    else:
        img_vis = image

    # If float, scale to 0-1 for matplotlib
    if img_vis.dtype != np.uint8:
        img_vis = np.clip(img_vis, 0, 1)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_vis)
    plt.scatter(pixel_coords_vis[:, 0], pixel_coords_vis[:, 1],
                c=marker_color, s=marker_size, alpha=0.7, label='Projected 3D Points')
    plt.axis('off')
    plt.title(f"2D projection of {pixel_coords_vis.shape[0]} 3D points")
    plt.legend()
    plt.show()


def visualize_timestamps_sync(pcl_stamps, cam_stamps, pairings):
    """
    pcl_stamps: list/array of LiDAR timestamps
    cam_stamps: list/array of camera timestamps
    pairings: list of (pcl_stamp, closest_cam_stamp) pairs
    """
    # y-positions: LiDAR on y=1, Cam on y=0
    y_pcl = 1
    y_cam = 0

    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot all LiDAR points (blue)
    ax.scatter(pcl_stamps, [y_pcl]*len(pcl_stamps), color='blue', label='LiDAR Points', zorder=2)
    # Plot all Cam points (green)
    ax.scatter(cam_stamps, [y_cam]*len(cam_stamps), color='green', label='Camera Frames', zorder=2)

    # Draw arrows for paired matches (red)
    for pcl_ts, cam_ts in pairings:
        ax.annotate(
            '', xy=(cam_ts, y_cam), xytext=(pcl_ts, y_pcl),
            arrowprops=dict(arrowstyle="->", color='red', lw=1.5), zorder=1
        )

    # Add some styling
    ax.set_yticks([y_cam, y_pcl])
    ax.set_yticklabels(['Camera', 'LiDAR'])
    ax.set_xlabel("Timestamp")
    ax.set_title("Time Synchronization: LiDAR ↔ Closest Camera Frames")
    ax.legend(loc='upper right')
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()