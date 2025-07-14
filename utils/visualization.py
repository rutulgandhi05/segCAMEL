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

def visualize_dino_semantic_image(dino_feat_tensor, feature_map_size, save_path=None):
    """
    Visualize DINO patch features as a 'semantic' image using PCA for RGB mapping.
    """
    num_patches, feat_dim = dino_feat_tensor.shape
    grid_w, grid_h = feature_map_size  # (w, h)
    pca = PCA(n_components=3)
    feat_rgb = pca.fit_transform(dino_feat_tensor)
    feat_rgb = (feat_rgb - feat_rgb.min()) / (feat_rgb.max() - feat_rgb.min())
    feat_rgb = feat_rgb.reshape(grid_h, grid_w, 3)  # (h, w, 3) for imshow

    plt.figure(figsize=(8, 6))
    plt.title("DINO Patchwise Semantic (PCA RGB)")
    plt.imshow(feat_rgb)
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Saved DINO semantic image to {save_path}")
    plt.show()
    plt.close()

def visualize_patch_3d_pointcloud(pts3d_patches, dino_feat_tensor, save_path=None, use_open3d=False):
    """
    Visualize DUSt3R 3D patch centers colored by DINO features (via PCA).
    """
    pca = PCA(n_components=3)
    feat_rgb = pca.fit_transform(dino_feat_tensor)
    feat_rgb = (feat_rgb - feat_rgb.min()) / (feat_rgb.max() - feat_rgb.min())
    if use_open3d and o3d is not None:
        PointCloudVisualizer.visualize_open3d(pts3d_patches, feat_rgb)
    else:
        PointCloudVisualizer.plot_pointcloud(pts3d_patches, feat_rgb, title="DUSt3R Patch Centers (PCA RGB)", save_path=save_path)

def visualize_lidar_semantics(lidar_xyz, assigned_feats, save_path=None, use_open3d=False):
    """
    Visualize LiDAR point cloud colored by assigned DINO features (via PCA).
    """
    pca = PCA(n_components=3)
    assigned_feat_rgb = pca.fit_transform(assigned_feats)
    assigned_feat_rgb = (assigned_feat_rgb - assigned_feat_rgb.min()) / (assigned_feat_rgb.max() - assigned_feat_rgb.min())
    if use_open3d and o3d is not None:
        PointCloudVisualizer.visualize_open3d(lidar_xyz, assigned_feat_rgb)
    else:
        PointCloudVisualizer.plot_pointcloud(lidar_xyz, assigned_feat_rgb, title="LiDAR Points (Assigned DINO Feature RGB)", save_path=save_path)


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