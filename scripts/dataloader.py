from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from scripts.dataset import load_scantinel_dataset_folder, load_hercules_dataset_folder
from PIL import Image
from utils.misc import scale_intrinsics
import torch

class ScantinelDataset(Dataset):
    def __init__(self, root_dir):
        """
        PyTorch Dataset wrapper for Scantinel FMCW LiDAR + Camera data.

        Args:
            root_dir (str or Path): Dataset folder.
        """
        self.root_dir = Path(root_dir)
        self.samples = load_scantinel_dataset_folder(self.root_dir)
        self.transform = transforms.PILToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with Image.open(sample["image"]) as img:
            image_tensor = self.transform(img.convert("RGB"))
        pointcloud = sample["pointcloud"]

        return {
            "pointcloud": pointcloud, 
            "image_tensor": image_tensor,         # torch.Tensor, shape [3, H, W]
            "intrinsics": sample["intrinsics"],   # np.ndarray, shape [3, 3]
            "timestamps": sample["timestamps"]    # list (lidar_ts, cam_ts)
        }

class HerculesDataset(Dataset):
    def __init__(self, root_dir, transform_factory=None, max_workers=None,
                 use_right_image=True, return_all_fields=True):
        """
        PyTorch Dataset wrapper for Hercules FMCW LiDAR + Camera data.

        Args:
            root_dir: sequence folder path
            transform_factory: TransformFactory from your Extractor
            max_workers: passed to loader
            use_right_image: prefer right view; fallback to left if right missing
            return_all_fields: ask loader to return XYZ+features (reflectivity, velocity, [intensity])
        """
        self.root_dir = Path(root_dir)
        print(f"Loading Hercules dataset from {self.root_dir}")
        self.samples = load_hercules_dataset_folder(self.root_dir, return_all_fields=return_all_fields, max_workers=max_workers)
        self.transform_factory = transform_factory
        self.use_right_image = use_right_image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # pointcloud -> torch.float32
        pointcloud_tensor = torch.from_numpy(sample["pointcloud"]).to(torch.float32)

        right_path = sample.get("right_image")
        left_path = sample.get("left_image")
        used_camera = "right" if self.use_right_image and right_path is not None else "left"
        img_path = right_path if (used_camera == "right") else left_path

        if img_path is None:
            raise FileNotFoundError(f"No image found for sample index {idx} (left/right both None)")

        img = Image.open(img_path).convert("RGB")
        orig_size = img.size  # (W, H)

        if self.transform_factory is not None:
            tf = self.transform_factory.get_transform(orig_size)
            image_tensor = tf.transform(img)  # [C,H,W] torch.Tensor
            input_size = tf.resize_size  # (W, H)
            feature_map_size = tf.feature_map_size  # (Wf, Hf)
        else:
            image_tensor = self.fallback_transform(img)
            input_size = img.size
            feature_map_size = (input_size[0] // 14, input_size[1] // 14)

        img.close()

        if used_camera == "right":
            intrinsics = sample["stereo_right_intrinsics"]
            extrinsics = sample["lidar_to_stereo_right_extrinsic"]
        else:
            intrinsics = sample["stereo_left_intrinsics"]
            extrinsics = sample["lidar_to_stereo_left_extrinsic"]

        intrinsics = torch.from_numpy(intrinsics).to(torch.float32)
        extrinsics = torch.from_numpy(extrinsics).to(torch.float32)
        scaled_intrinsics = torch.from_numpy(scale_intrinsics(intrinsics.numpy(), orig_size, input_size)).to(torch.float32)

        try:
            image_relpath = str(Path(img_path).relative_to(self.root_dir))
        except Exception:
            image_relpath = str(img_path)

        return {
            "pointcloud": pointcloud_tensor,
            "image_tensor": image_tensor,
            "timestamps": sample["timestamps"],
            "intrinsics": scaled_intrinsics,
            "extrinsics": extrinsics,
            "input_size": input_size,
            "feature_map_size": feature_map_size,
            "image_relpath": image_relpath,
            "used_camera": used_camera,
        }