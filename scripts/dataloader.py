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
            root_dir (str or Path): Dataset folder.
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
        # Choose which stereo image to use.  The loader guarantees that at
        # least one of left_image/right_image is not None.
        img_path = sample["right_image"] if self.use_right_image else sample["left_image"]
        # Load and convert to RGB; resize to the target resolution expected by the
        # transform factory
        with Image.open(img_path).convert("RGB") as img:
            # Many vision models expect specific input resolutions.  Resize
            # according to the factory provided by the caller.
            # If no factory is provided, images are returned at their original size.
            if self.transform_factory is not None:
                # Let the factory inspect the raw image size
                transform = self.transform_factory.get_transform((img.width, img.height))
                # Resize to the configured size (e.g. 672x378) using Lanczos for
                # high‑quality downsampling
                img = img.resize((transform.resize_size[0], transform.resize_size[1]), Image.LANCZOS)
                image_tensor = transform.transform(img)
                input_size = transform.resize_size
                feature_map_size = transform.feature_map_size
            else:
                # No transformation factory; use default PILToTensor
                transform = transforms.PILToTensor()
                image_tensor = transform(img)
                input_size = (img.width, img.height)
                feature_map_size = (img.width, img.height)
        

        pointcloud = sample["pointcloud"]
        pointcloud_tensor = torch.from_numpy(pointcloud)
        intrinsics = sample["stereo_right_intrinsics"] if self.use_right_image else sample["stereo_left_intrinsics"]
        # Scale intrinsics to match the resized image
        scaled_intrinsics = scale_intrinsics(intrinsics, (img.width, img.height), input_size)
        extrinsics = (
            sample["lidar_to_stereo_right_extrinsic"]
            if self.use_right_image
            else sample["lidar_to_stereo_left_extrinsic"]
        )

        return {
            "pointcloud": pointcloud_tensor,
            "image_tensor": image_tensor,
            "timestamps": sample["timestamps"],
            "intrinsics": torch.from_numpy(scaled_intrinsics),
            "extrinsics": torch.from_numpy(extrinsics),
            "input_size": input_size,
            "feature_map_size": feature_map_size
        }