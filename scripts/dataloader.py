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
    def __init__(self, root_dir, transform_factory=None, max_workers=8):
        """
        PyTorch Dataset wrapper for Hercules FMCW LiDAR + Camera data.

        Args:
            root_dir (str or Path): Dataset folder.
        """
        self.root_dir = Path(root_dir)
        print(f"Loading Hercules dataset from {self.root_dir}")
        self.samples = load_hercules_dataset_folder(self.root_dir, return_all_fields=True, max_workers=max_workers)
        self.transform_factory = transform_factory

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with Image.open(sample["right_image"]).convert("RGB") as img:
            img = img.resize((672, 378), Image.LANCZOS)
            transform = self.transform_factory.get_transform((img.width, img.height))
            image_tensor = transform(img)

        pointcloud = sample["pointcloud"]
        input_size = transform.resize_size
        feature_map_size = transform.feature_map_size

        intrinsics = sample["stereo_right_intrinsics"]
        intrinsics = scale_intrinsics(
                intrinsics, 
                (img.width, img.height),  # (W, H)
                input_size
            )

        return {
            "pointcloud": torch.from_numpy(pointcloud),  # torch.Tensor, shape [N, 4]
            "image_tensor": image_tensor,
            "timestamps": sample["timestamps"],
            "intrinsics": torch.from_numpy(intrinsics),
            "extrinsics": torch.from_numpy(sample["lidar_to_stereo_right_extrinsic"]),
            "input_size": input_size,
            "feature_map_size": feature_map_size
        }