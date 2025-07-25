from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from scripts.dataset import load_scantinel_dataset_folder, load_hercules_dataset_folder
from PIL import Image

class ScantinelDataset(Dataset):
    def __init__(self, root_dir):
        """
        PyTorch Dataset wrapper for Scantinel FMCW LiDAR + Camera data.

        Args:
            root_dir (str or Path): Dataset folder.
        """
        self.root_dir = Path(root_dir)
        self.samples = load_scantinel_dataset_folder(self.root_dir)
        self.transform = transforms.Compose([ 
            #transforms.Resize((512, 512)),  # Resize to 224x224
            transforms.PILToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")  # Convert to RGB format
        pointcloud = sample["pointcloud"]  # shape: (N, 6)
        image_tensor = self.transform(image)
        image.close()  # Close the image file to free resources


        return {
            "pointcloud": pointcloud, 
            "image_tensor": image_tensor,         # torch.Tensor, shape [3, H, W]
            "intrinsics": sample["intrinsics"],   # np.ndarray, shape [3, 3]
            "timestamps": sample["timestamps"]    # list (lidar_ts, cam_ts)
        }

class HerculesDataset(Dataset):
    def __init__(self, root_dir):
        """
        PyTorch Dataset wrapper for Hercules FMCW LiDAR + Camera data.

        Args:
            root_dir (str or Path): Dataset folder.
        """
        self.root_dir = Path(root_dir)
        self.samples = load_hercules_dataset_folder(self.root_dir, return_all_fields=True)
        self.transform = transforms.Compose([
            transforms.PILToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["right_image"]).convert("RGB")
        pointcloud = sample["pointcloud"]
        image_tensor = self.transform(image)
        image.close()
        
        return {
            "pointcloud": pointcloud,
            "image_tensor": image_tensor,
            "timestamps": sample["timestamps"],
            "intrinsics": sample["stereo_right_intrinsics"],
            "extrinsics": sample["lidar_to_stereo_right_extrinsic"]
        }