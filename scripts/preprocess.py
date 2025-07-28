import torch
from pathlib import Path
from tqdm import tqdm

from scripts.feature_extractor import Extractor
from scripts.dataloader import HerculesDataset
from torch.utils.data import DataLoader
from utils.misc import setup_logger
from scripts.project_2d_to_3d import LidarToImageProjector

logger = setup_logger("preprocess")


def custom_collate(batch):
    # Custom collate for variable-size fields (pointcloud etc.)
    collated = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        
        if isinstance(values[0], torch.Tensor) and all(v.shape == values[0].shape for v in values):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    return collated

def preprocess_and_save_hercules(
    root_dir,
    save_dir,
    device="cuda" if torch.cuda.is_available() else "cpu",
    workers=8
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    extractor = Extractor()
    dataset = HerculesDataset(root_dir, transform=extractor.transform_factory)  # Use extractor's transform
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=custom_collate)  # Try batch_size=4 or more if GPU fits!

    print(f"Using DINO model: {extractor.dino_model}")

    for idx, batch in enumerate(tqdm(dataloader, desc="Processing frames")):
        # --- Batch processing ---
        image_tensors = batch["image_tensor"]  # [B, C, H, W]
        if isinstance(image_tensors, list):
            image_tensors = torch.stack(image_tensors, dim=0)
        batch_size = image_tensors.shape[0]

        image_tensors = image_tensors.to(device)
        with torch.no_grad():
            features_batch = extractor.extract_dino_features(image_tensors)

        for i in range(batch_size):
            features = features_batch[i]
            dino_feat_tensor = features.flat().tensor.to(device)  # torch.Tensor

            # --- Pointcloud Handling ---
            pointcloud = batch["pointcloud"][i].to(device)
            lidar_xyz = pointcloud[:, :3].to(device)
            lidar_feats = pointcloud[:, 3:] if pointcloud.shape[1] > 3 else torch.zeros((lidar_xyz.shape[0], 1), dtype=pointcloud.dtype)
            lidar_feats = lidar_feats.to(device)

            # --- Intrinsics/Extrinsics Handling ---
            intrinsics = batch["intrinsics"][i].to(device)
            #if isinstance(intrinsics, torch.Tensor): intrinsics = intrinsics.squeeze().cpu().numpy()
            extrinsics = batch["extrinsics"][i].to(device)
            #if isinstance(extrinsics, torch.Tensor): extrinsics = extrinsics.squeeze().cpu().numpy()

            # --- Projector & Feature Assignment ---
            projector = LidarToImageProjector(
                intrinsic=intrinsics,
                extrinsic=extrinsics,
                image_size=batch["input_size"][i],
                feature_map_size=batch["feature_map_size"][i],
                patch_features=dino_feat_tensor
            )
            assigned_feats, mask = projector.assign_features(lidar_xyz.to(torch.float32))
            assigned_feats = assigned_feats.float().cpu()

            grid_size_manual = 0.05

            save_data = {
                "coord": lidar_xyz.float().cpu(),      # [N, 3]
                "feat": lidar_feats.float().cpu(),     # [N, F]
                "dino_feat": assigned_feats,           # [N, D]
                "grid_size": torch.tensor(grid_size_manual, dtype=torch.float32),
                "mask": mask.cpu(),  # [N]
                "image_tensor": image_tensors[i].cpu(),  # [C, H, W]
            }
            frame_idx = idx * batch_size + i
            save_path = save_dir / f"frame_{frame_idx:05d}.pth"
            torch.save(save_data, save_path)
            if frame_idx % 10 == 0 or frame_idx == (len(dataset)-1):
                print(f"[Saved] {save_path}")

if __name__ == "__main__":
    import time

    start_time = time.time()
    print("Starting preprocessing...")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    HERCULES_ROOT_DIR = "data/hercules/Mountain_01_Day/"
    HERCULES_SAVE_DIR = "data/hercules/Mountain_01_Day/processed_data"
    preprocess_and_save_hercules(HERCULES_ROOT_DIR, HERCULES_SAVE_DIR, workers=8)
    end_time = time.time()
    print("Preprocessing completed.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")