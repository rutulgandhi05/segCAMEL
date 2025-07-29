import os

from tqdm import tqdm
from pathlib import Path
from utils.misc import setup_logger
from models.PTv3.model import PointTransformerV3

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: Path):
        self.files = sorted(list(Path(root_dir).glob("*.pth")))
        assert len(self.files), f"No .pth files found in {root_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx])
        return {
            "coord": sample["coord"],
            "feat": sample["feat"],
            "dino_feat": sample["dino_feat"],
            "grid_size": float(sample.get("grid_size", 0.05))
        }

def distillation_loss(pred_feat, target_feat):
    pred = F.normalize(pred_feat, dim=1)
    target = F.normalize(target_feat, dim=1)
    return 1 - (pred * target).sum(dim=1).mean()

def safe_grid_coord(coord, grid_size, logger=None):
    if not torch.is_tensor(grid_size):
        grid_size = torch.tensor(grid_size, device=coord.device)
    else:
        grid_size = grid_size.to(coord.device)

    coord_min = coord.min(0)[0]
    grid_coord = ((coord - coord_min) / grid_size).floor().int()

    for axis in range(3):
        n_unique = len(torch.unique(grid_coord[:, axis]))
        if n_unique < 2:
            if logger:
                logger.warning(f"Axis {axis} of grid_coord has only {n_unique} unique value! Forcing two values.")
            grid_coord[:, axis] += torch.arange(grid_coord.shape[0], device=grid_coord.device) % 2

    return grid_coord

def train(
    data_dir=Path,
    epochs=20,
    batch_size=1,
    lr=1e-3,
    save_path=Path,
    device="cuda" if torch.cuda.is_available() else "cpu",
    input_mode="dino_only",   # Options: 'dino_only', 'vri_dino', 'coord_dino', 'coord_vri_dino'
):
    logger = setup_logger()
    print(f"Starting training on device: {device}")
    print(f"Training data: {data_dir}")
    print(f"Input mode: {input_mode}")

    dataset = PointCloudDataset(data_dir)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True,
                            persistent_workers=True
                            )

    print(f"Loaded {len(dataset)} preprocessed samples")

    # --- Infer input_dim robustly based on input_mode and first sample
    sample = dataset[0]
    coord = sample["coord"].to(device)
    feat = sample["feat"].to(device)
    dino_feat = sample["dino_feat"].to(device)

    print("feat.shape:", feat.dim())
    print("dino_feat.shape:", dino_feat.dim())

    if input_mode == "dino_only":
        input_feat = dino_feat
    elif input_mode == "vri_dino":
        input_feat = torch.cat([feat, dino_feat], dim=1)
    elif input_mode == "coord_dino":
        input_feat = torch.cat([coord, dino_feat], dim=1)
    elif input_mode == "coord_vri_dino":
        input_feat = torch.cat([coord, feat, dino_feat], dim=1)
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")

    input_dim = input_feat.shape[1]
    dino_dim = dino_feat.shape[1]
    print(f"Using input_dim={input_dim}, dino_dim={dino_dim}")
    
    model = PointTransformerV3(in_channels=input_dim).to(device)
    proj_head = torch.nn.Linear(model.out_channels, dino_dim).to(device) if hasattr(model, "out_channels") else torch.nn.Linear(64, dino_dim).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(proj_head.parameters()), lr=lr)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        proj_head.train()
        total_loss = 0
        print(f"\nEpoch {epoch + 1}/{epochs}")

        for batch_idx, samples in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            try:
                if isinstance(samples, list) or isinstance(samples, tuple):
                    samples = samples[0]

                coord = samples["coord"].to(device)
                feat = samples["feat"].to(device)
                dino_feat = samples["dino_feat"].to(device)
                grid_size = samples["grid_size"]

                if not (0.01 <= grid_size <= 1.0):
                    grid_size = 0.05

                grid_coord = safe_grid_coord(coord, grid_size, logger=logger)

                if grid_coord.max() > 2**15:
                    logger.warning(f"Grid coordinate overflow (max={grid_coord.max()}), using coarser grid_size")
                    grid_size = (coord.max(0)[0] - coord.min(0)[0]).max().item() / 10000
                    grid_coord = safe_grid_coord(coord, grid_size, logger=logger)

                num_points = coord.shape[0]
                batch_tensor  = torch.zeros(num_points, dtype=torch.long, device=device)
                offset = torch.tensor([num_points], dtype=torch.long, device=device)

                if input_mode == "dino_only":
                    input_feat = dino_feat
                elif input_mode == "vri_dino":
                    input_feat = torch.cat([feat, dino_feat], dim=1)
                elif input_mode == "coord_dino":
                    input_feat = torch.cat([coord, dino_feat], dim=1)
                elif input_mode == "coord_vri_dino":
                    input_feat = torch.cat([coord, feat, dino_feat], dim=1)
                
                # Debug info (first batch, first epoch)
                if epoch == 0 and batch_idx == 0:
                    print(f"Grid size used: {grid_size}")
                    print(f"Coord shape: {coord.shape}")
                    print(f"Input feat shape: {input_feat.shape}")
                    print(f"Grid coord shape: {grid_coord.shape}")
                    print(f"Offset: {offset}")
                    print(f"Batch shape: {batch_tensor.shape}")


                data_dict = {
                    "coord": coord,
                    "feat": input_feat,
                    "grid_coord": grid_coord,
                    "grid_size": grid_size,
                    "offset": offset,
                    "batch": batch_tensor ,
                }

                optimizer.zero_grad()
                output = model(data_dict)
                pred = output.feat
                pred_proj = proj_head(pred)

                valid_mask = dino_feat.abs().sum(dim=1) > 1e-6
                loss = distillation_loss(pred_proj[valid_mask], dino_feat[valid_mask])
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                raise e

        avg_loss = total_loss / len(dataloader)
        print(f"Avg Loss = {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "proj_head": proj_head.state_dict()
            }, save_path)
            print(f"Best model saved to {save_path} (loss = {avg_loss:.6f})")

    print("Training complete.")

if __name__ == "__main__":
    dataset_env = os.getenv("HERCULES_DATASET")
    if dataset_env is None:
        raise EnvironmentError("HERCULES_DATASET environment variable not set.")

    data_dir = Path(dataset_env) / "Mountain_01_Day" / "processed_data"
    train(
        data_dir=data_dir,
        epochs=20,
        batch_size=1,
        lr=1e-3,
        save_path=Path(dataset_env) / "checkpoints" / "best_model_hercules_MAD1_vrid.pth",
        input_mode="vri_dino"
    )
