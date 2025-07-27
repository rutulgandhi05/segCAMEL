from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.PTv3.model import PointTransformerV3
from utils.misc import setup_logger

class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: Path):
        self.files = sorted(list(Path(root_dir).glob("*.pth")))
        assert len(self.files), f"No .pth files found in {root_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])

def distillation_loss(pred_feat, target_feat):
    pred = F.normalize(pred_feat, dim=1)
    target = F.normalize(target_feat, dim=1)
    return 1 - (pred * target).sum(dim=1).mean()

def train(
    data_dir=Path("data/hercules/Mountain_01_Day/processed_data"),
    epochs=20,
    batch_size=1,
    lr=1e-3,
    save_path=Path("data/checkpoints/best_model_hercules_exp_size100.pth"),
    device="cuda" if torch.cuda.is_available() else "cpu",
    input_mode="vri_dino",   # Options: 'dino_only', 'vri_dino', 'coord_dino', 'coord_vri_dino'
):
    logger = setup_logger()
    logger.info(f"Starting training on device: {device}")
    logger.info(f"Training data: {data_dir}")
    logger.info(f"Input mode: {input_mode}")

    dataset = PointCloudDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info(f"Loaded {len(dataset)} preprocessed samples")

    # --- Infer input_dim robustly based on input_mode and first sample
    sample = dataset[0]
    coord = sample["coord"]
    feat = sample["feat"]
    dino_feat = sample["dino_feat"]

    # Build input_feat as will be used in training
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
    logger.info(f"Using input_dim={input_dim}")
    model = PointTransformerV3(in_channels=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        for batch_idx, batch_sample in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            try:
                # Unpack batch_sample: if batch_size=1, it's a list of one dict
                if isinstance(batch_sample, list):
                    sample = batch_sample[0]
                else:
                    sample = batch_sample

                # Move all tensors to device
                coord = sample["coord"].to(device)
                feat = sample["feat"].to(device)
                dino_feat = sample["dino_feat"].to(device)

                # --- Robust grid size for outdoor LiDAR ---
                default_grid_size = 0.05  # 5 cm
                grid_size = float(sample.get("grid_size", default_grid_size))
                if not (0.01 <= grid_size <= 1.0):
                    grid_size = default_grid_size

                # Quantize coordinates
                coord_min = coord.min(0)[0]
                grid_coord = ((coord - coord_min) / grid_size).floor().int()

                # Ensure grid_coord does not overflow int16
                if grid_coord.max() > 2**15:
                    logger.warning(f"Grid coordinate overflow (max={grid_coord.max()}), using coarser grid_size")
                    grid_size = (coord.max(0)[0] - coord.min(0)[0]).max().item() / 10000
                    grid_coord = ((coord - coord_min) / grid_size).floor().int()

                # Batch and offset arrays for single-frame (batch_size=1)
                num_points = coord.shape[0]
                batch = torch.zeros(num_points, dtype=torch.long, device=device)
                offset = torch.tensor([num_points], dtype=torch.long, device=device)

                # Build input features
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

                # Model input
                data_dict = {
                    "coord": coord,
                    "feat": input_feat,
                    "grid_coord": grid_coord,
                    "grid_size": grid_size,
                    "offset": offset,
                    "batch": batch,
                }

                # Debug info (first batch, first epoch)
                if epoch == 0 and batch_idx == 0:
                    logger.info(f"Grid size used: {grid_size}")
                    logger.info(f"Coord shape: {coord.shape}")
                    logger.info(f"Input feat shape: {input_feat.shape}")
                    logger.info(f"Grid coord shape: {grid_coord.shape}")
                    logger.info(f"Grid coord range: {grid_coord.min()} to {grid_coord.max()}")
                    logger.info(f"Offset: {offset}")
                    logger.info(f"Batch shape: {batch.shape}")

                # Forward pass
                optimizer.zero_grad()
                output = model(data_dict)
                pred = output.feat

                # Compute loss
                loss = distillation_loss(pred, dino_feat)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                logger.error(f"Coord shape: {coord.shape if 'coord' in locals() else 'undefined'}")
                logger.error(f"Feat shape: {feat.shape if 'feat' in locals() else 'undefined'}")
                logger.error(f"Grid size: {grid_size if 'grid_size' in locals() else 'undefined'}")
                raise e

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Avg Loss = {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Best model saved to {save_path} (loss = {avg_loss:.6f})")

    logger.info("Training complete.")

if __name__ == "__main__":
    # Set ablation mode here: 'dino_only', 'vri_dino', 'coord_dino', 'coord_vri_dino'
    train(input_mode="dino_only")
    # train(input_mode="vri_dino")
    # train(input_mode="coord_dino")
    # train(input_mode="coord_vri_dino")