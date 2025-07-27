from pathlib import Path
from datetime import datetime
from utils.misc import setup_logger
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.PTv3.model import PointTransformerV3


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
    epochs=30,
    batch_size=1,
    lr=1e-4,
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

    # -- Dynamically infer in_channels from first sample --
    sample = dataset[0]
    coord = sample["coord"]
    feat = sample["feat"]
    dino_feat = sample["dino_feat"]

    if input_mode == "dino_only":
        input_dim = dino_feat.shape[1]
    elif input_mode == "vri_dino":
        input_dim = feat.shape[1] + dino_feat.shape[1]
    elif input_mode == "coord_dino":
        input_dim = coord.shape[1] + dino_feat.shape[1]
    elif input_mode == "coord_vri_dino":
        input_dim = coord.shape[1] + feat.shape[1] + dino_feat.shape[1]
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")

    logger.info(f"Using input_dim={input_dim}")
    model = PointTransformerV3(in_channels=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        for batch_idx, sample in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            try:
                # Move all tensors to device
                coord = sample["coord"].to(device)
                feat = sample["feat"].to(device)
                dino_feat = sample["dino_feat"].to(device)
                grid_size = sample["grid_size"].to(device)

                # Handle batch dimension - if batch_size=1, we need to handle single samples
                if coord.dim() == 2:  # [N, 3] -> need to add batch info
                    batch_size_actual = 1
                    num_points = coord.shape[0]
                    
                    # Create batch indices (all points belong to batch 0)
                    batch = torch.zeros(num_points, dtype=torch.long, device=device)
                    offset = torch.tensor([num_points], dtype=torch.long, device=device)
                else:  # Already batched
                    batch_size_actual = coord.shape[0]
                    # Handle batched data - flatten if needed
                    if coord.dim() == 3:
                        coord = coord.view(-1, coord.shape[-1])
                        feat = feat.view(-1, feat.shape[-1])
                        dino_feat = dino_feat.view(-1, dino_feat.shape[-1])
                    
                    num_points = coord.shape[0]
                    batch = torch.arange(batch_size_actual, device=device).repeat_interleave(num_points // batch_size_actual)
                    offset = torch.tensor([num_points], dtype=torch.long, device=device)

                # Generate grid coordinates with proper scaling
                if isinstance(grid_size, torch.Tensor) and grid_size.dim() == 0:
                    grid_size_val = grid_size.item()
                else:
                    grid_size_val = grid_size
                
                # Use a reasonable grid size - too small grid sizes cause huge coordinate values
                # Clamp grid size to be at least 0.01 (1cm) to avoid huge grid coordinates
                min_grid_size = 0.01
                if grid_size_val < min_grid_size:
                    logger.warning(f"Grid size {grid_size_val} too small, using {min_grid_size}")
                    grid_size_val = min_grid_size
                
                # Create grid coordinates by quantizing the original coordinates
                coord_min = coord.min(0)[0]
                coord_max = coord.max(0)[0]
                coord_range = coord_max - coord_min
                
                # Ensure the quantized coordinates don't exceed reasonable bounds
                grid_coord = torch.div(
                    coord - coord_min, 
                    grid_size_val, 
                    rounding_mode="trunc"
                ).int()
                
                # Check if grid coordinates are within reasonable bounds (< 2^16 for each dimension)
                max_grid_coord = grid_coord.max()
                if max_grid_coord >= 2**15:  # Keep some safety margin
                    # Rescale grid size to keep coordinates reasonable
                    new_grid_size = float(coord_range.max()) / (2**14)  # Use 2^14 as max coordinate
                    logger.warning(f"Grid coordinates too large ({max_grid_coord}), rescaling grid size from {grid_size_val} to {new_grid_size}")
                    grid_size_val = new_grid_size
                    grid_coord = torch.div(
                        coord - coord_min, 
                        grid_size_val, 
                        rounding_mode="trunc"
                    ).int()

                # -- Select features based on input_mode --
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

                # Prepare data dictionary for the model
                data_dict = {
                    "coord": coord,
                    "feat": input_feat,
                    "grid_coord": grid_coord,
                    "grid_size": grid_size_val,
                    "offset": offset,
                    "batch": batch,
                }

                # Debug info (only for first batch of first epoch)
                if epoch == 0 and batch_idx == 0:
                    logger.info(f"Grid size used: {grid_size_val}")
                    logger.info(f"Coord shape: {coord.shape}")
                    logger.info(f"Coord range: {coord_min} to {coord_max}")
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