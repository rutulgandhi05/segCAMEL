import os

from tqdm import tqdm
from pathlib import Path
from utils.misc import setup_logger
from models.PTv3.model import PointTransformerV3

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import math

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
            "grid_size": float(sample.get("grid_size", 0.05)),
            "mask": sample["mask"], 
        }

def collate_for_ptv3(batch):
    collated = {
        "coord": [],
        "feat": [],
        "dino_feat": [],
        "grid_size": [],
        "mask": [],
        "batch": [],
        "offset": [],
    }

    offset = 0
    for batch_id, sample in enumerate(batch):
        N = sample["coord"].shape[0]
        collated["coord"].append(sample["coord"])
        collated["feat"].append(sample["feat"])
        collated["dino_feat"].append(sample["dino_feat"])
        collated["mask"].append(sample["mask"])
        collated["grid_size"].append(sample["grid_size"])
        collated["batch"].append(torch.full((N,), batch_id, dtype=torch.long))
        offset += N
        collated["offset"].append(offset)
    for k in ["coord", "feat", "dino_feat", "mask", "batch"]:
        collated[k] = torch.cat(collated[k], dim=0)
    collated["grid_size"] = torch.tensor(collated["grid_size"])
    collated["offset"] = torch.tensor(collated["offset"], dtype=torch.long)
    return collated

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
            grid_coord[:, axis] += torch.arange(grid_coord.shape[0], device=grid_coord.device) % 2

    return grid_coord

def train(
    data_dir=Path,
    epochs=20,
    batch_size=12,
    workers: int = 8,
    lr=2e-3,
    save_path=Path,
    prefetch_factor: int = 2,
    device="cuda" if torch.cuda.is_available() else "cpu",
    pct_start=0.04,
    total_steps=None,
):

    print(f"Starting training on device: {device}")
    print(f"Training data: {data_dir}")

    dataset = PointCloudDataset(data_dir)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=workers,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=prefetch_factor,
                            collate_fn=collate_for_ptv3
                            )

    print(f"Loaded {len(dataset)} preprocessed samples")
    steps_per_epoch = len(dataloader)
    est_total_steps = epochs * steps_per_epoch
    print(f"Batch size: {batch_size} | Epochs: {epochs} | Steps per epoch: {steps_per_epoch}")
    print(f"Estimated total steps: {est_total_steps}")
    if total_steps:
        epochs = math.ceil(total_steps / steps_per_epoch)
        print(f"Will train for {epochs} epochs to hit {total_steps} steps.")

    # --- Infer input_dim robustly based on input_mode and first sample
    sample = dataset[0]
    input_dim = sample["coord"].shape[1] + sample["feat"].shape[1]
    dino_dim = sample["dino_feat"].shape[1]
    print(f"Using input_dim={input_dim}, dino_dim={dino_dim}")

    model = PointTransformerV3(in_channels=input_dim).to(device)
    #proj_head = torch.nn.Linear(getattr(model, 'out_channels', 64), dino_dim).to(device)
    proj_head = torch.nn.Linear(64, dino_dim).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(proj_head.parameters()), lr=lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=pct_start,
        anneal_strategy="cos",
        div_factor=10,       # initial LR = max_lr / div_factor
        final_div_factor=100 # final LR = initial / final_div_factor
    )
    scaler = GradScaler(device=device)
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        proj_head.train()
        total_loss = 0
        print(f"\nEpoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            try:
                coord = batch["coord"].to(device)
                feat = batch["feat"].to(device)
                dino_feat = batch["dino_feat"].to(device)
                mask = batch["mask"].to(device)
                grid_size = batch["grid_size"].mean().item()
                batch_tensor = batch["batch"].to(device)
                offset = batch["offset"].to(device)

                input_feat = torch.cat([coord, feat], dim=1)
                grid_coord = safe_grid_coord(coord, grid_size)

                if batch_idx == 0:
                    print(f"[Batch 0] input_feat: min={input_feat.min().item()}, max={input_feat.max().item()}, mean={input_feat.mean().item()}, std={input_feat.std().item()}")
                    print(f"  Any NaN: {torch.isnan(input_feat).any().item()}, Any Inf: {torch.isinf(input_feat).any().item()}")

                # Basic validity checks
                if torch.isnan(input_feat).any() or torch.isinf(input_feat).any():
                    print(f"[ERROR][Batch {batch_idx}] NaN or Inf in input_feat, skipping batch")
                    continue
                if input_feat.abs().sum().item() == 0:
                    print(f"[WARN][Batch {batch_idx}] All-zero input_feat, skipping batch")
                    continue

                data_dict = {
                    "coord": coord,
                    "feat": input_feat,
                    "grid_coord": grid_coord,
                    "grid_size": grid_size,
                    "offset": offset,
                    "batch": batch_tensor,
                }

                optimizer.zero_grad()
                with autocast(device_type=device):
                    output = model(data_dict)
                    if torch.isnan(output.feat).any() or torch.isinf(output.feat).any():
                        print(f"[ERROR][Batch {batch_idx}] NaN or Inf in model output, skipping batch")
                        continue

                    # Check model output
                   
                    pred_proj = proj_head(output.feat)
                    if torch.isnan(pred_proj).any() or torch.isinf(pred_proj).any():
                        print(f"[ERROR][Batch {batch_idx}] NaN or Inf in proj_head output, skipping batch")
                        continue
                   
                    valid_mask = mask.bool()
                    

                    pred_valid = pred_proj[valid_mask]
                    dino_valid = dino_feat[valid_mask]
                    if pred_valid.shape[0] != dino_valid.shape[0]:
                        print(f"[ERROR][Batch {batch_idx}] Masked shape mismatch: {pred_valid.shape} vs {dino_valid.shape}, skipping batch")
                        continue

                    loss = distillation_loss(pred_valid, dino_valid)
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"[ERROR][Batch {batch_idx}] Loss is NaN or Inf, skipping batch")
                        continue
                

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += loss.item()

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                continue

            finally:
                # Explicitly delete large variables for memory safety
                for var in [
                    "coord", "feat", "dino_feat", "mask", "grid_size", "batch_tensor",
                    "offset", "input_feat", "grid_coord", "output", "pred_proj",
                    "valid_mask", "pred_valid", "dino_valid", "loss"
                ]:
                    if var in locals():
                        del locals()[var]
                del batch
                torch.cuda.empty_cache()

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
   
    DATA_DIR = Path(os.getenv("PREPROCESS_OUTPUT_DIR"))
    TRAIN_CHECKPOINT = Path(os.getenv("TRAIN_CHECKPOINT"))

    train(
        data_dir=DATA_DIR,
        epochs=10,
        workers=16,
        batch_size=12,
        prefetch_factor=4,
        lr=2e-3,
        save_path=TRAIN_CHECKPOINT,
    )
