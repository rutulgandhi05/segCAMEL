import os
import math
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from models.PTv3.model import PointTransformerV3
from utils.misc import _resolve_default_workers

def safe_grid_coord(coord, grid_size, logger=None):
    if not torch.is_tensor(grid_size):
        grid_size = torch.tensor(grid_size, device=coord.device,  dtype=coord.dtype)
    else:
        grid_size = grid_size.to(coord.device,  dtype=coord.dtype)

    coord_min = coord.min(dim=0, keepdim=True).values
    grid = torch.floor((coord - coord_min) / (grid_size + 1e-8)).to(torch.int32)
    return grid

class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: Path):
        self.files = sorted(list(Path(root_dir).glob("*.pth")))
        if not self.files:
            raise FileNotFoundError(f"No .pth files found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx])
        grid_coord = safe_grid_coord(sample["coord"], sample["grid_size"])
        return {
            "coord": sample["coord"],               # (N,3) float32
            "feat": sample["feat"],                 # (N,F) float32
            "dino_feat": sample["dino_feat"],       # (N,D) float32 or float16
            "mask": sample["mask"],                 # (N,) bool/uint8
            "grid_size": sample["grid_size"],       # scalar
            "grid_coord": grid_coord,               # (N,3) int32
        }

def collate_for_ptv3(batch):
    collated = {
        "coord": [],
        "feat": [],
        "dino_feat": [],
        "grid_size": [],
        "grid_coord": [],
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
        collated["grid_coord"].append(sample["grid_coord"])
        collated["batch"].append(torch.full((N,), batch_id, dtype=torch.long))
        offset += N
        collated["offset"].append(offset)

    for k in ["coord", "feat", "dino_feat", "mask", "batch", "grid_coord"]:
        collated[k] = torch.cat(collated[k], dim=0)
    collated["grid_size"] = torch.tensor(collated["grid_size"])
    collated["offset"] = torch.tensor(collated["offset"], dtype=torch.long)
    return collated

def distillation_loss(pred_feat, target_feat):
    pred = F.normalize(pred_feat, dim=1)
    target = F.normalize(target_feat, dim=1)
    return 1 - (pred * target).sum(dim=1).mean()

def _load_state(m, state):
    """
    Load a state_dict regardless of DataParallel wrapping differences.
    Tries raw, then strip 'module.', then add 'module.'.
    """
    try:
        m.load_state_dict(state)
        return
    except Exception:
        pass
    try:
        stripped = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                    for k, v in state.items()}
        m.load_state_dict(stripped)
        return
    except Exception:
        pass
    try:
        added = {("module." + k if not k.startswith("module.") else k): v
                 for k, v in state.items()}
        m.load_state_dict(added)
        return
    except Exception as e:
        raise e
    

def train(
    data_dir,
    output_dir,
    epochs=5,
    batch_size=12,
    workers=None,
    lr=2e-3,
    prefetch_factor=2,
    device="cuda" if torch.cuda.is_available() else "cpu",
    pct_start=0.04,
    total_steps=None,
    use_data_parallel=True
):
    device = torch.device(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print(f"Starting training on device: {device}")
    print(f"Training data: {data_dir}")

    dataset = PointCloudDataset(data_dir)
    if workers is None:
        workers = _resolve_default_workers()
    workers = max(1, int(workers))

    print(f"Using {workers} DataLoader workers for training...")

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
    proj_head = torch.nn.Linear(64, dino_dim).to(device)

    if use_data_parallel and torch.cuda.device_count() > 1 and str(device).startswith("cuda"):
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
        proj_head = torch.nn.DataParallel(proj_head)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(proj_head.parameters()), 
                                  lr=lr,
                                  weight_decay=0.05)

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
    start_epoch = 0

    latest_ckpt_path = output_dir / "latest_checkpoint.pth"
    best_ckpt_path = output_dir / "best_model.pth"


    # ---- RESUME LOGIC ----
    if latest_ckpt_path.exists():
        print(f"Resuming from checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        _load_state(model, checkpoint["model"])
        _load_state(proj_head, checkpoint["proj_head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # scaler may be absent or disabled; guard
        if "scaler" in checkpoint:
            try:
                scaler.load_state_dict(checkpoint["scaler"])
            except Exception:
                pass
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.6f}")

    # Autocast kwargs
    autocast_kwargs = {"device_type": "cuda", "dtype": torch.bfloat16} if device.type == "cuda" else {"device_type": "cpu"}


    for epoch in range(epochs):
        model.train()
        proj_head.train()
        total_loss = 0.0
        print(f"\nEpoch {epoch + 1}/{epochs}")

        skipped_batches = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            try:
                coord = batch["coord"].to(device, non_blocking=True)
                feat = batch["feat"].to(device, non_blocking=True).float()
                dino_feat = batch["dino_feat"].to(device, non_blocking=True).float()   # upcast in case saved fp16
                mask = batch["mask"].to(device, non_blocking=True).bool()
                grid_size = batch["grid_size"].mean().item()
                batch_tensor = batch["batch"].to(device, non_blocking=True)
                offset = batch["offset"].to(device, non_blocking=True)
                grid_coord = batch["grid_coord"].to(device, non_blocking=True)

                # Normalize coord/feat per-batch before concat
                coord = (coord - coord.mean(dim=0)) / (coord.std(dim=0) + 1e-6)
                feat = (feat - feat.mean(dim=0)) / (feat.std(dim=0) + 1e-6)
                input_feat = torch.cat([coord, feat], dim=1)

                if batch_idx == 0:
                    print(f"[Batch 0] input_feat stats: min={input_feat.min().item():.4f}, "
                          f"max={input_feat.max().item():.4f}, mean={input_feat.mean().item():.4f}, "
                          f"std={input_feat.std().item():.4f}")
                    
                if torch.isnan(input_feat).any() or torch.isinf(input_feat).any():
                    print(f"[ERROR][Batch {batch_idx}] NaN/Inf in input_feat, skipping batch")
                    skipped_batches += 1
                    continue
                if input_feat.abs().sum().item() == 0:
                    print(f"[WARN][Batch {batch_idx}] All-zero input_feat, skipping batch")
                    skipped_batches += 1
                    continue

                data_dict = {
                    "coord": coord,
                    "feat": input_feat,
                    "grid_coord": grid_coord,
                    "grid_size": grid_size,
                    "offset": offset,
                    "batch": batch_tensor,
                }

                optimizer.zero_grad(set_to_none=True)
                with autocast(**autocast_kwargs):
                    output = model(data_dict)
                    if torch.isnan(output.feat).any() or torch.isinf(output.feat).any():
                        print(f"[ERROR][Batch {batch_idx}] NaN/Inf in model output, skipping batch")
                        skipped_batches += 1
                        continue

                    pred_proj = proj_head(output.feat)
                    if torch.isnan(pred_proj).any() or torch.isinf(pred_proj).any():
                        print(f"[ERROR][Batch {batch_idx}] NaN/Inf in proj_head output, skipping batch")
                        skipped_batches += 1
                        continue
                   
                    # Distill on visible points only
                    valid_mask = mask
                    if valid_mask.sum().item() == 0:
                        print(f"[WARN][Batch {batch_idx}] No visible points; skipping batch")
                        skipped_batches += 1
                        continue

                    pred_valid = pred_proj[valid_mask]
                    dino_valid = dino_feat[valid_mask]
                    loss = distillation_loss(pred_valid, dino_valid)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(proj_head.parameters(), max_norm=1.0)
               
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += loss.item()

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                skipped_batches += 1
                continue

            finally:
                # Explicitly delete large variables (NO per-batch empty_cache)
                for var in [
                    "coord", "feat", "dino_feat", "mask", "grid_size", "batch_tensor",
                    "offset", "input_feat", "grid_coord", "output", "pred_proj",
                    "valid_mask", "pred_valid", "dino_valid", "loss"
                ]:
                    if var in locals():
                        del locals()[var]
                del batch
                #torch.cuda.empty_cache()    

        denom = len(dataloader) - skipped_batches
        if denom <= 0:
            print("[WARN] All batches skipped this epoch; not updating best model.")
            avg_loss = float("inf")
        else:
            avg_loss = total_loss / denom
        print(f"Avg Loss = {avg_loss:.6f}")


        # ---- Save latest (rolling) and epoch-suffixed checkpoints ----
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Get plain state_dicts (handle DP)
        model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        head_state = proj_head.module.state_dict() if isinstance(proj_head, torch.nn.DataParallel) else proj_head.state_dict()

        state = {
            "model": model_state,
            "proj_head": head_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss
        }

        latest_ckpt_epoch = Path(output_dir) / f"latest_checkpoint_epoch{epoch+1:04d}.pth"
        torch.save(state, latest_ckpt_epoch)
        latest_ckpt_path = Path(output_dir) / "latest_checkpoint.pth"
        torch.save(state, latest_ckpt_path)
        print(f"Saved: {latest_ckpt_epoch.name} and latest_checkpoint.pth")

        # ---- Save best ----
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(state, best_ckpt_path)
            print(f"Best model saved to {best_ckpt_path} (loss = {avg_loss:.6f})")

        print(f"[Epoch {epoch+1}] Skipped {skipped_batches} batches due to errors.")
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    print("Training complete.")

if __name__ == "__main__":
   
    DATA_DIR = Path(os.getenv("PREPROCESS_OUTPUT_DIR"))
    TRAIN_CHECKPOINTS = Path(os.getenv("TRAIN_CHECKPOINTS"))
    TRAIN_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    train(
        data_dir=DATA_DIR,
        output_dir=TRAIN_CHECKPOINTS,
        epochs=2,
        workers=None,
        batch_size=12,
        prefetch_factor=2,
        lr=2e-3,
        use_data_parallel=True
    )
