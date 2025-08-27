import os
import math
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple
from functools import partial
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from models.PTv3.model_ import PointTransformerV3
from utils.misc import _resolve_default_workers

import torch.multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    print("Spawn method already set or not available, using default.")
    pass


def safe_grid_coord(coord: torch.Tensor, grid_size, *, origin: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Build integer voxel coordinates from RAW coords (not normalized).
    grid_size: float or 0-D tensor
    origin: if provided (3,), use it as min; else compute per-sample min().
    """
    if not torch.is_tensor(grid_size):
        grid_size = torch.tensor(grid_size, device=coord.device, dtype=coord.dtype)
    else:
        grid_size = grid_size.to(device=coord.device, dtype=coord.dtype)

    if origin is None:
        origin = coord.min(dim=0, keepdim=True).values
    else:
        origin = origin.to(coord.device, coord.dtype).view(1, 3)

    grid = torch.floor((coord - origin) / (grid_size + 1e-8)).to(torch.int32)
    return grid


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: Path, voxel_size: float):
        self.files = sorted(list(Path(root_dir).glob("*.pth")))
        if not self.files:
            raise FileNotFoundError(f"No .pth files found in {root_dir}")
        self.voxel_size = voxel_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        rec = torch.load(f, map_location="cpu")

        coord = rec["coord"]                   # (N,3) float32
        feat  = rec.get("feat", torch.empty((coord.shape[0], 0), dtype=coord.dtype))
        mask  = rec.get("mask", torch.ones((coord.shape[0],), dtype=torch.bool))
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        # voxel grid (compute here for robustness, even if present in file)
        gsize = self.voxel_size if self.voxel_size is not None else float(rec.get("grid_size", 0.1))
        grid_coord = safe_grid_coord(coord, gsize)

        # robust image stem (may be None or missing)
        img_rel = rec.get("image_relpath", None)
        if isinstance(img_rel, (str, Path)) and str(img_rel):
            image_stem = Path(str(img_rel)).stem
        else:
            image_stem = ""

        out = {
            "coord": coord,                         # (N,3) float32
            "feat": feat,                           # (N,F) float32
            "mask": mask,                           # (N,) bool
            "grid_size": torch.tensor(float(gsize)),
            "grid_coord": grid_coord,               # (N,3) int32
            "lidar_stem": f.stem,
            "image_stem": image_stem,
        }
        if "dino_feat" in rec:
            out["dino_feat"] = rec["dino_feat"]     # (N,D) fp16/float32 OK
        return out


def _unique_first(indices_3d: torch.Tensor) -> torch.Tensor:
    """
    Return indices of the FIRST occurrence of each unique 3D voxel row in the ORIGINAL order,
    without using torch.unique(..., return_index=...).
    Assumes indices_3d are non-negative (true for (coord - min)/voxel_size).
    """
    if indices_3d.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    g = indices_3d.to(torch.int64).contiguous()  # (N, 3)
    max_vals = g.max(dim=0).values + 1           # (3,)
    max_vals = torch.clamp(max_vals, min=1)
    M_yz = max_vals[1] * max_vals[2]
    key = g[:, 0] * M_yz + g[:, 1] * max_vals[2] + g[:, 2]  # (N,)

    perm = torch.argsort(key, stable=True)
    key_sorted = key[perm]

    first_mask = torch.ones_like(key_sorted, dtype=torch.bool)
    first_mask[1:] = key_sorted[1:] != key_sorted[:-1]

    first_sorted = perm[first_mask]            # first occurrences in sorted order
    sel = first_sorted.sort().values           # back to ascending ORIGINAL indices
    return sel


def _multiscale_voxel_select(
    coord: torch.Tensor,
    grid_sizes: Tuple[float, float, float] = (0.05, 0.10, 0.20),
    r_bins: Tuple[float, float] = (30.0, 70.0),
) -> torch.Tensor:
    """
    Distance-aware voxel selection:
      - near (<= r_bins[0])    : small voxels (e.g., 5 cm)
      - mid  (r0, <= r_bins[1]): medium voxels (e.g., 10 cm)
      - far  (> r_bins[1])     : large voxels (e.g., 20 cm)
    Returns a 1-D index tensor selecting one point per voxel per bin.
    """
    if coord.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    r_xy = torch.linalg.norm(coord[:, :2], dim=1)

    near_mask = r_xy <= r_bins[0]
    mid_mask  = (r_xy > r_bins[0]) & (r_xy <= r_bins[1])
    far_mask  = r_xy > r_bins[1]

    sels = []
    for mask, gsize in zip((near_mask, mid_mask, far_mask), grid_sizes):
        if mask.any():
            sub_idx = mask.nonzero(as_tuple=False).squeeze(1)
            sub_coord = coord.index_select(0, sub_idx)
            sub_grid = safe_grid_coord(sub_coord, gsize)
            sel_local = _unique_first(sub_grid)
            sels.append(sub_idx.index_select(0, sel_local))

    if len(sels) == 0:
        return torch.empty((0,), dtype=torch.long)

    sel = torch.cat(sels, dim=0)
    return sel.sort().values


def collate_for_ptv3(batch, *, multiscale_voxel: bool = False):
    collated = {
        "coord": [], "feat": [], "grid_size": [], "grid_coord": [],
        "mask": [], "batch": [], "offset": [], "lidar_stems": [], "image_stems": [],
    }
    has_dino_feat = "dino_feat" in batch[0]
    if has_dino_feat:
        collated["dino_feat"] = []

    offset = 0
    for batch_id, sample in enumerate(batch):
        coord = sample["coord"]
        feat = sample["feat"]
        grid = sample["grid_coord"]
        mask = sample["mask"]
        gsize = float(sample["grid_size"])
        lidar_stem = sample["lidar_stem"]
        image_stem = sample["image_stem"]

        if multiscale_voxel:
            sel = _multiscale_voxel_select(coord, grid_sizes=(0.05, 0.10, 0.20), r_bins=(30.0, 70.0))
        else:
            sel = _unique_first(grid)

        if sel.numel() > 0:
            coord = coord.index_select(0, sel)
            feat  = feat.index_select(0, sel)
            grid  = grid.index_select(0, sel)
            mask  = mask.index_select(0, sel)
            if has_dino_feat:
                dino = sample["dino_feat"].index_select(0, sel)
        else:
            continue

        N = coord.shape[0]
        collated["coord"].append(coord)
        collated["feat"].append(feat)
        collated["grid_coord"].append(grid)
        collated["mask"].append(mask if mask.dtype == torch.bool else mask.to(torch.bool))
        collated["grid_size"].append(gsize)
        collated["lidar_stems"].append(lidar_stem)
        collated["image_stems"].append(image_stem)
        collated["batch"].append(torch.full((N,), batch_id, dtype=torch.long))
        offset += N
        collated["offset"].append(offset)
        if has_dino_feat:
            collated["dino_feat"].append(dino)

    for k in ["coord", "feat", "mask", "batch", "grid_coord"]:
        collated[k] = torch.cat(collated[k], dim=0)
    if has_dino_feat:
        collated["dino_feat"] = torch.cat(collated["dino_feat"], dim=0)
    collated["grid_size"] = torch.tensor(collated["grid_size"])
    collated["offset"] = torch.tensor(collated["offset"], dtype=torch.long)
    return collated


def distillation_loss(pred_feat, target_feat, eps=1e-6):
    pred = F.normalize(pred_feat, dim=1, eps=eps)
    target = F.normalize(target_feat, dim=1, eps=eps)
    return 1 - (pred * target).sum(dim=1).mean()


def _load_state(m, state):
    """Load a state_dict regardless of DataParallel wrapping differences."""
    try:
        m.load_state_dict(state); return
    except Exception:
        pass
    try:
        stripped = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
        m.load_state_dict(stripped); return
    except Exception:
        pass
    try:
        added = {("module." + k if not k.startswith("module.") else k): v for k, v in state.items()}
        m.load_state_dict(added); return
    except Exception as e:
        raise e


def train(
    data_dir,
    output_dir,
    epochs=20,
    batch_size=12,
    accum_steps=8,
    workers=None,
    lr=2e-3,
    prefetch_factor=2,
    device="cuda" if torch.cuda.is_available() else "cpu",
    pct_start=0.04,
    total_steps=None,
    use_data_parallel=True,
    voxel_size: float = 0.10,  # outdoor default
    multiscale_voxel: bool = False,
    feat_mode: str = "rvi"
):
    # ---- Reproducibility ----
    torch.use_deterministic_algorithms(False)
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    g = torch.Generator(device="cpu"); g.manual_seed(seed)

    device = torch.device(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print(f"Starting training on device: {device}")
    print(f"Training data: {data_dir}")

    dataset = PointCloudDataset(data_dir, voxel_size=voxel_size)
    if workers is None:
        workers = _resolve_default_workers()
    workers = max(1, int(workers))
    print(f"Using {workers} DataLoader workers for training...")

    collate_fn = partial(collate_for_ptv3, multiscale_voxel=multiscale_voxel)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
        multiprocessing_context="spawn",
        generator=g,
    )

    num_batches = len(dataloader)
    updates_per_epoch = math.ceil(max(1, num_batches) / max(1, accum_steps))

    print(f"Loaded {len(dataset)} preprocessed samples")
    print(f"Per-step batch size: {batch_size} | Accum steps: {accum_steps} | "
          f"Effective batch: {batch_size * accum_steps}")
    print(f"Epochs: {epochs} | Batches/epoch: {num_batches} | Updates/epoch: {updates_per_epoch} | Total Steps: {epochs * updates_per_epoch}")
    if multiscale_voxel:
        print("Multi-scale voxelization: ON (near=5cm, mid=10cm, far=20cm)")
    else:
        print(f"Single voxel size: {voxel_size:.3f} m")

    if total_steps:
        epochs = math.ceil(total_steps / updates_per_epoch)
        print(f"Will train for {epochs} epochs to hit ~{total_steps} optimizer steps.")

    # Probe sample + feat_mode selection (robust to missing keys)
    probe = dataset[0]
    feat_dim_total = probe["feat"].shape[1]
    col_map = {"r": 0, "v": 1, "i": 2}
    if feat_mode == "none":
        keep_cols = []
    else:
        keep_cols = [col_map[c] for c in feat_mode if c in col_map and col_map[c] < feat_dim_total]
    input_dim = probe["coord"].shape[1] + len(keep_cols)

    if "dino_feat" not in probe:
        raise RuntimeError("This training pipeline requires 'dino_feat' in preprocessed files for distillation.")
    dino_dim = probe["dino_feat"].shape[1]

    print(f"Using input_dim={input_dim} (feat_mode='{feat_mode}', keep_cols={keep_cols}), dino_dim={dino_dim}")

    model = PointTransformerV3(
        in_channels=input_dim,
        enable_flash=False,                          # classic attention
        enc_patch_size=(256, 256, 256, 256, 256),
        dec_patch_size=(256, 256, 256, 256),
        enable_rpe=False,                            # optional
        upcast_attention=True,                       # safer QK numerics
        upcast_softmax=True
    ).to(device)

    proj_head = torch.nn.Linear(64, dino_dim).to(device)

    if use_data_parallel and torch.cuda.device_count() > 1 and device.type == "cuda":
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
        proj_head = torch.nn.DataParallel(proj_head)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(proj_head.parameters()),
        lr=lr,
        weight_decay=0.05
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=updates_per_epoch,
        epochs=epochs,
        pct_start=pct_start,
        anneal_strategy="cos",
        div_factor=10,        # initial LR = max_lr / div_factor
        final_div_factor=100  # final LR = initial / final_div_factor
    )

    scaler = GradScaler(enabled=False)
    best_loss = float("inf")
    start_epoch = 0

    latest_ckpt_path = Path(output_dir) / "latest_checkpoint.pth"
    best_ckpt_path = Path(output_dir) / "best_model.pth"

    # Resume if latest exists
    if latest_ckpt_path.exists():
        print(f"Resuming from checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        _load_state(model, checkpoint["model"])
        _load_state(proj_head, checkpoint["proj_head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            try: scaler.load_state_dict(checkpoint["scaler"])
            except Exception: pass
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.6f}")

    for epoch in range(start_epoch, epochs):
        model.train(); proj_head.train()
        print(f"\nEpoch {epoch + 1}/{epochs}")
        skipped_batches = 0
        oom_streak = 0

        total_loss_raw = 0.0
        counted_batches = 0
        update_count = 0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            try:
                if batch["coord"].numel() == 0:
                    skipped_batches += 1; continue

                mask_cpu = batch["mask"].bool()
                if mask_cpu.sum().item() == 0:
                    skipped_batches += 1; continue
                idx_cpu = mask_cpu.nonzero(as_tuple=False).squeeze(1)

                coord = batch["coord"].to(device, non_blocking=True)
                feat  = batch["feat"].to(device, non_blocking=True).float()
                grid_coord = batch["grid_coord"].to(device, non_blocking=True)
                batch_tensor = batch["batch"].to(device, non_blocking=True)
                offset = batch["offset"].to(device, non_blocking=True)
                grid_size_val = float(batch["grid_size"].mean().item())

                # Feature-channel selection (matches feat_mode used for input_dim)
                cols = {"r": 0, "v": 1, "i": 2}
                if feat_mode == "none":
                    feat_sel = torch.empty(feat.shape[0], 0, device=feat.device, dtype=feat.dtype)
                else:
                    keep = [cols[c] for c in feat_mode if c in cols and cols[c] < feat.shape[1]]
                    feat_sel = feat[:, keep] if len(keep) else torch.empty(feat.shape[0], 0, device=feat.device, dtype=feat.dtype)

                # Normalize coord/feat per-batch then concatenate
                coord = (coord - coord.mean(dim=0)) / (coord.std(dim=0) + 1e-6)
                if feat_sel.numel() > 0:
                    feat_sel = (feat_sel - feat_sel.mean(dim=0)) / (feat_sel.std(dim=0) + 1e-6)
                input_feat = torch.cat([coord, feat_sel], dim=1)

                if batch_idx == 0:
                    print(f"[Batch 0] input_feat stats: mean={input_feat.mean().item():.4f}, std={input_feat.std().item():.4f}")

                # NaN/Inf guard
                if torch.isnan(input_feat).any() or torch.isinf(input_feat).any():
                    print(f"[ERROR][Batch {batch_idx}] NaN/Inf in input_feat, skipping batch")
                    skipped_batches += 1; continue
                if input_feat.numel() == 0:
                    skipped_batches += 1; continue

                data_dict = {
                    "coord": coord,
                    "feat": input_feat,
                    "grid_coord": grid_coord,
                    "grid_size": grid_size_val,
                    "offset": offset,
                    "batch": batch_tensor,
                }

                output = model(data_dict)
                feats = torch.nan_to_num(output.feat, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-20, 20)
                pred_proj = torch.nn.functional.linear(feats.float(), weight=proj_head.weight, bias=proj_head.bias)
                pred_proj = torch.nan_to_num(pred_proj, nan=0.0, posinf=1e4, neginf=-1e4)
                if torch.isnan(pred_proj).any() or torch.isinf(pred_proj).any():
                    print(f"[ERROR][Batch {batch_idx}] NaN/Inf in proj_head output, skipping batch")
                    skipped_batches += 1; continue

                # Distill on visible points only
                idx = idx_cpu.to(device, non_blocking=True)
                pred_valid = pred_proj.index_select(0, idx)
                dino_valid = batch["dino_feat"].index_select(0, idx_cpu).to(device, non_blocking=True).float()
                dino_valid = torch.nan_to_num(dino_valid, nan=0.0, posinf=1e4, neginf=-1e4)

                loss_raw = distillation_loss(pred_valid, dino_valid)
                loss = loss_raw / max(1, accum_steps)
                loss.backward()
                counted_batches += 1
                total_loss_raw += loss_raw.item()

                # Optimizer step on accumulation boundary
                if (counted_batches % accum_steps) == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(proj_head.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    update_count += 1
                    oom_streak = 0  # successful step resets OOM streak

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    skipped_batches += 1
                    oom_streak += 1
                    print(f"[OOM][Batch {batch_idx}] {str(e)} | streak={oom_streak}. Clearing cache and continuing.")
                    for var in [
                        "coord","feat","grid_coord","batch_tensor","offset",
                        "input_feat","output","feats","pred_proj",
                        "pred_valid","dino_valid","loss","loss_raw","idx","idx_cpu","mask_cpu","feat_sel"
                    ]:
                        if var in locals(): del locals()[var]
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if oom_streak >= 5:
                        print("[WARN] Too many consecutive OOMs; breaking to next epoch.")
                        break
                    continue
                else:
                    raise
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                skipped_batches += 1
                continue
            finally:
                for var in [
                    "coord","feat","grid_coord","batch_tensor","offset",
                    "input_feat","output","feats","pred_proj",
                    "pred_valid","dino_valid","loss","loss_raw","idx","idx_cpu","mask_cpu","feat_sel"
                ]:
                    if var in locals(): del locals()[var]
                del batch

        denom_batches = counted_batches if counted_batches > 0 else 1
        avg_loss = total_loss_raw / denom_batches
        print(f"Avg Loss (per batch) = {avg_loss:.6f} | Updates this epoch = {update_count} | Skipped batches = {skipped_batches}")

        # ---- Save latest and best checkpoints ----
        Path(output_dir).mkdir(parents=True, exist_ok=True)
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

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt_path = Path(output_dir) / "best_model.pth"
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
        epochs=30,
        workers=12,
        batch_size=4,
        accum_steps=8,
        prefetch_factor=2,
        lr=2e-3,
        use_data_parallel=False,
        voxel_size=0.10,      # 10 cm voxel size
        multiscale_voxel=False,
        feat_mode="rvi",
    )
