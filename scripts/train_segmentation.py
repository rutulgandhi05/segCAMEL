# train_segmentation.py
import os
import math
from pathlib import Path
from tqdm import tqdm
from typing import Optional
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
    pass


def _safe_grid_coord(coord: torch.Tensor, grid_size: float, origin: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Build integer voxel coordinates from RAW coords (not normalized)."""
    if origin is None:
        origin = coord.min(dim=0, keepdim=True).values
    else:
        origin = origin.view(1, 3).to(coord.device, coord.dtype)
    g = torch.floor((coord - origin) / (grid_size + 1e-8)).to(torch.int32)
    return g


def _visible_first_voxel_select(
    xyz: torch.Tensor, vis_mask: torch.Tensor, voxel_size: float, origin: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Fallback selector (only used if *_trainvox.pth is unavailable).
    Picks ONE point per voxel, preferring vis_mask==True. CPU-safe, stable order.
    """
    if xyz.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    x = xyz.detach().cpu().to(torch.float32)
    v = vis_mask.detach().cpu().to(torch.bool).view(-1)

    if origin is None:
        origin = x.min(dim=0, keepdim=True).values
    else:
        origin = origin.view(1, 3).to(x.dtype)

    g = torch.floor((x - origin) / max(float(voxel_size), 1e-8)).to(torch.int64)  # (N,3)
    max_vals = g.max(dim=0).values + 1
    max_vals = torch.clamp(max_vals, min=1)
    M_yz = max_vals[1] * max_vals[2]
    key = g[:, 0] * M_yz + g[:, 1] * max_vals[2] + g[:, 2]

    # prefer visible points: sort by (key, flag) where flag=0 for visible else 1
    flag = (~v).to(torch.int64)
    composite = key * 2 + flag
    perm = torch.argsort(composite, stable=True)

    key_sorted = key[perm]
    first_mask = torch.ones_like(key_sorted, dtype=torch.bool)
    first_mask[1:] = key_sorted[1:] != key_sorted[:-1]

    first_sorted = perm[first_mask]
    sel = first_sorted.sort().values
    return sel.to(torch.long)


# ----------------------------
# Dataset (train on *_trainvox.pth)
# ----------------------------
class TrainVoxDataset(torch.utils.data.Dataset):
    """
    Preferred path: loads compact visible-first voxel tables created in preprocess.py.
    If not found, falls back to dense *.pth and performs visible-first selection on the fly.
    """
    def __init__(self, root_dir: Path, fallback_voxel_size: float = 0.10):
        self.root_dir = Path(root_dir)
        self.files = sorted(self.root_dir.glob("*_trainvox.pth"))
        self.fallback_files = []
        self.fallback_voxel_size = float(fallback_voxel_size)

        if not self.files:
            # Fallback to dense files (slower, but keeps training usable)
            self.fallback_files = sorted(self.root_dir.glob("*.pth"))
            self.fallback_files = [f for f in self.fallback_files if not f.name.endswith("_trainvox.pth")]
            if not self.fallback_files:
                raise FileNotFoundError(f"No *_trainvox.pth or dense .pth files found in {root_dir}")

        # probe to validate keys / mode
        probe_path = self.files[0] if self.files else self.fallback_files[0]
        rec = torch.load(probe_path, map_location="cpu")
        self.using_trainvox = "vox_coord" in rec

    def __len__(self):
        return len(self.files) if self.using_trainvox else len(self.fallback_files)

    def __getitem__(self, idx):
        f = self.files[idx] if self.using_trainvox else self.fallback_files[idx]
        rec = torch.load(f, map_location="cpu")

        if self.using_trainvox:
            coord = rec["vox_coord"].to(torch.float32)                     # (M,3)
            feat  = rec.get("vox_feat", torch.empty((coord.shape[0], 0)))  # (M,F)
            mask  = rec.get("vox_mask", torch.ones((coord.shape[0],), dtype=torch.bool))
            dino  = rec.get("vox_dino", None)
            if dino is None:
                raise RuntimeError(f"{f.name} is missing 'vox_dino' needed for distillation.")
            gs = rec.get("grid_size", 0.10)
            grid_size = float(gs.item() if torch.is_tensor(gs) else gs)
            lidar_stem = rec.get("lidar_stem", Path(f).stem)
            image_stem = rec.get("image_stem", "")

            grid_coord = _safe_grid_coord(coord, grid_size)
            return {
                "coord": coord, "feat": feat, "mask": mask.to(torch.bool),
                "grid_coord": grid_coord, "grid_size": torch.tensor(grid_size),
                "dino_feat": dino,  # (M,D)
                "lidar_stem": str(lidar_stem), "image_stem": str(image_stem),
            }

        # ---- Fallback path (dense .pth -> visible-first voxel) ----
        coord = rec["coord"].to(torch.float32)                              # (N,3)
        feat  = rec.get("feat", torch.empty((coord.shape[0], 0))).to(torch.float32)
        mask  = rec.get("mask", torch.ones((coord.shape[0],), dtype=torch.bool)).to(torch.bool)
        dino  = rec.get("dino_feat", None)
        if dino is None:
            raise RuntimeError(f"{f.name} is missing 'dino_feat' (dense fallback requires it).")

        sel = _visible_first_voxel_select(coord, mask, self.fallback_voxel_size, origin=None)
        if sel.numel() == 0:
            sel = torch.arange(0, 0, dtype=torch.long)

        coord = coord.index_select(0, sel)
        feat  = feat.index_select(0, sel) if feat.numel() else feat
        mask  = mask.index_select(0, sel)
        dino  = dino.index_select(0, sel)

        grid_size = float(self.fallback_voxel_size)
        grid_coord = _safe_grid_coord(coord, grid_size)

        return {
            "coord": coord, "feat": feat, "mask": mask,
            "grid_coord": grid_coord, "grid_size": torch.tensor(grid_size),
            "dino_feat": dino,
            "lidar_stem": Path(f).stem, "image_stem": Path(f).stem,
        }


def collate_trainvox(batch):
    """
    Simple, robust collate: concatenate variable-length samples.
    """
    collated = {
        "coord": [], "feat": [], "mask": [], "grid_coord": [], "grid_size": [],
        "dino_feat": [], "batch": [], "offset": [],
        "lidar_stems": [], "image_stems": [],
    }
    offset = 0
    kept = 0
    for bid, sample in enumerate(batch):
        coord = sample["coord"]; feat = sample["feat"]; mask = sample["mask"].to(torch.bool)
        grid  = sample["grid_coord"]; gsize = float(sample["grid_size"])
        dino  = sample["dino_feat"]

        N = coord.shape[0]
        if N == 0:
            continue

        collated["coord"].append(coord)
        collated["feat"].append(feat)
        collated["mask"].append(mask)
        collated["grid_coord"].append(grid)
        collated["grid_size"].append(gsize)
        collated["dino_feat"].append(dino)

        collated["batch"].append(torch.full((N,), kept, dtype=torch.long))
        offset += N
        collated["offset"].append(offset)
        collated["lidar_stems"].append(sample.get("lidar_stem", ""))
        collated["image_stems"].append(sample.get("image_stem", ""))
        kept += 1

    if kept == 0:
        # Return empty tensors to avoid crashes; caller will skip this batch
        return {
            "coord": torch.zeros((0, 3)), "feat": torch.zeros((0, 0)), "mask": torch.zeros((0,), dtype=torch.bool),
            "grid_coord": torch.zeros((0, 3), dtype=torch.int32), "grid_size": torch.tensor([0.10]),
            "dino_feat": torch.zeros((0, 1)), "batch": torch.zeros((0,), dtype=torch.long),
            "offset": torch.zeros((0,), dtype=torch.long),
            "lidar_stems": [], "image_stems": [],
        }

    # concat
    for k in ["coord", "feat", "mask", "grid_coord", "dino_feat", "batch"]:
        collated[k] = torch.cat(collated[k], dim=0)
    collated["grid_size"] = torch.tensor(collated["grid_size"])
    collated["offset"]    = torch.tensor(collated["offset"], dtype=torch.long)
    return collated


def distillation_loss(pred_feat, target_feat, eps=1e-6):
    pred = F.normalize(pred_feat, dim=1, eps=eps)
    target = F.normalize(target_feat, dim=1, eps=eps)
    return 1 - (pred * target).sum(dim=1).mean()


@torch.no_grad()
def evaluate_distill_loss(
    model,
    proj_head,
    val_dir: Path,
    device: torch.device,
    *,
    workers: int = 8,
    batch_size: int = 8,
    voxel_size: float = 0.10,
    feat_mode: str = "rvi",
    prefetch_factor: int = 2,
) -> float:
    """
    Computes mean cosine distillation loss on a held-out directory written by preprocess.py.
    Uses the same visible-first voxelisation + normalisation as training for apples-to-apples evaluation.
    """
    model_was_training = model.training
    model.eval()
    if isinstance(proj_head, torch.nn.Module):
        proj_was_training = proj_head.training
        proj_head.eval()

    dataset = TrainVoxDataset(Path(val_dir), fallback_voxel_size=voxel_size)
    dataset.files = dataset.files[(len(dataset)*80)//100:]
    if len(dataset) == 0:
        print(f"[VAL] No samples under {val_dir}; returning NaN.")
        return float("nan")

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, int(workers)),
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_trainvox,
        multiprocessing_context="spawn",
    )

    # Feature cols to keep (mirror training)
    probe = dataset[0]
    feat_dim_total = probe["feat"].shape[1]
    col_map = {"r": 0, "v": 1, "i": 2}
    keep_cols = [] if feat_mode == "none" else [
        col_map[c] for c in feat_mode if c in col_map and col_map[c] < feat_dim_total
    ]

    total, count = 0.0, 0
    for batch in tqdm(dl, desc="Val distill loss", leave=False):
        if batch["coord"].numel() == 0:
            continue

        # visibility mask (match train fallback logic)
        mask_cpu = batch["mask"].bool()
        idx_cpu = mask_cpu.nonzero(as_tuple=False).squeeze(1)
        if idx_cpu.numel() == 0:
            idx_cpu = torch.arange(batch["coord"].shape[0], dtype=torch.long)

        # to device + normalise exactly like training
        coord = batch["coord"].to(device, non_blocking=True).float()
        feat  = batch["feat"].to(device, non_blocking=True).float()
        grid_coord = batch["grid_coord"].to(device, non_blocking=True)
        batch_tensor = batch["batch"].to(device, non_blocking=True)
        offset = batch["offset"].to(device, non_blocking=True)
        grid_size_val = float(batch["grid_size"].mean().item())

        if keep_cols:
            feat_sel = feat[:, keep_cols]
            feat_sel = (feat_sel - feat_sel.mean(dim=0)) / (feat_sel.std(dim=0) + 1e-6)
        else:
            feat_sel = torch.empty(feat.shape[0], 0, device=feat.device, dtype=feat.dtype)

        coord = (coord - coord.mean(dim=0)) / (coord.std(dim=0) + 1e-6)
        input_feat = torch.cat([coord, feat_sel], dim=1)

        data_dict = {
            "coord": coord,
            "feat": input_feat,
            "grid_coord": grid_coord,
            "grid_size": grid_size_val,
            "offset": offset,
            "batch": batch_tensor,
        }

        out = model(data_dict)
        feats64 = torch.nan_to_num(out.feat, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-20, 20)
        pred_proj = torch.nn.functional.linear(feats64.float(), weight=proj_head.weight, bias=proj_head.bias)
        pred_proj = torch.nan_to_num(pred_proj, nan=0.0, posinf=1e4, neginf=-1e4)

        idx = idx_cpu.to(device, non_blocking=True)
        if idx.numel() == 0:
            continue
        dino_valid = batch["dino_feat"].index_select(0, idx_cpu).to(device, non_blocking=True).float()
        dino_valid = torch.nan_to_num(dino_valid, nan=0.0, posinf=1e4, neginf=-1e4)
        pred_valid = pred_proj.index_select(0, idx)

        loss = distillation_loss(pred_valid, dino_valid)
        total += float(loss.item())
        count += 1

        # free ASAP
        del coord, feat, grid_coord, batch_tensor, offset, feat_sel, input_feat
        del out, feats64, pred_proj, pred_valid, dino_valid, idx, idx_cpu, mask_cpu

    # restore modes
    if model_was_training: model.train()
    if isinstance(proj_head, torch.nn.Module) and proj_was_training: proj_head.train()

    return total / max(count, 1)


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
    added = {("module." + k if not k.startswith("module.") else k): v for k, v in state.items()}
    m.load_state_dict(added)


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
    feat_mode: str = "rvi",
    voxel_size: float = 0.10,
):
    # ---- Repro ----
    torch.use_deterministic_algorithms(False)
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    g = torch.Generator(device="cpu"); g.manual_seed(seed)

    device = torch.device(device)
    if device.type == "cuda":
        # Small perf hint
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    data_dir = Path(data_dir)
    print(f"[INFO] Starting training on device: {device}")
    print(f"[INFO] Training data: {data_dir}")

    dataset = TrainVoxDataset(data_dir, fallback_voxel_size=voxel_size)
    dataset.files = dataset.files[:(len(dataset)*80)//100]  # Use 80% of data for training
    using_trainvox = dataset.using_trainvox

    if workers is None:
        workers = _resolve_default_workers()
    workers = max(1, int(workers))
    print(f"[INFO] Using {workers} DataLoader workers for training...")
    print("[INFO] " + ("Using *_trainvox.pth" if using_trainvox else "FALLBACK to dense .pth → visible-first voxelization on the fly"))

    if len(dataset) == 0:
        raise RuntimeError(f"No training samples found in {data_dir}. Did preprocessing write *_trainvox.pth or dense .pth?")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_trainvox,
        multiprocessing_context="spawn",
        generator=g,
    )

    num_batches = len(dataloader)
    updates_per_epoch = math.ceil(max(1, num_batches) / max(1, accum_steps))

    print(f"Loaded {len(dataset)} preprocessed samples")
    print(f"Per-step batch size: {batch_size} | Accum steps: {accum_steps} | "
          f"Effective batch: {batch_size * accum_steps}")
    print(f"Epochs: {epochs} | Batches/epoch: {num_batches} | Updates/epoch: {updates_per_epoch} | Total Steps: {epochs * updates_per_epoch}")
    if using_trainvox:
        print(f"Single voxel size: {voxel_size:.3f} m")

    if total_steps:
        epochs = math.ceil(total_steps / updates_per_epoch)
        print(f"Will train for {epochs} epochs to hit ~{total_steps} optimizer steps.")

    # Probe sample & feature dims
    probe = dataset[0]
    feat_dim_total = probe["feat"].shape[1]
    col_map = {"r": 0, "v": 1, "i": 2}
    if feat_mode == "none":
        keep_cols = []
    else:
        keep_cols = [col_map[c] for c in feat_mode if c in col_map and col_map[c] < feat_dim_total]
    input_dim = probe["coord"].shape[1] + len(keep_cols)

    if "dino_feat" not in probe:
        raise RuntimeError("Training files must include 'dino_feat' (teacher features) for distillation.")
    dino_dim = probe["dino_feat"].shape[1]

    print(f"Using input_dim={input_dim} (feat_mode='{feat_mode}', keep_cols={keep_cols}), dino_dim={dino_dim}")

    # Model + projection head
    model = PointTransformerV3(
        in_channels=input_dim,
        enable_flash=False,
        enc_patch_size=(256, 256, 256, 256, 256),
        dec_patch_size=(256, 256, 256, 256),
        enable_rpe=False,
        upcast_attention=True,
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
        div_factor=10,
        final_div_factor=100
    )

    scaler = GradScaler(enabled=False)
    best_loss = float("inf")
    start_epoch = 0

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = out_dir / "latest_checkpoint.pth"
    best_ckpt_path = out_dir / "best_model.pth"

    # Optional resume
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

        warn_once_all_mask = True

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            try:
                if batch["coord"].numel() == 0:
                    if batch_idx % 20 == 0:
                        print(f"[WARN][Batch {batch_idx}] Empty coord tensor, skipping")
                    skipped_batches += 1; continue

                mask_cpu = batch["mask"].bool()
                if mask_cpu.sum().item() == 0:
                    if warn_once_all_mask:
                        print("[WARN] All points masked as non-visible in this batch; "
                              "loss will be computed on all points as a fallback.")
                        warn_once_all_mask = False
                    idx_cpu = torch.arange(batch["coord"].shape[0], dtype=torch.long)
                else:
                    idx_cpu = mask_cpu.nonzero(as_tuple=False).squeeze(1)

                # Move to device
                coord = batch["coord"].to(device, non_blocking=True).float()
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

                # Normalize coord/feat per-batch; then build input
                coord = (coord - coord.mean(dim=0)) / (coord.std(dim=0) + 1e-6)
                if feat_sel.numel() > 0:
                    feat_sel = (feat_sel - feat_sel.mean(dim=0)) / (feat_sel.std(dim=0) + 1e-6)
                input_feat = torch.cat([coord, feat_sel], dim=1)

                if batch_idx == 0:
                    print(f"[Batch 0] input_feat stats: mean={input_feat.mean().item():.4f}, std={input_feat.std().item():.4f}")

                if torch.isnan(input_feat).any() or torch.isinf(input_feat).any():
                    print(f"[ERROR][Batch {batch_idx}] NaN/Inf in input_feat, skipping batch")
                    skipped_batches += 1; continue
                if input_feat.numel() == 0:
                    print(f"[ERROR][Batch {batch_idx}] input_feat is empty, skipping batch")
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

                # Distill on visible rows only (or all rows if fallback triggered)
                idx = idx_cpu.to(device, non_blocking=True)
                if idx.numel() == 0:
                    skipped_batches += 1; continue
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

        denom_batches = counted_batches if counted_batches > 0 else 1
        avg_loss = total_loss_raw / denom_batches
        print(f"Avg Loss (per batch) = {avg_loss:.6f} | Updates this epoch = {update_count} | Skipped batches = {skipped_batches}")

        val_loss = None
        try:
            val_loss = evaluate_distill_loss(
                model if not isinstance(model, torch.nn.DataParallel) else model.module,
                proj_head if not isinstance(proj_head, torch.nn.DataParallel) else proj_head.module,
                val_dir=Path(data_dir),
                device=device,
                workers=workers,
                batch_size=max(4, batch_size // 2),
                voxel_size=voxel_size,
                feat_mode=feat_mode,
                prefetch_factor=prefetch_factor,
            )
            print(f"[VAL] Mean distillation loss on held-out = {val_loss:.6f}")
        except Exception as e:
            print(f"[VAL] Skipping validation due to error: {e}")

        # ---- Save latest and best checkpoints ----
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

        latest_ckpt_epoch = out_dir / f"latest_checkpoint_epoch{epoch+1:04d}.pth"
        torch.save(state, latest_ckpt_epoch)
        torch.save(state, latest_ckpt_path)
        print(f"Saved: {latest_ckpt_epoch.name} and latest_checkpoint.pth")

        """ if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(state, best_ckpt_path)
            print(f"Best model saved to {best_ckpt_path} (loss = {avg_loss:.6f})") """
        
        score = val_loss if (val_loss is not None and not math.isnan(val_loss)) else avg_loss
        if score < best_loss:
            best_loss = score
            torch.save(state, best_ckpt_path)
            print(f"Best model saved to {best_ckpt_path} "
                  f"({'val' if val_loss is not None else 'train'} loss = {score:.6f})")

        print(f"[Epoch {epoch+1}] Skipped {skipped_batches} batches due to errors.")
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    DATA_DIR = Path(os.getenv("PREPROCESS_OUTPUT_DIR"))
    TRAIN_CHECKPOINTS = Path(os.getenv("TRAIN_CHECKPOINTS"))
    FEAT_MODE = os.getenv("FEAT_MODE", "rvi")  # "rvi", "rv", "none", etc.
    RESULT_DIR = Path(os.getenv("RESULT_DIR"))

    train(
        data_dir=DATA_DIR,
        output_dir=TRAIN_CHECKPOINTS,
        epochs=50,
        workers=None,
        batch_size=4,
        accum_steps=8,
        prefetch_factor=2,
        lr=2e-3,
        use_data_parallel=False,
        feat_mode=FEAT_MODE,
        voxel_size=0.10,
    )

    print(f"[INFO] Training finished. Checkpoints are in {TRAIN_CHECKPOINTS}")
    print(f"[INFO] Writing config   to {RESULT_DIR}...")
    with open(RESULT_DIR / "train_config.txt", "w") as f:
        f.write(f"DATA_DIR={DATA_DIR}\n")
        f.write(f"TRAIN_CHECKPOINTS={TRAIN_CHECKPOINTS}\n")
        f.write(f"FEAT_MODE={FEAT_MODE}\n")
        f.write(f"RESULT_DIR={RESULT_DIR}\n")
        f.write(f"WORKERS={16}\n")
        f.write(f"ACCUM_STEPS={8}\n")
        f.write(f"LR={2e-3}\n")
        f.write(f"USE_DATA_PARALLEL={False}\n")
        f.write(f"VOXEL_SIZE={0.10}\n")
        f.write(f"EPOCHS={50}\n")
        f.write(f"BATCH_SIZE={4}\n")
        f.write(f"PREFETCH_FACTOR={2}\n")
        f.write(f"GRADIENT_ACCUMULATION_STEPS={8}\n")

    print("[INFO] Done.")