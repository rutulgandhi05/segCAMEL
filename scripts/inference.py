import os
from pathlib import Path
from functools import partial
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.PTv3.model_ import PointTransformerV3

from scripts.train_segmentation import (
    PointCloudDataset,
    collate_for_ptv3,
    _load_state
)

# Inference
@torch.no_grad()
def run_inference(
    data_dir: Path,
    checkpoint_path: Path,
    out_dir: Path,
    *,
    voxel_size: float = 0.10,
    multiscale_voxel: bool = False,
    batch_size: int = 2,
    workers: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    feat_mode: str = "rvi",   # "none" | "ri" | "v" | "rvi"
):
    """
    Runs PTv3 forward (image-free) with the exact preprocessing used in training and
    writes per-sample outputs:
      - ptv3_feat (64-D), coord_norm, coord_raw, grid_coord, mask, grid_size, image_stem
      - speed (|v|) per point, if available from raw features (assumes feat=[reflectivity, velocity, intensity])
    """
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

    print(f"[inference] Data: {data_dir}")
    print(f"[inference] Checkpoint: {checkpoint_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PointCloudDataset(data_dir, voxel_size=voxel_size)
    collate_fn = partial(collate_for_ptv3, multiscale_voxel=multiscale_voxel)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
        collate_fn=collate_fn,
        multiprocessing_context="spawn",
        generator=g
    )

    # Probe dims (respect feat_mode selection)
    probe = dataset[0]
    feat_dim_total = probe["feat"].shape[1]
    col_map = {"r": 0, "v": 1, "i": 2}
    if feat_mode == "none":
        keep_cols = []
    else:
        keep_cols = [col_map[c] for c in feat_mode if c in col_map and col_map[c] < feat_dim_total]
    input_dim = probe["coord"].shape[1] + len(keep_cols)
    print(f"[inference] Using input_dim={input_dim} (feat_mode='{feat_mode}', keep_cols={keep_cols})")

    # Build model exactly like training
    model = PointTransformerV3(
        in_channels=input_dim,
        enable_flash=False,
        enc_patch_size=(256, 256, 256, 256, 256),
        dec_patch_size=(256, 256, 256, 256),
        enable_rpe=False,
        upcast_attention=True,
        upcast_softmax=True,
    ).to(device).eval()

    # Load checkpoint (model + head if present)
    print(f"[inference] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model" in ckpt:
        _load_state(model, ckpt["model"])
    else:
        _load_state(model, ckpt)

    for batch in tqdm(loader, desc="Inference"):

        # Move tensors
        coord = batch["coord"].to(device, non_blocking=True)
        feat  = batch["feat"].to(device, non_blocking=True).float()
        grid_coord = batch["grid_coord"].to(device, non_blocking=True)
        batch_tensor = batch["batch"].to(device, non_blocking=True)
        offset = batch["offset"].to(device, non_blocking=True)
        grid_size_val = batch["grid_size"].mean().item()

        # ---- Feature-channel selection (must match training) ----
        cols = {"r": 0, "v": 1, "i": 2}
        if feat_mode == "none":
            feat_sel = torch.empty(feat.shape[0], 0, device=feat.device, dtype=feat.dtype)
        else:
            keep = [cols[c] for c in feat_mode if c in cols and cols[c] < feat.shape[1]]
            feat_sel = feat[:, keep] if len(keep) else torch.empty(feat.shape[0], 0, device=feat.device, dtype=feat.dtype)

        # Same normalization as training (+ optional clamp to tame outliers)
        coord = (coord - coord.mean(dim=0)) / (coord.std(dim=0) + 1e-6)
        if feat_sel.numel() > 0:
            feat_sel = (feat_sel - feat_sel.mean(dim=0)) / (feat_sel.std(dim=0) + 1e-6)
        input_feat = torch.cat([coord, feat_sel], dim=1).clamp_(-8.0, 8.0)

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
   
        # Split and save per sample
        lidar_stems = batch["lidar_stems"]
        image_stems = batch["image_stems"]

        # NEW: derive per-point speed from RAW (CPU) features if available
        # assumes raw feat columns are [reflectivity, velocity, intensity]
        feat_raw_cpu = batch["feat"]  # this is still on CPU from the DataLoader
        has_vel = (feat_raw_cpu.ndim == 2) and (feat_raw_cpu.shape[1] >= 2)
        speed_all = feat_raw_cpu[:, 1].abs().clone() if has_vel else torch.empty((0,), dtype=torch.float32)

        B = len(lidar_stems)
        for b in range(B):
            idx_b = (batch_tensor == b).nonzero(as_tuple=False).squeeze(1)
            idx_b_cpu = idx_b.cpu()

            payload = {
                "lidar_stem": lidar_stems[b],
                "ptv3_feat": feats.index_select(0, idx_b).detach().cpu(),   # (Nb,64)
                "coord_norm": coord.index_select(0, idx_b).detach().cpu(),  # normalized coords used in model
                "coord_raw": batch["coord"].index_select(0, idx_b_cpu).clone(),
                "grid_coord": grid_coord.index_select(0, idx_b).detach().cpu(),
                "mask": batch["mask"].index_select(0, idx_b_cpu).cpu() if batch["mask"].numel() else torch.empty((0,), dtype=torch.bool),
                "grid_size": torch.tensor(grid_size_val),
                "image_stem": image_stems[b],
                "speed": speed_all.index_select(0, idx_b_cpu) if has_vel else torch.empty((0,), dtype=torch.float32),
            }

            save_path = out_dir / f"{image_stems[b]}_inference.pth"
            torch.save(payload, save_path)

        # Cleanup
        del coord, feat, grid_coord, batch_tensor, offset, input_feat, output, feats, image_stems, lidar_stems, payload, feat_raw_cpu, speed_all, has_vel, idx_b, idx_b_cpu

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[inference] Done.")


if __name__ == "__main__":
  
    DATA_DIR = Path(os.getenv("PREPROCESS_OUTPUT_DIR"))
    CHECKPOINTS_DIR = Path(os.getenv("TRAIN_CHECKPOINTS"))
    CHECKPOINT_PATH = CHECKPOINTS_DIR / "best_model.pth"
    OUT_DIR = Path(os.getenv("INFERENCE_OUTPUT_DIR"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_inference(
        data_dir=DATA_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        out_dir=OUT_DIR,
        voxel_size=0.10,
        multiscale_voxel=False,
        batch_size=4,
        workers=12,
        device="cuda" if torch.cuda.is_available() else "cpu",
        feat_mode="rvi",
    )
