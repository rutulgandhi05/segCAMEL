import os
from pathlib import Path
from functools import partial
import random
import numpy as np
import csv
import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.PTv3.model_ import PointTransformerV3

from scripts.train_segmentation import (
    PointCloudDataset,
    collate_for_ptv3,
    _load_state
)


def _unique_save_path(out_dir: Path, image_stem: str, lidar_stem: str) -> Path:
    """
    Create a collision-safe path: <image>__<lidar>[_N]_inference.pth
    so repeated basenames or multi-cam setups never overwrite.
    """
    base = f"{image_stem}__{lidar_stem}_inference.pth"
    path = out_dir / base
    if not path.exists():
        return path
    i = 1
    while True:
        cand = out_dir / f"{image_stem}__{lidar_stem}__{i}_inference.pth"
        if not cand.exists():
            return cand
        i += 1

def _manifest_open(out_dir: Path):
    """Open (append) a CSV manifest logging each frame's status."""
    mpath = out_dir / "inference_manifest.csv"
    new = not mpath.exists()
    f = open(mpath, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp_utc", "image_stem", "lidar_stem", "status", "reason", "path"])
    return f, w

def _save_empty_dump(out_dir: Path, image_stem: str, lidar_stem: str, grid_size: float):
    """
    Write an empty but well-formed dump so downstream code still works
    and counts can match 1:1 with preprocessing.
    """
    path = _unique_save_path(out_dir, image_stem, lidar_stem)
    payload = {
        "image_stem": image_stem,
        "lidar_stem": lidar_stem,
        "ptv3_feat": torch.empty((0, 64), dtype=torch.float32),  # N×64
        "coord_norm": torch.empty((0, 3), dtype=torch.float32),
        "coord_raw":  torch.empty((0, 3), dtype=torch.float32),
        "grid_coord": torch.empty((0, 3), dtype=torch.int32),
        "mask":       torch.empty((0,),   dtype=torch.bool),
        "grid_size":  float(grid_size),
        "speed":      torch.empty((0,),   dtype=torch.float32),
    }
    torch.save(payload, path)
    return path


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
    try:
        dataset_len = len(dataset)
    except Exception:
        dataset_len = None

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

    # bookkeeping + manifest
    manifest_fh, manifest_writer = _manifest_open(out_dir)
    written = 0
    empty_written = 0
    skipped = 0
    skipped_reasons = []

    for batch in tqdm(loader, desc="Inference"):

        # Move tensors (keep original CPU batch around for raw copies)
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

        try:
            output = model(data_dict)
            feats = torch.nan_to_num(output.feat, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-20, 20)
        except Exception as e:
            # whole-batch failure: mark all items as skipped
            now = datetime.datetime.utcnow().isoformat()
            for img_stem, lid_stem in zip(batch["image_stems"], batch["lidar_stems"]):
                manifest_writer.writerow([now, img_stem, lid_stem, "skipped", f"forward_error:{e}", ""])
                skipped += 1
                skipped_reasons.append(f"{img_stem}__{lid_stem}: forward_error:{e}")
            continue

        # Split and save per sample
        lidar_stems = batch["lidar_stems"]
        image_stems = batch["image_stems"]

        # derive per-point speed from RAW (CPU) features if available
        # assumes raw feat columns are [reflectivity, velocity, intensity]
        feat_raw_cpu = batch["feat"]  # this remains on CPU from DataLoader
        has_vel = (feat_raw_cpu.ndim == 2) and (feat_raw_cpu.shape[1] >= 2)
        speed_all = feat_raw_cpu[:, 1].abs().clone() if has_vel else torch.empty((0,), dtype=torch.float32)

        B = len(lidar_stems)
        for b in range(B):
            img_stem = image_stems[b]
            lid_stem = lidar_stems[b]
            now = datetime.datetime.utcnow().isoformat()

            try:
                # indices for this item
                idx_b = (batch_tensor == b).nonzero(as_tuple=False).squeeze(1)
                Ni = int(idx_b.numel())

                if Ni == 0:
                    # no points survived voxelization -> write empty dump
                    p = _save_empty_dump(out_dir, img_stem, lid_stem, grid_size_val)
                    manifest_writer.writerow([now, img_stem, lid_stem, "empty", "no_points_after_voxelization", str(p)])
                    empty_written += 1
                    written += 1
                    continue

                idx_b_cpu = idx_b.cpu()

                payload = {
                    "lidar_stem": lid_stem,
                    "ptv3_feat": feats.index_select(0, idx_b).detach().cpu(),   # (Nb,64)
                    "coord_norm": coord.index_select(0, idx_b).detach().cpu(),  # normalized coords used in model
                    "coord_raw": batch["coord"].index_select(0, idx_b_cpu).clone(),
                    "grid_coord": grid_coord.index_select(0, idx_b).detach().cpu(),
                    "mask": batch["mask"].index_select(0, idx_b_cpu).cpu() if batch["mask"].numel() else torch.empty((0,), dtype=torch.bool),
                    "grid_size": torch.tensor(grid_size_val),
                    "image_stem": img_stem,
                    "speed": speed_all.index_select(0, idx_b_cpu) if has_vel else torch.empty((0,), dtype=torch.float32),
                }

                save_path = _unique_save_path(out_dir, img_stem, lid_stem)
                torch.save(payload, save_path)
                manifest_writer.writerow([now, img_stem, lid_stem, "ok", "", str(save_path)])
                written += 1

            except Exception as e:
                skipped += 1
                skipped_reasons.append(f"{img_stem}__{lid_stem}: {e}")
                manifest_writer.writerow([now, img_stem, lid_stem, "skipped", str(e), ""])
                # continue to next item

        # Cleanup device allocations between batches
        del coord, feat, grid_coord, batch_tensor, offset, input_feat, output, feats

    # finalize manifest + summary
    try:
        manifest_fh.close()
    except Exception:
        pass

    wrote_files = len(list(Path(out_dir).glob("*_inference.pth")))
    if dataset_len is None:
        print(f"[inference] wrote_files={wrote_files} | empty={empty_written} | skipped={skipped}")
    else:
        delta = dataset_len - wrote_files
        print(f"[inference] dataset_len={dataset_len} | wrote_files={wrote_files} | empty={empty_written} | skipped={skipped} | delta={delta}")

    if skipped > 0:
        print("[inference] skipped frames (first 10):")
        for r in skipped_reasons[:10]:
            print("  -", r)
        print(f"[inference] Full list in {out_dir/'inference_manifest.csv'}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[inference] Saved inference outputs to:", out_dir)
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
