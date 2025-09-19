import itertools
import os
import csv
import datetime
import random
from pathlib import Path
from functools import partial
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.PTv3.model_ import PointTransformerV3

# Small utilities

def _load_state(m: torch.nn.Module, state):
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


def _safe_grid_coord(coord: torch.Tensor, grid_size: float, origin: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Build integer voxel coordinates from RAW coords (not normalized).
    origin: if provided (3,), use it as min; else compute per-sample min().
    """
    if origin is None:
        origin = coord.min(dim=0, keepdim=True).values
    else:
        origin = origin.view(1, 3).to(coord.device, coord.dtype)
    g = torch.floor((coord - origin) / (grid_size + 1e-8)).to(torch.int32)
    return g


def _unique_first_preferring_mask(indices_3d: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
    """
    Choose one index per (x,y,z) voxel row, preferring rows where visible_mask==True.
    Returns ascending ORIGINAL indices.
    """
    if indices_3d.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    g = indices_3d.to(torch.int64).contiguous()    # (N,3)
    vis = visible_mask.to(torch.bool).view(-1)

    # Build voxel key: key = x*(Ymax*Zmax) + y*Zmax + z
    max_vals = g.max(dim=0).values + 1
    max_vals = torch.clamp(max_vals, min=1)
    M_yz = max_vals[1] * max_vals[2]
    key = g[:, 0] * M_yz + g[:, 1] * max_vals[2] + g[:, 2]  # (N,)

    # Prefer visible points: stable sort by (key, flag) where flag=0 for visible, 1 for not visible
    flag = (~vis).to(torch.int64)
    composite = key * 2 + flag
    perm = torch.argsort(composite, stable=True)

    key_sorted = key[perm]
    first_mask = torch.ones_like(key_sorted, dtype=torch.bool)
    first_mask[1:] = key_sorted[1:] != key_sorted[:-1]

    first_sorted = perm[first_mask]
    sel = first_sorted.sort().values
    return sel


def _multiscale_voxel_select(
    coord: torch.Tensor,
    mask: torch.Tensor,
    grid_sizes: Tuple[float, float, float] = (0.05, 0.10, 0.20),
    r_bins: Tuple[float, float] = (30.0, 70.0),
) -> torch.Tensor:
    """
    Distance-aware voxel selection with visibility preference:
      - near (<= r_bins[0])    : small voxels (e.g., 5 cm)
      - mid  (r0, <= r_bins[1]): medium voxels (e.g., 10 cm)
      - far  (> r_bins[1])     : large voxels (e.g., 20 cm)
    Picks one point per voxel, preferring mask==True where available.
    Returns a 1-D index tensor selecting the chosen points (ascending original order).
    """
    if coord.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    r_xy = torch.linalg.norm(coord[:, :2], dim=1)
    near_mask = r_xy <= r_bins[0]
    mid_mask  = (r_xy > r_bins[0]) & (r_xy <= r_bins[1])
    far_mask  = r_xy > r_bins[1]

    sels: List[torch.Tensor] = []
    for bin_mask, gsize in zip((near_mask, mid_mask, far_mask), grid_sizes):
        if bin_mask.any():
            sub_idx   = bin_mask.nonzero(as_tuple=False).squeeze(1)
            sub_coord = coord.index_select(0, sub_idx)
            sub_grid  = _safe_grid_coord(sub_coord, float(gsize))
            sub_vis   = mask.index_select(0, sub_idx).to(torch.bool)
            sel_local = _unique_first_preferring_mask(sub_grid, sub_vis)
            if sel_local.numel() > 0:
                sels.append(sub_idx.index_select(0, sel_local))

    if len(sels) == 0:
        return torch.empty((0,), dtype=torch.long)

    sel = torch.cat(sels, dim=0)
    return sel.sort().values


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


def _safe_stem_from_rel(relpath) -> str:
    try:
        return Path(str(relpath)).stem
    except Exception:
        return ""


class InferenceDataset(torch.utils.data.Dataset):
    """
    Loads dense *.pth dumps written by preprocess.py (NOT *_trainvox.pth).
    Expects keys: coord (N,3), feat (N,F), mask (N,), grid_size (scalar), image_relpath (optional).
    """
    def __init__(self, root_dir: Path, voxel_size: Optional[float] = None):
        self.root_dir = Path(root_dir)
        files = sorted(self.root_dir.glob("*.pth"))
        # exclude train-voxel files if present in same folder
        self.files = [f for f in files if not f.name.endswith("_trainvox.pth")]
        if not self.files:
            raise FileNotFoundError(f"No dense .pth files found in {root_dir}")
        self.voxel_size = float(voxel_size) if voxel_size is not None else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        rec = torch.load(f, map_location="cpu")

        coord = rec["coord"].to(torch.float32)                        # (N,3)
        feat  = rec.get("feat", torch.empty((coord.shape[0], 0))).to(torch.float32)
        mask  = rec.get("mask", torch.ones((coord.shape[0],), dtype=torch.bool)).to(torch.bool)
        grid_size = float(rec.get("grid_size", 0.10))
        img_rel = rec.get("image_relpath", None)
        image_stem = _safe_stem_from_rel(img_rel) if img_rel is not None else ""

        out = {
            "coord": coord,
            "feat": feat,
            "mask": mask,
            "grid_size": torch.tensor(grid_size),
            "image_stem": image_stem,
            "lidar_stem": f.stem,
        }
        return out


def collate_for_ptv3(batch, *, multiscale_voxel: bool = False, voxel_size: float = 0.10):
    """
    Concatenate variable-length samples AFTER optional voxel downsampling.
    - If multiscale_voxel: distance-aware (5/10/20 cm) with visible-first preference.
    - Else: single voxel size (voxel_size) with visible-first preference.
    """
    collated = {
        "coord": [], "feat": [], "mask": [], "grid_coord": [], "grid_size": [],
        "batch": [], "offset": [], "lidar_stems": [], "image_stems": [],
    }
    offset = 0

    for bid, sample in enumerate(batch):
        coord = sample["coord"]
        feat  = sample["feat"]
        mask  = sample["mask"].to(torch.bool)
        gsize = float(sample["grid_size"])
        lidar_stem = sample["lidar_stem"]
        image_stem = sample["image_stem"]

        if multiscale_voxel:
            sel = _multiscale_voxel_select(coord, mask, grid_sizes=(0.05, 0.10, 0.20), r_bins=(30.0, 70.0))
            g_for_grid = 0.10  # any single number is fine for stats; we'll store mean later
        else:
            grid = _safe_grid_coord(coord, voxel_size)
            sel = _unique_first_preferring_mask(grid, mask)
            g_for_grid = voxel_size

        if sel.numel() > 0:
            coord = coord.index_select(0, sel)
            feat  = feat.index_select(0, sel) if feat.numel() else feat
            mask  = mask.index_select(0, sel)
            grid_coord = _safe_grid_coord(coord, g_for_grid)
        else:
            # keep empty sample (downstream will write an empty dump)
            grid_coord = torch.empty((0, 3), dtype=torch.int32)

        N = coord.shape[0]
        collated["coord"].append(coord)
        collated["feat"].append(feat)
        collated["mask"].append(mask)
        collated["grid_coord"].append(grid_coord)
        collated["grid_size"].append(g_for_grid)
        collated["lidar_stems"].append(lidar_stem)
        collated["image_stems"].append(image_stem)
        collated["batch"].append(torch.full((N,), bid, dtype=torch.long))
        offset += N
        collated["offset"].append(offset)

    # concat
    for k in ["coord", "feat", "mask", "grid_coord", "batch"]:
        collated[k] = torch.cat(collated[k], dim=0)
    collated["grid_size"] = torch.tensor(collated["grid_size"])
    collated["offset"]    = torch.tensor(collated["offset"], dtype=torch.long)
    return collated


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
    limit: Optional[int] = None,
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

    dataset = InferenceDataset(data_dir, voxel_size=voxel_size)

    if limit is not None and limit > 0:
        dataset.files = dataset.files[:limit]
        print(f"[inference] Limiting to first {limit} samples.") 
    try:
        dataset_len = len(dataset)
    except Exception:
        dataset_len = None

    collate_fn = partial(collate_for_ptv3, multiscale_voxel=multiscale_voxel, voxel_size=float(voxel_size))
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
    skipped_reasons: List[str] = []

    for batch in tqdm(loader, desc="Inference"):

        # Keep CPU copy for speed extraction later
        feat_cpu = batch["feat"]

        # Move tensors (device copy for forward)
        coord = batch["coord"].to(device, non_blocking=True)
        feat  = feat_cpu.to(device, non_blocking=True).float()
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
        has_vel = (feat_cpu.ndim == 2) and (feat_cpu.shape[1] >= 2)
        speed_all = feat_cpu[:, 1].abs().clone() if has_vel else torch.empty((0,), dtype=torch.float32)

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
        multiscale_voxel=False,   # set True to shrink dense frames faster at inference
        batch_size=4,
        workers=12,
        device="cuda" if torch.cuda.is_available() else "cpu",
        feat_mode="rvi",
    )
