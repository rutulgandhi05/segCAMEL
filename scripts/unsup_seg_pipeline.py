from pathlib import Path
from typing import Iterable, Dict, List, Optional
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scripts.metrics import compute_all_metrics, MetricsConfig


# Utilities
def _iter_inference_dumps(infer_dir: Path) -> Iterable[Path]:
    """Yield per-sample *_inference.pth files in a stable order."""
    infer_dir = Path(infer_dir)
    files = sorted(infer_dir.glob("*_inference.pth"))
    if not files:
        raise FileNotFoundError(f"No *_inference.pth files found in {infer_dir}")
    for f in files:
        yield f


def _load_dump(path: Path) -> Dict[str, torch.Tensor]:
    """
    Load a single inference dump produced by scripts/inference.py.
    Expected keys:
      - ptv3_feat: (N,64) float32
      - coord_norm: (N,3) float32
      - coord_raw: (N,3) float32
      - grid_coord: (N,3) int32
      - mask: (N,) bool
      - grid_size: scalar tensor (float)
      - image_stem: str
      - speed: (N,) float32  [optional, if saved by inference]
    """
    payload = torch.load(path, map_location="cpu")
    if "image_stem" not in payload:
        # backward-compat
        payload["image_stem"] = path.stem.replace("_inference", "")
    # ensure dtypes
    if payload["ptv3_feat"].dtype != torch.float32:
        payload["ptv3_feat"] = payload["ptv3_feat"].float()
    if payload["coord_norm"].dtype != torch.float32:
        payload["coord_norm"] = payload["coord_norm"].float()
    if payload["coord_raw"].dtype != torch.float32:
        payload["coord_raw"] = payload["coord_raw"].float()
    if payload["grid_coord"].dtype != torch.int32:
        payload["grid_coord"] = payload["grid_coord"].to(torch.int32)
    if payload["mask"].dtype != torch.bool:
        payload["mask"] = payload["mask"].to(torch.bool)
    # optional speed
    if "speed" in payload and payload["speed"] is not None:
        if payload["speed"].numel() and payload["speed"].dtype != torch.float32:
            payload["speed"] = payload["speed"].float()
    else:
        payload["speed"] = torch.empty((0,), dtype=torch.float32)
    return payload


# 1) Feature stream (from saved dumps — NO backbone inference here)
def extract_features(
    infer_dir: Path,
    *,
    return_iter: bool = False,
) -> Iterable[Dict[str, torch.Tensor]] | List[Dict[str, torch.Tensor]]:
    """
    Stream features from the per-sample files saved by scripts/inference.py.

    Yields dictionaries with:
      - file_stem: str
      - feat64: (N,64) float32    (student backbone features)
      - coord_norm: (N,3) float32      (normalized coords used by the model)
      - coord_raw: (N,3) float32      (raw coords used by the model)
      - grid_coord: (N,3) int32
      - mask: (N,) bool
      - grid_size: float
      - speed: (N,) float32 (optional; empty if not available)
    """
    def _stream():
        for f in _iter_inference_dumps(infer_dir):
            payload = _load_dump(f)
            yield {
                "file_stem": payload["image_stem"],
                "feat64": payload["ptv3_feat"],     # already 64-D backbone features
                "coord_norm": payload["coord_norm"],          # normalized coords (as in model)
                "coord_raw": payload["coord_raw"],
                "grid_coord": payload["grid_coord"],
                "mask": payload["mask"],
                "grid_size": float(payload["grid_size"].item()) if torch.is_tensor(payload["grid_size"]) else float(payload["grid_size"]),
                "speed": payload.get("speed", torch.empty((0,), dtype=torch.float32)),
            }

    return _stream() if return_iter else list(_stream())


# 2) Spherical mini-batch K-Means (cosine) over the feature stream
@torch.no_grad()
def learn_prototypes_from_dataset(
    infer_dir: Path,
    *,
    k: int = 20,
    max_passes: int = 2,
    sample_per_frame: int = 20000,
    seed: int = 0,
) -> torch.Tensor:
    """
    Learn K unit-norm prototypes in 64-D from saved feature dumps.
    - sample_per_frame: subsample up to this many points per file to limit compute.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Warm-start centroids from first pass
    centroids = None
    buf, seen = [], 0
    for item in extract_features(infer_dir, return_iter=False):
        X = item["feat64"]
        if X.numel() == 0:
            continue
        X = F.normalize(X, dim=1)
        if X.shape[0] > sample_per_frame:
            sel = torch.from_numpy(rng.choice(X.shape[0], size=sample_per_frame, replace=False)).long()
            X = X.index_select(0, sel)
        buf.append(X); seen += X.shape[0]
        if seen >= k * 50:  # heuristic: ~50 samples per centroid to seed
            X0 = torch.cat(buf, dim=0)
            perm = torch.randperm(X0.shape[0])
            centroids = F.normalize(X0[perm[:k]].clone(), dim=1)
            break
    if centroids is None:
        X0 = torch.cat(buf, dim=0) if buf else torch.randn(k, 64)
        centroids = F.normalize(X0[torch.randperm(X0.shape[0])[:k]].clone(), dim=1)

    # Mini-batch updates over multiple passes
    for _ in tqdm(range(max_passes), desc="Learning prototypes"):
        accum = torch.zeros_like(centroids)
        counts = torch.zeros((k,), dtype=torch.long)
        for item in extract_features(infer_dir, return_iter=True):
            X = item["feat64"]
            if X.numel() == 0:
                continue
            X = F.normalize(X, dim=1)
            if X.shape[0] > sample_per_frame:
                sel = torch.from_numpy(rng.choice(X.shape[0], size=sample_per_frame, replace=False)).long()
                X = X.index_select(0, sel)

            sim = X @ centroids.T    # cosine with unit-norm vectors
            idx = sim.argmax(dim=1)
            for c in range(k):
                m = (idx == c)
                if m.any():
                    accum[c] += X[m].sum(dim=0)
                    counts[c] += int(m.sum())

        for c in range(k):
            if counts[c] > 0:
                centroids[c] = accum[c] / counts[c].clamp(min=1)
        centroids = F.normalize(centroids, dim=1)

    return centroids.cpu()


# 3) Fast voxel-neighborhood smoothing (CPU)
def smooth_labels_voxel(
    grid_coord: torch.Tensor,
    labels: np.ndarray,
    iters: int = 1,
    neighbor_range: int = 1,   # 3x3x3 neighborhood by default
    min_component: int = 50,   # snap tiny components to neighbor mode
) -> np.ndarray:
    if grid_coord.numel() == 0 or labels.size == 0:
        return labels

    g = grid_coord.cpu().numpy().astype(np.int64)
    lbl = labels.copy()

    # pack 3D indices into 64-bit key (21 bits per axis)
    key = (g[:, 0] << 42) + (g[:, 1] << 21) + g[:, 2]
    vox2idx: Dict[int, List[int]] = defaultdict(list)
    for i, k in enumerate(key):
        vox2idx[int(k)].append(i)

    offs = np.array([(dx, dy, dz)
                     for dx in range(-neighbor_range, neighbor_range + 1)
                     for dy in range(-neighbor_range, neighbor_range + 1)
                     for dz in range(-neighbor_range, neighbor_range + 1)], dtype=np.int64)

    for _ in range(iters):
        new_lbl = lbl.copy()
        for k, idxs in vox2idx.items():
            if not idxs:
                continue
            z =  k        & ((1 << 21) - 1)
            y = (k >> 21) & ((1 << 21) - 1)
            x = (k >> 42) & ((1 << 21) - 1)
            neigh_keys = ((x + offs[:, 0]) << 42) + ((y + offs[:, 1]) << 21) + (z + offs[:, 2])

            votes = []
            for nk in neigh_keys:
                js = vox2idx.get(int(nk))
                if js:
                    votes.extend(lbl[js])
            if votes:
                vals, counts = np.unique(np.array(votes, dtype=lbl.dtype), return_counts=True)
                new_lbl[idxs] = vals[counts.argmax()]
        lbl = new_lbl

    if min_component > 0:
        counts = defaultdict(int)
        lv = list(zip(lbl.tolist(), key.tolist()))
        for pair in lv:
            counts[pair] += 1
        small = {pair for pair, c in counts.items() if c < min_component}
        if small:
            for i, pair in enumerate(lv):
                if pair in small:
                    x = (pair[1] >> 42) & ((1 << 21) - 1)
                    y = (pair[1] >> 21) & ((1 << 21) - 1)
                    z =  pair[1]        & ((1 << 21) - 1)
                    neigh_keys = ((x + offs[:, 0]) << 42) + ((y + offs[:, 1]) << 21) + (z + offs[:, 2])
                    votes = []
                    for nk in neigh_keys:
                        js = vox2idx.get(int(nk))
                        if js:
                            votes.extend(lbl[js])
                    if votes:
                        vals, counts2 = np.unique(np.array(votes, dtype=lbl.dtype), return_counts=True)
                        lbl[i] = vals[counts2.argmax()]
    return lbl


# 4) Segmentation (nearest prototype + optional smoothing)
@torch.no_grad()
def segment_dataset(
    infer_dir: Path,
    centroids: torch.Tensor,     # (K,64) unit-norm
    *,
    smoothing_iters: int = 1,
    neighbor_range: int = 1,
    min_component: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Returns: dict[file_stem] -> cluster_id (N,) np.int64
    """
    C = F.normalize(centroids.to(torch.float32), dim=1)
    results: Dict[str, np.ndarray] = {}

    for item in tqdm(extract_features(infer_dir, return_iter=False), desc="Segmenting dataset"):
        stem = item["file_stem"]
        Z = item["feat64"]  # (N,64)
        if Z.numel() == 0:
            results[stem] = np.empty((0,), dtype=np.int64)
            continue

        Z = F.normalize(Z, dim=1).to(torch.float32)
        idx = (Z @ C.T).argmax(dim=1).cpu().numpy().astype(np.int64)

        if smoothing_iters > 0:
            idx = smooth_labels_voxel(
                grid_coord=item["grid_coord"],
                labels=idx,
                iters=smoothing_iters,
                neighbor_range=neighbor_range,
                min_component=min_component,
            )
        results[stem] = idx

    return results


# 5) Optional: PLY export for quick inspection
def save_colorized_ply(
    path: Path,
    coord: torch.Tensor,     # (N,3) float32  (NOTE: these coords are normalized if taken from dumps)
    labels: np.ndarray,      # (N,) int
    palette: Optional[np.ndarray] = None,
):
    """
    Writes ASCII PLY colored by cluster id. If you want *raw-scale* coordinates,
    you must save them during preprocessing/inference and pass them here.
    """
    xyz = coord.cpu().numpy().astype(np.float32)
    lab = labels.astype(np.int32)
    K = int(lab.max()) + 1 if lab.size else 1
    if palette is None:
        rng = np.random.default_rng(0)
        palette = (rng.uniform(60, 255, size=(K, 3))).astype(np.uint8)
        palette[0] = np.array([200, 200, 200], dtype=np.uint8)
    rgb = palette[lab % palette.shape[0]]

    header = "\n".join([
        "ply", "format ascii 1.0", f"element vertex {xyz.shape[0]}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header"
    ])
    with open(path, "w") as f:
        f.write(header + "\n")
        for i in range(xyz.shape[0]):
            r, g, b = int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} {r} {g} {b}\n")


# 6) NEW: Metrics evaluation over the whole dataset
def evaluate_dataset_metrics(
    infer_dir: Path,
    labels_per_frame: Dict[str, np.ndarray],
    out_csv: Path,
    *,
    sample_n: int = 200_000,
    seed: int = 42,
    q_bins: int = 4,
    tau_list: Optional[List[float]] = None,       # defaults to [0.2,0.4,0.6] (quantiles)
    tau_policy: str = "quantile",
) -> Dict[str, Optional[float]]:
    """
    Aggregates features/labels (and optional speeds & visibility) across all frames,
    computes metrics via scripts.metrics.compute_all_metrics, and writes a CSV.

    Returns the metrics dict as well.
    """
    feats_all, labels_all, speeds_all, vis_all = [], [], [], []

    # Iterate inference dumps in a stable order
    for f in _iter_inference_dumps(infer_dir):
        payload = _load_dump(f)
        stem = payload["image_stem"]
        if stem not in labels_per_frame:
            continue

        lbl = labels_per_frame[stem]
        N_dump = payload["ptv3_feat"].shape[0]
        if len(lbl) != N_dump:
            # Mismatch should not happen if labels were produced from the same dumps
            raise ValueError(f"Label length mismatch for {stem}: labels={len(lbl)} vs dump={N_dump}")

        feats_all.append(payload["ptv3_feat"])
        labels_all.append(torch.as_tensor(lbl, dtype=torch.int64))
        # Optional speed & mask
        if "speed" in payload and payload["speed"].numel():
            speeds_all.append(payload["speed"])
        else:
            # Keep shapes aligned; if missing, skip later
            speeds_all.append(torch.empty((0,), dtype=torch.float32))
        vis_all.append(payload["mask"].to(torch.bool) if payload["mask"].numel() else torch.zeros(N_dump, dtype=torch.bool))

    if not feats_all:
        raise RuntimeError("No features found to evaluate metrics.")

    X = torch.cat(feats_all, dim=0).numpy()
    Y = torch.cat(labels_all, dim=0).numpy()
    M = torch.cat(vis_all, dim=0).numpy()
    V = None
    # If at least one frame had speed data and all frames concatenated match a non-zero size
    if any(s.numel() for s in speeds_all):
        # For frames without speeds we appended empty tensors; concat only non-empty
        speeds_nonempty = [s for s in speeds_all if s.numel()]
        if speeds_nonempty:
            V = torch.cat(speeds_nonempty, dim=0).numpy()

    cfg = MetricsConfig(sample_n=sample_n, seed=seed, q_bins=q_bins, tau_list=tau_list, tau_policy=tau_policy)
    results = compute_all_metrics(X, Y, speeds=V, visible_mask=M, token_labels_proj=None, cfg=cfg)

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as w:
        w.write("metric,value\n")
        for k, v in results.items():
            w.write(f"{k},{'' if v is None else v}\n")

    print(f"[metrics] wrote {out_csv}")
    return results
