from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Union
from collections import defaultdict
import json, io, zipfile
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---- robust import for metrics (package or top-level; safe fallback) ----
_HAS_METRICS = True
try:
    from scripts.metrics import compute_all_metrics, MetricsConfig  # package form
except Exception:
    try:
        from metrics import compute_all_metrics, MetricsConfig      # flat-file form
    except Exception:
        _HAS_METRICS = False
        class MetricsConfig:
            def __init__(self, **kwargs): pass
        def compute_all_metrics(*args, **kwargs):
            return {}

# ------------------------------------------------------------------------ #
#                                I/O utils                                 #
# ------------------------------------------------------------------------ #

def _load_dump(path: Path) -> Dict[str, torch.Tensor]:
    """Load a single *_inference.pth and normalize dtypes/fields."""
    payload = torch.load(path, map_location="cpu")
    if "image_stem" not in payload:
        payload["image_stem"] = path.stem.replace("_inference", "")
    # standardize dtypes
    payload["ptv3_feat"] = payload["ptv3_feat"].float()
    payload["coord_norm"] = payload["coord_norm"].float()
    payload["coord_raw"]  = payload["coord_raw"].float()
    payload["grid_coord"] = payload["grid_coord"].to(torch.int32)
    payload["mask"]       = payload["mask"].to(torch.bool)
    if "speed" in payload and payload["speed"] is not None:
        payload["speed"] = payload["speed"].float()
    else:
        payload["speed"] = torch.empty((0,), dtype=torch.float32)
    return payload

class DumpDataset(Dataset):
    """
    Dataset over inference dumps. Produces dict with:
      file_stem (unique), feat64, coord_norm, coord_raw, grid_coord, mask, grid_size, speed
    """
    def __init__(self, infer_dir: Path):
        self.paths = sorted(Path(infer_dir).glob("*_inference.pth"))
        if not self.paths:
            raise FileNotFoundError(f"No *_inference.pth files found in {infer_dir}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, i: int):
        path = self.paths[i]
        p = _load_dump(path)

        # --- build a UNIQUE stem ---
        stem_from_path = path.stem.replace("_inference", "")
        img = p.get("image_stem", "")
        lid = p.get("lidar_stem", "")
        if isinstance(img, str) and isinstance(lid, str) and img and lid:
            file_stem = f"{img}__{lid}"
        else:
            file_stem = stem_from_path  # unique per file on disk

        return {
            "file_stem":  file_stem,
            "feat64":     p["ptv3_feat"],
            "coord_norm": p["coord_norm"],
            "coord_raw":  p["coord_raw"],
            "grid_coord": p["grid_coord"],
            "mask":       p["mask"],
            "grid_size":  float(p["grid_size"].item()) if torch.is_tensor(p["grid_size"]) else float(p["grid_size"]),
            "speed":      p.get("speed", torch.empty((0,), dtype=torch.float32)),
        }

# --- top-level collate (picklable; replaces lambda) ---
def _collate_keep(batch):
    # DataLoader gives list[dict]; we keep it as-is and iterate inner dicts.
    return batch

def _make_loader(
    infer_dir: Path,
    *,
    workers: int = 8,
    prefetch: int = 4,
    batch_io: int = 32,
    pin_memory: bool = True,
) -> DataLoader:
    ds = DumpDataset(infer_dir)
    return DataLoader(
        ds,
        batch_size=max(1, int(batch_io)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=bool(pin_memory),
        persistent_workers=(workers > 0),
        prefetch_factor=max(2, int(prefetch)) if workers > 0 else None,
        collate_fn=_collate_keep,
    )

# ------------------------------------------------------------------------ #
#                           Simple JSON helpers                             #
# ------------------------------------------------------------------------ #

def save_run_config(path: Path, cfg: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

def load_run_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

# ------------------------------------------------------------------------ #
#                          Feature construction                            #
# ------------------------------------------------------------------------ #

def _build_features(item: Dict[str, torch.Tensor], feature_cfg: Optional[Dict] = None) -> torch.Tensor:
    """
    Build clustering features.
    - Always includes the 64-D distilled point features.
    - Optionally concatenates: range (m), height z (m), |v| speed (m/s).
      Each extra channel is scaled so magnitudes are comparable to unit-normed 64-D.
    """
    if feature_cfg is None:
        feature_cfg = {}
    X = item["feat64"]
    comps = [X]  # (N,64)

    if feature_cfg.get("use_range", False):
        r = torch.linalg.norm(item["coord_raw"], dim=1)
        r = (r / float(feature_cfg.get("range_scale", 50.0))).unsqueeze(1).to(torch.float32)
        comps.append(r)

    if feature_cfg.get("use_height", False):
        z = (item["coord_raw"][:, 2] / float(feature_cfg.get("height_scale", 5.0))).unsqueeze(1).to(torch.float32)
        comps.append(z)

    if feature_cfg.get("use_speed", False):
        s = item.get("speed", None)
        if s is not None and s.numel() == X.shape[0]:
            # --- range-aware dead-zone: tau(r) = clip(a*r, t_min, t_max) ---
            a     = float(feature_cfg.get("speed_deadzone_per_m", 0.02))  # m/s per meter
            t_min = float(feature_cfg.get("speed_deadzone_min", 0.10))    # m/s
            t_max = float(feature_cfg.get("speed_deadzone_max", 0.80))    # m/s
            r     = torch.linalg.norm(item["coord_raw"], dim=1)
            tau   = torch.clamp(a * r, min=t_min, max=t_max)
            s_filt = torch.where(s.abs() < tau, torch.zeros_like(s), s)
            s_scaled = (s_filt / float(feature_cfg.get("speed_scale", 25.0))).unsqueeze(1).to(torch.float32)
            comps.append(s_scaled)

    # optional: signed, two-channel variant (direction only when confident)
    if feature_cfg.get("use_speed_signed", False) and ("vel_signed" in item):
        v = item["vel_signed"]
        a     = float(feature_cfg.get("speed_deadzone_per_m", 0.02))
        t_min = float(feature_cfg.get("speed_deadzone_min", 0.10))
        t_max = float(feature_cfg.get("speed_deadzone_max", 0.80))
        r     = torch.linalg.norm(item["coord_raw"], dim=1)
        tau   = torch.clamp(a * r, min=t_min, max=t_max)
        v_pos = torch.where(v >  tau,  v, torch.zeros_like(v))
        v_neg = torch.where(v < -tau, -v, torch.zeros_like(v))
        comps.append((v_pos / float(feature_cfg.get("speed_scale", 25.0))).unsqueeze(1).to(torch.float32))
        comps.append((v_neg / float(feature_cfg.get("speed_scale", 25.0))).unsqueeze(1).to(torch.float32))
        
    return X if len(comps) == 1 else torch.cat(comps, dim=1)

# ------------------------------------------------------------------------ #
#        Distance bins + stratified subsampling (for proto learning)       #
# ------------------------------------------------------------------------ #

def _distance_bins(coord_raw: torch.Tensor, edges: List[float]) -> List[torch.Tensor]:
    r = torch.linalg.norm(coord_raw, dim=1)
    if edges is None or len(edges) <= 1:
        return [(r >= 0.0)]
    masks = []
    for i in range(len(edges)):
        if i == 0:
            lo, hi = 0.0, edges[1]
        elif i < len(edges) - 1:
            lo, hi = edges[i], edges[i+1]
        else:
            lo, hi = edges[-1], float("inf")
        masks.append((r >= lo) & (r < hi))
    return masks

def _align_ratios_to_bins(ratios: Optional[List[float]], bin_count: int) -> np.ndarray:
    if bin_count <= 0:
        return np.array([], dtype=np.float64)
    if ratios is None or len(ratios) == 0:
        arr = np.ones(bin_count, dtype=np.float64) / float(bin_count)
        return arr
    arr = np.array(ratios, dtype=np.float64)
    if arr.size < bin_count:
        pad_val = arr[-1] if arr.size > 0 else 1.0
        arr = np.pad(arr, (0, bin_count - arr.size), mode="constant", constant_values=pad_val)
    elif arr.size > bin_count:
        arr = arr[:bin_count]
    s = arr.sum()
    if s <= 0:
        arr = np.ones(bin_count, dtype=np.float64) / float(bin_count)
    else:
        arr = arr / s
    return arr

def _stratified_subsample(
    X: torch.Tensor,
    coord_raw: torch.Tensor,
    total_n: int,
    ratios: Optional[List[float]],
    edges: Optional[List[float]],
    rng: np.random.Generator,
    mask_filter: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    N = X.shape[0]
    if N == 0 or total_n <= 0:
        return X
    valid = mask_filter if (mask_filter is not None and mask_filter.numel() == N) else torch.ones((N,), dtype=torch.bool, device=X.device)
    bins = [b & valid for b in _distance_bins(coord_raw, edges or [])]
    bin_count = len(bins)
    if bin_count == 0:
        return X
    ratios_aligned = _align_ratios_to_bins(ratios, bin_count)

    idx_keep = []
    for bi, b in enumerate(bins):
        n_b = int(round(total_n * float(ratios_aligned[bi])))
        ids = torch.nonzero(b, as_tuple=False).squeeze(1).cpu().numpy()
        if ids.size == 0:
            continue
        if ids.size <= n_b:
            idx_keep.append(torch.from_numpy(ids))
        else:
            sel = rng.choice(ids, size=n_b, replace=False)
            idx_keep.append(torch.from_numpy(sel))
    if not idx_keep:
        return X
    idx_keep = torch.cat(idx_keep, dim=0).long()
    return X.index_select(0, idx_keep)

# ------------------------------------------------------------------------ #
#                     K-means (cosine) prototype learning                  #
# ------------------------------------------------------------------------ #

def _kmeanspp_init(x_unit: torch.Tensor, k: int, dev: torch.device, seed: int) -> torch.Tensor:
    assert x_unit.ndim == 2 and x_unit.shape[0] >= k, "Insufficient samples for k-means++ seeding"
    gen = torch.Generator(device=dev).manual_seed(seed)
    i0 = torch.randint(x_unit.shape[0], (1,), generator=gen, device=dev).item()
    cents = [x_unit[i0:i0+1]]
    for _ in range(1, k):
        sims = torch.stack([x_unit @ c.t() for c in cents], dim=2).squeeze(1)  # (N, C)
        max_sim = sims.max(dim=1).values
        d2 = (1.0 - max_sim).clamp_min_(0.0)
        probs = (d2 + 1e-12) / (d2.sum() + 1e-12)
        idx = torch.multinomial(probs, 1, generator=gen).item()
        cents.append(x_unit[idx:idx+1])
    return F.normalize(torch.cat(cents, dim=0), dim=1)

@torch.no_grad()
def learn_prototypes_from_dataset(
    infer_dir: Path,
    *,
    k: int = 20,
    max_passes: int = 2,
    sample_per_frame: int = 20000,
    seed: int = 0,
    use_visible_for_prototypes: bool = True,
    invisible_weight: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    update_chunk: int = 1_000_000,
    feature_cfg: Optional[Dict] = None,
    dist_edges: Optional[List[float]] = None,
    dist_ratios: Optional[List[float]] = None,
    use_fp16: Optional[bool] = None,
    dl_workers: int = 8,
    dl_prefetch: int = 4,
    dl_batch_io: int = 32,
    dl_pin_memory: bool = True,
) -> torch.Tensor:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    if use_fp16 is None:
        use_fp16 = (dev.type == "cuda")

    if dev.type == "cuda":
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass

    if feature_cfg is None: feature_cfg = {}
    if dist_edges is None:  dist_edges = [0.0, 20.0, 40.0]

    centroids = None
    buf, seen = [], 0
    loader_seed = _make_loader(infer_dir, workers=dl_workers, prefetch=dl_prefetch, batch_io=dl_batch_io, pin_memory=dl_pin_memory)
    for batch in tqdm(loader_seed, desc="Seeding prototypes"):
        for item in batch:
            X0 = _build_features(item, feature_cfg)
            M = item["mask"]
            filt = M if (use_visible_for_prototypes and M.numel()) else None
            Xs = _stratified_subsample(X0, item["coord_raw"], sample_per_frame, dist_ratios, dist_edges, rng, filt)
            if Xs.numel() == 0: continue
            Xs = F.normalize(Xs, dim=1)
            buf.append(Xs); seen += Xs.shape[0]
            if seen >= k * 50:
                Xcat = torch.cat(buf, dim=0).to(dev, non_blocking=True)
                centroids = _kmeanspp_init(Xcat, k, dev=dev, seed=seed)
                break
        if centroids is not None: break
    if centroids is None:
        Xcat = (torch.cat(buf, dim=0) if buf else torch.randn(k, 64)).to(dev, non_blocking=True)
        Xcat = F.normalize(Xcat, dim=1)
        centroids = _kmeanspp_init(Xcat, k, dev=dev, seed=seed)

    loader = _make_loader(infer_dir, workers=dl_workers, prefetch=dl_prefetch, batch_io=dl_batch_io, pin_memory=dl_pin_memory)
    for _ in tqdm(range(max_passes), desc="Learning prototypes"):
        accum = torch.zeros_like(centroids, device=dev, dtype=torch.float32)
        counts = torch.zeros((k,), dtype=torch.float32, device=dev)
        for batch in loader:
            for item in batch:
                X0 = _build_features(item, feature_cfg)
                M = item["mask"]
                filt = M if (use_visible_for_prototypes and M.numel()) else None
                X = _stratified_subsample(X0, item["coord_raw"], sample_per_frame, dist_ratios, dist_edges, rng, filt)
                if X.numel() == 0: continue

                X = F.normalize(X, dim=1).to(dev, non_blocking=True)
                if use_fp16 and dev.type == "cuda":
                    X_mm = X.to(torch.float16); C_mm = centroids.to(torch.float16)
                else:
                    X_mm = X; C_mm = centroids

                for s in range(0, X.shape[0], update_chunk):
                    Xe = X[s:s+update_chunk]; Ze = X_mm[s:s+update_chunk]
                    idx = (Ze @ C_mm.T).argmax(dim=1)
                    accum.index_add_(0, idx, Xe)
                    counts += torch.bincount(idx, minlength=k).to(counts.dtype)

        m = counts > 0
        centroids[m] = accum[m] / counts[m].unsqueeze(1).clamp(min=1e-6)
        centroids = F.normalize(centroids, dim=1)

    return centroids.detach().cpu()

# ------------------------------------------------------------------------ #
#                  vMF (von Mises–Fisher) mixture learning                 #
# ------------------------------------------------------------------------ #

def _kappa_from_Rbar(Rbar: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    R = Rbar.clamp(min=eps, max=1 - 1e-6)
    num = R * (dim - R * R)
    den = (1.0 - R * R).clamp_min(eps)
    kappa = num / den
    return kappa.clamp_min(1e-3)

@torch.no_grad()
def learn_vmf_from_dataset(
    infer_dir: Path,
    *,
    k: int = 20,
    max_passes: int = 3,
    sample_per_frame: int = 20000,
    seed: int = 0,
    use_visible_for_prototypes: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    update_chunk: int = 1_000_000,
    feature_cfg: Optional[Dict] = None,
    dist_edges: Optional[List[float]] = None,
    dist_ratios: Optional[List[float]] = None,
    use_fp16: Optional[bool] = None,
    dl_workers: int = 8,
    dl_prefetch: int = 4,
    dl_batch_io: int = 32,
    dl_pin_memory: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    if use_fp16 is None:
        use_fp16 = (dev.type == "cuda")
    if dev.type == "cuda":
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
    if feature_cfg is None: feature_cfg = {}
    if dist_edges is None:  dist_edges = [0.0, 20.0, 40.0]

    buf, seen = [], 0
    loader_seed = _make_loader(infer_dir, workers=dl_workers, prefetch=dl_prefetch, batch_io=dl_batch_io, pin_memory=dl_pin_memory)
    mu = None
    for batch in tqdm(loader_seed, desc="Seeding vMF means"):
        for item in batch:
            X0 = _build_features(item, feature_cfg)
            M = item["mask"]
            filt = M if (use_visible_for_prototypes and M.numel()) else None
            Xs = _stratified_subsample(X0, item["coord_raw"], sample_per_frame, dist_ratios, dist_edges, rng, filt)
            if Xs.numel() == 0: continue
            Xs = F.normalize(Xs, dim=1)
            buf.append(Xs); seen += Xs.shape[0]
            if seen >= k * 50:
                Xcat = torch.cat(buf, dim=0).to(dev, non_blocking=True)
                mu = _kmeanspp_init(Xcat, k, dev=dev, seed=seed)
                break
        if mu is not None: break
    if mu is None:
        Xcat = (torch.cat(buf, dim=0) if buf else torch.randn(k, 64)).to(dev, non_blocking=True)
        Xcat = F.normalize(Xcat, dim=1)
        mu = _kmeanspp_init(Xcat, k, dev=dev, seed=seed)

    D = int(mu.shape[1])
    kappa = torch.full((k,), 10.0, device=dev, dtype=torch.float32)

    loader = _make_loader(infer_dir, workers=dl_workers, prefetch=dl_prefetch, batch_io=dl_batch_io, pin_memory=dl_pin_memory)
    for _ in tqdm(range(max_passes), desc="vMF EM"):
        V = torch.zeros_like(mu, device=dev, dtype=torch.float32)
        Nk = torch.zeros((k,), device=dev, dtype=torch.float32)

        for batch in loader:
            for item in batch:
                X0 = _build_features(item, feature_cfg)
                M = item["mask"]
                filt = M if (use_visible_for_prototypes and M.numel()) else None
                X = _stratified_subsample(X0, item["coord_raw"], sample_per_frame, dist_ratios, dist_edges, rng, filt)
                if X.numel() == 0: continue

                X = F.normalize(X, dim=1).to(dev, non_blocking=True)
                if use_fp16 and dev.type == "cuda":
                    X_mm = X.to(torch.float16); MU_mm = mu.to(torch.float16); kappa_mm = kappa.to(torch.float16)
                else:
                    X_mm = X; MU_mm = mu; kappa_mm = kappa

                for s in range(0, X.shape[0], update_chunk):
                    Xe = X[s:s+update_chunk]
                    Ze = X_mm[s:s+update_chunk]
                    sim = Ze @ MU_mm.T
                    logits = sim * kappa_mm.unsqueeze(0)
                    logits_f32 = logits.to(torch.float32)
                    logits_f32 -= logits_f32.max(dim=1, keepdim=True).values
                    R = torch.softmax(logits_f32, dim=1)
                    V += R.T @ Xe
                    Nk += R.sum(dim=0).to(Nk.dtype)

        mu = F.normalize(V, dim=1).to(torch.float32)
        Vnorm = V.norm(dim=1).clamp_min(1e-8)
        Rbar = (Vnorm / Nk.clamp_min(1e-6)).clamp(0.0, 0.999999)
        kappa = _kappa_from_Rbar(Rbar, D)

    return mu.detach().cpu(), kappa.detach().cpu()

# ------------------------------------------------------------------------ #
#                   Voxel neighborhood label smoothing                     #
# ------------------------------------------------------------------------ #
def smooth_labels_voxel(
    grid_coord: torch.Tensor,
    labels: np.ndarray,
    iters: int = 1,
    neighbor_range: int = 1,
    min_component: int = 50,
    *,
    ignore_label: Optional[int] = -1,
    coord_raw: Optional[torch.Tensor] = None,
    range_gate_m: Optional[float] = None,
) -> np.ndarray:
    """
    Python-mode smoothing; keep it because it’s robust, but it can be slow.
    """
    if grid_coord.numel() == 0 or labels.size == 0 or iters <= 0:
        return labels

    g = grid_coord.cpu().numpy().astype(np.int64)
    mins = g.min(axis=0)
    g = g - mins

    lbl = labels.copy()

    key = (g[:, 0] << 42) + (g[:, 1] << 21) + (g[:, 2])
    from collections import defaultdict as _dd
    vox2idx: Dict[int, List[int]] = _dd(list)
    for i, k in enumerate(key):
        vox2idx[int(k)].append(i)

    vox_range: Optional[Dict[int, float]] = None
    if (coord_raw is not None) and (range_gate_m is not None):
        r = torch.linalg.norm(coord_raw, dim=1).cpu().numpy()
        vox_range = {}
        for k, idxs in vox2idx.items():
            if idxs:
                vox_range[k] = float(r[idxs].mean())

    offs = np.array([(dx, dy, dz)
                     for dx in range(-neighbor_range, neighbor_range + 1)
                     for dy in range(-neighbor_range, neighbor_range + 1)
                     for dz in range(-neighbor_range, neighbor_range + 1)],
                    dtype=np.int64)

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
                if not js:
                    continue
                if vox_range is not None:
                    r0 = vox_range.get(int(k), None)
                    r1 = vox_range.get(int(nk), None)
                    if (r0 is None) or (r1 is None) or (abs(r1 - r0) > float(range_gate_m)):
                        continue
                if ignore_label is None:
                    votes.extend(lbl[js])
                else:
                    votes.extend([v for v in lbl[js] if v != ignore_label])

            if votes:
                vals, counts = np.unique(np.array(votes, dtype=lbl.dtype), return_counts=True)
                new_lbl[idxs] = vals[counts.argmax()]
        lbl = new_lbl

    if min_component > 0:
        from collections import defaultdict as _dd2
        counts = _dd2(int)
        lv = list(zip(lbl.tolist(), key.tolist()))
        for pair in lv:
            if (ignore_label is not None) and (pair[0] == ignore_label):
                continue
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
                        if not js:
                            continue
                        if ignore_label is None:
                            votes.extend(lbl[js])
                        else:
                            votes.extend([v for v in lbl[js] if v != ignore_label])
                    if votes:
                        vals, c2 = np.unique(np.array(votes, dtype=lbl.dtype), return_counts=True)
                        lbl[i] = vals[c2.argmax()]

    return lbl

# ------------------------------------------------------------------------ #
#                         Segmentation (nearest proto)                     #
# ------------------------------------------------------------------------ #

def _npz_bytes(labels: np.ndarray, compress: bool = False) -> bytes:
    """Pack labels into an .npz (optionally compressed) and return bytes."""
    buff = io.BytesIO()
    if compress:
        np.savez_compressed(buff, labels=labels.astype(np.int32))
    else:
        np.savez(buff, labels=labels.astype(np.int32))
    return buff.getvalue()

@torch.no_grad()
def segment_dataset(
    infer_dir: Path,
    centroids: torch.Tensor,
    *,
    smoothing_iters: int = 1,
    neighbor_range: int = 1,
    min_component: int = 50,
    per_frame_hook: Optional[Callable[[str, np.ndarray, np.ndarray], None]] = None,
    collect_metrics: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    assign_chunk: int = 3_000_000,
    feature_cfg: Optional[Dict] = None,
    save_labels_dir: Optional[Path] = None,
    use_fp16: Optional[bool] = None,
    # ---- ZIP writing (optional) ----
    zip_labels_path: Optional[Path] = None,
    zip_mode: str = "w",
    zip_compress: int = zipfile.ZIP_STORED,            # <<<< default: STORE (no re-compress)
    npz_compress: bool = False,                         # <<<< inner .npz uncompressed by default
    # ---- Quality knobs (k-means cosine path) ----
    tau_reject: Optional[float] = None,
    noise_label: int = -1,
    range_gate_m: Optional[float] = None,
    # ---- DataLoader knobs ----
    dl_workers: int = 8,
    dl_prefetch: int = 4,
    dl_batch_io: int = 32,
    dl_pin_memory: bool = True,
    # ---- New: vMF & distance-aware thresholds ----
    mode: str = "kmeans",
    vmf_kappa: Optional[torch.Tensor] = None,
    posterior_tau: Optional[float] = None,
    tau_edges: Optional[List[float]] = None,
    tau_map: Optional[List[float]] = None,
    # ---- Speed guard: skip smoothing beyond this point count
    max_points_smooth: Optional[int] = 300_000,         # <<<< skip smoothing on very large frames
) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Dict[str, List[torch.Tensor]]]]:
    """
    Segment dataset and optionally persist labels (dir and/or a single ZIP).
    Fully DataLoader-based and **no resume**.
    """
    if feature_cfg is None:
        feature_cfg = {}

    dev = torch.device(device)
    if use_fp16 is None:
        use_fp16 = (dev.type == "cuda")
    if dev.type == "cuda":
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass

    C = F.normalize(centroids.to(torch.float32), dim=1).to(dev, non_blocking=True)
    C_mm = C.to(torch.float16) if (use_fp16 and dev.type == "cuda") else C
    K = C.shape[0]

    kappa_dev: Optional[torch.Tensor] = None
    if mode.lower() == "vmf":
        if vmf_kappa is None:
            raise ValueError("mode='vmf' requires vmf_kappa tensor of shape (K).")
        kappa_dev = vmf_kappa.to(dev, dtype=torch.float32, non_blocking=True)
        if kappa_dev.numel() != K:
            raise ValueError(f"vmf_kappa length {kappa_dev.numel()} != number of centroids {K}")
        kappa_dev = kappa_dev.clamp_min(1e-3)
        if posterior_tau is None and (tau_edges is None or tau_map is None):
            posterior_tau = 0.05

    tau_bins_edges = None
    tau_bins_vals = None
    if (tau_edges is not None) and (tau_map is not None):
        tau_bins_edges = torch.as_tensor(list(tau_edges), device=dev, dtype=torch.float32)
        tau_bins_vals  = torch.as_tensor(list(tau_map),  device=dev, dtype=torch.float32)

    results: Dict[str, np.ndarray] = {}
    accum = {"feats": [], "labels": [], "speeds": [], "mask": []} if collect_metrics else None

    if save_labels_dir is not None:
        save_labels_dir = Path(save_labels_dir)
        save_labels_dir.mkdir(parents=True, exist_ok=True)

    zf = None
    written_names = None
    if zip_labels_path is not None:
        zip_labels_path = Path(zip_labels_path)
        zip_labels_path.parent.mkdir(parents=True, exist_ok=True)
        zf = zipfile.ZipFile(zip_labels_path, mode=zip_mode, compression=zip_compress, allowZip64=True)
        written_names = set()

    loader = _make_loader(
        infer_dir,
        workers=dl_workers,
        prefetch=dl_prefetch,
        batch_io=dl_batch_io,
        pin_memory=dl_pin_memory,
    )

    try:
        for batch in tqdm(loader, desc="Segmenting + (opt) metrics/export"):
            for item in batch:
                stem = item["file_stem"]

                Z0 = _build_features(item, feature_cfg)
                if Z0.numel() == 0:
                    results[stem] = np.empty((0,), dtype=np.int64)
                    continue

                Z_norm = F.normalize(Z0, dim=1)
                N = Z_norm.shape[0]
                idx_out = np.empty((N,), dtype=np.int64)

                need_tau_bins = (tau_bins_edges is not None) and (tau_bins_vals is not None)
                r_all = None
                if need_tau_bins:
                    r_all = torch.linalg.norm(item["coord_raw"], dim=1).to(dev, non_blocking=True)

                for s in range(0, N, assign_chunk):
                    Ze_f32 = Z_norm[s:s+assign_chunk].to(dev, non_blocking=True)
                    Ze_mm  = Ze_f32.to(torch.float16) if (use_fp16 and dev.type == "cuda") else Ze_f32
                    sim    = Ze_mm @ C_mm.T

                    if mode.lower() == "vmf":
                        logits = sim.to(torch.float32) * kappa_dev.unsqueeze(0)
                        logits = logits - logits.max(dim=1, keepdim=True).values
                        post = torch.softmax(logits, dim=1)
                        max_post, hard = post.max(dim=1)
                        idx = hard.detach().cpu().numpy().astype(np.int64)

                        if (posterior_tau is not None) or need_tau_bins:
                            if need_tau_bins:
                                r_chunk = r_all[s:s+assign_chunk]
                                bin_idx = torch.bucketize(r_chunk, tau_bins_edges, right=False)
                                bin_idx = bin_idx.clamp(max=tau_bins_vals.numel()-1)
                                tau_eff = tau_bins_vals[bin_idx]
                                rej = (max_post < tau_eff).detach().cpu().numpy()
                            else:
                                rej = (max_post < float(posterior_tau)).detach().cpu().numpy()
                            if np.any(rej):
                                idx[rej] = int(noise_label)

                    else:
                        max_sim, hard = sim.max(dim=1)
                        idx = hard.detach().cpu().numpy().astype(np.int64)
                        if (tau_reject is not None) or need_tau_bins:
                            if need_tau_bins:
                                r_chunk = r_all[s:s+assign_chunk]
                                bin_idx = torch.bucketize(r_chunk, tau_bins_edges, right=False)
                                bin_idx = bin_idx.clamp(max=tau_bins_vals.numel()-1)
                                tau_eff = tau_bins_vals[bin_idx]
                                rej = (max_sim < tau_eff).detach().cpu().numpy()
                            else:
                                rej = (max_sim < float(tau_reject)).detach().cpu().numpy()
                            if np.any(rej):
                                idx[rej] = int(noise_label)

                    idx_out[s:s+idx.shape[0]] = idx

                # ---- Smoothing (guarded) ----
                do_smooth = (smoothing_iters > 0) and (max_points_smooth is None or N <= int(max_points_smooth))
                if do_smooth:
                    idx_out = smooth_labels_voxel(
                        grid_coord=item["grid_coord"],
                        labels=idx_out,
                        iters=smoothing_iters,
                        neighbor_range=neighbor_range,
                        min_component=min_component,
                        ignore_label=noise_label,
                        coord_raw=item["coord_raw"],
                        range_gate_m=range_gate_m,
                    )

                # ---- Persist outputs ----
                if save_labels_dir is not None:
                    np.savez_compressed(save_labels_dir / f"{stem}_clusters.npz", labels=idx_out.astype(np.int32))

                if zf is not None:
                    name = f"{stem}_clusters.npz"
                    if (written_names is not None) and (name in written_names):
                        k = 2
                        while f"{stem}__dup{k}_clusters.npz" in written_names:
                            k += 1
                        name = f"{stem}__dup{k}_clusters.npz"
                    zf.writestr(name, _npz_bytes(idx_out, compress=npz_compress))
                    if written_names is not None:
                        written_names.add(name)

                results[stem] = idx_out

                if per_frame_hook is not None:
                    per_frame_hook(stem, item["coord_raw"].cpu().numpy().astype(np.float32), idx_out)

                if collect_metrics and accum is not None:
                    accum["feats"].append(item["feat64"])
                    accum["labels"].append(torch.as_tensor(idx_out, dtype=torch.int64))
                    accum["mask"].append(item["mask"])
                    accum["speeds"].append(item.get("speed", torch.empty((0,), dtype=torch.float32)))
    finally:
        if zf is not None:
            zf.close()

    if collect_metrics and accum is not None:
        return results, accum
    return results

# ------------------------------------------------------------------------ #
#                              Metrics wrapper                             #
# ------------------------------------------------------------------------ #

""" def evaluate_accumulated_metrics(
    accum: Dict[str, List[torch.Tensor]],
    out_csv: Path,
    *,
    sample_n: int = 50_000,
    seed: int = 42,
    q_bins: int = 4,
    tau_list: Optional[List[float]] = None,
    tau_policy: str = "quantile",
) -> Dict[str, Optional[float]]:
    feats_all = accum["feats"]
    labels_all = accum["labels"]
    speeds_all = accum["speeds"]
    vis_all = accum["mask"]

    if not feats_all:
        raise RuntimeError("No features found to evaluate metrics.")

    X = torch.cat(feats_all, dim=0).numpy()
    Y = torch.cat(labels_all, dim=0).numpy()
    M = torch.cat(vis_all, dim=0).numpy()
    V = torch.cat(speeds_all, dim=0).numpy() if all(s.numel() > 0 for s in speeds_all) else None

    cfg = MetricsConfig(sample_n=sample_n, seed=seed, q_bins=q_bins, tau_list=tau_list, tau_policy=tau_policy)
    results = compute_all_metrics(X, Y, speeds=V, visible_mask=M, token_labels_proj=None, cfg=cfg) if _HAS_METRICS else {}

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as w:
        w.write("metric,value\n")
        for k, v in results.items():
            w.write(f"{k},{'' if v is None else v}\n")
    print(f"[metrics] wrote {out_csv}")
    return results """

# --- REPLACE the whole evaluate_accumulated_metrics with this version ---
def evaluate_accumulated_metrics(
    accum,
    out_csv,
    sample_n=200_000,
    distance="cosine",
    # --- NEW: motion & temporal knobs ---
    speed_tau_policy="value",           # "value" for fixed m/s thresholds
    speed_tau_list=(0.3, 0.5, 1.0),     # thresholds for F1 and bins (m/s); tune per your dataset
    # --- NEW: optional 2D alignment (pass an extra list at call-site, else leave None) ---
    labels2d_all=None
):
    """
    Streaming / two-pass metrics with motion & temporal extensions.
    - Exact CH/DBI on full data (z-scored, streaming).
    - Silhouette on a reservoir sample (keeps memory bounded).
    - Motion metrics: NMI(cluster, speed-bins), F1_dyn@taus, weighted within-cluster |v| variance.
    - Temporal consistency: size-weighted cosine similarity of matched cluster centroids across consecutive frames.
    - 2D alignment NMI: optional, if labels2d_all is provided (same shape lists as labels_all).
    """
    import math, gc, csv, os
    import numpy as np

    try:
        from sklearn.metrics import silhouette_score
    except Exception:
        silhouette_score = None

    feats_all = accum["feats"]
    labels_all = accum["labels"]
    speeds_all = accum["speeds"]
    vis_all = accum["mask"]

    # ---------- helpers ----------
    def _iter_chunks(include_2d=False):
        """Yield per-frame (X, y, m, v, y2d_optional). Filter on visibility & assigned labels later."""
        if include_2d and labels2d_all is not None:
            for Xi, yi, mi, vi, zi in zip(feats_all, labels_all, vis_all, speeds_all, labels2d_all):
                yield Xi.detach().cpu().float(), yi.detach().cpu().long(), mi.detach().cpu().bool(), \
                      (vi.detach().cpu().float() if vi is not None and vi.numel() else None), \
                      (zi.detach().cpu().long() if zi is not None and zi.numel() else None)
        else:
            for Xi, yi, mi, vi in zip(feats_all, labels_all, vis_all, speeds_all):
                yield Xi.detach().cpu().float(), yi.detach().cpu().long(), mi.detach().cpu().bool(), \
                      (vi.detach().cpu().float() if vi is not None and vi.numel() else None), None

    def _write_csv_row(path, row: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)


    # ---------- discover K, N ----------
    K = 0
    N_total = 0
    for Xi, yi, mi, vi, zi in _iter_chunks(include_2d=True):
        keep = mi & (yi >= 0)
        if keep.any():
            yk = yi[keep]
            K = max(K, int(yk.max().item()) + 1)
            N_total += int(yk.numel())
        del Xi, yi, mi, vi, zi, keep
        gc.collect()
    if K == 0 or N_total == 0:
        _write_csv_row(out_csv, {"N": 0, "K": 0, "CH": float("nan"), "DBI": float("nan"), "SIL": float("nan")})
        return

    # ---------- pass 1: global z-stats + reservoir + motion tallies + per-frame centroids ----------
    # Global z-stats
    sum_global = None
    sumsq_global = None
    n_seen = 0

    # Reservoir for silhouette
    cap = int(sample_n) if sample_n else 0
    samp_X = samp_y = None
    rng = np.random.RandomState(0)

    # Motion: bin edges & holders
    taus = np.array(speed_tau_list, dtype=np.float32)  # fixed thresholds (m/s)
    B = len(taus) + 1                                  # bins: <=t1, (t1,t2], ... , >t_last
    CM = np.zeros((K, B), dtype=np.int64)              # confusion: cluster x speed_bin
    # F1 per threshold: per-cluster pos/neg counts
    pos_counts = np.zeros((K, len(taus)), dtype=np.int64)
    neg_counts = np.zeros((K, len(taus)), dtype=np.int64)
    # Within-cluster |v| variance (Welford): per-cluster mean & M2
    vel_mean = np.zeros(K, dtype=np.float64)
    vel_M2   = np.zeros(K, dtype=np.float64)
    vel_n    = np.zeros(K, dtype=np.int64)

    # Temporal: per-frame centroids & sizes (for cosine matching)
    frame_centroids = []   # list of (K,64) np.float32
    frame_sizes     = []   # list of (K,) np.int64

    for Xi, yi, mi, vi, zi in _iter_chunks(include_2d=True):
        keep = mi & (yi >= 0)
        if not keep.any():
            # still append empty centroids for temporal indexing
            frame_centroids.append(np.zeros((K, 64), dtype=np.float32))
            frame_sizes.append(np.zeros(K, dtype=np.int64))
            del Xi, yi, mi, vi, zi, keep
            gc.collect()
            continue

        X = Xi[keep].numpy()
        y = yi[keep].numpy()
        n = X.shape[0]

        # init global stats
        if sum_global is None:
            D = X.shape[1]
            sum_global  = np.zeros(D, dtype=np.float64)
            sumsq_global = np.zeros(D, dtype=np.float64)

        sum_global  += X.sum(0, dtype=np.float64)
        sumsq_global += (X * X).sum(0, dtype=np.float64)

        # reservoir sample for silhouette
        if cap > 0:
            if samp_X is None:
                take = min(cap, n)
                idx = np.arange(take)
                samp_X = X[idx].copy()
                samp_y = y[idx].copy()
            else:
                for i in range(n):
                    j = rng.randint(0, n_seen + i + 1)
                    if j < cap:
                        samp_X[j] = X[i]
                        samp_y[j] = y[i]

        # motion tallies
        if vi is not None and vi.numel():
            sp = vi[keep].numpy()  # |v| in m/s
            # bins: np.digitize returns 1..len(taus); we want 0..B-1
            bin_ids = np.digitize(sp, taus, right=True)
            # Confusion counts per cluster x bin
            for k in range(K):
                mk = (y == k)
                if np.any(mk):
                    hist = np.bincount(bin_ids[mk], minlength=B)
                    CM[k, :] += hist.astype(np.int64)

            # F1 per threshold: pos/neg per cluster
            for t_idx, t in enumerate(taus):
                pos = (sp > t)
                for k in range(K):
                    mk = (y == k)
                    if np.any(mk):
                        pos_counts[k, t_idx] += int(np.count_nonzero(pos[mk]))
                        neg_counts[k, t_idx] += int(np.count_nonzero(~pos[mk]))

            # Within-cluster |v| variance (Welford)
            for k in range(K):
                mk = (y == k)
                if np.any(mk):
                    vk = sp[mk].astype(np.float64)
                    nk_old = vel_n[k]
                    vel_n[k] += vk.size
                    delta = vk.mean() - vel_mean[k] if vk.size > 0 else 0.0
                    vel_mean[k] += (vk.size * delta) / max(vel_n[k], 1)
                    # M2 update per sample for accuracy
                    for vv in vk:
                        d  = vv - vel_mean[k]
                        vel_M2[k] += d * d

        # Per-frame centroids in raw feature space (for temporal)
        # We use L2-normalized features for cosine later
        C_t = np.zeros((K, X.shape[1]), dtype=np.float32)
        S_t = np.zeros(K, dtype=np.int64)
        for k in range(K):
            mk = (y == k)
            if np.any(mk):
                Xk = X[mk]
                C_t[k] = Xk.mean(0).astype(np.float32)
                S_t[k] = Xk.shape[0]
        frame_centroids.append(C_t)
        frame_sizes.append(S_t)

        n_seen += n
        del Xi, yi, mi, vi, zi, keep, X, y
        gc.collect()

    # global z-scoring params
    mu = sum_global / float(n_seen)
    var = np.maximum(sumsq_global / float(n_seen) - mu * mu, 1e-8)
    std = np.sqrt(var, dtype=np.float64)

    # ---------- pass 2: CH/DBI in z-space (exact, full dataset) ----------
    mu_k_sum = np.zeros((K, mu.shape[0]), dtype=np.float64)
    sumsq_k  = np.zeros(K, dtype=np.float64)
    n_k      = np.zeros(K, dtype=np.int64)
    sum_global_z = np.zeros_like(mu, dtype=np.float64)

    for Xi, yi, mi, vi, zi in _iter_chunks(include_2d=False):
        keep = mi & (yi >= 0)
        if not keep.any():
            del Xi, yi, mi, vi
            gc.collect();  continue
        Xz = (Xi[keep].numpy() - mu) / std
        y  = yi[keep].numpy()
        for k in range(K):
            mk = (y == k)
            if np.any(mk):
                Xk = Xz[mk]
                n_k[k]     += Xk.shape[0]
                mu_k_sum[k] += Xk.sum(0, dtype=np.float64)
                sumsq_k[k]  += float((Xk * Xk).sum())
        sum_global_z += Xz.sum(0, dtype=np.float64)
        del Xi, yi, mi, vi, Xz, y
        gc.collect()

    mu_k = (mu_k_sum.T / np.maximum(n_k, 1)).T
    mu_k_norm2 = (mu_k * mu_k).sum(1)
    SW = float(np.sum(sumsq_k - n_k * mu_k_norm2))
    mu_g = sum_global_z / float(n_seen)
    SB = float(np.sum(n_k * np.sum((mu_k - mu_g) ** 2, axis=1)))
    CH = (SB / max(K - 1, 1)) / (SW / max(n_seen - K, 1))

    # DBI (Euclidean on z-scored space)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_k = np.sqrt(np.maximum((sumsq_k - n_k * mu_k_norm2) / np.maximum(n_k, 1), 0.0))
    cij = mu_k @ mu_k.T
    Md2 = np.maximum(mu_k_norm2[:, None] + mu_k_norm2[None, :] - 2.0 * cij, 1e-12)
    Md  = np.sqrt(Md2)
    R = np.full(K, 0.0)
    for k in range(K):
        if n_k[k] == 0:
            R[k] = np.nan
            continue
        ratios = (s_k[k] + s_k) / np.maximum(Md[k], 1e-8)
        ratios[k] = -np.inf
        R[k] = float(np.nanmax(ratios))
    DBI = float(np.nanmean(R))

    # ---------- Silhouette (sampled) ----------
    SIL = float("nan")
    if (samp_X is not None) and (silhouette_score is not None):
        if distance == "cosine":
            SIL = float(silhouette_score(samp_X, samp_y, metric="cosine"))
        else:
            samp_Xz = (samp_X - mu) / std
            SIL = float(silhouette_score(samp_Xz, samp_y, metric="euclidean"))

    # ---------- Motion metrics ----------
    # 4) NMI(cluster, speed-bins)
    def _entropy(p):
        p = p[p > 0]
        return -float(np.sum(p * np.log(p)))
    total = CM.sum()
    nmi = float("nan")
    if total > 0:
        Pxy = CM / total
        Px  = Pxy.sum(1, keepdims=True)
        Py  = Pxy.sum(0, keepdims=True)
        MI  = float(np.nansum(Pxy * (np.log(Pxy + 1e-12) - np.log(Px + 1e-12) - np.log(Py + 1e-12))))
        Hx  = _entropy((CM.sum(1) / total))
        Hy  = _entropy((CM.sum(0) / total))
        nmi = MI / max((Hx + Hy) / 2.0, 1e-12)

    # 5) F1 for dynamic vs static at thresholds
    f1s = {}
    for t_idx, t in enumerate(taus):
        # cluster majority mapping to predicted label at this threshold
        # if majority is "moving" (pos >= neg) then cluster predicts moving for all its points
        pos_k = pos_counts[:, t_idx]
        neg_k = neg_counts[:, t_idx]
        # global counts by majority decision
        TP = int((pos_k[pos_k >= neg_k]).sum())
        FP = int((neg_k[pos_k >= neg_k]).sum())
        TN = int((neg_k[pos_k <  neg_k]).sum())
        FN = int((pos_k[pos_k <  neg_k]).sum())
        precision = TP / max(TP + FP, 1)
        recall    = TP / max(TP + FN, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1s[f"F1_dyn@{float(t):.2f}mps"] = float(f1)

    # 6) Weighted within-cluster speed variance
    vel_var_k = np.zeros(K, dtype=np.float64)
    for k in range(K):
        if vel_n[k] > 1:
            vel_var_k[k] = vel_M2[k] / (vel_n[k] - 1)
        else:
            vel_var_k[k] = np.nan
    # size-weighted mean of per-cluster variance (ignore NaNs)
    weights = n_k.astype(np.float64)
    mask = ~np.isnan(vel_var_k)
    velvar_wmean = float(np.sum(weights[mask] * vel_var_k[mask]) / max(np.sum(weights[mask]), 1.0))

    # ---------- Temporal consistency (feature centroid matching) ----------
    # cosine similarity between matched centroids in consecutive frames, size-weighted
    def _l2norm(a, axis=1, eps=1e-12):
        n = np.sqrt(np.maximum((a * a).sum(axis=axis, keepdims=True), eps))
        return a / n
    temp_scores = []
    for t in range(len(frame_centroids) - 1):
        C0 = _l2norm(frame_centroids[t].astype(np.float64), axis=1)
        C1 = _l2norm(frame_centroids[t + 1].astype(np.float64), axis=1)
        S0 = frame_sizes[t].astype(np.float64)
        S1 = frame_sizes[t + 1].astype(np.float64)
        # cosine sim matrix
        S = C0 @ C1.T  # (K,K)
        # greedy matching (Hungarian would be nicer but this is light and stable)
        used0 = np.zeros(K, dtype=bool)
        used1 = np.zeros(K, dtype=bool)
        pairs = []
        # pick top-K pairs greedily
        for _ in range(K):
            idx = np.unravel_index(np.argmax(S, axis=None), S.shape)
            i, j = int(idx[0]), int(idx[1])
            if used0[i] or used1[j]:
                S[i, j] = -np.inf
                continue
            pairs.append((i, j, S[i, j]))
            used0[i] = True
            used1[j] = True
            S[i, :] = -np.inf
            S[:, j] = -np.inf
        # size-weighted mean cosine over pairs (min size to be conservative)
        num = 0.0
        den = 0.0
        for i, j, sim in pairs:
            w = min(S0[i], S1[j])
            num += w * max(float(sim), -1.0)
            den += w
        if den > 0:
            temp_scores.append(num / den)

    temporal_consistency = float(np.mean(temp_scores)) if len(temp_scores) else float("nan")

    # ---------- Optional: 2D alignment NMI ----------
    nmi_2d = float("nan")
    if labels2d_all is not None:
        # stream a cross-tab between 3D cluster ids and 2D token/segment ids
        # We don't know the max 2D id; build a dict of counters
        from collections import defaultdict
        counts_2d = defaultdict(lambda: defaultdict(int))  # k -> { z2d -> count }
        n2d_total = 0
        for Xi, yi, mi, vi, zi in _iter_chunks(include_2d=True):
            keep = mi & (yi >= 0) & (zi is not None)
            if not keep.any():
                del Xi, yi, mi, vi, zi, keep; gc.collect();  continue
            yk = yi[keep].numpy()
            zk = zi[keep].numpy()
            for k in range(K):
                mk = (yk == k)
                if np.any(mk):
                    unique, cnts = np.unique(zk[mk], return_counts=True)
                    for u, c in zip(unique.tolist(), cnts.tolist()):
                        counts_2d[k][int(u)] += int(c)
                    n2d_total += int(cnts.sum())
            del Xi, yi, mi, vi, zi, keep
            gc.collect()
        if n2d_total > 0:
            # build dense matrix
            # map 2D ids to 0..Z-1
            u2d = sorted({z for d in counts_2d.values() for z in d})
            zmap = {z:i for i,z in enumerate(u2d)}
            Z = len(u2d)
            M = np.zeros((K, Z), dtype=np.int64)
            for k, d in counts_2d.items():
                for z, c in d.items():
                    M[k, zmap[z]] = c
            tot = M.sum()
            Pxy = M / tot
            Px = Pxy.sum(1, keepdims=True)
            Py = Pxy.sum(0, keepdims=True)
            MI = float(np.nansum(Pxy * (np.log(Pxy + 1e-12) - np.log(Px + 1e-12) - np.log(Py + 1e-12))))
            def _H(p): 
                p = p[p > 0];  return -float(np.sum(p * np.log(p)))
            Hx = _H(M.sum(1) / tot);  Hy = _H(M.sum(0) / tot)
            nmi_2d = MI / max((Hx + Hy) / 2.0, 1e-12)

    # ---------- write CSV ----------
    row = {
        "N": int(n_seen), "K": int(K),
        "CH": float(CH), "DBI": float(DBI), "SIL": float(SIL),
        "NMI_speedbins": float(nmi),
        "velvar_wmean": float(velvar_wmean),
        "temporal_consistency": temporal_consistency,
    }
    row.update(f1s)                # add F1_dyn@... columns
    if not math.isnan(nmi_2d):
        row["NMI_2Dalignment"] = float(nmi_2d)

    _write_csv_row(out_csv, row)
# --- END replacement ---

