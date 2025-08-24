from pathlib import Path
from typing import Iterable, Dict, List, Optional, Callable, Tuple, Union
from collections import defaultdict
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scripts.metrics import compute_all_metrics, MetricsConfig


# ------------------------------- I/O utils ------------------------------- #

def _iter_inference_dumps(infer_dir: Path) -> Iterable[Path]:
    infer_dir = Path(infer_dir)
    files = sorted(infer_dir.glob("*_inference.pth"))
    if not files:
        raise FileNotFoundError(f"No *_inference.pth files found in {infer_dir}")
    for f in files:
        yield f


def _load_dump(path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if "image_stem" not in payload:
        payload["image_stem"] = path.stem.replace("_inference", "")
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


def extract_features(
    infer_dir: Path,
    *,
    return_iter: bool = False,
):
    """Stream features from per-sample *_inference.pth files."""
    def _stream():
        for f in _iter_inference_dumps(infer_dir):
            p = _load_dump(f)
            yield {
                "file_stem":  p["image_stem"],
                "feat64":     p["ptv3_feat"],
                "coord_norm": p["coord_norm"],
                "coord_raw":  p["coord_raw"],
                "grid_coord": p["grid_coord"],
                "mask":       p["mask"],
                "grid_size":  float(p["grid_size"].item()) if torch.is_tensor(p["grid_size"]) else float(p["grid_size"]),
                "speed":      p.get("speed", torch.empty((0,), dtype=torch.float32)),
            }
    return _stream() if return_iter else list(_stream())


# ----------------------- Run-config helpers ------------------------------ #

def save_run_config(path: Path, cfg: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

def load_run_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


# ----------------------- Feature construction (aug) ---------------------- #

def _build_features(
    item: Dict[str, torch.Tensor],
    feature_cfg: Optional[Dict] = None,
) -> torch.Tensor:
    """
    Build clustering features.
    - Always includes 64-D backbone features.
    - Optionally concatenates: range (meters), height z (meters), |v| speed (m/s).
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
            s = (s / float(feature_cfg.get("speed_scale", 30.0))).unsqueeze(1).to(torch.float32)
            comps.append(s)

    if len(comps) == 1:  # no aug
        return X
    return torch.cat(comps, dim=1)


# ------------------ Distance helpers (stratified sampling) --------------- #

def _distance_bins(coord_raw: torch.Tensor, edges: List[float]) -> List[torch.Tensor]:
    r = torch.linalg.norm(coord_raw, dim=1)
    masks = []
    for i, _ in enumerate(edges):
        if i == 0:
            lo, hi = 0.0, edges[1] if len(edges) > 1 else float("inf")
        elif i < len(edges) - 1:
            lo, hi = edges[i], edges[i+1]
        else:
            lo, hi = edges[-1], float("inf")
        masks.append((r >= lo) & (r < hi))
    return masks


def _stratified_subsample(
    X: torch.Tensor,
    coord_raw: torch.Tensor,
    total_n: int,
    ratios: List[float],
    edges: List[float],
    rng: np.random.Generator,
    mask_filter: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    N = X.shape[0]
    if N == 0 or total_n <= 0:
        return X

    valid = mask_filter if (mask_filter is not None and mask_filter.numel() == N) \
        else torch.ones((N,), dtype=torch.bool, device=X.device)

    bins = [b & valid for b in _distance_bins(coord_raw, edges)]

    ratios = np.array(ratios, dtype=np.float64)
    ratios = (ratios / (ratios.sum() if ratios.sum() > 0 else 1.0))

    idx_keep = []
    for bi, b in enumerate(bins):
        n_b = int(round(total_n * float(ratios[bi])))
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


# ------------------ K-means (cosine) prototype learning ------------------ #

@torch.no_grad()
def learn_prototypes_from_dataset(
    infer_dir: Path,
    *,
    k: int = 20,
    max_passes: int = 2,
    sample_per_frame: int = 20000,
    seed: int = 0,
    use_visible_for_prototypes: bool = True,
    invisible_weight: float = 1.0,  # only if use_visible_for_prototypes=False
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    update_chunk: int = 1_000_000,
    # optional:
    feature_cfg: Optional[Dict] = None,
    dist_edges: Optional[List[float]] = None,
    dist_ratios: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Learn K unit-norm prototypes with optional feature augmentation and distance-stratified subsampling.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    if feature_cfg is None:
        feature_cfg = {}
    if dist_edges is None:
        dist_edges = [0.0, 20.0, 40.0]
    if dist_ratios is None:
        dist_ratios = [0.5, 0.35, 0.15]

    # ---- Seed ----
    centroids = None
    buf, seen = [], 0
    for item in tqdm(extract_features(infer_dir, return_iter=True), desc="Seeding prototypes"):
        X0 = _build_features(item, feature_cfg)
        M = item["mask"]
        filt = M if (use_visible_for_prototypes and M.numel()) else None
        Xs = _stratified_subsample(X0, item["coord_raw"], sample_per_frame, dist_ratios, dist_edges, rng, filt)
        if Xs.numel() == 0:
            continue
        Xs = F.normalize(Xs, dim=1)
        buf.append(Xs); seen += Xs.shape[0]
        if seen >= k * 50:
            Xcat = torch.cat(buf, dim=0)
            centroids = F.normalize(Xcat[torch.randperm(Xcat.shape[0])[:k]].clone(), dim=1).to(dev)
            break
    if centroids is None:
        Xcat = torch.cat(buf, dim=0) if buf else torch.randn(k, 64)
        centroids = F.normalize(Xcat[torch.randperm(Xcat.shape[0])[:k]].clone(), dim=1).to(dev)

    # ---- Passes ----
    for _ in tqdm(range(max_passes), desc="Learning prototypes"):
        accum = torch.zeros_like(centroids, device=dev)
        counts = torch.zeros((k,), dtype=torch.float32, device=dev)
        for item in extract_features(infer_dir, return_iter=True):
            X0 = _build_features(item, feature_cfg)
            M = item["mask"]
            filt = M if (use_visible_for_prototypes and M.numel()) else None
            X = _stratified_subsample(X0, item["coord_raw"], sample_per_frame, dist_ratios, dist_edges, rng, filt)
            if X.numel() == 0:
                continue
            X = F.normalize(X, dim=1).to(dev, non_blocking=True)
            for s in range(0, X.shape[0], update_chunk):
                Xe = X[s:s+update_chunk]
                idx = (Xe @ centroids.T).argmax(dim=1)
                accum.index_add_(0, idx, Xe)
                counts += torch.bincount(idx, minlength=k).to(counts.dtype)

        m = counts > 0
        centroids[m] = accum[m] / counts[m].unsqueeze(1).clamp(min=1e-6)
        centroids = F.normalize(centroids, dim=1)

    return centroids.detach().cpu()


# ----------------------- Helper: soft assignment ------------------------- #

def _soft_assign(sim: torch.Tensor, tau: float = 15.0):
    logits = sim * float(tau)
    probs = torch.softmax(logits, dim=1)
    conf, _ = probs.max(dim=1)
    return probs, conf


# ------------- Fast voxel-neighborhood label smoothing (CPU) ------------- #

def smooth_labels_voxel(
    grid_coord: torch.Tensor,
    labels: np.ndarray,
    iters: int = 1,
    neighbor_range: int = 1,
    min_component: int = 50,
    visible_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    vis_vote_mode: str = "one_way",  # "off" | "strict" | "one_way"
    region_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> np.ndarray:
    """
    Visibility-aware voxel smoothing.
    - "one_way": visible points only take votes from visible; invisible can take votes from all.
    region_mask: if set, only update indices where region_mask==True (neighbors outside don’t vote).
    """
    if grid_coord.numel() == 0 or labels.size == 0:
        return labels

    g = grid_coord.cpu().numpy().astype(np.int64)
    lbl = labels.copy()
    vis = None if visible_mask is None else np.asarray(visible_mask, dtype=bool)
    reg = None if region_mask is None else np.asarray(region_mask, dtype=bool)

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
            if not idxs: continue
            idxs_np = np.array(idxs, dtype=np.int64)
            if reg is not None:
                idxs_np = idxs_np[reg[idxs_np]]
                if idxs_np.size == 0:
                    continue

            z =  k        & ((1 << 21) - 1)
            y = (k >> 21) & ((1 << 21) - 1)
            x = (k >> 42) & ((1 << 21) - 1)
            neigh_keys = ((x + offs[:, 0]) << 42) + ((y + offs[:, 1]) << 21) + (z + offs[:, 2])

            def collect_votes(filter_fn=None):
                votes = []
                for nk in neigh_keys:
                    js = vox2idx.get(int(nk))
                    if not js:
                        continue
                    js = np.array(js, dtype=np.int64)
                    if reg is not None:
                        js = js[reg[js]]
                    if filter_fn is not None:
                        js = filter_fn(js)
                    if js.size:
                        votes.extend(lbl[js])
                return votes

            if vis is None or vis_vote_mode == "off":
                votes = collect_votes()
                if votes:
                    vals, counts = np.unique(np.array(votes, dtype=lbl.dtype), return_counts=True)
                    new_lbl[idxs_np] = vals[counts.argmax()]
                continue

            idxs_vis_true  = idxs_np[vis[idxs_np]]
            idxs_vis_false = idxs_np[~vis[idxs_np]]

            if idxs_vis_true.size:
                votes_v = collect_votes(lambda js: js[vis[js] == True])
                if votes_v:
                    vals, counts = np.unique(np.array(votes_v, dtype=lbl.dtype), return_counts=True)
                    new_lbl[idxs_vis_true] = vals[counts.argmax()]

            if idxs_vis_false.size:
                votes_i = collect_votes(None if vis_vote_mode != "strict" else (lambda js: js[vis[js] == False]))
                if votes_i:
                    vals, counts = np.unique(np.array(votes_i, dtype=lbl.dtype), return_counts=True)
                    new_lbl[idxs_vis_false] = vals[counts.argmax()]

        lbl = new_lbl

    # Snap tiny components (global)
    if min_component > 0:
        counts = defaultdict(int)
        lv = list(zip(lbl.tolist(), key.tolist()))
        for pair in lv:
            counts[pair] += 1
        small = {pair for pair, c in counts.items() if c < min_component}
        if small:
            for i, pair in enumerate(lv):
                x = (pair[1] >> 42) & ((1 << 21) - 1)
                y = (pair[1] >> 21) & ((1 << 21) - 1)
                z =  pair[1]        & ((1 << 21) - 1)
                neigh_keys = ((x + offs[:, 0]) << 42) + ((y + offs[:, 1]) << 21) + (z + offs[:, 2])
                votes = []
                for nk in neigh_keys:
                    js = vox2idx.get(int(nk))
                    if js: votes.extend(lbl[js])
                if votes:
                    vals, c2 = np.unique(np.array(votes, dtype=lbl.dtype), return_counts=True)
                    lbl[i] = vals[c2.argmax()]
    return lbl


# --------------------- Optional: outlier pruning ------------------------- #

def _voxel_density_outliers(grid_coord: torch.Tensor, frac: float = 0.02) -> np.ndarray:
    if grid_coord.numel() == 0 or frac <= 0.0:
        return np.zeros((grid_coord.shape[0],), dtype=bool)
    g = grid_coord.cpu().numpy().astype(np.int64)
    key = (g[:, 0] << 42) + (g[:, 1] << 21) + g[:, 2]
    uniq, cnts = np.unique(key, return_counts=True)
    count_map = dict(zip(uniq.tolist(), cnts.tolist()))
    dens = np.array([count_map[int(k)] for k in key], dtype=np.int32)
    thr = np.quantile(dens, float(frac))
    return dens <= thr


def _fill_low_conf_by_neighbors(grid_coord: torch.Tensor, labels: np.ndarray, lowconf_mask: np.ndarray, neighbor_range: int = 1) -> np.ndarray:
    if not lowconf_mask.any():
        return labels
    g = grid_coord.cpu().numpy().astype(np.int64)
    key = (g[:, 0] << 42) + (g[:, 1] << 21) + g[:, 2]
    vox2idx: Dict[int, List[int]] = defaultdict(list)
    for i, k in enumerate(key):
        vox2idx[int(k)].append(i)
    offs = np.array([(dx, dy, dz)
                     for dx in range(-neighbor_range, neighbor_range + 1)
                     for dy in range(-neighbor_range, neighbor_range + 1)
                     for dz in range(-neighbor_range, neighbor_range + 1)], dtype=np.int64)
    new_lbl = labels.copy()
    for i in np.where(lowconf_mask)[0]:
        z =  key[i]        & ((1 << 21) - 1)
        y = (key[i] >> 21) & ((1 << 21) - 1)
        x = (key[i] >> 42) & ((1 << 21) - 1)
        neigh_keys = ((x + offs[:, 0]) << 42) + ((y + offs[:, 1]) << 21) + (z + offs[:, 2])
        votes = []
        for nk in neigh_keys:
            js = vox2idx.get(int(nk))
            if js:
                js = [j for j in js if not lowconf_mask[j]]
                if js: votes.extend(labels[js])
        if votes:
            vals, counts = np.unique(np.array(votes, dtype=labels.dtype), return_counts=True)
            new_lbl[i] = vals[counts.argmax()]
    return new_lbl


# ---------- Segmentation (nearest prototype + optional smoothing) --------- #

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
    soft_assign: bool = True,
    tau: float = 15.0,
    conf_th: float = 0.6,
    vis_aware_smooth: bool = True,
    vis_vote_mode: str = "one_way",
    outlier_prune: bool = False,
    outlier_frac: float = 0.02,
    # feature/smoothing:
    feature_cfg: Optional[Dict] = None,
    adaptive_smoothing: bool = True,
    dist_edges: Optional[List[float]] = None,
    bin_neighbor_ranges: Optional[List[int]] = None,
    bin_min_components: Optional[List[int]] = None,
    # NEW: save labels to disk (cluster → local)
    save_labels_dir: Optional[Path] = None,
) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Dict[str, List[torch.Tensor]]]]:
    """
    Segment dataset and optionally save per-frame label files:
      <save_labels_dir>/<file_stem>_clusters.npz  with {"labels": int32}
    """
    if feature_cfg is None:
        feature_cfg = {}
    if dist_edges is None:
        dist_edges = [0.0, 20.0, 40.0]
    if bin_neighbor_ranges is None:
        bin_neighbor_ranges = [neighbor_range, max(1, neighbor_range+1), max(1, neighbor_range+2)]
    if bin_min_components is None:
        bin_min_components = [min_component, int(min_component*1.5), int(min_component*2.0)]

    dev = torch.device(device)
    C = F.normalize(centroids.to(torch.float32), dim=1).to(dev, non_blocking=True)
    results: Dict[str, np.ndarray] = {}
    accum = {"feats": [], "labels": [], "speeds": [], "mask": []} if collect_metrics else None

    if save_labels_dir is not None:
        save_labels_dir = Path(save_labels_dir)
        save_labels_dir.mkdir(parents=True, exist_ok=True)

    for item in tqdm(extract_features(infer_dir, return_iter=True), desc="Segmenting + (opt) metrics/export"):
        stem = item["file_stem"]

        Z0 = _build_features(item, feature_cfg)
        if Z0.numel() == 0:
            lbl = np.empty((0,), dtype=np.int64)
            if save_labels_dir is not None:
                np.savez_compressed(save_labels_dir / f"{stem}_clusters.npz", labels=lbl.astype(np.int32))
            results[stem] = lbl
            continue

        Z_norm = F.normalize(Z0, dim=1)
        N = Z_norm.shape[0]
        idx_hard = np.empty((N,), dtype=np.int64)
        low_conf_mask = np.zeros((N,), dtype=bool)

        for s in range(0, N, assign_chunk):
            Ze = Z_norm[s:s+assign_chunk].to(dev, non_blocking=True)
            sim = Ze @ C.T
            idx = sim.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            idx_hard[s:s+idx.shape[0]] = idx

            if soft_assign:
                _, conf = _soft_assign(sim, tau=float(tau))
                conf_np = conf.detach().cpu().numpy()
                if item["mask"].numel():
                    vis_np = item["mask"].cpu().numpy()
                    lo = conf_np < float(conf_th)
                    slice_idx = np.arange(s, s+conf_np.shape[0])
                    low_conf_mask[slice_idx[vis_np[slice_idx]]] |= lo[vis_np[slice_idx]]
                else:
                    low_conf_mask[s:s+conf_np.shape[0]] |= (conf_np < float(conf_th))

        idx = idx_hard.copy()

        bin_masks = _distance_bins(item["coord_raw"], dist_edges)
        if outlier_prune:
            out_mask = _voxel_density_outliers(item["grid_coord"], frac=float(outlier_frac))
            vis_np = item["mask"].cpu().numpy() if item["mask"].numel() else np.zeros_like(out_mask, dtype=bool)
            low_conf_mask[vis_np] |= out_mask[vis_np]

        if smoothing_iters > 0 and adaptive_smoothing:
            for b, bm in enumerate(bin_masks):
                idx = smooth_labels_voxel(
                    grid_coord=item["grid_coord"],
                    labels=idx,
                    iters=smoothing_iters,
                    neighbor_range=int(bin_neighbor_ranges[min(b, len(bin_neighbor_ranges)-1)]),
                    min_component=int(bin_min_components[min(b, len(bin_min_components)-1)]),
                    visible_mask=(item["mask"] if vis_aware_smooth else None),
                    vis_vote_mode=("one_way" if vis_aware_smooth else "off"),
                    region_mask=bm.cpu().numpy(),
                )
        elif smoothing_iters > 0:
            idx = smooth_labels_voxel(
                grid_coord=item["grid_coord"],
                labels=idx,
                iters=smoothing_iters,
                neighbor_range=neighbor_range,
                min_component=min_component,
                visible_mask=(item["mask"] if vis_aware_smooth else None),
                vis_vote_mode=("one_way" if vis_aware_smooth else "off"),
            )

        if low_conf_mask.any():
            idx = _fill_low_conf_by_neighbors(
                grid_coord=item["grid_coord"],
                labels=idx,
                lowconf_mask=low_conf_mask,
                neighbor_range=max(1, neighbor_range),
            )

        # Save per-frame labels (compact)
        if save_labels_dir is not None:
            np.savez_compressed(save_labels_dir / f"{stem}_clusters.npz", labels=idx.astype(np.int32))

        results[stem] = idx

        if per_frame_hook is not None:
            per_frame_hook(stem, item["coord_raw"].cpu().numpy().astype(np.float32), idx)

        if collect_metrics and accum is not None:
            accum["feats"].append(item["feat64"])  # keep 64-D baseline for metrics comparability
            accum["labels"].append(torch.as_tensor(idx, dtype=torch.int64))
            accum["mask"].append(item["mask"])
            accum["speeds"].append(item.get("speed", torch.empty((0,), dtype=torch.float32)))

    if collect_metrics and accum is not None:
        return results, accum
    return results


# ----------------------- Metrics from accumulated pass -------------------- #

def evaluate_accumulated_metrics(
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
    results = compute_all_metrics(X, Y, speeds=V, visible_mask=M, token_labels_proj=None, cfg=cfg)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as w:
        w.write("metric,value\n")
        for k, v in results.items():
            w.write(f"{k},{'' if v is None else v}\n")
    print(f"[metrics] wrote {out_csv}")
    return results
