# scripts/metrics.py
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:
    torch = None  # type: ignore

# Optional: scikit-learn for standard clustering metrics
try:
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score,
        calinski_harabasz_score,
        normalized_mutual_info_score,
        f1_score,
    )
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False
    silhouette_score = None  # type: ignore
    davies_bouldin_score = None  # type: ignore
    calinski_harabasz_score = None  # type: ignore
    normalized_mutual_info_score = None  # type: ignore
    f1_score = None  # type: ignore


# ------------------------------ Utilities ------------------------------ #

def _to_numpy(x) -> np.ndarray:
    """Accepts numpy arrays or torch tensors and returns a NumPy array (CPU)."""
    if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else 42)


def _downsample_idx(n: int, sample_n: int, seed: Optional[int]) -> np.ndarray:
    """Uniformly sample up to sample_n indices from range(n)."""
    if sample_n is None or sample_n <= 0 or sample_n >= n:
        return np.arange(n, dtype=np.int64)
    g = _rng(seed)
    return g.choice(n, size=sample_n, replace=False)


def _has_enough_clusters(labels: np.ndarray, min_clusters: int = 2) -> bool:
    return len(np.unique(labels)) >= min_clusters


def _zscore(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + eps)


def _l2norm(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def _apply_mask(*arrays, mask: Optional[np.ndarray]):
    if mask is None:
        return arrays
    m = np.asarray(mask).astype(bool)
    return tuple(np.asarray(a)[m] for a in arrays)


def _quantile_bins(v: np.ndarray, q_bins: int = 4) -> np.ndarray:
    """Bin |v| into q_bins quantiles → labels in [0..q_bins-1]."""
    v_abs = np.abs(v)
    # Ensure unique, sorted quantiles (avoid identical cuts on constant arrays)
    qs = np.linspace(0, 1, q_bins + 1)[1:-1]
    cuts = np.unique(np.quantile(v_abs, qs))
    # np.digitize returns indices 0..len(cuts), which is the intended [0..q_bins-1] even if cuts shrank
    return np.digitize(v_abs, cuts, right=True)


def _weighted_within_cluster_variance(values: np.ndarray, labels: np.ndarray) -> float:
    """Weighted average of per-cluster variance of 'values' (1D)."""
    uniq, counts = np.unique(labels, return_counts=True)
    if len(uniq) <= 1:
        return float(np.var(values))
    total = values.shape[0]
    out = 0.0
    for c, k in zip(uniq, counts):
        vs = values[labels == c]
        out += (k / total) * float(np.var(vs))
    return out


def _majority_map_to_binary(labels: np.ndarray, pos_mask: np.ndarray) -> np.ndarray:
    """
    For each cluster id, assign it to {0,1} by majority of pos_mask within that cluster.
    Returns binary predictions per point (same shape as labels).
    """
    uniq = np.unique(labels)
    pred_bin = np.zeros_like(labels, dtype=np.int64)
    for c in uniq:
        idx = labels == c
        # majority vote
        pos = int(np.sum(pos_mask[idx]))
        neg = int(idx.sum() - pos)
        pred_class = 1 if pos >= neg else 0
        pred_bin[idx] = pred_class
    return pred_bin


# Intrinsic Metrics 

def compute_intrinsic_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    visible_mask: Optional[np.ndarray] = None,
    sample_n: int = 200_000,
    seed: Optional[int] = 42,
) -> Dict[str, Optional[float]]:
    """
    Computes label-free clustering metrics on features X with cluster assignments 'labels'.

    - Silhouette (cosine) on an optional downsample
    - Davies–Bouldin (Euclidean) on z-scored features
    - Calinski–Harabasz (Euclidean) on z-scored features

    Returns a dict with keys:
      silhouette_cosine_[all|vis|invis], dbi_euclid_[...], ch_euclid_[...]
    """
    out: Dict[str, Optional[float]] = {}

    def _calc(X_, y_, suffix: str):
        # Guard: need at least 2 clusters and > 1 sample for metrics
        if X_.shape[0] < 3 or not _has_enough_clusters(y_):
            out[f"silhouette_cosine_{suffix}"] = None
            out[f"dbi_euclid_{suffix}"] = None
            out[f"ch_euclid_{suffix}"] = None
            return

        # Downsample uniformly for silhouette (which is O(n^2) distance-wise)
        idx = _downsample_idx(X_.shape[0], sample_n, seed)
        Xs, ys = X_[idx], y_[idx]

        # Silhouette (cosine)
        if _HAS_SKLEARN and silhouette_score is not None:
            try:
                out[f"silhouette_cosine_{suffix}"] = float(silhouette_score(Xs, ys, metric="cosine"))
            except Exception:
                out[f"silhouette_cosine_{suffix}"] = None
        else:
            warnings.warn("scikit-learn not available; silhouette will be None.")
            out[f"silhouette_cosine_{suffix}"] = None

        # DBI / CH (Euclidean) on standardized features
        Xe = _zscore(X_)  # center-scale per feature
        if _HAS_SKLEARN and davies_bouldin_score is not None and calinski_harabasz_score is not None:
            try:
                out[f"dbi_euclid_{suffix}"] = float(davies_bouldin_score(Xe, y_))
            except Exception:
                out[f"dbi_euclid_{suffix}"] = None
            try:
                out[f"ch_euclid_{suffix}"] = float(calinski_harabasz_score(Xe, y_))
            except Exception:
                out[f"ch_euclid_{suffix}"] = None
        else:
            warnings.warn("scikit-learn not available; DBI/CH will be None.")
            out[f"dbi_euclid_{suffix}"] = None
            out[f"ch_euclid_{suffix}"] = None

    # Prepare arrays
    X = _to_numpy(X)
    labels = _to_numpy(labels).astype(np.int64)

    # All
    _calc(X, labels, "all")

    # Visible / invisible splits if provided
    if visible_mask is not None:
        vis_mask = np.asarray(visible_mask).astype(bool)
        inv_mask = ~vis_mask

        if vis_mask.any():
            _calc(X[vis_mask], labels[vis_mask], "vis")
        else:
            out["silhouette_cosine_vis"] = None
            out["dbi_euclid_vis"] = None
            out["ch_euclid_vis"] = None

        if inv_mask.any():
            _calc(X[inv_mask], labels[inv_mask], "invis")
        else:
            out["silhouette_cosine_invis"] = None
            out["dbi_euclid_invis"] = None
            out["ch_euclid_invis"] = None

    return out


# Velocity Metrics

def compute_velocity_metrics(
    labels: np.ndarray,
    speeds: np.ndarray,
    *,
    visible_mask: Optional[np.ndarray] = None,
    q_bins: int = 4,
    tau_list: Optional[Sequence[float]] = None,
    tau_policy: str = "quantile",  # "quantile" or "value"
) -> Dict[str, Optional[float]]:
    """
    Metrics that quantify how clusters relate to per-point speeds (|v|):

    - NMI(cluster_labels, speed_bins) with q_bins quantiles (default 4)
    - F1@τ for dynamic/static classification:
        dynamic = |v| > τ, clusters mapped to binary by majority
      (τ chosen by 'tau_policy': "quantile" uses speed quantiles; "value" uses raw thresholds)
    - Weighted within-cluster variance of |v|

    Returns keys:
      nmi_speedbins_[all|vis|invis],
      f1_dyn_tau{...}_[all|vis|invis]  (one per τ),
      velvar_wmean_[all|vis|invis]
    """
    out: Dict[str, Optional[float]] = {}

    labels = _to_numpy(labels).astype(np.int64)
    v = _to_numpy(speeds).astype(np.float32)
    vabs = np.abs(v)

    # Default τ list: 20/40/60th percentiles if none given (for "quantile" policy)
    if tau_list is None:
        tau_list = [0.2, 0.4, 0.6] if tau_policy == "quantile" else []

    def _calc(y: np.ndarray, v_: np.ndarray, suffix: str):
        if y.size == 0 or v_.size == 0:
            out[f"nmi_speedbins_{suffix}"] = None
            for tau in tau_list:
                out[f"f1_dyn_tau{tau}_{suffix}"] = None
            out[f"velvar_wmean_{suffix}"] = None
            return

        # NMI(cluster, speed-bins)
        if _HAS_SKLEARN and normalized_mutual_info_score is not None:
            try:
                sbins = _quantile_bins(v_, q_bins=q_bins)
                out[f"nmi_speedbins_{suffix}"] = float(normalized_mutual_info_score(y, sbins))
            except Exception:
                out[f"nmi_speedbins_{suffix}"] = None
        else:
            warnings.warn("scikit-learn not available; NMI will be None.")
            out[f"nmi_speedbins_{suffix}"] = None

        # F1@τ: dynamic mask versus cluster-majority mapped labels
        if _HAS_SKLEARN and f1_score is not None:
            for tau in tau_list:
                if tau_policy == "quantile":
                    thr = float(np.quantile(v_, tau))
                elif tau_policy == "value":
                    thr = float(tau)
                else:
                    raise ValueError("tau_policy must be 'quantile' or 'value'")
                dyn_true = (v_ > thr).astype(np.int64)
                dyn_pred = _majority_map_to_binary(y, dyn_true.astype(bool))
                try:
                    f1 = float(f1_score(dyn_true, dyn_pred, average="binary"))
                except Exception:
                    f1 = None
                out[f"f1_dyn_tau{tau}_{suffix}"] = f1
        else:
            warnings.warn("scikit-learn not available; F1 will be None.")
            for tau in tau_list:
                out[f"f1_dyn_tau{tau}_{suffix}"] = None

        # Weighted within-cluster |v| variance
        out[f"velvar_wmean_{suffix}"] = _weighted_within_cluster_variance(v_, y)

    # All
    _calc(labels, vabs, "all")

    # Visible / invisible splits if provided
    if visible_mask is not None:
        vis_mask = np.asarray(visible_mask).astype(bool)
        inv_mask = ~vis_mask

        _calc(labels[vis_mask], vabs[vis_mask], "vis") if vis_mask.any() else (
            out.setdefault("nmi_speedbins_vis", None),
            *[out.setdefault(f"f1_dyn_tau{t}_vis", None) for t in tau_list],
            out.setdefault("velvar_wmean_vis", None),
        )

        _calc(labels[inv_mask], vabs[inv_mask], "invis") if inv_mask.any() else (
            out.setdefault("nmi_speedbins_invis", None),
            *[out.setdefault(f"f1_dyn_tau{t}_invis", None) for t in tau_list],
            out.setdefault("velvar_wmean_invis", None),
        )

    return out


# 2D Alignment (Proxy)

def compute_2d_alignment_metrics(
    labels_3d: np.ndarray,
    labels_2dproj: np.ndarray,
    *,
    visible_mask: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """
    Compares 3D clusters to projected 2D token groups via NMI.

    'labels_2dproj' can include -1 for 'no token' / 'not visible'; these are excluded.
    Returns keys: nmi_2dproj_[all|vis|invis]
    """
    out: Dict[str, Optional[float]] = {}
    y3 = _to_numpy(labels_3d).astype(np.int64)
    y2 = _to_numpy(labels_2dproj).astype(np.int64)

    def _calc(y3_, y2_, suffix: str):
        # Filter out unlabeled 2D projections
        valid = y2_ >= 0
        if not valid.any() or not _has_enough_clusters(y3_[valid]):
            out[f"nmi_2dproj_{suffix}"] = None
            return
        if _HAS_SKLEARN and normalized_mutual_info_score is not None:
            try:
                out[f"nmi_2dproj_{suffix}"] = float(normalized_mutual_info_score(y3_[valid], y2_[valid]))
            except Exception:
                out[f"nmi_2dproj_{suffix}"] = None
        else:
            warnings.warn("scikit-learn not available; NMI will be None.")
            out[f"nmi_2dproj_{suffix}"] = None

    _calc(y3, y2, "all")

    if visible_mask is not None:
        vis = np.asarray(visible_mask).astype(bool)
        inv = ~vis
        if vis.any():
            _calc(y3[vis], y2[vis], "vis")
        else:
            out["nmi_2dproj_vis"] = None
        if inv.any():
            _calc(y3[inv], y2[inv], "invis")
        else:
            out["nmi_2dproj_invis"] = None

    return out


# Orchestrator

@dataclass
class MetricsConfig:
    sample_n: int = 200_000
    seed: int = 42
    q_bins: int = 4
    tau_list: Optional[Sequence[float]] = None  # defaults to [0.2,0.4,0.6] (quantiles)
    tau_policy: str = "quantile"               # or "value"


def compute_all_metrics(
    feats_64d: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    speeds: Optional[np.ndarray] = None,
    visible_mask: Optional[np.ndarray] = None,
    token_labels_proj: Optional[np.ndarray] = None,
    cfg: MetricsConfig = MetricsConfig(),
) -> Dict[str, Optional[float]]:
    """
    Convenience wrapper to compute all metrics you care about in one shot.

    Required:
      - feats_64d: (N,64) model features from inference
      - cluster_labels: (N,) k-means labels

    Optional:
      - speeds: (N,) per-point velocity magnitudes (or signed; abs is used)
      - visible_mask: (N,) boolean
      - token_labels_proj: (N,) projected 2D token group IDs (use -1 for 'no label')

    Returns a flat dict of scalar metrics (floats or None).
    """
    X = _to_numpy(feats_64d)
    y = _to_numpy(cluster_labels).astype(np.int64)

    out: Dict[str, Optional[float]] = {}
    out.update(
        compute_intrinsic_metrics(
            X, y, visible_mask=visible_mask, sample_n=cfg.sample_n, seed=cfg.seed
        )
    )

    if speeds is not None:
        out.update(
            compute_velocity_metrics(
                labels=y,
                speeds=speeds,
                visible_mask=visible_mask,
                q_bins=cfg.q_bins,
                tau_list=cfg.tau_list,
                tau_policy=cfg.tau_policy,
            )
        )

    if token_labels_proj is not None:
        out.update(
            compute_2d_alignment_metrics(
                labels_3d=y,
                labels_2dproj=token_labels_proj,
                visible_mask=visible_mask,
            )
        )

    return out
