from pathlib import Path
from typing import Optional
import json
import numpy as np
import torch
import os


# ----- Paths -----
INFER_DIR = Path(os.environ.get("INFERENCE_OUTPUT_DIR"))
OUT_DIR   = Path(os.environ.get("SEGMENTATION_OUT_DIR"))
K = 10  # back to stable baseline


# ----- Modes -----
# "fit_and_apply" : learn prototypes, segment, save everything
# "fit_only"      : learn prototypes, save prototypes+cfg (no segmentation)
# "apply_only"    : load prototypes from LOAD_PROTOTYPES_PATH, segment
# "view_labels"   : dump saved labels to PLY/PNGs (no compute)
# "view_o3d"      : **NEW** interactive Open3D viewer (no file writes)
MODE = "fit_and_apply"

LOAD_PROTOTYPES_PATH = OUT_DIR / "prototypes.pt"
RUN_CFG_JSON         = OUT_DIR / "run_config.json"
LABELS_DIR           = OUT_DIR / f"labels_k{K}"

# ----- Segmentation knobs -----
SMOOTH_ITERS = 1
NEIGHBOR_RANGE = 1
MIN_COMPONENT = 120

# ----- Viz / I-O -----
PLY_LIMIT = None
DO_OPEN3D_VIEW = False
SAVE_PNG = False
SAVE_PLY = False
PNG_W = 1600
PNG_H = 1200

# ----- Metrics -----
METRICS_SAMPLE_N = 50_000
METRICS_CSV = OUT_DIR / f"metrics_k{K}.csv"

# ----- Prototype learning behavior -----
USE_VISIBLE_FOR_PROTOS = True
INVISIBLE_WEIGHT = 1.0

# ----- Speed knobs (use these) -----
USE_GPU = torch.cuda.is_available()
PROTOTYPE_DEVICE = "cuda" if USE_GPU else "cpu"
SEGMENT_DEVICE   = "cuda" if USE_GPU else "cpu"
PROTOTYPE_CHUNK  = 1_000_000
ASSIGN_CHUNK     = 3_000_000
PREFETCH_BUFFER  = 4          # more overlap
USE_FP16_MATMUL  = True       # use fp16 for @ (confidence in fp32)
AUTO_CHUNK       = True       # auto-size chunk per frame

# ----- Post-DITR options -----
SOFT_ASSIGN = True
TAU = 15.0
CONF_TH = 0.6
VIS_AWARE_SMOOTH = True
VIS_VOTE_MODE = "one_way"
OUTLIER_PRUNE = False
OUTLIER_FRAC = 0.02

# ----- Distance-aware & Feature-aug (defaults off here) -----
DIST_EDGES = [0.0, 20.0, 40.0]
DIST_RATIOS = [0.5, 0.35, 0.15]
BIN_NEIGHBOR_RANGES = [1, 2, 3]
BIN_MIN_COMPONENTS  = [MIN_COMPONENT, int(MIN_COMPONENT*1.5), int(MIN_COMPONENT*2.0)]

FEATURE_CFG = {
    "use_range":  False,  "range_scale": 60.0,
    "use_height": False,  "height_scale": 6.0,
    "use_speed":  False,  "speed_scale": 25.0,
}

from scripts.unsup_seg_pipeline import (
    learn_prototypes_from_dataset,
    segment_dataset,
    evaluate_accumulated_metrics,
    save_run_config,
    load_run_config,
)

# ---------- Optional Open3D viewer (unchanged) ----------
try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False

# (helper functions for palette / rendering omitted for brevity — keep your current ones)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    if MODE in ("fit_only", "fit_and_apply"):
        print("[fit] Learning prototypes…")
        centroids = learn_prototypes_from_dataset(
            INFER_DIR,
            k=K,
            max_passes=2,
            sample_per_frame=20000,
            seed=0,
            use_visible_for_prototypes=USE_VISIBLE_FOR_PROTOS,
            invisible_weight=INVISIBLE_WEIGHT,
            device=PROTOTYPE_DEVICE,
            update_chunk=PROTOTYPE_CHUNK,
            feature_cfg=FEATURE_CFG,
            dist_edges=DIST_EDGES,
            dist_ratios=DIST_RATIOS,
            prefetch_buffer=PREFETCH_BUFFER,
            use_fp16=USE_FP16_MATMUL,
        )
        torch.save({"centroids": centroids}, OUT_DIR / "prototypes.pt")
        save_run_config(RUN_CFG_JSON, {"K": K})

        if MODE == "fit_only":
            print("[fit_only] Done.")
            return

    if MODE in ("apply_only", "fit_and_apply"):
        if MODE == "apply_only":
            centroids = torch.load(LOAD_PROTOTYPES_PATH, map_location="cpu")["centroids"]

        print("[apply] Segmenting dataset and saving per-frame label dumps…")
        results, accum = segment_dataset(
            INFER_DIR,
            centroids=centroids,
            smoothing_iters=SMOOTH_ITERS,
            neighbor_range=NEIGHBOR_RANGE,
            min_component=MIN_COMPONENT,
            per_frame_hook=None,
            collect_metrics=True,
            device=SEGMENT_DEVICE,
            assign_chunk=ASSIGN_CHUNK,
            soft_assign=SOFT_ASSIGN,
            tau=TAU,
            conf_th=CONF_TH,
            vis_aware_smooth=VIS_AWARE_SMOOTH,
            vis_vote_mode=VIS_VOTE_MODE,
            outlier_prune=OUTLIER_PRUNE,
            outlier_frac=OUTLIER_FRAC,
            feature_cfg=FEATURE_CFG,
            adaptive_smoothing=True,
            dist_edges=DIST_EDGES,
            bin_neighbor_ranges=BIN_NEIGHBOR_RANGES,
            bin_min_components=BIN_MIN_COMPONENTS,
            save_labels_dir=LABELS_DIR,
            prefetch_buffer=PREFETCH_BUFFER,
            use_fp16=USE_FP16_MATMUL,
            auto_chunk=AUTO_CHUNK,
        )

        print("[apply] Computing metrics…")
        evaluate_accumulated_metrics(
            accum,
            out_csv=METRICS_CSV,
            sample_n=METRICS_SAMPLE_N,
            seed=42,
            q_bins=4,
            tau_list=[0.2, 0.4, 0.6],
            tau_policy="quantile",
        )

    print(f"[done] Artifacts under: {OUT_DIR}")

if __name__ == "__main__":
    main()