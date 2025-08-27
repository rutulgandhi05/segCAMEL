from pathlib import Path
import time
import zipfile
import torch
import os
from typing import Optional, Tuple

# --- Core paths (env-driven, unchanged) ---
INFER_DIR   = Path(os.environ.get("INFERENCE_OUTPUT_DIR"))        # where *_inference.pth live
OUT_DIR     = Path(os.environ.get("SEGMENTATION_OUT_DIR"))        # root for outputs
K           = 8                                                   # number of clusters
ZIP_PATH    = OUT_DIR / f"labels_k{K}.zip"                        # labels go into this single zip (fresh each run)
PROTOS_PATH = OUT_DIR / "prototypes.pt"
RUN_CFG_JSON= OUT_DIR / "run_config.json"
METRICS_CSV = OUT_DIR / f"metrics_k{K}.csv"

# --- Execution toggles ---
RUN_INFERENCE = True    # set True to call inference.py first
DO_FIT        = True    # learn prototypes (False -> load from PROTOS_PATH)

# --- Segmentation mode ---
#   "kmeans": legacy cosine k-means + hard argmax (uses TAU_REJECT)
#   "vmf"   : von Mises–Fisher mixture + soft posteriors (uses POSTERIOR_TAU / per-bin posterior map)
MODE = "vmf"  # "kmeans" | "vmf"

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Inference settings (only used if RUN_INFERENCE=True) ---
DATA_DIR          = Path(os.getenv("PREPROCESS_OUTPUT_DIR"))
CKPT_PATH         = Path(os.getenv("TRAIN_CHECKPOINTS")) / "best_model.pth"
INFERENCE_BATCH   = 4
INFERENCE_WORKERS = 12
VOXEL_SIZE        = 0.10
FEAT_MODE         = "rvi"   # "none"|"ri"|"v"|"rvi" (must match training)

# --- Prototype learning (quality-first defaults) ---
MAX_PASSES        = 3
SAMPLE_PER_FRAME  = 50_000
USE_FP16_MATMUL   = True
SEED              = 0
DIST_EDGES        = [0.0, 15.0, 30.0, 60.0]
DIST_RATIOS       = [0.40, 0.35, 0.20, 0.05]

# --- Segmentation ---
SMOOTH_ITERS      = 2
NEIGHBOR_RANGE    = 1
MIN_COMPONENT     = 80
ASSIGN_CHUNK      = 4_000_000
USE_FP16_ASSIGN   = True
TAU_REJECT        = 0.10      # (k-means) low-confidence cosine → noise label
NOISE_LABEL       = -1
RANGE_GATE_M      = 1.5       # depth-aware smoothing (meters)

# --- vMF posterior thresholds (used only if MODE == "vmf") ---
POSTERIOR_TAU     = 0.10      # global posterior min threshold
TAU_EDGES         = [0.0, 15.0, 30.0, 60.0]   # per-distance posterior thresholds
TAU_MAP           = [0.08, 0.10, 0.12, 0.14]  # near→far; used if you prefer distance-aware instead of global

# --- Feature augmentation (better separation) ---
FEATURE_CFG = {
    "use_range":  True,  "range_scale": 100.0,
    "use_height": True,  "height_scale": 3.0,
    "use_speed":  True,  "speed_scale": 30.0,
}

# --- DataLoader I/O knobs ---
DL_WORKERS    = 8
DL_PREFETCH   = 4
DL_BATCH_IO   = 32
DL_PIN_MEMORY = True

# Robust import (package vs flat)
try:
    from scripts.unsup_seg_pipeline import (
        learn_prototypes_from_dataset,
        learn_vmf_from_dataset,
        segment_dataset,
        evaluate_accumulated_metrics,
        save_run_config,
    )
except Exception:
    from unsup_seg_pipeline import (
        learn_prototypes_from_dataset,
        learn_vmf_from_dataset,
        segment_dataset,
        evaluate_accumulated_metrics,
        save_run_config,
    )

# Optional inference hook: try package path first, then flat path.
_HAS_INFER = False
_inference_mod = None
try:
    import scripts.inference as _inference_mod
    _HAS_INFER = True
except Exception:
    try:
        import inference as _inference_mod
        _HAS_INFER = True
    except Exception:
        _inference_mod = None
        _HAS_INFER = False


def _count_infer_files(infer_dir: Path) -> int:
    return len(list(Path(infer_dir).glob("*_inference.pth")))

def _zip_entries_count(zip_path: Path) -> int:
    if not zip_path.exists():
        return 0
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return len(zf.namelist())
    except Exception:
        return 0

def _ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)

def _maybe_run_inference():
    if not RUN_INFERENCE:
        return
    if not _HAS_INFER or not hasattr(_inference_mod, "run_inference"):
        raise RuntimeError("RUN_INFERENCE=True but inference.run_inference() not importable.")
    print("[segment_once] Running inference…")
    t0 = time.time()
    _inference_mod.run_inference(
        data_dir=DATA_DIR,
        checkpoint_path=CKPT_PATH,
        out_dir=INFER_DIR,
        voxel_size=VOXEL_SIZE,
        multiscale_voxel=True,   # <<< enable multi-scale voxel selection
        batch_size=INFERENCE_BATCH,
        workers=INFERENCE_WORKERS,
        device=DEVICE,
        feat_mode=FEAT_MODE,
    )
    print(f"[segment_once] Inference done in {time.time()-t0:.1f}s")

def _fit_prototypes() -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
      centroids : (K, D) unit vectors
      kappa     : (K,) or None (k-means)
    """
    print(f"[segment_once] Learning prototypes (mode={MODE})…")
    t0 = time.time()

    if MODE.lower() == "vmf":
        mu, kappa = learn_vmf_from_dataset(
            INFER_DIR,
            k=K,
            max_passes=MAX_PASSES,
            sample_per_frame=SAMPLE_PER_FRAME,
            seed=SEED,
            use_visible_for_prototypes=True,
            device=DEVICE,
            update_chunk=1_000_000,
            feature_cfg=FEATURE_CFG,
            dist_edges=DIST_EDGES,
            dist_ratios=DIST_RATIOS,
            use_fp16=USE_FP16_MATMUL,
            # DataLoader knobs
            dl_workers=DL_WORKERS,
            dl_prefetch=DL_PREFETCH,
            dl_batch_io=DL_BATCH_IO,
            dl_pin_memory=DL_PIN_MEMORY,
        )
        torch.save({"centroids": mu, "kappa": kappa, "mode": "vmf"}, PROTOS_PATH)
    else:
        centroids = learn_prototypes_from_dataset(
            INFER_DIR,
            k=K,
            max_passes=MAX_PASSES,
            sample_per_frame=SAMPLE_PER_FRAME,
            seed=SEED,
            use_visible_for_prototypes=True,
            invisible_weight=1.0,
            device=DEVICE,
            update_chunk=1_000_000,
            feature_cfg=FEATURE_CFG,
            dist_edges=DIST_EDGES,
            dist_ratios=DIST_RATIOS,
            use_fp16=USE_FP16_MATMUL,
            # DataLoader knobs
            dl_workers=DL_WORKERS,
            dl_prefetch=DL_PREFETCH,
            dl_batch_io=DL_BATCH_IO,
            dl_pin_memory=DL_PIN_MEMORY,
        )
        torch.save({"centroids": centroids, "mode": "kmeans"}, PROTOS_PATH)
        mu, kappa = centroids, None

    save_run_config(RUN_CFG_JSON, {
        "mode": MODE,
        "K": K,
        "feature_cfg": FEATURE_CFG,
        "smooth_iters": SMOOTH_ITERS,
        "neighbor_range": NEIGHBOR_RANGE,
        "min_component": MIN_COMPONENT,
        "tau_reject": TAU_REJECT,
        "posterior_tau": POSTERIOR_TAU,
        "tau_edges": DIST_EDGES,
        "tau_map": TAU_MAP,
        "noise_label": NOISE_LABEL,
        "range_gate_m": RANGE_GATE_M,
        "dl_workers": DL_WORKERS,
        "dl_prefetch": DL_PREFETCH,
        "dl_batch_io": DL_BATCH_IO,
        "dl_pin_memory": DL_PIN_MEMORY,
        "zip_mode": "w",
    })
    print(f"[segment_once] Saved prototypes -> {PROTOS_PATH}  ({time.time()-t0:.1f}s)")
    return mu, kappa

def _apply_segmentation(centroids: torch.Tensor, kappa: Optional[torch.Tensor]):
    print(f"[segment_once] Segmenting dataset… (mode={MODE}, no resume)")
    t0 = time.time()

    # Choose thresholding strategy
    seg_kwargs = {}
    if MODE.lower() == "vmf":
        # Prefer distance-aware posterior thresholds; else fall back to global posterior_tau
        seg_kwargs.update(dict(
            mode="vmf",
            vmf_kappa=kappa,
            posterior_tau=POSTERIOR_TAU if (TAU_EDGES is None or TAU_MAP is None) else None,
            tau_edges=TAU_EDGES if (TAU_EDGES and TAU_MAP) else None,
            tau_map=TAU_MAP   if (TAU_EDGES and TAU_MAP) else None,
        ))
    else:
        # k-means cosine mode (legacy)
        seg_kwargs.update(dict(
            mode="kmeans",
            tau_reject=TAU_REJECT,
            tau_edges=None,
            tau_map=None,
        ))

    # Fresh ZIP each run (no resume): zip_mode="w"
    results, accum = segment_dataset(
        INFER_DIR,
        centroids=centroids,
        smoothing_iters=SMOOTH_ITERS,
        neighbor_range=NEIGHBOR_RANGE,
        min_component=MIN_COMPONENT,
        per_frame_hook=None,
        collect_metrics=True,
        device=DEVICE,
        assign_chunk=ASSIGN_CHUNK,
        feature_cfg=FEATURE_CFG,
        use_fp16=USE_FP16_ASSIGN,
        # ZIP-only persistence (fresh)
        save_labels_dir=None,
        zip_labels_path=ZIP_PATH,
        zip_mode="w",
        zip_compress=zipfile.ZIP_DEFLATED,
        # quality knobs (shared)
        noise_label=NOISE_LABEL,
        range_gate_m=RANGE_GATE_M,
        # DataLoader knobs
        dl_workers=DL_WORKERS,
        dl_prefetch=DL_PREFETCH,
        dl_batch_io=DL_BATCH_IO,
        dl_pin_memory=DL_PIN_MEMORY,
        # new mode-specific args
        **seg_kwargs,
    )
    proc = len(results)                      # processed frames (incl. empties)
    written = _zip_entries_count(ZIP_PATH)   # actually saved (non-empty only)
    skipped = max(0, proc - written)         # empties skipped by the pipeline
    print(f"[segment_once] Done in {time.time()-t0:.1f}s; "
          f"processed={proc}, written={written}, skipped_empties={skipped}; "
          f"zip -> {ZIP_PATH}")
    return accum

def _compute_metrics(accum):
    print("[segment_once] Computing metrics CSV…")
    evaluate_accumulated_metrics(
        accum,
        out_csv=METRICS_CSV,
        sample_n=50_000,
        seed=42,
        q_bins=4,
        tau_list=[0.2, 0.4, 0.6],
        tau_policy="quantile",
    )
    print(f"[segment_once] Wrote -> {METRICS_CSV}")

def main():
    _ensure_dirs()

    # (optional) inference
    _maybe_run_inference()

    # check coverage
    n_inf = _count_infer_files(INFER_DIR)
    if n_inf == 0:
        raise SystemExit(f"[segment_once] No *_inference.pth in {INFER_DIR}. Did you run inference?")

    # fit or load
    if DO_FIT:
        centroids, kappa = _fit_prototypes()
    else:
        if not PROTOS_PATH.exists():
            raise SystemExit(f"[segment_once] DO_FIT=False but prototypes not found at {PROTOS_PATH}")
        payload = torch.load(PROTOS_PATH, map_location="cpu")
        centroids = payload["centroids"]
        kappa = payload.get("kappa", None)
        print(f"[segment_once] Loaded prototypes from {PROTOS_PATH} (mode={payload.get('mode','kmeans')})")

    # segment + metrics
    accum = _apply_segmentation(centroids, kappa)
    _compute_metrics(accum)

    print("\n[summary]")
    print(f"  Mode      : {MODE}")
    print(f"  Prototypes: {PROTOS_PATH if PROTOS_PATH.exists() else '(not saved)'}")
    print(f"  Labels ZIP: {ZIP_PATH}")
    print(f"  Metrics   : {METRICS_CSV}")
    print("  Done.\n")

if __name__ == "__main__":
    main()
