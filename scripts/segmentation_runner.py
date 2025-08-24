from pathlib import Path
import sys
import time
import torch
import os
import zipfile


# --- Core paths (with safe fallbacks if envs are not set) ---
INFER_OUT_DIR = Path(os.environ.get("INFERENCE_OUTPUT_DIR"))  # *_inference.pth live here
OUT_DIR       = Path(os.environ.get("SEGMENTATION_OUT_DIR"))     # prototypes/labels/metrics root
K             = 10
LABELS_DIR    = OUT_DIR / f"labels_k{K}"              # used if SAVE_PER_FILE=True
ZIP_PATH      = OUT_DIR / f"labels_k{K}.zip"          # used if USE_ZIP=True
PROTOS_PATH   = OUT_DIR / "prototypes.pt"
RUN_CFG_JSON  = OUT_DIR / "run_config.json"

# --- Execution toggles ---
RUN_INFERENCE = True      # run your inference.py first (kept "as-is")
DO_FIT        = True      # learn prototypes (set False to reuse existing PROTOS_PATH)
DO_APPLY      = True      # segment dataset and persist labels
RESUME        = False     # skip frames already labeled (dir or zip, depending on settings)

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Inference settings (used only if RUN_INFERENCE=True) ---
DATA_DIR          = Path(os.getenv("PREPROCESS_OUTPUT_DIR"))       # input to inference
CKPT_DIR          = Path(os.getenv("TRAIN_CHECKPOINTS"))
CKPT_PATH         = CKPT_DIR / "best_model.pth"
INFERENCE_BATCH   = 4
INFERENCE_WORKERS = 12
VOXEL_SIZE        = 0.10
FEAT_MODE         = "rvi"   # "none"|"ri"|"v"|"rvi" (depends on your inference.py)

# --- Prototype learning ---
MAX_PASSES        = 2
SAMPLE_PER_FRAME  = 20000
PREFETCH_BUFFER   = 4
USE_FP16_MATMUL   = True
SEED              = 0
DIST_EDGES        = [0.0, 20.0, 40.0]
DIST_RATIOS       = [0.5, 0.35, 0.15]

# --- Segmentation ---
SMOOTH_ITERS      = 1
NEIGHBOR_RANGE    = 1
MIN_COMPONENT     = 120
ASSIGN_CHUNK      = 3_000_000
USE_FP16_ASSIGN   = True

# --- Label persistence options ---
# Save per-frame NPZs to LABELS_DIR?
SAVE_PER_FILE     = False
# Also/alternatively: write labels directly into a single ZIP (resume-friendly with append).
USE_ZIP           = True
ZIP_MODE          = "a"          # "w" to overwrite, "a" to append/resume
ZIP_SKIP_EXISTING = True         # when appending, skip entries already present

# Optional feature augmentation (kept off by default for DITR-aligned runs)
FEATURE_CFG = {
    "use_range":  False,  "range_scale": 60.0,
    "use_height": False,  "height_scale": 6.0,
    "use_speed":  False,  "speed_scale": 25.0,
}

# --- Metrics ---
METRICS_SAMPLE_N  = 50_000
from scripts.unsup_seg_pipeline import (
        extract_features,
        learn_prototypes_from_dataset,
        segment_dataset,
        evaluate_accumulated_metrics,
        save_run_config,
        resume_stream,
    )

# optional inference hook (only used if RUN_INFERENCE=True)
try:
    import scripts.inference as _inference_mod
    _HAS_INFER = True
except Exception:
    _inference_mod = None
    _HAS_INFER = False


# Helpers

def _count_infer_files(infer_dir: Path) -> int:
    return len(list(Path(infer_dir).glob("*_inference.pth")))

def _count_label_files(store: Path) -> int:
    """
    Count how many labeled frames exist, whether labels are stored in a directory
    (as *_clusters.npz) or inside a ZIP file.
    """
    store = Path(store)
    if store.is_file() and store.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(store, "r") as zf:
                return sum(1 for n in zf.namelist() if n.endswith("_clusters.npz"))
        except Exception:
            return 0
    return len(list(store.glob("*_clusters.npz")))

def _print_paths():
    labels_store_str = str(ZIP_PATH) if (USE_ZIP and not SAVE_PER_FILE) else (
        f"{LABELS_DIR} (+zip {ZIP_PATH})" if (USE_ZIP and SAVE_PER_FILE) else str(LABELS_DIR)
    )
    print("\n[Segmentation]")
    print(f" [paths] INFER_DIR   = {INFER_OUT_DIR}")
    print(f" [paths] OUT_DIR     = {OUT_DIR}")
    print(f" [paths] LABEL_STORE = {labels_store_str}")
    print(f" [conf ] K={K} | DEVICE={DEVICE}")
    print(f" [conf ] RUN_INFERENCE={RUN_INFERENCE} | DO_FIT={DO_FIT} | DO_APPLY={DO_APPLY} | RESUME={RESUME}")
    print(f" [conf ] SAVE_PER_FILE={SAVE_PER_FILE} | USE_ZIP={USE_ZIP} (mode={ZIP_MODE}, skip_existing={ZIP_SKIP_EXISTING})")

def _ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_PER_FILE:
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
    if USE_ZIP:
        ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)

def _maybe_run_inference():
    if not RUN_INFERENCE:
        return
    if not _HAS_INFER or not hasattr(_inference_mod, "run_inference"):
        sys.exit("[runner] RUN_INFERENCE=True but inference.run_inference() not importable in this env.")
    print("[runner] Running inference…")
    t0 = time.time()
    _inference_mod.run_inference(
        data_dir=DATA_DIR,
        checkpoint_path=CKPT_PATH,
        out_dir=INFER_OUT_DIR,
        voxel_size=VOXEL_SIZE,
        multiscale_voxel=False,
        batch_size=INFERENCE_BATCH,
        workers=INFERENCE_WORKERS,
        device=DEVICE,
        feat_mode=FEAT_MODE,
    )
    print(f"[runner] Inference done in {time.time()-t0:.1f}s")

def _fit_prototypes() -> torch.Tensor:
    print("[fit] Learning prototypes…")
    t0 = time.time()
    centroids = learn_prototypes_from_dataset(
        INFER_OUT_DIR,
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
        prefetch_buffer=PREFETCH_BUFFER,
        use_fp16=USE_FP16_MATMUL,
    )
    torch.save({"centroids": centroids}, PROTOS_PATH)
    save_run_config(RUN_CFG_JSON, {
        "K": K,
        "feature_cfg": FEATURE_CFG,
        "smooth_iters": SMOOTH_ITERS,
        "neighbor_range": NEIGHBOR_RANGE,
        "min_component": MIN_COMPONENT,
        "save_per_file": SAVE_PER_FILE,
        "use_zip": USE_ZIP,
    })
    print(f"[fit] Saved prototypes -> {PROTOS_PATH}  ({time.time()-t0:.1f}s)")
    return centroids

def _apply_segmentation(centroids: torch.Tensor):
    print("[apply] Segmenting dataset… (resume:", RESUME, ")")
    t0 = time.time()

    # Choose the label store used for resume checks
    labels_store = ZIP_PATH if (USE_ZIP and not SAVE_PER_FILE) else LABELS_DIR
    source = (resume_stream(INFER_OUT_DIR, labels_store) if RESUME else INFER_OUT_DIR)

    results, accum = segment_dataset(
        source,
        centroids=centroids,
        smoothing_iters=SMOOTH_ITERS,
        neighbor_range=NEIGHBOR_RANGE,
        min_component=MIN_COMPONENT,
        per_frame_hook=None,
        collect_metrics=True,
        device=DEVICE,
        assign_chunk=ASSIGN_CHUNK,
        feature_cfg=FEATURE_CFG,
        save_labels_dir=(LABELS_DIR if SAVE_PER_FILE else None),
        prefetch_buffer=PREFETCH_BUFFER,
        use_fp16=USE_FP16_ASSIGN,
        # --- ZIP persistence (new) ---
        zip_labels_path=(ZIP_PATH if USE_ZIP else None),
        zip_mode=ZIP_MODE,
        zip_compress=zipfile.ZIP_DEFLATED,
        zip_skip_existing=ZIP_SKIP_EXISTING,
    )
    total_now = _count_label_files(labels_store)
    print(f"[apply] Done in {time.time()-t0:.1f}s; wrote {len(results)} frames (total now: {total_now})")
    return accum

def _compute_metrics(accum):
    print("[metrics] Computing and writing CSV…")
    out_csv = OUT_DIR / f"metrics_k{K}.csv"
    evaluate_accumulated_metrics(
        accum,
        out_csv=out_csv,
        sample_n=METRICS_SAMPLE_N,
        seed=42,
        q_bins=4,
        tau_list=[0.2, 0.4, 0.6],
        tau_policy="quantile",
    )
    print(f"[metrics] Wrote -> {out_csv}")

def main():
    _print_paths()
    _ensure_dirs()

    # 0) (Optional) Inference
    _maybe_run_inference()

    # 1) Sanity on inference coverage
    n_inf = _count_infer_files(INFER_OUT_DIR)
    if n_inf == 0:
        sys.exit(f"[runner] No *_inference.pth in {INFER_OUT_DIR}. Did you run inference to this folder?")
    print(f"[runner] Found {n_inf} inference dumps.")

    # 2) Fit or load prototypes
    if DO_FIT:
        centroids = _fit_prototypes()
    else:
        if not PROTOS_PATH.exists():
            sys.exit(f"[runner] DO_FIT=False but prototypes not found at {PROTOS_PATH}")
        centroids = torch.load(PROTOS_PATH, map_location="cpu")["centroids"]
        print(f"[fit] Loaded prototypes from {PROTOS_PATH}")

    # 3) Apply segmentation (resume-safe) + collect metrics
    if DO_APPLY:
        accum = _apply_segmentation(centroids)
        # 4) Metrics
        _compute_metrics(accum)

    # 5) Summary
    labels_store = ZIP_PATH if (USE_ZIP and not SAVE_PER_FILE) else LABELS_DIR
    print("\n[summary]")
    print(f"  Prototypes: {PROTOS_PATH if PROTOS_PATH.exists() else '(not saved)'}")
    print(f"  Labels    : {labels_store}  (count={_count_label_files(labels_store)})")
    print(f"  Metrics   : {OUT_DIR / f'metrics_k{K}.csv'}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
