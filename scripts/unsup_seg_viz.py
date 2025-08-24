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

# ----- Segmentation knobs (unchanged for fit/apply modes) -----
SMOOTH_ITERS = 1
NEIGHBOR_RANGE = 1
MIN_COMPONENT = 120

# ----- Viz / I-O -----
PLY_LIMIT = None
DO_OPEN3D_VIEW = False  # not used in view_o3d mode
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

# ----- Speed knobs -----
USE_GPU = torch.cuda.is_available()
PROTOTYPE_DEVICE = "cuda" if USE_GPU else "cpu"
SEGMENT_DEVICE   = "cuda" if USE_GPU else "cpu"
PROTOTYPE_CHUNK  = 1_000_000
ASSIGN_CHUNK     = 3_000_000

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

# ---------- Open3D (optional) ----------
try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False


def _make_palette(num_classes: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pal = (rng.uniform(60, 255, size=(max(1, num_classes), 3))).astype(np.uint8)
    return pal


def _labels_to_colors(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return (palette[labels % palette.shape[0]].astype(np.float32) / 255.0)


def _o3d_show_and_snapshot(xyz: np.ndarray, rgb: np.ndarray, title: str, png_path: Optional[Path]):
    if not _HAS_O3D:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if rgb is not None and rgb.size:
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    if SAVE_PLY or SAVE_PNG:
        # used by "view_labels" mode only
        if SAVE_PLY:
            pass  # handled elsewhere
        if SAVE_PNG and png_path is not None:
            try:
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False, width=PNG_W, height=PNG_H, window_name=title)
                vis.add_geometry(pcd)
                vis.get_render_option().point_size = 2.0
                vis.update_renderer()
                ctr = vis.get_view_control(); ctr.rotate(0.0, 0.0)
                vis.poll_events(); vis.update_renderer()
                vis.capture_screen_image(str(png_path), do_render=True)
                vis.destroy_window()
            except Exception:
                pass


def _save_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    rgb8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    header = "\n".join([
        "ply","format ascii 1.0",f"element vertex {xyz.shape[0]}",
        "property float x","property float y","property float z",
        "property uchar red","property uchar green","property uchar blue","end_header"
    ])
    with open(path, "w") as f:
        f.write(header + "\n")
        for i in range(xyz.shape[0]):
            r,g,b = int(rgb8[i,0]), int(rgb8[i,1]), int(rgb8[i,2])
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} {r} {g} {b}\n")


def _export_labels_to_ply(infer_dir: Path, labels_dir: Path, out_dir: Path, palette: np.ndarray):
    """Existing non-interactive renderer (kept for 'view_labels' mode)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = out_dir / "ply"; png_dir = out_dir / "png"
    if SAVE_PLY: ply_dir.mkdir(exist_ok=True)
    if SAVE_PNG: png_dir.mkdir(exist_ok=True)

    dumps = {p.stem.replace("_inference", ""): p for p in sorted(infer_dir.glob("*_inference.pth"))}
    for npz in sorted(labels_dir.glob("*_clusters.npz")):
        stem = npz.stem.replace("_clusters", "")
        lab = np.load(npz)["labels"].astype(np.int64)
        dump = dumps.get(stem, None)
        if dump is None:
            print(f"[view_labels] missing dump for {stem}, skipping.")
            continue
        coord = torch.load(dump, map_location="cpu")["coord_raw"].numpy().astype(np.float32)
        cols = _labels_to_colors(lab, palette)
        if SAVE_PLY:
            _save_ply(ply_dir / f"{stem}.ply", coord, cols)
        if _HAS_O3D and SAVE_PNG:
            png_path = (png_dir / f"{stem}.png")
            _o3d_show_and_snapshot(coord, cols, title=f"Seg: {stem}", png_path=png_path)


# --------------------- NEW: interactive Open3D viewer -------------------- #

def _view_labels_o3d(infer_dir: Path, labels_dir: Path, palette: np.ndarray, win_w=1600, win_h=1200):
    if not _HAS_O3D:
        print("Open3D is not installed on this machine.")
        return

    # Build stem -> paths maps and intersect
    dump_map = {p.stem.replace("_inference", ""): p for p in sorted(infer_dir.glob("*_inference.pth"))}
    label_map = {p.stem.replace("_clusters", ""): p for p in sorted(labels_dir.glob("*_clusters.npz"))}
    stems = sorted(set(dump_map.keys()).intersection(label_map.keys()))
    if not stems:
        print("No overlapping frames found between inference dumps and labels dir.")
        return

    # Visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Unsupervised Segmentation Viewer", width=win_w, height=win_h)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    ro = vis.get_render_option()
    ro.point_size = 2.0

    idx = {"i": 0}

    def load_frame(i: int):
        stem = stems[i]
        coord = torch.load(dump_map[stem], map_location="cpu")["coord_raw"].numpy().astype(np.float32)
        lab   = np.load(label_map[stem])["labels"].astype(np.int64)
        cols  = _labels_to_colors(lab, palette)
        pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        vis.update_geometry(pcd)
        vis.get_view_control().set_zoom(0.7)
        vis.update_renderer()
        vis.poll_events()
        vis.get_render_option().point_size = ro.point_size
        print(f"[view_o3d] frame {i+1}/{len(stems)}: {stem}  (N={coord.shape[0]})")

    def next_frame(visu):
        idx["i"] = (idx["i"] + 1) % len(stems)
        load_frame(idx["i"])
        return False

    def prev_frame(visu):
        idx["i"] = (idx["i"] - 1) % len(stems)
        load_frame(idx["i"])
        return False

    def inc_point_size(visu):
        ro.point_size = min(ro.point_size + 1.0, 10.0)
        vis.get_render_option().point_size = ro.point_size
        vis.update_renderer()
        return False

    def dec_point_size(visu):
        ro.point_size = max(ro.point_size - 1.0, 1.0)
        vis.get_render_option().point_size = ro.point_size
        vis.update_renderer()
        return False

    def reset_view(visu):
        vis.get_view_control().reset_view_point(True)
        return False

    # Key bindings
    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("P"), prev_frame)
    vis.register_key_callback(ord("]"), inc_point_size)
    vis.register_key_callback(ord("["), dec_point_size)
    vis.register_key_callback(ord("R"), reset_view)
    vis.register_key_callback(ord("Q"), lambda v: True)  # close on Q

    load_frame(idx["i"])
    vis.run()
    vis.destroy_window()


# ------------------------------ main ------------------------------------- #

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    palette = _make_palette(K)

    if MODE == "view_labels":
        print("[mode=view_labels] Rendering saved label files to PLY/PNG…")
        _export_labels_to_ply(INFER_DIR, LABELS_DIR, OUT_DIR, palette)
        return

    if MODE == "view_o3d":
        print("[mode=view_o3d] Interactive Open3D viewer (no file writes).")
        _view_labels_o3d(INFER_DIR, LABELS_DIR, palette, win_w=PNG_W, win_h=PNG_H)
        return

    # ---- fit/apply modes unchanged below ----
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
        )
        torch.save({"centroids": centroids}, OUT_DIR / "prototypes.pt")
        print("[fit] Prototypes saved:", OUT_DIR / "prototypes.pt")

        save_run_config(RUN_CFG_JSON, {
            "K": K,
            "feature_cfg": FEATURE_CFG,
            "dist_edges": DIST_EDGES,
            "dist_ratios": DIST_RATIOS,
            "soft_assign": SOFT_ASSIGN,
            "tau": TAU,
            "conf_th": CONF_TH,
            "vis_aware_smooth": VIS_AWARE_SMOOTH,
            "vis_vote_mode": VIS_VOTE_MODE,
            "smooth_iters": SMOOTH_ITERS,
            "neighbor_range": NEIGHBOR_RANGE,
            "min_component": MIN_COMPONENT,
            "bin_neighbor_ranges": BIN_NEIGHBOR_RANGES,
            "bin_min_components": BIN_MIN_COMPONENTS,
            "outlier_prune": OUTLIER_PRUNE,
            "outlier_frac": OUTLIER_FRAC,
        })
        if MODE == "fit_only":
            print("[fit_only] Done.")
            return

    if MODE in ("apply_only", "fit_and_apply"):
        if MODE == "apply_only":
            print("[apply_only] Loading prototypes + run config…")
            centroids = torch.load(LOAD_PROTOTYPES_PATH, map_location="cpu")["centroids"]
            if RUN_CFG_JSON.exists():
                cfg = load_run_config(RUN_CFG_JSON)
                globals().update({
                    "K": cfg.get("K", K),
                    "FEATURE_CFG": cfg.get("feature_cfg", FEATURE_CFG),
                    "DIST_EDGES": cfg.get("dist_edges", DIST_EDGES),
                    "DIST_RATIOS": cfg.get("dist_ratios", DIST_RATIOS),
                    "SOFT_ASSIGN": cfg.get("soft_assign", SOFT_ASSIGN),
                    "TAU": cfg.get("tau", TAU),
                    "CONF_TH": cfg.get("conf_th", CONF_TH),
                    "VIS_AWARE_SMOOTH": cfg.get("vis_aware_smooth", VIS_AWARE_SMOOTH),
                    "VIS_VOTE_MODE": cfg.get("vis_vote_mode", VIS_VOTE_MODE),
                    "SMOOTH_ITERS": cfg.get("smooth_iters", SMOOTH_ITERS),
                    "NEIGHBOR_RANGE": cfg.get("neighbor_range", NEIGHBOR_RANGE),
                    "MIN_COMPONENT": cfg.get("min_component", MIN_COMPONENT),
                    "BIN_NEIGHBOR_RANGES": cfg.get("bin_neighbor_ranges", BIN_NEIGHBOR_RANGES),
                    "BIN_MIN_COMPONENTS": cfg.get("bin_min_components", BIN_MIN_COMPONENTS),
                    "OUTLIER_PRUNE": cfg.get("outlier_prune", OUTLIER_PRUNE),
                    "OUTLIER_FRAC": cfg.get("outlier_frac", OUTLIER_FRAC),
                })

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

        # Optional: also render PLY/PNG from labels
        # _export_labels_to_ply(INFER_DIR, LABELS_DIR, OUT_DIR, palette)

    print(f"[done] Artifacts under: {OUT_DIR}")
    print(f"      - labels dir: {LABELS_DIR}")
    print("      - prototypes.pt, run_config.json (if fitting)")
    

if __name__ == "__main__":
    main()