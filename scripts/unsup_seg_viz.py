# scripts/one_shot_unsup_seg_viz.py
# One-shot unsupervised segmentation + visualization over inference dumps.
# - Consumes:  .../inference_out/*_inference.pth  (produced by scripts/inference.py)
# - Learns K spherical prototypes (cosine) on 64-D features.
# - Segments every frame (nearest prototype + voxel smoothing).
# - Exports colorized PLYs.
# - Optionally visualizes in Open3D and saves PNG snapshots (best-effort).
#
# Usage:
#   1) Set the constants below (INFER_DIR, OUT_DIR, ...)
#   2) python -m scripts.one_shot_unsup_seg_viz
#
# Notes:
#   - Coordinates in dumps are the normalized coords used by the model.
#     If you need raw-scale XYZ, save and pass them during inference.

from pathlib import Path
from typing import Optional, Dict, Iterable
import numpy as np
import torch

# ---- Adjust these constants ----
INFER_DIR = Path("data/14082025_0250_segcamel_train_with_vel_md1_ld1_sd1/inference_output")   # folder containing *_inference.pth
OUT_DIR   = Path("data/14082025_0250_segcamel_train_with_vel_md1_ld1_sd1/unsup_outputs")          # where to save prototypes, labels, PLYs, PNGs
K = 20                                       # number of clusters (try 10–40 for highway scenes)
SMOOTH_ITERS = 2                             # 0=off, 1–2 recommended
NEIGHBOR_RANGE = 1                           # 3x3x3 voxel neighborhood
MIN_COMPONENT = 150                           # snap tiny blobs to neighbor mode
PLY_LIMIT = None                             # set to an int to only export first N PLYs; None = all
DO_OPEN3D_VIEW = True                        # interactive viewer (if available)
SAVE_PNG = True                              # save PNG snapshot with Open3D (if available)
PNG_W = 1600                                 # snapshot width
PNG_H = 1200                                 # snapshot height
# --------------------------------

# Import pipeline pieces (no redundancy)
from scripts.unsup_seg_pipeline import (
    extract_features,
    learn_prototypes_from_dataset,
    segment_dataset,
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
    if num_classes > 0:
        pal[0] = np.array([200, 200, 200], dtype=np.uint8)
    return pal


def _labels_to_colors(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    cols = palette[labels % palette.shape[0]].astype(np.float32) / 255.0
    return cols


def _o3d_show_and_snapshot(xyz: np.ndarray, rgb: np.ndarray, title: str, png_path: Optional[Path]):
    """
    Best-effort Open3D visualization. Shows interactive window if DO_OPEN3D_VIEW is True.
    Saves a PNG snapshot to png_path when requested; if running headless, snapshot may fail
    depending on Open3D build. That's OK — we keep the rest of the pipeline working.
    """
    if not _HAS_O3D:
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if rgb is not None and rgb.size:
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))

    # Interactive viewer (optional)
    if DO_OPEN3D_VIEW:
        try:
            o3d.visualization.draw_geometries([pcd], window_name=title)
        except Exception:
            pass  # ignore UI errors on headless nodes

    # Snapshot (optional)
    if SAVE_PNG and png_path is not None:
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=PNG_W, height=PNG_H, window_name=title)
            vis.add_geometry(pcd)
            vis.get_render_option().point_size = 2.0
            vis.update_renderer()
            # Try a reasonable viewpoint:
            ctr = vis.get_view_control()
            ctr.rotate(0.0, 0.0)  # noop but forces internal update
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(str(png_path), do_render=True)
            vis.destroy_window()
        except Exception:
            # Offscreen capture may fail on some builds; just skip
            pass


def _save_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    """
    Save colored PLY (ASCII). xyz: (N,3) float32, rgb: (N,3) in [0,1].
    """
    rgb8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {xyz.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ])
    with open(path, "w") as f:
        f.write(header + "\n")
        for i in range(xyz.shape[0]):
            r, g, b = int(rgb8[i, 0]), int(rgb8[i, 1]), int(rgb8[i, 2])
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} {r} {g} {b}\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ply_dir = OUT_DIR / "ply"
    png_dir = OUT_DIR / "png"
    (ply_dir).mkdir(exist_ok=True)
    if SAVE_PNG:
        png_dir.mkdir(exist_ok=True)

    # 1) Learn prototypes from saved feature dumps
    print("[one-shot] Learning prototypes...")
    centroids = learn_prototypes_from_dataset(
        INFER_DIR,
        k=K,
        max_passes=2,
        sample_per_frame=20000,
        seed=0,
    )
    torch.save({"centroids": centroids}, OUT_DIR / "prototypes.pt")
    print("[one-shot] Prototypes saved:", OUT_DIR / "prototypes.pt")

    # 2) Segment all frames
    print("[one-shot] Segmenting dataset...")
    results = segment_dataset(
        INFER_DIR,
        centroids=centroids,
        smoothing_iters=SMOOTH_ITERS,
        neighbor_range=NEIGHBOR_RANGE,
        min_component=MIN_COMPONENT,
    )
    # Save label arrays for reuse
    #for stem, labels in results.items():
    #    np.save(OUT_DIR / f"{stem}_labels.npy", labels)

    # 3) Export PLYs (+ Open3D snapshot/visualize)
    print("[one-shot] Exporting PLYs...")
    palette = _make_palette(int(max((lab.max() if lab.size else -1) for lab in results.values()) + 1))
    exported = 0

    # Re-stream features to get coords in the same order we produced labels
    for item in extract_features(INFER_DIR, return_iter=True):
        stem = item["file_stem"]
        if stem not in results:
            continue
        labels = results[stem]
        coord = item["coord_raw"].cpu().numpy().astype(np.float32)  # NOTE: raw coords
        colors = _labels_to_colors(labels, palette)

        ply_path = ply_dir / f"{stem}.ply"
        #_save_ply(ply_path, coord, colors)

        if _HAS_O3D and (DO_OPEN3D_VIEW or SAVE_PNG):
            png_path = (png_dir / f"{stem}.png") if SAVE_PNG else None
            _o3d_show_and_snapshot(coord, colors, title=f"Seg: {stem}", png_path=png_path)

        exported += 1
        if PLY_LIMIT is not None and exported >= int(PLY_LIMIT):
            break
        
        
    print(f"[one-shot] Done. Prototypes + labels + PLYs saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()
