from pathlib import Path
from typing import Optional
import numpy as np
import torch


INFER_DIR  = Path("/path/to/inference_output")
OUT_DIR    = Path("/path/to/unsup_outputs")
K          = 10  # used for palette sizing only
LABELS_DIR = OUT_DIR / f"labels_k{K}"

#   "labels" -> export saved labels to PLY and/or PNG (no interactive window)
#   "o3d"    -> interactive Open3D viewer
VIEW_MODE = "o3d"  # "labels" or "o3d"

# Exports (only used in VIEW_MODE="labels")
SAVE_PLY = False
SAVE_PNG = False
PLY_LIMIT: Optional[int] = None  # set to an int to cap number of exported frames
PNG_W, PNG_H = 1600, 1200

# Interactive Open3D toggles (used in both modes where relevant)
DO_OPEN3D_VIEW = True  # if True in "labels" mode, opens window instead of headless snapshot

# Implementation
try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False


def _make_palette(num_classes: int, seed: int = 0) -> np.ndarray:
    """Generate a simple, bright color palette for clusters."""
    num_classes = max(1, num_classes)
    rng = np.random.default_rng(seed)
    return (rng.uniform(60, 255, size=(num_classes, 3))).astype(np.uint8)


def _labels_to_colors(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return (palette[labels % palette.shape[0]].astype(np.float32) / 255.0)


def _o3d_show_and_snapshot(xyz: np.ndarray, rgb: np.ndarray, title: str, png_path: Optional[Path]):
    """Headless snapshot (PNG) and/or interactive view via Open3D."""
    if not _HAS_O3D:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if rgb is not None and rgb.size:
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))

    if DO_OPEN3D_VIEW:
        o3d.visualization.draw_geometries([pcd], window_name=title)
        return

    if png_path is not None:
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=PNG_W, height=PNG_H, window_name=title)
            vis.add_geometry(pcd)
            vis.get_render_option().point_size = 2.0
            vis.update_renderer()
            vis.capture_screen_image(str(png_path), do_render=True)
            vis.destroy_window()
        except Exception:
            pass  # snapshot optional; don't crash the script


def _save_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    """Minimal ASCII PLY writer (fast enough, avoids external deps)."""
    rgb8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    header = "\n".join([
        "ply", "format ascii 1.0", f"element vertex {xyz.shape[0]}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header"
    ])
    with open(path, "w") as f:
        f.write(header + "\n")
        for i in range(xyz.shape[0]):
            r, g, b = int(rgb8[i, 0]), int(rgb8[i, 1]), int(rgb8[i, 2])
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} {r} {g} {b}\n")


def _export_labels_to_ply_png(infer_dir: Path, labels_dir: Path, out_dir: Path, palette: np.ndarray):
    """Export each saved labels npz to PLY and/or PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = out_dir / "ply"
    png_dir = out_dir / "png"
    if SAVE_PLY:
        ply_dir.mkdir(exist_ok=True)
    if SAVE_PNG:
        png_dir.mkdir(exist_ok=True)

    # Map stem -> inference dump path
    dumps = {p.stem.replace("_inference", ""): p for p in sorted(infer_dir.glob("*_inference.pth"))}

    exported = 0
    for npz in sorted(labels_dir.glob("*_clusters.npz")):
        stem = npz.stem.replace("_clusters", "")
        dump = dumps.get(stem, None)
        if dump is None:
            print(f"[export] missing dump for {stem}, skipping.")
            continue

        # Load point coords and labels
        try:
            lab = np.load(npz)["labels"].astype(np.int64)
            coord = torch.load(dump, map_location="cpu")["coord_raw"].numpy().astype(np.float32)
        except Exception as e:
            print(f"[export] error loading {stem}: {e}")
            continue

        # Colorize
        cols = _labels_to_colors(lab, palette)

        # Save artifacts
        if SAVE_PLY:
            _save_ply(ply_dir / f"{stem}.ply", coord, cols)
        if _HAS_O3D and (DO_OPEN3D_VIEW or SAVE_PNG):
            png_path = (png_dir / f"{stem}.png") if SAVE_PNG else None
            _o3d_show_and_snapshot(coord, cols, title=f"Seg: {stem}", png_path=png_path)

        exported += 1
        if PLY_LIMIT is not None and exported >= int(PLY_LIMIT):
            break

    print(f"[export] done. exported={exported}, PLY={SAVE_PLY}, PNG={SAVE_PNG}")


def _view_labels_o3d(infer_dir: Path, labels_dir: Path, palette: np.ndarray, win_w=1600, win_h=1200):
    """Interactive point cloud browser over saved segmentation labels."""
    if not _HAS_O3D:
        print("Open3D is not installed.")
        return

    dump_map = {p.stem.replace("_inference", ""): p for p in sorted(infer_dir.glob("*_inference.pth"))}
    label_map = {p.stem.replace("_clusters", ""): p for p in sorted(labels_dir.glob("*_clusters.npz"))}
    stems = sorted(set(dump_map.keys()).intersection(label_map.keys()))
    if not stems:
        print("No overlapping frames found between inference dumps and labels dir.")
        return

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
        print(f"[o3d] frame {i+1}/{len(stems)}: {stem} (N={coord.shape[0]})")

    def next_frame(_): idx["i"] = (idx["i"] + 1) % len(stems); load_frame(idx["i"]); return False
    def prev_frame(_): idx["i"] = (idx["i"] - 1) % len(stems); load_frame(idx["i"]); return False
    def inc_ps(_): ro.point_size = min(ro.point_size + 1.0, 10.0); vis.get_render_option().point_size = ro.point_size; vis.update_renderer(); return False
    def dec_ps(_): ro.point_size = max(ro.point_size - 1.0, 1.0);  vis.get_render_option().point_size = ro.point_size; vis.update_renderer(); return False
    def reset_v(_): vis.get_view_control().reset_view_point(True); return False
    def quit_v(_): return True

    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("P"), prev_frame)
    vis.register_key_callback(ord("]"), inc_ps)
    vis.register_key_callback(ord("["), dec_ps)
    vis.register_key_callback(ord("R"), reset_v)
    vis.register_key_callback(ord("Q"), quit_v)

    load_frame(idx["i"])
    vis.run()
    vis.destroy_window()


def _sanity_print():
    n_inf = len(list(INFER_DIR.glob("*_inference.pth")))
    n_lbl = len(list(LABELS_DIR.glob("*_clusters.npz")))
    print(f"[paths] INFER_DIR = {INFER_DIR}  ({n_inf} dumps)")
    print(f"[paths] LABELS_DIR = {LABELS_DIR} ({n_lbl} label files)")
    print(f"[mode ] VIEW_MODE = {VIEW_MODE} | PLY={SAVE_PLY} PNG={SAVE_PNG} O3D={_HAS_O3D}")


def main():
    # Basic checks
    if not INFER_DIR.exists():
        raise FileNotFoundError(f"INFER_DIR not found: {INFER_DIR}")
    if not LABELS_DIR.exists():
        raise FileNotFoundError(f"LABELS_DIR not found: {LABELS_DIR}")

    _sanity_print()

    # Build a palette. If K is off, we still work; color wraps by modulus.
    palette = _make_palette(K)

    if VIEW_MODE == "labels":
        _export_labels_to_ply_png(INFER_DIR, LABELS_DIR, OUT_DIR, palette)
        return
    if VIEW_MODE == "o3d":
        _view_labels_o3d(INFER_DIR, LABELS_DIR, palette, win_w=PNG_W, win_h=PNG_H)
        return

    raise ValueError(f"Unknown VIEW_MODE={VIEW_MODE} (use 'labels' or 'o3d').")


if __name__ == "__main__":
    main()
