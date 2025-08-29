from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import torch
import io
import zipfile

# ==============================
# EDIT THESE CONSTANTS
# ==============================
INFER_DIR  = Path("data/28082025_1348_segcamel_train_output/28082025_2002_inference_output")
OUT_DIR    = Path("data/28082025_1348_segcamel_train_output/28082025_2002_unsup_outputs")
K          = 10  # used for palette sizing only
LABELS_DIR = OUT_DIR / f"labels_k{K}"
LABELS_ZIP = OUT_DIR / f"labels_k{K}.zip"   # viewer can read zipped labels too
PREFER_ZIP = True                           # prefer reading labels from ZIP if it exists

#   "labels" -> export saved labels to PLY and/or PNG (no interactive window)
#   "o3d"    -> interactive Open3D viewer
VIEW_MODE = "labels"  # "labels" or "o3d"

# Exports (only used in VIEW_MODE="labels")
SAVE_PLY = True
SAVE_PNG = False
PLY_LIMIT: Optional[int] = None  # set to an int to cap number of exported frames
PNG_W, PNG_H = 1600, 1200

# Interactive Open3D toggles (used in both modes where relevant)
DO_OPEN3D_VIEW = False  # if True in "labels" mode, opens window instead of headless snapshot

# ==============================
# Implementation
# ==============================

# Optional: Open3D (soft dependency)
try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False


def _make_palette(num_classes: int, seed: int = 0) -> np.ndarray:
    num_classes = max(1, num_classes)
    rng = np.random.default_rng(seed)
    return (rng.uniform(60, 255, size=(num_classes, 3))).astype(np.uint8)


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
            pass


def _save_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
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


# ---------------- ZIP/Dir label store helpers ---------------- #

def _labels_store_exists() -> Tuple[bool, str]:
    zip_ok = LABELS_ZIP.exists()
    dir_ok = LABELS_DIR.exists()
    if PREFER_ZIP and zip_ok:
        return True, "zip"
    if dir_ok:
        return True, "dir"
    if zip_ok:
        return True, "zip"
    return False, "none"

def _zip_list_labels(zip_path: Path) -> Dict[str, str]:
    mapping = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for n in zf.namelist():
            if n.endswith("_clusters.npz"):
                stem = Path(n).stem.replace("_clusters", "")
                mapping[stem] = n
    return mapping

def _zip_load_labels(zip_path: Path, name_in_zip: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(name_in_zip, "r") as fh:
            with np.load(fh) as data:
                return data["labels"].astype(np.int64)

# ---------------- Dump map (robust) ---------------- #

def _build_dump_maps(infer_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    dumps = sorted(list(infer_dir.glob("*_inference.pth"))[:10])
    by_filename = {p.stem.replace("_inference", ""): p for p in dumps}
    by_payload = {}
    for p in dumps:
        try:
            payload = torch.load(p, map_location="cpu")
            s = payload.get("image_stem", None)
            if isinstance(s, str):
                by_payload[s] = p
        except Exception:
            pass
    return by_filename, by_payload

def _resolve_dump_path(label_stem: str, by_filename: Dict[str, Path], by_payload: Dict[str, Path]) -> Optional[Path]:
    if label_stem in by_filename:
        return by_filename[label_stem]
    if label_stem in by_payload:
        return by_payload[label_stem]
    return None


# -------- Camera reset (version-agnostic) -------- #

def _reset_camera(vis, pcd, zoom=0.7):
    """
    Robust camera reset that works across Open3D versions:
    1) Try Visualizer.reset_view_point(True) if available.
    2) Otherwise compute bbox and set lookat/front/up/zoom manually.
    """
    # Try Visualizer API (exists in many builds)
    try:
        if hasattr(vis, "reset_view_point"):
            vis.reset_view_point(True)
            ctr = vis.get_view_control()
            ctr.set_zoom(zoom)
            return
    except Exception:
        pass

    # Fallback: manual fit using bbox
    try:
        bbox = pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = np.linalg.norm(bbox.get_extent()) + 1e-6

        ctr = vis.get_view_control()
        # Reasonable defaults for LiDAR-like clouds
        ctr.set_lookat(center.tolist())
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_up([0.0, -1.0, 0.0])
        # Zoom ~ how tight the fit is; 0.35–0.8 usually fine
        ctr.set_zoom(zoom)
    except Exception:
        # last resort: just try to update renderer
        pass


# ---------------- Export: labels -> PLY/PNG ---------------- #

def _export_labels_to_ply_png(infer_dir: Path, out_dir: Path, palette: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = out_dir / "ply"
    png_dir = out_dir / "png"
    if SAVE_PLY:
        ply_dir.mkdir(exist_ok=True)
    if SAVE_PNG:
        png_dir.mkdir(exist_ok=True)

    by_fn, by_pl = _build_dump_maps(infer_dir)

    exists, mode = _labels_store_exists()
    if not exists:
        raise FileNotFoundError(f"No labels store found. Looked for dir={LABELS_DIR} and zip={LABELS_ZIP}")

    if mode == "zip":
        label_map = _zip_list_labels(LABELS_ZIP)  # stem -> name_in_zip
        load_labels = lambda stem: _zip_load_labels(LABELS_ZIP, label_map[stem])
    else:
        label_map = {p.stem.replace("_clusters", ""): p for p in sorted(LABELS_DIR.glob("*_clusters.npz"))}
        def load_labels(stem: str) -> np.ndarray:
            return np.load(label_map[stem])["labels"].astype(np.int64)

    exported = 0
    for stem in sorted(label_map.keys()):
        dump = _resolve_dump_path(stem, by_fn, by_pl)
        if dump is None:
            print(f"[export] missing dump for stem={stem}, skipping.")
            continue

        try:
            lab = load_labels(stem)
            coord = torch.load(dump, map_location="cpu")["coord_raw"].numpy().astype(np.float32)
        except Exception as e:
            print(f"[export] error loading {stem}: {e}")
            continue

        cols = _labels_to_colors(lab, palette)

        if SAVE_PLY:
            _save_ply(ply_dir / f"{stem}.ply", coord, cols)
        if _HAS_O3D and (DO_OPEN3D_VIEW or SAVE_PNG):
            png_path = (png_dir / f"{stem}.png") if SAVE_PNG else None
            _o3d_show_and_snapshot(coord, cols, title=f"Seg: {stem}", png_path=png_path)

        exported += 1
        if PLY_LIMIT is not None and exported >= int(PLY_LIMIT):
            break

    print(f"[export] done. exported={exported}, PLY={SAVE_PLY}, PNG={SAVE_PNG} (store={mode})")


# ---------------- Interactive Open3D viewer ---------------- #

def _view_labels_o3d(infer_dir: Path, palette: np.ndarray, win_w=1600, win_h=1200):
    if not _HAS_O3D:
        print("Open3D is not installed.")
        return

    by_fn, by_pl = _build_dump_maps(infer_dir)

    exists, mode = _labels_store_exists()
    if not exists:
        print(f"No labels found. Expected dir: {LABELS_DIR} or zip: {LABELS_ZIP}")
        return

    if mode == "zip":
        label_map = _zip_list_labels(LABELS_ZIP)  # stem -> name_in_zip
        load_labels = lambda stem: _zip_load_labels(LABELS_ZIP, label_map[stem])
        label_stems = sorted(label_map.keys())
    else:
        label_map = {p.stem.replace("_clusters", ""): p for p in sorted(LABELS_DIR.glob("*_clusters.npz"))}
        def load_labels(stem: str) -> np.ndarray:
            return np.load(label_map[stem])["labels"].astype(np.int64)
        label_stems = sorted(label_map.keys())

    stems = [s for s in label_stems if _resolve_dump_path(s, by_fn, by_pl) is not None]
    if not stems:
        print("No overlapping frames found between inference dumps and labels (check naming).")
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
        dump_path = _resolve_dump_path(stem, by_fn, by_pl)
        coord = torch.load(dump_path, map_location="cpu")["coord_raw"].numpy().astype(np.float32)
        lab   = load_labels(stem)
        cols  = _labels_to_colors(lab, palette)

        pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
        if cols.size:
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        vis.update_geometry(pcd)

        # Robust camera reset (no more blank window)
        _reset_camera(vis, pcd, zoom=0.7)

        vis.update_renderer()
        vis.poll_events()
        vis.get_render_option().point_size = ro.point_size
        print(f"[o3d] frame {i+1}/{len(stems)}: {stem} (N={coord.shape[0]})")

    def next_frame(_): idx["i"] = (idx["i"] + 1) % len(stems); load_frame(idx["i"]); return False
    def prev_frame(_): idx["i"] = (idx["i"] - 1) % len(stems); load_frame(idx["i"]); return False
    def inc_ps(_): ro.point_size = min(ro.point_size + 1.0, 10.0); vis.get_render_option().point_size = ro.point_size; vis.update_renderer(); return False
    def dec_ps(_): ro.point_size = max(ro.point_size - 1.0, 1.0);  vis.get_render_option().point_size = ro.point_size; vis.update_renderer(); return False
    def reset_v(_): _reset_camera(vis, pcd, zoom=0.7); return False  # patched
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


# ---------------- Sanity + main ---------------- #

def _count_infer() -> int:
    return len(list(INFER_DIR.glob("*_inference.pth")))

def _count_labels() -> Tuple[int, str]:
    if PREFER_ZIP and LABELS_ZIP.exists():
        try:
            with zipfile.ZipFile(LABELS_ZIP, "r") as zf:
                return sum(1 for n in zf.namelist() if n.endswith("_clusters.npz")), "zip"
        except Exception:
            return 0, "zip"
    if LABELS_DIR.exists():
        return len(list(LABELS_DIR.glob("*_clusters.npz"))), "dir"
    if LABELS_ZIP.exists():
        try:
            with zipfile.ZipFile(LABELS_ZIP, "r") as zf:
                return sum(1 for n in zf.namelist() if n.endswith("_clusters.npz")), "zip"
        except Exception:
            return 0, "zip"
    return 0, "none"

def _sanity_print():
    n_inf = _count_infer()
    n_lbl, store = _count_labels()
    print(f"[paths] INFER_DIR = {INFER_DIR}  ({n_inf} dumps)")
    if store == "dir":
        print(f"[paths] LABELS_DIR = {LABELS_DIR} ({n_lbl} label files)")
    elif store == "zip":
        print(f"[paths] LABELS_ZIP = {LABELS_ZIP} ({n_lbl} label entries)")
    else:
        print(f"[paths] No labels found (checked {LABELS_DIR} and {LABELS_ZIP})")
    print(f"[mode ] VIEW_MODE = {VIEW_MODE} | PLY={SAVE_PLY} PNG={SAVE_PNG} O3D={_HAS_O3D} | store={store}")

def main():
    if not INFER_DIR.exists():
        raise FileNotFoundError(f"INFER_DIR not found: {INFER_DIR}")

    _sanity_print()

    palette = _make_palette(K)

    if VIEW_MODE == "labels":
        _export_labels_to_ply_png(INFER_DIR, OUT_DIR, palette)
        return
    if VIEW_MODE == "o3d":
        _view_labels_o3d(INFER_DIR, palette, win_w=PNG_W, win_h=PNG_H)
        return

    raise ValueError(f"Unknown VIEW_MODE={VIEW_MODE} (use 'labels' or 'o3d').")


if __name__ == "__main__":
    main()
