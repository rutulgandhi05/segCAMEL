from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import torch
import io
import zipfile
try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False

# ==============================
# EDIT THESE CONSTANTS
# ==============================
INFER_DIR  = Path("data/09092025_0900_segcamel_train_output_epoch_50/09092025_1317_inference_output_parking_lot_02_Day")
OUT_DIR    = Path("data/09092025_0900_segcamel_train_output_epoch_50/09092025_1317_unsup_outputs_parking_lot_02_Day")
K          = 10  # used for palette sizing only
LABELS_DIR = OUT_DIR / f"labels_k{K}"
LABELS_ZIP = OUT_DIR / f"labels_k{K}.zip"   # viewer can read zipped labels too
PREFER_ZIP = True                           # prefer reading labels from ZIP if it exists

#   "labels" -> export saved labels to PLY and/or PNG (no interactive window)
#   "o3d"    -> interactive Open3D viewer
VIEW_MODE = "o3d"  # "labels" or "o3d"

# Exports (only used in VIEW_MODE="labels")
SAVE_PLY = False
SAVE_PNG = False
PLY_LIMIT: Optional[int] = None  # set to an int to cap number of exported frames
PNG_W, PNG_H = 1600, 1200

# Interactive Open3D toggles (used in both modes where relevant)
DO_OPEN3D_VIEW = False  # if True in "labels" mode, opens window instead of headless snapshot

# Visualization knobs
NOISE_LABEL = -1                  # must match your pipeline
FADE_NON_VISIBLE = True           # default: gray-out points outside image mask
FADE_COLOR = np.array([180, 180, 180], dtype=np.uint8)  # grey for non-visible
NOISE_COLOR = np.array([128, 128, 128], dtype=np.uint8) # grey for noise label

# ---------------- Palette & coloring ----------------

def _make_palette(k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(k, 3), dtype=np.uint8)
    # ensure good spread and avoid very dark colors
    base = np.maximum(base, 40)
    return base


def _labels_to_colors(
    labels: np.ndarray,
    palette: np.ndarray,
    noise_label: int = NOISE_LABEL
) -> np.ndarray:
    """Map labels -> RGB, with a dedicated gray for noise_label."""
    if labels.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    colors = np.empty((labels.size, 3), dtype=np.uint8)
    noise_mask = (labels == noise_label)

    # clamp negatives to 0 to avoid modulo surprises
    safe_lab = labels.copy()
    safe_lab[safe_lab < 0] = 0

    colors[~noise_mask] = palette[safe_lab[~noise_mask] % palette.shape[0]]
    colors[noise_mask]  = NOISE_COLOR
    return colors.astype(np.float32) / 255.0

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
    dumps = sorted(list(infer_dir.glob("*_inference.pth")))
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
    # Try Visualizer.reset_view_point if present
    try:
        if hasattr(vis, "reset_view_point"):
            vis.reset_view_point(True)
            ctr = vis.get_view_control()
            ctr.set_zoom(zoom)
            return
    except Exception:
        pass
    # Fallback: manual fit
    try:
        bbox = pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        ctr = vis.get_view_control()
        ctr.set_lookat(center.tolist())
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_up([0.0, -1.0, 0.0])
        ctr.set_zoom(zoom)
    except Exception:
        pass


# ---------------- Export: labels -> PLY/PNG ---------------- #

def _save_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    if not _HAS_O3D:
        return
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    p.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    o3d.io.write_point_cloud(str(path), p, write_ascii=False, compressed=True)

def _o3d_show_and_snapshot(xyz: np.ndarray, rgb: np.ndarray, title: str, png_path: Optional[Path]):
    if not _HAS_O3D:
        return
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=PNG_W, height=PNG_H, visible=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    vis.add_geometry(pcd)
    _reset_camera(vis, pcd, zoom=0.7)
    vis.poll_events(); vis.update_renderer()
    if png_path is not None:
        o3d.io.write_image(str(png_path), vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

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

        # Defensive length check
        if lab.shape[0] != coord.shape[0]:
            n = min(lab.shape[0], coord.shape[0])
            print(f"[export][warn] len(labels)={lab.shape[0]} != N={coord.shape[0]} → truncating to {n}.")
            lab = lab[:n]; coord = coord[:n]

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
    
    # stateful toggles
    state = {
        "i": 0,
        "mode": "labels",    # "labels" or "range"
        "fade": FADE_NON_VISIBLE,
        "palette_seed": 0,
        "palette": palette,
    }

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Unsupervised Segmentation Viewer", width=win_w, height=win_h)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    ro = vis.get_render_option()
    ro.point_size = 2.0

    def _compute_colors(stem: str, payload, labels: np.ndarray) -> np.ndarray:
        if state["mode"] == "range":
            # grayscale by Euclidean distance
            xyz = payload["coord_raw"].numpy().astype(np.float32)
            r = np.linalg.norm(xyz, axis=1)
            r = (r - r.min()) / (r.ptp() + 1e-6)
            gray = np.stack([r, r, r], axis=1)
            if state["fade"] and "mask" in payload:
                mask = payload["mask"].numpy().astype(bool)
                gray[~mask] = FADE_COLOR / 255.0
            return gray.astype(np.float32)

        # label coloring
        cols = _labels_to_colors(labels, state["palette"])
        if state["fade"] and "mask" in payload:
            mask = payload["mask"].numpy().astype(bool)
            cols[~mask] = FADE_COLOR / 255.0
        return cols

    def load_frame(i: int):
            stem = stems[i]
            dump_path = _resolve_dump_path(stem, by_fn, by_pl)
            payload = torch.load(dump_path, map_location="cpu")
            coord = payload["coord_raw"].numpy().astype(np.float32)
            lab   = load_labels(stem)

            # Defensive length check
            if lab.shape[0] != coord.shape[0]:
                n = min(lab.shape[0], coord.shape[0])
                print(f"[o3d][warn] len(labels)={lab.shape[0]} != N={coord.shape[0]} → truncating to {n}.")
                lab = lab[:n]; coord = coord[:n]

            # (re)size palette if needed (max label can exceed K)
            max_lab = int(lab[lab >= 0].max()) + 1 if (lab.size and (lab >= 0).any()) else K
            if max_lab > state["palette"].shape[0]:
                state["palette"] = _make_palette(max_lab, seed=state["palette_seed"])

            cols  = _compute_colors(stem, payload, lab)

            pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
            if cols.size:
                pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
            vis.update_geometry(pcd)

            _reset_camera(vis, pcd, zoom=0.7)
            vis.update_renderer()
            vis.poll_events()
            vis.get_render_option().point_size = ro.point_size
            print(f"[o3d] frame {i+1}/{len(stems)}: {stem} (N={coord.shape[0]}) mode={state['mode']} fade={state['fade']}")

    def next_frame(_): state["i"] = (state["i"] + 1) % len(stems); load_frame(state["i"]); return False
    def prev_frame(_): state["i"] = (state["i"] - 1) % len(stems); load_frame(state["i"]); return False
    def inc_ps(_): ro.point_size = min(ro.point_size + 1.0, 10.0); vis.get_render_option().point_size = ro.point_size; vis.update_renderer(); return False
    def dec_ps(_): ro.point_size = max(ro.point_size - 1.0, 1.0);  vis.get_render_option().point_size = ro.point_size; vis.update_renderer(); return False
    def reset_v(_): _reset_camera(vis, pcd, zoom=0.7); return False
    def toggle_mode(_):
        state["mode"] = "range" if state["mode"] == "labels" else "labels"
        load_frame(state["i"]); return False
    def toggle_fade(_):
        state["fade"] = not state["fade"]; load_frame(state["i"]); return False
    def cycle_palette(_):
        state["palette_seed"] += 1
        state["palette"] = _make_palette(max(K, state["palette"].shape[0]), seed=state["palette_seed"])
        load_frame(state["i"]); return False
    def quit_v(_): return True

    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("P"), prev_frame)
    vis.register_key_callback(ord("]"), inc_ps)
    vis.register_key_callback(ord("["), dec_ps)
    vis.register_key_callback(ord("R"), reset_v)
    vis.register_key_callback(ord("G"), toggle_mode)   # labels <-> range
    vis.register_key_callback(ord("V"), toggle_fade)   # fade non-visible
    vis.register_key_callback(ord("C"), cycle_palette)
    vis.register_key_callback(ord("Q"), quit_v)

    load_frame(state["i"])
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
