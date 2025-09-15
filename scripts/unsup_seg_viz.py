from pathlib import Path
from typing import Optional, Dict, Tuple, Callable
import numpy as np
import torch
import zipfile
from tqdm import tqdm
try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False


# ---- CURRENT run (the one you want to view) ----
INFER_DIR  = Path(r"data\09092025_0900_segcamel_train_output_epoch_50\12092025_0532_inference_output_parking_lot_02_Day_nospd")
OUT_DIR    = Path(r"data\09092025_0900_segcamel_train_output_epoch_50\12092025_0532_unsup_outputs_parking_lot_02_Day_nospd")
K          = 10  # used only for initial palette sizing
LABELS_DIR = OUT_DIR / f"labels_k{K}"
LABELS_ZIP = OUT_DIR / f"labels_k{K}.zip"
PREFER_ZIP = True

# ---- REFERENCE labels (the run whose colors you want to copy) ----
ALIGN_TO_REF   = True  # turn OFF to color current labels directly
REF_LABELS_DIR = Path(r"data\09092025_0900_segcamel_train_output_epoch_50\12092025_0532_unsup_outputs_parking_lot_02_Day") / f"labels_k{K}"
REF_LABELS_ZIP = Path(r"data\09092025_0900_segcamel_train_output_epoch_50\12092025_0532_unsup_outputs_parking_lot_02_Day") / f"labels_k{K}.zip"
REF_PREFER_ZIP = True

#   "labels" -> export saved labels to PLY and/or PNG (no interactive window)
#   "o3d"    -> interactive Open3D viewer
VIEW_MODE = "o3d"  # "labels" or "o3d"

# Exports (only used in VIEW_MODE="labels")
SAVE_PLY = False
SAVE_PNG = False
PLY_LIMIT: Optional[int] = None  # set to an int to cap number of exported frames
PNG_W, PNG_H = 1600, 1200

# Visualization knobs
NOISE_LABEL = -1
FADE_NON_VISIBLE = True
FADE_COLOR  = np.array([180, 180, 180], dtype=np.uint8)
NOISE_COLOR = np.array([128, 128, 128], dtype=np.uint8)

# ==============================
# Palette & helper functions
# ==============================
def _make_palette(k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(k, 3), dtype=np.uint8)
    base = np.maximum(base, 40)  # avoid near-black
    return base

def _labels_to_colors(labels: np.ndarray, palette: np.ndarray, noise_label: int = NOISE_LABEL) -> np.ndarray:
    if labels.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    colors = np.empty((labels.size, 3), dtype=np.uint8)
    noise_mask = (labels == noise_label)
    safe_lab = labels.copy()
    safe_lab[safe_lab < 0] = 0
    idx = safe_lab % palette.shape[0]
    colors[~noise_mask] = palette[idx[~noise_mask]]
    colors[noise_mask]  = NOISE_COLOR
    return colors.astype(np.float32) / 255.0

# ==============================
# Label stores (dir/zip)
# ==============================
def _labels_store_exists(dir_path: Path, zip_path: Path, prefer_zip: bool) -> Tuple[bool, str]:
    zip_ok = zip_path.exists()
    dir_ok = dir_path.exists()
    if prefer_zip and zip_ok:
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

def _build_current_label_loader() -> Tuple[Callable[[str], np.ndarray], Dict[str, str], str]:
    ok, mode = _labels_store_exists(LABELS_DIR, LABELS_ZIP, PREFER_ZIP)
    if not ok:
        raise FileNotFoundError(f"No labels found for CURRENT run (checked {LABELS_DIR} and {LABELS_ZIP}).")
    if mode == "zip":
        label_map = _zip_list_labels(LABELS_ZIP)
        loader = lambda stem: _zip_load_labels(LABELS_ZIP, label_map[stem])
    else:
        label_map = {p.stem.replace("_clusters", ""): str(p) for p in sorted(LABELS_DIR.glob("*_clusters.npz"))}
        def loader(stem: str) -> np.ndarray:
            return np.load(label_map[stem])["labels"].astype(np.int64)
    return loader, label_map, mode

def _build_ref_label_loader() -> Tuple[Optional[Callable[[str], np.ndarray]], Dict[str, str], str]:
    ok, mode = _labels_store_exists(REF_LABELS_DIR, REF_LABELS_ZIP, REF_PREFER_ZIP)
    if not ok:
        return None, {}, "none"
    if mode == "zip":
        label_map = _zip_list_labels(REF_LABELS_ZIP)
        loader = lambda stem: _zip_load_labels(REF_LABELS_ZIP, label_map[stem])
    else:
        label_map = {p.stem.replace("_clusters", ""): str(p) for p in sorted(REF_LABELS_DIR.glob("*_clusters.npz"))}
        def loader(stem: str) -> np.ndarray:
            return np.load(label_map[stem])["labels"].astype(np.int64)
    return loader, label_map, mode

# ==============================
# Inference dump mapping
# ==============================
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

def _resolve_dump_path(stem: str, by_filename: Dict[str, Path], by_payload: Dict[str, Path]) -> Optional[Path]:
    if stem in by_filename: return by_filename[stem]
    if stem in by_payload:  return by_payload[stem]
    return None

# ==============================
# Label alignment to reference
# ==============================
def _remap_to_reference(cur: np.ndarray, ref: np.ndarray, noise_label: int = NOISE_LABEL) -> np.ndarray:
    """
    Greedy overlap matching: map each current cluster ID to the reference cluster ID
    with which it overlaps the most. Noise label is preserved as-is.
    """
    cur = cur.astype(np.int64)
    ref = ref.astype(np.int64)
    out = cur.copy()

    # indices where both are valid (>=0)
    m = (cur >= 0) & (ref >= 0)
    if not np.any(m):
        return out  # nothing to map

    cur_ids = np.unique(cur[m])
    ref_ids = np.unique(ref[m])

    # small contingency table with compact indices
    cur_index = {c: i for i, c in enumerate(cur_ids)}
    ref_index = {r: j for j, r in enumerate(ref_ids)}
    C = np.zeros((len(cur_ids), len(ref_ids)), dtype=np.int64)

    # accumulate overlaps
    # (vectorized by building pairs)
    pairs = np.stack([cur[m], ref[m]], axis=1)
    # count by pairs
    # (simple loop is okay; arrays are per-frame)
    for c_id, r_id in pairs:
        C[cur_index[c_id], ref_index[r_id]] += 1

    # greedy assignment
    mapping: Dict[int, int] = {}
    C_work = C.copy()
    used_rows = set()
    used_cols = set()
    while True:
        # find max cell not used
        best = None
        best_val = 0
        for i in range(C_work.shape[0]):
            if i in used_rows: continue
            for j in range(C_work.shape[1]):
                if j in used_cols: continue
                v = C_work[i, j]
                if v > best_val:
                    best_val = v
                    best = (i, j)
        if best is None or best_val == 0:
            break
        i, j = best
        mapping[cur_ids[i]] = ref_ids[j]
        used_rows.add(i); used_cols.add(j)

    # apply mapping
    for c_id, r_id in mapping.items():
        out[cur == c_id] = r_id

    # keep noise label untouched
    out[cur == noise_label] = noise_label
    return out

# ==============================
# Open3D helpers
# ==============================
def _reset_camera(vis, pcd, zoom=0.7):
    try:
        if hasattr(vis, "reset_view_point"):
            vis.reset_view_point(True)
            ctr = vis.get_view_control()
            ctr.set_zoom(zoom)
            return
    except Exception:
        pass
    try:
        bbox = pcd.get_axis_aligned_bounding_box()
        ctr = vis.get_view_control()
        ctr.set_lookat(bbox.get_center().tolist())
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_up([0.0, -1.0, 0.0])
        ctr.set_zoom(zoom)
    except Exception:
        pass

def _save_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    if not _HAS_O3D: return
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    p.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    o3d.io.write_point_cloud(str(path), p, write_ascii=False, compressed=True)

def _o3d_show_and_snapshot(xyz: np.ndarray, rgb: np.ndarray, title: str, png_path: Optional[Path]):
    if not _HAS_O3D: return
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

# ==============================
# Export mode (PLY/PNG)
# ==============================
def _export_labels_to_ply_png(infer_dir: Path, out_dir: Path, palette: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = out_dir / "ply"
    png_dir = out_dir / "png"
    if SAVE_PLY: ply_dir.mkdir(exist_ok=True)
    if SAVE_PNG: png_dir.mkdir(exist_ok=True)

    cur_loader, cur_map, _ = _build_current_label_loader()
    ref_loader, ref_map, _ = _build_ref_label_loader() if ALIGN_TO_REF else (None, {}, "none")

    by_fn, by_pl = _build_dump_maps(infer_dir)
    exported = 0

    for stem in sorted(cur_map.keys()):
        dump = _resolve_dump_path(stem, by_fn, by_pl)
        if dump is None:
            print(f"[export] missing dump for stem={stem}, skipping.")
            continue

        try:
            cur_lab = cur_loader(stem)
            payload = torch.load(dump, map_location="cpu")
            coord   = payload["coord_raw"].numpy().astype(np.float32)
        except Exception as e:
            print(f"[export] error loading {stem}: {e}")
            continue

        # defensive length check
        if cur_lab.shape[0] != coord.shape[0]:
            n = min(cur_lab.shape[0], coord.shape[0])
            print(f"[export][warn] len(labels)={cur_lab.shape[0]} != N={coord.shape[0]} → truncating to {n}.")
            cur_lab = cur_lab[:n]; coord = coord[:n]

        # optional remap to reference IDs
        if ALIGN_TO_REF and ref_loader is not None and stem in ref_map:
            try:
                ref_lab = ref_loader(stem)
                if ref_lab.shape[0] != cur_lab.shape[0]:
                    n = min(ref_lab.shape[0], cur_lab.shape[0])
                    ref_lab = ref_lab[:n]; cur_lab = cur_lab[:n]; coord = coord[:n]
                cur_lab = _remap_to_reference(cur_lab, ref_lab, noise_label=NOISE_LABEL)
            except Exception as e:
                print(f"[export][align] failed for {stem}: {e}")

        # ensure palette large enough
        max_label = int(cur_lab[cur_lab >= 0].max()) + 1 if (cur_lab.size and (cur_lab >= 0).any()) else K
        if max_label > palette.shape[0]:
            palette = _make_palette(max_label, seed=0)

        cols = _labels_to_colors(cur_lab, palette)
        if FADE_NON_VISIBLE and "mask" in payload:
            mask = payload["mask"].numpy().astype(bool)
            cols[~mask] = FADE_COLOR / 255.0

        if SAVE_PLY:
            _save_ply(ply_dir / f"{stem}.ply", coord, cols)
        if _HAS_O3D and SAVE_PNG:
            _o3d_show_and_snapshot(coord, cols, title=f"Seg: {stem}", png_path=png_dir / f"{stem}.png")

        exported += 1
        if PLY_LIMIT is not None and exported >= int(PLY_LIMIT):
            break

    print(f"[export] done. exported={exported}, PLY={SAVE_PLY}, PNG={SAVE_PNG}, align={ALIGN_TO_REF}")

# ==============================
# Interactive viewer
# ==============================
def _view_labels_o3d(infer_dir: Path, palette_init: np.ndarray, win_w=1600, win_h=1200):
    if not _HAS_O3D:
        print("Open3D is not installed.")
        return

    cur_loader, cur_map, _ = _build_current_label_loader()
    ref_loader, ref_map, _ = _build_ref_label_loader() if ALIGN_TO_REF else (None, {}, "none")
    by_fn, by_pl = _build_dump_maps(infer_dir)

    stems = [s for s in sorted(cur_map.keys()) if _resolve_dump_path(s, by_fn, by_pl) is not None]
    if not stems:
        print("No overlapping frames found between inference dumps and labels.")
        return

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Unsupervised Segmentation Viewer", width=win_w, height=win_h)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    ro = vis.get_render_option(); ro.point_size = 2.0

    state = {
        "i": 0,
        "mode": "labels",     # "labels" or "range"
        "fade": FADE_NON_VISIBLE,
        "palette": palette_init.copy(),
        "palette_seed": 0
    }

    def _compute_colors(payload, labels: np.ndarray) -> np.ndarray:
        if state["mode"] == "range":
            xyz = payload["coord_raw"].numpy().astype(np.float32)
            r = np.linalg.norm(xyz, axis=1)
            r = (r - r.min()) / (np.ptp(r) + 1e-6)
            cols = np.stack([r, r, r], axis=1).astype(np.float32)
        else:
            cols = _labels_to_colors(labels, state["palette"])
        if state["fade"] and "mask" in payload:
            mask = payload["mask"].numpy().astype(bool)
            cols[~mask] = FADE_COLOR / 255.0
        return cols

    def load_frame(i: int):
        stem = stems[i]
        dump_path = _resolve_dump_path(stem, by_fn, by_pl)
        payload = torch.load(dump_path, map_location="cpu")
        coord   = payload["coord_raw"].numpy().astype(np.float32)
        cur_lab = cur_loader(stem)

        # truncate lengths if needed
        if cur_lab.shape[0] != coord.shape[0]:
            n = min(cur_lab.shape[0], coord.shape[0])
            cur_lab = cur_lab[:n]; coord = coord[:n]

        # optional remap to reference for consistent coloring
        if ALIGN_TO_REF and ref_loader is not None and stem in ref_map:
            try:
                ref_lab = ref_loader(stem)
                if ref_lab.shape[0] != cur_lab.shape[0]:
                    n = min(ref_lab.shape[0], cur_lab.shape[0])
                    ref_lab = ref_lab[:n]; cur_lab = cur_lab[:n]; coord = coord[:n]
                cur_lab = _remap_to_reference(cur_lab, ref_lab, noise_label=NOISE_LABEL)

                # ensure palette large enough for reference IDs
                max_label = int(cur_lab[cur_lab >= 0].max()) + 1 if (cur_lab.size and (cur_lab >= 0).any()) else K
                if max_label > state["palette"].shape[0]:
                    state["palette"] = _make_palette(max_label, seed=state["palette_seed"])
            except Exception as e:
                print(f"[o3d][align] failed for {stem}: {e}")

        pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
        cols = _compute_colors(payload, cur_lab)
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        vis.update_geometry(pcd)
        _reset_camera(vis, pcd, zoom=0.7)
        vis.update_renderer(); vis.poll_events()
        vis.get_render_option().point_size = ro.point_size
        print(f"[o3d] frame {i+1}/{len(stems)}: {stem} mode={state['mode']} fade={state['fade']} align={ALIGN_TO_REF}")

    def next_frame(_): state["i"] = (state["i"] + 1) % len(stems); load_frame(state["i"]); return False
    def prev_frame(_): state["i"] = (state["i"] - 1) % len(stems); load_frame(state["i"]); return False
    def inc_ps(_): ro.point_size = min(ro.point_size + 1.0, 10.0); vis.get_render_option().point_size = ro.point_size; vis.update_renderer(); return False
    def dec_ps(_): ro.point_size = max(ro.point_size - 1.0, 1.0);  vis.get_render_option().point_size = ro.point_size; vis.update_renderer(); return False
    def reset_v(_): _reset_camera(vis, pcd, zoom=0.7); return False
    def toggle_mode(_):
        state["mode"] = "range" if state["mode"] == "labels" else "labels"
        load_frame(state["i"]); return False
    def toggle_fade(_): state["fade"] = not state["fade"]; load_frame(state["i"]); return False
    def cycle_palette(_):
        state["palette_seed"] += 1
        # keep size; just reseed for a different look
        sz = max(K, state["palette"].shape[0])
        state["palette"] = _make_palette(sz, seed=state["palette_seed"])
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

# ==============================
# Sanity + main
# ==============================
def _count_infer() -> int:
    return len(list(INFER_DIR.glob("*_inference.pth")))

def _count_labels(dir_path: Path, zip_path: Path, prefer_zip: bool) -> Tuple[int, str]:
    ok, mode = _labels_store_exists(dir_path, zip_path, prefer_zip)
    if not ok: return 0, "none"
    if mode == "zip":
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                return sum(1 for n in zf.namelist() if n.endswith("_clusters.npz")), "zip"
        except Exception:
            return 0, "zip"
    else:
        return len(list(dir_path.glob("*_clusters.npz"))), "dir"

def _sanity_print():
    n_inf = _count_infer()
    n_cur, cur_store = _count_labels(LABELS_DIR, LABELS_ZIP, PREFER_ZIP)
    print(f"[paths] INFER_DIR   = {INFER_DIR}  ({n_inf} dumps)")
    print(f"[paths] CUR labels  = {LABELS_DIR if cur_store=='dir' else LABELS_ZIP} ({n_cur} files) store={cur_store}")
    if ALIGN_TO_REF:
        n_ref, ref_store = _count_labels(REF_LABELS_DIR, REF_LABELS_ZIP, REF_PREFER_ZIP)
        print(f"[paths] REF labels  = {REF_LABELS_DIR if ref_store=='dir' else REF_LABELS_ZIP} ({n_ref} files) store={ref_store}")
    print(f"[mode ] VIEW_MODE={VIEW_MODE} | SAVE_PLY={SAVE_PLY} SAVE_PNG={SAVE_PNG} O3D={_HAS_O3D} align={ALIGN_TO_REF}")

def main():
    if not INFER_DIR.exists():
        raise FileNotFoundError(f"INFER_DIR not found: {INFER_DIR}")
    _sanity_print()

    palette = _make_palette(K, seed=0)

    if VIEW_MODE == "labels":
        _export_labels_to_ply_png(INFER_DIR, OUT_DIR, palette)
        return
    if VIEW_MODE == "o3d":
        _view_labels_o3d(INFER_DIR, palette, win_w=PNG_W, win_h=PNG_H)
        return

    raise ValueError(f"Unknown VIEW_MODE={VIEW_MODE} (use 'labels' or 'o3d').")

if __name__ == "__main__":
    main()
