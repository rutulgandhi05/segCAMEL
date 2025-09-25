from pathlib import Path
from typing import Optional, Dict, Tuple, Callable
import numpy as np
import torch
import zipfile
from tqdm import tqdm
from datetime import datetime
import math

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False


# ---- CURRENT run (the one you want to view) ----
INFER_DIR  = Path(r"data\22092025_0509_segcamel_train_output_epoch_50_rvi\20250924_0137_inference_output_river_island_01_Day_rvi")
OUT_DIR    = Path(r"data\22092025_0509_segcamel_train_output_epoch_50_rvi\20250925_1147_unsup_outputs_river_island_01_Day_rvi")
K          = 12 # used only for initial palette sizing
LABELS_DIR = OUT_DIR / f"labels_k{K}"
LABELS_ZIP = OUT_DIR / f"labels_k{K}.zip"
PREFER_ZIP = True

# ---- REFERENCE labels (the run whose colors you want to copy) ----
# For cross-run comparability (e.g., align A3 colors to A2), point these to the reference run.
ALIGN_TO_REF   = True
REF_LABELS_DIR = OUT_DIR / f"labels_k{K}"   # change to your baseline run path if needed
REF_LABELS_ZIP = OUT_DIR / f"labels_k{K}.zip"
REF_PREFER_ZIP = True

#   "labels" -> export saved labels to PLY and/or PNG (no interactive window)
#   "o3d"    -> interactive Open3D viewer
VIEW_MODE = "o3d"  # "labels" or "o3d"

# Exports (only used in VIEW_MODE="labels")
SAVE_PLY = False
SAVE_PNG = False
PLY_LIMIT: Optional[int] = None  # cap number of exported frames
PNG_W, PNG_H = 1600, 1200

# Visualization knobs
NOISE_LABEL = -1
FADE_NON_VISIBLE = True
FADE_COLOR  = np.array([180, 180, 180], dtype=np.uint8)
NOISE_COLOR = np.array([128, 128, 128], dtype=np.uint8)

# Distinct palette instead of random (improves cluster separability by color)
USE_DISTINCT_PALETTE = True

# Label coloring policy in "labels" mode: "id" or "by_speed"
LABEL_COLOR_POLICY = "id"

# --- Speed visualisation defaults ---
# Modes: "off" (labels), "mag" (|v|), "signed" (v±), "fused" (labels tinted by |v|)
SPEED_VIS_MODE = "off"
SPEED_TAU = 0.5            # m/s threshold shown in stats & fused blending
SPEED_CLIP_PCT = 99.0      # percentile for color scaling (robust to outliers)

# Debug / logging
VERBOSE = True

def _dbg(msg: str):
    if VERBOSE:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")

# ==============================
# Palette & helper functions
# ==============================
def _hsv_to_rgb(h, s, v):
    """h in [0,1], s,v in [0,1]"""
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0: r,g,b = v,t,p
    elif i == 1: r,g,b = q,v,p
    elif i == 2: r,g,b = p,v,t
    elif i == 3: r,g,b = p,q,v
    elif i == 4: r,g,b = t,p,v
    else: r,g,b = v,p,q
    return int(r*255), int(g*255), int(b*255)

def _make_palette_distinct(k: int, seed: int = 0) -> np.ndarray:
    """Evenly spaced hues with golden-angle offsets; high contrast."""
    rng = np.random.default_rng(seed)
    base = []
    phi = (math.sqrt(5) - 1) / 2.0  # golden ratio conjugate ~0.618
    h0 = rng.random()
    for i in range(k):
        h = (h0 + i * phi) % 1.0
        s = 0.85
        v = 0.95
        base.append(_hsv_to_rgb(h, s, v))
    return np.array(base, dtype=np.uint8)

def _make_palette_random(k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(k, 3), dtype=np.uint8)
    base = np.maximum(base, 40)  # avoid near-black
    return base

def _make_palette(k: int, seed: int = 0) -> np.ndarray:
    if USE_DISTINCT_PALETTE:
        return _make_palette_distinct(k, seed=seed)
    return _make_palette_random(k, seed=seed)

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

# --- speed colormaps ---
def _cmap_mag(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    c = np.zeros((x.size, 3), dtype=np.float32)
    mid = 0.5
    lo = x < mid
    hi = ~lo
    if np.any(lo):
        t = (x[lo] / mid)[:, None]
        c[lo] = np.array([0.0, 0.0, 1.0]) * (1 - t) + np.array([0.0, 1.0, 1.0]) * t
    if np.any(hi):
        t = ((x[hi] - mid) / (1 - mid))[:, None]
        c[hi] = np.array([0.0, 1.0, 1.0]) * (1 - t) + np.array([1.0, 1.0, 0.0]) * t
    return c

def _cmap_signed(v: np.ndarray, vmax: Optional[float] = None) -> np.ndarray:
    v = v.astype(np.float32)
    if vmax is None or vmax <= 0:
        vmax = np.percentile(np.abs(v), SPEED_CLIP_PCT) + 1e-6
    x = np.clip(v / vmax, -1.0, 1.0)
    c = np.zeros((v.size, 3), dtype=np.float32)
    neg = x < 0
    if np.any(neg):
        t = (-x[neg])[:, None]
        c[neg] = np.array([1.0, 1.0, 1.0]) * (1 - t) + np.array([0.0, 0.0, 1.0]) * t
    pos = ~neg
    if np.any(pos):
        t = (x[pos])[:, None]
        c[pos] = np.array([1.0, 1.0, 1.0]) * (1 - t) + np.array([1.0, 0.0, 0.0]) * t
    return c

def _get_speed_payload(payload) -> Optional[np.ndarray]:
    """
    Returns |v| as float array if available, else None.
    Accepts either 'speed' (|v|) or 'vel_signed' tensors in payload.
    """
    v = payload.get("speed", None)
    if v is None:
        vs = payload.get("vel_signed", None)
        if vs is not None:
            v = torch.as_tensor(vs).float().abs()
    if v is None:
        return None
    return v.numpy().astype(np.float32).reshape(-1)

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
    for p in tqdm(dumps, desc="Scan dumps for payload stems", leave=False):
        try:
            payload = torch.load(p, map_location="cpu")
            s = payload.get("image_stem", None)
            if isinstance(s, str):
                by_payload[s] = p
        except Exception:
            pass
    return by_filename, by_payload

def _resolve_dump_path(stem: str, by_filename: Dict[str, Path], by_payload: Dict[str, Path]) -> Optional[Path]:
    return by_filename.get(stem) or by_payload.get(stem)

# ==============================
# Label alignment to reference
# ==============================
def _remap_to_reference(cur: np.ndarray, ref: np.ndarray, noise_label: int = NOISE_LABEL) -> np.ndarray:
    cur = cur.astype(np.int64)
    ref = ref.astype(np.int64)
    out = cur.copy()

    m = (cur >= 0) & (ref >= 0)
    if not np.any(m):
        return out

    cur_ids = np.unique(cur[m])
    ref_ids = np.unique(ref[m])
    cur_index = {c: i for i, c in enumerate(cur_ids)}
    ref_index = {r: j for j, r in enumerate(ref_ids)}
    C = np.zeros((len(cur_ids), len(ref_ids)), dtype=np.int64)

    pairs = np.stack([cur[m], ref[m]], axis=1)
    for c_id, r_id in pairs:
        C[cur_index[c_id], ref_index[r_id]] += 1

    mapping: Dict[int, int] = {}
    C_work = C.copy()
    used_rows, used_cols = set(), set()
    while True:
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

    for c_id, r_id in mapping.items():
        out[cur == c_id] = r_id
    out[cur == noise_label] = noise_label
    return out

# ==============================
# Open3D helpers
# ==============================
def _reset_camera(vis, pcd, zoom=0.9):
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
    _reset_camera(vis, pcd, zoom=1.0)
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

        if cur_lab.shape[0] != coord.shape[0]:
            n = min(cur_lab.shape[0], coord.shape[0])
            print(f"[export][warn] len(labels)={cur_lab.shape[0]} != N={coord.shape[0]} → truncating to {n}.")
            cur_lab = cur_lab[:n]; coord = coord[:n]

        if ALIGN_TO_REF and ref_loader is not None and stem in ref_map:
            try:
                ref_lab = ref_loader(stem)
                if ref_lab.shape[0] != cur_lab.shape[0]:
                    n = min(ref_lab.shape[0], cur_lab.shape[0])
                    ref_lab = ref_lab[:n]; cur_lab = cur_lab[:n]; coord = coord[:n]
                cur_lab = _remap_to_reference(cur_lab, ref_lab, noise_label=NOISE_LABEL)
            except Exception as e:
                print(f"[export][align] failed for {stem}: {e}")

        max_label = int(cur_lab[cur_lab >= 0].max()) + 1 if (cur_lab.size and (cur_lab >= 0).any()) else K
        if max_label > palette.shape[0]:
            palette = _make_palette(max_label, seed=0)

        # plain ID-based colors
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
        "mode": "labels",           # "labels" or "range" (legacy)
        "fade": FADE_NON_VISIBLE,
        "palette": palette_init.copy(),
        "palette_seed": 0,
        "speed_mode": SPEED_VIS_MODE,   # "off", "mag", "signed", "fused"
        "speed_tau": float(SPEED_TAU),
        "label_color_policy": LABEL_COLOR_POLICY   # "id" or "by_speed"
    }

    def _speed_stats(v: np.ndarray) -> Tuple[float, float]:
        if v.size == 0:
            return 0.0, 0.0
        nz_frac = float((np.abs(v) > state["speed_tau"]).mean())
        vmax = float(np.percentile(np.abs(v), SPEED_CLIP_PCT)) + 1e-6
        return nz_frac, vmax

    def _by_speed_colors(payload, labels: np.ndarray) -> Optional[np.ndarray]:
        v = payload.get("speed", None)
        if v is None:
            return None
        v = v.numpy().astype(np.float32).reshape(-1)
        lab = labels.astype(np.int64)
        m = lab >= 0
        if not np.any(m):
            return None
        # mean |v| per label
        vmax = float(np.percentile(np.abs(v[m]), SPEED_CLIP_PCT)) + 1e-6
        mean_v = {}
        for lid in np.unique(lab[m]):
            li = (lab == lid)
            mean_v[int(lid)] = float(np.mean(np.abs(v[li])))
        # normalize to [0,1] for palette mapping (cool->warm)
        lids = sorted(mean_v.keys())
        vals = np.array([min(mean_v[l], vmax)/vmax for l in lids], dtype=np.float32)
        # build a small gradient palette blue->red
        pal = np.zeros((len(lids), 3), dtype=np.float32)
        for i, t in enumerate(vals):
            pal[i] = _cmap_signed(np.array([ (t - 0.5)*2.0 ]), vmax=1.0)[0]  # [-1,1] -> blue/white/red
        # assign
        id2row = {lid: i for i, lid in enumerate(lids)}
        cols = np.empty((lab.size, 3), dtype=np.float32)
        for idx in range(lab.size):
            lid = int(lab[idx])
            if lid < 0:
                cols[idx] = NOISE_COLOR / 255.0
            else:
                cols[idx] = pal[id2row[lid]]
        # print a short summary
        top = sorted(mean_v.items(), key=lambda kv: kv[1], reverse=True)[:5]
        _dbg("Top labels by mean |v|: " + ", ".join([f"{lid}:{mv:.2f}" for lid,mv in top]))
        return cols

    def _compute_colors(payload, labels: np.ndarray) -> np.ndarray:
        # --- speed-only modes ---
        if state["speed_mode"] == "mag":
            v = _get_speed_payload(payload)
            if v is None:
                return _labels_to_colors(labels, state["palette"])
            vmax = float(np.percentile(np.abs(v), SPEED_CLIP_PCT)) + 1e-6
            x = np.clip(np.abs(v) / vmax, 0.0, 1.0)
            cols = _cmap_mag(x)

        elif state["speed_mode"] == "signed":
            vs = payload.get("vel_signed", None)
            v = vs.numpy().astype(np.float32).reshape(-1) if vs is not None else _get_speed_payload(payload)
            if v is None:
                return _labels_to_colors(labels, state["palette"])
            vmax = float(np.percentile(np.abs(v), SPEED_CLIP_PCT)) + 1e-6
            cols = _cmap_signed(v, vmax=vmax)

        elif state["speed_mode"] == "fused":
            # Base label colors
            base = _labels_to_colors(labels, state["palette"])
            v = _get_speed_payload(payload)
            if v is None:
                cols = base
            else:
                vabs = np.abs(v)
                vmax = float(np.percentile(vabs, SPEED_CLIP_PCT)) + 1e-6
                tau  = float(state["speed_tau"])
                # Normalized |v| and a smooth mover mask around τ
                x = np.clip(vabs / vmax, 0.0, 1.0)
                delta = max(0.3, 0.2 * vmax)  # soft transition width
                t = np.clip((vabs - tau) / max(delta, 1e-6), 0.0, 1.0)
                m = t * t * (3 - 2 * t)       # smoothstep in [0,1]
                # Red-ish overlay for movers
                overlay = np.tile(np.array([[1.0, 0.25, 0.25]], dtype=np.float32), (base.shape[0], 1))
                alpha = 0.70
                a = (alpha * m)[:, None]
                cols = (1 - a) * base + a * overlay

            if state["fade"] and "mask" in payload:
                mask = payload["mask"].numpy().astype(bool)
                cols[~mask] = FADE_COLOR / 255.0
            return cols

        else:
            # Label modes (no speed override)
            if state["mode"] == "range":
                xyz = payload["coord_raw"].numpy().astype(np.float32)
                r = np.linalg.norm(xyz, axis=1)
                r = (r - r.min()) / (np.ptp(r) + 1e-6)
                cols = np.stack([r, r, r], axis=1).astype(np.float32)
            else:
                if state["label_color_policy"] == "by_speed":
                    cols = _by_speed_colors(payload, labels)
                    if cols is None:
                        cols = _labels_to_colors(labels, state["palette"])
                else:
                    cols = _labels_to_colors(labels, state["palette"])

        # Common fade path for non-fused modes
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

        if cur_lab.shape[0] != coord.shape[0]:
            n = min(cur_lab.shape[0], coord.shape[0])
            cur_lab = cur_lab[:n]; coord = coord[:n]

        if ALIGN_TO_REF and ref_loader is not None and stem in ref_map:
            try:
                ref_lab = ref_loader(stem)
                if ref_lab.shape[0] != cur_lab.shape[0]:
                    n = min(ref_lab.shape[0], cur_lab.shape[0])
                    ref_lab = ref_lab[:n]; cur_lab = cur_lab[:n]; coord = coord[:n]
                cur_lab = _remap_to_reference(cur_lab, ref_lab, noise_label=NOISE_LABEL)
                max_label = int(cur_lab[cur_lab >= 0].max()) + 1 if (cur_lab.size and (cur_lab >= 0).any()) else K
                if max_label > state["palette"].shape[0]:
                    state["palette"] = _make_palette(max_label, seed=state["palette_seed"])
            except Exception:
                pass

        pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
        cols = _compute_colors(payload, cur_lab)
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        vis.update_geometry(pcd)
        _reset_camera(vis, pcd, zoom=1.0)
        vis.update_renderer(); vis.poll_events()
        vis.get_render_option().point_size = ro.point_size

        # console stats for speed modes
        if state["speed_mode"] in ("mag", "signed", "fused"):
            v = payload.get("vel_signed", None) if state["speed_mode"] == "signed" else payload.get("speed", None)
            if v is None:  # fall back to magnitude if signed missing
                v = payload.get("speed", None)
            if v is not None:
                v = v.numpy().astype(np.float32).reshape(-1)
                nz_frac, vmax = _speed_stats(v)
                print(f"[o3d] frame {i+1}/{len(stems)}: {stem}  mode={state['speed_mode']}  τ={state['speed_tau']:.2f} m/s"
                      f" | movers(>|τ|)={nz_frac*100:.1f}%  vmax@{SPEED_CLIP_PCT}p={vmax:.2f} m/s")
        else:
            print(f"[o3d] frame {i+1}/{len(stems)}: {stem} mode={state['mode']} policy={state['label_color_policy']} fade={state['fade']} align={ALIGN_TO_REF}")

    # Key handlers
    def next_frame(_): state["i"] = (state["i"] + 1) % len(stems); load_frame(state["i"]); return False
    def prev_frame(_): state["i"] = (state["i"] - 1) % len(stems); load_frame(state["i"]); return False
    def inc_ps(_): ro.point_size = min(ro.point_size + 1.0, 10.0); vis.get_render_option().point_size = ro.point_size; vis.update_renderer(); return False
    def dec_ps(_): ro.point_size = max(ro.point_size - 1.0, 1.0);  vis.get_render_option().point_size = ro.point_size; vis.update_renderer(); return False
    def reset_v(_): _reset_camera(vis, pcd, zoom=1.0); return False

    def toggle_mode(_):
        state["mode"] = "range" if state["mode"] == "labels" else "labels"
        state["speed_mode"] = "off"  # leave speed modes when toggling legacy
        load_frame(state["i"]); return False

    def cycle_speed(_):
        order = ["off", "mag", "signed", "fused"]   # added "fused"
        i = order.index(state["speed_mode"]) if state["speed_mode"] in order else 0
        state["speed_mode"] = order[(i + 1) % len(order)]
        load_frame(state["i"]); return False

    def cycle_label_policy(_):
        state["label_color_policy"] = "by_speed" if state["label_color_policy"] == "id" else "id"
        load_frame(state["i"]); return False

    def tau_inc(_):
        state["speed_tau"] = state["speed_tau"] + 0.5
        _dbg(f"[viewer] τ -> {state['speed_tau']:.2f} m/s"); load_frame(state["i"]); return False
    def tau_dec(_):
        state["speed_tau"] = state["speed_tau"] - 0.5
        _dbg(f"[viewer] τ -> {state['speed_tau']:.2f} m/s"); load_frame(state["i"]); return False

    def toggle_fade(_): state["fade"] = not state["fade"]; load_frame(state["i"]); return False
    def cycle_palette(_):
        state["palette_seed"] += 1
        sz = max(K, state["palette"].shape[0])
        state["palette"] = _make_palette(sz, seed=state["palette_seed"])
        load_frame(state["i"]); return False
    def quit_v(_): return True

    # Bindings
    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("P"), prev_frame)
    vis.register_key_callback(ord("]"), inc_ps)
    vis.register_key_callback(ord("["), dec_ps)
    vis.register_key_callback(ord("R"), reset_v)
    vis.register_key_callback(ord("G"), toggle_mode)     # labels <-> range
    vis.register_key_callback(ord("S"), cycle_speed)     # off -> |v| -> signed -> fused
    vis.register_key_callback(ord("M"), cycle_label_policy)  # id <-> by_speed
    # τ controls (robust to keyboard layouts)
    vis.register_key_callback(ord('='), tau_inc)
    vis.register_key_callback(ord('+'), tau_inc)
    vis.register_key_callback(ord('-'), tau_dec)
    vis.register_key_callback(ord('_'), tau_dec)
    vis.register_key_callback(ord('K'), tau_inc)         # backups
    vis.register_key_callback(ord('J'), tau_dec)
    vis.register_key_callback(ord("V"), toggle_fade)
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
    print(f"[mode ] VIEW_MODE={VIEW_MODE} | O3D={_HAS_O3D} align={ALIGN_TO_REF}")

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
