import io
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import tarfile

import numpy as np
from tqdm import tqdm

from hercules.aeva import load_aeva_bin  # keep your parser as-is
from utils.files import read_mcap_file
from scantinel.parse_mcap_pcl import parse_pcl
from utils.misc import find_closest, _resolve_default_workers


# -----------------------------
# Helpers
# -----------------------------
def _safe_K(default_hw=(640, 480)):
    # Fallback intrinsics if YAML missing or malformed
    fx, fy = float(default_hw[0]), float(default_hw[1])
    cx, cy = float(default_hw[0]) / 2.0, float(default_hw[1]) / 2.0
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def _load_intrinsics_yaml(file_path: Path):
    """
    Robust YAML intrinsics loader.
    Returns (K 3x3 float32, D or None). Falls back to identity-like K if parsing fails.
    """
    if not file_path.exists():
        return _safe_K(), None

    # Try PyYAML first (preferred)
    try:
        import yaml  # type: ignore
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        # Common layouts:
        # - K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        # - camera_matrix: { data: [...] }
        if "K" in data:
            kv = np.array(data["K"], dtype=np.float32).reshape(3, 3)
        elif "camera_matrix" in data and "data" in data["camera_matrix"]:
            kv = np.array(data["camera_matrix"]["data"], dtype=np.float32).reshape(3, 3)
        else:
            kv = _safe_K()

        # Distortion optional
        D = None
        for key in ("D", "distortion", "distortion_coefficients"):
            if key in data:
                arr = data[key]["data"] if isinstance(data[key], dict) and "data" in data[key] else data[key]
                D = np.asarray(arr, dtype=np.float32).reshape(-1)
                break

        return kv.astype(np.float32), (D if D is not None and D.size > 0 else None)

    except Exception:
        # Very lightweight fallback parser when YAML is unavailable / malformed
        try:
            lines = file_path.read_text().splitlines()
            # Scan for a line that contains 9 numbers for K
            def _nums(s): 
                # split on space or tab
                return np.fromstring(s.replace(",", " "), sep=" ", dtype=np.float32)

            K = None
            for ln in lines:
                vals = _nums(ln)
                if vals.size == 9:
                    K = vals.reshape(3, 3)
                    break
            if K is None:
                K = _safe_K()
            # no reliable D in fallback
            return K.astype(np.float32), None
        except Exception:
            return _safe_K(), None


def _load_extrinsics_txt(file_path: Path):
    """
    Parse stereo_lidar.txt by labels, not by line indices.
    Expected keys (case-insensitive substrings):
      - 'Tr_lidar_to_leftcam'
      - 'Tr_lidar_to_rightcam'
    Returns (lidar_to_left_4x4, lidar_to_right_4x4). On failure -> (I, I).
    """
    I = np.eye(4, dtype=np.float32)
    if not file_path.exists():
        return I, I

    left = None
    right = None
    try:
        lines = file_path.read_text().splitlines()
        for line in lines:
            s = line.strip()
            if not s or ":" not in s or s.startswith("#"):
                continue
            key, vals = s.split(":", 1)
            key_l = key.strip().lower()
            nums = [float(x) for x in vals.strip().split()]
            if len(nums) != 12:
                continue
            mat = np.array(nums, dtype=np.float32).reshape(3, 4)
            M = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]]).astype(np.float32)
            if "lidar" in key_l and "left" in key_l:
                left = M
            elif "lidar" in key_l and "right" in key_l:
                right = M
        return (left if left is not None else I), (right if right is not None else I)
    except Exception:
        return I, I


def _first_existing_path(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # return first candidate even if missing, so glob() just yields empty


def _glob_images(folder: Path):
    if not folder.exists():
        return []
    imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        imgs.extend(sorted(folder.glob(ext)))
    return imgs


def _int_stem(p: Path):
    try:
        return int(p.stem)
    except Exception:
        return None


def _extract_if_needed(tar_path: Path, target_dir: Path):
    """
    Extract only if target_dir is missing or empty.
    """
    if not tar_path.exists():
        return
    
    print(f" [DATASET] Extracting {tar_path} ...")
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=target_dir)


def _process_hercules_bin(
    bin_path: Path,
    return_all_fields: bool,
    left_stamps: list[int],
    right_stamps: list[int],
    left_dict: dict[int, Path],
    right_dict: dict[int, Path],
    stereo_left_intr: np.ndarray,
    stereo_right_intr: np.ndarray,
    lidar_to_left_ext: np.ndarray,
    lidar_to_right_ext: np.ndarray,
):
    pc = load_aeva_bin(bin_path, return_all_fields=return_all_fields)
    if pc is None:
        return None

    ts = int(bin_path.stem)
    l_ts = find_closest(left_stamps, ts) if left_stamps else None
    r_ts = find_closest(right_stamps, ts) if right_stamps else None

    l_img = left_dict.get(l_ts)
    r_img = right_dict.get(r_ts)
    if l_img is None and r_img is None:
        return None

    # Prefer right if available, else left
    if r_img is not None:
        used_camera = "right"
        used_image = r_img
        K_used = stereo_right_intr
        T_used = lidar_to_right_ext
    else:
        used_camera = "left"
        used_image = l_img
        K_used = stereo_left_intr
        T_used = lidar_to_left_ext

    return {
        "pointcloud": pc,
        # convenience single-view (matched) fields:
        "image": used_image,
        "intrinsics": K_used,                  # 3x3 float32
        "extrinsics": T_used,                  # 4x4 float32 (LiDAR -> used camera)
        "used_camera": used_camera,            # "right" or "left"

        # keep the original dual-view info too (for debugging / multi-view use):
        "left_image": l_img,
        "right_image": r_img,
        "timestamps": [ts, l_ts, r_ts],
        "stereo_left_intrinsics": stereo_left_intr,
        "stereo_right_intrinsics": stereo_right_intr,
        "lidar_to_stereo_left_extrinsic": lidar_to_left_ext,
        "lidar_to_stereo_right_extrinsic": lidar_to_right_ext,
    }


def load_hercules_dataset_folder(dataset_folder: Path, return_all_fields=False, max_workers: int = None):
    """
    Hercules dataset.
    Returns a list[dict] per LiDAR bin:
      - pointcloud (np.ndarray)
      - image (Path)               # matched to used_camera
      - intrinsics (np.ndarray 3x3)
      - extrinsics (np.ndarray 4x4)
      - used_camera ("right"/"left")
      - left_image / right_image (Path or None)
      - timestamps [lidar_ts, left_ts, right_ts]
      - stereo_left_intrinsics / stereo_right_intrinsics (np.ndarray 3x3)
      - lidar_to_stereo_left_extrinsic / lidar_to_stereo_right_extrinsic (np.ndarray 4x4)
    """
    dataset_folder = Path(dataset_folder)

    # number of workers to use for the ProcessPool
    if max_workers is None:
        max_workers = _resolve_default_workers()
    max_workers = max(1, int(max_workers))  # Ensure at least one worker

    # --- Extraction (idempotent) ---
    lidar_zip = _first_existing_path(
        dataset_folder / "Aeva_data" / "LiDAR.tar.gz",
        dataset_folder / "Avea_data" / "LiDAR.tar.gz",
        dataset_folder / "Aeva_data" / "lidar.tar.gz",
        dataset_folder / "Avea_data" / "lidar.tar.gz",
    )
    _extract_if_needed(lidar_zip, dataset_folder / "Aeva_data")

    image_left_zip = _first_existing_path(
        dataset_folder / "Image" / "stereo_left.tar.gz",
        dataset_folder / "Image" / "Stereo_left.tar.gz",
    )
    _extract_if_needed(image_left_zip, dataset_folder / "Image")

    image_right_zip = _first_existing_path(
        dataset_folder / "Image" / "stereo_right.tar.gz",
        dataset_folder / "Image" / "Stereo_right.tar.gz",
    )
    _extract_if_needed(image_right_zip, dataset_folder / "Image")

    # Paths
    lidar_folder = _first_existing_path(
        dataset_folder / "Aeva_data" / "LiDAR" / "Aeva",
        dataset_folder / "Avea_data" / "LiDAR" / "Aeva",
    )
    left_img_folder = dataset_folder / "Image" / "stereo_left"
    right_img_folder = dataset_folder / "Image" / "stereo_right"
    calib_folder = dataset_folder / "Calibration"

    # --- Calibration ---
    stereo_left_intr, _ = _load_intrinsics_yaml(calib_folder / "stereo_left.yaml")
    stereo_right_intr, _ = _load_intrinsics_yaml(calib_folder / "stereo_right.yaml")
    lidar_to_left_ext, lidar_to_right_ext = _load_extrinsics_txt(calib_folder / "stereo_lidar.txt")

    # --- Images ---
    left_images = _glob_images(left_img_folder)
    right_images = _glob_images(right_img_folder)

    left_stamps = [s for s in (_int_stem(p) for p in left_images) if s is not None]
    right_stamps = [s for s in (_int_stem(p) for p in right_images) if s is not None]
    left_dict = {int(p.stem): p for p in left_images if _int_stem(p) is not None}
    right_dict = {int(p.stem): p for p in right_images if _int_stem(p) is not None}

    # --- LiDAR files ---
    bin_files = sorted(lidar_folder.glob("*.bin"))

    print(f" [DATASET] Found {len(bin_files)} LiDAR files, {len(left_images)} left images, {len(right_images)} right images in {dataset_folder.name}")

    process_fn = partial(
        _process_hercules_bin,
        return_all_fields=return_all_fields,
        left_stamps=left_stamps,
        right_stamps=right_stamps,
        left_dict=left_dict,
        right_dict=right_dict,
        stereo_left_intr=stereo_left_intr,
        stereo_right_intr=stereo_right_intr,
        lidar_to_left_ext=lidar_to_left_ext,
        lidar_to_right_ext=lidar_to_right_ext,
    )

    # --- Parallel execution ---
    paired_samples: list[dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        for result in tqdm(exe.map(process_fn, bin_files), total=len(bin_files), desc="Loading & pairing", unit="file"):
            if result is not None:
                paired_samples.append(result)

    return paired_samples

def load_scantinel_dataset_folder(dataset_folder: Path):
    """
    Load paired LiDAR and Camera data from Scantinel dataset folder.

    Args:
        dataset_folder (Path): Path to the dataset root.

    Returns:
        list of dicts: Each dict contains 'pointcloud', 'image', 'timestamps', and 'intrinsics'.
    """
    import tqdm as _tqdm  # local alias to avoid shadowing

    # Find all FMCW and Camera MCAP files
    lidar_files = sorted(dataset_folder.glob("*FMCW.mcap"))
    cam_files = sorted(dataset_folder.glob("*CAM.mcap"))

    if not lidar_files or not cam_files:
        raise ValueError(f"No LiDAR or camera files found in the dataset folder: {dataset_folder}")

    # Read all LiDAR messages
    lidar_data = []
    for lidar_file in _tqdm.tqdm(lidar_files, desc="Loading LiDAR"):
        msgs = read_mcap_file(lidar_file, ["/FMCW_pointclouds[0]"])
        lidar_data.extend((msg.proto_msg.data, msg.log_time) for msg in msgs)

    # Read all Camera messages
    cam_data = []
    for cam_file in _tqdm.tqdm(cam_files, desc="Loading Camera"):
        msgs = read_mcap_file(cam_file, ["/camera"])
        cam_data.extend((msg.proto_msg.data, msg.log_time) for msg in msgs)

    cam_stamps = [msg[1] for msg in cam_data]

    intrinsics = np.array([[640, 0, 320], [0, 480, 240], [0, 0, 1]], dtype=np.float32)

    paired_samples = []
    for pcl in _tqdm.tqdm(lidar_data, desc="Pairing LiDAR and Camera data"):
        # Decode LiDAR
        pointcloud, ts_lidar = pcl
        pointcloud = parse_pcl(pointcloud, point_stride=40, dtype=np.float32, num_fields=10)

        ts_cam = find_closest(cam_stamps, ts_lidar)
        cam = next((msg for msg in cam_data if msg[1] == ts_cam), None)

        sample = {
            "pointcloud": pointcloud,  # shape: (N, 6)
            "image": io.BytesIO(cam[0]),
            "intrinsics": intrinsics,
            "timestamps": [ts_lidar.timestamp(), ts_cam.timestamp()],
        }
        paired_samples.append(sample)

    return paired_samples


if __name__ == "__main__":
    hercules_f = Path("data/hercules/Mountain_01_Day")
    data = load_hercules_dataset_folder(hercules_f, return_all_fields=True)
    print(f"Loaded {len(data)} paired samples from {hercules_f}")
