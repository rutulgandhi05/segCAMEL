import io
import logging
from pathlib import Path
from functools import partial, lru_cache
from concurrent.futures import ProcessPoolExecutor
import tarfile
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple, Any
from PIL import Image 
from hercules.aeva import load_aeva_bin
from utils.files import read_mcap_file
from scantinel.parse_mcap_pcl import parse_pcl
from utils.misc import find_closest, _resolve_default_workers

def _safe_K(default_hw=(640, 480)) -> np.ndarray:
    print("Using safe K")
    fx, fy = float(default_hw[0]), float(default_hw[1])
    cx, cy = float(default_hw[0]) / 2.0, float(default_hw[1]) / 2.0
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def _validate_intrinsics(K: np.ndarray) -> bool:
    if K.shape != (3, 3):
        return False
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return (fx > 0 and fy > 0 and cx > 0 and cy > 0 and fx < 10000 and fy < 10000)


@lru_cache(maxsize=32)
def _load_intrinsics_yaml_cached(file_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    return _load_intrinsics_yaml(Path(file_path))


def _load_intrinsics_yaml(file_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Very lightweight parser: expects numeric line for K on line ~3 and D on ~6.
    Returns (K, D). If not present/malformed -> (identity_K, None).
    """
    if not file_path.exists():
        print(f"Intrinsics file not found: {file_path}")
        return _safe_K(), None
    try:
        lines = file_path.read_text().splitlines()
        def _nums(s): return np.fromstring(s, sep="\t" if "\t" in s else " ", dtype=np.float32)
        K = _nums(lines[3]) if len(lines) > 3 else np.array([])
        K = K.reshape(3, 3) if K.size == 9 else _safe_K()
        D = _nums(lines[6]) if len(lines) > 6 else None
        return K.astype(np.float32), (D.astype(np.float32) if D is not None and D.size > 0 else None)
    except Exception:
        return _safe_K(), None

@lru_cache(maxsize=32)
def _load_extrinsics_txt_cached(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    return _load_extrinsics_txt(Path(file_path))


def _load_extrinsics_txt(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    I = np.eye(4, dtype=np.float32)
    if not file_path.exists():
        print(f"Extrinsics file not found: {file_path}")
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
            try:
                nums = [float(x) for x in vals.strip().split()]
            except ValueError:
                print(f"Invalid numbers in extrinsics line: {line}")
                continue
            if len(nums) != 12:
                continue
            mat = np.array(nums, dtype=np.float32).reshape(3, 4)
            M = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]]).astype(np.float32)
            det = np.linalg.det(M[:3, :3])
            if abs(det - 1.0) > 0.1:
                print(f"Suspicious rotation matrix determinant: {det}")
            if "lidar" in key_l and "left" in key_l:
                left = M
            elif "lidar" in key_l and "right" in key_l:
                right = M
        return (left if left is not None else I), (right if right is not None else I)
    except Exception as e:
        print(f"Failed to parse extrinsics from {file_path}: {e}")
        return I, I


@lru_cache(maxsize=64)
def _find_existing_path_cached(*candidates: str) -> Optional[Path]:
    for path_str in candidates:
        p = Path(path_str)
        if p.exists():
            return p
    return None


def _first_existing_path(*candidates: Path) -> Path:
    path_strs = tuple(str(p) for p in candidates)
    result = _find_existing_path_cached(*path_strs)
    return result if result else candidates[0]


def _glob_images(folder: Path, extensions=("*.png", "*.jpg", "*.jpeg")) -> List[Path]:
    if not folder.exists():
        return []
    imgs = []
    for ext in extensions:
        imgs.extend(sorted(folder.glob(ext)))
    return imgs


def _int_stem(p: Path) -> Optional[int]:
    try:
        return int(p.stem)
    except (ValueError, TypeError):
        return None


def _extract_if_needed(tar_path: Path, target_dir: Path, force: bool = False) -> bool:
    if not tar_path.exists():
        print(f"Archive not found: {tar_path}")
        return False
    if not force and target_dir.exists() and any(target_dir.iterdir()):
        return False
    print(f"Extracting {tar_path} to {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
        return True
    except Exception as e:
        print(f"Failed to extract {tar_path}: {e}")
        return False


def _safe_load_aeva_bin(bin_path: Path, return_all_fields: bool = True) -> np.ndarray:
    try:
        data = load_aeva_bin(bin_path, return_all_fields=return_all_fields)
        if isinstance(data, np.ndarray):
            if data.ndim == 2 and data.shape[1] >= 3:
                return data.astype(np.float32)
            elif data.ndim == 1:
                n_fields = 10 if return_all_fields else 3
                if data.size % n_fields == 0:
                    return data.reshape(-1, n_fields).astype(np.float32)
        elif isinstance(data, dict):
            if all(k in data for k in ['x', 'y', 'z']):
                xyz = np.stack([data['x'], data['y'], data['z']], axis=1)
                if return_all_fields:
                    extra = []
                    for k in ['reflectivity', 'velocity', 'intensity']:
                        if k in data:
                            extra.append(np.asarray(data[k]).reshape(-1, 1))
                    if extra:
                        return np.hstack([xyz] + extra).astype(np.float32)
                return xyz.astype(np.float32)
        elif isinstance(data, (tuple, list)) and len(data) > 0:
            arr = np.asarray(data[0], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr
    except Exception as e:
        print(f"Failed loading {bin_path}: {e}")
    n_cols = 10 if return_all_fields else 3
    return np.zeros((0, n_cols), dtype=np.float32)


def _process_hercules_bin(
    bin_path: Path,
    return_all_fields: bool,
    left_stamps: List[int],
    right_stamps: List[int],
    left_dict: Dict[int, Path],
    right_dict: Dict[int, Path],
    stereo_left_intr: np.ndarray,
    stereo_right_intr: np.ndarray,
    lidar_to_left_ext: np.ndarray,
    lidar_to_right_ext: np.ndarray,
) -> Optional[Dict]:
    pointcloud = load_aeva_bin(bin_path, return_all_fields)
    if pointcloud is None:
        return None

    ts = int(bin_path.stem)
    l_ts = find_closest(left_stamps, ts) if left_stamps else None
    r_ts = find_closest(right_stamps, ts) if right_stamps else None

    l_img = left_dict.get(l_ts)
    r_img = right_dict.get(r_ts)
    if l_img is None and r_img is None:
        return None

    return {
        "pointcloud": pointcloud,              
        "left_image": l_img,
        "right_image": r_img,
        "timestamps": [ts, l_ts, r_ts],
        "stereo_left_intrinsics":stereo_left_intr,
        "stereo_right_intrinsics": stereo_right_intr,
        "lidar_to_stereo_left_extrinsic": lidar_to_left_ext,
        "lidar_to_stereo_right_extrinsic": lidar_to_right_ext,
    }


def load_hercules_dataset_folder(
    dataset_folder: Path,
    return_all_fields: bool = False,
    max_workers: Optional[int] = None
) -> List[Dict]:
    dataset_folder = Path(dataset_folder)
    print(f"Loading Hercules dataset from {dataset_folder}")

    if max_workers is None:
        max_workers = _resolve_default_workers()
    max_workers = max(1, int(max_workers))

    extraction_paths = {
        'lidar': [
            dataset_folder / "Aeva_data" / "LiDAR.tar.gz",
            dataset_folder / "Avea_data" / "LiDAR.tar.gz",
            dataset_folder / "Aeva_data" / "lidar.tar.gz",
        ],
        'left': [
            dataset_folder / "Image" / "stereo_left.tar.gz",
            dataset_folder / "Image" / "Stereo_left.tar.gz",
        ],
        'right': [
            dataset_folder / "Image" / "stereo_right.tar.gz",
            dataset_folder / "Image" / "Stereo_right.tar.gz",
        ]
    }
    for paths in extraction_paths.values():
        archive = _first_existing_path(*paths)
        if archive.exists():
            _extract_if_needed(archive, archive.parent, force=True)

    lidar_folder = _first_existing_path(
        dataset_folder / "Aeva_data" / "LiDAR" / "Aeva",
        dataset_folder / "Avea_data" / "LiDAR" / "Aeva",
    )
    left_img_folder = dataset_folder / "Image" / "stereo_left"
    right_img_folder = dataset_folder / "Image" / "stereo_right"
    calib_folder = dataset_folder / "Calibration"

    stereo_left_yaml = _first_existing_path(
        calib_folder / "stereo_left.yaml",
        calib_folder / "Stereo_left.yaml",
        calib_folder / "Stereo_Left.yaml",
    )
    stereo_left_intr, _ = _load_intrinsics_yaml_cached(str(stereo_left_yaml))

    stereo_right_yaml = _first_existing_path(
        calib_folder / "stereo_right.yaml",
        calib_folder / "Stereo_right.yaml",
        calib_folder / "Stereo_Right.yaml",
    )
    stereo_right_intr, _ = _load_intrinsics_yaml_cached(str(stereo_right_yaml))

    stereo_lidar_txt = _first_existing_path(
        calib_folder / "stereo_lidar.txt",
        calib_folder / "Stereo_lidar.txt",
        calib_folder / "Stereo_Lidar.txt",
        calib_folder / "Stereo_LiDAR.txt",
    )
    lidar_to_left_ext, lidar_to_right_ext = _load_extrinsics_txt_cached(str(stereo_lidar_txt))

    left_images = _glob_images(left_img_folder)
    right_images = _glob_images(right_img_folder)

    left_stamps = [s for s in (_int_stem(p) for p in left_images) if s is not None]
    right_stamps = [s for s in (_int_stem(p) for p in right_images) if s is not None]
    left_dict = {_int_stem(p): p for p in left_images if _int_stem(p) is not None}
    right_dict = {_int_stem(p): p for p in right_images if _int_stem(p) is not None}

    bin_files = sorted(lidar_folder.glob("*.bin"))
    print(f"Found {len(bin_files)} LiDAR files, {len(left_images)} left images, {len(right_images)} right images")

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

    paired_samples: List[Dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_fn, bin_files))
        for result in tqdm(results, desc="Processing samples", unit="file"):
            if result is not None:
                paired_samples.append(result)

    print(f"Successfully paired {len(paired_samples)}/{len(bin_files)} samples")
    return paired_samples


def load_scantinel_dataset_folder(dataset_folder: Path) -> List[Dict]:
    print(f"Loading Scantinel dataset from {dataset_folder}")
    lidar_files = sorted(dataset_folder.glob("*FMCW.mcap"))
    cam_files = sorted(dataset_folder.glob("*CAM.mcap"))
    if not lidar_files or not cam_files:
        raise ValueError(f"No LiDAR or camera files found in {dataset_folder}")

    lidar_data = []
    for lidar_file in tqdm(lidar_files, desc="Loading LiDAR"):
        try:
            msgs = read_mcap_file(lidar_file, ["/FMCW_pointclouds[0]"])
            lidar_data.extend((msg.proto_msg.data, msg.log_time) for msg in msgs)
        except Exception as e:
            print(f"Failed to read {lidar_file}: {e}")

    cam_data = []
    for cam_file in tqdm(cam_files, desc="Loading Camera"):
        try:
            msgs = read_mcap_file(cam_file, ["/camera"])
            cam_data.extend((msg.proto_msg.data, msg.log_time) for msg in msgs)
        except Exception as e:
            print(f"Failed to read {cam_file}: {e}")

    cam_stamps = [msg[1] for msg in cam_data]
    intrinsics = np.array([[640, 0, 320], [0, 480, 240], [0, 0, 1]], dtype=np.float32)

    paired_samples = []
    for pcl in tqdm(lidar_data, desc="Pairing data"):
        try:
            pointcloud, ts_lidar = pcl
            pointcloud = parse_pcl(pointcloud, point_stride=40, dtype=np.float32, num_fields=10)
            ts_cam = find_closest(cam_stamps, ts_lidar)
            cam = next((msg for msg in cam_data if msg[1] == ts_cam), None)
            if cam is None:
                continue

            sample = {
                "pointcloud": pointcloud,
                "image": io.BytesIO(cam[0]),
                "intrinsics": intrinsics,
                "timestamps": [ts_lidar.timestamp(), ts_cam.timestamp()],
            }
            paired_samples.append(sample)
        except Exception as e:
            print(f"Failed to process sample: {e}")

    print(f"Successfully paired {len(paired_samples)} samples")
    return paired_samples


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    hercules_f = Path("data/hercules/Mountain_01_Day")
    if hercules_f.exists():
        data = load_hercules_dataset_folder(hercules_f, return_all_fields=True)
        print(f"Loaded {len(data)} paired samples from {hercules_f}")
