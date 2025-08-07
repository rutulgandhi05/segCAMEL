import io
import numpy as np

from pathlib import Path
from tqdm import tqdm
from hercules.aeva import load_aeva_bin
from utils.files import read_mcap_file
from scantinel.parse_mcap_pcl import parse_pcl
from utils.misc import find_closest
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils.misc import _resolve_default_workers

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

    return {
        "pointcloud": pc,
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
    Efficient loader for the Hercules dataset from a folder.
    Returns a list of dictionaries, each with all fields needed for further batching.
    """
    # Determine the number of workers to use for the ProcessPool
    if max_workers is None:
        max_workers = _resolve_default_workers()
    # Ensure at least one worker
    max_workers = max(1, int(max_workers))

    # Paths
    lidar_folder = dataset_folder / "Avea_data" / "LiDAR" / "Aeva"
    left_img_folder = dataset_folder / "Image" / "stereo_left"
    right_img_folder = dataset_folder / "Image" / "stereo_right"
    calib_folder = dataset_folder / "Calibration"

    # Load calibration intrinsics
    def load_intrinsics(file_path: Path):
        if not file_path.exists():
            return None, None
        lines = file_path.read_text().splitlines()
        K = np.fromstring(lines[3], sep="\t" if "\t" in lines[3] else " ", dtype=np.float32)
        K = K.reshape(3, 3) if K.size == 9 else None
        D = np.fromstring(lines[6], sep="\t" if "\t" in lines[6] else " ", dtype=np.float32)
        return K, D

    stereo_left_intr, _ = load_intrinsics(calib_folder / "stereo_left.yaml")
    stereo_right_intr, _ = load_intrinsics(calib_folder / "stereo_right.yaml")

    # Load extrinsics from text file.  Each extrinsic is stored as a 3x4 matrix
    # preceded by a label in the first column.  The helper returns a 4x4 matrix
    # in homogeneous coordinates.
    lines = (calib_folder / "stereo_lidar.txt").read_text().splitlines()
    def _get_ext(idx):
        mat = np.fromstring(" ".join(lines[idx].split()[1:]), sep=" ", dtype=np.float32).reshape(3, 4)
        return np.vstack([mat, [0,0,0,1]]).astype(np.float32)

    lidar_to_left_ext  = _get_ext(1)
    lidar_to_right_ext = _get_ext(4)

    left_images  = sorted(left_img_folder.glob("*.png"))  if left_img_folder.exists()  else []
    right_images = sorted(right_img_folder.glob("*.png")) if right_img_folder.exists() else []

    left_stamps = sorted(int(p.stem) for p in left_images)
    right_stamps = sorted(int(p.stem) for p in right_images)
    left_dict  = {int(p.stem): p for p in left_images}
    right_dict = {int(p.stem): p for p in right_images}

    # Files
    bin_files = sorted(lidar_folder.glob("*.bin"))
    
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

    # --- 6. Parallel execution ---
    # Parallel execution
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

    # Find all FMCW and Camera MCAP files
    lidar_files = sorted(dataset_folder.glob("*FMCW.mcap"))
    cam_files = sorted(dataset_folder.glob("*CAM.mcap"))

    if not lidar_files or not cam_files:
        raise ValueError(f"No LiDAR or camera files found in the dataset folder: {dataset_folder}")

    # Read all LiDAR messages
    lidar_data = []
    for lidar_file in tqdm.tqdm(lidar_files, desc="Loading LiDAR"):
        msgs = read_mcap_file(lidar_file, ["/FMCW_pointclouds[0]"])
        lidar_data.extend((msg.proto_msg.data, msg.log_time) for msg in msgs)

    # Read all Camera messages
    cam_data = []
    for cam_file in tqdm.tqdm(cam_files, desc="Loading Camera"):
        msgs = read_mcap_file(cam_file, ["/camera"])
        cam_data.extend((msg.proto_msg.data, msg.log_time) for msg in msgs)

    cam_stamps = [msg[1] for msg in cam_data]

    intrinsics = np.array([[640, 0, 320], [0, 480, 240], [0, 0, 1]], dtype=np.float32)

    paired_samples = []
    for pcl in tqdm.tqdm(lidar_data, desc="Pairing LiDAR and Camera data"):
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
