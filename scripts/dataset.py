import io

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from hercules.aeva import load_aeva_bin
from utils.files import read_mcap_file
from scantinel.parse_mcap_pcl import parse_pcl
from utils.misc import find_closest_stamp

def load_hercules_dataset_folder(dataset_folder: Path, return_all_fields=False):
    """
    Efficient loader for the Hercules dataset from a folder.
    Returns a list of dictionaries, each with all fields needed for further batching.
    """
    # Paths
    lidar_folder = dataset_folder / "Avea_data" / "LiDAR" / "Aeva"
    left_img_folder = dataset_folder / "Image" / "stereo_left"
    right_img_folder = dataset_folder / "Image" / "stereo_right"
    calib_folder = dataset_folder / "Calibration"
    #sensor_data_folder = dataset_folder / "Sensor_data"

    # Load calibration intrinsics
    def load_intrinsics(file_path):
        if file_path.exists():
            with file_path.open("r") as f:
                lines = f.readlines()
                intrinsic = np.array(lines[3].strip().replace("\t", " ").split(" "), dtype=np.float32)
                intrinsic = intrinsic.reshape(3, 3) if intrinsic.size == 9 else None
                distortion = np.array(lines[6].strip().replace("\t", " ").split(" "), dtype=np.float32)
        else:
            intrinsic, distortion = None, None
        return intrinsic, distortion

    stereo_left_intr, stereo_left_dist = load_intrinsics(calib_folder / "stereo_left.yaml")
    stereo_right_intr, stereo_right_dist = load_intrinsics(calib_folder / "stereo_right.yaml")

    # Extrinsics
    stereo_lidar_path = calib_folder / "stereo_lidar.txt"
    lines = stereo_lidar_path.read_text().splitlines()
    def get_extrinsic(line_idx):
        arr = np.array([float(x) for x in lines[line_idx].split()[1:]], dtype=np.float32).reshape(3, 4)
        return np.vstack([arr, np.array([0,0,0,1], dtype=np.float32)])

    lidar_to_left_ext = get_extrinsic(1)
    lidar_to_right_ext = get_extrinsic(4)

    # Files
    bin_files = sorted(lidar_folder.glob("*.bin"))
    left_images = sorted(left_img_folder.glob("*.png")) if left_img_folder.exists() else []
    right_images = sorted(right_img_folder.glob("*.png")) if right_img_folder.exists() else []
    left_stamps = [int(img.stem) for img in left_images]
    right_stamps = [int(img.stem) for img in right_images]

    # Load point clouds and pair with images
    paired_samples = []
    for bin_file in tqdm(bin_files, desc="Loading LiDAR files", unit="file", leave=False):
        point_cloud = load_aeva_bin(bin_file, return_all_fields=return_all_fields)
        if point_cloud is None:
            continue
        bin_stamp = int(bin_file.stem)
        left_stamp = find_closest_stamp(left_stamps, bin_stamp) if left_stamps else None
        right_stamp = find_closest_stamp(right_stamps, bin_stamp) if right_stamps else None
        left_image = next((img for img in left_images if int(img.stem) == left_stamp), None) if left_stamp is not None else None
        right_image = next((img for img in right_images if int(img.stem) == right_stamp), None) if right_stamp is not None else None
        if left_image is None and right_image is None:
            continue  # Skip if no matching images
        paired_samples.append({
            "pointcloud": point_cloud,
            "left_image": left_image,
            "right_image": right_image,
            "timestamps": [bin_stamp, left_stamp, right_stamp],
            "stereo_left_intrinsics": stereo_left_intr,
            "stereo_right_intrinsics": stereo_right_intr,
            "lidar_to_stereo_left_extrinsic": lidar_to_left_ext,
            "lidar_to_stereo_right_extrinsic": lidar_to_right_ext,
        })
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

        ts_cam = find_closest_stamp(cam_stamps, ts_lidar)
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
