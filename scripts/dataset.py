from pathlib import Path
import numpy as np
import yaml
from PIL import Image
from hercules.aeva import load_aeva_bin
import tqdm
import io
from utils.files import read_mcap_file
from utils.misc import find_closest_stamp, find_closest_cam_stamp_to_aeva
from scantinel.parse_mcap_pcl import parse_pcl
import pandas as pd

def load_hercules_dataset_folder(dataset_folder: Path, return_all_fields=False):
    """
    Load the Hercules dataset from a specified folder.

    Args:
        dataset_folder (Path): Path to the dataset folder.
        return_all_fields (bool): If True, returns all fields from the LiDAR point cloud.   
    Returns:
        dict: A dictionary containing:
            - point_clouds: List of point clouds (numpy arrays).
            - point_cloud_paths: List of paths to the LiDAR binary files.
            - stereo_left_images: List of paths to left stereo images.
            - stereo_right_images: List of paths to right stereo images.
            - stereo_left_intrinsics: Intrinsic parameters of the left stereo camera.
            - stereo_right_intrinsics: Intrinsic parameters of the right stereo camera.
            - stereo_left_distortion: Distortion coefficients of the left stereo camera.
            - stereo_right_distortion: Distortion coefficients of the right stereo camera.
            - lidar_to_stereo_left_extrinsic: Extrinsic matrix from LiDAR to left stereo camera.
            - lidar_to_stereo_right_extrinsic: Extrinsic matrix from LiDAR to right stereo camera.
    """
    # Paths
    lidar_folder = dataset_folder / "Avea_data" / "LiDAR" / "Aeva"
    left_img_folder = dataset_folder / "Image" / "stereo_left"
    right_img_folder = dataset_folder / "Image" / "stereo_right"
    calib_folder = dataset_folder / "Calibration"
    sensor_data_folder = dataset_folder / "Sensor_data"

    # Load calibration intrinsics
    stereo_left_intr_path = calib_folder / "stereo_left.yaml"
    stereo_right_intr_path = calib_folder / "stereo_right.yaml"

    data_stamp = sensor_data_folder / "data_stamp.csv"
    if data_stamp.exists():
        data_stamp_df = pd.read_csv(data_stamp, header=None)
        data_stamp_df = data_stamp_df[data_stamp_df[1].isin(["aeva", "stereo_left", "stereo_right"])]
    else:
        data_stamp_df = None

    if stereo_left_intr_path.exists():
        with stereo_left_intr_path.open("r") as f:
            stereo_left_intr_data = f.readlines()
            stereo_left_intrinsic = stereo_left_intr_data[3].strip().replace("\t"," ").split(" ")
            stereo_left_intrinsic = np.array(stereo_left_intrinsic, dtype=np.float32)
            if stereo_left_intrinsic.size == 9:
                stereo_left_intrinsic = stereo_left_intrinsic.reshape(3, 3)
            stereo_left_distortion = stereo_left_intr_data[6].strip().replace("\t"," ").split(" ")
            stereo_left_distortion = np.array(stereo_left_distortion, dtype=np.float32)
    else:
        stereo_left_intrinsic = None
        stereo_left_distortion = None

    if stereo_right_intr_path.exists():
        with stereo_right_intr_path.open("r") as f:
            stereo_right_intr_data = f.readlines()
            stereo_right_intrinsic = stereo_right_intr_data[3].strip().replace("\t"," ").split(" ")
            stereo_right_intrinsic = np.array(stereo_right_intrinsic, dtype=np.float32)
            if stereo_right_intrinsic.size == 9:
                stereo_right_intrinsic = stereo_right_intrinsic.reshape(3, 3)
            stereo_right_distortion = stereo_right_intr_data[6].strip().replace("\t"," ").split(" ")
            stereo_right_distortion = np.array(stereo_right_distortion, dtype=np.float32)
    else:
        stereo_right_intrinsic = None
        stereo_right_distortion = None

    # Load extrinsic between LiDAR and stereo cameras from text file
    stereo_lidar_path = calib_folder / "stereo_lidar.txt"
    stereo_lidar_lines = stereo_lidar_path.open("r").read().splitlines()
    # second line: left camera extrinsic, fourth line: right camera extrinsic

    stereo_left_lidar = [float(x) for x in stereo_lidar_lines[1].split()[1:]]
    stereo_right_lidar = [float(x) for x in stereo_lidar_lines[4].split()[1:]]

    # Reshape into 3x4 matrices, then expand to 4x4 homogeneous if needed
    lidar_to_left_extrinsic = np.array(stereo_left_lidar, dtype=np.float32).reshape(3, 4)
    lidar_to_right_extrinsic = np.array(stereo_right_lidar, dtype=np.float32).reshape(3, 4)
    # Convert to 4x4
    lidar_to_left_extrinsic = np.vstack([lidar_to_left_extrinsic, np.array([0,0,0,1], dtype=np.float32)])
    lidar_to_right_extrinsic = np.vstack([lidar_to_right_extrinsic, np.array([0,0,0,1], dtype=np.float32)])

    # List data files
    bin_files = sorted(lidar_folder.glob("*.bin"))
    left_images = sorted(left_img_folder.glob("*.png")) if left_img_folder.exists() else []
    right_images = sorted(right_img_folder.glob("*.png")) if right_img_folder.exists() else []

    # Load point clouds
    paired_samples = []
    for bin_file in tqdm.tqdm(bin_files[:50], desc="Loading point clouds", unit="file", leave=False): ##########################################
        point_cloud = load_aeva_bin(bin_file, return_all_fields=return_all_fields)
        closest_left_image = find_closest_cam_stamp_to_aeva(int(bin_file.stem), data_stamp_df, sensor="stereo_left") if data_stamp_df is not None else None
        closest_right_image = find_closest_cam_stamp_to_aeva(int(bin_file.stem), data_stamp_df, sensor="stereo_right") if data_stamp_df is not None else None
        if closest_left_image is not None:
            left_image = next((img for img in left_images if img.stem == str(closest_left_image)), None)
        else:
            left_image = None
        if closest_right_image is not None:
            right_image = next((img for img in right_images if img.stem == str(closest_right_image)), None)
        else:
            right_image = None
        if point_cloud is None:
            continue
        
        paired_samples.append({
            "pointcloud": point_cloud,
            "left_image": left_image,
            "right_image": right_image,
            "timestamps": [int(bin_file.stem), int(closest_left_image), int(closest_right_image)],
            "stereo_left_intrinsics": stereo_left_intrinsic,
            "stereo_right_intrinsics": stereo_right_intrinsic,
            "lidar_to_stereo_left_extrinsic": lidar_to_left_extrinsic,
            "lidar_to_stereo_right_extrinsic": lidar_to_right_extrinsic,
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
    from datetime import datetime
    hercules_f = Path("data/raw/dataset1")

    data = load_hercules_dataset_folder(hercules_f, return_all_fields=True)

    x = [datetime.fromtimestamp(int(pcl.stem)/1000000000) for pcl in data["point_cloud_paths"]][:50]
    y = [datetime.fromtimestamp(int(img.stem)/1000000000) for img in data["stereo_left_images"][:len(x)]]

    from matplotlib import pyplot as plt

    # compare timestamps
    plt.figure(figsize=(10, 5))
    plt.plot(x, np.zeros_like(x), 'ro', label='LiDAR Timestamps')
    plt.plot(y, np.ones_like(y), 'bo', label='Camera Timestamps')
    plt.xlabel('Timestamp')
    plt.yticks([0, 1], ['LiDAR', 'Camera'])
    plt.title('LiDAR and Camera Timestamps Comparison')
    plt.legend()
    plt.grid()
    plt.show()