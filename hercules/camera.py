import cv2
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from misc.mcap_writer import write_to_mcap
from mcap_protobuf.reader import read_protobuf_messages as PBReader
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage as PBCompreesedImage
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration as PBCameraInfo


def read_stereo_image(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"File not found: {image_path}")
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    compressed_msg_data = encimg.tobytes() 

    return compressed_msg_data

def read_stereo_image_for_foxglove(image_path: Path, timestamp: int):
    """
    Reads a stereo image and constructs a Foxglove-compatible PointCloud message.
    
    Args:
      image_path (str): Path to the .bin file.
      timestamp (int, optional): Timestamp in nanoseconds used for selecting the record format.
    
    Returns:
      PB_PointCloud: A Foxglove-compatible PointCloud message.
    """
    compressed_msg_data = read_stereo_image(image_path)

    camera_info_msg = PBCameraInfo(**{
            "frame_id": "stereo_info",
            "height": 1080,
            "width": 1920,
            "distortion_model": 'plumb_bob',
            "timestamp": datetime.fromtimestamp(timestamp/1000000000)
        })

    msg = {
        "timestamp": datetime.fromtimestamp(timestamp/1000000000),  # Timestamp can be customized as needed.
        "frame_id": "stereo",
        "data": compressed_msg_data,
        "format": "png",
    }

    return [PBCompreesedImage(**msg), camera_info_msg]

def test():
    # Example usage

    fmcw_mcap_msgs_all = list(PBReader(source='data/1724813019217024258.mcap',  log_time_order=True, topics=['hercules/aeva']))
   
    fmcw_start_stamp = fmcw_mcap_msgs_all[0].log_time
    fmcw_end_stamp = fmcw_mcap_msgs_all[-1].log_time
    print(f"FMCW start: {fmcw_start_stamp}, end: {fmcw_end_stamp}")

    image_file_paths = list(Path("Z:\\01c_non-scantinel_sensor_data\\20250408_HeRCULES_dataset\\Mountain_01_Day\\Image\\stereo_left").glob("*.png"))
    image_file_paths = [image_path for image_path in tqdm(image_file_paths, desc='Filtering for relevant data.') if fmcw_start_stamp <= datetime.fromtimestamp(int(image_path.stem)/1000000000) <= fmcw_end_stamp]
    output_mcap_path = Path("data/1724813019217024258_stereo.mcap")
   
    foxglove_msgs = [(read_stereo_image_for_foxglove(image_path, timestamp=int(image_path.stem)), int(image_path.stem)) for image_path in tqdm(image_file_paths, desc="Gererating messages for MCAP")]

    write_to_mcap(foxglove_msgs, output_mcap_path, topic="hercules/stereo")
    print(f"Converted to {output_mcap_path}")


if __name__ == '__main__':
    test()