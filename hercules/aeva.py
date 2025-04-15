import struct

from pathlib import Path
from datetime import datetime
from misc.mcap_writer import write_to_mcap
from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField as PBfield
from foxglove_schemas_protobuf.Pose_pb2 import Pose as PB_Pose
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion as PB_Quaternion
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3 as PB_Vector3
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud as PB_PointCloud
from hercules.transforms import get_stereo_lidar_transforms_msgs

INTENSITY_THRESHOLD = 1691936557946849179

def read_aeva_bin(bin_path: Path):
    if not bin_path.exists():
        raise FileNotFoundError(f"File not found: {bin_path}")
    
    point_format = '<fffffiBf' if int(bin_path.stem) > INTENSITY_THRESHOLD else '<fffffiB'  
    point_size = struct.calcsize(point_format)

    with open(bin_path, 'rb') as f:
        raw_data = f.read()
    total_len = len(raw_data)
    remainder = total_len % point_size
    if remainder != 0:
        print(f"Warning: File contains {remainder} extra byte(s). Trimming extra bytes.")
        raw_data = raw_data[:total_len - remainder]
    
    return point_size, raw_data


def read_aeva_bin_for_foxglove(bin_path: Path, timestamp: int):
    """
    Reads a .bin file and constructs a Foxglove-compatible PointCloud message.
    It trims any extra incomplete bytes, ensuring the binary data length is a multiple
    of the expected record size.
    
    Args:
      bin_path (str): Path to the .bin file.
      timestamp (int, optional): Timestamp in nanoseconds used for selecting the record format.
    
    Returns:
      PB_PointCloud: A Foxglove-compatible PointCloud message.
    """
    point_size, raw_data = read_aeva_bin(bin_path)

    msg = {
        "timestamp": datetime.fromtimestamp(timestamp/1000000000),  # Timestamp can be customized as needed.
        "frame_id": "aeva",
        "pose": PB_Pose(**{
            "position": PB_Vector3(**{"x": 0.0, "y": 0.0, "z": 0.0}),
            "orientation": PB_Quaternion(**{"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})
        }),
        "point_stride": point_size,
        "fields": [
            PBfield(**{"name": "x", "offset": 0, "type": PBfield.FLOAT32}),
            PBfield(**{"name": "y", "offset": 4, "type": PBfield.FLOAT32}),
            PBfield(**{"name": "z", "offset": 8, "type": PBfield.FLOAT32}),
            PBfield(**{"name": "reflectivity", "offset": 12, "type": PBfield.FLOAT32}),
            PBfield(**{"name": "velocity", "offset": 16, "type": PBfield.FLOAT32}),
            PBfield(**{"name": "time_offset_ns", "offset": 20, "type": PBfield.INT32}),
            PBfield(**{"name": "line_index", "offset": 24, "type": PBfield.UINT8}),
            PBfield(**{"name": "intensity", "offset": 25, "type": PBfield.FLOAT32}),
        ],
        "data": raw_data,
    }
    return PB_PointCloud(**msg)



def test():
    # Example usage
    bin_file_paths = list(Path("data/aeva").glob("*.bin"))
    output_aeva_mcap_path = Path("data/1724813019217024258_aeva.mcap")
    output_transform_mcap_path = Path("data/1724813019217024258_aeva_transforms.mcap")

    transformantion_matrix_lcam = [[0.022023532009715, -0.999669977147658, 0.013225007652863, 0.254288466421238],
                            [-0.049191798363281, -0.014295738779819, -0.998687037477971, -0.108349681409513],
                            [0.998546509188032, 0.021344054027771, -0.049490406606293,  -0.136209414128968]
                            ]
    
    foxglove_msgs = [(read_aeva_bin_for_foxglove(bin_path, timestamp=int(bin_path.stem)), int(bin_path.stem)) for bin_path in bin_file_paths]
    transform_msgs = [get_stereo_lidar_transforms_msgs(int(bin_path.stem), transformantion_matrix_lcam=transformantion_matrix_lcam) for bin_path in bin_file_paths]
    all_transform_msgs = [msg for sublist in transform_msgs for msg in sublist]

    write_to_mcap(foxglove_msgs, output_aeva_mcap_path, topic="hercules/aeva")
    print(f"Converted Aeva data to {output_aeva_mcap_path}")

    write_to_mcap(all_transform_msgs, output_transform_mcap_path, topic="hercules/transform")
    print(f"Converted Aeva transform data to {output_transform_mcap_path}")

if __name__ == '__main__':
    test()