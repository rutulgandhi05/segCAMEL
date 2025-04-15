from pathlib import Path
from tqdm import tqdm
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud as PB_PointCloud
from mcap_protobuf.writer import Writer

def write_to_mcap(foxglove_msgs: list[tuple[PB_PointCloud, int]], output_mcap_path: Path, topic="/hercules/"):
    """
    Converts a .bin file to a Foxglove point cloud message and writes it to an MCAP file.
    This function repeats the same frame for num_frames with a given frame interval.
    
    Args:
      bin_paths (str): List of paths to the .bin files.
      output_mcap_path (str): Output MCAP file path.
    """
    try:
        with open(output_mcap_path, "wb") as f, Writer(f) as mcap_writer:
            for msg in tqdm(foxglove_msgs, desc="Writing to MCAP", unit="message"):
                if not isinstance(msg[0], PB_PointCloud):
                    stero_msg, camera_info_msg = msg[0]
                    mcap_writer.write_message(
                        topic=topic,
                        message=stero_msg,
                        log_time=msg[1],  # Use the timestamp from the filename.
                    )
                    mcap_writer.write_message(
                        topic=topic + "/camera_info",
                        message=camera_info_msg,
                        log_time=msg[1],  # Use the timestamp from the filename.
                    )
                else:
                    mcap_writer.write_message(
                        topic=topic,
                        message=msg[0],
                        log_time=msg[1],  # Use the timestamp from the filename.
                    )
    except Exception as e:
        raise ValueError(f"Error writing to MCAP: {e}")