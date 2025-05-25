import toml

from tqdm import tqdm
from pathlib import Path   
from mcap_protobuf.writer import Writer
from datetime import timedelta, datetime
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud as PB_PointCloud

def read_toml_file(file_path: Path):
    """
    Reads a TOML file and returns its content as a dictionary.
    
    Args:
        file_path (str): Path to the TOML file.
        
    Returns:
        dict: Content of the TOML file.
    """
    try:
        with file_path.open("r") as f:
            data = toml.load(f)
        return data
    except Exception as e:
        print(f"Error reading TOML file: {e}")
        return None
    
    
def read_txt_file(file_path: Path):
    """
    Reads a text file and returns its content as a list of lines.
    
    Args:
        file_path (str): Path to the text file.
        
    Returns:
        list: List of lines in the text file.
    """
    try:
        with file_path.open("r") as f:
            lines = f.readlines()
        return [line.strip() for line in lines]
    except Exception as e:
        print(f"Error reading text file: {e}")
        return None
    

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
                if isinstance(msg[0], list):
                    stero_msg, camera_info_msg = msg[0]
                    mcap_writer.write_message(
                        topic=topic,
                        message=stero_msg,
                        log_time=msg[1],  
                    )
                    mcap_writer.write_message(
                        topic=topic + "/camera_info",
                        message=camera_info_msg,
                        log_time=msg[1],  
                    )
                else:
                    mcap_writer.write_message(
                        topic=topic,
                        message=msg[0],
                        log_time=msg[1], 
                    )
    except Exception as e:
        raise ValueError(f"Error writing to MCAP: {e}")
    

def get_chunks(data: list[Path]):
    if not data:
        return []
    
    result = []
    current_group = []
    start_time = None

    for name in tqdm(data, desc="Creating chunks..."):
        timestamp = datetime.fromtimestamp(int(name.stem) / 1000000000)
        current_time = timestamp
        if start_time is None:
            start_time = current_time
        
        if current_time - start_time <= timedelta(minutes=1):
            current_group.append(name)
        else:
            result.append(current_group)
            current_group = [name]
            start_time = current_time

    if current_group:
            result.append(current_group)

    return result