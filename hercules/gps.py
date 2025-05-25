import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from utils.files import write_to_mcap
from mcap_protobuf.reader import read_protobuf_messages as PBReader
from google.protobuf.timestamp_pb2 import Timestamp as PBTimestamp
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix


def read_gps_csv(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    return pd.read_csv(csv_path, sep=",", header=None, names= ["timestamp", "latitude", "longitude", "altitude", "cov_0", "cov_1", "cov_2", "cov_3", "cov_4", "cov_5", "cov_6", "cov_7", "cov_8"])

def read_gps_csv_for_foxglove(csv_path: Path):
    """
    Reads a GPS CSV file and constructs Foxglove-compatible LocationFix messages.
    
    Args:
        csv_path (Path): Path to the GPS CSV file.
    
    Returns:
        list: A list of Foxglove-compatible LocationFix messages.
    """
    df = read_gps_csv(csv_path)
    location_fix_msgs = []

    for row in tqdm(df.itertuples(index=False), desc="Processing GPS data", total=len(df)):
        timestamp = row.timestamp
        latitude = row.latitude
        longitude = row.longitude
        altitude = row.altitude
        covariance = [row.cov_0, row.cov_1, row.cov_2, row.cov_3, row.cov_4, row.cov_5, row.cov_6, row.cov_7, row.cov_8]

        location_fix_msg = LocationFix(**{
            "frame_id": "gps",
            "timestamp": datetime.fromtimestamp(timestamp/1000000000),
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "position_covariance": covariance,
            "position_covariance_type": LocationFix.KNOWN,
        })
        
        location_fix_msgs.append((location_fix_msg, int(timestamp)))


    return location_fix_msgs


if __name__ == '__main__':
    # Example usage

    fmcw_mcap_msgs_all = list(PBReader(source='data/1724813019217024258_aeva.mcap',  log_time_order=True, topics=['hercules/aeva']))
    fmcw_start_stamp = fmcw_mcap_msgs_all[0].log_time
    fmcw_end_stamp = fmcw_mcap_msgs_all[-1].log_time
    print(f"FMCW start: {fmcw_start_stamp}, end: {fmcw_end_stamp}")

    output_mcap_path = Path("data/1724813019217024258_gps.mcap")

    foxglove_msgs = read_gps_csv_for_foxglove(Path("Z:\\01c_non-scantinel_sensor_data\\20250408_HeRCULES_dataset\\Mountain_01_Day\\Sensor_data\\gps.csv"))
    foxglove_msgs = [(msg, timestamp) for msg, timestamp in foxglove_msgs if fmcw_start_stamp <= datetime.fromtimestamp(timestamp/1000000000) <= fmcw_end_stamp]


    write_to_mcap(foxglove_msgs, output_mcap_path, topic="hercules/gps")
    print(f"Converted to {output_mcap_path}")