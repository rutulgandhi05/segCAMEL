
from bisect import bisect_left
import logging
import pandas as pd

def find_closest_stamp(stamps, target_stamp):
    """
    Find the closest timestamp in a list of timestamps to a target timestamp.
    
    Args:
        stamps (list): List of timestamps.
        target_stamp (float): The target timestamp to find the closest match for.
        
    Returns:
        float: The closest timestamp to the target.
    """

    i = bisect_left(stamps, target_stamp)
    #return min(stamps, key=lambda x: abs(x - target_stamp))
    return min(stamps[max(0, i-1): i+2], key=lambda t: abs(target_stamp - t))
    

def setup_logger(name="train"):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    return logger

def find_closest_cam_stamp_to_aeva(aeva_stamp: int, data_stamp: pd.DataFrame, sensor="stereo_left"):
    """
    Find the closest timestamp for a given Aeva timestamp in the DataFrame.
    
    Parameters:
        aeva_stamp (int): The Aeva timestamp to match.
        data_stamp (pd.DataFrame): DataFrame containing timestamps and sensor names.
        sensor (str): The sensor name to filter by (default is "stereo_left").
    
    Returns:
        int: The closest timestamp for the specified sensor.
    """
    df = data_stamp.copy()
    # Filter DataFrame for the specified sensor
    sensor_df = df[df[1] == sensor]
    
    if sensor_df.empty:
        return None  # No matching sensor data
    
    # Find the closest timestamp
    closest_row = sensor_df.iloc[(sensor_df[0] - aeva_stamp).abs().argsort()[:1]]
    return closest_row[0].values[0]
