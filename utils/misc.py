
from bisect import bisect_left
import logging

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

    