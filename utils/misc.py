
from bisect import bisect_left
import logging
import pandas as pd
import numpy as np
import os
import re

def find_closest(stamps, target):
        """Binary‐search nearest stamp."""
        i = bisect_left(stamps, target)
        if i == 0:
            return stamps[0]
        if i == len(stamps):
            return stamps[-1]
        before, after = stamps[i-1], stamps[i]
        return before if (target - before) <= (after - target) else after


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


def scale_intrinsics(K_orig: np.ndarray, orig_size: tuple, new_size: tuple):
    w_orig, h_orig = orig_size
    w_new, h_new = new_size
    scale_x = w_new / w_orig
    scale_y = h_new / h_orig

    K_new = K_orig.copy()
    K_new[0, 0] *= scale_x   # fx
    K_new[1, 1] *= scale_y   # fy
    K_new[0, 2] *= scale_x   # cx
    K_new[1, 2] *= scale_y   # cy
    return K_new

def _resolve_default_workers() -> int:
    """Resolve a sensible default worker count based on the environment.

    This helper inspects Slurm environment variables to infer the number of
    CPU cores assigned to the current job.  If such variables are not set or
    cannot be parsed, it falls back to the machine's CPU count.

    Returns:
        An integer >= 1 representing the maximum number of worker processes
        suitable for a :class:`~concurrent.futures.ProcessPoolExecutor`.
    """
    # Check common Slurm variables.  Some may be formatted like "32(x2)".
    slurm_vars = [
        "SLURM_CPUS_ON_NODE",
        "SLURM_JOB_CPUS_PER_NODE",
        "SLURM_CPUS_PER_TASK",
    ]
    for var in slurm_vars:
        value = os.environ.get(var)
        if value:
            match = re.match(r"(\d+)", str(value))
            if match:
                try:
                    n = int(match.group(1))
                    if n > 0:
                        return n
                except ValueError:
                    pass
    # Fallback to system CPU count
    return os.cpu_count() or 1