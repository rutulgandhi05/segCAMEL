import numpy as np

def parse_pcl(data_bytes: bytes, point_stride=40, dtype=np.float32, num_fields=10):
    """
    Converts raw LiDAR bytes into a numpy array of shape [N, num_fields].

    Args:
        data_bytes (bytes): Raw binary point cloud data.
        point_stride (int): Bytes per point (default 40).
        dtype: Numpy dtype of the fields (default float32).
        num_fields (int): Number of fields per point (default 10).

    Returns:
        np.ndarray: Array of shape (N, num_fields)
    """
    assert point_stride == num_fields * np.dtype(dtype).itemsize, \
        f"Stride {point_stride} doesn't match {num_fields} * {dtype}"
    
    raw_array = np.frombuffer(data_bytes, dtype=dtype)
    points = raw_array.reshape(-1, num_fields)
    return points