import struct
import numpy as np

from pathlib import Path

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
    
    return point_size, raw_data, point_format


def load_aeva_bin(bin_path, return_all_fields=False):
    """
    Loads an Aeva .bin file and returns a dict of numpy arrays for each field, or just xyz.
    Args:
        bin_path: str or Path to .bin file
        return_all_fields: if True, returns dict of all fields as numpy arrays
    Returns:
        points: (N, 3) numpy array of x, y, z
        (optionally) fields: dict of all fields as numpy arrays
    """
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"File not found: {bin_path}")
    
    point_size, raw_data, point_format = read_aeva_bin(bin_path)

    if point_size == 29:
        field_names = ['x', 'y', 'z', 'reflectivity', 'velocity', 'time_offset_ns', 'line_index', 'intensity']
    elif point_size == 25:
        field_names = ['x', 'y', 'z', 'reflectivity', 'velocity', 'time_offset_ns', 'line_index']
    else:
        raise ValueError(f"Unexpected point size: {point_size}")
        

    n_points = len(raw_data) // point_size
    all_fields = {k: [] for k in field_names}
    unpack = struct.Struct(point_format).unpack_from

    for i in range(n_points):
        offset = i * point_size 
        vals = unpack(raw_data, offset)
        for k, v in zip(field_names, vals):
            all_fields[k].append(v)
            
    # Convert to numpy arrays
    for k in all_fields:
        all_fields[k] = np.asarray(all_fields[k], dtype=np.float32 if k != 'time_offset_ns' and k != 'line_index' else np.int32)
    
    all_fields_trimmed = np.stack([all_fields['x'], 
                           all_fields['y'], 
                           all_fields['z'], 
                           all_fields['reflectivity'], 
                           all_fields['velocity']], 
                           axis=1) if  "intensity" not in field_names else np.stack([all_fields['x'], 
                                                                                     all_fields['y'], 
                                                                                     all_fields['z'], 
                                                                                     all_fields['reflectivity'], 
                                                                                     all_fields['velocity'], 
                                                                                     all_fields['intensity']], 
                                                                                     axis=1)
    if return_all_fields:
        return all_fields_trimmed
    return np.stack([all_fields['x'], all_fields['y'], all_fields['z']], axis=1)
