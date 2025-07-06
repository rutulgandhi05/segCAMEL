def find_closest_stamp(stamps, target_stamp):
    """
    Find the closest timestamp in a list of timestamps to a target timestamp.
    
    Args:
        stamps (list): List of timestamps.
        target_stamp (float): The target timestamp to find the closest match for.
        
    Returns:
        float: The closest timestamp to the target.
    """
    return min(stamps, key=lambda x: abs(x - target_stamp))