#############################################################
# Temporal Data Processing Functions                         #
# Transforms a paticular timestamp to a fixed descrete time  #
# This will represent the interval from the reference point  #
##############################################################


from datetime import datetime, timedelta

def convert_to_fixed_intervals(timestamp, reference_point = '2000-01-01 00:00:00', task = 'Hours'):
    """
    Converts a timestamp to a discrete value using fixed minute intervals.
    
    Args:
        timestamp (str): The original timestamp (format: 'YYYY-MM-DD HH:MM:SS').
        reference_point (str): The reference timestamp for calculation.
        interval_minutes (int): The size of each time interval in minutes.
    
    Returns:
        int: The discrete timestamp based on fixed intervals.
    """
    # Parse timestamps into datetime objects
    timestamp_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    reference_dt = datetime.strptime(reference_point, "%Y-%m-%d %H:%M:%S")
    
    # Calculate total minutes since the reference point
    if task == 'Hours':
        since_ref = int((timestamp_dt - reference_dt).total_seconds() / 3600)
    elif task == 'Minutes':
        # Calculate total minutes since the reference point
        since_ref = int((timestamp_dt - reference_dt).total_seconds() / 60)
    
    # Map to discrete intervals
    #discrete_time = int(minutes_since_ref // interval_minutes)
    return since_ref

def recover_from_fixed_intervals(discrete_time, reference_point = '2000-01-01 00:00:00', interval_minutes=1):
    """
    Recovers the original timestamp from a discrete value based on fixed intervals.
    
    Args:
        discrete_time (int): The discrete timestamp.
        reference_point (str): The reference timestamp for calculation.
        interval_minutes (int): The size of each time interval in minutes.
    
    Returns:
        str: The approximate recovered timestamp (format: 'YYYY-MM-DD HH:MM:SS').
    """
    # Parse the reference point into a datetime object
    reference_dt = datetime.strptime(reference_point, "%Y-%m-%d %H:%M:%S")
    
    # Calculate minutes from discrete time
    minutes_since_ref = discrete_time * interval_minutes
    
    # Add the minutes back to the reference datetime
    recovered_timestamp = reference_dt + timedelta(minutes=minutes_since_ref)
    return recovered_timestamp.strftime("%Y-%m-%d %H:%M:%S")

def calculate_midpoint(timestamp1, timestamp2, timestamp_format="%Y-%m-%d %H:%M:%S"):
    """
    Computes the midpoint between two timestamps.
    
    Args:
        timestamp1 (str): The first timestamp (format: 'YYYY-MM-DD HH:MM:SS').
        timestamp2 (str): The second timestamp (format: 'YYYY-MM-DD HH:MM:SS').
        timestamp_format (str): The format in which the timestamps are given. Default is '%Y-%m-%d %H:%M:%S'.
    
    Returns:
        str: The midpoint timestamp in the same format.
    """
    # Parse timestamps into datetime objects
    timestamp1_dt = datetime.strptime(timestamp1, timestamp_format)
    timestamp2_dt = datetime.strptime(timestamp2, timestamp_format)
    
    # Calculate the difference and find the midpoint
    delta = timestamp2_dt - timestamp1_dt
    midpoint = timestamp1_dt + delta / 2
    
    # Return the midpoint as a string
    return midpoint.strftime(timestamp_format)



if __name__ == "__main__":
    # Example Usage
    reference = "2000-01-01 00:00:00"
    # Example Usage
    timestamp1 = "2000-01-01 01:00:00"
    timestamp2 = "2000-01-02 02:20:00"
    midpoint = calculate_midpoint(timestamp1, timestamp2)
    print(f"Midpoint: {midpoint}")

    # Convert to discrete intervals
    discrete_time = convert_to_fixed_intervals(timestamp1, reference)
    print(f"Discrete Time: {discrete_time}")

    # Recover the original timestamp
    recovered_timestamp = recover_from_fixed_intervals(discrete_time, reference, interval_minutes=1)
    print(f"Recovered Timestamp: {recovered_timestamp}")