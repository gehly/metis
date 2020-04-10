import numpy as np
from datetime import datetime
import sys
import os

sys.path.append('../')

from utilities.constants import Re
from sensors.visibility_functions import compute_transits


if __name__ == '__main__':
    
    cwd = os.getcwd()
    metis_ind = cwd.find('metis')
    metis_dir = cwd[0:metis_ind+6]
    
            
    # Object list
    obj_id_list = [25544, 40146]
    
    # Time window
    UTC0 = datetime(2019, 1, 1, 0, 0, 0)
    UTCf = datetime(2019, 1, 8, 0, 0, 0)
    UTC_window = [UTC0, UTCf]
    
    # Sensor data
    sensor_data_file = ''    
    
    # Generate transit dictionary
    transit_dict = compute_transits(UTC_window, obj_id_list, sensor_data_file)
    
    
    print(transit_dict)
    
    
    # Generate output file
    
    
    
    
    
    
    
    