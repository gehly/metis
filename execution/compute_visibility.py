import numpy as np
from datetime import datetime, timedelta
import sys
import os

#from skyfield.api import Loader, utc

sys.path.append('../')

from utilities.constants import Re
from utilities.tle_functions import launch2tle, tletime2datetime
from sensors.visibility_functions import compute_visible_passes
from sensors.visibility_functions import generate_visibility_file
from sensors.sensors import define_sensors


if __name__ == '__main__':
    
    cwd = os.getcwd()
    metis_ind = cwd.find('metis')
    metis_dir = cwd[0:metis_ind+6]
#    load = Loader(os.path.join(metis_dir, 'skyfield_data'))
#    ephemeris = load('de430t.bsp')
#    ts = load.timescale()
            
    
#    obj_id_list = [24870, 24871, 24873]
    
#    tle_dict = {}
#    tle_dict[25544] = {}
#    tle_dict[25544]['line1_list'] = ['1 25544U 98067A   18260.18078571  .00001804  00000-0  34815-4 0  9992']
#    tle_dict[25544]['line2_list'] = ['2 25544  51.6412 280.2571 0004861 165.0174 287.2188 15.53860328132822']
#    UTC = tletime2datetime(tle_dict[25544]['line1_list'][0])
    
#    UTC = datetime(2021, 3, 20, 12, 0, 0)
    
#    sensor_id_list = ['UNSW Falcon', 'UNSW Viper', 'NJC Falcon', 'CMU Falcon', 'RMIT ROO']
    
    
    # Times for visibility check
#    ndays = 3
#    dt = 10
#    UTC0 = ts.utc(UTC.replace(tzinfo=utc)).utc
#    print(UTC0)
#    sec_array = list(range(0,86400*ndays,dt))
#    skyfield_times = ts.utc(UTC0[0], UTC0[1]+2, UTC0[2],
#                            UTC0[3], UTC0[4], sec_array)
#    
#    print(skyfield_times[0])
#    mistake
    
    
#    obj_id_list = [45727, 47967, 29648, 46113, 24846, 23528, 26624, 35491]
    # obj_id_list = [47967]
    # obj_id_list = [37379, 41240, 45551, 20580, 36585, 40697]
    obj_id_list = [41240, 37379]
    sensor_id_list = ['UNSW Falcon', 'CMU Falcon']
    
    # sensor_id_list = ['CMU Falcon']
    # obj_id_list = [37379]
    
    UTC0 = datetime(2024, 5, 8, 12, 0, 0)
    delta_t = 1.*86400.
    dt = 1.
    UTC_list = [UTC0 + timedelta(seconds=ti) for ti in list(np.arange(0, delta_t, dt))]
    
    sensor_dict = define_sensors(sensor_id_list)
    vis_dict, rso_dict = compute_visible_passes(UTC_list, obj_id_list, sensor_dict)
                                                
    
    
#    print(vis_dict)

    
    # Generate output file
    vis_file_min_el = 10.
#    outdir = os.path.join(metis_dir, 'skyfield_data')
#    vis_file = os.path.join(outdir, 'test_visible_passes.csv')

    outdir = r'C:\Users\sgehly\Documents\students\masters\bas_andriessen\data'
    vis_file = os.path.join(outdir, 'bas_test_case_visible_passes_2024_05_08_1sec.csv')
    #generate_visibility_file(vis_dict, vis_file, vis_file_min_el)
    generate_visibility_file(vis_dict, rso_dict, UTC_list, outdir, vis_file, 
                             vis_file_min_el)
    
    
    
    
    
    
    