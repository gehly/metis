import numpy as np
from datetime import datetime
import sys
import os

from skyfield.api import Loader, utc

sys.path.append('../')

from utilities.astrodynamics import launch2tle
from utilities.constants import Re
from skyfield_visibility import compute_visible_passes
from skyfield_visibility import generate_visibility_file


if __name__ == '__main__':
    
    cwd = os.getcwd()
    metis_dir = cwd[0:-7]
    load = Loader(os.path.join(metis_dir, 'skyfield_data'))
    ephemeris = load('de430t.bsp')
    ts = load.timescale()
    
    # Given orbit data
#    obj_id = 90001
#    launch_elem = {}
#    launch_elem['ra'] = Re + 505.
#    launch_elem['rp'] = Re + 500.
#    launch_elem['i'] = 97.3479
#    launch_elem['RAAN'] = 309.966
#    launch_elem['w'] = 0.
#    launch_elem['M'] = 0.
#    launch_elem['date'] = datetime(2018, 9, 1, 5, 0, 0)
    
    obj_id = 90002
    launch_elem = {}
    launch_elem['ra'] = Re + 505.
    launch_elem['rp'] = Re + 500.
    launch_elem['i'] = 97.6266
    launch_elem['RAAN'] = 317.567
    launch_elem['w'] = 0.
    launch_elem['M'] = 0.
    launch_elem['date'] = datetime(2018, 9, 30, 5, 0, 0)
        
    
    obj_id_list = [obj_id]
    launch_elem_list = [launch_elem]
    
    
    tle_dict = launch2tle(obj_id_list, launch_elem_list)
    
    sensor_id_list = ['PSU Falcon', 'NJC Falcon', 'FLC Falcon', 'OJC Falcon',
                      'UNSW Falcon', 'Perth Falcon']
    
    
    # Times for visibility check
    ndays = 90
    dt = 60
    UTC0 = ts.utc(launch_elem['date'].replace(tzinfo=utc)).utc
    sec_array = list(range(0,86400*ndays,dt))
    skyfield_times = ts.utc(UTC0[0], UTC0[1], UTC0[2],
                            UTC0[3], UTC0[4], sec_array)
    
    vis_dict = compute_visible_passes(skyfield_times, obj_id_list,
                                      sensor_id_list, ephemeris, tle_dict)
    
    
    print(vis_dict)
    
    
    # Generate output file
    vis_file_min_el = 10.
    outdir = os.path.join(metis_dir, 'skyfield_data')
    vis_file = os.path.join(outdir, 'C2_visible_passes.csv')
    generate_visibility_file(vis_dict, vis_file, vis_file_min_el)