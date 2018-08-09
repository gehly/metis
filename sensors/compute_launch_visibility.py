import numpy as np
from datetime import datetime
import sys
import os

from skyfield.api import Loader, utc

sys.path.append('../')

from utilities.tle_functions import launch2tle
from utilities.constants import Re
from visibility_functions import compute_visible_passes
from visibility_functions import generate_visibility_file


if __name__ == '__main__':
    
    cwd = os.getcwd()
    metis_dir = cwd[0:-7]
    load = Loader(os.path.join(metis_dir, 'skyfield_data'))
    ephemeris = load('de430t.bsp')
    ts = load.timescale()
    
    launch_elem_dict = {}
    
    # Given orbit data
    obj_id = 90001
    launch_elem_dict[obj_id] = {}
    launch_elem_dict[obj_id]['ra'] = Re + 505.
    launch_elem_dict[obj_id]['rp'] = Re + 500.
    launch_elem_dict[obj_id]['i'] = 97.3479
    launch_elem_dict[obj_id]['RAAN'] = 309.966
    launch_elem_dict[obj_id]['w'] = 0.
    launch_elem_dict[obj_id]['M'] = 0.
    launch_elem_dict[obj_id]['UTC'] = datetime(2018, 11, 30, 5, 0, 0)

    obj_id = 90002
    launch_elem_dict[obj_id] = {}
    launch_elem_dict[obj_id]['ra'] = Re + 505.
    launch_elem_dict[obj_id]['rp'] = Re + 500.
    launch_elem_dict[obj_id]['i'] = 97.6266
    launch_elem_dict[obj_id]['RAAN'] = 317.567
    launch_elem_dict[obj_id]['w'] = 0.
    launch_elem_dict[obj_id]['M'] = 0.
    launch_elem_dict[obj_id]['UTC'] = datetime(2018, 11, 30, 5, 0, 0)
        
    
    obj_id_list = [90001, 90002]
    
    
    tle_dict = launch2tle(obj_id_list, launch_elem_dict)
    
    sensor_id_list = ['UNSW Falcon', 'ADFA UHF Radio', 'Stromlo Laser']
    
    
    # Times for visibility check
    ndays = 3
    dt = 10
    UTC0 = ts.utc(launch_elem_dict[90001]['UTC'].replace(tzinfo=utc)).utc
    sec_array = list(range(0,86400*ndays,dt))
    skyfield_times = ts.utc(UTC0[0], UTC0[1], UTC0[2],
                            UTC0[3], UTC0[4], sec_array)
    
    vis_dict = compute_visible_passes(skyfield_times, obj_id_list,
                                      sensor_id_list, ephemeris, tle_dict)
    
    
    print(vis_dict)
    
    
    # Generate output file
    vis_file_min_el = 10.
    outdir = os.path.join(metis_dir, 'skyfield_data')
    vis_file = os.path.join(outdir, 'Fleet_visible_passes.csv')
    generate_visibility_file(vis_dict, vis_file, vis_file_min_el)
    
    
    
    
    
    
    