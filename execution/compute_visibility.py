import numpy as np
from datetime import datetime
import sys
import os

from skyfield.api import Loader, utc

sys.path.append('../')

from utilities.constants import Re
from utilities.tle_functions import launch2tle, tletime2datetime
from sensors.visibility_functions import compute_visible_passes
from sensors.visibility_functions import generate_visibility_file


if __name__ == '__main__':
    
    cwd = os.getcwd()
    metis_ind = cwd.find('metis')
    metis_dir = cwd[0:metis_ind+6]
    load = Loader(os.path.join(metis_dir, 'skyfield_data'))
    ephemeris = load('de430t.bsp')
    ts = load.timescale()
            
    
    obj_id_list = [25544]
#    UTC = datetime(2018, 9, 15, 8, 0, 0)
    
    tle_dict = {}
    tle_dict[25544] = {}
    tle_dict[25544]['line1_list'] = ['1 25544U 98067A   18260.18078571  .00001804  00000-0  34815-4 0  9992']
    tle_dict[25544]['line2_list'] = ['2 25544  51.6412 280.2571 0004861 165.0174 287.2188 15.53860328132822']
    UTC = tletime2datetime(tle_dict[25544]['line1_list'][0])
    

    
    sensor_id_list = ['Stromlo Optical', 'Zimmerwald Optical',
                      'Arequipa Optical', 'Haleakala Optical',
                      'Yarragadee Optical']
    
    
    # Times for visibility check
    ndays = 7
    dt = 10
    UTC0 = ts.utc(UTC.replace(tzinfo=utc)).utc
    sec_array = list(range(0,86400*ndays,dt))
    skyfield_times = ts.utc(UTC0[0], UTC0[1]+2, UTC0[2],
                            UTC0[3], UTC0[4], sec_array)
    
    vis_dict = compute_visible_passes(skyfield_times, obj_id_list,
                                      sensor_id_list, ephemeris, tle_dict)
    
    
    print(vis_dict)
    
    
    # Generate output file
    vis_file_min_el = 10.
    outdir = os.path.join(metis_dir, 'skyfield_data')
    vis_file = os.path.join(outdir, 'iac_visible_passes2.csv')
    generate_visibility_file(vis_dict, vis_file, vis_file_min_el)
    
    
    
    
    
    
    