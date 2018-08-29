import numpy as np
from datetime import datetime
import sys
import os

from skyfield.api import Loader, utc

sys.path.append('../')

from utilities.constants import Re
from visibility_functions import compute_visible_passes
from visibility_functions import generate_visibility_file


if __name__ == '__main__':
    
    cwd = os.getcwd()
    metis_dir = cwd[0:-7]
    load = Loader(os.path.join(metis_dir, 'skyfield_data'))
    ephemeris = load('de430t.bsp')
    ts = load.timescale()
            
    
    obj_id_list = [25544]
    UTC = datetime(2018, 7, 12, 9, 0, 0)
    
    sensor_id_list = ['Stromlo Laser', 'Zimmerwald Laser',
                      'Arequipa Laser', 'Haleakala Laser']
    
    
    # Times for visibility check
    ndays = 3
    dt = 10
    UTC0 = ts.utc(UTC.replace(tzinfo=utc)).utc
    sec_array = list(range(0,86400*ndays,dt))
    skyfield_times = ts.utc(UTC0[0], UTC0[1], UTC0[2],
                            UTC0[3], UTC0[4], sec_array)
    
    vis_dict = compute_visible_passes(skyfield_times, obj_id_list,
                                      sensor_id_list, ephemeris)
    
    
    print(vis_dict)
    
    
    # Generate output file
    vis_file_min_el = 10.
    outdir = os.path.join(metis_dir, 'skyfield_data')
    vis_file = os.path.join(outdir, 'iac_visible_passes.csv')
    generate_visibility_file(vis_dict, vis_file, vis_file_min_el)
    
    
    
    
    
    
    