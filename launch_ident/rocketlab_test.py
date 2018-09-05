import numpy as np
from datetime import datetime
import sys
import os
from pathlib import Path

sys.path.append('../')

cwd = os.getcwd()
ind = cwd.find('metis')
metis_dir = Path(cwd[0:ind+5])

from skyfield.api import Loader, utc

from utilities.tle_functions import launchecef2tle, propagate_TLE, gcrf2tle
from utilities.tle_functions import tledict2dataframe, kep2tle
from utilities.constants import Re
from sensors.visibility_functions import compute_visible_passes
from sensors.visibility_functions import generate_visibility_file

if __name__ == '__main__':
    
    load = Loader(os.path.join(metis_dir, 'skyfield_data'))
    ephemeris = load('de430t.bsp')
    ts = load.timescale()
    
    # RocketLab launch data from June 2018
#    UTC = datetime(2018,6,23,4,10,21)
#    r_ITRF = 0.001*np.array([[-3651380.321], [1598487.431], [-5610448.359]])
#    v_ITRF = 0.001*np.array([[5276.523548], [-3242.081015], [-4349.310553]])
    
    # Rasit TLE (modified drag related terms)
#    line1 = '1 90001U 18999B   18178.09260417 +.00000000 +00000-0 +00000-0 0  9998'
#    line2 = '2 90001 084.9994 273.8351 0013325 288.2757 306.6473 15.21113008601617'
    
    # Compute TLE dictionary and dataframe at this time
    obj_id = 90001
    obj_id_list = [obj_id]
#    ecef_dict = {}
#    ecef_dict[obj_id] = {}
#    ecef_dict[obj_id]['UTC'] = UTC
#    ecef_dict[obj_id]['r_ITRF'] = r_ITRF
#    ecef_dict[obj_id]['v_ITRF'] = v_ITRF
#    tle_dict, tle_df = launchecef2tle(obj_id_list, ecef_dict)

#    tle_dict = {}
#    tle_dict[obj_id] = {}
#    tle_dict[obj_id]['line1_list'] = [line1]
#    tle_dict[obj_id]['line2_list'] = [line2]   
    
    UTC = datetime(2018, 11, 30, 4, 0, 0)
    kep_dict = {}
    kep_dict[obj_id] = {}
    kep_dict[obj_id]['UTC'] = UTC
#    kep_dict[obj_id]['a'] = 6880.619167272355
#    kep_dict[obj_id]['e'] = 0.0013325
#    kep_dict[obj_id]['i'] = 84.9994
#    kep_dict[obj_id]['RAAN'] = 273.8351
#    kep_dict[obj_id]['w'] = 288.2757
#    kep_dict[obj_id]['M'] = 306.6473
    
    kep_dict[obj_id]['a'] = Re + 500.
    kep_dict[obj_id]['e'] = 0.001
    kep_dict[obj_id]['i'] = 85.
    kep_dict[obj_id]['RAAN'] = 280.
    kep_dict[obj_id]['w'] = 300.
    kep_dict[obj_id]['M'] = 300.
    
    tle_dict, tle_df = kep2tle(obj_id_list, kep_dict)
    
    print(tle_dict)
    
    # Propagate to November
    
#    UTC_list = [UTC]
#    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)
#    r_GCRF = output_state[obj_id]['r_GCRF'][0]
#    v_GCRF = output_state[obj_id]['v_GCRF'][0]
#    
#    # Compute TLE data for this time
#    line1, line2 = gcrf2tle(obj_id, r_GCRF, v_GCRF, UTC)
#
#    # Add to dictionary
#    tle_dict[obj_id] = {}
#    tle_dict[obj_id]['UTC_list'] = [UTC]
#    tle_dict[obj_id]['line1_list'] = [line1]
#    tle_dict[obj_id]['line2_list'] = [line2]
#    
#    print(tle_dict)
    
    # Generate pandas dataframe
#    tle_df = tledict2dataframe(tle_dict)
    
    sensor_id_list = ['Stromlo Optical', 'Zimmerwald Optical',
                      'Arequipa Optical', 'Haleakala Optical',
                      'Yarragadee Optical', 'RMIT ROO', 'UNSW Falcon',
                      'FLC Falcon', 'Mamalluca Falcon']
    
    # Times for visibility check
    ndays = 3
    dt = 10
    UTC0 = ts.utc(UTC.replace(tzinfo=utc)).utc
    sec_array = list(range(0,86400*ndays,dt))
    skyfield_times = ts.utc(UTC0[0], UTC0[1], UTC0[2],
                            UTC0[3], UTC0[4], sec_array)
    
    vis_dict = compute_visible_passes(skyfield_times, obj_id_list,
                                      sensor_id_list, ephemeris, tle_dict)
    
    
    print(vis_dict)
    
    
    # Generate output file
    vis_file_min_el = 10.
    outdir = os.getcwd()
    vis_file = os.path.join(outdir, 'rocketlab_visible_passes.csv')
    generate_visibility_file(vis_dict, vis_file, vis_file_min_el)
