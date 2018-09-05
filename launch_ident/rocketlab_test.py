import numpy as np
from datetime import datetime
import sys

sys.path.append('../')

from utilities.tle_functions import launchecef2tle, propagate_TLE, gcrf2tle
from utilities.tle_functions import tledict2dataframe

if __name__ == '__main__':
    
    
    # RocketLab launch data from June 2018
    UTC = datetime(2018,6,23,4,10,21)
    r_ITRF = 0.001*np.array([[-3651380.321], [1598487.431], [-5610448.359]])
    v_ITRF = 0.001*np.array([[5276.523548], [-3242.081015], [-4349.310553]])
    
    # Compute TLE dictionary and dataframe at this time
    obj_id = 90001
    obj_id_list = [obj_id]
    ecef_dict = {}
    ecef_dict[obj_id] = {}
    ecef_dict[obj_id]['UTC'] = UTC
    ecef_dict[obj_id]['r_ITRF'] = r_ITRF
    ecef_dict[obj_id]['v_ITRF'] = v_ITRF
    tle_dict, tle_df = launchecef2tle(obj_id_list, ecef_dict)
    
    print(tle_dict)
    
    # Propagate to November
    UTC = datetime(2018, 11, 21, 4, 0, 0)
    UTC_list = [UTC]
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)
    r_GCRF = output_state[obj_id]['r_GCRF'][0]
    v_GCRF = output_state[obj_id]['v_GCRF'][0]
    
    # Compute TLE data for this time
    line1, line2 = gcrf2tle(obj_id, r_GCRF, v_GCRF, UTC)

    # Add to dictionary
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    # Generate pandas dataframe
    tle_df = tledict2dataframe(tle_dict)
    
    sensor_id_list = ['Stromlo Optical', 'Zimmerwald Optical',
                      'Arequipa Optical', 'Haleakala Optical',
                      'Yarragadee Optical', 'RMIT ROO', 'UNSW Falcon',
                      'FLC Falcon', 'Mamalluca Falcon']
    
    
    
    UTC2 = datetime(2018, 12, 9, 12, 0, 0)
    
    # Times for visibility check
    ndays = 3
    dt = 10
    UTC0 = ts.utc(launch_elem_dict[90003]['UTC'].replace(tzinfo=utc)).utc
    sec_array = list(range(0,86400*ndays,dt))
    skyfield_times = ts.utc(UTC0[0], UTC0[1], UTC0[2],
                            UTC0[3], UTC0[4], sec_array)
    
    vis_dict = compute_visible_passes(skyfield_times, obj_id_list,
                                      sensor_id_list, ephemeris, tle_dict)
    
    
    print(vis_dict)
    
    
    # Generate output file
    vis_file_min_el = 10.
    outdir = os.path.join(metis_dir, 'skyfield_data')
    vis_file = os.path.join(outdir, 'iac_visible_passes.csv')
    generate_visibility_file(vis_dict, vis_file, vis_file_min_el)
