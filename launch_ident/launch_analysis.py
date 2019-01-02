import numpy as np
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

sys.path.append('../')

cwd = os.getcwd()
ind = cwd.find('metis')
metis_dir = Path(cwd[0:ind+5])

from skyfield.api import Loader, utc

from utilities.tle_functions import gcrf2tle, launchecef2tle, kep2tle, tletime2datetime
from utilities.constants import Re
from utilities.astrodynamics import sunsynch_RAAN, sunsynch_inclination, element_conversion
from sensors.visibility_functions import compute_visible_passes
from sensors.visibility_functions import generate_visibility_file



def pslv_analysis():
    
    obj_id = 90000

    # Using Lawrence orbit parameters
    h = 505.
    a = Re + h
    e = 1e-4
    
    LTAN = 10.
    launch_dt = datetime(2018, 11, 19, 16, 30, 0)
    
    RAAN = sunsynch_RAAN(launch_dt, LTAN)
    i = sunsynch_inclination(a, e)
    
    w = 0.
    M = 180.
    
    kep_dict = {}
    kep_dict[obj_id] = {}
    kep_dict[obj_id]['a'] = a
    kep_dict[obj_id]['e'] = e
    kep_dict[obj_id]['i'] = i
    kep_dict[obj_id]['RAAN'] = RAAN
    kep_dict[obj_id]['w'] = w
    kep_dict[obj_id]['M'] = M
    kep_dict[obj_id]['UTC'] = launch_dt
    
    tle_dict, tle_df = kep2tle([obj_id], kep_dict)
    
    print(tle_dict)
    
    print(a,e,i,RAAN,w,M)
    
    UTC = datetime(2018, 11, 19, 16, 30, 0)
    
    load = Loader(os.path.join(metis_dir, 'skyfield_data'))
    ephemeris = load('de430t.bsp')
    ts = load.timescale()
    
    sensor_id_list = ['Stromlo Optical', 'RMIT ROO', 'UNSW Falcon',
                      'FLC Falcon', 'Mamalluca Falcon']
    
    # Times for visibility check
    ndays = 14
    dt = 10
    obj_id_list = [obj_id]
    UTC0 = ts.utc(UTC.replace(tzinfo=utc)).utc
    sec_array = list(range(0,86400*ndays,dt))
    skyfield_times = ts.utc(UTC0[0], UTC0[1], UTC0[2],
                            UTC0[3], UTC0[4], sec_array)
    
    vis_dict = compute_visible_passes(skyfield_times, obj_id_list,
                                      sensor_id_list, ephemeris, tle_dict)
    
    
    print(vis_dict)
    
    
    # Generate output file
    vis_file_min_el = 0.
    outdir = os.getcwd()
    vis_file = os.path.join(outdir, 'PSLV_visible_passes.csv')
    generate_visibility_file(vis_dict, vis_file, vis_file_min_el)
    
    return



def rocketlab_analysis():
    
#    # Using Thomas excel file
#    obj_id = 90000
#    UTC = datetime(2018, 6, 23, 2, 13, 21)   	
#    
#    ecef_dict = {}   
#    ecef_dict[obj_id] = {}
#    
#    r_ITRF = np.array([-3651380.321,	1598487.431,	-5610448.359]) * 0.001
#    v_ITRF = np.array([5276.523548, 	-3242.081015,	-4349.310553]) * 0.001
#    
#    ecef_dict[obj_id]['r_ITRF'] = r_ITRF.reshape(3,1)
#    ecef_dict[obj_id]['v_ITRF'] = v_ITRF.reshape(3,1)
#    ecef_dict[obj_id]['UTC'] = UTC
    
    # Using Rasit TLE
#    obj_id_list = [43690, 43691, 43692, 43693, 43694, 43695, 43696, 43697]
    obj_id_list = [43166, 43164, 43690, 43691, 43692, 43693, 43694, 43695, 43696, 43697]
#    line1 = '1 99999U 18999B   18315.16116898 +.00000500 +00000-0 +32002-2 0  9993'
#    line2 = '2 99999 085.0168 090.4036 0012411 292.8392 108.1000 15.20833469601616'
#    
#    line1 = '1 99999U 18999B   18315.19693718 +.00000500 +00000-0 +60940-2 0  9999'
#    line2 = '2 99999 085.0165 102.9279 0012642 291.6624 115.0006 15.20806704601617'
    
#    line1 = '1 43690U 18088A   18315.20213355  .00000372 -11738-5  00000+0 0  9993'
#    line2 = '2 43690  85.0339 102.9499 0224293 222.7416 214.1638 15.71130100    06'
#
#    UTC = tletime2datetime(line1)
#    tle_dict = {}
#    tle_dict[obj_id] = {}
#    tle_dict[obj_id]['line1_list'] = [line1]
#    tle_dict[obj_id]['line2_list'] = [line2]
#    tle_dict[obj_id]['UTC_list'] = [UTC]
    
    
    UTC = datetime(2018, 11, 29, 0, 0, 0)
    
    load = Loader(os.path.join(metis_dir, 'skyfield_data'))
    ephemeris = load('de430t.bsp')
    ts = load.timescale()
    
    sensor_id_list = ['RMIT ROO', 'UNSW Falcon', 'USAFA Falcon', 'Mamalluca Falcon']
    
    # Times for visibility check
    ndays = 5
    dt = 10
    UTC0 = ts.utc(UTC.replace(tzinfo=utc)).utc
    sec_array = list(range(0,86400*ndays,dt))
    skyfield_times = ts.utc(UTC0[0], UTC0[1], UTC0[2],
                            UTC0[3], UTC0[4], sec_array)
    
    vis_dict = compute_visible_passes(skyfield_times, obj_id_list,
                                      sensor_id_list, ephemeris)
    
    
    print(vis_dict)
    
    
    # Generate output file
    vis_file_min_el = 0.
    outdir = os.getcwd()
    vis_file = os.path.join(outdir, 'RocketLab_visible_passes_2018_11_29.csv')
    generate_visibility_file(vis_dict, vis_file, vis_file_min_el)
    
    return 
    

def spacex_ssoa_analysis():
    
#    # Using Thomas excel file
#    obj_id = 90000
#    UTC = datetime(2018, 6, 23, 2, 13, 21)   	
#    
#    ecef_dict = {}   
#    ecef_dict[obj_id] = {}
#    
#    r_ITRF = np.array([-3651380.321,	1598487.431,	-5610448.359]) * 0.001
#    v_ITRF = np.array([5276.523548, 	-3242.081015,	-4349.310553]) * 0.001
#    
#    ecef_dict[obj_id]['r_ITRF'] = r_ITRF.reshape(3,1)
#    ecef_dict[obj_id]['v_ITRF'] = v_ITRF.reshape(3,1)
#    ecef_dict[obj_id]['UTC'] = UTC
    
    # Using Rasit TLE
#    obj_id_list = [43690, 43691, 43692, 43693, 43694, 43695, 43696, 43697]
    obj_id_list = [43758, 43763]
#    line1 = '1 99999U 18999B   18315.16116898 +.00000500 +00000-0 +32002-2 0  9993'
#    line2 = '2 99999 085.0168 090.4036 0012411 292.8392 108.1000 15.20833469601616'
#    
#    line1 = '1 99999U 18999B   18315.19693718 +.00000500 +00000-0 +60940-2 0  9999'
#    line2 = '2 99999 085.0165 102.9279 0012642 291.6624 115.0006 15.20806704601617'
    
#    line1 = '1 43690U 18088A   18315.20213355  .00000372 -11738-5  00000+0 0  9993'
#    line2 = '2 43690  85.0339 102.9499 0224293 222.7416 214.1638 15.71130100    06'
#
#    UTC = tletime2datetime(line1)
#    tle_dict = {}
#    tle_dict[obj_id] = {}
#    tle_dict[obj_id]['line1_list'] = [line1]
#    tle_dict[obj_id]['line2_list'] = [line2]
#    tle_dict[obj_id]['UTC_list'] = [UTC]
    
    
    UTC = datetime(2018, 12, 3, 0, 0, 0)
    
    load = Loader(os.path.join(metis_dir, 'skyfield_data'))
    ephemeris = load('de430t.bsp')
    ts = load.timescale()
    
    sensor_id_list = ['RMIT ROO', 'UNSW Falcon', 'USAFA Falcon', 'Mamalluca Falcon']
    
    # Times for visibility check
    ndays = 5
    dt = 10
    UTC0 = ts.utc(UTC.replace(tzinfo=utc)).utc
    sec_array = list(range(0,86400*ndays,dt))
    skyfield_times = ts.utc(UTC0[0], UTC0[1], UTC0[2],
                            UTC0[3], UTC0[4], sec_array)
    
    vis_dict = compute_visible_passes(skyfield_times, obj_id_list,
                                      sensor_id_list, ephemeris)
    
    
    print(vis_dict)
    
    
    # Generate output file
    vis_file_min_el = 0.
    outdir = os.getcwd()
    vis_file = os.path.join(outdir, 'SSOA_visible_passes_2018_12_04.csv')
    generate_visibility_file(vis_dict, vis_file, vis_file_min_el)
    
    return 
    
    
    
    
if __name__ == '__main__':    
    
    
    spacex_ssoa_analysis()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    