import numpy as np
from math import pi, asin, atan2
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
import pickle
import getpass
import os
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

import estimation.analysis_functions as analysis
import estimation.estimation_functions as est
import dynamics.dynamics_functions as dyn
import sensors.measurement_functions as mfunc
import sensors.sensors as sens
import sensors.visibility_functions as visfunc
import utilities.astrodynamics as astro
import utilities.coordinate_systems as coord
import utilities.eop_functions as eop
from utilities.constants import GME, arcsec2rad





def leo_tracklets_marco():
    
    username = input('space-track username: ')
    password = getpass.getpass('space-track password: ')
    
    
    obj_id = 53323
    sensor_id = 'Leiden Optical'
    orbit_regime = 'LEO'
    UTC0 = datetime(2022, 8, 2, 22, 56, 47)
    dt_interval = 0.5
    dt_max = 10.
    noise = 5.
    
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
    state_params['radius_m'] = 1.
    state_params['albedo'] = 0.1

    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = dyn.ode_twobody
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'
    
    # Sensor parameters
    sensor_id_list = [sensor_id]
    sensor_params = sens.define_sensors(sensor_id_list)
    sensor_params['eop_alldata'] = eop_alldata
    sensor_params['XYs_df'] = XYs_df
    
    for sensor_id in sensor_id_list:
        sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
        sigma_dict = {}
        sigma_dict['ra'] = noise*arcsec2rad   # rad
        sigma_dict['dec'] = noise*arcsec2rad  # rad
        sensor_params[sensor_id]['sigma_dict'] = sigma_dict
        
        
    
    params_dict = {}
    params_dict['sensor_params'] = sensor_params
    params_dict['state_params'] = state_params
    params_dict['int_params'] = int_params
    
    
    
    tracklet_dict = {}
    tracklet_dict = \
        mfunc.tracklet_generator(obj_id, UTC0, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, orbit_regime,
                                 username=username, password=password)
    
    
    UTC0 = datetime(2022, 8, 3, 23, 52, 55)
    dt_interval = 0.5
    dt_max = 10.
    tracklet_dict = \
        mfunc.tracklet_generator(obj_id, UTC0, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, orbit_regime,
                                 username=username, password=password)
        
    
    UTC0 = datetime(2022, 8, 6, 23, 34, 23)
    dt_interval = 0.5
    dt_max = 10.
    tracklet_dict = \
        mfunc.tracklet_generator(obj_id, UTC0, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, orbit_regime,
                                 username=username, password=password)

    
    UTC0 = datetime(2022, 8, 7, 22, 56, 57)
    dt_interval = 0.5
    dt_max = 30.
    tracklet_dict = \
        mfunc.tracklet_generator(obj_id, UTC0, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, orbit_regime,
                                 username=username, password=password)
        
        
    UTC0 = datetime(2022, 8, 9, 23, 15, 38)
    dt_interval = 0.5
    dt_max = 12.
    tracklet_dict = \
        mfunc.tracklet_generator(obj_id, UTC0, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, orbit_regime,
                                 username=username, password=password)
        
        
    UTC0 = datetime(2022, 8, 10, 22, 38, 8)
    dt_interval = 0.5
    dt_max = 10.
    tracklet_dict = \
        mfunc.tracklet_generator(obj_id, UTC0, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, orbit_regime,
                                 username=username, password=password)
        
        
    UTC0 = datetime(2022, 8, 12, 22, 56, 54)
    dt_interval = 0.5
    dt_max = 10.
    tracklet_dict = \
        mfunc.tracklet_generator(obj_id, UTC0, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, orbit_regime,
                                 username=username, password=password)
    
    
    
    print(tracklet_dict)
    
    setup_file = os.path.join('test_cases', 'twobody_leo_marco_noise5.pkl')
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [tracklet_dict, params_dict], pklFile, -1 )
    pklFile.close()
    
    
    
    return


if __name__ == '__main__':
    
    
    leo_tracklets_marco()