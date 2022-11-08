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

from iod import iod_functions as iod
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
from utilities import tle_functions as tle





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


def geo_tracklets():
    
    username = input('space-track username: ')
    password = getpass.getpass('space-track password: ')
    
    
    qzs1r_norad = 49336
    qzs2_norad = 42738
    qzs3_norad = 42917
    qzs4_norad = 42965
    
    ses15_norad = 42709
    amos5_norad = 37950
    coms1_norad = 36744
    
    
    # Common parameter setup
    tracklet_dict = {}    
    sensor_id = 'RMIT ROO'
    orbit_regime = 'GEO'    
    dt_interval = 10.
    dt_max = 600.
    noise = 0.    
    
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
        sensor_params[sensor_id]['moon_angle_lim'] = 0.
                
    
    params_dict = {}
    params_dict['sensor_params'] = sensor_params
    params_dict['state_params'] = state_params
    params_dict['int_params'] = int_params
       
    
    
    ###########################################################################
    # QZS1R
    ###########################################################################
    
    obj_id = qzs1r_norad
    UTC0 = datetime(2022, 11, 7, 11, 0, 0)
    state_dict = tle.propagate_TLE([obj_id], [UTC0], username=username,
                                   password=password)
    r0 = state_dict[obj_id]['r_GCRF'][0]
    v0 = state_dict[obj_id]['v_GCRF'][0]
    Xo = np.concatenate((r0, v0), axis=0)
    
    tracklet_dict = \
        mfunc.tracklet_generator(Xo, UTC0, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, orbit_regime)
    
    
    UTC = datetime(2022, 11, 8, 14, 0, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict = \
        mfunc.tracklet_generator(Xk, UTC, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 9, 16, 0, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict = \
        mfunc.tracklet_generator(Xk, UTC, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, orbit_regime)
        
        
    ###########################################################################
    # QZS2
    ###########################################################################
    
    obj_id = qzs2_norad
    UTC0 = datetime(2022, 11, 7, 11, 10, 0)
    state_dict = tle.propagate_TLE([obj_id], [UTC0], username=username,
                                   password=password)
    r0 = state_dict[obj_id]['r_GCRF'][0]
    v0 = state_dict[obj_id]['v_GCRF'][0]
    Xo = np.concatenate((r0, v0), axis=0)
    
    
    tracklet_dict = \
        mfunc.tracklet_generator(Xo, UTC0, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, orbit_regime)
    
    
    UTC = datetime(2022, 11, 8, 14, 10, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict = \
        mfunc.tracklet_generator(Xk, UTC, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 9, 16, 10, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict = \
        mfunc.tracklet_generator(Xk, UTC, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, orbit_regime)
        
    
    
    
    
    
    print(tracklet_dict)
    
    setup_file = os.path.join('test_cases', 'twobody_geo_10min_noise0.pkl')
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [tracklet_dict, params_dict], pklFile, -1 )
    pklFile.close()
    
    
    
    return


def tracklet_association():
    
    # Load data
    setup_file = os.path.join('test_cases', 'twobody_geo_10min_noise0.pkl')
    pklFile = open(setup_file, 'rb' )
    data = pickle.load( pklFile )
    tracklet_dict = data[0]
    params_dict = data[1]
    pklFile.close()
    
    
    tracklet1 = tracklet_dict[0]
    tracklet2 = tracklet_dict[1]
    
    tk_list = [tracklet1['tk_list'][0], tracklet1['tk_list'][-1], tracklet2['tk_list'][-1]]
    Yk_list = [tracklet1['Yk_list'][0], tracklet1['Yk_list'][-1], tracklet2['Yk_list'][-1]]
    sensor_id_list = [tracklet1['sensor_id_list'][0],
                      tracklet1['sensor_id_list'][-1],
                      tracklet2['sensor_id_list'][-1]]
    
    sensor_params = params_dict['sensor_params']
    orbit_regime = tracklet1['orbit_regime']
    
    print(tk_list)
    print(Yk_list)
    print(sensor_id_list)

    
    # Execute function
    X_list, M_list = iod.gooding_angles_iod(tk_list, Yk_list, sensor_id_list,
                                            sensor_params, orbit_regime=orbit_regime,
                                            search_mode='middle_out',
                                            periapsis_check=True,
                                            rootfind='zeros')
    
    
    print('Final Answers')
    print('X_list', X_list)
    print('M_list', M_list)
    
    for ii in range(len(M_list)):
        
        elem_ii = astro.cart2kep(X_list[ii])
        
        print('')
        print(ii)
        print('Mi', M_list[ii])
        print('Xi', X_list[ii])
        print('elem', elem_ii)
        
        
    # True orbits
    qzs1r_norad = 49336
    qzs2_norad = 42738
    qzs3_norad = 42917
    qzs4_norad = 42965
    
    ses15_norad = 42709
    amos5_norad = 37950
    coms1_norad = 36744
    
    
    obj_id = qzs1r_norad
    state_dict = tle.propagate_TLE([obj_id], tk_list)
    
    r0 = state_dict[obj_id]['r_GCRF'][0]
    v0 = state_dict[obj_id]['v_GCRF'][0]
    Xo_true = np.concatenate((r0, v0), axis=0)
    elem_true = astro.cart2kep(Xo_true)
    
    print('QZS-1R Elem Truth: ', elem_true)
    print('QZS-1R Xo Err: ', np.linalg.norm(X_list[0] - Xo_true))
    
    
    return


if __name__ == '__main__':
    
    
#    leo_tracklets_marco()
    
#    geo_tracklets()
    
    tracklet_association()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    