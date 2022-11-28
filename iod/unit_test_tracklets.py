import numpy as np
from math import pi, asin, atan2
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
import pickle
import getpass
import os
import inspect
import copy
import pandas as pd
import time

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from iod import iod_functions_jit as iod
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


def geo_tracklets(tracklets_file, noise):
    
    username = input('space-track username: ')
    password = getpass.getpass('space-track password: ')
    
    
    qzs1r_norad = 49336
    qzs2_norad = 42738
    qzs3_norad = 42917
    qzs4_norad = 42965
    
    amos5_norad = 37950
    coms1_norad = 36744
    
    
    # Common parameter setup
    tracklet_dict = {}   
    truth_dict = {}
    sensor_id = 'RMIT ROO'
    orbit_regime = 'GEO'    
    dt_interval = 10.
    dt_max = 600.
    
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
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xo, UTC0, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, truth_dict, orbit_regime)
    
    
    UTC = datetime(2022, 11, 8, 14, 0, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 9, 16, 0, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 10, 10, 0, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 11, 12, 0, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 12, 13, 0, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 13, 9, 0, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                 params_dict, tracklet_dict, truth_dict, orbit_regime)
        
        
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
    
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xo, UTC0, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
    
    
    UTC = datetime(2022, 11, 8, 14, 10, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 9, 16, 10, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 10, 10, 10, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 11, 12, 10, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 12, 13, 10, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 13, 9, 10, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    
    ###########################################################################
    # QZS3
    ###########################################################################
    
    obj_id = qzs3_norad
    UTC0 = datetime(2022, 11, 7, 11, 20, 0)
    state_dict = tle.propagate_TLE([obj_id], [UTC0], username=username,
                                    password=password)
    r0 = state_dict[obj_id]['r_GCRF'][0]
    v0 = state_dict[obj_id]['v_GCRF'][0]
    Xo = np.concatenate((r0, v0), axis=0)
    
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xo, UTC0, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
    
    
    UTC = datetime(2022, 11, 8, 14, 20, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 9, 16, 20, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 10, 10, 20, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 11, 12, 20, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 12, 13, 20, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 13, 9, 20, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    
    ###########################################################################
    # QZS4
    ###########################################################################
    
    obj_id = qzs4_norad
    UTC0 = datetime(2022, 11, 7, 11, 30, 0)
    state_dict = tle.propagate_TLE([obj_id], [UTC0], username=username,
                                    password=password)
    r0 = state_dict[obj_id]['r_GCRF'][0]
    v0 = state_dict[obj_id]['v_GCRF'][0]
    Xo = np.concatenate((r0, v0), axis=0)
    
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xo, UTC0, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
    
    
    UTC = datetime(2022, 11, 8, 14, 30, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 9, 16, 30, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 10, 10, 30, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 11, 12, 30, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 12, 13, 30, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 13, 9, 30, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
               
    ###########################################################################
    # AMOS-5
    ###########################################################################
    
    obj_id = amos5_norad
    UTC0 = datetime(2022, 11, 7, 11, 50, 0)
    state_dict = tle.propagate_TLE([obj_id], [UTC0], username=username,
                                    password=password)
    r0 = state_dict[obj_id]['r_GCRF'][0]
    v0 = state_dict[obj_id]['v_GCRF'][0]
    Xo = np.concatenate((r0, v0), axis=0)
    
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xo, UTC0, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
    
    
    UTC = datetime(2022, 11, 8, 14, 50, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 9, 16, 50, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 10, 10, 50, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 11, 12, 50, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 12, 13, 50, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 13, 9, 50, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    ###########################################################################
    # COMS-1
    ###########################################################################
    
    obj_id = coms1_norad
    UTC0 = datetime(2022, 11, 7, 12, 0, 0)
    state_dict = tle.propagate_TLE([obj_id], [UTC0], username=username,
                                    password=password)
    r0 = state_dict[obj_id]['r_GCRF'][0]
    v0 = state_dict[obj_id]['v_GCRF'][0]
    Xo = np.concatenate((r0, v0), axis=0)
    
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xo, UTC0, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
    
    
    UTC = datetime(2022, 11, 8, 15, 0, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 9, 17, 0, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 10, 11, 0, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 11, 13, 0, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 12, 14, 0, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
        
    UTC = datetime(2022, 11, 13, 10, 0, 0)
    tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
    Xk = Xout[-1,:].reshape(6,1)
    
    tracklet_dict, truth_dict = \
        mfunc.tracklet_generator(obj_id, Xk, UTC, dt_interval, dt_max, sensor_id,
                                  params_dict, tracklet_dict, truth_dict, orbit_regime)
    
    
    
    print(tracklet_dict)
    
    pklFile = open( tracklets_file, 'wb' )
    pickle.dump( [tracklet_dict, params_dict, truth_dict], pklFile, -1 )
    pklFile.close()
    
    
    
    return


def test_tracklet_association():
    
    # True orbits
    qzs1r_norad = 49336
    qzs2_norad = 42738
    qzs3_norad = 42917
    qzs4_norad = 42965
    
    amos5_norad = 37950
    coms1_norad = 36744
    
    
    # Load data
    setup_file = os.path.join('test_cases', 'twobody_geo_10min_noise0.pkl')
    pklFile = open(setup_file, 'rb' )
    data = pickle.load( pklFile )
    tracklet_dict = data[0]
    params_dict = data[1]
    pklFile.close()
    
    
    ###########################################################################
    # QZS-1R Self-Association
    ###########################################################################
    
    tracklet1 = tracklet_dict[0]
    tracklet2 = tracklet_dict[2]
    
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
    

    obj_id = qzs1r_norad
    state_dict = tle.propagate_TLE([obj_id], tk_list)
    
    r0 = state_dict[obj_id]['r_GCRF'][0]
    v0 = state_dict[obj_id]['v_GCRF'][0]
    Xo_true = np.concatenate((r0, v0), axis=0)
    elem_true = astro.cart2kep(Xo_true)
    
    print('QZS-1R Elem Truth: ', elem_true)
    
    
    
    print('Final Answers')
    print('X_list', X_list)
    print('M_list', M_list)
    
    for ii in range(len(M_list)):
        
        elem_ii = astro.cart2kep(X_list[ii])
        resids, ra_rms, dec_rms = \
            compute_resids(X_list[ii], tk_list[0], tracklet1, tracklet2,
                          params_dict)
        
        
        
        print('')
        print(ii)
        print('Mi', M_list[ii])
        print('Xi', X_list[ii])
        print('elem', elem_ii)
        print('QZS-1R Xo Err: ', np.linalg.norm(X_list[ii] - Xo_true))
        print('RA Resids RMS [arcsec]: ', ra_rms)
        print('DEC Resids RMS [arcsec]: ', dec_rms)
        
        

        

    ###########################################################################
    # QZS-2 Self-Association
    ###########################################################################
    
    tracklet1 = tracklet_dict[3]
    tracklet2 = tracklet_dict[5]
    
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
    

    obj_id = qzs2_norad
    state_dict = tle.propagate_TLE([obj_id], tk_list)
    
    r0 = state_dict[obj_id]['r_GCRF'][0]
    v0 = state_dict[obj_id]['v_GCRF'][0]
    Xo_true = np.concatenate((r0, v0), axis=0)
    elem_true = astro.cart2kep(Xo_true)
    
    print('QZS-2 Elem Truth: ', elem_true)
    
    
    
    print('Final Answers')
    print('X_list', X_list)
    print('M_list', M_list)
    
    for ii in range(len(M_list)):
        
        elem_ii = astro.cart2kep(X_list[ii])
        resids, ra_rms, dec_rms = \
            compute_resids(X_list[ii], tk_list[0], tracklet1, tracklet2,
                          params_dict)
        
        print('')
        print(ii)
        print('Mi', M_list[ii])
        print('Xi', X_list[ii])
        print('elem', elem_ii)
        print('QZS-2 Xo Err: ', np.linalg.norm(X_list[ii] - Xo_true))
        print('RA Resids RMS [arcsec]: ', ra_rms)
        print('DEC Resids RMS [arcsec]: ', dec_rms)
        
        
    ###########################################################################
    # QZS-1R and QZS-2 Cross-Association
    ###########################################################################
    
    tracklet1 = tracklet_dict[0]
    tracklet2 = tracklet_dict[5]
    
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
    

#    obj_id = qzs2_norad
#    state_dict = tle.propagate_TLE([obj_id], tk_list)
#    
#    r0 = state_dict[obj_id]['r_GCRF'][0]
#    v0 = state_dict[obj_id]['v_GCRF'][0]
#    Xo_true = np.concatenate((r0, v0), axis=0)
#    elem_true = astro.cart2kep(Xo_true)
#    
#    print('QZS-2 Elem Truth: ', elem_true)
    
    
    
    print('Final Answers')
    print('X_list', X_list)
    print('M_list', M_list)
    
    for ii in range(len(M_list)):
        
        elem_ii = astro.cart2kep(X_list[ii])
        resids, ra_rms, dec_rms = \
            compute_resids(X_list[ii], tk_list[0], tracklet1, tracklet2,
                           params_dict)
        
        print('')
        print(ii)
        print('Mi', M_list[ii])
        print('Xi', X_list[ii])
        print('elem', elem_ii)
#        print('QZS-2 Xo Err: ', np.linalg.norm(X_list[ii] - Xo_true))
        print('RA Resids RMS [arcsec]: ', ra_rms)
        print('DEC Resids RMS [arcsec]: ', dec_rms)
    
    
    return


def process_tracklets_full(tracklet_file, csv_file, correlation_file):
    
    # Load data
    pklFile = open(tracklet_file, 'rb' )
    data = pickle.load( pklFile )
    tracklet_dict = data[0]
    params_dict = data[1]
    truth_dict = data[2]
    pklFile.close()
    
    # EOP data
    sensor_params = params_dict['sensor_params']
    eop_alldata = sensor_params['eop_alldata']
    XYs_df = sensor_params['XYs_df']
    
    # Initialize output
    df_list = []
    correlation_dict = {}
    
    # Execution time
    gooding_time = 0.
    resids_time = 0.
    
    # Exclusion times
    exclude_short = 12*3600.
    exclude_long = 10.*86400.
    
    # Loop through tracklets and compute association
    tracklet_id_list = list(tracklet_dict.keys())
    case_id = 0
    for ii in tracklet_id_list:
        tracklet_ii = tracklet_dict[ii]
        
        for jj in tracklet_id_list[ii+1:]:
            tracklet_jj = tracklet_dict[jj]
            
            # Check times and switch if needed
            if tracklet_jj['tk_list'][0] > tracklet_ii['tk_list'][-1]:
                tracklet1 = copy.deepcopy(tracklet_ii)
                tracklet2 = copy.deepcopy(tracklet_jj)
                tracklet1_id = ii
                tracklet2_id = jj
            else:
                tracklet1 = copy.deepcopy(tracklet_jj)
                tracklet2 = copy.deepcopy(tracklet_ii)
                tracklet1_id = jj
                tracklet2_id = ii
            
            # Check exclusion criteria
            if (tracklet2['tk_list'][0] - tracklet1['tk_list'][-1]).total_seconds() < exclude_short:
                continue
            
            if (tracklet2['tk_list'][0] - tracklet1['tk_list'][-1]).total_seconds() > exclude_long:
                continue
            
            
            case_id += 1
            
            
            print('')
            print('case_id', case_id)
            print('tracklet1')
            print(tracklet1['obj_id'])
            print(tracklet1['tk_list'][0])
            print('tracklet2')
            print(tracklet2['obj_id'])
            print(tracklet2['tk_list'][0])
    
            
            # Set up for correlation
            tk_list = [tracklet1['tk_list'][0], tracklet1['tk_list'][-1], tracklet2['tk_list'][-1]]
            Yk_list = [tracklet1['Yk_list'][0], tracklet1['Yk_list'][-1], tracklet2['Yk_list'][-1]]
            sensor_id_list = [tracklet1['sensor_id_list'][0],
                              tracklet1['sensor_id_list'][-1],
                              tracklet2['sensor_id_list'][-1]]
            
            sensor_params = params_dict['sensor_params']
            orbit_regime = tracklet1['orbit_regime']
            
            # print(tracklet1['tk_list'])
            # print(tracklet2['tk_list'])
            
            print(tk_list)
            print(Yk_list)
            print(sensor_id_list)
            
            # Compute true association details
            obj_id = tracklet1['obj_id']
            Xo_true = truth_dict[obj_id][tk_list[0]]
            elem_true = astro.cart2kep(Xo_true)
            sma = elem_true[0]
            period = astro.sma2period(sma)
            M_true = int(np.floor((tk_list[-1]-tk_list[0]).total_seconds()/period))
            
            print('Tracklet1 Elem Truth: ', elem_true)
            print('Xo_true', Xo_true)
            print('sma', sma)
            print('period', period/3600.)
            print('M_frac', (tk_list[-1]-tk_list[0]).total_seconds()/period)
            print('M_true', M_true)
                
            # Run Gooding IOD
            start = time.time()
            X_list, M_list = iod.gooding_angles_iod(tk_list, Yk_list, sensor_id_list,
                                                    sensor_params, 
                                                    eop_alldata=eop_alldata,
                                                    XYs_df=XYs_df,
                                                    orbit_regime=orbit_regime,
                                                    search_mode='middle_out',
                                                    periapsis_check=True,
                                                    rootfind='min', debug=False)

            
            gooding_time += time.time() - start
            
            
            
            # print('Final Answers')
            # print('X_list', X_list)
            # print('M_list', M_list)
    
            # If no solutions found, record basic data
            if len(M_list) == 0:
                
                if tracklet1['obj_id'] == tracklet2['obj_id']:
                    corr_truth = True
                else:
                    corr_truth = False
                
                correlation_dict[case_id] = {}
                correlation_dict[case_id]['tracklet1_id'] = tracklet1_id
                correlation_dict[case_id]['tracklet2_id'] = tracklet2_id
                correlation_dict[case_id]['corr_truth_list'] = [corr_truth]
                correlation_dict[case_id]['obj1_id'] = tracklet1['obj_id']
                correlation_dict[case_id]['obj2_id'] = tracklet2['obj_id']
                correlation_dict[case_id]['Xo_true'] = Xo_true
                correlation_dict[case_id]['Xo_list'] = []
                correlation_dict[case_id]['M_list'] = []
                correlation_dict[case_id]['resids_list'] = []
                correlation_dict[case_id]['ra_rms_list'] = []
                correlation_dict[case_id]['dec_rms_list'] = []
                
                
                df_list.append([case_id, tracklet1['obj_id'], tracklet2['obj_id'],
                               tracklet1['tk_list'][0], tracklet2['tk_list'][0],
                               0, np.inf, np.inf, np.inf, corr_truth])
            
            
            # Compute sensor locations and observed measurements at all times
            # not used for IOD solution
            start = time.time()
            tk_list1, Yk_list1, Rmat1 = reduce_tracklets(tracklet1, tracklet2,
                                                         params_dict)
            resids_time += time.time() - start
            
            
            for ind in range(len(M_list)):
                
                
                elem = astro.cart2kep(X_list[ind])
                start = time.time()
                resids, ra_rms, dec_rms = \
                    compute_resids(X_list[ind], tk_list[0], tk_list1, Yk_list1,
                                   Rmat1, params_dict)
                    
                resids_time += time.time() - start
                                    
                
                Xo_err = np.linalg.norm(X_list[ind] - Xo_true)
                
                # True correlation status
                if (tracklet1['obj_id'] == tracklet2['obj_id']) and (M_list[ind] == M_true):
                    corr_truth = True
                else:
                    corr_truth = False
                
                print('')
                print(ind)
                print('Mi', M_list[ind])
                # print('Xi', X_list[ind])
                # print('elem', elem)
                print('Xo Err: ', Xo_err)
                print('RA Resids RMS [arcsec]: ', ra_rms)
                print('DEC Resids RMS [arcsec]: ', dec_rms)
                
                df_list.append([case_id, tracklet1['obj_id'], tracklet2['obj_id'],
                               tracklet1['tk_list'][0], tracklet2['tk_list'][0],
                               M_list[ind], Xo_err, ra_rms, dec_rms, corr_truth])
                
                # print(df_list)
                
                if ind == 0:
                    correlation_dict[case_id] = {}
                    correlation_dict[case_id]['tracklet1_id'] = tracklet1_id
                    correlation_dict[case_id]['tracklet2_id'] = tracklet2_id
                    correlation_dict[case_id]['corr_truth_list'] = [corr_truth]
                    correlation_dict[case_id]['obj1_id'] = tracklet1['obj_id']
                    correlation_dict[case_id]['obj2_id'] = tracklet2['obj_id']
                    correlation_dict[case_id]['Xo_true'] = Xo_true
                    correlation_dict[case_id]['Xo_list'] = X_list
                    correlation_dict[case_id]['M_list'] = M_list
                    correlation_dict[case_id]['resids_list'] = [resids]
                    correlation_dict[case_id]['ra_rms_list'] = [ra_rms]
                    correlation_dict[case_id]['dec_rms_list'] = [dec_rms]
                    
                else:
                    correlation_dict[case_id]['corr_truth_list'].append(corr_truth)
                    correlation_dict[case_id]['resids_list'].append(resids)
                    correlation_dict[case_id]['ra_rms_list'].append(ra_rms)
                    correlation_dict[case_id]['dec_rms_list'].append(dec_rms)
                

                
            print('Gooding Time: ', gooding_time)
            print('Resids Time: ', resids_time)

            
    df = pd.DataFrame(df_list, columns=['Case ID', 'Tracklet1_Obj_ID',
                                        'Tracklet2_Obj_ID', 't_10', 't_20',
                                        'M [rev]', 'Xo Err', 'RA rms',
                                        'DEC rms', 'Correlation Truth'])

    df.to_csv(csv_file)
    
    
    pklFile = open( correlation_file, 'wb' )
    pickle.dump( [correlation_dict, tracklet_dict, params_dict, truth_dict], pklFile, -1 )
    pklFile.close()
    
    
    return


def reduce_tracklets(tracklet1, tracklet2, params_dict):
    
    # Break out inputs
    state_params = params_dict['state_params']
    int_params = params_dict['int_params']
    sensor_params = params_dict['sensor_params']
    eop_alldata = sensor_params['eop_alldata']
    XYs_df = sensor_params['XYs_df']
    
    # Retrieve tracklet data
    tk_list1 = tracklet1['tk_list']
    Yk_list1 = tracklet1['Yk_list']
    sensor_id_list1 = tracklet1['sensor_id_list']
    
    tk_list2 = tracklet2['tk_list']
    Yk_list2 = tracklet2['Yk_list']
    sensor_id_list2 = tracklet2['sensor_id_list']
    
    # Combine into single lists
    tk_list1.extend(tk_list2)
    Yk_list1.extend(Yk_list2)
    sensor_id_list1.extend(sensor_id_list2)
    
    # Remove entries used to compute Gooding IOD solution
    del tk_list1[-1]
    del tk_list1[0]
    del Yk_list1[-1]
    del Yk_list1[0]
    del sensor_id_list1[-1]
    del sensor_id_list1[0]
    del tk_list2[-1]
    del Yk_list2[-1]
    del sensor_id_list2[-1]
    
    # Compute sensor ECI coordinates
    Rmat = np.zeros((3,len(tk_list1)))
    for kk in range(len(tk_list1)):
        tk = tk_list1[kk]
        sensor_id = sensor_id_list1[kk]        
        site_ecef = sensor_params[sensor_id]['site_ecef']
        
        # Compute sensor location in ECI
        EOP_data = eop.get_eop_data(eop_alldata, tk)
        site_eci, dum = coord.itrf2gcrf(site_ecef, np.zeros((3,1)), tk,
                                        EOP_data, XYs_df)
        
        Rmat[:,kk] = site_eci.flatten()

    
    return tk_list1, Yk_list1, Rmat


def compute_resids(Xo, UTC0, tk_list1, Yk_list1, Rmat1, params_dict):
    

    
    # Break out inputs
    state_params = params_dict['state_params']
    int_params = params_dict['int_params']
    sensor_params = params_dict['sensor_params']
    
    # Propagate initial orbit, compute measurements and resids
    resids = np.zeros((2, len(tk_list1)))
    loop_start = time.time()
    eop_time = 0.
    astro_time = 0.
    meas_time = 0.
    for kk in range(len(tk_list1)):
        tk = tk_list1[kk]
        Yk = Yk_list1[kk]
        sensor_eci = Rmat1[:,kk].reshape(3,1)
        
        start = time.time()
        Xk = astro.element_conversion(Xo, 1, 1, dt=(tk-UTC0).total_seconds())
        astro_time += time.time() - start
        
        start = time.time()
        r_eci = Xk[0:3].reshape(3,1)
        rho_eci = r_eci - sensor_eci
        ra_calc = atan2(rho_eci[1], rho_eci[0])
        dec_calc = asin(rho_eci[2]/np.linalg.norm(rho_eci))
        Y_calc = np.reshape([ra_calc, dec_calc], (2,1))
        meas_time += time.time() - start
        
        
        diff = Yk - Y_calc
        if diff[0] > np.pi:
            diff[0] -= 2.*np.pi
        if diff[0] < -np.pi:
            diff[0] += 2.*np.pi

        resids[:,kk] = diff.flatten()
        
    # print('orbit prop/meas time', time.time() - loop_start)
    # print('eop time', eop_time)
    # print('astro time', astro_time)
    # print('meas time', meas_time)
        
    ra_resids = resids[0,:]
    dec_resids = resids[1,:]
    
    ra_rms = np.sqrt(np.dot(ra_resids, ra_resids)/len(ra_resids))*(1./arcsec2rad)
    dec_rms = np.sqrt(np.dot(dec_resids, dec_resids)/len(dec_resids))*(1./arcsec2rad)
    
    # print('ra std', np.std(ra_resids)*(1./arcsec2rad))
    # print('dec std', np.std(dec_resids)*(1./arcsec2rad))
    
    # mistake
    
    return resids, ra_rms, dec_rms


def check_tracklet_dict(tracklets_file):
    
    # Load data
    pklFile = open(tracklets_file, 'rb' )
    data = pickle.load( pklFile )
    tracklet_dict = data[0]
    params_dict = data[1]
    pklFile.close()
    
    
    for tracklet_id in tracklet_dict:
        print('obj_id', tracklet_dict[tracklet_id]['obj_id'])
        print('t0', tracklet_dict[tracklet_id]['tk_list'][0])
    
    
    return


if __name__ == '__main__':
    
    
    fdir = r'D:\documents\research_projects\iod\data\sim\debug\2022_11_23_twobody_geo_6obj_7day_10min'
    tracklets_file = os.path.join(fdir, 'twobody_geo_6obj_10min_noise0.pkl')
    csv_file = os.path.join(fdir, 'twobody_geo_6obj_10min_noise0_corr_summary_min.csv')
    correlation_file = os.path.join(fdir, 'twobody_geo_6obj_10min_noise1_correlation.pkl')
    
    # noise = 0.
    
#    leo_tracklets_marco()
    
    
    # geo_tracklets(tracklets_file, noise)
    
    # test_tracklet_association()
    
    # check_tracklet_dict(tracklets_file)
    
    
    # start = time.time()
    
    # process_tracklets_full(tracklets_file, csv_file, correlation_file)
    
    
    # print('Full run time', time.time() - start)
    
    
    # ra_lim = 500.
    # dec_lim = 500
    # analysis.evaluate_tracklet_correlation(correlation_file, ra_lim, dec_lim)
    
    
    # qzs1r_norad = 49336
    # qzs2_norad = 42738
    # qzs3_norad = 42917
    # qzs4_norad = 42965
    
    # amos5_norad = 37950
    # coms1_norad = 36744
    
    # obj_id_list = [49336, 42738, 42917, 42965, 37950, 36744]
    
    # UTC0 = datetime(2022, 11, 7, 11, 0, 0)
    # state_dict = tle.propagate_TLE(obj_id_list, [UTC0])
    
    # for obj_id in obj_id_list:
    #     r0 = state_dict[obj_id]['r_GCRF'][0]
    #     v0 = state_dict[obj_id]['v_GCRF'][0]
    #     Xo = np.concatenate((r0, v0), axis=0)
        
    #     elem = astro.cart2kep(Xo)
        
    #     print('')
    #     print(obj_id)
    #     print(elem)
    
    
    
    
    
    
    
    
    
    
    
    
    
    