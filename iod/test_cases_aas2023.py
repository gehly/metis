import numpy as np
from math import pi, asin, atan2
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
import pickle
import getpass
import os
import inspect
import pandas as pd

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
    
    setup_file = os.path.join('test_cases', 'twobody_geo_10min_noise5.pkl')
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [tracklet_dict, params_dict], pklFile, -1 )
    pklFile.close()
    
    
    
    return


def test_tracklet_association():
    
    # True orbits
    qzs1r_norad = 49336
    qzs2_norad = 42738
    qzs3_norad = 42917
    qzs4_norad = 42965
    
    ses15_norad = 42709
    amos5_norad = 37950
    coms1_norad = 36744
    
    
    # Load data
    setup_file = os.path.join('test_cases', 'twobody_geo_10min_noise5.pkl')
    pklFile = open(setup_file, 'rb' )
    data = pickle.load( pklFile )
    tracklet_dict = data[0]
    params_dict = data[1]
    pklFile.close()
    
    
    ###########################################################################
    # QZS-1R Self-Association
    ###########################################################################
    
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
        
        
    mistake
        

    ###########################################################################
    # QZS-2 Self-Association
    ###########################################################################
#    
#    tracklet1 = tracklet_dict[3]
#    tracklet2 = tracklet_dict[4]
#    
#    tk_list = [tracklet1['tk_list'][0], tracklet1['tk_list'][-1], tracklet2['tk_list'][-1]]
#    Yk_list = [tracklet1['Yk_list'][0], tracklet1['Yk_list'][-1], tracklet2['Yk_list'][-1]]
#    sensor_id_list = [tracklet1['sensor_id_list'][0],
#                      tracklet1['sensor_id_list'][-1],
#                      tracklet2['sensor_id_list'][-1]]
#    
#    sensor_params = params_dict['sensor_params']
#    orbit_regime = tracklet1['orbit_regime']
#    
#    print(tk_list)
#    print(Yk_list)
#    print(sensor_id_list)
#
#    # Execute function
#    X_list, M_list = iod.gooding_angles_iod(tk_list, Yk_list, sensor_id_list,
#                                            sensor_params, orbit_regime=orbit_regime,
#                                            search_mode='middle_out',
#                                            periapsis_check=True,
#                                            rootfind='zeros')
#    
#
#    obj_id = qzs2_norad
#    state_dict = tle.propagate_TLE([obj_id], tk_list)
#    
#    r0 = state_dict[obj_id]['r_GCRF'][0]
#    v0 = state_dict[obj_id]['v_GCRF'][0]
#    Xo_true = np.concatenate((r0, v0), axis=0)
#    elem_true = astro.cart2kep(Xo_true)
#    
#    print('QZS-2 Elem Truth: ', elem_true)
#    
#    
#    
#    print('Final Answers')
#    print('X_list', X_list)
#    print('M_list', M_list)
#    
#    for ii in range(len(M_list)):
#        
#        elem_ii = astro.cart2kep(X_list[ii])
#        resids, ra_rms, dec_rms = \
#            compute_resids(X_list[ii], tk_list[0], tracklet1, tracklet2,
#                           params_dict)
#        
#        print('')
#        print(ii)
#        print('Mi', M_list[ii])
#        print('Xi', X_list[ii])
#        print('elem', elem_ii)
#        print('QZS-2 Xo Err: ', np.linalg.norm(X_list[ii] - Xo_true))
#        print('RA Resids RMS [arcsec]: ', ra_rms)
#        print('DEC Resids RMS [arcsec]: ', dec_rms)
        
        
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


def compute_resids(Xo, UTC0, tracklet1, tracklet2, params_dict):
    
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
    
    # Combine into single lists
    tk_list1.extend(tk_list2)
    Yk_list1.extend(Yk_list2)
    sensor_id_list1.extend(sensor_id_list2)
    
    # Propagate initial orbit, compute measurements and resids
    resids = np.zeros((2, len(tk_list1)))
    for kk in range(len(tk_list1)):
        tk = tk_list1[kk]
        Yk = Yk_list1[kk]
        sensor_id = sensor_id_list1[kk]
        EOP_data = eop.get_eop_data(eop_alldata, tk)
        
        tout, Xout = dyn.general_dynamics(Xo, [UTC0, tk], state_params,
                                          int_params)
        Xk = Xout[-1,:].reshape(6,1)
        Yprop = mfunc.compute_measurement(Xk, state_params, sensor_params,
                                          sensor_id, tk, EOP_data, XYs_df)
        
        diff = Yk - Yprop
        if diff[0] > np.pi:
            diff[0] -= 2.*np.pi
        if diff[0] < -np.pi:
            diff[0] += 2.*np.pi

        resids[:,kk] = diff.flatten()
        
    ra_resids = resids[0,:]
    dec_resids = resids[1,:]
    
    ra_rms = np.sqrt(np.dot(ra_resids, ra_resids))*(1./arcsec2rad)
    dec_rms = np.sqrt(np.dot(dec_resids, dec_resids))*(1./arcsec2rad)
    
    
    

    
    return resids, ra_rms, dec_rms


def geo_twobody_setup(setup_file):
    
    # Object IDs
    qzs1r_norad = 49336
    qzs2_norad = 42738
    qzs3_norad = 42917
    qzs4_norad = 42965
    
    ses15_norad = 42709
    amos5_norad = 37950
    coms1_norad = 36744
    
    # Initial state vectors from TLE data
    obj_id_list = [qzs1r_norad, qzs2_norad, qzs3_norad, qzs4_norad,
                   ses15_norad, amos5_norad, coms1_norad]
    UTC0 = datetime(2022, 11, 7, 11, 0, 0)
    tle_dict = tle.propagate_TLE(obj_id_list, [UTC0])
    
    # Build truth dict
    truth_dict = {}
    truth_dict[UTC0] = {}
    for obj_id in obj_id_list:
    
        r0 = tle_dict[obj_id]['r_GCRF'][0]
        v0 = tle_dict[obj_id]['v_GCRF'][0]
        Xt = np.concatenate((r0, v0), axis=0)
        
        truth_dict[UTC0][obj_id] = Xt
        
        
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
    state_params['radius_m'] = 1.
    state_params['albedo'] = 0.1
    state_params['bodies_to_create'] = ['Earth']
    state_params['global_frame_origin'] = 'Earth'
    state_params['global_frame_orientation'] = 'J2000'
    state_params['central_bodies'] = ['Earth']
    state_params['sph_deg'] = 0
    state_params['sph_ord'] = 0
    state_params['mass'] = 400.
    state_params['Cd'] = 0.
    state_params['Cr'] = 0.
    state_params['drag_area_m2'] = 4.
    state_params['srp_area_m2'] = 4.
    
    # Integration function and additional settings    
    int_params = {}
    int_params['integrator'] = 'tudat'
    int_params['tudat_integrator'] = 'rkf78'
    int_params['step'] = 10.
    int_params['max_step'] = 1000.
    int_params['min_step'] = 1.
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'
    
    # Sensor and measurement parameters
    sensor_id_list = ['RMIT ROO']
    sensor_params = sens.define_sensors(sensor_id_list)
    sensor_params['eop_alldata'] = eop_alldata
    sensor_params['XYs_df'] = XYs_df
    
    for sensor_id in sensor_id_list:
        sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
        sigma_dict = {}
        sigma_dict['ra'] = 5.*arcsec2rad   # rad
        sigma_dict['dec'] = 5.*arcsec2rad  # rad
        sensor_params[sensor_id]['sigma_dict'] = sigma_dict
        sensor_params[sensor_id]['lam_clutter'] = 5.
        FOV_hlim = sensor_params[sensor_id]['FOV_hlim']
        FOV_vlim = sensor_params[sensor_id]['FOV_vlim']        
        sensor_params[sensor_id]['V_sensor'] = (FOV_hlim[1] - FOV_hlim[0])*(FOV_vlim[1] - FOV_vlim[0])
        
        
    # Save truth and params
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [truth_dict, state_params, int_params, sensor_params], pklFile, -1 )
    pklFile.close()
    
    
    return


def tracklet_visibility(vis_file, prev_file, truth_file):
    
    
    pklFile = open(prev_file, 'rb' )
    data = pickle.load( pklFile )
    truth_dict = data[0]
    state_params = data[1]
    int_params = data[2]
    sensor_params = data[3]
    pklFile.close()
    
    # Break out inputs
    eop_alldata = sensor_params['eop_alldata']
    XYs_df = sensor_params['XYs_df']
    sensor_id_list = ['RMIT ROO']
    
    

    # Time vector
    prev_tk_list = sorted(list(truth_dict.keys()))
    UTC0 = prev_tk_list[-1]
    obj_id_list = list(truth_dict[UTC0].keys())
    ndays = 0.5
    tvec = np.arange(0., ndays*86400.+1, 10.)
    tk_list = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    
    

    # Loop over times to build truth dict and check visibility conditions
    vis_dict = {}
    for kk in range(len(tk_list)):
        
        tk = tk_list[kk]
        
        
        if kk > 0:
            
            # Prior truth state
            tk_prior = tk_list[kk-1]
            X = np.array([])
            for obj_id in obj_id_list:
                Xj = truth_dict[tk_prior][obj_id]
                X = np.append(X, Xj)

            
            tin = [tk_prior, tk]
            tout, Xout = dyn.general_dynamics(X, tin, state_params, int_params)
            X = Xout[-1,:]

            ind = 0
            truth_dict[tk] = {}
            for obj_id in obj_id_list:                
                truth_dict[tk][obj_id] = X[ind:ind+6].reshape(6,1)
                ind += 6
        
        # Check visibility conditions
        EOP_data = eop.get_eop_data(eop_alldata, tk)
        
        for sensor_id in sensor_id_list:
            
            vis_obj_tk = []
            for obj_id in obj_id_list:
                
                # print(obj_id)
                # print(truth_dict[tk])
        
                Xj = truth_dict[tk][obj_id]          
                          
                if visfunc.check_visibility(Xj, state_params, sensor_params,
                                            sensor_id, tk, EOP_data, XYs_df):
                    
                    vis_obj_tk.append(obj_id)
            
            if len(vis_obj_tk) > 0:
                vis_dict[tk] = vis_obj_tk

                    
    # Generate visibility report
    vis_times = sorted(list(vis_dict.keys()))
    output = {}
    output['Time [UTC]'] = vis_times
    
    for obj_id in obj_id_list:
        output[obj_id] = []
        
    for tk in vis_times:
        for obj_id in obj_id_list:
            if obj_id in vis_dict[tk]:
                output[obj_id].append(True)
            else:
                output[obj_id].append(False)
                
            
    # Form dataframe and CSV output
    output_df = pd.DataFrame.from_dict(output)
    output_df.to_csv(vis_file)
    
    # Save truth and params
    pklFile = open( truth_file, 'wb' )
    pickle.dump( [truth_dict, state_params, int_params, sensor_params], pklFile, -1 )
    pklFile.close()


    return


def generate_tracklets():
    
    
    
    # # Check visibility conditions
    # EOP_data = eop.get_eop_data(eop_alldata, tk)
    # Zk_list = []
    # sensor_kk_list = []
    # for sensor_id in sensor_id_list:
        
    #     sensor = sensor_params[sensor_id]
    #     p_det = filter_params['p_det']
    #     center_flag = True            
    #     for Xj in truth_dict[tk_list[kk]]['Xt_list']:            
                      
    #         if visfunc.check_visibility(Xj, state_params, sensor_params,
    #                                     sensor_id, UTC, EOP_data, XYs_df):
                
    #             # Incorporate missed detection
    #             if np.random.rand() > p_det:
    #                 continue
                
    #             # Compute measurements
    #             zj = mfunc.compute_measurement(Xj, state_params, sensor_params,
    #                                            sensor_id, UTC, EOP_data, XYs_df,
    #                                            meas_types=sensor['meas_types'])
                
    #             # Store first measurement for each sensor as FOV center
    #             if center_flag:
    #                 center = zj.copy()
    #                 center_flag = False
                
    #             # Add noise and store measurement data
    #             zj[0] += np.random.randn()*sigma_dict['ra']
    #             zj[1] += np.random.randn()*sigma_dict['dec']
                
    #             Zk_list.append(zj)
    #             sensor_kk_list.append(sensor_id)
        
    #     # Incorporate clutter measurements
    #     n_clutter = ss.poisson.rvs(sensor['lam_clutter'])

    #     # Compute clutter meas in RA/DEC, uniform over FOV
    #     for c_ind in range(n_clutter):
    #         FOV_hlim = sensor['FOV_hlim']
    #         FOV_vlim = sensor['FOV_vlim']
    #         ra  = center[0] + (FOV_hlim[1]-FOV_hlim[0])*(np.random.rand() - 0.5)
    #         dec = center[1] + (FOV_vlim[1]-FOV_vlim[0])*(np.random.rand() - 0.5)

    #         zclutter = np.reshape([ra, dec], (2,1))
    #         Zk_list.append(zclutter)
    #         sensor_kk_list.append(sensor_id)

    # # If measurements were collected, randomize order and store
    # if len(Zk_list) > 0:
        
    #     inds = list(range(len(Zk_list)))
    #     random.shuffle(inds)
        
    #     meas_dict[UTC] = {}
    #     meas_dict[UTC]['Zk_list'] = [Zk_list[ii] for ii in inds]
    #     meas_dict[UTC]['sensor_id_list'] = [sensor_kk_list[ii] for ii in inds]
    
    
    return


def check_truth():
    
    fdir = r'D:\documents\research_projects\iod\data\sim\test\aas2023_geo_6obj_7day'

    
    fname = 'geo_twobody_6obj_7day_truth_9.pkl'    
    truth_file = os.path.join(fdir, fname)
    
    pklFile = open(truth_file, 'rb' )
    data = pickle.load( pklFile )
    truth_dict = data[0]
    pklFile.close()
    
    tk_list = sorted(list(truth_dict.keys()))
    t0 = tk_list[0]
    tf = tk_list[-1]
    
    print(t0)
    print(tf)
    
    print(len(tk_list))

    
    obj_id_list = list(truth_dict[t0].keys())
    obj_id = obj_id_list[0]
    
    Xo = truth_dict[t0][obj_id]
    tdays = []
    Xerr = np.zeros((6,len(tk_list)))
    for tk in tk_list:
        dt_sec = (tk - t0).total_seconds()
        tdays.append(dt_sec/86400.)
        Xnum = truth_dict[tk][obj_id]
        Xanal = astro.element_conversion(Xo, 1, 1, dt=dt_sec)
        
        kk = tk_list.index(tk)
        Xerr[:,kk] = (Xanal - Xnum).flatten()
        
        
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tdays, Xerr[0,:], 'k.')
    plt.ylabel('X [km]')
    plt.subplot(3,1,2)
    plt.plot(tdays, Xerr[1,:], 'k.')
    plt.ylabel('Y [km]')
    plt.subplot(3,1,3)
    plt.plot(tdays, Xerr[2,:], 'k.')
    plt.ylabel('Z [km]')
    plt.xlabel('Time [days]')
    
    plt.show()
    
    
    return





if __name__ == '__main__':
    
    plt.close('all')
    
    
#    leo_tracklets_marco()
    
#    geo_tracklets()
    
    # test_tracklet_association()
    
    fdir = r'D:\documents\research_projects\iod\data\sim\test\aas2023_geo_6obj_7day'
    
    fname = 'geo_twobody_6obj_7day_setup.pkl'
    setup_file = os.path.join(fdir, fname)  
    
    fname = 'geo_twobody_6obj_7day_truth_8.pkl'    
    prev_file = os.path.join(fdir, fname)
    
    fname = 'geo_twobody_6obj_7day_visibility_9.csv'
    vis_file = os.path.join(fdir, fname)
    
    fname = 'geo_twobody_6obj_7day_truth_9.pkl'    
    truth_file = os.path.join(fdir, fname)
    
    # geo_twobody_setup(setup_file)
    
    
    
    # tracklet_visibility(vis_file, prev_file, truth_file)
    
    check_truth()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    