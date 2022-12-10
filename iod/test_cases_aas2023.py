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


def geo_tracklet_visibility():
    
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
    truth_dict[UTC0]['Xt_list'] = []
    for obj_id in obj_id_list:
    
        r0 = tle_dict[obj_id]['r_GCRF'][0]
        v0 = tle_dict[obj_id]['v_GCRF'][0]
        Xt = np.concatenate((r0, v0), axis=0)
        
        truth_dict[UTC0]['Xt_list'].append(Xt)
        
        
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


    # Time vector
    tk_list = []
    ndays = 7.
    tvec = np.arange(0., ndays*86400.+1., 10.)
    tk_list.append([UTC0 + timedelta(seconds=ti) for ti in tvec])
    
    
        
    
    
    return





if __name__ == '__main__':
    
    
#    leo_tracklets_marco()
    
#    geo_tracklets()
    
    test_tracklet_association()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    