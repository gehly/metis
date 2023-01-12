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
import scipy.stats as ss
import random
import copy
import time

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from data_processing import measurement_analysis as ma
from data_processing import data_processing as proc
from iod import iod_functions_jit as iod
import estimation.analysis_functions as analysis
import estimation.estimation_functions as est
from estimation import multitarget_functions2 as mult
import dynamics.dynamics_functions as dyn
import sensors.measurement_functions as mfunc
import sensors.sensors as sens
import sensors.visibility_functions as visfunc
import utilities.astrodynamics as astro
import utilities.coordinate_systems as coord
import utilities.eop_functions as eop
from utilities.constants import GME, arcsec2rad
from utilities import tle_functions as tle





# def leo_tracklets_marco():
    
#     username = input('space-track username: ')
#     password = getpass.getpass('space-track password: ')
    
    
#     obj_id = 53323
#     sensor_id = 'Leiden Optical'
#     orbit_regime = 'LEO'
#     UTC0 = datetime(2022, 8, 2, 22, 56, 47)
#     dt_interval = 0.5
#     dt_max = 10.
#     noise = 5.
    
    
#     # Retrieve latest EOP data from celestrak.com
#     eop_alldata = eop.get_celestrak_eop_alldata()
        
#     # Retrieve polar motion data from file
#     XYs_df = eop.get_XYs2006_alldata()
    
#     # Define state parameters
#     state_params = {}
#     state_params['GM'] = GME
#     state_params['radius_m'] = 1.
#     state_params['albedo'] = 0.1

#     # Integration function and additional settings
#     int_params = {}
#     int_params['integrator'] = 'solve_ivp'
#     int_params['ode_integrator'] = 'DOP853'
#     int_params['intfcn'] = dyn.ode_twobody
#     int_params['rtol'] = 1e-12
#     int_params['atol'] = 1e-12
#     int_params['time_format'] = 'datetime'
    
#     # Sensor parameters
#     sensor_id_list = [sensor_id]
#     sensor_params = sens.define_sensors(sensor_id_list)
#     sensor_params['eop_alldata'] = eop_alldata
#     sensor_params['XYs_df'] = XYs_df
    
#     for sensor_id in sensor_id_list:
#         sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
#         sigma_dict = {}
#         sigma_dict['ra'] = noise*arcsec2rad   # rad
#         sigma_dict['dec'] = noise*arcsec2rad  # rad
#         sensor_params[sensor_id]['sigma_dict'] = sigma_dict
        
        
    
#     params_dict = {}
#     params_dict['sensor_params'] = sensor_params
#     params_dict['state_params'] = state_params
#     params_dict['int_params'] = int_params
    
    
    
#     tracklet_dict = {}
#     tracklet_dict = \
#         mfunc.tracklet_generator(obj_id, UTC0, dt_interval, dt_max, sensor_id,
#                                  params_dict, tracklet_dict, orbit_regime,
#                                  username=username, password=password)
    
    
#     UTC0 = datetime(2022, 8, 3, 23, 52, 55)
#     dt_interval = 0.5
#     dt_max = 10.
#     tracklet_dict = \
#         mfunc.tracklet_generator(obj_id, UTC0, dt_interval, dt_max, sensor_id,
#                                  params_dict, tracklet_dict, orbit_regime,
#                                  username=username, password=password)
        
    
#     UTC0 = datetime(2022, 8, 6, 23, 34, 23)
#     dt_interval = 0.5
#     dt_max = 10.
#     tracklet_dict = \
#         mfunc.tracklet_generator(obj_id, UTC0, dt_interval, dt_max, sensor_id,
#                                  params_dict, tracklet_dict, orbit_regime,
#                                  username=username, password=password)

    
#     UTC0 = datetime(2022, 8, 7, 22, 56, 57)
#     dt_interval = 0.5
#     dt_max = 30.
#     tracklet_dict = \
#         mfunc.tracklet_generator(obj_id, UTC0, dt_interval, dt_max, sensor_id,
#                                  params_dict, tracklet_dict, orbit_regime,
#                                  username=username, password=password)
        
        
#     UTC0 = datetime(2022, 8, 9, 23, 15, 38)
#     dt_interval = 0.5
#     dt_max = 12.
#     tracklet_dict = \
#         mfunc.tracklet_generator(obj_id, UTC0, dt_interval, dt_max, sensor_id,
#                                  params_dict, tracklet_dict, orbit_regime,
#                                  username=username, password=password)
        
        
#     UTC0 = datetime(2022, 8, 10, 22, 38, 8)
#     dt_interval = 0.5
#     dt_max = 10.
#     tracklet_dict = \
#         mfunc.tracklet_generator(obj_id, UTC0, dt_interval, dt_max, sensor_id,
#                                  params_dict, tracklet_dict, orbit_regime,
#                                  username=username, password=password)
        
        
#     UTC0 = datetime(2022, 8, 12, 22, 56, 54)
#     dt_interval = 0.5
#     dt_max = 10.
#     tracklet_dict = \
#         mfunc.tracklet_generator(obj_id, UTC0, dt_interval, dt_max, sensor_id,
#                                  params_dict, tracklet_dict, orbit_regime,
#                                  username=username, password=password)
    
    
    
#     print(tracklet_dict)
    
#     setup_file = os.path.join('test_cases', 'twobody_leo_marco_noise5.pkl')
#     pklFile = open( setup_file, 'wb' )
#     pickle.dump( [tracklet_dict, params_dict], pklFile, -1 )
#     pklFile.close()
    
    
    
#     return


# def geo_tracklets():
    
#     username = input('space-track username: ')
#     password = getpass.getpass('space-track password: ')
    
    
#     qzs1r_norad = 49336
#     qzs2_norad = 42738
#     qzs3_norad = 42917
#     qzs4_norad = 42965
    
#     ses15_norad = 42709
#     amos5_norad = 37950
#     coms1_norad = 36744
    
    
#     # Common parameter setup
#     tracklet_dict = {}    
#     sensor_id = 'RMIT ROO'
#     orbit_regime = 'GEO'    
#     dt_interval = 10.
#     dt_max = 600.
#     noise = 5.    
    
#     # Retrieve latest EOP data from celestrak.com
#     eop_alldata = eop.get_celestrak_eop_alldata()
        
#     # Retrieve polar motion data from file
#     XYs_df = eop.get_XYs2006_alldata()
    
#     # Define state parameters
#     state_params = {}
#     state_params['GM'] = GME
#     state_params['radius_m'] = 1.
#     state_params['albedo'] = 0.1

#     # Integration function and additional settings
#     int_params = {}
#     int_params['integrator'] = 'solve_ivp'
#     int_params['ode_integrator'] = 'DOP853'
#     int_params['intfcn'] = dyn.ode_twobody
#     int_params['rtol'] = 1e-12
#     int_params['atol'] = 1e-12
#     int_params['time_format'] = 'datetime'
    
#     # Sensor parameters
#     sensor_id_list = [sensor_id]
#     sensor_params = sens.define_sensors(sensor_id_list)
#     sensor_params['eop_alldata'] = eop_alldata
#     sensor_params['XYs_df'] = XYs_df
    
#     for sensor_id in sensor_id_list:
#         sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
#         sigma_dict = {}
#         sigma_dict['ra'] = noise*arcsec2rad   # rad
#         sigma_dict['dec'] = noise*arcsec2rad  # rad
#         sensor_params[sensor_id]['sigma_dict'] = sigma_dict
#         sensor_params[sensor_id]['moon_angle_lim'] = 0.
                
    
#     params_dict = {}
#     params_dict['sensor_params'] = sensor_params
#     params_dict['state_params'] = state_params
#     params_dict['int_params'] = int_params
       
    
    
#     ###########################################################################
#     # QZS1R
#     ###########################################################################
    
#     obj_id = qzs1r_norad
#     UTC0 = datetime(2022, 11, 7, 11, 0, 0)
#     state_dict = tle.propagate_TLE([obj_id], [UTC0], username=username,
#                                    password=password)
#     r0 = state_dict[obj_id]['r_GCRF'][0]
#     v0 = state_dict[obj_id]['v_GCRF'][0]
#     Xo = np.concatenate((r0, v0), axis=0)
    
#     tracklet_dict = \
#         mfunc.tracklet_generator(Xo, UTC0, dt_interval, dt_max, sensor_id,
#                                  params_dict, tracklet_dict, orbit_regime)
    
    
#     UTC = datetime(2022, 11, 8, 14, 0, 0)
#     tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
#     Xk = Xout[-1,:].reshape(6,1)
    
#     tracklet_dict = \
#         mfunc.tracklet_generator(Xk, UTC, dt_interval, dt_max, sensor_id,
#                                  params_dict, tracklet_dict, orbit_regime)
        
#     UTC = datetime(2022, 11, 9, 16, 0, 0)
#     tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
#     Xk = Xout[-1,:].reshape(6,1)
    
#     tracklet_dict = \
#         mfunc.tracklet_generator(Xk, UTC, dt_interval, dt_max, sensor_id,
#                                  params_dict, tracklet_dict, orbit_regime)
        
        
#     ###########################################################################
#     # QZS2
#     ###########################################################################
    
#     obj_id = qzs2_norad
#     UTC0 = datetime(2022, 11, 7, 11, 10, 0)
#     state_dict = tle.propagate_TLE([obj_id], [UTC0], username=username,
#                                    password=password)
#     r0 = state_dict[obj_id]['r_GCRF'][0]
#     v0 = state_dict[obj_id]['v_GCRF'][0]
#     Xo = np.concatenate((r0, v0), axis=0)
    
    
#     tracklet_dict = \
#         mfunc.tracklet_generator(Xo, UTC0, dt_interval, dt_max, sensor_id,
#                                  params_dict, tracklet_dict, orbit_regime)
    
    
#     UTC = datetime(2022, 11, 8, 14, 10, 0)
#     tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
#     Xk = Xout[-1,:].reshape(6,1)
    
#     tracklet_dict = \
#         mfunc.tracklet_generator(Xk, UTC, dt_interval, dt_max, sensor_id,
#                                  params_dict, tracklet_dict, orbit_regime)
        
#     UTC = datetime(2022, 11, 9, 16, 10, 0)
#     tout, Xout = dyn.general_dynamics(Xo, [UTC0, UTC], state_params, int_params)
#     Xk = Xout[-1,:].reshape(6,1)
    
#     tracklet_dict = \
#         mfunc.tracklet_generator(Xk, UTC, dt_interval, dt_max, sensor_id,
#                                  params_dict, tracklet_dict, orbit_regime)
        
    
    
    
    
    
#     print(tracklet_dict)
    
#     setup_file = os.path.join('test_cases', 'twobody_geo_10min_noise5.pkl')
#     pklFile = open( setup_file, 'wb' )
#     pickle.dump( [tracklet_dict, params_dict], pklFile, -1 )
#     pklFile.close()
    
    
    
#     return


# def test_tracklet_association():
    
#     # True orbits
#     qzs1r_norad = 49336
#     qzs2_norad = 42738
#     qzs3_norad = 42917
#     qzs4_norad = 42965
    
#     ses15_norad = 42709
#     amos5_norad = 37950
#     coms1_norad = 36744
    
    
#     # Load data
#     setup_file = os.path.join('test_cases', 'twobody_geo_10min_noise5.pkl')
#     pklFile = open(setup_file, 'rb' )
#     data = pickle.load( pklFile )
#     tracklet_dict = data[0]
#     params_dict = data[1]
#     pklFile.close()
    
    
#     ###########################################################################
#     # QZS-1R Self-Association
#     ###########################################################################
    
#     tracklet1 = tracklet_dict[0]
#     tracklet2 = tracklet_dict[1]
    
#     tk_list = [tracklet1['tk_list'][0], tracklet1['tk_list'][-1], tracklet2['tk_list'][-1]]
#     Yk_list = [tracklet1['Yk_list'][0], tracklet1['Yk_list'][-1], tracklet2['Yk_list'][-1]]
#     sensor_id_list = [tracklet1['sensor_id_list'][0],
#                       tracklet1['sensor_id_list'][-1],
#                       tracklet2['sensor_id_list'][-1]]
    
#     sensor_params = params_dict['sensor_params']
#     orbit_regime = tracklet1['orbit_regime']
    
#     print(tk_list)
#     print(Yk_list)
#     print(sensor_id_list)

    
#     # Execute function
#     X_list, M_list = iod.gooding_angles_iod(tk_list, Yk_list, sensor_id_list,
#                                             sensor_params, orbit_regime=orbit_regime,
#                                             search_mode='middle_out',
#                                             periapsis_check=True,
#                                             rootfind='zeros')
    

#     obj_id = qzs1r_norad
#     state_dict = tle.propagate_TLE([obj_id], tk_list)
    
#     r0 = state_dict[obj_id]['r_GCRF'][0]
#     v0 = state_dict[obj_id]['v_GCRF'][0]
#     Xo_true = np.concatenate((r0, v0), axis=0)
#     elem_true = astro.cart2kep(Xo_true)
    
#     print('QZS-1R Elem Truth: ', elem_true)
    
    
    
#     print('Final Answers')
#     print('X_list', X_list)
#     print('M_list', M_list)
    
#     for ii in range(len(M_list)):
        
#         elem_ii = astro.cart2kep(X_list[ii])
#         resids, ra_rms, dec_rms = \
#             compute_resids(X_list[ii], tk_list[0], tracklet1, tracklet2,
#                            params_dict)
        
        
        
#         print('')
#         print(ii)
#         print('Mi', M_list[ii])
#         print('Xi', X_list[ii])
#         print('elem', elem_ii)
#         print('QZS-1R Xo Err: ', np.linalg.norm(X_list[ii] - Xo_true))
#         print('RA Resids RMS [arcsec]: ', ra_rms)
#         print('DEC Resids RMS [arcsec]: ', dec_rms)
        
        
#     mistake
        

#     ###########################################################################
#     # QZS-2 Self-Association
#     ###########################################################################
# #    
# #    tracklet1 = tracklet_dict[3]
# #    tracklet2 = tracklet_dict[4]
# #    
# #    tk_list = [tracklet1['tk_list'][0], tracklet1['tk_list'][-1], tracklet2['tk_list'][-1]]
# #    Yk_list = [tracklet1['Yk_list'][0], tracklet1['Yk_list'][-1], tracklet2['Yk_list'][-1]]
# #    sensor_id_list = [tracklet1['sensor_id_list'][0],
# #                      tracklet1['sensor_id_list'][-1],
# #                      tracklet2['sensor_id_list'][-1]]
# #    
# #    sensor_params = params_dict['sensor_params']
# #    orbit_regime = tracklet1['orbit_regime']
# #    
# #    print(tk_list)
# #    print(Yk_list)
# #    print(sensor_id_list)
# #
# #    # Execute function
# #    X_list, M_list = iod.gooding_angles_iod(tk_list, Yk_list, sensor_id_list,
# #                                            sensor_params, orbit_regime=orbit_regime,
# #                                            search_mode='middle_out',
# #                                            periapsis_check=True,
# #                                            rootfind='zeros')
# #    
# #
# #    obj_id = qzs2_norad
# #    state_dict = tle.propagate_TLE([obj_id], tk_list)
# #    
# #    r0 = state_dict[obj_id]['r_GCRF'][0]
# #    v0 = state_dict[obj_id]['v_GCRF'][0]
# #    Xo_true = np.concatenate((r0, v0), axis=0)
# #    elem_true = astro.cart2kep(Xo_true)
# #    
# #    print('QZS-2 Elem Truth: ', elem_true)
# #    
# #    
# #    
# #    print('Final Answers')
# #    print('X_list', X_list)
# #    print('M_list', M_list)
# #    
# #    for ii in range(len(M_list)):
# #        
# #        elem_ii = astro.cart2kep(X_list[ii])
# #        resids, ra_rms, dec_rms = \
# #            compute_resids(X_list[ii], tk_list[0], tracklet1, tracklet2,
# #                           params_dict)
# #        
# #        print('')
# #        print(ii)
# #        print('Mi', M_list[ii])
# #        print('Xi', X_list[ii])
# #        print('elem', elem_ii)
# #        print('QZS-2 Xo Err: ', np.linalg.norm(X_list[ii] - Xo_true))
# #        print('RA Resids RMS [arcsec]: ', ra_rms)
# #        print('DEC Resids RMS [arcsec]: ', dec_rms)
        
        
#     ###########################################################################
#     # QZS-1R and QZS-2 Cross-Association
#     ###########################################################################
    
#     tracklet1 = tracklet_dict[0]
#     tracklet2 = tracklet_dict[5]
    
#     tk_list = [tracklet1['tk_list'][0], tracklet1['tk_list'][-1], tracklet2['tk_list'][-1]]
#     Yk_list = [tracklet1['Yk_list'][0], tracklet1['Yk_list'][-1], tracklet2['Yk_list'][-1]]
#     sensor_id_list = [tracklet1['sensor_id_list'][0],
#                       tracklet1['sensor_id_list'][-1],
#                       tracklet2['sensor_id_list'][-1]]
    
#     sensor_params = params_dict['sensor_params']
#     orbit_regime = tracklet1['orbit_regime']
    
#     print(tk_list)
#     print(Yk_list)
#     print(sensor_id_list)

#     # Execute function
#     X_list, M_list = iod.gooding_angles_iod(tk_list, Yk_list, sensor_id_list,
#                                             sensor_params, orbit_regime=orbit_regime,
#                                             search_mode='middle_out',
#                                             periapsis_check=True,
#                                             rootfind='zeros')
    

# #    obj_id = qzs2_norad
# #    state_dict = tle.propagate_TLE([obj_id], tk_list)
# #    
# #    r0 = state_dict[obj_id]['r_GCRF'][0]
# #    v0 = state_dict[obj_id]['v_GCRF'][0]
# #    Xo_true = np.concatenate((r0, v0), axis=0)
# #    elem_true = astro.cart2kep(Xo_true)
# #    
# #    print('QZS-2 Elem Truth: ', elem_true)
    
    
    
#     print('Final Answers')
#     print('X_list', X_list)
#     print('M_list', M_list)
    
#     for ii in range(len(M_list)):
        
#         elem_ii = astro.cart2kep(X_list[ii])
#         resids, ra_rms, dec_rms = \
#             compute_resids(X_list[ii], tk_list[0], tracklet1, tracklet2,
#                            params_dict)
        
#         print('')
#         print(ii)
#         print('Mi', M_list[ii])
#         print('Xi', X_list[ii])
#         print('elem', elem_ii)
# #        print('QZS-2 Xo Err: ', np.linalg.norm(X_list[ii] - Xo_true))
#         print('RA Resids RMS [arcsec]: ', ra_rms)
#         print('DEC Resids RMS [arcsec]: ', dec_rms)
    
    
#     return


# def compute_resids(Xo, UTC0, tracklet1, tracklet2, params_dict):
    
#     # Break out inputs
#     state_params = params_dict['state_params']
#     int_params = params_dict['int_params']
#     sensor_params = params_dict['sensor_params']
#     eop_alldata = sensor_params['eop_alldata']
#     XYs_df = sensor_params['XYs_df']
    
#     # Retrieve tracklet data
#     tk_list1 = tracklet1['tk_list']
#     Yk_list1 = tracklet1['Yk_list']
#     sensor_id_list1 = tracklet1['sensor_id_list']
    
#     tk_list2 = tracklet2['tk_list']
#     Yk_list2 = tracklet2['Yk_list']
#     sensor_id_list2 = tracklet2['sensor_id_list']
    
#     # Remove entries used to compute Gooding IOD solution
#     del tk_list1[-1]
#     del tk_list1[0]
#     del Yk_list1[-1]
#     del Yk_list1[0]
#     del sensor_id_list1[-1]
#     del sensor_id_list1[0]
#     del tk_list2[-1]
#     del Yk_list2[-1]
#     del sensor_id_list2[-1]
    
#     # Combine into single lists
#     tk_list1.extend(tk_list2)
#     Yk_list1.extend(Yk_list2)
#     sensor_id_list1.extend(sensor_id_list2)
    
#     # Propagate initial orbit, compute measurements and resids
#     resids = np.zeros((2, len(tk_list1)))
#     for kk in range(len(tk_list1)):
#         tk = tk_list1[kk]
#         Yk = Yk_list1[kk]
#         sensor_id = sensor_id_list1[kk]
#         EOP_data = eop.get_eop_data(eop_alldata, tk)
        
#         tout, Xout = dyn.general_dynamics(Xo, [UTC0, tk], state_params,
#                                           int_params)
#         Xk = Xout[-1,:].reshape(6,1)
#         Yprop = mfunc.compute_measurement(Xk, state_params, sensor_params,
#                                           sensor_id, tk, EOP_data, XYs_df)
        
#         diff = Yk - Yprop
#         if diff[0] > np.pi:
#             diff[0] -= 2.*np.pi
#         if diff[0] < -np.pi:
#             diff[0] += 2.*np.pi

#         resids[:,kk] = diff.flatten()
        
#     ra_resids = resids[0,:]
#     dec_resids = resids[1,:]
    
#     ra_rms = np.sqrt(np.dot(ra_resids, ra_resids))*(1./arcsec2rad)
#     dec_rms = np.sqrt(np.dot(dec_resids, dec_resids))*(1./arcsec2rad)
    
    
    

    
#     return resids, ra_rms, dec_rms


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


def geo_perturbed_setup(setup_file):
    
    # Object IDs
    qzs1r_norad = 49336
    qzs2_norad = 42738
    qzs3_norad = 42917
    qzs4_norad = 42965
    
    amos5_norad = 37950
    coms1_norad = 36744
    
    # Initial state vectors from TLE data
    obj_id_list = [qzs1r_norad, qzs2_norad, qzs3_norad, qzs4_norad,
                   amos5_norad, coms1_norad]
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
    state_params['bodies_to_create'] = ['Earth', 'Sun', 'Moon']
    state_params['global_frame_origin'] = 'Earth'
    state_params['global_frame_orientation'] = 'J2000'
    state_params['central_bodies'] = ['Earth']
    state_params['sph_deg'] = 8
    state_params['sph_ord'] = 8
    state_params['mass'] = 2000.
    state_params['Cd'] = 0.
    state_params['Cr'] = 1.2
    state_params['drag_area_m2'] = 0.1
    state_params['srp_area_m2'] = 40.
    
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


def choose_ids(obj):    
    
    # QZSS Satellite IDs
    qzs1_norad = 37158
    qzs2_norad = 42738
    qzs3_norad = 42917
    qzs4_norad = 42965
    qzs1r_norad = 49336
    
    qzs1_sp3_id = 'PJ01'
    qzs1r_sp3_id = 'PJ04'
    qzs2_sp3_id = 'PJ02'    
    qzs3_sp3_id = 'PJ07'    
    qzs4_sp3_id = 'PJ03'   
    
    if obj == 'qzs1':
        norad_id = qzs1_norad
        sp3_id = qzs1_sp3_id
    elif obj == 'qzs1r':
        norad_id = qzs1r_norad
        sp3_id = qzs1r_sp3_id
    elif obj == 'qzs2':
        norad_id = qzs2_norad
        sp3_id = qzs2_sp3_id
    elif obj == 'qzs3':
        norad_id = qzs3_norad
        sp3_id = qzs3_sp3_id
    elif obj == 'qzs4':
        norad_id = qzs4_norad
        sp3_id = qzs4_sp3_id
    
    return norad_id, sp3_id


def geo_real_tracklets(truth_file, meas_file):
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
    state_params['radius_m'] = 1.
    state_params['albedo'] = 0.1
    state_params['bodies_to_create'] = ['Earth', 'Sun', 'Moon']
    state_params['global_frame_origin'] = 'Earth'
    state_params['global_frame_orientation'] = 'J2000'
    state_params['central_bodies'] = ['Earth']
    state_params['sph_deg'] = 4
    state_params['sph_ord'] = 4
    state_params['mass'] = 2000.
    state_params['Cd'] = 0.
    state_params['Cr'] = 1.2
    state_params['drag_area_m2'] = 0.1
    state_params['srp_area_m2'] = 40.
    
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
    sensor_id = 'RMIT ROO'
    sensor_id_list = ['RMIT ROO']
    sensor_params = sens.define_sensors(sensor_id_list)
    sensor_params['eop_alldata'] = eop_alldata
    sensor_params['XYs_df'] = XYs_df
    
    for sensor_id in sensor_id_list:
        sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
        sigma_dict = {}
        sigma_dict['ra'] = 10.*arcsec2rad   # rad
        sigma_dict['dec'] = 10.*arcsec2rad  # rad
        sensor_params[sensor_id]['sigma_dict'] = sigma_dict
        sensor_params[sensor_id]['lam_clutter'] = 1.
        sensor_params[sensor_id]['p_det'] = 0.99
        FOV_hlim = sensor_params[sensor_id]['FOV_hlim']
        FOV_vlim = sensor_params[sensor_id]['FOV_vlim']        
        sensor_params[sensor_id]['V_sensor'] = (FOV_hlim[1] - FOV_hlim[0])*(FOV_vlim[1] - FOV_vlim[0])

    
    
    # Files and directories
    fdir = r'D:\documents\research_projects\iod\data\real\2022_11_GEO'
    
    csv_list = []
    sp3_list = []
    obj_id_list = []
    sp3_id_list = []
    
    date = '2022_11_07'
    obj = 'qzs1r'
    obj_id, sp3_id = choose_ids(obj)
    fname = os.path.join(fdir, date, obj, 'PNGs_angles.csv')
    csv_list.append(fname)
    fname = os.path.join(fdir, date, 'truth', 'qzf22351.sp3')
    sp3_list.append(fname)
    obj_id_list.append(obj_id)
    sp3_id_list.append(sp3_id)
    
    date = '2022_11_07'
    obj = 'qzs3'
    obj_id, sp3_id = choose_ids(obj)
    fname = os.path.join(fdir, date, obj, 'PNGs_angles.csv')
    csv_list.append(fname)
    fname = os.path.join(fdir, date, 'truth', 'qzf22351.sp3')
    sp3_list.append(fname)
    obj_id_list.append(obj_id)
    sp3_id_list.append(sp3_id)
    
    date = '2022_11_07'
    obj = 'qzs4'
    obj_id, sp3_id = choose_ids(obj)
    fname = os.path.join(fdir, date, obj, 'PNGs_angles.csv')
    csv_list.append(fname)
    fname = os.path.join(fdir, date, 'truth', 'qzf22351.sp3')
    sp3_list.append(fname)
    obj_id_list.append(obj_id)
    sp3_id_list.append(sp3_id)
    
    date = '2022_11_08'
    obj = 'qzs1r'
    obj_id, sp3_id = choose_ids(obj)
    fname = os.path.join(fdir, date, obj, 'PNGs_angles1.csv')
    csv_list.append(fname)
    fname = os.path.join(fdir, date, 'truth', 'qzf22352.sp3')
    sp3_list.append(fname)
    obj_id_list.append(obj_id)
    sp3_id_list.append(sp3_id)
    
    date = '2022_11_08'
    obj = 'qzs1r'
    obj_id, sp3_id = choose_ids(obj)
    fname = os.path.join(fdir, date, obj, 'PNGs_angles2.csv')
    csv_list.append(fname)
    fname = os.path.join(fdir, date, 'truth', 'qzf22352.sp3')
    sp3_list.append(fname)
    obj_id_list.append(obj_id)
    sp3_id_list.append(sp3_id)
    
    date = '2022_11_08'
    obj = 'qzs3'
    obj_id, sp3_id = choose_ids(obj)
    fname = os.path.join(fdir, date, obj, 'PNGs_angles1.csv')
    csv_list.append(fname)
    fname = os.path.join(fdir, date, 'truth', 'qzf22352.sp3')
    sp3_list.append(fname)
    obj_id_list.append(obj_id)
    sp3_id_list.append(sp3_id)
    
    date = '2022_11_08'
    obj = 'qzs3'
    obj_id, sp3_id = choose_ids(obj)
    fname = os.path.join(fdir, date, obj, 'PNGs_angles2.csv')
    csv_list.append(fname)
    fname = os.path.join(fdir, date, 'truth', 'qzf22352.sp3')
    sp3_list.append(fname)
    obj_id_list.append(obj_id)
    sp3_id_list.append(sp3_id)
    
    date = '2022_11_08'
    obj = 'qzs4'
    obj_id, sp3_id = choose_ids(obj)
    fname = os.path.join(fdir, date, obj, 'PNGs_angles1.csv')
    csv_list.append(fname)
    fname = os.path.join(fdir, date, 'truth', 'qzf22352.sp3')
    sp3_list.append(fname)
    obj_id_list.append(obj_id)
    sp3_id_list.append(sp3_id)
    
    date = '2022_11_08'
    obj = 'qzs4'
    obj_id, sp3_id = choose_ids(obj)
    fname = os.path.join(fdir, date, obj, 'PNGs_angles2.csv')
    csv_list.append(fname)
    fname = os.path.join(fdir, date, 'truth', 'qzf22352.sp3')
    sp3_list.append(fname)
    obj_id_list.append(obj_id)
    sp3_id_list.append(sp3_id)
        
    date = '2022_11_09'
    obj = 'qzs1r'
    obj_id, sp3_id = choose_ids(obj)
    fname = os.path.join(fdir, date, obj, 'PNGs_angles1.csv')
    csv_list.append(fname)
    fname = os.path.join(fdir, date, 'truth', 'qzf22353.sp3')
    sp3_list.append(fname)
    obj_id_list.append(obj_id)
    sp3_id_list.append(sp3_id)
    
    date = '2022_11_09'
    obj = 'qzs1r'
    obj_id, sp3_id = choose_ids(obj)
    fname = os.path.join(fdir, date, obj, 'PNGs_angles2.csv')
    csv_list.append(fname)
    fname = os.path.join(fdir, date, 'truth', 'qzf22353.sp3')
    sp3_list.append(fname)
    obj_id_list.append(obj_id)
    sp3_id_list.append(sp3_id)

    date = '2022_11_09'
    obj = 'qzs3'
    obj_id, sp3_id = choose_ids(obj)
    fname = os.path.join(fdir, date, obj, 'PNGs_angles.csv')
    csv_list.append(fname)
    fname = os.path.join(fdir, date, 'truth', 'qzf22353.sp3')
    sp3_list.append(fname)
    obj_id_list.append(obj_id)
    sp3_id_list.append(sp3_id)
    
    date = '2022_11_09'
    obj = 'qzs4'
    obj_id, sp3_id = choose_ids(obj)
    fname = os.path.join(fdir, date, obj, 'PNGs_angles.csv')
    csv_list.append(fname)
    fname = os.path.join(fdir, date, 'truth', 'qzf22353.sp3')
    sp3_list.append(fname)
    obj_id_list.append(obj_id)
    sp3_id_list.append(sp3_id)

    
    truth_dict = {}
    meas_dict = {}
    tracklet_dict = {}
    tracklet_id = 0
    for ind in range(len(csv_list)):
        
        csv_fname = csv_list[ind]
        sp3_fname = sp3_list[ind]
        obj_id = obj_id_list[ind]
        sp3_id = sp3_id_list[ind]

        meas_time_offset = 0.
        ra_bias = 0.
        dec_bias = 0.
        
        # Retrieve times and measurements
        UTC_list, ra_list, dec_list = ma.read_roo_csv_data(csv_fname, meas_time_offset, ra_bias, dec_bias)
        Zk_list = [np.reshape([ra_list[ii], dec_list[ii]], (2,1)) for ii in range(len(ra_list))]
        
        tracklet_dict[tracklet_id] = {}
        tracklet_dict[tracklet_id]['orbit_regime'] = 'GEO'
        tracklet_dict[tracklet_id]['obj_id'] = obj_id
        tracklet_dict[tracklet_id]['tk_list'] = UTC_list
        tracklet_dict[tracklet_id]['Zk_list'] = Zk_list
        tracklet_dict[tracklet_id]['sensor_id_list'] = [sensor_id]*len(UTC_list)
        
        tracklet_id += 1
        
        # Generate truth data
        r_eci_list = proc.interpolate_sp3_truth(UTC_list, sp3_fname, sp3_id,
                                                eop_alldata, XYs_df)
        
        for kk in range(len(UTC_list)):
            tk = UTC_list[kk]
            r_eci = r_eci_list[kk]
            Zk = Zk_list[kk]
            
            center = mfunc.compute_measurement(r_eci, state_params,
                                               sensor_params, sensor_id, tk,
                                               meas_types=['ra', 'dec'])
            

            
            if tk not in truth_dict:
                truth_dict[tk] = {}
            
            truth_dict[tk][obj_id] = np.reshape(r_eci, (3,1))
            
            if kk > 0 and kk < (len(UTC_list) -1):
                meas_dict[tk] = {}
                meas_dict[tk]['Zk_list'] = [Zk]
                meas_dict[tk]['sensor_id_list'] = [sensor_id]
                meas_dict[tk]['center_list'] = [center]
    
    print(tracklet_dict)
    
    
    # Save truth file and meas file
    pklFile = open( truth_file, 'wb' )
    pickle.dump( [truth_dict, state_params, int_params, sensor_params], pklFile, -1 )
    pklFile.close()
    
    pklFile = open( meas_file, 'wb' )
    pickle.dump( [tracklet_dict, meas_dict, sensor_params], pklFile, -1 )
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


def check_truth(truth_file):
    
    
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
    Xerr = np.zeros((len(Xo),len(tk_list)))
    a_plot = []
    e_plot = []
    i_plot = []
    raan_plot = []
    w_plot = []
    ta_plot = []
    for tk in tk_list:
        dt_sec = (tk - t0).total_seconds()
        tdays.append(dt_sec/86400.)
        Xnum = truth_dict[tk][obj_id]
        Xanal = astro.element_conversion(Xo, 1, 1, dt=dt_sec)
        
        kk = tk_list.index(tk)
        Xerr[:,kk] = (Xanal - Xnum).flatten()
        
        elem = astro.cart2kep(Xnum)
        a_plot.append(elem[0])
        e_plot.append(elem[1])
        i_plot.append(elem[2])
        raan_plot.append(elem[3])
        w_plot.append(elem[4])
        ta_plot.append(elem[5])
        
        
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
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tdays, a_plot, 'k.')
    plt.ylabel('SMA [km]')
    plt.subplot(3,1,2)
    plt.plot(tdays, e_plot, 'k.')
    plt.ylabel('ECC')
    plt.subplot(3,1,3)
    plt.plot(tdays, ta_plot, 'k.')
    plt.ylabel('TA [deg]')
    plt.xlabel('Time [days]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tdays, i_plot, 'k.')
    plt.ylabel('INC [deg]')
    plt.subplot(3,1,2)
    plt.plot(tdays, raan_plot, 'k.')
    plt.ylabel('RAAN [deg]')
    plt.subplot(3,1,3)
    plt.plot(tdays, w_plot, 'k.')
    plt.ylabel('AOP [deg]')
    plt.xlabel('Time [days]')
    
    plt.show()
    
    
    return


def consolidate_visibility():
    
    fdir = r'D:\documents\research_projects\iod\data\sim\test\aas2023_geo_6obj_7day'

    
    df_list = []
    for jj in range(1,15):
        fname = 'geo_perturbed_6obj_7day_visibility_' + str(jj) + '.csv'
        vis_file = os.path.join(fdir, fname)
        
        df = pd.read_csv(vis_file)
        df.reset_index(drop=True, inplace=True)
        df_list.append(df)
        
    vis_df = pd.concat(df_list, ignore_index=True, axis=0)
    
    
    # Object IDs
    qzs1r_norad = 49336
    qzs2_norad = 42738
    qzs3_norad = 42917
    qzs4_norad = 42965
    
    # ses15_norad = 42709
    amos5_norad = 37950
    coms1_norad = 36744
    
    # Initial state vectors from TLE data
    obj_id_list = [qzs1r_norad, qzs2_norad, qzs3_norad, qzs4_norad,
                   amos5_norad, coms1_norad]
    
    vis_dict = {}
    vis_dict['UTC'] = vis_df['Time [UTC]'].tolist()
    for obj_id in obj_id_list:
        vis_dict[obj_id] = vis_df[str(obj_id)].tolist()
    
    vis_df2 = pd.DataFrame.from_dict(vis_dict)
    
    print(vis_df2)
    
    vis_df2 = vis_df2.drop_duplicates()
    
    print(vis_df2)
    
    fname = 'geo_perturbed_6obj_7day_visibility.csv'
    outfile = os.path.join(fdir, fname)
    vis_df2.to_csv(outfile)
    
    
    return


def compute_obs_times(vis_file, pass_length, obs_time_file):
    
    vis_df = pd.read_csv(vis_file)
    
    # Object IDs
    qzs1r_norad = 49336
    qzs2_norad = 42738
    qzs3_norad = 42917
    qzs4_norad = 42965
    
    # ses15_norad = 42709
    amos5_norad = 37950
    coms1_norad = 36744
    
    
    obj_id_list = [qzs1r_norad, qzs2_norad, qzs3_norad, qzs4_norad,
                   amos5_norad, coms1_norad]
    
    
    # Initialize dict
    obs_times = {}
    for obj_id in obj_id_list:
        obs_times[obj_id] = {}
        obs_times[obj_id]['tk_list'] = []
        
        
    # For each day, find visible times and build out dictionary
    UTC_list = vis_df['UTC'].tolist()
    vis_df['UTC'] = [datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S') for dt_str in UTC_list]
    UTC0 = vis_df['UTC'].tolist()[0]

    date = datetime(UTC0.year, UTC0.month, UTC0.day)
    for ii in range(7):
        
        obs_times = single_day_obs_times(vis_df, date, obs_times, pass_length)
        
        
        obs_times = single_day_obs_times(vis_df, date, obs_times, pass_length)
        
        print('')
        print(obs_times)
        
        
        date = date + timedelta(days=1)
        
        
    print('\n\n')
    
    for obj_id in obs_times:
        
        print('\n')
        print('obj_id', obj_id)
        print(obs_times[obj_id]['tk_list'])
        
        
        
    # Save truth and params
    pklFile = open( obs_time_file, 'wb' )
    pickle.dump( [obs_times], pklFile, -1 )
    pklFile.close()
    
    return


def single_day_obs_times(vis_df, date, obs_times, pass_length):
    
    # Retrieve data from inputs
    obj_id_list = sorted(list(obs_times.keys()))

    # Reduce vis_df
    vis_df2 = vis_df.loc[(vis_df['UTC'] > date) & (vis_df['UTC'] < date + timedelta(days=1.))]
    UTC_list = vis_df2['UTC'].tolist()
    
    print(vis_df)
    print(vis_df2)
    
    # Delete previously used times
    del_list = []
    for obj_id in obs_times:
        tk_list = obs_times[obj_id]['tk_list']
        for tk in tk_list:
            
            if tk in UTC_list:
                # del_list.append(UTC_list.index(tk))
                del_list.append(int(vis_df2.index[vis_df2['UTC'] == tk][0]))
            
    if len(del_list) > 0:
        del_list = sorted(del_list)
        print(del_list)
        print(vis_df2)
        
        vis_df2 = vis_df2.drop(del_list)
        UTC_list = vis_df2['UTC'].tolist()
        
        print(vis_df2)
        
        print('')

    
    # Find number of visible times
    obj_ind_dict = {}
    ntimes_list = []
    for obj_id in obj_id_list:
        obj_ind_dict[obj_id] = [ind for ind, vis_flag in enumerate(vis_df2[str(obj_id)]) if vis_flag]
        ntimes_list.append(len(obj_ind_dict[obj_id]))
        
        
        # if obj_id == 36744 and date == datetime(2022,11,10):
        #     print(obj_ind_dict[obj_id])
        #     print(ntimes_list)
        #     mistake
        
    # Work backward from least to most visible object
    ntimes_inds = sorted(range(len(ntimes_list)), key=lambda k: ntimes_list[k])
    
    print(ntimes_list)
    print(ntimes_inds)
    
    sorted_obj = [obj_id_list[ii] for ii in ntimes_inds]
    
    print(obj_id_list)
    print(sorted_obj)
    
    # if date == datetime(2022,11,10):
    #     mistake
    
    while len(sorted_obj) > 0:
        
        obj_id = sorted_obj[0]
        print(obj_id)
        
        # Check there are at least enough entries
        min_entries = int(pass_length/10) + 1
        if len(obj_ind_dict[obj_id]) < min_entries:
            del sorted_obj[0]
            continue
        
        # Retrieve last obs time for this object
        tk_list = obs_times[obj_id]['tk_list']
        if len(tk_list) > 0:
            tk_prior = tk_list[-1]
        else:
            tk_prior = datetime(2000, 1, 1)
            
            
        # Select first available block with at least min_entries consecutive
        ind = 0
        test_block = [obj_ind_dict[obj_id][ind]]
        while len(test_block) < min_entries:
            
            if ind >= len(obj_ind_dict[obj_id]):
                break
            
            
            # print('obj_id', obj_id)
            # print('ind', ind)
            # print('len obj_ind_dict[obj_id]', len(obj_ind_dict[obj_id]))
            # print('obj_ind_dict[obj_id][ind]', obj_ind_dict[obj_id][ind])
            # print('UTC list', len(UTC_list))
            
            # Check if time is too close to previous pass
            tk_new = UTC_list[obj_ind_dict[obj_id][ind]]
            tdiff = (tk_new - tk_prior).total_seconds()/3600.
            # if tdiff > 3.5 and tdiff < 26:
            if tdiff < 3.5:
                ind += 1
                test_block = [obj_ind_dict[obj_id][ind]]
                continue

            
            print(tk_new)
            print(ind)
            
            if (UTC_list[obj_ind_dict[obj_id][ind+1]] - UTC_list[obj_ind_dict[obj_id][ind]]).total_seconds() < 11.:
                test_block.append(obj_ind_dict[obj_id][ind+1])
            else:
                test_block = [obj_ind_dict[obj_id][ind+1]]
                
            ind += 1
            
            
            
        if len(test_block) < min_entries:
            del sorted_obj[0]
            continue
            
        print(obj_id)
        print(test_block)
        
        # Store entries for this object and delete object from list
        tk_list_new = [UTC_list[kk] for kk in test_block]
        obs_times[obj_id]['tk_list'].extend(tk_list_new)
        del sorted_obj[0]
        
        print(obs_times)
        
        # Delete these entries from other objects
        for obj_id2 in sorted_obj:
            for del_ind in test_block:
                if del_ind in obj_ind_dict[obj_id2]:
                    del obj_ind_dict[obj_id2][obj_ind_dict[obj_id2].index(del_ind)]
            
            
        print(obj_ind_dict)
        
        
        
        
    print('')
    print(obs_times)
    

    
    
    return obs_times


def compute_obs_times2(vis_file, pass_length, obs_time_file):
    
    vis_df = pd.read_csv(vis_file)
    UTC_list = vis_df['UTC'].tolist()
    vis_df['UTC'] = [datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S') for dt_str in UTC_list]
    
    # Object IDs
    qzs1r_norad = 49336
    qzs2_norad = 42738
    qzs3_norad = 42917
    qzs4_norad = 42965
    
    # ses15_norad = 42709
    amos5_norad = 37950
    coms1_norad = 36744
    
    
    obj_id_list = [qzs1r_norad, qzs2_norad, qzs3_norad, qzs4_norad,
                   amos5_norad, coms1_norad]
    
    
    # Initialize dict
    obs_times = {}
    for obj_id in obj_id_list:
        obs_times[obj_id] = {}
        obs_times[obj_id]['tk_list'] = []
        
    # Choose obs times
    obj_id = qzs4_norad
    tk_list =      [datetime(2022,11,7,11,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)]
    tk_list.extend([datetime(2022,11,7,15,9,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,8,9,55,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,8,14,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,9,12,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,9,13,30,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,10,10,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,10,14,30,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,11,11,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,11,13,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,12,10,30,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,12,12,30,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,13,11,10,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,13,14,40,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    
    # Check visibility
    vis_df2 = vis_df.loc[(vis_df[str(obj_id)])]
    vis_times = vis_df2['UTC'].tolist()
    print(vis_df2)

    for tk in tk_list:
        if tk not in vis_times:
            print(obj_id, ' not visible at time ', tk)

    print(len(tk_list))
    
    obs_times[obj_id]['tk_list'] = tk_list
    
    
    obj_id = qzs1r_norad
    tk_list =      [datetime(2022,11,7,11,10,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)]
    tk_list.extend([datetime(2022,11,7,18,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,8,10,5,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,8,14,10,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,9,12,10,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,9,17,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,10,10,30,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,10,14,40,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,11,11,10,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,11,17,10,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,12,12,40,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,12,16,30,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,13,10,50,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,13,17,40,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    
    # Check visibility
    vis_df2 = vis_df.loc[(vis_df[str(obj_id)])]
    vis_times = vis_df2['UTC'].tolist()
    print(vis_df2)

    for tk in tk_list:
        if tk not in vis_times:
            print(obj_id, ' not visible at time ', tk)

    print(len(tk_list))
    
    obs_times[obj_id]['tk_list'] = tk_list
    
    
    obj_id = qzs2_norad
    tk_list =      [datetime(2022,11,7,13,40,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)]
    tk_list.extend([datetime(2022,11,7,17,50,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,8,14,20,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,8,16,30,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,9,14,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,9,17,10,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,10,14,50,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,10,18,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,11,13,25,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,11,17,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,12,14,20,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,12,16,40,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,13,13,50,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,13,17,50,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    
    # Check visibility
    vis_df2 = vis_df.loc[(vis_df[str(obj_id)])]
    vis_times = vis_df2['UTC'].tolist()
    print(vis_df2)

    for tk in tk_list:
        if tk not in vis_times:
            print(obj_id, ' not visible at time ', tk)

    print(len(tk_list))
    
    obs_times[obj_id]['tk_list'] = tk_list
    
    
    
    obj_id = qzs3_norad
    tk_list =      [datetime(2022,11,7,11,20,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)]
    tk_list.extend([datetime(2022,11,7,18,10,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,8,10,15,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,8,14,30,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,9,12,20,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,9,17,20,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,10,10,40,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,10,15,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,11,11,20,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,11,17,20,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,12,12,50,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,12,16,50,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,13,11,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,13,18,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    
    
    # Check visibility
    vis_df2 = vis_df.loc[(vis_df[str(obj_id)])]
    vis_times = vis_df2['UTC'].tolist()
    print(vis_df2)

    for tk in tk_list:
        if tk not in vis_times:
            print(obj_id, ' not visible at time ', tk)

    print(len(tk_list))
    
    obs_times[obj_id]['tk_list'] = tk_list
    
    
    obj_id = amos5_norad
    tk_list =      [datetime(2022,11,7,11,30,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)]
    tk_list.extend([datetime(2022,11,7,17,40,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,8,10,25,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,8,14,40,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,9,12,30,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,9,17,30,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,10,10,50,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,10,15,10,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,11,10,30,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,11,14,30,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,12,13,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,12,17,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,13,11,20,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,13,16,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    
    
    # Check visibility
    vis_df2 = vis_df.loc[(vis_df[str(obj_id)])]
    vis_times = vis_df2['UTC'].tolist()
    print(vis_df2)

    for tk in tk_list:
        if tk not in vis_times:
            print(obj_id, ' not visible at time ', tk)

    print(len(tk_list))
    
    obs_times[obj_id]['tk_list'] = tk_list
    
    
    
    obj_id = coms1_norad
    tk_list =      [datetime(2022,11,7,11,40,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)]
    tk_list.extend([datetime(2022,11,7,17,30,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,8,10,35,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,8,16,20,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,9,12,40,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,9,15,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,10,10,10,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,10,17,50,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,11,12,0,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,11,14,20,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,12,13,10,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,12,17,10,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,13,11,30,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    tk_list.extend([datetime(2022,11,13,15,30,0) + timedelta(seconds=ii) for ii in np.arange(0,pass_length+1,10)])
    
    
    # Check visibility
    vis_df2 = vis_df.loc[(vis_df[str(obj_id)])]
    vis_times = vis_df2['UTC'].tolist()
    print(vis_df2)

    for tk in tk_list:
        if tk not in vis_times:
            print(obj_id, ' not visible at time ', tk)

    print(len(tk_list))
    
    obs_times[obj_id]['tk_list'] = tk_list
    
        
     
        
        
    # Save truth and params
    pklFile = open( obs_time_file, 'wb' )
    pickle.dump( [obs_times], pklFile, -1 )
    pklFile.close()
    
    return


def generate_meas_file(noise, lam_c, p_det, orbit_regime, truth_file, obs_time_file, meas_file,
                       obj_id_list=[], reduce_meas=True):
    
    gap_length = 100.  # seconds
    
    # Load truth and observation time data
    pklFile = open(truth_file, 'rb' )
    data = pickle.load( pklFile )
    truth_dict = data[0]
    state_params = data[1]
    int_params = data[2]
    sensor_params = data[3]
    pklFile.close()
    
    pklFile = open(obs_time_file, 'rb' )
    data = pickle.load( pklFile )
    obs_times = data[0]
    pklFile.close()
    
    eop_alldata = sensor_params['eop_alldata']
    XYs_df = sensor_params['XYs_df']
    
    # Retrieve sensors
    sensor_id_list = list(sensor_params.keys())
    del sensor_id_list[sensor_id_list.index('eop_alldata')]
    del sensor_id_list[sensor_id_list.index('XYs_df')]
    
    # Update sensor params noise and lam_clutter
    for sensor_id in sensor_id_list:
        sensor_params[sensor_id]['sigma_dict']['ra'] = max(noise,1.)*arcsec2rad
        sensor_params[sensor_id]['sigma_dict']['dec'] = max(noise,1.)*arcsec2rad
        sensor_params[sensor_id]['lam_clutter'] = max(lam_c, 1)
        sensor_params[sensor_id]['p_det'] = min(p_det, 0.99)
    
    # Form obj_id_list from obs_times
    if len(obj_id_list) == 0:
        obj_id_list = sorted(list(obs_times.keys()))    
    
    
    
    # Loop over objects
    meas_dict = {}
    tracklet_dict = {}
    tracklet_id = 0
    for obj_id in obj_id_list:  
        
        
        # Form object sublist for later checks if in FOV
        obj_id_sublist = list(set(obj_id_list) - set([obj_id]))
        
        
        # Retrieve tk_list and form tracklet time sublists
        tk_list = obs_times[obj_id]['tk_list']
        
        print(obj_id)
        print(tk_list)
        # mistake

        
        tracklet_sublists = []
        for kk in range(len(tk_list)):
            
            if kk == 0:
                tk_tracklet = [tk_list[kk]]
                continue
            
            tdiff = (tk_list[kk] - tk_list[kk-1]).total_seconds()
            
            # Still part of same tracklet
            if tdiff < gap_length:
                tk_tracklet.append(tk_list[kk])
                
            # Tracklet has ended, start over
            else:
                tracklet_sublists.append(tk_tracklet)
                tk_tracklet = [tk_list[kk]]
                
            # If reached the end, store tracklet
            if kk == (len(tk_list) - 1):
                tracklet_sublists.append(tk_tracklet)
                
                
        print(obj_id)
        
        for tk_tracklet in tracklet_sublists:
            print('')
            print(tk_tracklet)

        
        # Loop over sensors
        for sensor_id in sensor_id_list:  
            
            
            # Sensor data
            # p_det = sensor_params[sensor_id]['p_det']
            # lam_clutter = sensor_params[sensor_id]['lam_clutter']
            FOV_hlim = sensor_params[sensor_id]['FOV_hlim']
            FOV_vlim = sensor_params[sensor_id]['FOV_vlim']
            
            # print(sensor_params[sensor_id])
            # mistake
            
            # Loop over tracklet times
            for tk_tracklet in tracklet_sublists:
                
                # Initialize tracklet data
                tracklet_dict[tracklet_id] = {}
                tracklet_dict[tracklet_id]['orbit_regime'] = orbit_regime
                tracklet_dict[tracklet_id]['obj_id'] = obj_id
                tracklet_dict[tracklet_id]['tk_list'] = []
                tracklet_dict[tracklet_id]['Zk_list'] = []
                tracklet_dict[tracklet_id]['sensor_id_list'] = []
                
                for tk in tk_tracklet:
                    
                    print('')
                    print(tk)
                    
                    # Initialize meas_dict entry
                    if tk in meas_dict:
                        print('meas_dict', meas_dict)
                        print(tk)
                        mistake
                        
                    meas_Zk = []
                    meas_sensor_id = []
                    meas_center = []
                    
                    # Retrieve true state
                    Xj = truth_dict[tk][obj_id]
            
                    # Check/confirm visibility
                    EOP_data = eop.get_eop_data(eop_alldata, tk)
                    if not visfunc.check_visibility(Xj, state_params, sensor_params,
                                                    sensor_id, tk, EOP_data, XYs_df):
                        
                        print('error not visible')
                        print('obj_id', obj_id)
                        print('tk', tk)
                        print('sensor_id', sensor_id)
                        
                        mistake
                
                    # Compute measurements
                    zj = mfunc.compute_measurement(Xj, state_params, sensor_params,
                                                   sensor_id, tk, EOP_data, XYs_df)
                    
                    # Save as center for this observation
                    center = zj.copy()
                    
                    # Add noise
                    zj[0] += np.random.randn()*noise*arcsec2rad
                    zj[1] += np.random.randn()*noise*arcsec2rad
                    
                    # Incorporate p_det for main object
                    if np.random.rand() <= p_det:
                        
                        # Store tracklet data
                        tracklet_dict[tracklet_id]['tk_list'].append(tk)
                        tracklet_dict[tracklet_id]['Zk_list'].append(zj)
                        tracklet_dict[tracklet_id]['sensor_id_list'].append(sensor_id)
                        
                        
                        # First and last entries in trackelts will be used for
                        # Gooding IOD solution. Skip these entries for the 
                        # meas_dict supplied to the filter to avoid double
                        # counting measurements
                        # if tk_tracklet.index(tk) == 0 or tk_tracklet.index(tk) == (len(tk_tracklet) - 1):
                        #     continue
                        
                        # print('add to meas dict')
                        
                        # Store meas data
                        meas_Zk.append(zj)
                        meas_sensor_id.append(sensor_id)
                        meas_center.append(center)
                        

                    # Check all other objects if visible and generate measurements
                    for obj_id2 in obj_id_sublist:
                        
                        print('check other obj visible')
                        print('obj_id2', obj_id2)
                        
                        # True object state
                        X2 = truth_dict[tk][obj_id2]
                    
                        # If object visible, apply p_det and noise and store meas_dict
                        if not visfunc.check_visibility(X2, state_params, sensor_params,
                                                        sensor_id, tk, EOP_data, XYs_df):
                            continue
                        
                        
                        print('object pass visibility check')
                        
                        z2 = mfunc.compute_measurement(X2, state_params, sensor_params,
                                                       sensor_id, tk, EOP_data, XYs_df)
                        # Angle rollover in RA
                        z2_test = z2 - center
                        if z2_test[0] > np.pi:
                            z2_test[0] -= 2.*np.pi
                        if z2_test[0] < -np.pi:
                            z2_test[0] += 2.*np.pi                        
                        
                        if (z2_test[0] < FOV_hlim[0] or z2_test[0] > FOV_hlim[1] 
                            or z2_test[1] < FOV_vlim[0] or z2_test[1] > FOV_vlim[1]):
                            
                            continue
                        
                        print('object is in FOV')
                        
                        
                        if np.random.rand() > p_det:
                            continue
                        
                        print('object detected!')
                        
                        # Add noise and store
                        z2[0] += np.random.randn()*noise*arcsec2rad
                        z2[1] += np.random.randn()*noise*arcsec2rad
                        
                        meas_Zk.append(z2)
                        meas_sensor_id.append(sensor_id)
                        meas_center.append(center)
                    
                    
                    # Generate clutter and store
                    n_clutter = ss.poisson.rvs(lam_c)

                    # Compute clutter meas in RA/DEC, uniform over FOV
                    for c_ind in range(n_clutter):
                        ra  = center[0] + (FOV_hlim[1]-FOV_hlim[0])*(np.random.rand() - 0.5)
                        dec = center[1] + (FOV_vlim[1]-FOV_vlim[0])*(np.random.rand() - 0.5)
                        
                        # Angle rollover in RA
                        if ra > np.pi:
                            ra -= 2.*np.pi
                        if ra < -np.pi:
                            ra += 2.*np.pi

                        zclutter = np.reshape([ra, dec], (2,1))
                        meas_Zk.append(zclutter)
                        meas_sensor_id.append(sensor_id)
                        meas_center.append(center)
                        
                    
                    # If measurements were collected, randomize order and store
                    if len(meas_Zk) > 0:
                        
                        inds = list(range(len(meas_Zk)))
                        random.shuffle(inds)
                        
                        meas_dict[tk] = {}
                        meas_dict[tk]['Zk_list'] = [meas_Zk[ii] for ii in inds]
                        meas_dict[tk]['sensor_id_list'] = [meas_sensor_id[ii] for ii in inds]
                        meas_dict[tk]['center_list'] = [meas_center[ii] for ii in inds]
                        
                        print('tk', tk)
                        print('meas dict tk')
                        print(meas_dict[tk])
                        print('')
                    
            
                # Summary and cleanup
                print('tracklet_dict entry')
                print(tracklet_dict[tracklet_id])
                
                # print(meas_dict.keys())
                
                # First and last entries in trackelts will be used for
                # Gooding IOD solution. Skip these entries for the 
                # meas_dict supplied to the filter to avoid double
                # counting measurements
                
                if reduce_meas:
                    t0 = tracklet_dict[tracklet_id]['tk_list'][0]
                    tf = tracklet_dict[tracklet_id]['tk_list'][-1]
                    del meas_dict[t0]
                    del meas_dict[tf]
                
                # print(meas_dict.keys())
                
                # mistake
                
                
                # Increment tracklet index
                tracklet_id += 1
    
    
    # Save data
    pklFile = open( meas_file, 'wb' )
    pickle.dump( [tracklet_dict, meas_dict, sensor_params], pklFile, -1 )
    pklFile.close()
    
    
    return


def check_meas_file(meas_file):
    
    # Load data
    pklFile = open(meas_file, 'rb' )
    data = pickle.load( pklFile )
    tracklet_dict = data[0]
    meas_dict = data[1]
    sensor_params = data[2]
    pklFile.close()
    
    
    t0 = datetime(2022, 11, 7, 11, 0, 0)
    
    sensor_id_list = list(sensor_params.keys())
    del sensor_id_list[sensor_id_list.index('eop_alldata')]
    del sensor_id_list[sensor_id_list.index('XYs_df')]
    
    # Tracklet data
    obj_id_list = []
    for tracklet_id in tracklet_dict:
        obj_id_list.append(tracklet_dict[tracklet_id]['obj_id'])
        
    obj_id_list = sorted(list(set(obj_id_list)))
    
    plot_dict = {}
    for tracklet_id in tracklet_dict:
        
        tracklet = tracklet_dict[tracklet_id]
        obj_id = tracklet['obj_id']
        tk_list = tracklet['tk_list']
        tracklet_sensor_id = tracklet['sensor_id_list']
        
        print('')
        print('tracklet_id', tracklet_id)
        print('obj_id', tracklet['obj_id'])
        print('orbit_regime', tracklet['orbit_regime'])
        print('nmeas', len(tracklet['Zk_list']))
        print('t0', tracklet['tk_list'][0])
        print('tf', tracklet['tk_list'][-1])
        
        if obj_id not in plot_dict:
            plot_dict[obj_id] = {}
            plot_dict[obj_id]['thrs'] = []
            plot_dict[obj_id]['sensor_index'] = []
            plot_dict[obj_id]['object_index'] = []
            
        for kk in range(len(tk_list)):
            tk = tk_list[kk]
            ind = sensor_id_list.index(tracklet_sensor_id[kk])
            plot_dict[obj_id]['thrs'].append((tk-t0).total_seconds()/3600.)
            plot_dict[obj_id]['sensor_index'].append(ind)
            plot_dict[obj_id]['object_index'].append(obj_id_list.index(obj_id))
            
            
        
        
        
    # Measurement and pre-fit residuals data
    meas_tk = sorted(meas_dict.keys())
    meas_tplot = [(tk - t0).total_seconds()/3600. for tk in meas_tk]
    nmeas_plot = []
    ra_plot = []
    dec_plot = []
    resids_tplot = []
    for tk in meas_tk:
            
        Zk_list = meas_dict[tk]['Zk_list']
        center_list = meas_dict[tk]['center_list']
        
        # sensor_id_list = meas_dict[tk]['sensor_id_list']
        # meas_sensor_id = meas_dict[tk]['sensor_id_list']
        # meas_sensor_index = [sensor_id_list.index(sensor_id) for sensor_id in meas_sensor_id]
        
        nmeas_plot.append(len(Zk_list))
        
        for ii in range(len(Zk_list)):
            zi = Zk_list[ii]
            center = center_list[ii]
            resid = (zi - center)*(1./arcsec2rad)
            ra_plot.append(resid[0])
            dec_plot.append(resid[1])
            resids_tplot.append((tk - t0).total_seconds()/3600.)
            
    
    
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    clist = ['r', 'g', 'b', 'c', 'k', 'y', 'm', 'c']
    for ii in range(len(obj_id_list)):
        obj_id = obj_id_list[ii]
        color = clist[ii]
        
        ax1.plot(plot_dict[obj_id]['thrs'], plot_dict[obj_id]['object_index'], '.', color=color)
        
    plt.xlabel('Time [hours] since ' + t0.strftime('%Y-%m-%d %H:%M:%S') + ' UTC')
    # ax1.set_yticks([0])
    # ax1.set_yticklabels(sensor_id_list, rotation=90, verticalalignment='center')
    ax1.set_yticks(list(range(len(obj_id_list))))
    ax1.set_yticklabels([str(obj_id) for obj_id in obj_id_list], rotation=0, verticalalignment='center')
    ax1.set_ylim([-1, len(obj_id_list) ])
    
    # plt.legend(obj_id_list)
    # plt.title('Tracklets')
    
    
    plt.figure()
    plt.plot(meas_tplot, nmeas_plot, 'k.')
    plt.ylabel('Number of Meas')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(resids_tplot, ra_plot, 'k.')
    plt.ylabel('RA [arcsec]')
    # plt.title('Prefit Residuals')
    plt.subplot(2,1,2)
    plt.plot(resids_tplot, dec_plot, 'k.')
    plt.ylabel('DEC [arcsec]')
    plt.xlabel('Time [hours] since ' + t0.strftime('%Y-%m-%d %H:%M:%S') + ' UTC')
    
                
    plt.show() 
    
    ra_mean = np.mean(ra_plot)
    ra_std = np.std(ra_plot)
    ra_rms = float(np.sqrt((1./len(ra_plot)) * sum([ra**2. for ra in ra_plot])))
    
    dec_mean = np.mean(dec_plot)
    dec_std = np.std(dec_plot)
    dec_rms = float(np.sqrt((1./len(dec_plot)) * sum([dec**2. for dec in dec_plot])))
    
    print('')
    print('Resids Mean, STD, and RMS')
    print('RA [arcsec]: ', ra_mean, ra_std, ra_rms)
    print('DEC [arcsec]: ', dec_mean, dec_std, dec_rms)
    
    
    return


def process_tracklets_full(meas_file, truth_file, csv_file, correlation_file):
    

    
    # Don't use sensor_params from truth file it has been updated for meas
    pklFile = open(truth_file, 'rb' )
    data = pickle.load( pklFile )
    truth_dict = data[0]
    state_params = data[1]
    int_params = data[2]
    pklFile.close()
        
    # Load measurement data and sensor params
    pklFile = open(meas_file, 'rb' )
    data = pickle.load( pklFile )
    tracklet_dict = data[0]
    meas_dict = data[1]
    sensor_params = data[2]
    pklFile.close()
    

    
    params_dict = {}
    params_dict['sensor_params'] = sensor_params
    params_dict['state_params'] = state_params
    params_dict['int_params'] = int_params
    
    
    # EOP data
    eop_alldata = sensor_params['eop_alldata']
    XYs_df = sensor_params['XYs_df']
    
    # Initialize output
    df_list = []
    correlation_dict = {}
    
    # Execution time
    gooding_time = 0.
    resids_time = 0.
    
    # Exclusion times
    exclude_short = 0.*3600.
    exclude_long = 3.*86400.
    

    # Loop through tracklets and compute association
    tracklet_id_list = list(tracklet_dict.keys())
    case_id = 0
    for ind in range(len(tracklet_id_list)):
        ii = tracklet_id_list[ind]
        tracklet_ii = tracklet_dict[ii]
        
        for ind2 in range(ind+1, len(tracklet_id_list)):            
        # for jj in tracklet_id_list[ii+1:]:
            
            jj = tracklet_id_list[ind2]
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
            Zk_list = [tracklet1['Zk_list'][0], tracklet1['Zk_list'][-1], tracklet2['Zk_list'][-1]]
            sensor_id_list = [tracklet1['sensor_id_list'][0],
                              tracklet1['sensor_id_list'][-1],
                              tracklet2['sensor_id_list'][-1]]
            
            orbit_regime = tracklet1['orbit_regime']
            
            # print(tracklet1['tk_list'])
            # print(tracklet2['tk_list'])
            
            print(tk_list)
            print(Zk_list)
            print(sensor_id_list)
            
            # Compute true association details
            obj_id = tracklet1['obj_id']
            Xo_true = truth_dict[tk_list[0]][obj_id]
            
            if len(Xo_true) == 6:
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
            X_list, M_list = iod.gooding_angles_iod(tk_list, Zk_list, sensor_id_list,
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
            tk_list1, Zk_list1, Rmat1 = reduce_tracklets(tracklet1, tracklet2,
                                                         params_dict)
            resids_time += time.time() - start
            
            
            for ind in range(len(M_list)):
                
                
                elem = astro.cart2kep(X_list[ind])
                start = time.time()
                resids, ra_rms, dec_rms = \
                    compute_resids(X_list[ind], tk_list[0], tk_list1, Zk_list1,
                                   Rmat1, params_dict)
                    
                resids_time += time.time() - start
                                    
                if len(Xo_true) == 3:
                    Xo_err = np.linalg.norm(X_list[ind][0:3] - Xo_true)
                else:
                    Xo_err = np.linalg.norm(X_list[ind] - Xo_true)
                
                # True correlation status
                if (tracklet1['obj_id'] == tracklet2['obj_id']): # and (M_list[ind] == M_true):
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
    pickle.dump( [correlation_dict, tracklet_dict, params_dict, truth_dict,
                  gooding_time, resids_time], pklFile, -1 )
    pklFile.close()
    
    
    return


def reduce_tracklets(tracklet1, tracklet2, params_dict):
    
    # Break out inputs
    # state_params = params_dict['state_params']
    # int_params = params_dict['int_params']
    sensor_params = params_dict['sensor_params']
    eop_alldata = sensor_params['eop_alldata']
    XYs_df = sensor_params['XYs_df']
    
    # Retrieve tracklet data
    tk_list1 = tracklet1['tk_list']
    Zk_list1 = tracklet1['Zk_list']
    sensor_id_list1 = tracklet1['sensor_id_list']
    
    tk_list2 = tracklet2['tk_list']
    Zk_list2 = tracklet2['Zk_list']
    sensor_id_list2 = tracklet2['sensor_id_list']
    
    # Combine into single lists
    tk_list1.extend(tk_list2)
    Zk_list1.extend(Zk_list2)
    sensor_id_list1.extend(sensor_id_list2)
    
    # Remove entries used to compute Gooding IOD solution
    del tk_list1[-1]
    del tk_list1[0]
    del Zk_list1[-1]
    del Zk_list1[0]
    del sensor_id_list1[-1]
    del sensor_id_list1[0]
    del tk_list2[-1]
    del Zk_list2[-1]
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

    
    return tk_list1, Zk_list1, Rmat


def compute_resids(Xo, UTC0, tk_list1, Zk_list1, Rmat1, params_dict):
    

    
    # Break out inputs
    # state_params = params_dict['state_params']
    # int_params = params_dict['int_params']
    # sensor_params = params_dict['sensor_params']
    
    # Propagate initial orbit, compute measurements and resids
    resids = np.zeros((2, len(tk_list1)))
    loop_start = time.time()
    eop_time = 0.
    astro_time = 0.
    meas_time = 0.
    for kk in range(len(tk_list1)):
        tk = tk_list1[kk]
        Zk = Zk_list1[kk]
        sensor_eci = Rmat1[:,kk].reshape(3,1)
        
        start = time.time()
        Xk = astro.element_conversion(Xo, 1, 1, dt=(tk-UTC0).total_seconds())
        astro_time += time.time() - start
        
        start = time.time()
        r_eci = Xk[0:3].reshape(3,1)
        rho_eci = r_eci - sensor_eci
        ra_calc = atan2(rho_eci[1], rho_eci[0])
        dec_calc = asin(rho_eci[2]/np.linalg.norm(rho_eci))
        Z_calc = np.reshape([ra_calc, dec_calc], (2,1))
        meas_time += time.time() - start
        
        
        diff = Zk - Z_calc
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


def tracklets_to_birth_model(correlation_file, ra_lim, dec_lim, birth_type='simple'):
    
    # Load correlation data
    pklFile = open(correlation_file, 'rb' )
    data = pickle.load( pklFile )
    correlation_dict = data[0]
    tracklet_dict = data[1]
    params_dict = data[2]
    truth_dict = data[3]
    pklFile.close()
    
    # Params
    state_params = params_dict['state_params']
    int_params = params_dict['int_params']
    
    gmm_params = {}
    gmm_params['prune_T'] = 1e-3
    gmm_params['merge_U'] = 36.
    
    
    # Initialize
    birth_time_dict = {}
    label_truth_dict = {}
    tk_meas_del = []
    # P_birth = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    
    # From Batch Fit
    
    X_birth = np.array([[ 3.17140426e+04],
                       [-1.04196373e+04],
                       [-2.06293835e+04],
                       [ 4.84449965e-01],
                       [ 3.18679816e+00],
                       [-6.87770665e-01]])
    
    
    P_birth = 100.*np.array([[ 5.10084093e-01, -1.87085288e-01, -3.21505063e-01,
                              -5.18406494e-05,  3.40826466e-06,  3.85866423e-05],
                            [-1.87085288e-01,  6.96794260e-02,  1.18108035e-01,
                              1.99392669e-05, -3.81221250e-06, -1.50122083e-05],
                            [-3.21505063e-01,  1.18108035e-01,  2.03867746e-01,
                              3.28755384e-05, -2.47623187e-06, -2.44474851e-05],
                            [-5.18406494e-05,  1.99392669e-05,  3.28755384e-05,
                              8.01900884e-09, -7.94260286e-09, -6.48798448e-09],
                            [ 3.40826466e-06, -3.81221250e-06, -2.47623187e-06,
                              -7.94260286e-09,  2.10729027e-08,  7.37667284e-09],
                            [ 3.85866423e-05, -1.50122083e-05, -2.44474851e-05,
                              -6.48798448e-09,  7.37667284e-09,  5.33221536e-09]])
    
    
    # # Initial covariance, compute by unscented transform
    # P_elem = np.diag([1., 1e-8, 0.0001, 0.0001, 0.0001, 0.0001])
    # m_elem = np.reshape([42164.1, 1e-3, 1., 10., 10., 10.], (6,1))
    # transform_fcn = est.unscented_kep2cart
    # dum, P_birth, dum2 = est.unscented_transform(m_elem, P_elem, transform_fcn, 
    #                                              {}, alpha=1e-4, pnorm=2.)
    
    # print(P_birth)
    # mistake
    
    
    # Simple model - just initialize near truth
    if birth_type == 'simple':
        
        tracklet_id_list = sorted(list(tracklet_dict.keys()))
        
        for tracklet_id in tracklet_id_list:
            
            # Retrieve tracklet data
            tracklet = tracklet_dict[tracklet_id]
            obj_id = tracklet['obj_id']
            tk_list = tracklet['tk_list']
            
            # Retrieve truth data for initial tracklet time
            tk = tk_list[1]
            X_true = truth_dict[tk][obj_id]
            
            # Perturbed initial filter state
            # pert_vect = np.multiply(np.sqrt(np.diag(P_birth)), np.random.randn(6))
            # X_init = X_true + np.reshape(pert_vect, (6, 1))
            
            # Generate and store birth component for this time
            birth_model = {}
            birth_model[1] = {}
            birth_model[1]['r_birth'] = 0.05
            birth_model[1]['weights'] = [1.]
            birth_model[1]['means'] = [X_birth]
            birth_model[1]['covars'] = [P_birth]
            
            birth_time_dict[tk] = birth_model
            
            label = (tk, 1)
            label_truth_dict[label] = obj_id
            
            break
            
            
    # GMM model using mean state from Gooding IOD
    if birth_type == 'gooding_gmm':
        
        # Get estimated correlation status
        corr_est_dict = analysis.evaluate_tracklet_correlation(correlation_file, ra_lim, dec_lim)
        
        # Build birth model
        for tracklet_id in corr_est_dict:
            
            # Set up propagation to second tracklet time
            t0 = tracklet_dict[tracklet_id]['tk_list'][0]
            tk = tracklet_dict[tracklet_id]['tk_list'][1]
            tin = [t0, tk]
            
            if t0 > datetime(2022, 11, 7, 14, 0, 0):
                continue
            
            # Retrieve and merge GMM components
            means0 = corr_est_dict[tracklet_id]['means']
            ncomp = len(means0)
            # weights0 = [1./ncomp]*ncomp
            # covars0 = [P_birth]*ncomp
            # GMM_dict = {}
            # GMM_dict['weights'] = weights0
            # GMM_dict['means'] = means0
            # GMM_dict['covars'] = covars0
            # GMM_dict = est.merge_GMM(GMM_dict, gmm_params)
            
            # means1 = GMM_dict['means']
            
            
            # Propagate Gooding IOD states to second tracklet time for birth model
            means = []
            
            for jj in range(ncomp):
                Xo = means0[jj]
                tout, intout = dyn.general_dynamics(Xo, tin, state_params, int_params)
                Xk = intout[-1,:].reshape(6,1)
                means.append(Xk)

            if tk in birth_time_dict:
                birth_id_list = sorted(list(birth_time_dict[tk].keys()))
                birth_id = birth_id_list[-1] + 1
            else:
                birth_id = 1                
                birth_time_dict[tk] = {}
                

            birth_time_dict[tk][birth_id] = {}
            birth_time_dict[tk][birth_id]['r_birth'] = 0.01
            birth_time_dict[tk][birth_id]['weights'] = [1./ncomp]*ncomp
            birth_time_dict[tk][birth_id]['means'] = means
            birth_time_dict[tk][birth_id]['covars'] = [P_birth]*ncomp

            label = (tk, birth_id)
            label_truth_dict[label] = tracklet_dict[tracklet_id]['obj_id']
            
            
    if birth_type == 'gooding_batch':
        
        # Get estimated correlation status
        corr_est_dict = analysis.evaluate_tracklet_correlation(correlation_file, ra_lim, dec_lim)
        
        # Build birth model
        for tracklet_id in corr_est_dict:
            
            # Set up propagation to second tracklet time
            t0 = tracklet_dict[tracklet_id]['tk_list'][0]
            tk = tracklet_dict[tracklet_id]['tk_list'][1]
            # tin = [t0, tk]
            
            if t0 > datetime(2022, 11, 8, 0, 0, 0):
                continue
            
            # Retrieve initial state
            means0 = corr_est_dict[tracklet_id]['means']
            ncomp = len(means0)
            
            print(corr_est_dict[tracklet_id])
            
            
            # Setup and run batch estimator
            Xo = means0[0]
            t0_batch = datetime(t0.year, t0.month, t0.day)
            tf_batch = t0_batch + timedelta(days=1.)
            X_birth, P_birth, tk_meas_del_ii = run_batch(Xo, tracklet_dict, params_dict, truth_dict,
                                                         t0_batch, tf_batch)
            tk_birth = tk_meas_del_ii[0]
            tk_meas_del.extend(tk_meas_del_ii)
        
            birth_model = {}
            birth_model[1] = {}
            birth_model[1]['r_birth'] = 0.01
            birth_model[1]['weights'] = [1.]
            birth_model[1]['means'] = [X_birth]
            birth_model[1]['covars'] = [P_birth]
            
            birth_time_dict[tk_birth] = birth_model
            
            label = (tk_birth, 1)
            label_truth_dict[label] = tracklet_dict[tracklet_id]['obj_id']
            
            obj_id = tracklet_dict[tracklet_id]['obj_id']
            X_true = truth_dict[tk_birth][obj_id]
            
            
            print('X_birth', X_birth)
            print('X_true', X_true)
        
    
    print(birth_time_dict)
    print(sorted(list(birth_time_dict.keys())))
    print(len(birth_time_dict.keys()))
    
    print(label_truth_dict)
    

    return birth_time_dict, label_truth_dict, tk_meas_del


def tudat_geo_lmb_setup_no_birth(truth_file, meas_file, setup_file):
    
    
    # Don't use sensor_params from truth file it has been updated for meas
    pklFile = open(truth_file, 'rb' )
    data = pickle.load( pklFile )
    truth_dict = data[0]
    state_params = data[1]
    int_params = data[2]
    pklFile.close()
        
    # Load measurement data and sensor params
    pklFile = open(meas_file, 'rb' )
    data = pickle.load( pklFile )
    tracklet_dict = data[0]
    meas_dict = data[1]
    sensor_params = data[2]
    pklFile.close()
    
    # Setup filter params
    # LMB Birth Model
    birth_model = {}
    
    # Filter parameters
    filter_params = {}
    filter_params['Q'] = 1e-15 * np.diag([1, 1, 1])
    filter_params['snc_flag'] = 'gamma'
    filter_params['gap_seconds'] = 900.
    filter_params['alpha'] = 1e-4
    filter_params['pnorm'] = 2.
    filter_params['prune_T'] = 1e-3
    filter_params['merge_U'] = 36.
    filter_params['H_max'] = 1000
    filter_params['H_max_birth'] = 5
    filter_params['T_max'] = 100
    filter_params['T_threshold'] = 1e-3
    filter_params['p_surv'] = 1.
    filter_params['birth_model'] = birth_model
    
    # Additions to other parameter dictionaries
    state_params['nstates'] = 6
    
    
    
    # Initial state LMB for filter
    tk_list = sorted(truth_dict.keys())
    obj_id_list = sorted(truth_dict[tk_list[0]].keys())
    if 42709 in obj_id_list:
        del obj_id_list[obj_id_list.index(42709)]
    
    print(obj_id_list)
    
    # Initial covariance, compute by unscented transform
    # P_elem = np.diag([1., 1e-8, 0.0001, 0.0001, 0.0001, 0.0001])
    # m_elem = np.reshape([42164.1, 1e-3, 1., 10., 10., 10.], (6,1))
    # transform_fcn = est.unscented_kep2cart
    # dum, P, dum2 = est.unscented_transform(m_elem, P_elem, transform_fcn, 
    #                                        {}, alpha=1e-4, pnorm=2.)
    
    # print(P)
    # print(np.sqrt(np.diag(P)))
    
    P = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    

    label_truth_dict = {}
    LMB_dict = {}
    ii = 1
    for obj_id in obj_id_list:
        
        X_true = truth_dict[tk_list[0]][obj_id]
        pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(6))
        X_init = X_true + np.reshape(pert_vect, (6, 1))
        
        LMB_dict[(tk_list[0], ii)] = {}
        LMB_dict[(tk_list[0], ii)]['r'] = 0.999
        LMB_dict[(tk_list[0], ii)]['weights'] = [1.]
        LMB_dict[(tk_list[0], ii)]['means'] = [X_init]
        LMB_dict[(tk_list[0], ii)]['covars'] = [P]
        
        label = (tk_list[0], ii)
        label_truth_dict[label] = obj_id
        
        ii += 1
        
        
    print(LMB_dict)

        
    state_dict = {}
    state_dict[tk_list[0]] = {}
    state_dict[tk_list[0]]['LMB_dict'] = LMB_dict
    
    
    birth_time_dict = {}
    

    # Save final setup file
    params_dict = {}
    params_dict['state_params'] = state_params
    params_dict['filter_params'] = filter_params
    params_dict['int_params'] = int_params
    params_dict['sensor_params'] = sensor_params
    
    meas_fcn = mfunc.unscented_radec
                
    # pklFile = open( setup_file, 'wb' )
    # pickle.dump( [state_dict, meas_fcn, meas_dict, params_dict, truth_dict], pklFile, -1 )
    # pklFile.close()
    
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [state_dict, meas_fcn, meas_dict, params_dict, truth_dict, birth_time_dict, label_truth_dict], pklFile, -1 )
    pklFile.close()
    
    
    return


def tudat_geo_lmb_setup_birth(truth_file, meas_file, correlation_file,
                              ra_lim, dec_lim, birth_type, setup_file):
    
    
    # Don't use sensor_params from truth file it has been updated for meas
    pklFile = open(truth_file, 'rb' )
    data = pickle.load( pklFile )
    truth_dict = data[0]
    state_params = data[1]
    int_params = data[2]
    pklFile.close()
        
    # Load measurement data and sensor params
    pklFile = open(meas_file, 'rb' )
    data = pickle.load( pklFile )
    tracklet_dict = data[0]
    meas_dict = data[1]
    sensor_params = data[2]
    pklFile.close()
    
    # Setup filter params
    # LMB Birth Model
    birth_time_dict, label_truth_dict, tk_meas_del = \
        tracklets_to_birth_model(correlation_file, ra_lim, dec_lim, birth_type)
        
    mistake
    
    # Filter parameters
    filter_params = {}
    filter_params['Q'] = 1e-15 * np.diag([1, 1, 1])
    filter_params['snc_flag'] = 'gamma'
    filter_params['gap_seconds'] = 900.
    filter_params['alpha'] = 1e-4
    filter_params['pnorm'] = 2.
    filter_params['prune_T'] = 1e-3
    filter_params['merge_U'] = 36.
    filter_params['H_max'] = 1000
    filter_params['H_max_birth'] = 5
    filter_params['T_max'] = 100
    filter_params['T_threshold'] = 1e-3
    filter_params['p_surv'] = 1.0
    # filter_params['birth_model'] = birth_model
    
    # Additions to other parameter dictionaries
    state_params['nstates'] = 6
    
    # Reduce measurement dictionary, remove entries used for IOD
    for tk in tk_meas_del:
        if tk in meas_dict:
            del meas_dict[tk]
    
    
    
    # Initial state LMB for filter
    tk_list = sorted(truth_dict.keys())
    obj_id_list = sorted(truth_dict[tk_list[0]].keys())
    if 42709 in obj_id_list:
        del obj_id_list[obj_id_list.index(42709)]
    
    print(obj_id_list)
    
    # Initial covariance, compute by unscented transform
    # P_elem = np.diag([1., 1e-8, 0.0001, 0.0001, 0.0001, 0.0001])
    # m_elem = np.reshape([42164.1, 1e-3, 1., 10., 10., 10.], (6,1))
    # transform_fcn = est.unscented_kep2cart
    # dum, P, dum2 = est.unscented_transform(m_elem, P_elem, transform_fcn, 
    #                                        {}, alpha=1e-4, pnorm=2.)
    
    # print(P)
    # print(np.sqrt(np.diag(P)))
    
    # P = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    

    
    # LMB_dict = {}
    # ii = 1
    # for obj_id in obj_id_list:
        
    #     X_true = truth_dict[tk_list[0]][obj_id]
    #     pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(6))
    #     X_init = X_true + np.reshape(pert_vect, (6, 1))
        
    #     LMB_dict[(tk_list[0], ii)] = {}
    #     LMB_dict[(tk_list[0], ii)]['r'] = 0.999
    #     LMB_dict[(tk_list[0], ii)]['weights'] = [1.]
    #     LMB_dict[(tk_list[0], ii)]['means'] = [X_init]
    #     LMB_dict[(tk_list[0], ii)]['covars'] = [P]
        
    #     ii += 1
        
        
    # print(LMB_dict)

    # Initialize empty
    state_dict = {}
    state_dict[tk_list[0]] = {}
    state_dict[tk_list[0]]['LMB_dict'] = {}
    
    
    
    # Save final setup file
    params_dict = {}
    params_dict['state_params'] = state_params
    params_dict['filter_params'] = filter_params
    params_dict['int_params'] = int_params
    params_dict['sensor_params'] = sensor_params
    
    meas_fcn = mfunc.unscented_radec
                
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [state_dict, meas_fcn, meas_dict, params_dict, truth_dict, birth_time_dict, label_truth_dict], pklFile, -1 )
    pklFile.close()
    
    
    return


def tudat_geo_setup_singletarget(mult_setup_file, obs_time_file, single_setup_file):
    
    
    # Don't use sensor_params from truth file it has been updated for meas
    pklFile = open(mult_setup_file, 'rb' )
    data = pickle.load( pklFile )
    state_dict = data[0]
    meas_fcn = data[1]
    meas_dict = data[2]
    params_dict = data[3]
    truth_dict = data[4]
    pklFile.close()
    
    pklFile = open(obs_time_file, 'rb' )
    data = pickle.load( pklFile )
    obs_times = data[0]
    pklFile.close()
        
    
    # Reformulate to fit single target
    obj_id = 37950
    
    # Date range
    t0 = datetime(2022, 11, 7, 0, 0, 0)
    tf = datetime(2022, 11, 12, 0, 0, 0)
    
    tk_truth = sorted(list(truth_dict.keys()))
    truth_dict2 = {}
    for tk in tk_truth:
        
        if tk > t0 and tk < tf:
            
            X = truth_dict[tk][obj_id]
            truth_dict2[tk] = X.copy()
    
    
    print(truth_dict2.keys())
    
    tk_meas = sorted(list(meas_dict.keys()))
    meas_dict2 = {}
    meas_dict2['tk_list'] = []
    meas_dict2['Yk_list'] = []
    meas_dict2['sensor_id_list'] = []
    for tk in tk_meas:
        
        if tk > t0 and tk < tf:
            
            if tk in obs_times[obj_id]['tk_list']:
                
                Zk_list = meas_dict[tk]['Zk_list']
                sensor_id_list = meas_dict[tk]['sensor_id_list']
                
                if len(Zk_list) == 1:
                    Yk = Zk_list[0]
                    sensor_id = sensor_id_list[0]
                    meas_dict2['tk_list'].append(tk)
                    meas_dict2['Yk_list'].append(Yk)
                    meas_dict2['sensor_id_list'].append(sensor_id)
                    
                    
                else:
                    mistake
                
                
    print(meas_dict2)
    
    
    
    
    # Initial state dict
    # LMB_dict = state_dict[tk_truth[0]]['LMB_dict']
    # weights = LMB_dict[(tk_truth[0], 2)]['weights']
    # means = LMB_dict[(tk_truth[0], 2)]['means']
    # covars = LMB_dict[(tk_truth[0], 2)]['covars']

    P = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    Xo_true = truth_dict2[tk_truth[0]]
    pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(6))
    X_init = Xo_true + np.reshape(pert_vect, (6, 1))
        
    state_dict2 = {}
    state_dict2[tk_truth[0]] = {}
    state_dict2[tk_truth[0]]['X'] = X_init
    state_dict2[tk_truth[0]]['P'] = P
    
    
    
    # # Save final setup file
    # params_dict = {}
    # params_dict['state_params'] = state_params
    # params_dict['filter_params'] = filter_params
    # params_dict['int_params'] = int_params
    # params_dict['sensor_params'] = sensor_params
    
    # meas_fcn = mfunc.unscented_radec
                
    pklFile = open( single_setup_file, 'wb' )
    pickle.dump( [state_dict2, meas_fcn, meas_dict2, params_dict, truth_dict2], pklFile, -1 )
    pklFile.close()
    
    
    return


def tudat_geo_setup_singletarget2(truth_file, meas_file, obj_id, truth_file2, meas_file2):
    
    # Don't use sensor_params from truth file it has been updated for meas
    pklFile = open(truth_file, 'rb' )
    data = pickle.load( pklFile )
    truth_dict = data[0]
    state_params = data[1]
    int_params = data[2]
    pklFile.close()
        
    # Load measurement data and sensor params
    pklFile = open(meas_file, 'rb' )
    data = pickle.load( pklFile )
    tracklet_dict = data[0]
    meas_dict = data[1]
    sensor_params = data[2]
    pklFile.close()
    
    
    
    # Reduce truth, tracklet, meas dictionaries to single object
    tracklet_id_list = list(tracklet_dict.keys())
    tk_tracklet_full = []
    for tracklet_id in tracklet_id_list:
        if tracklet_dict[tracklet_id]['obj_id'] != obj_id:
            del tracklet_dict[tracklet_id]
        else:
            tk_tracklet_full.extend(tracklet_dict[tracklet_id]['tk_list'])
            
            
    print('tk tracklet full', tk_tracklet_full)
            
    tk_truth = sorted(list(truth_dict.keys()))  
    for tk in tk_truth:
        obj_id_list = list(truth_dict[tk].keys())
        
        for obj_jj in obj_id_list:
            if obj_jj != obj_id:
                del truth_dict[tk][obj_jj]
                
        if tk in meas_dict and tk not in tk_tracklet_full:
            del meas_dict[tk]
            
    
    print(meas_dict)
    print(tracklet_dict)
        
      
        
    # Save truth and params
    pklFile = open( truth_file2, 'wb' )
    pickle.dump( [truth_dict, state_params, int_params, sensor_params], pklFile, -1 )
    pklFile.close()
    
    # Save data
    pklFile = open( meas_file2, 'wb' )
    pickle.dump( [tracklet_dict, meas_dict, sensor_params], pklFile, -1 )
    pklFile.close()
    
    
    return


def run_batch(Xo, tracklet_dict, params_dict, truth_dict, t0, tf):
    
    
    # t0 = datetime(2022, 11, 7, 0, 0, 0)
    # tf = datetime(2022, 11, 8, 0, 0, 0)
    
    
    # Reformat truth data for single target filter
    tk_truth = sorted(list(truth_dict.keys()))
    truth_dict2 = {}
    for tk in tk_truth:
        
        if tk > t0 and tk < tf:
            
            X = truth_dict[tk][obj_id]
            truth_dict2[tk] = X.copy()
        
        
    # Form measurement dict from tracklet dict    
    meas_dict = {}
    meas_dict['tk_list'] = []
    meas_dict['Yk_list'] = []
    meas_dict['sensor_id_list'] = []
    tracklet_id_list = sorted(list(tracklet_dict.keys()))
    for tracklet_id in tracklet_id_list:
        
        tk_list = tracklet_dict[tracklet_id]['tk_list']
        
        if tk_list[0] > t0 and tk_list[-1] < tf:
            
            Zk_list = tracklet_dict[tracklet_id]['Zk_list']
            sensor_id_list = tracklet_dict[tracklet_id]['sensor_id_list']
            
            ind_list = list(np.arange(0,len(tk_list), 2))
            last_ind = len(tk_list)-1
            if last_ind not in ind_list:
                ind_list.append(last_ind)
            
            meas_dict['tk_list'].extend([tk_list[ii] for ii in ind_list])
            meas_dict['Yk_list'].extend([Zk_list[ii] for ii in ind_list])
            meas_dict['sensor_id_list'].extend([sensor_id_list[ii] for ii in ind_list])
            
    
    t0_state = sorted(meas_dict['tk_list'])[0]
    
    
    # Setup state dict    
    state_dict = {}
    state_dict[t0_state] = {}
    state_dict[t0_state]['X'] = Xo
    state_dict[t0_state]['P'] = 100.*np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    
    
    # Update params
    int_params = {}
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = dyn.ode_twobody_stm
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'
    
    
    meas_fcn = mfunc.H_radec
    # params_dict['filter_params']['alpha'] = 1e-4
    # params_dict['filter_params']['Q'] = 1e-15 * np.diag([1, 1, 1])
    # params_dict['int_params']['tudat_integrator'] = 'rkf78'
    
    params_dict['int_params'] = int_params
    
    
    filter_output, full_state_output = est.ls_batch(state_dict, truth_dict2, meas_dict, meas_fcn, params_dict)    
    analysis.compute_orbit_errors(filter_output, full_state_output, truth_dict2)
    
    t0_truth = sorted(truth_dict2.keys())[0]
    
    print('')
    print('t0_truth', t0_truth)
    print(full_state_output[t0_truth])
    
    print('')
    print('t0_state', t0_state)
    print(full_state_output[t0_state])
    
    X_birth = filter_output[t0_state]['X']
    P_birth = 100.*filter_output[t0_state]['P']
    tk_meas_del = meas_dict['tk_list']
    
    return X_birth, P_birth, tk_meas_del


def run_singletarget_filter(setup_file):
    
        
    pklFile = open(setup_file, 'rb' )
    data = pickle.load( pklFile )
    state_dict = data[0]
    meas_fcn = data[1]
    meas_dict = data[2]
    params_dict = data[3]
    truth_dict = data[4]
    pklFile.close()
    

    
    # # Don't use sensor_params from truth file it has been updated for meas
    # pklFile = open(truth_file, 'rb' )
    # data = pickle.load( pklFile )
    # truth_dict = data[0]
    # state_params = data[1]
    # int_params = data[2]
    # pklFile.close()
        
    # # Load measurement data and sensor params
    # pklFile = open(meas_file, 'rb' )
    # data = pickle.load( pklFile )
    # tracklet_dict = data[0]
    # meas_dict = data[1]
    # sensor_params = data[2]
    # pklFile.close()
    
    t0 = datetime(2022, 11, 7, 0, 0, 0)
    tf = datetime(2022, 11, 8, 0, 0, 0)
    
    
    # Reformat truth data for single target filter
    tk_truth = sorted(list(truth_dict.keys()))
    truth_dict2 = {}
    for tk in tk_truth:
        
        if tk > t0 and tk < tf:
            
            X = truth_dict[tk][obj_id]
            truth_dict2[tk] = X.copy()
        
        
    # Reformat meas_dict for single target filter
    
    
    tk_meas = sorted(list(meas_dict.keys()))
    meas_dict2 = {}
    meas_dict2['tk_list'] = []
    meas_dict2['Yk_list'] = []
    meas_dict2['sensor_id_list'] = []
    for tk in tk_meas:
        
        if tk > t0 and tk < tf:
            
            Zk_list = meas_dict[tk]['Zk_list']
            sensor_id_list = meas_dict[tk]['sensor_id_list']
            
            if len(Zk_list) == 1:
                Yk = Zk_list[0]
                sensor_id = sensor_id_list[0]
                meas_dict2['tk_list'].append(tk)
                meas_dict2['Yk_list'].append(Yk)
                meas_dict2['sensor_id_list'].append(sensor_id)
                
                
            else:
                mistake
    
    
    # Reformat state dict
    t0_state = sorted(state_dict.keys())[0]    
    LMB_dict = state_dict[t0_state]['LMB_dict']
    label = list(LMB_dict.keys())[0]
    X_init = LMB_dict[label]['means'][0]
    P = LMB_dict[label]['covars'][0]
    
    state_dict2 = {}
    state_dict2[t0_state] = {}
    state_dict2[t0_state]['X'] = X_init
    state_dict2[t0_state]['P'] = P
    
    
    # Update params
    int_params = {}
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = dyn.ode_twobody_stm
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'
    
    
    meas_fcn = mfunc.H_radec
    # params_dict['filter_params']['alpha'] = 1e-4
    # params_dict['filter_params']['Q'] = 1e-15 * np.diag([1, 1, 1])
    # params_dict['int_params']['tudat_integrator'] = 'rkf78'
    
    params_dict['int_params'] = int_params
    
    
    # # Reduced dynamics model
    # state_params = params_dict['state_params']
    # state_params['bodies_to_create'] = ['Earth', 'Sun', 'Moon']
    # state_params['sph_deg'] = 2
    # state_params['sph_ord'] = 0
    # state_params['mass'] = 400.
    # state_params['Cd'] = 2.2
    # state_params['Cr'] = 1.5
    # state_params['drag_area_m2'] = 4.
    # state_params['srp_area_m2'] = 4.
    
    # params_dict['state_params'] = state_params
    
    # print(params_dict['state_params'])
    
    # mistake
    
    filter_output, full_state_output = est.ls_batch(state_dict2, truth_dict2, meas_dict2, meas_fcn, params_dict)    
    analysis.compute_orbit_errors(filter_output, full_state_output, truth_dict2)
    
    t0_truth = sorted(truth_dict2.keys())[0]
    
    print('')
    print('t0_truth', t0_truth)
    print(full_state_output[t0_truth])
    
    print('')
    print('t0_meas', tk_meas[0])
    print(full_state_output[tk_meas[0]])
    
    # UKF Test
    # filter_output, full_state_output = est.ls_ukf(state_dict2, truth_dict2, meas_dict2, meas_fcn, params_dict)
    # analysis.compute_orbit_errors(filter_output, filter_output, truth_dict2)
    
    # Unscented Batch Test
    # filter_output, full_state_output = est.unscented_batch(state_dict2, truth_dict2, meas_dict2, meas_fcn, params_dict)
    # analysis.compute_orbit_errors(filter_output, full_state_output, truth_dict2)
        
    
    return


def run_multitarget_filter(setup_file, prev_results, results_file):
    
    # Load setup
    pklFile = open(setup_file, 'rb' )
    data = pickle.load( pklFile )
    state_dict = data[0]
    meas_fcn = data[1]
    meas_dict = data[2]
    params_dict = data[3]
    truth_dict = data[4]
    tracklet_dict = data[5]
    # birth_time_dict = data[5]
    # label_truth_dict = data[6]
    pklFile.close()
    
    # # Load previous results and reset state_dict
    # pklFile = open(prev_results, 'rb' )
    # data = pickle.load( pklFile )
    # state_dict = data[0]
    # pklFile.close()
    
    
    # Reduce meas and birth dict to times of interest
    t0 = datetime(2022, 11, 7, 0, 0, 0)
    tf = datetime(2022, 11, 10, 0, 0, 0)
    tk_list = sorted(list(meas_dict.keys()))
    tk_list2 = sorted(list(birth_time_dict.keys()))

    
    for tk in tk_list:
        if tk < t0 or tk > tf:
            del meas_dict[tk]
            
    for tk in tk_list2:
        if tk < t0 or tk > tf:
            del birth_time_dict[tk]
            
    print(meas_dict.keys())
    print(len(meas_dict.keys()))
    
    print(meas_dict.keys())
    print(birth_time_dict.keys())
    

    filter_output, full_state_output = mult.lmb_filter(state_dict, truth_dict, meas_dict, tracklet_dict, meas_fcn, params_dict)
    
    
    pklFile = open( results_file, 'wb' )
    pickle.dump( [filter_output, full_state_output, params_dict, truth_dict, label_truth_dict], pklFile, -1 )
    pklFile.close()

    
    return


def combine_results():
    
    
    fdir = r'D:\documents\research_projects\iod\data\sim\test\aas2023_geo_6obj_7day\2023_01_02_geo_twobody_fixed_corr'
    

    filter_output_full = {}
    
    for ii in range(1,3):
    
        fname = 'geo_twobody_6obj_7day_goodingbirth_results_' + str(ii) + '.pkl'
        results_file = os.path.join(fdir, fname)
        
        pklFile = open(results_file, 'rb' )
        data = pickle.load( pklFile )
        filter_output = data[0]
        full_state_output = data[1]
        params_dict = data[2]
        truth_dict = data[3]
        label_truth_dict = data[4]
        pklFile.close()
        
        filter_output_full.update(filter_output)
        
    full_state_output = filter_output_full
        
    fname = 'geo_twobody_6obj_7day_goodingbirth_results_full.pkl'
    full_results_file = os.path.join(fdir, fname)
    
    pklFile = open( full_results_file, 'wb' )
    pickle.dump( [filter_output_full, full_state_output, params_dict, truth_dict, label_truth_dict], pklFile, -1 )
    pklFile.close()
    
    
    return


def multitarget_analysis(results_file, setup_file):
    
    pklFile = open(results_file, 'rb' )
    data = pickle.load( pklFile )
    filter_output = data[0]
    full_state_output = data[1]
    params_dict = data[2]
    truth_dict = data[3]
    label_truth_dict = data[4]
    pklFile.close()
    
    # Load setup
    pklFile = open(setup_file, 'rb' )
    data = pickle.load( pklFile )
    meas_dict = data[2]
    pklFile.close()
    
    
    
    
    analysis.lmb_orbit_errors2(filter_output, filter_output, truth_dict, meas_dict, label_truth_dict)
    
    return



if __name__ == '__main__':
    
    plt.close('all')
    
    
#    leo_tracklets_marco()
    
#    geo_tracklets()
    
    # test_tracklet_association()
    
    fdir = r'D:\documents\research_projects\iod\data\aas2023_preprint'
    truthdir = os.path.join(fdir, 'truth')
    visdir = os.path.join(fdir, 'visibility')
    measdir = os.path.join(fdir, 'meas')
    trackdir = os.path.join(fdir, 'tracklet_corr')
    # filterdir = r'D:\documents\research_projects\iod\data\aas2023_preprint\geo_twobody_1obj_3day'
    filterdir = r'D:\documents\research_projects\iod\data\sim\test\aas2023_geo_6obj_7day\2023_01_12_geo_twobody_singletarget'
    
    
    # fname = 'geo_twobody_6obj_7day_truth_13.pkl'    
    # prev_file = os.path.join(fdir, fname)
    
    # fname = 'geo_twobody_6obj_7day_visibility.csv'
    # vis_file = os.path.join(visdir, fname)
    
    # fname = 'geo_twobody_6obj_7day_truth.pkl'    
    # truth_file = os.path.join(truthdir, fname)
    
    fname = 'geo_twobody_1obj_7day_truth.pkl'    
    truth_file = os.path.join(filterdir, fname)
    
    
    fname = 'geo_twobody_6obj_7day_obstime_2pass_300sec.pkl'
    obs_time_file = os.path.join(filterdir, fname)
    
    # fname = 'geo_twobody_6obj_7day_meas_2pass_300sec_noise1_lam0_pd1.pkl'
    # fname = 'geo_real_3obj_3day_meas.pkl'
    # meas_file = os.path.join(measdir, fname)
    
    fname = 'geo_twobody_1obj_7day_meas_2pass_300sec_noise1_lam0_pd1.pkl'
    meas_file = os.path.join(filterdir, fname)
    
    
    
    fname = 'geo_real_3obj_3day_corr_summary.csv'
    corr_csv = os.path.join(trackdir, fname)
    
    # fname = 'geo_real_3obj_3day_corr.pkl'
    
    fname = 'geo_twobody_1obj_7day_corr_2pass_300sec_noise1_lam0_pd1.pkl'
    corr_pkl = os.path.join(filterdir, fname)
    
    fname = 'geo_twobody_1obj_7day_setup_noise1_lam0_pd1_batchbirth.pkl'
    setup_file = os.path.join(filterdir, fname)  
    
    
    # fname = 'geo_twobody_1obj_7day_batchbirth3_results_1.pkl'
    # prev_results = os.path.join(filterdir, fname)
    
    # fname = 'geo_twobody_1obj_7day_batchbirth_results_1.pkl'
    # results_file = os.path.join(filterdir, fname)
    
    
    
    
    # geo_perturbed_setup(setup_file)
    
    
    
    # tracklet_visibility(vis_file, prev_file, truth_file)
    
    # check_truth(truth_file)
    
    # consolidate_visibility()
    
    pass_length = 300.
    # compute_obs_times2(vis_file, pass_length, obs_time_file)
    
    # pklFile = open(obs_time_file, 'rb' )
    # data = pickle.load( pklFile )
    # obs_times = data[0]
    # pklFile.close()
    
    
    # print(obs_times)
    
    ###########################################################################
    # Tracklets and Measurements
    ###########################################################################
    
    # geo_real_tracklets(truth_file, meas_file)
    
    obj_id = 49336
    # tudat_geo_setup_singletarget2(truth_file, meas_file, obj_id, truth_file2, meas_file2)
    
    
    noise = 1.
    lam_c = 0.
    p_det = 1.
    orbit_regime = 'GEO'
    obj_id_list = [49336]
    # generate_meas_file(noise, lam_c, p_det, orbit_regime, truth_file,
    #                    obs_time_file, meas_file, obj_id_list, reduce_meas=False)
    
    # check_meas_file(meas_file)
    
    
    
    ###########################################################################
    # Tracklet Correlation
    ###########################################################################
    
    
    # process_tracklets_full(meas_file2, truth_file2, corr_csv, corr_pkl)
    
    ra_lim = 50.
    dec_lim = 50.
    
    plot_flag = True
    
    # corr_est_dict = analysis.evaluate_tracklet_correlation(corr_pkl, ra_lim, dec_lim, plot_flag)
    
    lim_list = [50., 200., 500.]
    # analysis.boxplot_corr_errors(corr_pkl, lim_list, True)
    
    
    
    
    ###########################################################################
    # Filter Setup
    ###########################################################################
    
    # birth_type = 'gooding_gmm'
    
    # birth_time_dict, label_truth_dict = tracklets_to_birth_model(corr_pkl, ra_lim, dec_lim, birth_type)
    
    # print(birth_time_dict)
    
    
    # tudat_geo_lmb_setup_no_birth(truth_file2, meas_file2, setup_file)
    
    
    
    ra_lim = 50.
    dec_lim = 50.
    birth_type = 'gooding_batch'
    # tudat_geo_lmb_setup_birth(truth_file2, meas_file2, corr_pkl,
    #                           ra_lim, dec_lim, birth_type, setup_file)
    
    
    # fname = 'geo_twobody_singletarget_setup.pkl'
    # single_setup_file = os.path.join(fdir2, fname)
    
    # tudat_geo_setup_singletarget(setup_file, obs_time_file, single_setup_file)
    
    
    
    ###########################################################################
    # Run Filter
    ###########################################################################
    
    
    
    # run_singletarget_filter(setup_file)
    
    
    
    # Run Filter
    # run_multitarget_filter(setup_file, prev_results, results_file)
    
    # combine_results()
    
    
    
    # multitarget_analysis(results_file, setup_file)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    