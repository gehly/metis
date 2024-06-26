import numpy as np
from math import pi, asin, atan2
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
import pickle
import os
import inspect

# Load tudatpy modules  
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)


from estimation import analysis_functions as analysis
from estimation import estimation_functions as est
from dynamics import dynamics_functions as dyn
from sensors import measurement_functions as mfunc
from sensors import sensors as sens
from sensors import visibility_functions as visfunc
from utilities import astrodynamics as astro
from utilities import coordinate_systems as coord
from utilities import eop_functions as eop
from utilities import tle_functions as tle
from utilities.constants import GME, J2E, wE, Re, arcsec2rad




###############################################################################
# Orbit Dynamics with Perturbations
###############################################################################


def geo_j2_setup():
    
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
    state_params['J2'] = J2E
    state_params['dtheta'] = wE  # rad/s
    state_params['R'] = Re  # km
    state_params['Cd'] = 2.2*0.
    state_params['A_m'] = 1e-8    # km^2/kg
    state_params['radius_m'] = 1.
    state_params['albedo'] = 0.1
    
    # Filter parameters
    filter_params = {}
    filter_params['Q'] = 1e-16 * np.diag([1, 1, 1])
    filter_params['gap_seconds'] = 900.
    filter_params['alpha'] = 1e-4
    filter_params['pnorm'] = 2.
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'ode'
    int_params['ode_integrator'] = 'dop853'
    int_params['intfcn'] = dyn.ode_twobody_j2_drag
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'

    # Time vector
    tk_list = []
    for hr in range(24):
        UTC = datetime(2021, 6, 21, hr, 0, 0)
        tvec = np.arange(0., 601., 60.)
        tk_list.extend([UTC + timedelta(seconds=ti) for ti in tvec])

    # Inital State
    elem = [42164.1, 0.001, 0., 90., 0., 0.]
    X_true = np.reshape(astro.kep2cart(elem), (6,1))
    P = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(6))
    X_init = X_true + np.reshape(pert_vect, (6, 1))
    
    state_dict = {}
    state_dict[tk_list[0]] = {}
    state_dict[tk_list[0]]['X'] = X_init
    state_dict[tk_list[0]]['P'] = P
    
    
    # Sensor and measurement parameters
    sensor_id_list = ['UNSW Falcon']
    sensor_params = sens.define_sensors(sensor_id_list)
    sensor_params['eop_alldata'] = eop_alldata
    sensor_params['XYs_df'] = XYs_df
    
    for sensor_id in sensor_id_list:
        sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
        sigma_dict = {}
        sigma_dict['ra'] = 5.*arcsec2rad   # rad
        sigma_dict['dec'] = 5.*arcsec2rad  # rad
        sensor_params[sensor_id]['sigma_dict'] = sigma_dict
    print(sensor_params)

    # Generate truth and measurements
    truth_dict = {}
    meas_fcn = mfunc.H_radec
    meas_dict = {}
    meas_dict['tk_list'] = []
    meas_dict['Yk_list'] = []
    meas_dict['sensor_id_list'] = []
    X = X_true.copy()
    for kk in range(len(tk_list)):
        
        if kk > 0:
            tin = [tk_list[kk-1], tk_list[kk]]
            tout, Xout = dyn.general_dynamics(X, tin, state_params, int_params)
            X = Xout[-1,:].reshape(6, 1)
        
        truth_dict[tk_list[kk]] = X
        
        # Check visibility conditions and compute measurements
        UTC = tk_list[kk]
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        
        for sensor_id in sensor_id_list:
            sensor = sensor_params[sensor_id]
            if visfunc.check_visibility(X, state_params, sensor_params,
                                        sensor_id, UTC, EOP_data, XYs_df):
                
                # Compute measurements
                Yk = mfunc.compute_measurement(X, state_params, sensor_params,
                                               sensor_id, UTC, EOP_data, XYs_df,
                                               meas_types=sensor['meas_types'])
                
                Yk[0] += np.random.randn()*sigma_dict['ra']
                Yk[1] += np.random.randn()*sigma_dict['dec']
                
                meas_dict['tk_list'].append(UTC)
                meas_dict['Yk_list'].append(Yk)
                meas_dict['sensor_id_list'].append(sensor_id)
                

    # Plot data
    tplot = [(tk - tk_list[0]).total_seconds()/3600. for tk in tk_list]
    xplot = []
    yplot = []
    zplot = []
    for tk in tk_list:
        X = truth_dict[tk]
        xplot.append(X[0])
        yplot.append(X[1])
        zplot.append(X[2])
        
    meas_tk = meas_dict['tk_list']
    meas_tplot = [(tk - tk_list[0]).total_seconds()/3600. for tk in meas_tk]
    meas_sensor_id = meas_dict['sensor_id_list']
    meas_sensor_index = [sensor_id_list.index(sensor_id) for sensor_id in meas_sensor_id]
    
        
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tplot, xplot, 'k.')
    plt.ylabel('X [km]')
    plt.subplot(3,1,2)
    plt.plot(tplot, yplot, 'k.')
    plt.ylabel('Y [km]')
    plt.subplot(3,1,3)
    plt.plot(tplot, zplot, 'k.')
    plt.ylabel('Z [km]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.plot(meas_tplot, meas_sensor_index, 'k.')
    plt.xlabel('Time [hours]')
    plt.xlim([0, 25])
    plt.yticks([0], ['UNSW Falcon'])
    plt.ylabel('Sensor ID')
    
                
    plt.show()   
    
    print(meas_dict)
    
    params_dict = {}
    params_dict['state_params'] = state_params
    params_dict['filter_params'] = filter_params
    params_dict['int_params'] = int_params
    params_dict['sensor_params'] = sensor_params
                
    setup_file = os.path.join('advanced_test', 'geo_j2_setup.pkl')
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [state_dict,  meas_fcn, meas_dict, params_dict, truth_dict], pklFile, -1 )
    pklFile.close()
                
    
    
    
    return


def sso_j2_drag_setup():
    
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
    state_params['J2'] = J2E
    state_params['dtheta'] = wE  # rad/s
    state_params['R'] = Re  # km
    state_params['Cd'] = 2.2
    state_params['A_m'] = 1e-8    # km^2/kg
    
    state_params['radius_m'] = 1.
    state_params['albedo'] = 0.1
    state_params['laser_lim'] = 1e6
    
    # Filter parameters
    filter_params = {}
    filter_params['Q'] = 1e-16 * np.diag([1, 1, 1])
    filter_params['gap_seconds'] = 900.
    filter_params['alpha'] = 1e-4
    filter_params['pnorm'] = 2.
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'ode'
    int_params['ode_integrator'] = 'dop853'
    int_params['intfcn'] = dyn.ode_twobody_j2_drag
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'

    # Time vector
    tvec = np.arange(0., 18341., 20.)
#    UTC0 = datetime(1999, 10, 3, 23, 11, 9, 181400)
    UTC0 = datetime(1999, 10, 4, 1, 45, 0)
#    UTC0 = datetime(2000, 1, 1, 12, 0, 0)
    tk_list = [UTC0 + timedelta(seconds=ti) for ti in tvec]

    # Inital State
#    X_true = np.reshape([757.700301, 5222.606566, 4851.49977,
#                         2.213250611, 4.678372741, -5.371314404], (6,1))
#    P = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
#    pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(6))
#    X_init = X_true + np.reshape(pert_vect, (6, 1))
    
    beta = state_params['Cd']*state_params['A_m']
    X_true = np.reshape([757.700301, 5222.606566, 4851.49977,
                         2.213250611, 4.678372741, -5.371314404, beta], (7,1))
    P = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6, 1e-18])
    pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(7))
    X_init = X_true + np.reshape(pert_vect, (7, 1))
    
    print(X_init)
    
    
    # Check initial satellite location
    r_GCRF = X_true[0:3].reshape(3,1)
    v_GCRF = X_true[3:6].reshape(3,1)
    EOP_data = eop.get_eop_data(eop_alldata, UTC0)
    r_ecef, dum = coord.gcrf2itrf(r_GCRF, v_GCRF, UTC0, EOP_data, XYs_df)
    
    lat, lon, ht = coord.ecef2latlonht(r_ecef)
    print(lat, lon, ht)
    
#    mistake
    
    state_dict = {}
    state_dict[tk_list[0]] = {}
    state_dict[tk_list[0]]['X'] = X_init
    state_dict[tk_list[0]]['P'] = P
    
    
    # Sensor and measurement parameters
    sensor_id_list = ['Born s101', 'Born s337', 'Born s394']
    sensor_params = sens.define_sensors(sensor_id_list)
    sensor_params['eop_alldata'] = eop_alldata
    sensor_params['XYs_df'] = XYs_df
    
    for sensor_id in sensor_id_list:
        sensor_params[sensor_id]['meas_types'] = ['rg', 'ra', 'dec']
        sigma_dict = {}
        sigma_dict['rg'] = 0.001  # km
        sigma_dict['ra'] = 5.*arcsec2rad   # rad
        sigma_dict['dec'] = 5.*arcsec2rad  # rad
        sensor_params[sensor_id]['sigma_dict'] = sigma_dict

    # Generate truth and measurements
    truth_dict = {}
    meas_fcn = mfunc.H_rgradec
    meas_dict = {}
    meas_dict['tk_list'] = []
    meas_dict['Yk_list'] = []
    meas_dict['sensor_id_list'] = []
    X = X_true.copy()
    n = len(X)
    for kk in range(len(tk_list)):
        
        if kk > 0:
            tin = [tk_list[kk-1], tk_list[kk]]
            tout, Xout = dyn.general_dynamics(X, tin, state_params, int_params)
            X = Xout[-1,:].reshape(n, 1)
        
        truth_dict[tk_list[kk]] = X
        
        # Check visibility conditions and compute measurements
        UTC = tk_list[kk]
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        
        for sensor_id in sensor_id_list:
            sensor = sensor_params[sensor_id]
            sigma_dict = sensor['sigma_dict']
            meas_types = sensor['meas_types']
            if visfunc.check_visibility(X, state_params, sensor_params,
                                        sensor_id, UTC, EOP_data, XYs_df):
                
                # Compute measurements
                Yk = mfunc.compute_measurement(X, state_params, sensor_params,
                                               sensor_id, UTC, EOP_data, XYs_df,
                                               meas_types=sensor['meas_types'])
                
                for mtype in meas_types:
                    mind = meas_types.index(mtype)
                    Yk[mind] += np.random.randn()*sigma_dict[mtype]

                meas_dict['tk_list'].append(UTC)
                meas_dict['Yk_list'].append(Yk)
                meas_dict['sensor_id_list'].append(sensor_id)
                

    # Plot data
    tplot = [(tk - tk_list[0]).total_seconds()/3600. for tk in tk_list]
    xplot = []
    yplot = []
    zplot = []
    for tk in tk_list:
        X = truth_dict[tk]
        xplot.append(X[0])
        yplot.append(X[1])
        zplot.append(X[2])
        
    meas_tk = meas_dict['tk_list']
    meas_tplot = [(tk - tk_list[0]).total_seconds()/3600. for tk in meas_tk]
    meas_sensor_id = meas_dict['sensor_id_list']
    meas_sensor_index = [sensor_id_list.index(sensor_id) for sensor_id in meas_sensor_id]
    
        
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tplot, xplot, 'k.')
    plt.ylabel('X [km]')
    plt.subplot(3,1,2)
    plt.plot(tplot, yplot, 'k.')
    plt.ylabel('Y [km]')
    plt.subplot(3,1,3)
    plt.plot(tplot, zplot, 'k.')
    plt.ylabel('Z [km]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.plot(meas_tplot, meas_sensor_index, 'k.')
    plt.xlabel('Time [hours]')
    plt.xlim([0, 5.5])
    plt.yticks([0, 1, 2], ['s101', 's337', 's394'])
    plt.ylabel('Sensor ID')
    
                
    plt.show()   
    
    print(meas_dict)

    params_dict = {}
    params_dict['state_params'] = state_params
    params_dict['filter_params'] = filter_params
    params_dict['int_params'] = int_params
    params_dict['sensor_params'] = sensor_params
                
    setup_file = os.path.join('advanced_test', 'sso_j2_drag_beta_setup.pkl')
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [state_dict,  meas_fcn, meas_dict, params_dict, truth_dict], pklFile, -1 )
    pklFile.close()
    
    
    
    return


def tudat_geo_setup():
        
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
    state_params['mass'] = 400.
    state_params['Cd'] = 2.2
    state_params['Cr'] = 1.2
    state_params['drag_area_m2'] = 4.
    state_params['srp_area_m2'] = 4.
    
    # Filter parameters
    filter_params = {}
    filter_params['Q'] = 1e-16 * np.diag([1, 1, 1])
    filter_params['gap_seconds'] = 900.
    filter_params['alpha'] = 1e-4
    filter_params['pnorm'] = 2.
    
    # Integration function and additional settings    
    int_params = {}
    int_params['integrator'] = 'tudat'
    int_params['tudat_integrator'] = 'rk4'
    int_params['step'] = 10.
    int_params['max_step'] = 100.
    int_params['min_step'] = 1.
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'

    # Time vector
    tk_list = []
    for hr in range(24):
        UTC = datetime(2021, 6, 21, hr, 0, 0)
        tvec = np.arange(0., 601., 60.)
        tk_list.extend([UTC + timedelta(seconds=ti) for ti in tvec])

    # Inital State
    elem = [42164.1, 0.001, 0.1, 90., 0., 0.]
    X_true = np.reshape(astro.kep2cart(elem), (6,1))
    P = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(6))
    X_init = X_true + np.reshape(pert_vect, (6, 1))
    
    state_dict = {}
    state_dict[tk_list[0]] = {}
    state_dict[tk_list[0]]['X'] = X_init
    state_dict[tk_list[0]]['P'] = P
    
    
    # Sensor and measurement parameters
    sensor_id_list = ['UNSW Falcon']
    sensor_params = sens.define_sensors(sensor_id_list)
    sensor_params['eop_alldata'] = eop_alldata
    sensor_params['XYs_df'] = XYs_df
    
    for sensor_id in sensor_id_list:
        sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
        sigma_dict = {}
        sigma_dict['ra'] = 5.*arcsec2rad   # rad
        sigma_dict['dec'] = 5.*arcsec2rad  # rad
        sensor_params[sensor_id]['sigma_dict'] = sigma_dict
#    print(sensor_params)
    
#    for sensor_id in sensor_id_list:
#        sensor_params[sensor_id]['meas_types'] = ['rg', 'ra', 'dec']
#        sigma_dict = {}
#        sigma_dict['rg'] = 0.001  # km
#        sigma_dict['ra'] = 5.*arcsec2rad   # rad
#        sigma_dict['dec'] = 5.*arcsec2rad  # rad
#        sensor_params[sensor_id]['sigma_dict'] = sigma_dict
#    print(sensor_params)

    # Generate truth and measurements
    truth_dict = {}
    meas_fcn = mfunc.H_radec
    meas_dict = {}
    meas_dict['tk_list'] = []
    meas_dict['Yk_list'] = []
    meas_dict['sensor_id_list'] = []
    X = X_true.copy()
    for kk in range(len(tk_list)):
        
        if kk > 0:
            tin = [tk_list[kk-1], tk_list[kk]]
            tout, Xout = dyn.general_dynamics(X, tin, state_params, int_params)
            X = Xout[-1,:].reshape(6, 1)
        
        truth_dict[tk_list[kk]] = X
        
        # Check visibility conditions and compute measurements
        UTC = tk_list[kk]
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        
        for sensor_id in sensor_id_list:
            sensor = sensor_params[sensor_id]
            if visfunc.check_visibility(X, state_params, sensor_params,
                                        sensor_id, UTC, EOP_data, XYs_df):
                
                # Compute measurements
                Yk = mfunc.compute_measurement(X, state_params, sensor_params,
                                               sensor_id, UTC, EOP_data, XYs_df,
                                               meas_types=sensor['meas_types'])
                
                Yk[0] += np.random.randn()*sigma_dict['ra']
                Yk[1] += np.random.randn()*sigma_dict['dec']
                
                meas_dict['tk_list'].append(UTC)
                meas_dict['Yk_list'].append(Yk)
                meas_dict['sensor_id_list'].append(sensor_id)
                

    # Plot data
    tplot = [(tk - tk_list[0]).total_seconds()/3600. for tk in tk_list]
    xplot = []
    yplot = []
    zplot = []
    for tk in tk_list:
        X = truth_dict[tk]
        xplot.append(X[0])
        yplot.append(X[1])
        zplot.append(X[2])
        
    meas_tk = meas_dict['tk_list']
    meas_tplot = [(tk - tk_list[0]).total_seconds()/3600. for tk in meas_tk]
    meas_sensor_id = meas_dict['sensor_id_list']
    meas_sensor_index = [sensor_id_list.index(sensor_id) for sensor_id in meas_sensor_id]
    
        
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tplot, xplot, 'k.')
    plt.ylabel('X [km]')
    plt.subplot(3,1,2)
    plt.plot(tplot, yplot, 'k.')
    plt.ylabel('Y [km]')
    plt.subplot(3,1,3)
    plt.plot(tplot, zplot, 'k.')
    plt.ylabel('Z [km]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.plot(meas_tplot, meas_sensor_index, 'k.')
    plt.xlabel('Time [hours]')
    plt.xlim([0, 25])
    plt.yticks([0], ['UNSW Falcon'])
    plt.ylabel('Sensor ID')
    
                
    plt.show()   
    
    print(meas_dict)
    
    params_dict = {}
    params_dict['state_params'] = state_params
    params_dict['filter_params'] = filter_params
    params_dict['int_params'] = int_params
    params_dict['sensor_params'] = sensor_params
                
    setup_file = os.path.join('advanced_test', 'tudat_geo_perturbed_setup.pkl')
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [state_dict, meas_fcn, meas_dict, params_dict, truth_dict], pklFile, -1 )
    pklFile.close()
                
    
    
    return 


def tudat_geo_7day_setup():
    
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
    state_params['mass'] = 400.
    state_params['Cd'] = 2.2
    state_params['Cr'] = 1.2
    state_params['drag_area_m2'] = 4.
    state_params['srp_area_m2'] = 4.
    
    # Filter parameters
    filter_params = {}
    filter_params['Q'] = 1e-16 * np.diag([1, 1, 1])
    filter_params['gap_seconds'] = 900.
    filter_params['alpha'] = 1e-4
    filter_params['pnorm'] = 2.
    
    # Integration function and additional settings    
    int_params = {}
    int_params['integrator'] = 'tudat'
    int_params['tudat_integrator'] = 'rk4'
    int_params['step'] = 10.
    int_params['max_step'] = 100.
    int_params['min_step'] = 1.
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'


    # Initial state
    qzs1r_norad = 49336
    qzs2_norad = 42738
    qzs3_norad = 42917
    qzs4_norad = 42965
    amos5_norad = 37950
    coms1_norad = 36744
    
    obj_id = qzs3_norad
    UTC0 = datetime(2022, 11, 7, 11, 0, 0)
    state_dict = tle.propagate_TLE([obj_id], [UTC0])
    r0 = state_dict[obj_id]['r_GCRF'][0]
    v0 = state_dict[obj_id]['v_GCRF'][0]
    Xo_true = np.concatenate((r0, v0), axis=0)
    
    P = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(6))
    X_init = Xo_true + np.reshape(pert_vect, (6, 1))
    
    

    # Time vector
    tk_list = []
    for ii in range(61):
        tk_list.append(UTC0 + timedelta(seconds=ii*10.))
    
    UTC = datetime(2022, 11, 8, 14, 0, 0)
    for ii in range(61):
        tk_list.append(UTC + timedelta(seconds=ii*10.))

        
    UTC = datetime(2022, 11, 9, 16, 0, 0)
    for ii in range(61):
        tk_list.append(UTC + timedelta(seconds=ii*10.))

        
    UTC = datetime(2022, 11, 10, 10, 0, 0)
    for ii in range(61):
        tk_list.append(UTC + timedelta(seconds=ii*10.))

        
    UTC = datetime(2022, 11, 11, 12, 0, 0)
    for ii in range(61):
        tk_list.append(UTC + timedelta(seconds=ii*10.))

        
    UTC = datetime(2022, 11, 12, 13, 0, 0)
    for ii in range(61):
        tk_list.append(UTC + timedelta(seconds=ii*10.))

        
    UTC = datetime(2022, 11, 13, 11, 0, 0)
    for ii in range(61):
        tk_list.append(UTC + timedelta(seconds=ii*10.))


    # Inital State for filter
    state_dict = {}
    state_dict[tk_list[0]] = {}
    state_dict[tk_list[0]]['X'] = X_init
    state_dict[tk_list[0]]['P'] = P
    
    
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
        sensor_params[sensor_id]['moon_angle_lim'] = 0.
#    print(sensor_params)
    
#    for sensor_id in sensor_id_list:
#        sensor_params[sensor_id]['meas_types'] = ['rg', 'ra', 'dec']
#        sigma_dict = {}
#        sigma_dict['rg'] = 0.001  # km
#        sigma_dict['ra'] = 5.*arcsec2rad   # rad
#        sigma_dict['dec'] = 5.*arcsec2rad  # rad
#        sensor_params[sensor_id]['sigma_dict'] = sigma_dict
#    print(sensor_params)

    # Generate truth and measurements
    truth_dict = {}
    meas_fcn = mfunc.H_radec
    meas_dict = {}
    meas_dict['tk_list'] = []
    meas_dict['Yk_list'] = []
    meas_dict['sensor_id_list'] = []
    X = Xo_true.copy()
    for kk in range(len(tk_list)):
        
        if kk > 0:
            tin = [tk_list[kk-1], tk_list[kk]]
            tout, Xout = dyn.general_dynamics(X, tin, state_params, int_params)
            X = Xout[-1,:].reshape(6, 1)
        
        truth_dict[tk_list[kk]] = X
        
        # Check visibility conditions and compute measurements
        UTC = tk_list[kk]
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        
        for sensor_id in sensor_id_list:
            sensor = sensor_params[sensor_id]
            if visfunc.check_visibility(X, state_params, sensor_params,
                                        sensor_id, UTC, EOP_data, XYs_df):
                
                # Compute measurements
                Yk = mfunc.compute_measurement(X, state_params, sensor_params,
                                               sensor_id, UTC, EOP_data, XYs_df,
                                               meas_types=sensor['meas_types'])
                
                Yk[0] += np.random.randn()*sigma_dict['ra']
                Yk[1] += np.random.randn()*sigma_dict['dec']
                
                meas_dict['tk_list'].append(UTC)
                meas_dict['Yk_list'].append(Yk)
                meas_dict['sensor_id_list'].append(sensor_id)
                

    # Plot data
    tplot = [(tk - tk_list[0]).total_seconds()/3600. for tk in tk_list]
    xplot = []
    yplot = []
    zplot = []
    for tk in tk_list:
        X = truth_dict[tk]
        xplot.append(X[0])
        yplot.append(X[1])
        zplot.append(X[2])
        
    meas_tk = meas_dict['tk_list']
    meas_tplot = [(tk - tk_list[0]).total_seconds()/3600. for tk in meas_tk]
    meas_sensor_id = meas_dict['sensor_id_list']
    meas_sensor_index = [sensor_id_list.index(sensor_id) for sensor_id in meas_sensor_id]
    
        
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tplot, xplot, 'k.')
    plt.ylabel('X [km]')
    plt.subplot(3,1,2)
    plt.plot(tplot, yplot, 'k.')
    plt.ylabel('Y [km]')
    plt.subplot(3,1,3)
    plt.plot(tplot, zplot, 'k.')
    plt.ylabel('Z [km]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.plot(meas_tplot, meas_sensor_index, 'k.')
    plt.xlabel('Time [hours]')
    # plt.xlim([0, 25])
    plt.yticks([0], ['RMIT ROO'])
    plt.ylabel('Sensor ID')
    
                
    plt.show()   
    
    print(meas_dict)
    
    params_dict = {}
    params_dict['state_params'] = state_params
    params_dict['filter_params'] = filter_params
    params_dict['int_params'] = int_params
    params_dict['sensor_params'] = sensor_params
                
    setup_file = os.path.join('advanced_test', 'tudat_geo_perturbed_7day_setup.pkl')
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [state_dict, meas_fcn, meas_dict, params_dict, truth_dict], pklFile, -1 )
    pklFile.close()
    
    return


def execute_tudat_ukf():
    
    # setup_file = os.path.join('advanced_test', 'tudat_geo_perturbed_setup.pkl')    
    setup_file = os.path.join('advanced_test', 'tudat_geo_perturbed_7day_setup.pkl')
    
    pklFile = open(setup_file, 'rb' )
    data = pickle.load( pklFile )
    state_dict = data[0]
    meas_fcn = data[1]
    meas_dict = data[2]
    params_dict = data[3]
    truth_dict = data[4]
    pklFile.close()
    
    
    meas_fcn = mfunc.unscented_radec
    params_dict['filter_params']['alpha'] = 1e-4
    params_dict['filter_params']['Q'] = 1e-15 * np.diag([1, 1, 1])
    params_dict['int_params']['tudat_integrator'] = 'rkf78'
    
    
    # Reduced dynamics model
    state_params = params_dict['state_params']
    state_params['bodies_to_create'] = ['Earth', 'Sun', 'Moon']
    state_params['sph_deg'] = 2
    state_params['sph_ord'] = 0
    state_params['mass'] = 400.
    state_params['Cd'] = 2.2
    state_params['Cr'] = 1.5
    state_params['drag_area_m2'] = 4.
    state_params['srp_area_m2'] = 4.
    
    params_dict['state_params'] = state_params
    
    # print(params_dict['state_params'])
    
    # mistake
    
    # UKF Test
    filter_output, full_state_output = est.ls_ukf(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    analysis.compute_orbit_errors(filter_output, filter_output, truth_dict)
    
    
    return


def execute_test():
    
        
    # setup_file = os.path.join('advanced_test', 'geo_j2_setup.pkl')
    setup_file = os.path.join('advanced_test', 'sso_j2_drag_beta_setup.pkl')
    
    pklFile = open(setup_file, 'rb' )
    data = pickle.load( pklFile )
    state_dict = data[0]
    meas_fcn = data[1]
    meas_dict = data[2]
    params_dict = data[3]
    truth_dict = data[4]
    pklFile.close()
    
    params_dict['int_params']['intfcn'] = dyn.ode_twobody_j2_drag_stm
    # params_dict['filter_params']['Q'] = 1e-16 * np.diag([1, 1, 1])
    
    
    
    filter_output, full_state_output = est.ls_batch(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)    
    analysis.compute_orbit_errors(filter_output, full_state_output, truth_dict)
    
    
    filter_output, full_state_output = est.ls_ekf(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    analysis.compute_orbit_errors(filter_output, filter_output, truth_dict)




if __name__ == '__main__':
    
    plt.close('all')
    

    # geo_j2_setup()
    
    # sso_j2_drag_setup()
    
    # tudat_geo_setup()
    
    # tudat_geo_7day_setup()
    
    execute_tudat_ukf()
    
    # execute_test()