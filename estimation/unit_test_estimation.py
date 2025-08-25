import numpy as np
from math import pi, asin, atan2
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
import pickle
import os
import inspect

# # Load tudatpy modules  
# from tudatpy.kernel.interface import spice
# from tudatpy.kernel import numerical_simulation
# from tudatpy.kernel.numerical_simulation import environment_setup
# from tudatpy.kernel.numerical_simulation import propagation_setup
# from tudatpy.kernel.astro import element_conversion
# from tudatpy.kernel import constants
# from tudatpy.util import result2array

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
from utilities import tle_functions as tle
from utilities.constants import GME, arcsec2rad



###############################################################################
# Linear Motion Test
###############################################################################

def linear_motion_setup():
    
    # Define state parameters
    state_params = {}
    
    filter_params = {}
    filter_params['Q'] = np.diag([1e-8])
    filter_params['gap_seconds'] = 1e-12
    filter_params['alpha'] = 1e-4
    filter_params['pnorm'] = 1.2
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = dyn.ode_linear1d
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'seconds'

    # Time vector
    tk_list = np.arange(0.,900.1,10.)
    
    # Inital State
    X_true = np.array([[0.],[5.]])
    P = np.array([[10000., 0.],[0., 25.]])
    pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(2))
    X_init = X_true + np.reshape(pert_vect, (2, 1))
    
    
    state_dict = {}
    state_dict[tk_list[0]] = {}
    state_dict[tk_list[0]]['X'] = X_init
    state_dict[tk_list[0]]['P'] = P
    
    # Generate Truth and Measurements
    truth_dict = {}
    meas_dict = {}
    meas_dict['tk_list'] = []
    meas_dict['Yk_list'] = []
    meas_dict['sensor_id_list'] = []
    sensor_params = {}
    sensor_params[1] = {}
    sig_rg = 1.
    sensor_params[1]['sigma_dict'] = {}
    sensor_params[1]['sigma_dict']['rg'] = sig_rg
    sensor_params[1]['meas_types'] = ['rg']
    meas_fcn = mfunc.H_linear1d_rg
    outlier_inds = []
    X = X_true.copy()
    
    for kk in range(len(tk_list)):
        
        if kk > 0:
            tin = [tk_list[kk-1], tk_list[kk]]
            tout, Xout = dyn.general_dynamics(X, tin, state_params, int_params)
            X = Xout[-1,:].reshape(2, 1)
        
        truth_dict[tk_list[kk]] = X
        
        if kk in outlier_inds:
            rg_noise = 100.*sig_rg*np.random.randn()
        else:
            rg_noise = sig_rg*np.random.randn()
        
        rg_meas = float(X[0]) + rg_noise
        meas_dict['tk_list'].append(tk_list[kk])
        meas_dict['Yk_list'].append(np.array([[rg_meas]]))
        meas_dict['sensor_id_list'].append(1)
        
    params_dict = {}
    params_dict['state_params'] = state_params
    params_dict['filter_params'] = filter_params
    params_dict['int_params'] = int_params
    params_dict['sensor_params'] = sensor_params
        

    return state_dict, meas_fcn, meas_dict, params_dict, truth_dict






def execute_linear1d_test():
    
    state_dict, meas_fcn, meas_dict, params_dict, truth_dict =\
        linear_motion_setup()
        
    params_dict['int_params']['intfcn'] = dyn.ode_linear1d_stm
    
    
    # Batch Test
    batch_output, full_state_output = est.ls_batch(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    analysis.compute_linear1d_errors(batch_output, truth_dict)
    
    # # Lp-norm Batch Test
    # filter_output, full_state_output = est.lp_batch(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    # analysis.compute_linear1d_errors(filter_output, truth_dict)
    
    
    print('Batch output')
    tf = sorted(batch_output.keys())[-1]
    Xf_batch = batch_output[tf]['X']
    Pf_batch = batch_output[tf]['P']
    
    print('Xf_batch', Xf_batch)
    print('Pf_batch', Pf_batch)
    
    
    # EKF Test
    # params_dict['filter_params']['gap_seconds'] = 0.
    ekf_output, full_state_output = est.ls_ekf(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    analysis.compute_linear1d_errors(ekf_output, truth_dict)
    
    print('KF output')
    Xf_kf = ekf_output[tf]['X']
    Pf_kf = ekf_output[tf]['P']
    
    print('Xf_ekf', Xf_kf)
    print('Pf_ekf', Pf_kf)
    
    
    # EKF Smoothing
    # params_dict['filter_params']['gap_seconds'] = 0.
    ekf_output, full_state_output = est.ls_ekf(state_dict, truth_dict, meas_dict, meas_fcn, params_dict, smoothing=True)
    analysis.compute_linear1d_errors(ekf_output, truth_dict, smoothing=True)
    
    print('KF smoothing output')
    Xf_kf = ekf_output[tf]['X']
    Pf_kf = ekf_output[tf]['P']
    
    print('Xf_ekf', Xf_kf)
    print('Pf_ekf', Pf_kf)
        
    
    # # Unscented Batch Test
    # params_dict['int_params']['intfcn'] = dyn.ode_linear1d_ukf
    # meas_fcn = mfunc.unscented_linear1d_rg
    # ubatch_output, full_state_output = est.unscented_batch(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    # analysis.compute_linear1d_errors(ubatch_output, truth_dict)    
    
    # # UKF Test
    # ukf_output, full_state_output = est.ls_ukf(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    # analysis.compute_linear1d_errors(ukf_output, truth_dict)
    
    
    
    
    return

###############################################################################
# Constant Acceleration Test (Ball Dropping)
###############################################################################

def balldrop_setup():
    
    # Define state parameters
    acc = 9.81  #m/s^2
    state_params = {}
    state_params['acc'] = acc
    
    # Filter params
    filter_params = {}
    filter_params['Q'] = np.diag([1e-8])
    filter_params['gap_seconds'] = 10.
    filter_params['alpha'] = 1e-4
    filter_params['pnorm'] = 1.2
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = dyn.ode_balldrop
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'seconds'

    # Time vector
    tk_list = np.arange(0.,100.1,1.)
    
    # Inital State
    X_true = np.array([[0.],[0.]])
    P = np.array([[4., 0.],[0., 1.]])
    pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(2))
    X_init = X_true + np.reshape(pert_vect, (2, 1))
    
    state_dict = {}
    state_dict[tk_list[0]] = {}
    state_dict[tk_list[0]]['X'] = X_init
    state_dict[tk_list[0]]['P'] = P
    
    # Generate Truth and Measurements
    truth_dict = {}
    meas_dict = {}
    meas_dict['tk_list'] = []
    meas_dict['Yk_list'] = []
    meas_dict['sensor_id_list'] = []
    sensor_params = {}
    sensor_params[1] = {}
    sig_y = 0.1
    sig_dy = 0.01
    sensor_params[1]['sigma_dict'] = {}
    sensor_params[1]['sigma_dict']['y'] = sig_y
    sensor_params[1]['sigma_dict']['dy'] = sig_dy
    sensor_params[1]['meas_types'] = ['y', 'dy']
    meas_fcn = mfunc.H_balldrop
    outlier_inds = []
    X = X_true.copy()
    
    for kk in range(len(tk_list)):
        
        if kk > 0:
            tin = [tk_list[kk-1], tk_list[kk]]
            tout, Xout = dyn.general_dynamics(X, tin, state_params, int_params)
            X = Xout[-1,:].reshape(2, 1)
        
        truth_dict[tk_list[kk]] = X
        
        if kk in outlier_inds:
            y_noise = 100.*sig_y*np.random.randn()
        else:
            y_noise = sig_y*np.random.randn()
            
        dy_noise = sig_dy*np.random.randn()
        
        y_meas = float(X[0]) + y_noise
        dy_meas = float(X[1]) + dy_noise
        meas_dict['tk_list'].append(tk_list[kk])
        meas_dict['Yk_list'].append(np.array([[y_meas], [dy_meas]]))
        meas_dict['sensor_id_list'].append(1)
        
    params_dict = {}
    params_dict['state_params'] = state_params
    params_dict['filter_params'] = filter_params
    params_dict['int_params'] = int_params
    params_dict['sensor_params'] = sensor_params
        

    return state_dict, meas_fcn, meas_dict, params_dict, truth_dict


def execute_balldrop_test():
    
    state_dict, meas_fcn, meas_dict, params_dict, truth_dict =\
        balldrop_setup()
        
    params_dict['int_params']['intfcn'] = dyn.ode_balldrop_stm
    
    
    # Batch Test
    filter_output, full_state_output = est.ls_batch(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    analysis.compute_balldrop_errors(filter_output, truth_dict)
    
    # Lp-norm Batch Test
    filter_output, full_state_output = est.lp_batch(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    analysis.compute_balldrop_errors(filter_output, truth_dict)
    
    # EKF Test
    filter_output, full_state_output = est.ls_ekf(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    analysis.compute_balldrop_errors(filter_output, truth_dict)
        
    # Unscented Batch Test
    params_dict['int_params']['intfcn'] = dyn.ode_balldrop_ukf
    meas_fcn = mfunc.unscented_balldrop
    
    ubatch_output, full_state_output = est.unscented_batch(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    analysis.compute_balldrop_errors(ubatch_output, truth_dict) 
    
    # UKF Test
    filter_output, full_state_output = est.ls_ukf(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    analysis.compute_balldrop_errors(filter_output, truth_dict)
        
    
    return







###############################################################################
# Orbit Dynamics Test (Two-Body Orbit)
###############################################################################


def twobody_geo_setup():
        
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
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
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = dyn.ode_twobody
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'

    # Time vector
    tk_list = []
    for hr in range(24):
        UTC = datetime(2021, 6, 21, hr, 0, 0)
        tvec = np.arange(0., 601., 60.)
        tk_list.extend([UTC + timedelta(seconds=ti) for ti in tvec])
        
    # tvec = np.arange(0., 86401., 1800.)
    # UTC = datetime(2021, 6, 21, 0, 0, 0)
    # tk_list.extend([UTC + timedelta(seconds=ti) for ti in tvec])

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
                
    setup_file = os.path.join('unit_test', 'twobody_geo_setup.pkl')
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [state_dict, meas_fcn, meas_dict, params_dict, truth_dict], pklFile, -1 )
    pklFile.close()
                
    
    
    return 





#def twobody_leo_setup():
#    
#    
#    # Retrieve latest EOP data from celestrak.com
#    eop_alldata = eop.get_celestrak_eop_alldata()
#        
#    # Retrieve polar motion data from file
#    XYs_df = eop.get_XYs2006_alldata()
#    
#    # Define state parameters
#    state_params = {}
#    state_params['GM'] = GME
#    state_params['radius_m'] = 1.
#    state_params['albedo'] = 0.1
#    state_params['laser_lim'] = 1e6
#    state_params['Q'] = 1e-16 * np.diag([1, 1, 1])
#    state_params['gap_seconds'] = 900.
#    state_params['alpha'] = 1e-4
#    
#    # Integration function and additional settings
#    int_params = {}
#    int_params['integrator'] = 'ode'
#    int_params['ode_integrator'] = 'dop853'
#    int_params['intfcn'] = dyn.ode_twobody
#    int_params['rtol'] = 1e-12
#    int_params['atol'] = 1e-12
#    int_params['time_format'] = 'datetime'
#
#    # Time vector
#    tvec = np.arange(0., 3600.*12. + 1., 10.)
#    UTC0 = datetime(2021, 6, 1, 0, 0, 0)
#    tk_list = [UTC0 + timedelta(seconds=ti) for ti in tvec]
#
#    # Inital State
#    elem = [7000., 0.01, 98., 0., 0., 0.]
#    X_true = np.reshape(astro.kep2cart(elem), (6,1))
#    P = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
#    pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(6))
#    X_init = X_true + np.reshape(pert_vect, (6, 1))
#    
#    state_dict = {}
#    state_dict[tk_list[0]] = {}
#    state_dict[tk_list[0]]['X'] = X_init
#    state_dict[tk_list[0]]['P'] = P
#    
#    
#    # Sensor and measurement parameters
#    sensor_id_list = ['Zimmerwald Laser', 'Stromlo Laser', 'Arequipa Laser', 'Haleakala Laser', 'Yarragadee Laser']
#    sensor_params = sens.define_sensors(sensor_id_list)
#    sensor_params['eop_alldata'] = eop_alldata
#    sensor_params['XYs_df'] = XYs_df
#    
#    for sensor_id in sensor_id_list:
#        sensor_params[sensor_id]['meas_types'] = ['rg', 'ra', 'dec']
#        sigma_dict = {}
#        sigma_dict['rg'] = 0.001  # km
#        sigma_dict['ra'] = 5.*arcsec2rad   # rad
#        sigma_dict['dec'] = 5.*arcsec2rad  # rad
#        sensor_params[sensor_id]['sigma_dict'] = sigma_dict
#
#    # Generate truth and measurements
#    truth_dict = {}
#    meas_fcn = mfunc.H_rgradec
#    meas_dict = {}
#    meas_dict['tk_list'] = []
#    meas_dict['Yk_list'] = []
#    meas_dict['sensor_id_list'] = []
#    X = X_true.copy()
#    for kk in range(len(tk_list)):
#        
#        if kk > 0:
#            tin = [tk_list[kk-1], tk_list[kk]]
#            tout, Xout = dyn.general_dynamics(X, tin, state_params, int_params)
#            X = Xout[-1,:].reshape(6, 1)
#        
#        truth_dict[tk_list[kk]] = X
#        
#        # Check visibility conditions and compute measurements
#        UTC = tk_list[kk]
#        EOP_data = eop.get_eop_data(eop_alldata, UTC)
#        
#        for sensor_id in sensor_id_list:
#            sensor = sensor_params[sensor_id]
#            sigma_dict = sensor['sigma_dict']
#            meas_types = sensor['meas_types']
#            if visfunc.check_visibility(X, state_params, sensor, UTC, EOP_data, XYs_df):
#                
#                # Compute measurements
#                Yk = mfunc.compute_measurement(X, state_params, sensor, UTC,
#                                         EOP_data, XYs_df, meas_types)
#                
#                for mtype in meas_types:
#                    mind = meas_types.index(mtype)
#                    Yk[mind] += np.random.randn()*sigma_dict[mtype]
#
#                meas_dict['tk_list'].append(UTC)
#                meas_dict['Yk_list'].append(Yk)
#                meas_dict['sensor_id_list'].append(sensor_id)
#                
#
#    # Plot data
#    tplot = [(tk - tk_list[0]).total_seconds()/3600. for tk in tk_list]
#    xplot = []
#    yplot = []
#    zplot = []
#    for tk in tk_list:
#        X = truth_dict[tk]
#        xplot.append(X[0])
#        yplot.append(X[1])
#        zplot.append(X[2])
#        
#    meas_tk = meas_dict['tk_list']
#    meas_tplot = [(tk - tk_list[0]).total_seconds()/3600. for tk in meas_tk]
#    meas_sensor_id = meas_dict['sensor_id_list']
#    meas_sensor_index = [sensor_id_list.index(sensor_id) for sensor_id in meas_sensor_id]
#    
#        
#    
#    plt.figure()
#    plt.subplot(3,1,1)
#    plt.plot(tplot, xplot, 'k.')
#    plt.ylabel('X [km]')
#    plt.subplot(3,1,2)
#    plt.plot(tplot, yplot, 'k.')
#    plt.ylabel('Y [km]')
#    plt.subplot(3,1,3)
#    plt.plot(tplot, zplot, 'k.')
#    plt.ylabel('Z [km]')
#    plt.xlabel('Time [hours]')
#    
#    plt.figure()
#    plt.plot(meas_tplot, meas_sensor_index, 'k.')
#    plt.xlabel('Time [hours]')
#    plt.xlim([0, 12])
#    plt.yticks([0, 1, 2, 3, 4], ['Zimmerwald', 'Stromlo', 'Arequipa', 'Haleakala', 'Yarragadee'])
#    plt.ylabel('Sensor ID')
#    
#                
#    plt.show()   
#    
#    print(meas_dict)
#                
#    setup_file = os.path.join('unit_test', 'twobody_leo_setup.pkl')
#    pklFile = open( setup_file, 'wb' )
#    pickle.dump( [state_dict, state_params, int_params, meas_fcn, meas_dict, sensor_params, truth_dict], pklFile, -1 )
#    pklFile.close()
#    
#    
#    return


def twobody_born_setup():
    
    # Use this for LEO test case (better measurement visibilty)
    
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
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
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = dyn.ode_twobody
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
    X_true = np.reshape([757.700301, 5222.606566, 4851.49977,
                         2.213250611, 4.678372741, -5.371314404], (6,1))
    P = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(6))
    X_init = X_true + np.reshape(pert_vect, (6, 1))
    
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
                
    setup_file = os.path.join('unit_test', 'twobody_born_setup.pkl')
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [state_dict, meas_fcn, meas_dict, params_dict, truth_dict], pklFile, -1 )
    pklFile.close()
    
    
    return


def twobody_starlink_setup():
    
    # Use this for LEO test case (better measurement visibilty)
    
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
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
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = dyn.ode_twobody
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'

    # Time vector
    tvec = np.arange(0., 8.*3600.+1., 10.)
#    UTC0 = datetime(1999, 10, 3, 23, 11, 9, 181400)
    UTC0 = datetime(2025, 7, 29, 12, 0, 0)
#    UTC0 = datetime(2000, 1, 1, 12, 0, 0)
    tk_list = [UTC0 + timedelta(seconds=ti) for ti in tvec]

    # Inital State
    X_true = np.reshape([ 4.48960010e+03, -9.70051996e+02, -5.18042169e+03,
                         -9.74421049e-1,   7.19247706,     -2.19294121], (6,1))
    P = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(6))
    X_init = X_true + np.reshape(pert_vect, (6, 1))
    
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
    sensor_id_list = ['TIRA', 'ALTAIR']
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
    plt.xlim([0, 8])
    plt.yticks([0, 1], ['TIRA', 'ALTAIR'])
    plt.ylabel('Sensor ID')
    
                
    plt.show()   
    
    print(meas_dict)
    
    params_dict = {}
    params_dict['state_params'] = state_params
    params_dict['filter_params'] = filter_params
    params_dict['int_params'] = int_params
    params_dict['sensor_params'] = sensor_params
                
    setup_file = os.path.join('unit_test', 'twobody_starlink_setup.pkl')
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [state_dict, meas_fcn, meas_dict, params_dict, truth_dict], pklFile, -1 )
    pklFile.close()
    
    
    return






def execute_twobody_test():
    
        
    setup_file = os.path.join('unit_test', 'twobody_starlink_setup.pkl')
    
    pklFile = open(setup_file, 'rb' )
    data = pickle.load( pklFile )
    state_dict = data[0]
    meas_fcn = data[1]
    meas_dict = data[2]
    params_dict = data[3]
    truth_dict = data[4]
    pklFile.close()
        
    params_dict['int_params']['intfcn'] = dyn.ode_twobody_stm
#    params_dict['filter_params']['Q'] = 1e-16 * np.diag([1, 1, 1])
        
    # Batch Test
    filter_output, full_state_output = est.ls_batch(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)    
    analysis.compute_orbit_errors(filter_output, full_state_output, truth_dict)
    
    # # Lp-Norm Batch Test
    # params_dict['filter_params']['pnorm'] = 1.2
    # filter_output, full_state_output = est.lp_batch(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)    
    # analysis.compute_orbit_errors(filter_output, full_state_output, truth_dict)
    
    
    # EKF Test
    filter_output, full_state_output = est.ls_ekf(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    analysis.compute_orbit_errors(filter_output, filter_output, truth_dict)
    
    
    # Unscented Batch Test
    params_dict['int_params']['intfcn'] = dyn.ode_twobody_ukf
    # meas_fcn = mfunc.unscented_radec
    meas_fcn = mfunc.unscented_rgradec
    params_dict['filter_params']['alpha'] = 1e-2
    
    
#    meas_fcn = mfunc.unscented_radec
#    sensor_id_list = ['Born s101', 'Born s337', 'Born s394']
#    for sensor_id in sensor_id_list:
#        sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
#        sigma_dict = {}
##        sigma_dict['rg'] = 0.001  # km
#        sigma_dict['ra'] = 5.*arcsec2rad   # rad
#        sigma_dict['dec'] = 5.*arcsec2rad  # rad
#        sensor_params[sensor_id]['sigma_dict'] = sigma_dict
#        
#    Yk_list = meas_dict['Yk_list']
#    Yk_list2 = [Yk[1:3] for Yk in Yk_list]
#    meas_dict['Yk_list'] = Yk_list2
#        
#    meas_fcn = mfunc.unscented_rg
#    sensor_id_list = ['Born s101', 'Born s337', 'Born s394']
#    for sensor_id in sensor_id_list:
#        sensor_params[sensor_id]['meas_types'] = ['rg']
#        sigma_dict = {}
#        sigma_dict['rg'] = 0.001  # km
##        sigma_dict['ra'] = 5.*arcsec2rad   # rad
##        sigma_dict['dec'] = 5.*arcsec2rad  # rad
#        sensor_params[sensor_id]['sigma_dict'] = sigma_dict
#        
#    Yk_list = meas_dict['Yk_list']
#    Yk_list2 = [Yk[0] for Yk in Yk_list]
#    meas_dict['Yk_list'] = Yk_list2
#    
    
    
    filter_output, full_state_output = est.unscented_batch(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    analysis.compute_orbit_errors(filter_output, full_state_output, truth_dict)
    
    # UKF Test
    filter_output, full_state_output = est.ls_ukf(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    analysis.compute_orbit_errors(filter_output, filter_output, truth_dict)
    

        
    
    return







if __name__ == '__main__':
    
    plt.close('all')
    
    # execute_linear1d_test()
    
    # execute_balldrop_test()
    
    # twobody_geo_setup()
    
    # twobody_born_setup()
    
    # twobody_starlink_setup()

    
    execute_twobody_test()
    
















