import numpy as np
from math import pi, asin, atan2
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
import pickle
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
from sensors.sensors import define_sensors
from sensors.visibility_functions import check_visibility
from sensors.measurements import compute_measurement
from utilities.astrodynamics import kep2cart
from utilities.constants import GME
from utilities.coordinate_systems import itrf2gcrf, gcrf2itrf, ecef2enu
from utilities.eop_functions import get_celestrak_eop_alldata, get_eop_data
from utilities.eop_functions import get_XYs2006_alldata



###############################################################################
# Linear Motion Test
###############################################################################

def linear_motion_setup():
    
    # Define state parameters
    state_params = {}
    state_params['Q'] = np.diag([1e-8])
    state_params['gap_seconds'] = 100.
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'ode'
    int_params['ode_integrator'] = 'dop853'
    int_params['intfcn'] = dyn.ode_linear1d
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'seconds'

    # Time vector
    tk_list = np.arange(0.,900.1,10.)
    
    # Inital State
    X_true = np.array([[0.],[5.]])
    P = np.array([[10000., 0.],[0., 25.]])
#    pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(2))
#    X_init = X_true + np.reshape(pert_vect, (2, 1))
    
    X_init = np.array([[100.], [0.]])
    
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
    sig_rg = 0.5
    sensor_params[1]['sigma_dict'] = {}
    sensor_params[1]['sigma_dict']['rg'] = sig_rg
    sensor_params[1]['meas_types'] = ['rg']
    meas_fcn = H_linear1d_rg
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
        

    return state_dict, state_params, int_params, meas_fcn, meas_dict, sensor_params, truth_dict


def H_linear1d_rg(tk, Xref, state_params, sensor_params, sensor_id):
    
    # Break out state
    x = float(Xref[0])
    
    # Measurement information
    sensor_kk = sensor_params[sensor_id]
    meas_types = sensor_kk['meas_types']
    sigma_dict = sensor_kk['sigma_dict']
    p = len(meas_types)
    Rk = np.zeros((p, p))
    for ii in range(p):
        mtype = meas_types[ii]
        sig = sigma_dict[mtype]
        Rk[ii,ii] = sig**2.   
    
    # Hk_til and Gi
    Hk_til = np.array([[1., 0.]])
    Gk = np.array([[x]])
    
    return Hk_til, Gk, Rk



def execute_linear1d_test():
    
    state_dict, state_params, int_params, meas_fcn, meas_dict, sensor_params, truth_dict =\
        linear_motion_setup()
        
    int_params['intfcn'] = dyn.ode_linear1d_stm
    
    
    # Batch Test
    # filter_output, full_state_output = est.ls_batch(state_dict, truth_dict, meas_dict, meas_fcn, state_params, sensor_params, int_params)
    # analysis.compute_linear1d_errors(filter_output, truth_dict)
    
    # EKF Test
    filter_output, full_state_output = est.ls_ekf(state_dict, truth_dict, meas_dict, meas_fcn, state_params, sensor_params, int_params)
    analysis.compute_linear1d_errors(filter_output, truth_dict)
        
        
        
    
    return

###############################################################################
# Constant Acceleration Test (Ball Dropping)
###############################################################################

def balldrop_setup():
    
    # Define state parameters
    acc = 9.81  #m/s^2
    state_params = {}
    state_params['acc'] = acc
    state_params['Q'] = np.diag([1e-2])
    state_params['gap_seconds'] = 10.
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'ode'
    int_params['ode_integrator'] = 'dop853'
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
    meas_fcn = H_balldrop
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

    return state_dict, state_params, int_params, meas_fcn, meas_dict, sensor_params, truth_dict


def execute_balldrop_test():
    
    state_dict, state_params, int_params, meas_fcn, meas_dict, sensor_params, truth_dict =\
        balldrop_setup()
        
    int_params['intfcn'] = dyn.ode_balldrop_stm
    
    
    # Batch Test
    filter_output, full_state_output = est.ls_batch(state_dict, truth_dict, meas_dict, meas_fcn, state_params, sensor_params, int_params)
    analysis.compute_balldrop_errors(filter_output, truth_dict)
    
    # EKF Test
    filter_output, full_state_output = est.ls_ekf(state_dict, truth_dict, meas_dict, meas_fcn, state_params, sensor_params, int_params)
    analysis.compute_balldrop_errors(filter_output, truth_dict)
        
        
        
    
    return



def H_balldrop(tk, Xref, state_params, sensor_params, sensor_id):
    
    # Break out state
    y = float(Xref[0])
    dy = float(Xref[1])
    
    # Measurement information
    sensor_kk = sensor_params[sensor_id]
    meas_types = sensor_kk['meas_types']
    sigma_dict = sensor_kk['sigma_dict']
    p = len(meas_types)
    Rk = np.zeros((p, p))
    for ii in range(p):
        mtype = meas_types[ii]
        sig = sigma_dict[mtype]
        Rk[ii,ii] = sig**2.   
    
    # Hk_til and Gi
    Hk_til = np.diag([1.,1.])
    Gk = np.array([[y],[dy]])
    
    return Hk_til, Gk, Rk



###############################################################################
# Orbit Dynamics Test (Two-Body Orbit)
###############################################################################


def twobody_leo_setup():
    
    arcsec2rad = pi/(3600.*180.)
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = get_XYs2006_alldata()
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
    state_params['radius_m'] = 1.
    state_params['albedo'] = 0.1
    state_params['laser_lim'] = 1e6
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'ode'
    int_params['ode_integrator'] = 'dop853'
    int_params['intfcn'] = dyn.ode_twobody
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'

    # Time vector
    tvec = np.arange(0., 3600.*12. + 1., 10.)
    UTC0 = datetime(2021, 6, 1, 0, 0, 0)
    tk_list = [UTC0 + timedelta(seconds=ti) for ti in tvec]

    # Inital State
    elem = [7000., 0.01, 98., 0., 0., 0.]
    X_true = np.reshape(kep2cart(elem), (6,1))
    P = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(6))
    X_init = X_true + np.reshape(pert_vect, (6, 1))
    
    state_dict = {}
    state_dict[tk_list[0]] = {}
    state_dict[tk_list[0]]['X'] = X_init
    state_dict[tk_list[0]]['P'] = P
    
    
    # Sensor and measurement parameters
    sensor_id_list = ['Zimmerwald Laser', 'Stromlo Laser', 'Arequipa Laser', 'Haleakala Laser', 'Yarragadee Laser']
    sensor_params = define_sensors(sensor_id_list)
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
    meas_fcn = est.H_rgradec
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
        EOP_data = get_eop_data(eop_alldata, UTC)
        
        for sensor_id in sensor_id_list:
            sensor = sensor_params[sensor_id]
            sigma_dict = sensor['sigma_dict']
            meas_types = sensor['meas_types']
            if check_visibility(X, state_params, sensor, UTC, EOP_data, XYs_df):
                
                # Compute measurements
                Yk = compute_measurement(X, state_params, sensor, UTC,
                                         EOP_data, XYs_df, meas_types)
                
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
    plt.ylabel('Sensor ID')
    
                
    plt.show()   
    
    print(meas_dict)
                
    pklFile = open( 'twobody_leo_setup.pkl', 'wb' )
    pickle.dump( [state_dict, state_params, int_params, meas_fcn, meas_dict, sensor_params, truth_dict], pklFile, -1 )
    pklFile.close()
    
    
    return



def twobody_geo_setup():
    
    arcsec2rad = pi/(3600.*180.)
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = get_XYs2006_alldata()
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
    state_params['radius_m'] = 1.
    state_params['albedo'] = 0.1
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'ode'
    int_params['ode_integrator'] = 'dop853'
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

    # Inital State
    elem = [42164.1, 0.001, 0., 90., 0., 0.]
    X_true = np.reshape(kep2cart(elem), (6,1))
    P = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(6))
    X_init = X_true + np.reshape(pert_vect, (6, 1))
    
    state_dict = {}
    state_dict[tk_list[0]] = {}
    state_dict[tk_list[0]]['X'] = X_init
    state_dict[tk_list[0]]['P'] = P
    
    
    # Sensor and measurement parameters
    sensor_id_list = ['UNSW Falcon']
    sensor_params = define_sensors(sensor_id_list)
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
    meas_fcn = est.H_radec
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
        EOP_data = get_eop_data(eop_alldata, UTC)
        
        for sensor_id in sensor_id_list:
            sensor = sensor_params[sensor_id]
            if check_visibility(X, state_params, sensor, UTC, EOP_data, XYs_df):
                
                # Compute measurements
                Yk = compute_measurement(X, state_params, sensor, UTC,
                                         EOP_data, XYs_df,
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
    plt.ylabel('Sensor ID')
    
                
    plt.show()   
    
    print(meas_dict)
                
    pklFile = open( 'twobody_geo_setup.pkl', 'wb' )
    pickle.dump( [state_dict, state_params, int_params, meas_fcn, meas_dict, sensor_params, truth_dict], pklFile, -1 )
    pklFile.close()
                
    
    
    return 





def execute_twobody_test():
    
    arcsec2rad = pi/(3600.*180.)
    
    pklFile = open('twobody_leo_setup.pkl', 'rb' )
    data = pickle.load( pklFile )
    state_dict = data[0]
    state_params = data[1]
    int_params = data[2]
    meas_fcn = data[3]
    meas_dict = data[4]
    sensor_params = data[5]
    truth_dict = data[6]
    pklFile.close()
        
    int_params['intfcn'] = dyn.ode_twobody_stm
        
    filter_output, full_state_output = est.ls_batch(state_dict, truth_dict, meas_dict, meas_fcn, state_params, sensor_params, int_params)
    
    # Compute errors
    n = 6
    p = len(meas_dict['Yk_list'][0])
    X_err = np.zeros((n, len(filter_output)))
    resids = np.zeros((p, len(filter_output)))
    sig_x = np.zeros(len(filter_output),)
    sig_y = np.zeros(len(filter_output),)
    sig_z = np.zeros(len(filter_output),)
    sig_dx = np.zeros(len(filter_output),)
    sig_dy = np.zeros(len(filter_output),)
    sig_dz = np.zeros(len(filter_output),)
    
    tk_list = list(filter_output.keys())
    t0 = sorted(truth_dict.keys())[0]
    thrs = [(tk - t0).total_seconds()/3600. for tk in tk_list]
    for kk in range(len(filter_output)):
        tk = tk_list[kk]
        X = filter_output[tk]['X']
        P = filter_output[tk]['P']
        resids[:,kk] = filter_output[tk]['resids'].flatten()
        
        X_true = truth_dict[tk]
        X_err[:,kk] = (X - X_true).flatten()
        sig_x[kk] = np.sqrt(P[0,0])
        sig_y[kk] = np.sqrt(P[1,1])
        sig_z[kk] = np.sqrt(P[2,2])
        sig_dx[kk] = np.sqrt(P[3,3])
        sig_dy[kk] = np.sqrt(P[4,4])
        sig_dz[kk] = np.sqrt(P[5,5])
        
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, X_err[0,:], 'k.')
    plt.plot(thrs, 3*sig_x, 'k--')
    plt.plot(thrs, -3*sig_x, 'k--')
    plt.ylabel('X Err [km]')
    
    plt.subplot(3,1,2)
    plt.plot(thrs, X_err[1,:], 'k.')
    plt.plot(thrs, 3*sig_y, 'k--')
    plt.plot(thrs, -3*sig_y, 'k--')
    plt.ylabel('Y Err [km]')
    
    plt.subplot(3,1,3)
    plt.plot(thrs, X_err[2,:], 'k.')
    plt.plot(thrs, 3*sig_z, 'k--')
    plt.plot(thrs, -3*sig_z, 'k--')
    plt.ylabel('Z Err [km]')

    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, X_err[3,:], 'k.')
    plt.plot(thrs, 3*sig_dx, 'k--')
    plt.plot(thrs, -3*sig_dx, 'k--')
    plt.ylabel('dX Err [km/s]')
    
    plt.subplot(3,1,2)
    plt.plot(thrs, X_err[4,:], 'k.')
    plt.plot(thrs, 3*sig_dy, 'k--')
    plt.plot(thrs, -3*sig_dy, 'k--')
    plt.ylabel('dY Err [km/s]')
    
    plt.subplot(3,1,3)
    plt.plot(thrs, X_err[5,:], 'k.')
    plt.plot(thrs, 3*sig_dz, 'k--')
    plt.plot(thrs, -3*sig_dz, 'k--')
    plt.ylabel('dZ Err [km/s]')

    plt.xlabel('Time [hours]')
    
    plt.figure()
    
    if p == 3:
        plt.subplot(3,1,1)
        plt.plot(thrs, resids[0,:]*1000., 'k.')
        plt.ylabel('Range [m]')
        
        plt.subplot(3,1,2)
        plt.plot(thrs, resids[1,:]/arcsec2rad, 'k.')
        plt.ylabel('RA [arcsec]')
        
        plt.subplot(3,1,3)
        plt.plot(thrs, resids[2,:]/arcsec2rad, 'k.')
        plt.ylabel('DEC [arcsec]')
        
        plt.xlabel('Time [hours]')
        
    elif p == 2:
        
        plt.subplot(1,1,1)
        plt.plot(thrs, resids[0,:]/arcsec2rad, 'k.')
        plt.ylabel('RA [arcsec]')
        
        plt.subplot(2,1,1)
        plt.plot(thrs, resids[1,:]/arcsec2rad, 'k.')
        plt.ylabel('DEC [arcsec]')
        
        plt.xlabel('Time [hours]')
        
    
    plt.show()
    
        
        
        
    
    return















if __name__ == '__main__':
    
    plt.close('all')
    
    execute_linear1d_test()
    
    # execute_balldrop_test()
    
#    twobody_geo_setup()
    
#    twobody_leo_setup()
    
    # execute_twobody_test()
















