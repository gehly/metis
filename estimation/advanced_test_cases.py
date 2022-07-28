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

from utilities.constants import GME, arcsec2rad
import utilities.eop_functions as eop




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
    state_params['radius_m'] = 1.
    state_params['albedo'] = 0.1
    state_params['Q'] = 1e-12 * np.diag([1, 1, 1])
    state_params['gap_seconds'] = 900.
    
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
    plt.xlim([0, 25])
    plt.yticks([0], ['UNSW Falcon'])
    plt.ylabel('Sensor ID')
    
                
    plt.show()   
    
    print(meas_dict)
                
    setup_file = os.path.join('advanced_test', 'geo_j2_setup.pkl')
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [state_dict, state_params, int_params, meas_fcn, meas_dict, sensor_params, truth_dict], pklFile, -1 )
    pklFile.close()
                
    
    
    
    return