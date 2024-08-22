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

def spring_mass_setup():
    
    # Define state parameters
    state_params = {}
    state_params['m'] = 100.
    state_params['k'] = 10.
    state_params['c'] = 1.
    
    filter_params = {}
    filter_params['Q'] = np.diag([1e-8])
    filter_params['gap_seconds'] = 100.
    filter_params['alpha'] = 1e-4
    filter_params['pnorm'] = 1.2
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = dyn.ode_spring_mass_damper
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'seconds'

    # Time vector
    tk_list = np.arange(0.,100.1,1.)
    
    # Inital State
    X_true = np.array([[0.],[1.]])
    P = np.array([[100., 0.],[0., 2.]])
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
    sig_pos = 0.01
    sensor_params[1]['sigma_dict'] = {}
    sensor_params[1]['sigma_dict']['pos'] = sig_pos
    sensor_params[1]['meas_types'] = ['pos']
    meas_fcn = mfunc.H_linear1d_pos
    outlier_inds = []
    X = X_true.copy()
    
    for kk in range(len(tk_list)):
        
        if kk > 0:
            tin = [tk_list[kk-1], tk_list[kk]]
            tout, Xout = dyn.general_dynamics(X, tin, state_params, int_params)
            X = Xout[-1,:].reshape(2, 1)
        
        truth_dict[tk_list[kk]] = X
        
        if kk in outlier_inds:
            pos_noise = 100.*sig_pos*np.random.randn()
        else:
            pos_noise = sig_pos*np.random.randn()
        
        pos_meas = float(X[0,0]) + pos_noise
        meas_dict['tk_list'].append(tk_list[kk])
        meas_dict['Yk_list'].append(np.array([[pos_meas]]))
        meas_dict['sensor_id_list'].append(1)
        
    params_dict = {}
    params_dict['state_params'] = state_params
    params_dict['filter_params'] = filter_params
    params_dict['int_params'] = int_params
    params_dict['sensor_params'] = sensor_params
    
    x_plot = []
    dx_plot = []
    pos_plot = []
    resids = []
    for kk in range(len(tk_list)):
        tk = tk_list[kk]
        x_plot.append(truth_dict[tk][0,0])
        dx_plot.append(truth_dict[tk][1,0])
        pos_plot.append(meas_dict['Yk_list'][kk][0,0])
        resids.append(pos_plot[kk] - x_plot[kk])
    
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(tk_list, x_plot, 'k.')
    plt.title('Spring-Mass-Damper Setup')
    plt.ylabel('True Pos [m]')
    plt.subplot(4,1,2)
    plt.plot(tk_list, dx_plot, 'k.')
    plt.ylabel('True Vel [m/s]')
    plt.subplot(4,1,3)
    plt.plot(tk_list, pos_plot, 'k.')
    plt.ylabel('Meas Pos [m]')
    plt.subplot(4,1,4)
    plt.plot(tk_list, resids, 'k.')
    plt.ylim([-0.03, 0.03])
    plt.ylabel('Resids [m]')    
    plt.xlabel('Time [sec]')
    
    plt.show()
    
    setup_file = os.path.join('unit_test', 'ocbe_spring_mass_setup.pkl')
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [state_dict, meas_fcn, meas_dict, params_dict, truth_dict], pklFile, -1 )
    pklFile.close()
        

    return 


def execute_spring_mass_damper_test():
    
    setup_file = os.path.join('unit_test', 'ocbe_spring_mass_setup.pkl')
    
    pklFile = open(setup_file, 'rb' )
    data = pickle.load( pklFile )
    state_dict = data[0]
    meas_fcn = data[1]
    meas_dict = data[2]
    params_dict = data[3]
    truth_dict = data[4]
    pklFile.close()
        
    params_dict['int_params']['intfcn'] = dyn.ode_spring_mass_damper_stm
    
    
    # Batch Test
    batch_output, full_state_output = est.ls_batch(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    analysis.compute_linear1d_errors(batch_output, truth_dict)
    
    
    
    # EKF Test
    # params_dict['filter_params']['gap_seconds'] = 0.
    ekf_output, full_state_output = est.ls_ekf(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    analysis.compute_linear1d_errors(ekf_output, truth_dict)
        
    
    
    return



if __name__ == '__main__':
    
    
    plt.close('all')
    
    # spring_mass_setup()
    
    execute_spring_mass_damper_test()