import numpy as np
from math import pi
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta

sys.path.append('../')

from estimation.estimation_functions import ls_batch
from dynamics.dynamics_functions import general_dynamics
from dynamics.dynamics_functions import ode_balldrop, ode_balldrop_stm
from dynamics.dynamics_functions import ode_twobody, ode_twobody_stm
from sensors.sensors import define_sensors
from sensors.visibility_functions import check_visibility
from sensors.measurements import compute_measurement
from utilities.astrodynamics import kep2cart
from utilities.constants import GME
from utilities.eop_functions import get_celestrak_eop_alldata, get_eop_data
from utilities.eop_functions import get_XYs2006_alldata



###############################################################################
# Constant Acceleration Test (Ball Dropping)
###############################################################################

def balldrop_setup():
    
    # Define state parameters
    acc = 9.81  #m/s^2
    state_params = {}
    state_params['acc'] = acc
    state_params['Q'] = np.diag([1e-12,1e-12])
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'ode'
    int_params['ode_integrator'] = 'dop853'
    int_params['intfcn'] = ode_balldrop
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_system'] = 'seconds'

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
    sig_y = 0.01
    sig_dy = 0.001
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
            tout, Xout = general_dynamics(X, tin, state_params, int_params)
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
        
    int_params['intfcn'] = ode_balldrop_stm
        
    filter_output = ls_batch(state_dict, meas_dict, meas_fcn, state_params, sensor_params, int_params)
    
    # Compute errors
    n = 2
    p = 2
    X_err = np.zeros((n, len(filter_output)))
    resids = np.zeros((p, len(filter_output)))
    sig_y = np.zeros(len(filter_output),)
    sig_dy = np.zeros(len(filter_output),)
    tk_list = list(filter_output.keys())
    for kk in range(len(filter_output)):
        tk = tk_list[kk]
        X = filter_output[tk]['X']
        P = filter_output[tk]['P']
        resids[:,kk] = filter_output[tk]['resids'].flatten()
        
        X_true = truth_dict[tk]
        X_err[:,kk] = (X - X_true).flatten()
        sig_y[kk] = np.sqrt(P[0,0])
        sig_dy[kk] = np.sqrt(P[1,1])
        
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tk_list, X_err[0,:], 'k.')
    plt.plot(tk_list, 3*sig_y, 'k--')
    plt.plot(tk_list, -3*sig_y, 'k--')
    plt.ylabel('Pos Err [m]')
    
    plt.subplot(2,1,2)
    plt.plot(tk_list, X_err[1,:], 'k.')
    plt.plot(tk_list, 3*sig_dy, 'k--')
    plt.plot(tk_list, -3*sig_dy, 'k--')
    plt.ylabel('Vel Err [m/s]')
    plt.xlabel('Time [sec]')
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tk_list, resids[0,:], 'k.')
    plt.ylabel('Y Resids [m]')
    
    plt.subplot(2,1,2)
    plt.plot(tk_list, resids[1,:], 'k.')
    plt.ylabel('dY Resids [m/s]')
    plt.xlabel('Time [sec]')
    
    plt.show()
    
        
        
        
    
    return



def H_balldrop(tk, Xref, state_params, sensor_kk):
    
    # Break out state
    y = float(Xref[0])
    dy = float(Xref[1])
    
    # Measurement information
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


def twobody_setup():
    
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
    int_params['intfcn'] = ode_twobody
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_system'] = 'datetime'

    # Time vector
    tvec = np.arange(0., 86400.*0.5 + 1., 10.)
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
    sensor_id_list = ['CMU Falcon', 'PSU Falcon']
    sensor_params = define_sensors(sensor_id_list)
    sensor_params[sensor_id_list[0]]['meas_types'] = ['ra', 'dec']
    sigma_dict = {}
    sigma_dict['ra'] = 5.*arcsec2rad   # rad
    sigma_dict['dec'] = 5.*arcsec2rad  # rad
    sensor_params[sensor_id_list[0]]['sigma_dict'] = sigma_dict
    print(sensor_params)

    # Generate truth and measurements
    truth_dict = {}
    meas_dict = {}
    meas_dict['tk_list'] = []
    meas_dict['Yk_list'] = []
    meas_dict['sensor_id_list'] = []
    X = X_true.copy()
    for kk in range(len(tk_list)):
        
        if kk > 0:
            tin = [tk_list[kk-1], tk_list[kk]]
            tout, Xout = general_dynamics(X, tin, state_params, int_params)
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
    
                
    plt.show()   
    
    print(meas_dict)
                
    
    
    return 
























if __name__ == '__main__':
    
    plt.close('all')
    
#    execute_balldrop_test()
    
    twobody_setup()
















