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









def unit_test_auction():
    '''
    Example assignment problem from [1] Blackman and Popoli 
    
    '''
    
    # C is cost matrix to minimize
    C = np.array([[10.,    5.,   8.,   9.],
                  [7.,   100.,  20., 100.],
                  [100.,  21., 100., 100.],
                  [100.,  15.,  17., 100.],
                  [100., 100.,  16.,  22.]])
    
    # A is score matrix to maximize
    A = 100.*np.ones((5,4)) - C
    
    # Compute assignment
    row_index, score, eps = auction(A)
    
    print(row_index, score, eps)
    
    truth = [7., 15., 16., 9.]
    test_sum = 0.
    for ii in range(4):
        print(C[row_index[ii],ii])
        test_sum += C[row_index[ii],ii] - truth[ii]
        
    if test_sum == 0.:
        print('pass')
    
    
    
    return


def unit_test_murty():
    '''
    Example assignment problem from [1] Blackman and Popoli
    '''
    
    # C is cost matrix to minimize
    C = np.array([[10.,    5.,   8.,   9.],
                  [7.,   100.,  20., 100.],
                  [100.,  21., 100., 100.],
                  [100.,  15.,  17., 100.],
                  [100., 100.,  16.,  22.]])
    
    # A is score matrix to maximize
    A = 100.*np.ones((5,4)) - C
    
    # Compute assignment
    kbest = 4
    final_list = murty(A, kbest)
    
    print(final_list)
    
    for row_index in final_list:
        for ii in range(4):
            print(C[row_index[ii], ii])
    
    
    
    return



def geo_2obj_twobody_setup():
    
    
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
    
    # Filter parameters
    filter_params = {}
    filter_params['Q'] = 1e-16 * np.diag([1, 1, 1])
    filter_params['gap_seconds'] = 900.
    filter_params['alpha'] = 1e-4
    filter_params['pnorm'] = 2.
    
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
    for hr in [9, 10, 11, 12, 13, 14, 15]:
        UTC = datetime(2021, 6, 21, hr, 0, 0)
        tvec = np.arange(0., 601., 60.)
        tk_list.extend([UTC + timedelta(seconds=ti) for ti in tvec])

    # Inital State
    elem1 = [42164.1, 0.001, 0.1, 90., 0., 0.]
    X1_true = np.reshape(astro.kep2cart(elem1), (6,1))
    P1 = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    pert_vect1 = np.multiply(np.sqrt(np.diag(P1)), np.random.randn(6))
    X1_init = X1_true + np.reshape(pert_vect1, (6, 1))
    
    elem2 = [42164.1, 0.001, 0.1, 90., 0., 1.]
    X2_true = np.reshape(astro.kep2cart(elem2), (6,1))
    P2 = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    pert_vect2 = np.multiply(np.sqrt(np.diag(P2)), np.random.randn(6))
    X2_init = X2_true + np.reshape(pert_vect2, (6, 1))
    
    
    state_dict = {}
    state_dict[tk_list[0]] = {}
    state_dict[tk_list[0]]['weights'] = [1., 1.]
    state_dict[tk_list[0]]['means'] = [X1_init, X2_init]
    state_dict[tk_list[0]]['covars'] = [P1, P2]
    
    
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
    meas_fcn = mfunc.unscented_radec
    meas_dict = {}
    X = np.concatentate((X1_true, X2_true), axis=0)
    for kk in range(len(tk_list)):
        
        if kk > 0:
            tin = [tk_list[kk-1], tk_list[kk]]
            tout, Xout = dyn.general_dynamics(X, tin, state_params, int_params)
            X = Xout[-1,:].reshape(6, 1)
        
        X1_t = X[0:6].reshape(6,1)
        X2_t = X[6:12].reshape(6,1)
        truth_dict[tk_list[kk]] = {}
        truth_dict[tk_list[kk]]['Xt_list'] = [X1_t, X2_t]
        
        # Check visibility conditions and compute measurements
        UTC = tk_list[kk]
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        Zk_list = []
        sensor_id_list = []
        # Loop over sensors and objects
        for sensor_id in sensor_id_list:  
        
            for Xj in truth_dict[tk_list[kk]]['Xt_list']:            
                          
                if visfunc.check_visibility(Xj, state_params, sensor_params,
                                            sensor_id, UTC, EOP_data, XYs_df):
                    
                    # Incorporate missed detection here
                    
                    # Compute measurements
                    zj = mfunc.compute_measurement(Xj, state_params, sensor_params,
                                                   sensor_id, UTC, EOP_data, XYs_df,
                                                   meas_types=sensor['meas_types'])
                    
                    zj[0] += np.random.randn()*sigma_dict['ra']
                    zj[1] += np.random.randn()*sigma_dict['dec']
                    
                    Zk_list.append(zj)
                    sensor_id_list.append(sensor_id)
            
            # Incorporate clutter measurements here

        # If measurements were collected, randomize order and store
        if len(Zk_list) > 0:
            
            inds = list(range(len(Zk_list)))
            random.shuffle(inds)
            
            meas_dict[UTC] = {}
            meas_dict['Zk_list'] = [Zk_list[ii] for ii in inds]
            meas_dict['sensor_id_list'] = [sensor_id_list[ii] for ii in inds]
                

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
                
    setup_file = os.path.join('unit_test', 'tudat_geo_perturbed_setup.pkl')
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [state_dict, meas_fcn, meas_dict, params_dict, truth_dict], pklFile, -1 )
    pklFile.close()
                
    
    
    return




if __name__ == '__main__':
    
    
    
    
    
    
    
    
    
    
    
    
    
#    unit_test_auction()
    
    
    # unit_test_murty()
    
    
    
    

#    # A is score matrix to maximize
#    A = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                  [2, 10, 3, 6, 2, 12, 6, 9, 6, 10],
#                  [3, 11, 1, 9, 4, 15, 5, 4, 9, 12],
#                  [4, 6, 5, 4, 0, 3, 4, 6, 10, 11],
#                  [5, 0, 6, 8, 1, 10, 3, 7, 8, 13],
#                  [6, 11, 0, 6, 5, 9, 2, 5, 3, 8],
#                  [7, 9, 2, 5, 6, 5, 1, 3, 6, 6],
#                  [8, 8, 6, 9, 4, 0, 8, 2, 1, 5],
#                  [10, 12, 11, 6, 5, 10, 9, 1, 6, 7],
#                  [9, 10, 4, 8, 0, 9, 1, 0, 5, 9]])
#
#    
#    # Compute assignment
#    row_index, score, eps = auction(A)
#    
#    print(row_index, score, eps)
#
#    for ii in range(10):
#        print(A[row_index[ii],ii])
#    
