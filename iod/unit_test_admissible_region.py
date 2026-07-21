###############################################################################
# This file contains code to test Admissible Region IOD functions.
#
# References:
#
#  [1] DeMars, K.J. and Jah, M.K., "Probabilistic Initial Orbit Determination
#      Using Gaussian Mixture Models," JGCD, 2013.
#
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import inspect
import sys

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)


import admissible_region as ar

import sensors.measurement_functions as mfunc
import sensors.sensors as sens
import utilities.astrodynamics as astro
import utilities.coordinate_systems as coord
import utilities.eop_functions as eop
import utilities.time_systems as timesys



arcsec2rad = np.pi/(3600.*180.)



def unit_test_optical_car_gmm_demars():
    '''
    Test case from DeMars and Jah Section II [1].
    
    '''
    
    # Physical constants
    GM = 398600.4415*1e9            # m^3/s^2
    Re = 6378.1370*1000.            # m
    wE = 7.2921158553e-5            # rad/s
       
    # Measurement vector
    ra = np.radians(10.)
    dec = np.radians(-2.)
    dra = np.radians(15.)/3600.
    ddec = np.radians(3.)/3600.
    
    Zk = np.reshape([ra, dec, dra, ddec], (4,1))
    
    # Ground station
    q_vect = Re*np.array([[np.cos(np.radians(30.))],
                          [                     0.],
                          [np.sin(np.radians(30.))]])
    
    w_vect = np.array([[0.], [0.], [wE]])
    dq_vect = np.cross(w_vect, q_vect, axis=0)
        
    # Vector of range values
    rho_vect = np.arange(0., 50000., 5.)*1000.      # m
    
    # CAR limits
    a_max = 50000.*1000.            # m
    a_min = 0.                      # m
    e_max = 0.4
    
    # Desired maximum standard deviation in range 
    # These are not from DeMars and produce CAR with many fewer components
    # than shown in the paper
    sigma_rho_desired = 5000.       # m
    sigma_drho_desired = 80.        # m/s
    
    # Measurement noise (DeMars Section V.B.)
    sigma_dict = {}
    sigma_dict['ra'] = 0.4*arcsec2rad
    sigma_dict['dec'] = 0.4*arcsec2rad
    sigma_dict['dra'] = 0.07*arcsec2rad
    sigma_dict['ddec'] = 0.07*arcsec2rad
    
    # Set up parameters for CAR GMM function
    params = {}
    params['GM'] = GM
    params['Re'] = Re
    params['a_max'] = a_max
    params['a_min'] = a_min
    params['e_max'] = e_max
    params['sigma_rho_desired'] = sigma_rho_desired
    params['sigma_drho_desired'] = sigma_drho_desired
    params['sigma_dict'] = sigma_dict
    
    # Compute CAR GMM in range, range-rate space
    GMM_rgrr = ar.optical_car_gmm(rho_vect, Zk, q_vect, dq_vect, params, plot_flag=True)
    
    # Convert GMM to Cartesian ECI
    GMM_eci = ar.car_gmm_to_eci(GMM_rgrr, Zk, q_vect, dq_vect, params)
    
    # Check weights
    print('sum weights', sum(GMM_eci[0]))
    
    
    return


def unit_test_optical_car_gmm_gehly():
    '''
    Test case from Gehly Ch 4.
    
    '''
    
    # Physical constants
    GM = 398600.4415*1e9            # m^3/s^2
    Re = 6378.1370*1000.            # m
    wE = 7.2921158553e-5            # rad/s
       
    # Measurement vector
    ra = np.radians(10.)
    dec = np.radians(-2.)
    dra = np.radians(15.)/3600.
    ddec = np.radians(3.)/3600.
    
    Zk = np.reshape([ra, dec, dra, ddec], (4,1))
    
    # Ground station
    q_vect = Re*np.array([[np.cos(np.radians(30.))],
                          [                     0.],
                          [np.sin(np.radians(30.))]])
    
    w_vect = np.array([[0.], [0.], [wE]])
    dq_vect = np.cross(w_vect, q_vect, axis=0)
        
    # Vector of range values
    rho_vect = np.arange(0., 50000., 5.)*1000.      # m
    
    # CAR limits
    a_max = 42565.*1000.            # m
    a_min = 41764.*1000.            # m
    e_max = 0.1
    
    # Desired maximum standard deviation in range 
    sigma_rho_desired = 500.       # m
    sigma_drho_desired = 20.        # m/s
    
    # Measurement noise (DeMars Section V.B.)
    sigma_dict = {}
    sigma_dict['ra'] = 0.4*arcsec2rad
    sigma_dict['dec'] = 0.4*arcsec2rad
    sigma_dict['dra'] = 0.07*arcsec2rad
    sigma_dict['ddec'] = 0.07*arcsec2rad
    
    # Set up parameters for CAR GMM function
    params = {}
    params['GM'] = GM
    params['Re'] = Re
    params['a_max'] = a_max
    params['a_min'] = a_min
    params['e_max'] = e_max
    params['sigma_rho_desired'] = sigma_rho_desired
    params['sigma_drho_desired'] = sigma_drho_desired
    params['sigma_dict'] = sigma_dict
    
    # Compute CAR GMM in range, range-rate space
    GMM_rgrr = ar.optical_car_gmm(rho_vect, Zk, q_vect, dq_vect, params, plot_flag=True)
    
    # Convert GMM to Cartesian ECI
    GMM_eci = ar.car_gmm_to_eci(GMM_rgrr, Zk, q_vect, dq_vect, params)
    
    # Check weights
    print('sum weights', sum(GMM_eci[0]))
    
    return


def unit_test_optical_car_gmm_leo():
    '''
    
    
    '''
    
    # Physical constants
    GM = 398600.4415*1e9            # m^3/s^2
    Re = 6378.1370*1000.            # m
    wE = 7.2921158553e-5            # rad/s
    
    
    # Generic LEO Orbit
    elem = [6878., 0.01, 75., 90., 10., 10.]
    Xo = np.reshape(astro.kep2cart(elem), (6,1))
    
    
    # Time vector
    UTC0 = datetime(2021, 6, 21, 0, 20, 0)
    tvec = np.arange(0, 15*60., 0.1)
    UTC_list = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    
    # Sensor data
    sensor_id_list = ['Leiden Optical']
    sensor_params = sens.define_sensors(sensor_id_list)
    
    for sensor_id in sensor_id_list:
        sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Compute measurements
    Yk_list = []
    sensor_id_time_list = []
    rho_list = []
    tk_list = []
    Xk_truth = []
    for UTC in UTC_list:
        
        dt_sec = (UTC - UTC0).total_seconds()
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        Xk = astro.element_conversion(Xo, 1, 1, dt=dt_sec)
        
        for sensor_id in sensor_id_list:
            
            sensor = sensor_params[sensor_id]
            all_meas = mfunc.compute_measurement(Xk, {}, sensor_params, sensor_id, UTC, EOP_data,
                                                 XYs_df, meas_types=['ra', 'dec', 'rg', 'az', 'el'])
            
            el = float(all_meas[4])
            if el > 0.:
                
#                print('')
#                print(UTC)
#                print(sensor_id)
#                print(all_meas)
            
                tk_list.append(UTC)
                Xk_truth.append(Xk)
                Yk = all_meas[0:2].reshape(2,1)
                Yk_list.append(Yk)
                sensor_id_time_list.append(sensor_id)
                rho_list.append(float(all_meas[2]))

                
    # Select first, middle, last entries
    mid = int(len(Yk_list)/2)
    tk_list = [tk_list[ii] for ii in [0, 1, 2]]
    Yk_list = [Yk_list[ii] for ii in [0, 1, 2]]
    sensor_id_time_list = [sensor_id_time_list[ii] for ii in [0, 1, 2]]
    rho_list = [rho_list[ii] for ii in [0, 1, 2]]
    Xo_truth = Xk_truth[0]
    
    print(tk_list)
    print(Yk_list)
    print(sensor_id_time_list)
    print(rho_list)
    
    
    
    mistake
    
    
       
    # # Measurement vector
    # ra = np.radians(10.)
    # dec = np.radians(-2.)
    # dra = np.radians(15.)/3600.
    # ddec = np.radians(3.)/3600.
    
    # Zk = np.reshape([ra, dec, dra, ddec], (4,1))
    
    # Ground station
    q_vect = Re*np.array([[np.cos(np.radians(30.))],
                          [                     0.],
                          [np.sin(np.radians(30.))]])
    
    w_vect = np.array([[0.], [0.], [wE]])
    dq_vect = np.cross(w_vect, q_vect, axis=0)
        
    # Vector of range values
    rho_vect = np.arange(0., 10000., 5.)*1000.      # m
    
    # CAR limits
    a_max = 7378.*1000.            # m
    a_min = 6378.*1000.            # m
    e_max = 0.1
    
    # Desired maximum standard deviation in range 
    sigma_rho_desired = 500.       # m
    sigma_drho_desired = 20.        # m/s
    
    # Measurement noise (DeMars Section V.B.)
    sigma_dict = {}
    sigma_dict['ra'] = 0.4*arcsec2rad
    sigma_dict['dec'] = 0.4*arcsec2rad
    sigma_dict['dra'] = 0.07*arcsec2rad
    sigma_dict['ddec'] = 0.07*arcsec2rad
    
    # Set up parameters for CAR GMM function
    params = {}
    params['GM'] = GM
    params['Re'] = Re
    params['a_max'] = a_max
    params['a_min'] = a_min
    params['e_max'] = e_max
    params['sigma_rho_desired'] = sigma_rho_desired
    params['sigma_drho_desired'] = sigma_drho_desired
    params['sigma_dict'] = sigma_dict
    
    # Compute CAR GMM in range, range-rate space
    GMM_rgrr = ar.optical_car_gmm(rho_vect, Zk, q_vect, dq_vect, params, plot_flag=True)
    
    # Convert GMM to Cartesian ECI
    GMM_eci = ar.car_gmm_to_eci(GMM_rgrr, Zk, q_vect, dq_vect, params)
    
    # Check weights
    print('sum weights', sum(GMM_eci[0]))
    
    return



if __name__ == '__main__':
    
    plt.close('all')
    
    # unit_test_optical_car_gmm_demars()
    
    # unit_test_optical_car_gmm_gehly()
    
    unit_test_optical_car_gmm_leo()
    