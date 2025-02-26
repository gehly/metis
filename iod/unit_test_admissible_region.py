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


import admissible_region as ar


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



if __name__ == '__main__':
    
    plt.close('all')
    
    unit_test_optical_car_gmm_demars()
    
    unit_test_optical_car_gmm_gehly()
    