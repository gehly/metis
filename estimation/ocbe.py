import numpy as np
import scipy
import math
import sys
import os
import inspect
import copy
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from dynamics import dynamics_functions as dyn
from sensors import measurement_functions as mfunc
from utilities import astrodynamics as astro
from utilities import coordinate_systems as coord
from utilities.constants import arcsec2rad



def bl_ocbe(state_dict, truth_dict, meas_dict, meas_fcn, params_dict):
    '''
    This function implements the Ballistic Linear Optimal Control Based 
    Estimator (BL-OCBE).

    Parameters
    ------
    state_dict : dictionary
        initial state and covariance for filter execution
    truth_dict : dictionary
        true state at all times
    meas_dict : dictionary
        measurement data over time for the filter and parameters (noise, etc)
    meas_fcn : function handle
        function for measurements
    params_dict: dictionary
        state, integrator, filter parameters    
    
    Returns
    ------
    filter_output : dictionary
        output state, covariance, and post-fit residuals at measurement times
    full_state_output : dictionary
        output state and covariance at all truth times
        
    '''
    
    # Break out params
    state_params = params_dict['state_params']
    int_params = params_dict['int_params']
    sensor_params = params_dict['sensor_params']
    filter_params = params_dict['filter_params']
    
    # State information
    state_tk = sorted(state_dict.keys())[-1]
    Xo_ref = state_dict[state_tk]['X']
    Po_bar = state_dict[state_tk]['P']
    Q = filter_params['Q']
    gap_seconds = filter_params['gap_seconds']
    time_format = int_params['time_format']
    
    # Setup
    n = len(Xo_ref)
    q = int(Q.shape[0])

    # Initialize output
    filter_output = {}

    # Measurement times
    tk_list = meas_dict['tk_list']
    Yk_list = meas_dict['Yk_list']
    sensor_id_list = meas_dict['sensor_id_list']
    
    # Number of epochs
    N = len(tk_list)

    # Initialize
    adjoint0 = np.zeros((n, 1))
    P = Po_bar
    Zref = np.concatenate((Xo_ref, adjoint0), axis=0)
    nz = len(Zref)
    xhat = np.zeros((nz, 1))
    phi = np.identity(nz)
    phi0_v = np.reshape(phi, (nz**2, 1))
    
    # Loop over times
    for kk in range(N):
    
        # Current and previous time
        if kk == 0:
            tk_prior = state_tk
        else:
            tk_prior = tk_list[kk-1]

        tk = tk_list[kk]
        
        if time_format == 'seconds':
            delta_t = tk - tk_prior
        elif time_format == 'JD':
            delta_t = (tk - tk_prior)*86400.
        elif time_format == 'datetime':
            delta_t = (tk - tk_prior).total_seconds()
            
            
        # Propagate to next time
        # Initial Conditions for Integration Routine
        Zref_prior = Zref
        xhat_prior = xhat
        P_prior = P
        int0 = np.concatenate((Zref_prior, phi0_v))
        
        # Integrate Zref and STM
        if tk_prior == tk:
            intout = int0.T
        else:
            int0 = int0.flatten()
            tin = [tk_prior, tk]            
            tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)

        # Extract values for later calculations
        xout = intout[-1,:]
        Zref = xout[0:nz].reshape(nz, 1)
        phi_v = xout[nz:].reshape(nz**2, 1)
        phi = np.reshape(phi_v, (nz, nz))
        
        phi_xx = phi[0:n,0:n]
        phi_xp = phi[0:n,n:nz]
        phi_px = phi[n:nz,0:n]
        phi_pp = phi[n:nz,n:nz]
        
        # Error check identities
        # Lubey Dissertation Eq 3.56
        if np.max(phi_px) > 1e-15:
            print('phi_px not zero', phi_px)
        if np.max(np.dot(phi_pp, phi_xx.T) - np.eye(n)) > 1e-15:
            print('phi_pp not inverse of phi_xx.T')
            print(phi_pp)
            print(phi_xx)
            
        prod = np.dot(phi_xp, phi_xx.T)
        if np.max(prod - prod.T) > 1e-15:
            print('phi_xp*phi_xx.T not symmetric')
            print(phi_xp)
            print(phi_xx)
            print(prod)
        
        
        # Time Update: a priori state and covar at tk
        # Lubey Dissertation Eq 3.63
        xbar = np.dot(phi_xx, xhat_prior)
        Pbar = np.dot(phi_xx, np.dot(P_prior, phi_xx.T)) - np.dot(phi_xp, phi_xx.T)
        
        
        # Measurement Update: posterior state and covar at tk            
        # Retrieve measurement data
        Yk = Yk_list[kk]
        sensor_id = sensor_id_list[kk]

        # Compute prefit residuals and  Kalman gain
        Hk_til, Gk, Rk = meas_fcn(tk, Zref, state_params, sensor_params, sensor_id)
        yk = Yk - Gk
        
    
    
    return