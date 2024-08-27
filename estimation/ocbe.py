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
from estimation import estimation_functions as est
from sensors import measurement_functions as mfunc
from utilities import astrodynamics as astro
from utilities import coordinate_systems as coord
from utilities.constants import arcsec2rad



def bl_ocbe(state_dict, truth_dict, meas_dict, meas_fcn, params_dict,
            smoothing=False):
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
        
        
    References
    ------
    [1] Lubey, "Maneuver Detection and Reconstruction in Data Sparse Systems 
        with an Optimal Control Based Estimator," PhD Dissertation, 2015.
    
    [2] Lubey and Scheeres, "Automated State and Dynamics Estimation in
        Dynamically Mismodeled Systems with Information from Optimal Control 
        Policies," FUSION 2015.
        
    '''
    
    # Note from Lubey dissertation P39
    # xbar_km1_km1 = xhat_km1_km1
    # Pbar_km1_km1 = Phat_km1_km1
    
    # Break out params
    state_params = params_dict['state_params']
    int_params = params_dict['int_params']
    sensor_params = params_dict['sensor_params']
    filter_params = params_dict['filter_params']
    state_params['Q'] = filter_params['Q']
    
    # State information
    state_tk = sorted(state_dict.keys())[-1]
    Xo_ref = state_dict[state_tk]['X']
    Po_bar = state_dict[state_tk]['P']
    Q = filter_params['Q']
    B = state_params['B']
    gap_seconds = filter_params['gap_seconds']
    time_format = int_params['time_format']
    
    # Setup
    n = len(Xo_ref)
    q = int(Q.shape[0])

    # Initialize output
    filter_output = {}
    smoother_data = {}

    # Measurement times
    tk_list = meas_dict['tk_list']
    Yk_list = meas_dict['Yk_list']
    sensor_id_list = meas_dict['sensor_id_list']
    
    # Number of epochs
    N = len(tk_list)

    # Initialize
    adjoint0 = np.zeros((n, 1))
    Phat_k = Po_bar
    Zref_k = np.concatenate((Xo_ref, adjoint0), axis=0)
    nz = len(Zref_k)
    xhat_k = np.zeros((n, 1))
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
        Zref_km1 = Zref_k
        xhat_km1 = xhat_k
        Phat_km1 = Phat_k
        int0 = np.concatenate((Zref_km1, phi0_v))
        
        # Integrate Zref and STM
        if tk_prior == tk:
            intout = int0.T
        else:
            int0 = int0.flatten()
            tin = [tk_prior, tk]            
            tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)

        # Extract values for later calculations
        xout = intout[-1,:]
        Zref_k = xout[0:nz].reshape(nz, 1)
        Xref_k = Zref_k[0:n].reshape(n, 1)
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
        # Lubey FUSION Eq 7-8
        xbar_k_km1 = np.dot(phi_xx, xhat_km1)
        Pbar_k_km1 = np.dot(phi_xx, np.dot(Phat_km1, phi_xx.T)) - np.dot(phi_xp, phi_xx.T)
        
        print('')
        print(tk)
        print(Xref_k)
        print(phi)
        print(xbar_k_km1)
        print(Pbar_k_km1)
        

        # Measurement Update           
        # Retrieve measurement data
        Yk = Yk_list[kk]
        sensor_id = sensor_id_list[kk]

        # Compute prefit residuals and measurement mapping matrix
        # Lubey FUSION Eq 9-10
        Hk_til, Gk, Rk = meas_fcn(tk, Zref_k, state_params, sensor_params, sensor_id)
        yk = Yk - Gk
        
        # Predicted residuals (innovations) and covariance
        Bk = yk - np.dot(Hk_til, xbar_k_km1)
        P_Bk = np.dot(Hk_til, np.dot(Pbar_k_km1, Hk_til.T)) + Rk 
        inv_Pbk = est.cholesky_inv(P_Bk)
        
        # Compute prior and posterior gain matrices
        # Lubey FUSION Eq 14-15
        L_km1 = Phat_km1 @ phi_xx.T @ Hk_til.T @ inv_Pbk
        L_k = Pbar_k_km1 @ Hk_til.T @ inv_Pbk
        
        # Updated state deviation vectors, covariances, adjoint deviation
        # Lubey FUSION Eq 11-13, 16-17
        xhat_km1_k = xhat_km1 + L_km1 @ Bk
        xhat_k = xbar_k_km1 + L_k @ Bk
        phat_km1_k = -est.cholesky_inv(Phat_km1) @ L_km1 @ Bk
        
        Phat_km1_k = Phat_km1 - L_km1 @ P_Bk @ L_km1.T
        P1 = np.eye(n) - L_k @ Hk_til
        P2 = L_k @ Rk @ L_k.T
        Phat_k = P1 @ Pbar_k_km1 @ P1.T + P2
        
        # Post-fit residuals and updated state
        resids = yk - np.dot(Hk_til, xhat_k)
        Xk = Xref_k + xhat_k
        
        # Store output
        filter_output[tk] = {}
        filter_output[tk]['Xref'] = Xref_k
        filter_output[tk]['X'] = Xk
        filter_output[tk]['P'] = Phat_k
        filter_output[tk]['resids'] = resids
        
        smoother_data[tk] = {}
        smoother_data[tk]['Xref'] = Xref_k
        smoother_data[tk]['xhat'] = xhat_k
        smoother_data[tk]['phi_xx'] = phi_xx
        smoother_data[tk]['phi_pp'] = phi_pp
        smoother_data[tk]['Pbar'] = Pbar_k_km1
        smoother_data[tk]['P'] = Phat_k
        
        # Overwrite previous time step
        # Note this is equivalent to a one step smoothing 
        # if smoothing the entire trajectory, not much point keeping this
        if tk > tk_prior and not smoothing:
            filter_output[tk_prior]['X'] = filter_output[tk_prior]['Xref'] + xhat_km1_k
            filter_output[tk_prior]['P'] = Phat_km1_k
        
    
    # Compute control inputs
    
    
    
    if smoothing:
        
        # Initialize
        xhat_kp1_l = smoother_data[tk_list[-1]]['xhat']
        Phat_kp1_l = smoother_data[tk_list[-1]]['P']
        filter_output[tk_list[-1]]['X_kl'] = filter_output[tk_list[-1]]['X']
        filter_output[tk_list[-1]]['P_kl'] = filter_output[tk_list[-1]]['P']
        filter_output[tk_list[-1]]['resids_kl'] = filter_output[tk_list[-1]]['resids']
        
        # Note that I set u(t0) = 0 and solve for u(t_k+1) throughout smoother
        # Lubey dissertation Fig 4.3-4.4 seem to suggest he does otherwise, set
        # final time to zero and solve backwards so nonzero value at t0.
        filter_output[tk_list[0]]['u_kl'] = 0
        
        # Loop backwards through time        
        for kk in range(N-2,-1,-1):
            
            # Retrieve data for this iteration
            t_k = tk_list[kk]
            t_kp1 = tk_list[kk+1]
            
            Xref_k = smoother_data[t_k]['Xref']
            xhat_k_k = smoother_data[t_k]['xhat']
            Phat_k_k = smoother_data[t_k]['P']
            phi_xx = smoother_data[t_kp1]['phi_xx']
            phi_pp = smoother_data[t_kp1]['phi_pp']
            Pbar_kp1_k = smoother_data[t_kp1]['Pbar']
            
            # Retrieve measurement data
            Yk = Yk_list[kk]
            sensor_id = sensor_id_list[kk]
    
            # Compute prefit residuals and  Kalman gain
            Hk_til, Gk, Rk = meas_fcn(tk, Xref_k, state_params, sensor_params, sensor_id)
            yk = Yk - Gk
            
            # Compute smoothed state estimate and covariance
            Sk = Phat_k_k @ phi_xx.T @ est.cholesky_inv(Pbar_kp1_k)
            xhat_k_l = xhat_k_k + Sk @ (xhat_kp1_l - np.dot(phi_xx, xhat_k_k))
            Phat_k_l = Phat_k_k + Sk @ (Phat_kp1_l - Pbar_kp1_k) @ Sk.T
            
            # Compute control inputs
            u = -Q @ B.T @ phi_pp @ est.cholesky_inv(Phat_k_k) @ (xhat_k_k - xhat_k_l)
            
            # Store output
            filter_output[t_k]['X_kl'] = Xref_k + xhat_k_l
            filter_output[t_k]['P_kl'] = Phat_k_l
            filter_output[t_kp1]['u_kl'] = u
            filter_output[t_k]['resids_kl'] = yk - np.dot(Hk_til, xhat_k_l)
            
            
            # Reset for next iteration
            xhat_kp1_l = xhat_k_l.copy()
            Phat_kp1_l = Phat_k_l.copy()
    
    
    
    return filter_output, 0


