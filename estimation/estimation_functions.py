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
from utilities import attitude as att
from utilities import coordinate_systems as coord
from utilities.constants import arcsec2rad




###############################################################################
# Batch Estimation
###############################################################################

def ls_batch(state_dict, truth_dict, meas_dict, meas_fcn, params_dict):
    '''
    This function implements the linearized batch estimator for the least
    squares cost function.

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
    state_params : dictionary
        physical parameters of spacecraft and central body
    sensor_params : dictionary
        location, constraint, noise parameters of sensors
    int_params : dictionary
        numerical integration parameters

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

    # State information
    state_tk = sorted(state_dict.keys())[-1]
    Xo_ref = state_dict[state_tk]['X']
    Po_bar = state_dict[state_tk]['P']

    # Setup
    cholPo = np.linalg.inv(np.linalg.cholesky(Po_bar))
    invPo_bar = np.dot(cholPo.T, cholPo)
    Xo_ref_print = Xo_ref.copy()

    n = len(Xo_ref)

    # Initialize output
    filter_output = {}

    # Measurement times
    tk_list = meas_dict['tk_list']
    Yk_list = meas_dict['Yk_list']
    sensor_id_list = meas_dict['sensor_id_list']
    nmeas = sum([len(Yk) for Yk in Yk_list])

    # Number of epochs
    N = len(tk_list)

    # Initialize
    maxiters = 10
    xo_bar = np.zeros((n, 1))
    xo_hat = np.zeros((n, 1))
    phi0 = np.identity(n)
    phi0_v = np.reshape(phi0, (n**2, 1))

    # Begin Loop
    iters = 0
    xo_hat_mag = 1
    rms_prior = 1e6
    xhat_crit = 1e-5
    rms_crit = 1e-4
    conv_flag = False    
    while not conv_flag:

        # Increment loop counter and exit if necessary
        iters += 1
        if iters > maxiters:
            iters -= 1
            print('Solution did not converge in ', iters, ' iterations')
            print('Last xo_hat magnitude: ', xo_hat_mag)
            break

        # Initialze values for this iteration
        Xref_list = []
        phi_list = []
        resids_list = []
        resids_sum = 0.
        phi_v = phi0_v.copy()
        Xref = Xo_ref.copy()
        Lambda = invPo_bar.copy()
        Nstate = np.dot(Lambda, xo_bar)

        # Loop over times        
        for kk in range(N):
            
#            print('\nkk = ', kk)
            
            # Current and previous time
            if kk == 0:
                tk_prior = state_tk
            else:
                tk_prior = tk_list[kk-1]

            tk = tk_list[kk]

            # Read the next observation
            Yk = Yk_list[kk]
            sensor_id = sensor_id_list[kk]

            # Initialize
            Xref_prior = Xref.copy()

            # Initial Conditions for Integration Routine
            int0 = np.concatenate((Xref_prior, phi_v))

            # Integrate Xref and STM
            if tk_prior == tk:
                intout = int0.T
            else:
                int0 = int0.flatten()
                tin = [tk_prior, tk]
                
                tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)

            # Extract values for later calculations
            xout = intout[-1,:]
            Xref = xout[0:n].reshape(n, 1)
            phi_v = xout[n:].reshape(n**2, 1)
            phi = np.reshape(phi_v, (n, n))
            
#            print('\n\n')
#            print('Xref', Xref)

            # Accumulate the normal equations
            Hk_til, Gk, Rk = meas_fcn(tk, Xref, state_params, sensor_params, sensor_id)
            yk = Yk - Gk
            Hk = np.dot(Hk_til, phi)
            cholRk = np.linalg.inv(np.linalg.cholesky(Rk))
            invRk = np.dot(cholRk.T, cholRk)
                        
            Lambda += np.dot(Hk.T, np.dot(invRk, Hk))
            Nstate += np.dot(Hk.T, np.dot(invRk, yk))
            
            # Store output
            resids_list.append(yk)
            Xref_list.append(Xref)
            phi_list.append(phi)
            resids_sum += float(np.dot(yk.T, np.dot(invRk, yk)))
            
            # print(kk)
            # print(tk)
            # print(int0)
            # print(Xref)
            # print(Yk)
            # print(Gk)
            # print(yk)
            
            # if kk > 2:
            #     mistake


        # print(Lambda)
        # print(np.linalg.eig(Lambda))


        # Solve the normal equations
        cholLam_inv = np.linalg.inv(np.linalg.cholesky(Lambda))
        Po = np.dot(cholLam_inv.T, cholLam_inv)     
        xo_hat = np.dot(Po, Nstate)
        xo_hat_mag = np.linalg.norm(xo_hat)

        # Update for next batch iteration
        Xo_ref = Xo_ref + xo_hat
        xo_bar = xo_bar - xo_hat
        
        # Evaluate xo_hat_mag and resids for convergence
#        if xo_hat_mag < xhat_crit:
#            conv_flag = True
            
        resids_rms = np.sqrt(resids_sum/nmeas)
        resids_diff = abs(resids_rms - rms_prior)/rms_prior
        if resids_diff < rms_crit:
            conv_flag = True
            
        rms_prior = float(resids_rms)
        

        print('Iteration Number: ', iters)
        print('xo_hat_mag = ', xo_hat_mag)
        print('delta-X = ', xo_hat)
        print('Xo', Xo_ref_print)
        print('X', Xo_ref)
        print('resids_rms = ', resids_rms)
        print('resids_diff = ', resids_diff)
        
#        # DEBUG
#        t_hrs = [(tt - tk_list[0]).total_seconds()/3600. for tt in tk_list]
#        rg_resids = [res[0]*1000. for res in resids_list]
#        ra_resids = [res[1]*(1./arcsec2rad) for res in resids_list]
#        dec_resids = [res[2]*(1./arcsec2rad) for res in resids_list]
#        
#        plt.figure()
#        plt.subplot(3,1,1)
#        plt.plot(t_hrs, rg_resids, 'k.')
#        plt.ylabel('Range [m]')
#        plt.subplot(3,1,2)
#        plt.plot(t_hrs, ra_resids, 'k.')
#        plt.ylabel('RA [arcsec]')
#        plt.subplot(3,1,3)
#        plt.plot(t_hrs, dec_resids, 'k.')
#        plt.ylabel('DEC [arcsec]')
#        plt.xlabel('Time [hours]')
#        
#        plt.show()
        
#        if iters > 1:
#            mistake

#    # Form output
#    for kk in range(N):
#        tk = tk_list[kk]
#        X = Xref_list[kk]
#        resids = resids_list[kk]
#        phi = phi_list[kk]
#        P = np.dot(phi, np.dot(Po, phi.T))
#
#        filter_output[tk] = {}
#        filter_output[tk]['X'] = X
#        filter_output[tk]['P'] = P
#        filter_output[tk]['resids'] = resids
        
    
    # Integrate over full time
    tk_truth = list(truth_dict.keys())
    phi_v = phi0_v.copy()
    Xref = Xo_ref.copy()
    full_state_output = {}
    for kk in range(len(tk_truth)):
        
        # Current and previous time
        if kk == 0:
            tk_prior = state_tk
        else:
            tk_prior = tk_truth[kk-1]
            
        tk = tk_truth[kk]
        
        # Initial Conditions for Integration Routine
        Xref_prior = Xref.copy()
        int0 = np.concatenate((Xref_prior, phi_v))

        # Integrate Xref and STM
        if tk_prior == tk:
            intout = int0.T
        else:
            int0 = int0.flatten()
            tin = [tk_prior, tk]
            
            tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)

        # Extract values for later calculations
        xout = intout[-1,:]
        Xref = xout[0:n].reshape(n, 1)
        phi_v = xout[n:].reshape(n**2, 1)
        phi = np.reshape(phi_v, (n, n))
        P = np.dot(phi, np.dot(Po, phi.T))
        
        full_state_output[tk] = {}
        full_state_output[tk]['X'] = Xref
        full_state_output[tk]['P'] = P
        
        if tk in tk_list:
            filter_output[tk] = {}
            filter_output[tk]['X'] = Xref
            filter_output[tk]['P'] = P
            filter_output[tk]['resids'] = resids_list[tk_list.index(tk)]
    

    return filter_output, full_state_output


def lp_batch(state_dict, truth_dict, meas_dict, meas_fcn, params_dict):
    '''
    This function implements the linearized batch estimator for the minimum
    Lp-norm cost function, not including L1-norm (Least Absolute Deviations).
    Selection of pnorm = 2 will produce the least squares estimate with 
    slightly more computational cost.
    
    The only allowable values of pnorm are pnorm > 1.
    
    For measurement data with outliers, it is recommended to set the input
    parameter pnorm to a value 1 < pnorm < 2, for example pnorm = 1.2.

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
    state_params : dictionary
        physical parameters of spacecraft and central body
    sensor_params : dictionary
        location, constraint, noise parameters of sensors
    int_params : dictionary
        numerical integration parameters

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
    
    # Cost function parameter
    pnorm = filter_params['pnorm']
    
    # Rescale noise for pnorm distribution
    scale = (math.gamma(3./pnorm)/math.gamma(1./pnorm)) * pnorm**(2./pnorm)
    print(scale)

    # Setup
    cholPo = np.linalg.inv(np.linalg.cholesky(Po_bar))
    invPo_bar = np.dot(cholPo.T, cholPo)
    Xo_ref_print = Xo_ref.copy()

    n = len(Xo_ref)

    # Initialize output
    filter_output = {}

    # Measurement times
    tk_list = meas_dict['tk_list']
    Yk_list = meas_dict['Yk_list']
    sensor_id_list = meas_dict['sensor_id_list']
    nmeas = sum([len(Yk) for Yk in Yk_list])

    # Number of epochs
    N = len(tk_list)

    # Initialize    
    xo_bar = np.zeros((n, 1))
    xo_hat = np.zeros((n, 1))
    phi0 = np.identity(n)
    phi0_v = np.reshape(phi0, (n**2, 1))

    # Begin Loop
    iters = 0
    maxiters = 10
    newt_maxiters = 100
    xo_hat_mag = 1
    rms_prior = 1e6
    xhat_crit = 1e-5
    rms_crit = 1e-4
    newt_crit = 1e-10
    conv_flag = False    
    while not conv_flag:

        # Increment loop counter and exit if necessary
        iters += 1
        if iters > maxiters:
            iters -= 1
            print('Solution did not converge in ', iters, ' iterations')
            print('Last xo_hat magnitude: ', xo_hat_mag)
            break

        # Initialze values for this iteration
        Xref_list = []
        phi_list = []
        resids_list = []
        Hk_list = []
        Rk_list = []
        resids_sum = 0.
        phi_v = phi0_v.copy()
        Xref = Xo_ref.copy()
        

        # Loop over times        
        for kk in range(N):
            
#            print('\nkk = ', kk)
            
            # Current and previous time
            if kk == 0:
                tk_prior = state_tk
            else:
                tk_prior = tk_list[kk-1]

            tk = tk_list[kk]

            # Read the next observation
            Yk = Yk_list[kk]
            sensor_id = sensor_id_list[kk]

            # Initialize
            Xref_prior = Xref.copy()

            # Initial Conditions for Integration Routine
            int0 = np.concatenate((Xref_prior, phi_v))

            # Integrate Xref and STM
            if tk_prior == tk:
                intout = int0.T
            else:
                int0 = int0.flatten()
                tin = [tk_prior, tk]
                
                tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)

            # Extract values for later calculations
            xout = intout[-1,:]
            Xref = xout[0:n].reshape(n, 1)
            phi_v = xout[n:].reshape(n**2, 1)
            phi = np.reshape(phi_v, (n, n))
            
#            print('\n\n')
#            print('Xref', Xref)

            # Compute and store data
            Hk_til, Gk, Rk = meas_fcn(tk, Xref, state_params, sensor_params, sensor_id)            
            yk = Yk - Gk
            Hk = np.dot(Hk_til, phi)
            
            cholRk = np.linalg.inv(np.linalg.cholesky(Rk))
            invRk = np.dot(cholRk.T, cholRk)            
            resids_sum += float(np.dot(yk.T, np.dot(invRk, yk)))
            
#            Rk = scale*Rk

            # Store output
            resids_list.append(yk)
            Xref_list.append(Xref)
            phi_list.append(phi)
            Hk_list.append(Hk)
            Rk_list.append(Rk)
            
            
            
            # print(kk)
            # print(tk)
            # print(int0)
            # print(Xref)
            # print(Yk)
            # print(Gk)
            # print(yk)
            
            # if kk > 2:
            #     mistake


        # print(Lambda)
        # print(np.linalg.eig(Lambda))


        # Solve the normal equations
        # Newton Raphson iteration to get best xo_hat
        newt_diff = 1.
        xo_bar_newt = xo_bar.copy()

        newt_iters = 0
        while newt_diff > newt_crit:

            newt_iters += 1
            if newt_iters > newt_maxiters:
                print('Newton iteration #', newt_iters)
                print('Newton iteration difference magnitude:', newt_diff)
                break
            
            # Initialize for this iteration
            Lambda = invPo_bar.copy()
            Nstate = np.dot(Lambda, xo_bar_newt)
            
            # Loop over times
            for kk in range(N):
                
                # Retrieve values
                yk = resids_list[kk]
                Hk = Hk_list[kk]
                Rk = Rk_list[kk]
                
                # Compute weighting matrix
                W_vect = abs(yk - np.dot(Hk, xo_hat))**(pnorm-2.)
                W = np.diag(W_vect.flatten())
                
                # Compute inverse of Rk
                cholRk = np.linalg.inv(np.linalg.cholesky(Rk))
                invRk = np.dot(cholRk.T, cholRk)
                invRk_pover2 = np.diag(np.diag(Rk)**(-pnorm/2.))
#                invRk_pover2 = invRk
                
#                print(invRk)
#                print(invRk_pover2)
                
#                mistake
            
                # Accumulate quantities of interest
                Lambda += (pnorm-1.)*\
                    np.dot(Hk.T, np.dot(W, np.dot(invRk_pover2, Hk)))
                abs_vect = np.multiply(abs(yk-np.dot(Hk, xo_hat))**(pnorm-1.),
                                       np.sign(yk-np.dot(Hk, xo_hat)))
                Nstate += np.dot(Hk.T, np.dot(invRk_pover2, abs_vect))
                
            # Solve the normal equations
            cholLam_inv = np.linalg.inv(np.linalg.cholesky(Lambda))
            Po = np.dot(cholLam_inv.T, cholLam_inv)   
            
            if pnorm > 2.:
                alpha = 1.
            else:
                alpha = pnorm - 1.

            xo_hat += alpha * np.dot(Po, Nstate)
            xo_bar_newt = xo_bar - xo_hat
            xo_hat_mag = np.linalg.norm(xo_hat)
            newt_diff = alpha * np.linalg.norm(np.dot(Po, Nstate))
        

        # Update for next batch iteration
        Xo_ref = Xo_ref + xo_hat
        xo_bar = xo_bar - xo_hat
        
        # Evaluate xo_hat_mag and resids for convergence
#        if xo_hat_mag < xhat_crit:
#            conv_flag = True
            
        resids_rms = np.sqrt(resids_sum/nmeas)
        resids_diff = abs(resids_rms - rms_prior)/rms_prior
        if resids_diff < rms_crit:
            conv_flag = True
            
        rms_prior = float(resids_rms)
        

        print('Iteration Number: ', iters)
        print('xo_hat_mag = ', xo_hat_mag)
        print('delta-X = ', xo_hat)
        print('Xo', Xo_ref_print)
        print('X', Xo_ref)
        print('resids_rms = ', resids_rms)
        print('resids_diff = ', resids_diff)
        
#        # DEBUG
#        t_hrs = [(tt - tk_list[0]).total_seconds()/3600. for tt in tk_list]
#        rg_resids = [res[0]*1000. for res in resids_list]
#        ra_resids = [res[1]*(1./arcsec2rad) for res in resids_list]
#        dec_resids = [res[2]*(1./arcsec2rad) for res in resids_list]
#        
#        plt.figure()
#        plt.subplot(3,1,1)
#        plt.plot(t_hrs, rg_resids, 'k.')
#        plt.ylabel('Range [m]')
#        plt.subplot(3,1,2)
#        plt.plot(t_hrs, ra_resids, 'k.')
#        plt.ylabel('RA [arcsec]')
#        plt.subplot(3,1,3)
#        plt.plot(t_hrs, dec_resids, 'k.')
#        plt.ylabel('DEC [arcsec]')
#        plt.xlabel('Time [hours]')
#        
#        plt.show()
        
#        if iters > 1:
#            mistake

#    # Form output
#    for kk in range(N):
#        tk = tk_list[kk]
#        X = Xref_list[kk]
#        resids = resids_list[kk]
#        phi = phi_list[kk]
#        P = np.dot(phi, np.dot(Po, phi.T))
#
#        filter_output[tk] = {}
#        filter_output[tk]['X'] = X
#        filter_output[tk]['P'] = P
#        filter_output[tk]['resids'] = resids
        
    
    # Integrate over full time
    tk_truth = list(truth_dict.keys())
    phi_v = phi0_v.copy()
    Xref = Xo_ref.copy()
    full_state_output = {}
    for kk in range(len(tk_truth)):
        
        # Current and previous time
        if kk == 0:
            tk_prior = state_tk
        else:
            tk_prior = tk_truth[kk-1]
            
        tk = tk_truth[kk]
        
        # Initial Conditions for Integration Routine
        Xref_prior = Xref.copy()
        int0 = np.concatenate((Xref_prior, phi_v))

        # Integrate Xref and STM
        if tk_prior == tk:
            intout = int0.T
        else:
            int0 = int0.flatten()
            tin = [tk_prior, tk]
            
            tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)

        # Extract values for later calculations
        xout = intout[-1,:]
        Xref = xout[0:n].reshape(n, 1)
        phi_v = xout[n:].reshape(n**2, 1)
        phi = np.reshape(phi_v, (n, n))
        P = np.dot(phi, np.dot(Po, phi.T))
        
        full_state_output[tk] = {}
        full_state_output[tk]['X'] = Xref
        full_state_output[tk]['P'] = P
        
        if tk in tk_list:
            filter_output[tk] = {}
            filter_output[tk]['X'] = Xref
            filter_output[tk]['P'] = P
            filter_output[tk]['resids'] = resids_list[tk_list.index(tk)]
    

    return filter_output, full_state_output


def unscented_batch(state_dict, truth_dict, meas_dict, meas_fcn, params_dict):
    '''
    This function implements the unscented batch estimator for the least
    squares cost function.

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
    state_params : dictionary
        physical parameters of spacecraft and central body
    sensor_params : dictionary
        location, constraint, noise parameters of sensors
    int_params : dictionary
        numerical integration parameters

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
    Xo = state_dict[state_tk]['X']
    Po = state_dict[state_tk]['P']

    # Setup
    n = len(Xo)
    
    # Prior information about the distribution
    pnorm = 2.
    kurt = math.gamma(5./pnorm)*math.gamma(1./pnorm)/(math.gamma(3./pnorm)**2.)
    beta = kurt - 1.
    kappa = kurt - float(n)
    
    # Compute sigma point weights
    alpha = filter_params['alpha']
    lam = alpha**2.*(n + kappa) - n
    gam = np.sqrt(n + lam)
    Wm = 1./(2.*(n + lam)) * np.ones(2*n,)
    Wc = Wm.copy()
    Wm = np.insert(Wm, 0, lam/(n + lam))
    Wc = np.insert(Wc, 0, lam/(n + lam) + (1 - alpha**2 + beta))
    diagWc = np.diag(Wc)

    # Initialize output
    filter_output = {}

    # Measurement times
    tk_list = meas_dict['tk_list']
    Yk_list = meas_dict['Yk_list']
    sensor_id_list = meas_dict['sensor_id_list']
    nmeas = sum([len(Yk) for Yk in Yk_list])
    
#    print('nmeas', nmeas)

    # Number of epochs
    N = len(tk_list)

    # Initialize 
    X = Xo.copy()
    P = Po.copy()
    maxiters = 10
    xdiff = 1
    rms_prior = 1e6
    xdiff_crit = 1e-5
    rms_crit = 1e-4
    conv_flag = False 
    
    # Begin loop
    iters = 0
    while not conv_flag:

        # Increment loop counter and exit if necessary
        iters += 1
        if iters > maxiters:
            iters -= 1
            print('Solution did not converge in ', iters, ' iterations')
            print('Last xdiff magnitude: ', xdiff)
            break

        # Initialze values for this iteration
        # Reset P every iteration???
        P = Po.copy()
        
        # Compute Sigma Points
        sqP = np.linalg.cholesky(P)
        Xrep = np.tile(X, (1, n))
        chi0 = np.concatenate((X, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
        chi_v = np.reshape(chi0, (n*(2*n+1), 1), order='F')  
        chi_diff0 = chi0 - np.dot(X, np.ones((1, 2*n+1)))

        # Loop over times
        meas_ind = 0
        Y_bar = np.zeros((nmeas, 1))
        Y_til = np.zeros((nmeas, 1))
        gamma_til_mat = np.zeros((nmeas, 2*n+1))
        Rk_list = []
        resids_list = []
        resids_sum = 0.
        for kk in range(N):
            
#            print('\nkk = ', kk)
            
            # Current and previous time
            if kk == 0:
                tk_prior = state_tk
            else:
                tk_prior = tk_list[kk-1]

            tk = tk_list[kk]

            # Read the next observation
            Yk = Yk_list[kk]
            sensor_id = sensor_id_list[kk]
            p = len(Yk)

            # Initial Conditions for Integration Routine
            int0 = chi_v.copy()

            # Integrate Xref and STM
            if tk_prior == tk:
                intout = int0.T
            else:
                int0 = int0.flatten()
                tin = [tk_prior, tk]
                
                tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)

            # Extract values for later calculations
            chi_v = intout[-1,:]
            chi = np.reshape(chi_v, (n, 2*n+1), order='F')
            
            # Compute measurement for each sigma point
            gamma_til_k, Rk = meas_fcn(tk, chi, state_params, sensor_params, sensor_id)
            
            # Standard implementation computes ybar as the mean of the sigma
            # point, but using Po_bar each iteration can cause these to have
            # large spread and produce poor ybar calculation
            # ybar = np.dot(gamma_til_k, Wm.T)
            
            # Instead, use only the first column of gamma_til_k, corresponding
            # to the mean state calculated with the best updated value of X(t0)
            ybar = gamma_til_k[:,0]
            
            # Reshape and continue
            ybar = np.reshape(ybar, (p, 1))
            resids = Yk - ybar
            cholRk = np.linalg.inv(np.linalg.cholesky(Rk))
            invRk = np.dot(cholRk.T, cholRk)
            
            # Accumulate measurements and computed measurements
            Y_til[meas_ind:meas_ind+p] = Yk
            Y_bar[meas_ind:meas_ind+p] = ybar
            gamma_til_mat[meas_ind:meas_ind+p, :] = gamma_til_k  
            Rk_list.append(Rk)
            
            # Store output
            resids_list.append(resids)
            resids_sum += float(np.dot(resids.T, np.dot(invRk, resids)))
            
#            print('kk', kk)
#            print('Yk', Yk)
#            print('ybar', ybar)
#            
#            if kk == 27:
#                print(gamma_til_k)
#                print(Rk)
#                mistake
            
            
            
#            Xk = np.dot(chi, Wm.T)
#            Xk = np.reshape(Xk, (n, 1))
#            chi_diff = chi - np.dot(Xk, np.ones((1, (2*n+1))))
#            Pk = np.dot(chi_diff, np.dot(diagWc, chi_diff.T))
#            
#            if tk not in filter_output:
#                filter_output[tk] = {}
#            filter_output[tk]['X'] = Xk
#            filter_output[tk]['P'] = Pk
#            filter_output[tk]['resids'] = resids
            
            # Increment measurement index
            meas_ind += p


#        # DEBUG
#        t_hrs = [(tt - tk_list[0]).total_seconds()/3600. for tt in tk_list]
#        rg_resids = [res[0]*1000. for res in resids_list]
#        ra_resids = [res[1]*(1./arcsec2rad) for res in resids_list]
#        dec_resids = [res[2]*(1./arcsec2rad) for res in resids_list]
#        
#        plt.figure()
#        plt.subplot(3,1,1)
#        plt.plot(t_hrs, rg_resids, 'k.')
#        plt.ylabel('Range [m]')
#        plt.subplot(3,1,2)
#        plt.plot(t_hrs, ra_resids, 'k.')
#        plt.ylabel('RA [arcsec]')
#        plt.subplot(3,1,3)
#        plt.plot(t_hrs, dec_resids, 'k.')
#        plt.ylabel('DEC [arcsec]')
#        plt.xlabel('Time [hours]')
#        
#        plt.show()

        # Compute covariances
        Rk_full = scipy.linalg.block_diag(*Rk_list)
        Y_diff = gamma_til_mat - np.dot(Y_bar, np.ones((1, 2*n+1)))
        
#        print(Rk_full.shape)
#        print(gamma_til_mat.shape)
#        print(Y_diff.shape)
#        
#        mistake
        
        Pyy = np.dot(Y_diff, np.dot(diagWc, Y_diff.T)) + Rk_full
        Pxy = np.dot(chi_diff0, np.dot(diagWc, Y_diff.T)) 
        
#        print(Pyy)
#        print(np.linalg.eig(Pyy))
        
#        print(Y_diff.shape)
#        print(diagWc.shape)
#        print(Rk_full.shape)
#        
#        np.linalg.cholesky(Rk_full)
#        np.linalg.cholesky(np.dot(Y_diff, Y_diff.T))

        # Compute Kalman Gain
        cholPyy_inv = np.linalg.inv(np.linalg.cholesky(Pyy))
        Pyy_inv = np.dot(cholPyy_inv.T, cholPyy_inv) 
        
#        Pyy_inv = np.linalg.inv(Pyy)
        
        K = np.dot(Pxy, Pyy_inv)
        
#        K = np.dot(Pxy, np.linalg.inv(Pyy))
        

        # Compute updated state and covariance    
        X += np.dot(K, Y_til-Y_bar)
#        P = P - np.dot(K, np.dot(Pyy, K.T))
        
        # Joseph Form        
        cholPbar = np.linalg.inv(np.linalg.cholesky(P))
        invPbar = np.dot(cholPbar.T, cholPbar)
        P1 = (np.eye(n) - np.dot(np.dot(K, np.dot(Pyy, K.T)), invPbar))
        P2 = np.dot(K, np.dot(Rk_full, K.T))
        P = np.dot(P1, np.dot(P, P1.T)) + P2
        
        
        xdiff = np.linalg.norm(np.dot(K, Y_til-Y_bar))
        
        # Evaluate xo_hat_mag and resids for convergence
#        if xdiff < xdiff_crit:
#            conv_flag = True
            
        resids_rms = np.sqrt(resids_sum/nmeas)
        resids_diff = abs(resids_rms - rms_prior)/rms_prior
        if resids_diff < rms_crit:
            conv_flag = True
            
        rms_prior = float(resids_rms)
        

        print('Iteration Number: ', iters)
        print('xdiff = ', xdiff)
        print('delta-X = ', np.dot(K, Y_til-Y_bar))
        print('Xo', Xo)
        print('X', X)
        print('resids_rms = ', resids_rms)
        print('resids_diff = ', resids_diff)
        
        
#        
#        if iters > 3:
#            mistake

    
    
    # Setup for full_state_output
    Xo = X.copy()
    Po = P.copy()
    
    # Compute Sigma Points
    sqP = np.linalg.cholesky(Po)
    Xrep = np.tile(Xo, (1, n))
    chi0 = np.concatenate((Xo, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
    chi_v = np.reshape(chi0, (n*(2*n+1), 1), order='F')
    
    # Integrate over full time
    tk_truth = list(truth_dict.keys())
    full_state_output = {}
    for kk in range(len(tk_truth)):
        
        # Current and previous time
        if kk == 0:
            tk_prior = state_tk
        else:
            tk_prior = tk_truth[kk-1]
            
        tk = tk_truth[kk]

        # Initial Conditions for Integration Routine
        int0 = chi_v.copy()

        # Integrate Xref and STM
        if tk_prior == tk:
            intout = int0.T
        else:
            int0 = int0.flatten()
            tin = [tk_prior, tk]
            
            tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)

        # Extract values for later calculations
        chi_v = intout[-1,:]
        chi = np.reshape(chi_v, (n, 2*n+1), order='F')
    
        # Store output
        Xk = np.dot(chi, Wm.T)
        Xk = np.reshape(Xk, (n, 1))
        chi_diff = chi - np.dot(Xk, np.ones((1, (2*n+1))))
        Pk = np.dot(chi_diff, np.dot(diagWc, chi_diff.T))
        
        full_state_output[tk] = {}
        full_state_output[tk]['X'] = Xk
        full_state_output[tk]['P'] = Pk
        
        if tk in tk_list:
            filter_output[tk] = {}
            filter_output[tk]['X'] = Xk
            filter_output[tk]['P'] = Pk
            filter_output[tk]['resids'] = resids_list[tk_list.index(tk)]

    return filter_output, full_state_output


###############################################################################
# Sequential Estimation
###############################################################################

def ls_ekf(state_dict, truth_dict, meas_dict, meas_fcn, params_dict,
           smoothing=False):    
    '''
    This function implements the linearized Extended Kalman Filter for the 
    least squares cost function.

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
    state_params : dictionary
        physical parameters and constants
    sensor_params : dictionary
        location, constraint, noise parameters of sensors
    int_params : dictionary
        numerical integration parameters

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
    smoother_data = {}

    # Measurement times
    tk_list = meas_dict['tk_list']
    Yk_list = meas_dict['Yk_list']
    sensor_id_list = meas_dict['sensor_id_list']
    
    # Number of epochs
    N = len(tk_list)

    # Initialize
    xhat = np.zeros((n, 1))
    P = Po_bar
    Xref = Xo_ref
    phi = np.identity(n)
    phi0_v = np.reshape(phi, (n**2, 1))
    conv_flag = False
    count = 0
    
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
        Xref_prior = Xref
        xhat_prior = xhat
        P_prior = P
        int0 = np.concatenate((Xref_prior, phi0_v))

        # Integrate Xref and STM
        if tk_prior == tk:
            intout = int0.T
        else:
            int0 = int0.flatten()
            tin = [tk_prior, tk]
            
            tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)

        # Extract values for later calculations
        xout = intout[-1,:]
        Xref = xout[0:n].reshape(n, 1)
        phi_v = xout[n:].reshape(n**2, 1)
        phi = np.reshape(phi_v, (n, n))
        
        # Time Update: a priori state and covar at tk
        
        # State Noise Compensation
        # Zero out SNC for long time gaps
        if delta_t > gap_seconds:        
            Gamma = np.zeros((n,q))
        else:
            Gamma = np.zeros((n,q))
            Gamma[0:q,:] = (delta_t**2./2) * np.eye(q)
            Gamma[q:2*q,:] = delta_t * np.eye(q)
#            Gamma = delta_t * np.concatenate((np.eye(q)*delta_t/2., np.eye(q)))

        xbar = np.dot(phi, xhat_prior)
        Pbar = np.dot(phi, np.dot(P_prior, phi.T)) + np.dot(Gamma, np.dot(Q, Gamma.T))
        
        # Measurement Update: posterior state and covar at tk            
        # Retrieve measurement data
        Yk = Yk_list[kk]
        sensor_id = sensor_id_list[kk]

        # Compute prefit residuals and  Kalman gain
        Hk_til, Gk, Rk = meas_fcn(tk, Xref, state_params, sensor_params, sensor_id)
        yk = Yk - Gk
        
        K1 = np.dot(Pbar, Hk_til.T)
        K2 = np.dot(Hk_til, np.dot(Pbar, Hk_til.T)) + Rk        
        Kk = np.dot(K1, np.linalg.inv(K2))
        
        # Predicted residuals
        Bk = yk - np.dot(Hk_til, xbar)
        P_bk = K2
        
        # Measurement update (Joseph form of covariance)
        xhat = xbar + np.dot(Kk, Bk)
        P1 = np.eye(n) - np.dot(Kk, Hk_til)
        P2 = np.dot(Kk, np.dot(Rk, Kk.T))
        P = np.dot(P1, np.dot(Pbar, P1.T)) + P2
        
        # P1 = np.eye(n) - np.dot(Kk, Hk_til)
        # P = np.dot(P1, Pbar)
        
        # Re-symmetric covariance
#        P = 0.5 * (P + P.T)
        
        # Post-fit residuals and updated state
        resids = yk - np.dot(Hk_til, xhat)
        Xk = Xref + xhat
        
        # Store output
        filter_output[tk] = {}
        filter_output[tk]['X'] = Xk
        filter_output[tk]['P'] = P
        filter_output[tk]['resids'] = resids
        
        smoother_data[tk] = {}
        smoother_data[tk]['Xref'] = Xref
        smoother_data[tk]['xhat'] = xhat
        smoother_data[tk]['phi'] = phi
        smoother_data[tk]['Pbar'] = Pbar
        smoother_data[tk]['P'] = P
        
        
        
#        print('\n')
#        print('tk', tk)
#        print('xbar', xbar)
#        print('xhat', xhat)
#        print('Xref', Xref)
#        print('Xk', Xk)
#        print('Yk', Yk)
#        print('Hk_til', Hk_til)
#        print('Kk', Kk)
#        print('resids', resids)
#        print('Gamma', Gamma)
#        print('phi', phi)
#        print('P_prior', P_prior)
#        print('Pbar', Pbar)
#        print('P', P)
        
#        if kk > 2:
#             mistake
        
        # Check convergence criteria and set flag to use EKF
#        if kk > 10:
#        P_diff = np.trace(P)/np.trace(P_prior)
#        print('\n')
#        print(kk)
#        print(P_diff)
        
#        if P_diff > 0.9 and P_diff < 1.0:
#            conv_flag = True
#        else:
#            conv_flag = False
        
        if count > 5:
            conv_flag = True

        # Don't use EKF after big gaps
        if delta_t > gap_seconds:
            conv_flag = False
            count = 0
            
        count += 1
        
        # Don't use EKF for smoothing cases
        if smoothing:
            conv_flag = False
        

        # After filter convergence, update reference trajectory
        if conv_flag:
            Xref = Xk
            xhat = np.zeros((n, 1))
            
            
    
    
    # Smoothing
    if smoothing:
        
        # Initialize
        xhat_kp1_l = smoother_data[tk_list[-1]]['xhat']
        Phat_kp1_l = smoother_data[tk_list[-1]]['P']
        filter_output[tk_list[-1]]['X_kl'] = filter_output[tk_list[-1]]['X']
        filter_output[tk_list[-1]]['P_kl'] = filter_output[tk_list[-1]]['P']
        filter_output[tk_list[-1]]['resids_kl'] = filter_output[tk_list[-1]]['resids']
        
        # Loop backwards through time        
        for kk in range(N-2,-1,-1):
            
            # Retrieve data for this iteration
            t_k = tk_list[kk]
            t_kp1 = tk_list[kk+1]
            
            Xref_k = smoother_data[t_k]['Xref']
            xhat_k_k = smoother_data[t_k]['xhat']
            Phat_k_k = smoother_data[t_k]['P']
            phi = smoother_data[t_kp1]['phi']
            Pbar_kp1_k = smoother_data[t_kp1]['Pbar']
            
            # Retrieve measurement data
            Yk = Yk_list[kk]
            sensor_id = sensor_id_list[kk]
    
            # Compute prefit residuals and  Kalman gain
            Hk_til, Gk, Rk = meas_fcn(tk, Xref_k, state_params, sensor_params, sensor_id)
            yk = Yk - Gk
            
            # Compute smoothed state estimate and covariance
            Sk = Phat_k_k @ phi.T @ cholesky_inv(Pbar_kp1_k)
            xhat_k_l = xhat_k_k + Sk @ (xhat_kp1_l - np.dot(phi, xhat_k_k))
            Phat_k_l = Phat_k_k + Sk @ (Phat_kp1_l - Pbar_kp1_k) @ Sk.T
            
            # Store output
            filter_output[t_k]['X_kl'] = Xref_k + xhat_k_l
            filter_output[t_k]['P_kl'] = Phat_k_l
            filter_output[t_k]['resids_kl'] = yk - np.dot(Hk_til, xhat_k_l)            
            
            # Reset for next iteration
            xhat_kp1_l = xhat_k_l.copy()
            Phat_kp1_l = Phat_k_l.copy()
        
        
    
            
    # TODO Generation of full_state_output not working correctly
    # Use filter_output for error analysis
    
    full_state_output = {}
            
#    # Integrate over full time
#    tk_truth = list(truth_dict.keys())
#    Xk = Xo_ref.copy()
#    Pk = Po_bar.copy()
#    full_state_output = {}
#    for kk in range(len(tk_truth)):
#        
#        # Current and previous time
#        if kk == 0:
#            tk_prior = state_tk
#        else:
#            tk_prior = tk_truth[kk-1]
#            
#        tk = tk_truth[kk]
#        
#        
#        # If current time is in filter output, retrieve values from filter 
#        # state
#        if tk in filter_output:
#            Xk = filter_output[tk]['X']
#            Pk = filter_output[tk]['P']
#        
#        # If not, then integrate to get estimated state/covar for this time
#        
#            # Initial Conditions for Integration Routine
#            Xk_prior = Xk.copy()
#            Pk_prior = Pk.copy()
#            int0 = np.concatenate((Xk_prior, phi0_v))
#    
#            # Integrate Xref and STM
#            if tk_prior == tk:
#                intout = int0.T
#            else:
#                int0 = int0.flatten()
#                tin = [tk_prior, tk]
#                
#                tout, intout = general_dynamics(int0, tin, state_params, int_params)
#
#            # Extract values for later calculations
#            xout = intout[-1,:]
#            Xk = xout[0:n].reshape(n, 1)
#            phi_v = xout[n:].reshape(n**2, 1)
#            phi = np.reshape(phi_v, (n, n))
#            Pk = np.dot(phi, np.dot(Pk_prior, phi.T))
#        
#        full_state_output[tk] = {}
#        full_state_output[tk]['X'] = Xk
#        full_state_output[tk]['P'] = Pk
    
    
    return filter_output, full_state_output


def ls_ukf(state_dict, truth_dict, meas_dict, meas_fcn, params_dict):    
    '''
    This function implements the Unscented Kalman Filter for the least
    squares cost function.

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
    state_params : dictionary
        physical parameters and constants
    sensor_params : dictionary
        location, constraint, noise parameters of sensors
    int_params : dictionary
        numerical integration parameters

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
    Xk = state_dict[state_tk]['X']
    P = state_dict[state_tk]['P']
    Q = filter_params['Q']
    gap_seconds = filter_params['gap_seconds']
    time_format = int_params['time_format']

    n = len(Xk)
    q = int(Q.shape[0])
    
    # Prior information about the distribution
    pnorm = 2.
    kurt = math.gamma(5./pnorm)*math.gamma(1./pnorm)/(math.gamma(3./pnorm)**2.)
    beta = kurt - 1.
    kappa = kurt - float(n)
    
    # Compute sigma point weights
    alpha = filter_params['alpha']
    lam = alpha**2.*(n + kappa) - n
    gam = np.sqrt(n + lam)
    Wm = 1./(2.*(n + lam)) * np.ones(2*n,)
    Wc = Wm.copy()
    Wm = np.insert(Wm, 0, lam/(n + lam))
    Wc = np.insert(Wc, 0, lam/(n + lam) + (1 - alpha**2 + beta))
    diagWc = np.diag(Wc)

    # Initialize output
    filter_output = {}

    # Measurement times
    tk_list = meas_dict['tk_list']
    Yk_list = meas_dict['Yk_list']
    sensor_id_list = meas_dict['sensor_id_list']
    
    # Number of epochs
    N = len(tk_list)
  
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

        # Compute sigma points matrix
        sqP = np.linalg.cholesky(P)
        Xrep = np.tile(Xk, (1, n))
        chi = np.concatenate((Xk, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
        chi_v = np.reshape(chi, (n*(2*n+1), 1), order='F')
        
        # Propagate to next time
        if tk_prior == tk:
            intout = chi_v.T
        else:
            int0 = chi_v.flatten()
            tin = [tk_prior, tk]
            
            tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)

        # Extract values for later calculations
        chi_v = intout[-1,:]
        chi = np.reshape(chi_v, (n, 2*n+1), order='F')
       
        # State Noise Compensation
        # Zero out SNC for long time gaps
        if delta_t > gap_seconds:        
            Gamma = np.zeros((n,q))
        else:
            Gamma = np.zeros((n,q))
            Gamma[0:q,:] = (delta_t**2./2) * np.eye(q)
            Gamma[q:2*q,:] = delta_t * np.eye(q)
#            Gamma = delta_t * np.concatenate((np.eye(q)*delta_t/2., np.eye(q)))

        Xbar = np.dot(chi, Wm.T)
        Xbar = np.reshape(Xbar, (n, 1))
        chi_diff = chi - np.dot(Xbar, np.ones((1, (2*n+1))))
        Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T)) + np.dot(Gamma, np.dot(Q, Gamma.T))
        
        print('')
        print('kk', kk)
        # print('Pbar', Pbar)
        # print('eig', np.linalg.eig(Pbar))
        print('det', np.linalg.det(Pbar))

        # Recompute sigma points to incorporate process noise
        sqP = np.linalg.cholesky(Pbar)
        Xrep = np.tile(Xbar, (1, n))
        chi_bar = np.concatenate((Xbar, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1) 
        chi_diff = chi_bar - np.dot(Xbar, np.ones((1, (2*n+1))))
        
        # Measurement Update: posterior state and covar at tk            
        # Retrieve measurement data
        Yk = Yk_list[kk]
        sensor_id = sensor_id_list[kk]
        
        # Computed measurements and covariance
        gamma_til_k, Rk = meas_fcn(tk, chi_bar, state_params, sensor_params, sensor_id)
        ybar = np.dot(gamma_til_k, Wm.T)
        ybar = np.reshape(ybar, (len(ybar), 1))
        Y_diff = gamma_til_k - np.dot(ybar, np.ones((1, (2*n+1))))
        Pyy = np.dot(Y_diff, np.dot(diagWc, Y_diff.T)) + Rk
        Pxy = np.dot(chi_diff,  np.dot(diagWc, Y_diff.T))
        
        print('Yk', Yk)
        print('ybar', ybar)
        
        # Kalman gain and measurement update
        Kk = np.dot(Pxy, np.linalg.inv(Pyy))
        Xk = Xbar + np.dot(Kk, Yk-ybar)
        
        # Basic covariance update
#        P = Pbar - np.dot(K, np.dot(Pyy, K.T))
        
        # Re-symmetric covariance     
#        P = 0.5 * (P + P.T)
        
        # Joseph form
        cholPbar = np.linalg.inv(np.linalg.cholesky(Pbar))
        invPbar = np.dot(cholPbar.T, cholPbar)
        P1 = (np.eye(n) - np.dot(np.dot(Kk, np.dot(Pyy, Kk.T)), invPbar))
        P2 = np.dot(Kk, np.dot(Rk, Kk.T))
        P = np.dot(P1, np.dot(Pbar, P1.T)) + P2

        # Recompute measurments using final state to get resids
        sqP = np.linalg.cholesky(P)
        Xrep = np.tile(Xk, (1, n))
        chi_k = np.concatenate((Xk, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
        gamma_til_post, dum = meas_fcn(tk, chi_k, state_params, sensor_params, sensor_id)
        ybar_post = np.dot(gamma_til_post, Wm.T)
        ybar_post = np.reshape(ybar_post, (len(ybar), 1))
        
        # Post-fit residuals and updated state
        resids = Yk - ybar_post
        
        # Store output
        filter_output[tk] = {}
        filter_output[tk]['X'] = Xk
        filter_output[tk]['P'] = P
        filter_output[tk]['resids'] = resids
        
        print('\n')
        print('tk', tk)
        print('Xbar', Xbar)
        print('Xk', Xk)
        print('Yk', Yk)
        print('ybar', ybar)
        print('ybar_post', ybar_post)
        print('Kk', Kk)
        print('resids', resids)
        print('Pbar', Pbar)
        print('P', P)
#        
        if kk > 2:
             mistake

            
    # TODO Generation of full_state_output not working correctly
    # Use filter_output for error analysis
    
    full_state_output = {}
            

    
    return filter_output, full_state_output


def ls_ukf_attitude(state_dict, truth_dict, meas_dict, meas_fcn, params_dict):    
    '''
    This function implements the Unscented Kalman Filter for the least
    squares cost function incorporate attitude state estimation.

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
    state_params : dictionary
        physical parameters and constants
    sensor_params : dictionary
        location, constraint, noise parameters of sensors
    int_params : dictionary
        numerical integration parameters

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
    Xk = state_dict[state_tk]['X']
    P = state_dict[state_tk]['P']
    Q = filter_params['Q']
    gap_seconds = filter_params['gap_seconds']
    time_format = int_params['time_format']

    n = len(Xk)
    q = int(Q.shape[0])
    
    # Prior information about the distribution
    pnorm = 2.
    kurt = math.gamma(5./pnorm)*math.gamma(1./pnorm)/(math.gamma(3./pnorm)**2.)
    beta = kurt - 1.
    kappa = kurt - float(n)
    
    # Compute sigma point weights
    alpha = filter_params['alpha']
    lam = alpha**2.*(n + kappa) - n
    gam = np.sqrt(n + lam)
    Wm = 1./(2.*(n + lam)) * np.ones(2*n,)
    Wc = Wm.copy()
    Wm = np.insert(Wm, 0, lam/(n + lam))
    Wc = np.insert(Wc, 0, lam/(n + lam) + (1 - alpha**2 + beta))
    diagWc = np.diag(Wc)

    # Initialize output
    filter_output = {}

    # Measurement times
    tk_list = meas_dict['tk_list']
    Yk_list = meas_dict['Yk_list']
    sensor_id_list = meas_dict['sensor_id_list']
    
    # Number of epochs
    N = len(tk_list)
  
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

        # Compute sigma points matrix
        sqP = np.linalg.cholesky(P)
        Xrep = np.tile(Xk, (1, n))
        chi = np.concatenate((Xk, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
        chi_v = np.reshape(chi, (n*(2*n+1), 1), order='F')
        
        # Propagate to next time
        if tk_prior == tk:
            intout = chi_v.T
        else:
            int0 = chi_v.flatten()
            tin = [tk_prior, tk]
            
            tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)

        # Extract values for later calculations
        chi_v = intout[-1,:]
        chi = np.reshape(chi_v, (n, 2*n+1), order='F')
       
        # State Noise Compensation
        # Zero out SNC for long time gaps
        if delta_t > gap_seconds:        
            Gamma = np.zeros((n,q))
        else:
            Gamma = np.zeros((n,q))
            Gamma[0:q,:] = (delta_t**2./2) * np.eye(q)
            Gamma[q:2*q,:] = delta_t * np.eye(q)
#            Gamma = delta_t * np.concatenate((np.eye(q)*delta_t/2., np.eye(q)))

        Xbar = np.dot(chi, Wm.T)
        Xbar = np.reshape(Xbar, (n, 1))
        chi_diff = chi - np.dot(Xbar, np.ones((1, (2*n+1))))
        Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T)) + np.dot(Gamma, np.dot(Q, Gamma.T))
        
        print('')
        print('kk', kk)
        print('Pbar', Pbar)
        print('eig', np.linalg.eig(Pbar))
        print('det', np.linalg.det(Pbar))

        # Recompute sigma points to incorporate process noise
        sqP = np.linalg.cholesky(Pbar)
        Xrep = np.tile(Xbar, (1, n))
        chi_bar = np.concatenate((Xbar, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1) 
        chi_diff = chi_bar - np.dot(Xbar, np.ones((1, (2*n+1))))
        
        # Measurement Update: posterior state and covar at tk            
        # Retrieve measurement data
        Yk = Yk_list[kk]
        sensor_id = sensor_id_list[kk]
        
        # Computed measurements and covariance
        gamma_til_k, Rk = meas_fcn(tk, chi_bar, state_params, sensor_params, sensor_id)
        ybar = np.dot(gamma_til_k, Wm.T)
        ybar = np.reshape(ybar, (len(ybar), 1))
        Y_diff = gamma_til_k - np.dot(ybar, np.ones((1, (2*n+1))))
        Pyy = np.dot(Y_diff, np.dot(diagWc, Y_diff.T)) + Rk
        Pxy = np.dot(chi_diff,  np.dot(diagWc, Y_diff.T))
        
        print('Yk', Yk)
        print('ybar', ybar)
        
        # Kalman gain and measurement update
        Kk = np.dot(Pxy, np.linalg.inv(Pyy))
        Xk = Xbar + np.dot(Kk, Yk-ybar)
        
        # Basic covariance update
#        P = Pbar - np.dot(K, np.dot(Pyy, K.T))
        
        # Re-symmetric covariance     
#        P = 0.5 * (P + P.T)
        
        # Joseph form
        cholPbar = np.linalg.inv(np.linalg.cholesky(Pbar))
        invPbar = np.dot(cholPbar.T, cholPbar)
        P1 = (np.eye(n) - np.dot(np.dot(Kk, np.dot(Pyy, Kk.T)), invPbar))
        P2 = np.dot(Kk, np.dot(Rk, Kk.T))
        P = np.dot(P1, np.dot(Pbar, P1.T)) + P2

        # Recompute measurments using final state to get resids
        sqP = np.linalg.cholesky(P)
        Xrep = np.tile(Xk, (1, n))
        chi_k = np.concatenate((Xk, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
        gamma_til_post, dum = meas_fcn(tk, chi_k, state_params, sensor_params, sensor_id)
        ybar_post = np.dot(gamma_til_post, Wm.T)
        ybar_post = np.reshape(ybar_post, (len(ybar), 1))
        
        # Post-fit residuals and updated state
        resids = Yk - ybar_post
        
        # Store output
        filter_output[tk] = {}
        filter_output[tk]['X'] = Xk
        filter_output[tk]['P'] = P
        filter_output[tk]['resids'] = resids
        
#        print('\n')
#        print('tk', tk)
#        print('Xbar', Xbar)
#        print('Xk', Xk)
#        print('Yk', Yk)
#        print('ybar', ybar)
#        print('ybar_post', ybar_post)
#        print('Kk', Kk)
#        print('resids', resids)
#        print('Pbar', Pbar)
#        print('P', P)
#        
#        if kk > 2:
#             mistake

            
    # TODO Generation of full_state_output not working correctly
    # Use filter_output for error analysis
    
    full_state_output = {}
            

    
    return filter_output, full_state_output


def ukf_6dof_predictor(tin, X, P, int_params, state_params, filter_params):
    
    # Retrieve parameters
    Q = filter_params['Q']
    sig_u = filter_params['sig_u']
    sig_v = filter_params['sig_v']
    Wm = filter_params['Wm']
    diagWc = filter_params['diagWc']
    gam = filter_params['gam']
    gap_seconds = filter_params['gap_seconds']
    time_format = int_params['time_format']
    
    # Initialize the GRP error sigma points
    sqP = np.linalg.cholesky(P)
    Xgrp = np.zeros((12,1))
    Xgrp[0:6] = X[0:6]
    Xgrp[9:12] = X[10:13]
    Xrep = np.tile(Xgrp, (1, 12))
    chi_grp = np.concatenate((Xgrp, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)

    # Form sigma point matrix for quaternions
    chi_quat = np.zeros((13, 25))
    chi_quat[:, 0] = X.flatten()
    chi_quat[0:6, 1:25] = chi_grp[0:6, 1:25]
    chi_quat[10:13, 1:25] = chi_grp[9:12, 1:25]
    
    # Compute the delta quaternion sigma points
    qmean = X[6:10].reshape(4,1)
    for jj in range(24):
        dp = chi_grp[6:9, jj+1].reshape(3,1)
        dq = att.grp2quat(dp, 1)
        qj = att.quat_composition(dq, qmean)
        chi_quat[6:10, jj+1] = qj.flatten()   
    
    # Vector for integration function
    chi_v = np.reshape(chi_quat, (13*25, 1), order='F')

    # Retrieve integration times
    tk_prior = tin[0]
    tk = tin[-1]
    if time_format == 'seconds':
        delta_t = tk - tk_prior
    elif time_format == 'JD':
        delta_t = (tk - tk_prior)*86400.
    elif time_format == 'datetime':
        delta_t = (tk - tk_prior).total_seconds()
    
    # Integrate chi
    if tk_prior == tk:
        intout = chi_v.T
    else:
        int0 = chi_v.flatten()        
        tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)

    # Extract values for later calculations
    chi_v = intout[-1,:]
    chi_bar = np.reshape(chi_v, (13, 25), order='F')
    
    # Form sigma point matrix for GRPs
    chi_grp = np.zeros((12, 25))
    chi_grp[0:6, :] = chi_bar[0:6, :]
    chi_grp[9:12, :] = chi_bar[9:12, :]
    
    # Compute the delta quaternion sigma points
    qmean = chi_bar[6:10, 0].reshape(4,1)
    qinv = att.quat_inverse(qmean)    
    for jj in range(1, 25):
        qj = chi_bar[6:10, jj].reshape(4,1)
        dq = att.quat_composition(qj, qinv)
        dp = att.quat2grp(dq, 1)
        chi_grp[6:9, jj] = dp.flatten()
    

    Xgrp = np.dot(chi_grp, Wm.T)
    Xgrp = np.reshape(Xgrp, (12, 1))
    chi_diff = chi_grp - np.dot(Xgrp, np.ones((1, 25)))
    
    if delta_t > 100.:
        Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T))
    else:
        print('\n Process Noise')
        Gamma1 = np.eye(3) * 0.5*delta_t**2.
        Gamma2 = np.eye(3) * delta_t
        Gamma = np.concatenate((Gamma1, Gamma2), axis=0)  
        
        Qatt1 = np.eye(3) * (sig_v**2 - (1./6.)*sig_u**2.*delta_t**2.)
        Qatt2 = np.eye(3) * sig_u**2.
        Qatt = np.zeros((6,6))
        Qatt[0:3, 0:3] = Qatt1
        Qatt[3:6, 3:6] = Qatt2
        Qatt *= delta_t/2.
        
        Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T))
        Pbar[0:6, 0:6] += np.dot(Gamma, np.dot(Q, Gamma.T))
        Pbar[6:12, 6:12] += Qatt

    # Re-symmetric pos def
    # Pbar = 0.5 * (Pbar + Pbar.T)
    
#    print(Pbar)
#    print(np.linalg.eig(Pbar))
#    
#    print(Xgrp)
#    print(qmean)
#    mistake
    
    
    return Xgrp, Pbar, qmean


def ukf_6dof_corrector(Xgrp, Pbar, qmean, Yi, ti, n, alpha, sun_gcrf,
                       sensor, EOP_data, XYs_df, spacecraftConfig, surfaces):
    
    
    #Compute Weights
    beta = 2.
    kappa = 3. - 12.
    lam = alpha**2 * (12. + kappa) - 12.
    gam = np.sqrt(12. + lam)

    Wm = 1./(2.*(12. + lam)) * np.ones((1,2*12))
    Wm = list(Wm.flatten())
    Wc = copy.copy(Wm)
    Wm.insert(0,lam/(12. + lam))
    Wc.insert(0,lam/(12. + lam) + (1 - alpha**2 + beta))
    Wm = np.asarray(Wm)
    diagWc = np.diag(Wc)  
    
    # Sensor parameters
    meas_types = sensor['meas_types']
    sigma_dict = sensor['sigma_dict']
    p = len(meas_types)
    
    # Measurement noise
    var = []
    for mt in meas_types:
        var.append(sigma_dict[mt]**2.)
    Rk = np.diag(var)
    
    
    # Recompute sigma points to incorporate process noise
    sqP = np.linalg.cholesky(Pbar)
    Xrep = np.tile(Xgrp, (1, 12))
    chi_grp = np.concatenate((Xgrp, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1) 
    chi_diff = chi_grp - np.dot(Xgrp, np.ones((1, 25)))
    
    # Compute sigma points with quaternions
    chi_quat = np.zeros((13, 25))
    chi_quat[0:6, :] = chi_grp[0:6, :]
    chi_quat[10:13, :] = chi_grp[9:12, :]
    chi_quat[6:10, 0] = qmean.flatten()
    for jj in range(24):
        dp = chi_grp[6:9, jj+1].reshape(3,1)
        dq = att.grp2quat(dp, 1)
        qj = att.quat_composition(dq, qmean)
        chi_quat[6:10, jj+1] = qj.flatten() 
    
    # Computed measurements    
    meas_bar = np.zeros((p, 25))
    for jj in range(chi_quat.shape[1]):
        Xj = chi_quat[:,jj]
        Yj = compute_measurement(Xj, sun_gcrf, sensor, spacecraftConfig,
                                 surfaces, ti, EOP_data,
                                 sensor['meas_types'], XYs_df)
        meas_bar[:,jj] = Yj.flatten()
        
#        print(jj)
#        print(Yj)
    
#    print(Wm)
    Ybar = np.dot(meas_bar, Wm.T)
    Ybar = np.reshape(Ybar, (p, 1))
    Y_diff = meas_bar - np.dot(Ybar, np.ones((1, 25)))
    Pyy = np.dot(Y_diff, np.dot(diagWc, Y_diff.T))
    Pxy = np.dot(chi_diff,  np.dot(diagWc, Y_diff.T))

    Pyy += Rk
    
#    print(meas_bar)
#    print(Yi)
#    print(Ybar)
#    print(Pyy)

    # Measurement Update
    K = np.dot(Pxy, np.linalg.inv(Pyy))
    Xgrp = Xgrp + np.dot(K, Yi-Ybar)
    
    # Compute updated quaternion and full state vector
    dp = Xgrp[6:9].reshape(3,1)
    dq = att.grp2quat(dp, 1)
    q = att.quat_composition(dq, qmean)
    X = np.zeros((13, 1))
    X[0:6] = Xgrp[0:6]
    X[6:10] = q
    X[10:13] = Xgrp[9:12]
    
#        # Regular update
#        P = Pbar - np.dot(K, np.dot(Pyy, K.T))
#
#        # Re-symmetric pos def
#        P = 0.5 * (P + P.T)
    
    # Joseph Form
    cholPbar = np.linalg.inv(np.linalg.cholesky(Pbar))
    invPbar = np.dot(cholPbar.T, cholPbar)
    P1 = (np.identity(12) - np.dot(np.dot(K, np.dot(Pyy, K.T)), invPbar))
    P = np.dot(P1, np.dot(Pbar, P1.T)) + np.dot(K, np.dot(Rk, K.T))
    
#    print('posterior')
#    print(X)
#    print(P)
#    print(Ybar)
#    print(Yi - Ybar)
#        print(Pyy)
#        print(Rk)
#        print(Pxy)
    

#    # Gaussian Likelihood
    beta = compute_gaussian(Yi, Ybar, Pyy)
#    beta_list.append(beta)
    
    return X, P, beta


###############################################################################
# Non-Gaussian Estimation
###############################################################################

def aegis_ukf(state_dict, truth_dict, meas_dict, meas_fcn, params_dict):    
    '''
    This function implements the Adaptive Entropy-based Gaussian Information
    Syntheis (AEGIS) Unscented Kalman Filter for the least
    squares cost function.

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
    state_params : dictionary
        physical parameters and constants
    sensor_params : dictionary
        location, constraint, noise parameters of sensors
    int_params : dictionary
        numerical integration parameters

    Returns
    ------
    filter_output : dictionary
        output state, covariance, and post-fit residuals at measurement times
    full_state_output : dictionary
        output state and covariance at all truth times
        
        
    References
    ------
    [1] DeMars, K.J., "Entropy-based Approach for Uncertainty Propagation of
        Nonlinear Dynamical Systems," JGCD 2013.
        
    '''
    
    # Break out inputs
    state_params = params_dict['state_params']
    filter_params = params_dict['filter_params']
    
    # State information
    state_tk = sorted(state_dict.keys())[-1]
    weights = state_dict[state_tk]['weights']
    means = state_dict[state_tk]['means']
    covars = state_dict[state_tk]['covars']
    GMM_dict = {}
    GMM_dict['weights'] = weights
    GMM_dict['means'] = means
    GMM_dict['covars'] = covars
    nstates = len(means[0])    
    
    # Prior information about the distribution
    pnorm = 2.
    kurt = math.gamma(5./pnorm)*math.gamma(1./pnorm)/(math.gamma(3./pnorm)**2.)
    beta = kurt - 1.
    kappa = kurt - float(nstates)

    # Compute sigma point weights
    alpha = filter_params['alpha']
    lam = alpha**2.*(nstates + kappa) - nstates
    gam = np.sqrt(nstates + lam)
    Wm = 1./(2.*(nstates + lam)) * np.ones(2*nstates,)
    Wc = Wm.copy()
    Wm = np.insert(Wm, 0, lam/(nstates + lam))
    Wc = np.insert(Wc, 0, lam/(nstates + lam) + (1 - alpha**2 + beta))
    diagWc = np.diag(Wc)
    filter_params['gam'] = gam
    filter_params['Wm'] = Wm
    filter_params['diagWc'] = diagWc

    # Initialize output
    filter_output = {}

    # Measurement times
    tk_list = meas_dict['tk_list']
    Yk_list = meas_dict['Yk_list']
    sensor_id_list = meas_dict['sensor_id_list']
    
    # Number of epochs
    N = len(tk_list)
  
    # Loop over times
    for kk in range(N):
    
        # Current and previous time
        if kk == 0:
            tk_prior = state_tk
        else:
            tk_prior = tk_list[kk-1]

        tk = tk_list[kk]

        # Predictor Step
        tin = [tk_prior, tk]
        GMM_bar = aegis_predictor(GMM_dict, tin, params_dict)
        
        # Corrector Step
        Yk = Yk_list[kk]
        sensor_id = sensor_id_list[kk]
        GMM_dict, resids_k = aegis_corrector(GMM_bar, tk, Yk, sensor_id,
                                             meas_fcn, params_dict)
        
        # Store output
        filter_output[tk] = {}
        filter_output[tk]['weights'] = GMM_dict['weights']
        filter_output[tk]['means'] = GMM_dict['means']
        filter_output[tk]['covars'] = GMM_dict['covars']
        filter_output[tk]['resids'] = resids_k
        
        
    # TODO Generation of full_state_output not working correctly
    # Use filter_output for error analysis
    full_state_output = {}
    
    return filter_output, full_state_output


def aegis_predictor(GMM_dict, tin, params_dict):
    '''
    
    
    '''
    
    # Break out inputs
    filter_params = params_dict['filter_params']
    state_params = params_dict['state_params']
    int_params = params_dict['int_params']
    
    # Copy input to ensure pass by value
    GMM_dict = copy.deepcopy(GMM_dict)
    filter_params = copy.deepcopy(filter_params)
    state_params = copy.deepcopy(state_params)
    int_params = copy.deepcopy(int_params)
    
    # Retrieve parameters
    Q = filter_params['Q']
    gam = filter_params['gam']
    Wm = filter_params['Wm']
    diagWc = filter_params['diagWc']    
    gap_seconds = filter_params['gap_seconds']        
    time_format = int_params['time_format']     
    
    # Fudge to work with general_dynamics
    state_params['split_T'] = filter_params['split_T']
    state_params['alpha'] = filter_params['alpha']
    
    q = int(Q.shape[0])
    
    tk_prior = tin[0]
    tk = tin[1]
    
    if time_format == 'seconds':
        delta_t = tk - tk_prior
    elif time_format == 'JD':
        delta_t = (tk - tk_prior)*86400.
    elif time_format == 'datetime':
        delta_t = (tk - tk_prior).total_seconds()
    
    # Check if propagation is needed
    if delta_t == 0.:
        return GMM_dict
    
    # Initialize for integrator
    # Retrieve current GMM
    weights = GMM_dict['weights']
    means = GMM_dict['means']
    covars = GMM_dict['covars']
    t0_list = [tk_prior]*len(weights)
    
    # For each GMM component, there should be 1 entropy, n states, and 2n+1
    # sigma points. 
#    ncomp = len(weights)
    nstates = len(means[0])
    npoints = nstates*2 + 1
    state_params['nstates'] = nstates
    state_params['npoints'] = npoints

    # Loop over components
    jj = 0
    while jj < len(weights):
        
#        print('\nstart loop')
#        print('jj', jj)
#        print('ncomp', len(weights))
#        print('t0', t0_list[jj])
        
        # Retrieve component values
        wj = weights[jj]
        mj = means[jj]
        Pj = covars[jj]
        ej = gaussian_entropy(Pj)
        tin = [t0_list[jj], tk]
            
        # Compute sigma points
        sqP = np.linalg.cholesky(Pj)
        Xrep = np.tile(mj, (1, nstates))
        chi = np.concatenate((mj, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
        chi_v = np.reshape(chi, (nstates*npoints, 1), order='F')
        
        # Setup initial state for integrator
        int0 = np.zeros(nstates*npoints+1,)
        int0[0] = ej
        int0[1:1+(nstates*npoints)] = chi_v.flatten()
        state_params['ncomp'] = 1
        
        # Integrate entropy and sigma point dynamics
        if tin[0] == tk:
            intout = np.reshape(int0, (1, len(int0)))
            split_flag = False
        else:
            tout, intout, split_flag = \
                dyn.general_dynamics(int0, tin, state_params, int_params)

        # Retrieve output state        
        chi_v = intout[-1, 1:1+(nstates*npoints)]
        chi = np.reshape(chi_v, (nstates, npoints), order='F')

        Xbar = np.dot(chi, Wm.T)
        Xbar = np.reshape(Xbar, (nstates, 1))
        chi_diff = chi - np.dot(Xbar, np.ones((1, npoints)))
        Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T))
        
        # Compute the current time
        if time_format == 'seconds':
            t = t0_list[jj] + tout[-1]
        elif time_format == 'JD':
            t = t0_list[jj] + tout[-1]/86400.
        elif time_format == 'datetime':
            t = t0_list[jj] + timedelta(seconds=tout[-1])
            
#        print('post integration')
#        print('t', t)
#        print('split_flag', split_flag)
        
            
        # Split if needed
        if split_flag:
            
            # Split the component
            GMM_in = {}
            GMM_in['weights'] = [wj]
            GMM_in['means'] = [Xbar]
            GMM_in['covars'] = [Pbar]
            GMM_out = split_GMM(GMM_in, N=3)
            w_split = GMM_out['weights']
            m_split = GMM_out['means']
            P_split = GMM_out['covars']
            
            # Replace current component and add others
            for comp in range(len(w_split)):

                # Compute weights, entropy and sigma points
                # Note: split_GMM function multiplies by wj, no need to 
                # repeat here
                w = w_split[comp]   
                m = m_split[comp]
                P = P_split[comp]
                
                if comp == 0:
                    weights[jj] = w
                    means[jj] = m
                    covars[jj] = P
                    t0_list[jj] = t
                else:
                    weights.append(w)
                    means.append(m)
                    covars.append(P)
                    t0_list.append(t)
        
        # If no split, update mean and covariance from propagator
        else:                
            means[jj] = Xbar
            covars[jj] = Pbar
            
        # If final time is reached, go to next component
        if t >= tk:
            jj += 1
            

    # After tk is reached, incorporate process noise
    # State Noise Compensation
    # Zero out SNC for long time gaps
    if delta_t < gap_seconds:        

        Gamma = np.zeros((nstates,q))
        Gamma[0:q,:] = (delta_t**2./2) * np.eye(q)
        Gamma[q:2*q,:] = delta_t * np.eye(q)
        
        for jj in range(len(weights)):
            covars[jj] += np.dot(Gamma, np.dot(Q, Gamma.T))
            
    # Form output
    GMM_bar = {}
    GMM_bar['weights'] = weights
    GMM_bar['means'] = means
    GMM_bar['covars'] = covars    
    
    return GMM_bar





def aegis_corrector(GMM_bar, tk, Yk, sensor_id, meas_fcn, params_dict):
    '''
    
    
    '''
    
    # Retrieve inputs
    filter_params = params_dict['filter_params']
    state_params = params_dict['state_params']
    sensor_params = params_dict['sensor_params']
    gam = filter_params['gam']
    Wm = filter_params['Wm']
    diagWc = filter_params['diagWc']
    
    # Break out GMM
    weights0 = GMM_bar['weights']
    means0 = GMM_bar['means']
    covars0 = GMM_bar['covars']
    nstates = len(means0[0])
    npoints = nstates*2 + 1
    
    # Loop over components and compute measurement update
    means = []
    covars = []
    beta_list = []
    for jj in range(len(weights0)):        
        
        mj = means0[jj]
        Pj = covars0[jj]
        
        # Compute sigma points
        sqP = np.linalg.cholesky(Pj)
        Xrep = np.tile(mj, (1, nstates))
        chi = np.concatenate((mj, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
        chi_diff = chi - np.dot(mj, np.ones((1, npoints)))

        # Computed measurements and covariance
        gamma_til_k, Rk = meas_fcn(tk, chi, state_params, sensor_params, sensor_id)
        ybar = np.dot(gamma_til_k, Wm.T)
        ybar = np.reshape(ybar, (len(ybar), 1))
        Y_diff = gamma_til_k - np.dot(ybar, np.ones((1, (2*nstates+1))))
        Pyy = np.dot(Y_diff, np.dot(diagWc, Y_diff.T)) + Rk
        Pxy = np.dot(chi_diff,  np.dot(diagWc, Y_diff.T))
        
        print('Yk', Yk)
        print('ybar', ybar)
        
        # Kalman gain and measurement update
        Kk = np.dot(Pxy, np.linalg.inv(Pyy))
        mf = mj + np.dot(Kk, Yk-ybar)
        
        # Joseph form
        cholPbar = np.linalg.inv(np.linalg.cholesky(Pj))
        invPbar = np.dot(cholPbar.T, cholPbar)
        P1 = (np.eye(nstates) - np.dot(np.dot(Kk, np.dot(Pyy, Kk.T)), invPbar))
        P2 = np.dot(Kk, np.dot(Rk, Kk.T))
        Pf = np.dot(P1, np.dot(Pj, P1.T)) + P2
        
        # Compute Gaussian likelihood
        beta_j = gaussian_likelihood(Yk, ybar, Pyy)

        # Store output
        means.append(mf)
        covars.append(Pf)
        beta_list.append(beta_j)
        
    # Normalize updated weights
#    denom = np.dot(beta_list, weights0)    
#    weights = [a1*a2/denom for a1,a2 in zip(weights0, beta_list)]
    weights = list(np.multiply(beta_list, weights0)/np.dot(beta_list, weights0))
    
    # Merge and Prune components
    GMM_in = {}
    GMM_in['weights'] = weights
    GMM_in['means'] = means
    GMM_in['covars'] = covars
    GMM_dict = merge_GMM(GMM_in, filter_params)
    
    # Compute post-fit residuals by merging all components
    params = {}
    params['prune_T'] = 0.
    params['merge_U'] = 1e10
    GMM_resids = merge_GMM(GMM_dict, params)
    Xhat = GMM_resids['means'][0]
    
    ybar = mfunc.compute_measurement(Xhat, state_params, sensor_params,
                                     sensor_id, tk)
    resids_k = Yk - ybar
    
    return GMM_dict, resids_k


def split_GMM(GMM0, N=3):
    '''
    This function splits a single gaussian PDF into multiple components.
    For a multivariate PDF, it will split along the axis corresponding to the
    largest eigenvalue (greatest uncertainty). The function splits along only
    one axis.

    Parameters
    ------

    '''

    # Break out GMM
    w0 = GMM0['weights'][0]
    m0 = GMM0['means'][0]
    P0 = GMM0['covars'][0]
    n = len(m0)

    # Get splitting library info
    wbar, mbar, sigbar = split_gaussian_library(N)

    # Decompose covariance matrix
    lam, V = np.linalg.eig(P0)

    # Find largest eigenvalue and corresponding eigenvector
    kk = np.argmax(lam)
    lam0_k = lam[kk]
    vk = V[:,kk].reshape(n, 1)

    # Compute updated weights
    w = [w0 * wi for wi in wbar]

    # All sigma values are equal, just use first entry
    lam[kk] *= sigbar[0]**2
    Lam = np.diag(lam)

    # Compute updated means, covars
    m = [m0 + np.sqrt(lam0_k)*mbar_ii*vk for mbar_ii in mbar]
    P = [np.dot(V, np.dot(Lam, V.T))]*N
    
    # Form output
    GMM = {}
    GMM['weights'] = w
    GMM['means'] = m
    GMM['covars'] = P

    return GMM


def merge_GMM(GMM0, params) :
    '''    
    This function examines a GMM containing multiple components. It removes
    components with weights below a given threshold, and merges components that
    are close together (small NL2 distance).
    
    Parameters


    References
    ------
    [1] Vo and Ma, "The Gaussian Mixture Probability Hypothesis Density
        Filter," 2006.
    
    '''
    
    # Break out GMM
    w0 = GMM0['weights']
    m0 = GMM0['means']
    P0 = GMM0['covars']

    # Break out inputs
    T = params['prune_T']
    U = params['merge_U']
    
    # Number of states
    nstates = len(m0[0])

    # Only keep GM components whose weight is above the threshold   
    # This applies DeMars threshold instead of Vo which just uses T
    wmax = max(w0)
    w = [w0[ii] for ii in range(len(w0)) if w0[ii] > T*wmax]
    m = [m0[ii] for ii in range(len(w0)) if w0[ii] > T*wmax]
    P = [P0[ii] for ii in range(len(w0)) if w0[ii] > T*wmax]
    
    # Normalize weights
    w = list(np.asarray(w)/sum(w)*sum(w0))

    # Loop to merge components that are close
    wf = []
    mf = []
    Pf = []
    I = np.arange(0, len(w))
    while len(I) != 0:

        # Loop over components to see if they are close to j
        # Note, at least one will be when i == j        
        L = []
        wsum = 0.
        msum = np.zeros((nstates, 1))
        jj = np.argmax(w)
        for ii in range(len(w)):
            Pi = P[ii]
            invP = np.linalg.inv(Pi)
            prod = np.dot((m[ii] - m[jj]).T, np.dot(invP,(m[ii] - m[jj])))
            if prod <= U:
                L.append(ii)
                wsum += w[ii]
                msum += w[ii]*m[ii]                

        # Compute final w,m,P
        wf.append(wsum)
        mf_bar = (1./wsum)*msum
        mf.append(mf_bar)

        Psum = np.zeros((nstates, nstates))
        for ii in range(len(L)):
            Psum += w[L[ii]]*(P[L[ii]] + np.dot((mf_bar - m[L[ii]]),
                                                (mf_bar - m[L[ii]]).T))
        Pf_bar = (1./wsum)*Psum
        Pf.append(Pf_bar)

        # Reduce w,m,P
        I = list(set(I).difference(set(L)))
        w = [w[ii] for ii in I]
        m = [m[ii] for ii in I]
        P = [P[ii] for ii in I]

        # Reset I        
        I = np.arange(0, len(w))

    # Normalize weights
    wf = list(np.asarray(wf)/sum(wf)*sum(w0))
    
    # Output
    GMM = {}
    GMM['weights'] = wf
    GMM['means'] = mf
    GMM['covars'] = Pf

    return GMM


def split_gaussian_library(N=3):
    '''
    This function outputs the splitting library for GM components. All outputs
    are given to split a univariate standard normal distribution (m=0,sig=1).

    Parameters
    ------
    N : int (optional)
        number of components to split into (3, 4, or 5) (default = 3)

    Returns
    ------
    w : list
        component weights
    m : list
        component means (univariate)
    sig : list 
        component sigmas (univariate)

    '''

    if N == 3:
        w = [0.2252246249136750, 0.5495507501726501, 0.2252246249136750]
        m = [-1.057515461475881, 0., 1.057515461475881]
        sig = [0.6715662886640760]*3

    elif N == 4:
        w = [0.1238046161618835, 0.3761953838381165, 0.3761953838381165,
             0.1238046161618835]
        m = [-1.437464136328835, -0.455886223973523, 0.455886223973523,
             1.437464136328835]
        sig = [0.5276007226175397]*4

    elif N == 5:
        w = [0.0763216490701042, 0.2474417859474436, 0.3524731299649044,
             0.2474417859474436, 0.0763216490701042]
        m = [-1.689972911128078, -0.800928383429953, 0., 0.800928383429953,
             1.689972911128078]
        sig = [0.4422555386310084]*5

    return w, m, sig


def gaussian_entropy(P) :
    '''
    This function computes the entropy of a Gaussian PDF given the covariance.
    
    Parameters
    ------
    P : nxn numpy array
        covariance matrix
    
    Returns
    ------
    H : float
        differential entropy
        
    Reference
    ------
    DeMars, K.J., Bishop, R.H., Jah, M.K., "Entropy-Based Approach for 
        Uncertainty Propagation of Nonlinear Dynamical Systems," JGCD 2013.
    '''

    if np.linalg.det(2*math.pi*math.e*P) < 0. :
        print(np.linalg.det(2*math.pi*math.e*P))
        print(np.linalg.eig(P))
        P2 = scipy.linalg.sqrtm(P)
        P3 = np.real(np.dot(P2,P2.T))
        print(np.linalg.eig(P3))
        print(P3 - P)
        mistake

    # Differential Entropy (Eq. 5)
    H = 0.5 * math.log(np.linalg.det(2.*math.pi*math.e*P))
    
#    print(np.linalg.det(2.*math.pi*math.e*P))

    # Renyi Entropy (Eq. 8)
    # kappa = 0.5
    # R = 0.5 * log(np.linalg.det(2*pi*(kappa**(1/(1-kappa)))*P))

    entropy = H

    return entropy


###############################################################################
# General Utilities
###############################################################################


def cholesky_inv(P):
    
    cholP_inv = np.linalg.inv(np.linalg.cholesky(P))
    Pinv = np.dot(cholP_inv.T, cholP_inv)
    
    return Pinv


def gaussian_likelihood(x, m, P):
    '''
    This function computes the likelihood of the multivariate gaussian pdf
    for a random vector x, assuming mean m and covariance P.  

    Parameters
    ------
    x : nx1 numpy array
        instance of a random vector
    m : nx1 numpy array
        mean
    P : nxn numpy array
        covariance

    Returns
    ------
    pg : float
        multivariate gaussian likelihood   

    '''

    K1 = np.sqrt(np.linalg.det(2*math.pi*(P)))
    K2 = np.exp(-0.5 * np.dot((x-m).T, np.dot(np.linalg.inv(P),(x-m))))
    pg = float((1./K1) * K2)

    return pg


def gmm_samples(GMM_dict, N):

    # Break out GMM
    w = GMM_dict['weights']
    m = GMM_dict['means']
    P = GMM_dict['covars']

    # Loop over components and generate samples
    for jj in range(len(w)):
        wj = w[jj]
        mj = m[jj]
        Pj = P[jj]

        mcj = np.random.multivariate_normal(mj.flatten(),Pj,int(wj*N))

        if jj == 0 :
            mc_points = mcj
        else :
            mc_points = np.concatenate((mc_points,mcj))

    return mc_points


def unscented_transform(m1, P1, transform_fcn, inputs, alpha=1., pnorm=2.):
    '''
    This function computes the unscented transform for a p-norm
    distribution and user defined transform function.

    Parameters
    ------
    m1 : nx1 numpy array
      mean state vector
    P1 : nxn numpy array
      covariance matrix
    transform_fcn : function handle
      name of transform function
    inputs : dictionary
      input parameters for transform function
    alpha : float, optional
      sigma point distribution parameter (default=1)
    pnorm : float, optional
      value of p-norm distribution (default=2)

    Returns
    ------
    m2 : mx1 numpy array
      transformed mean state vector
    P2 : mxm numpy array
      transformed covariance matrix
    Pcross : nxm numpy array
      cross correlation covariance matrix
    '''

    # Number of States
    L = len(m1)

    # Prior information about the distribution
    kurt = math.gamma(5./pnorm)*math.gamma(1./pnorm)/(math.gamma(3./pnorm)**2.)
    beta = kurt - 1.
    kappa = kurt - float(L)

    # Compute sigma point weights
    lam = alpha**2.*(L + kappa) - L
    gam = np.sqrt(L + lam)
    Wm = 1./(2.*(L + lam)) * np.ones((1, 2*L))
    Wm = list(Wm.flatten())
    Wc = copy.copy(Wm)
    Wm.insert(0, lam/(L + lam))
    Wc.insert(0, lam/(L + lam) + (1 - alpha**2 + beta))
    Wm = np.asarray(Wm)
    diagWc = np.diag(Wc)

    # Compute chi - baseline sigma points
    sqP = np.linalg.cholesky(P1)
    Xrep = np.tile(m1, (1, L))
    chi = np.concatenate((m1, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
    chi_diff = chi - np.dot(m1, np.ones((1, (2*L+1))))
    
    # Compute transformed sigma points
    Y = transform_fcn(chi, inputs)
    row2 = int(Y.shape[0])
    col2 = int(Y.shape[1])

    # Compute mean and covar
    m2 = np.dot(Y, Wm.T)
    m2 = np.reshape(m2, (row2, 1))
    Y_diff = Y - np.dot(m2, np.ones((1, col2)))
    P2 = np.dot(Y_diff, np.dot(diagWc, Y_diff.T))
    Pcross = np.dot(chi_diff,  np.dot(diagWc, Y_diff.T))

    return m2, P2, Pcross


def unscented_kep2cart(chi, inputs):
    '''
    Function for use with unscented_transform.
    Converts sigma point matrix from keplerian elements to inertial cartesian
    coordinates.

    Parameters
    ------
    chi : L x (2L+1) numpy array
      sigma point matrix
    inputs : dictionary
      input parameters

    Returns
    ------
    Y : m x (2L+1) numpy array
      transformed sigma point matrix
    '''

    # Size of input/output
    L = int(chi.shape[1])
    Y = np.zeros((6, L))

    for jj in range(L):

        # Pull out column of chi
        elem = chi[:,jj]

        # Convert to ECI
        Xeci = astro.kep2cart(elem)
        Y[:,jj] = Xeci.flatten()

    return Y


def unscented_ric2eci(chi, inputs):
    '''
    Function for use with unscented_transform.
    Converts sigma point matrix from RIC coordinates to ECI.

    Parameters
    ------
    chi : L x (2L+1) numpy array
      sigma point matrix
    inputs : dictionary
      input parameters

    Returns
    ------
    Y : m x (2L+1) numpy array
      transformed sigma point matrix
    '''
    
    # TODO: Some discrepancies exist in how to transform 6x6 covariance beteen
    # RIC and ECI, in particular the velocity components. The NASA CARA code
    # matches a description in Tapley, Schutz, Born Section 4.16.1, which uses
    # a simple linear transform and zeros for the pos-vel cross-correlations.
    # This seems like an oversimplification. Schaub and Junkins Section 14.3
    # and Example 14.1 include a cross(omega, r) term in the transformation
    # of velocity between ECI and RIC, and use of an unscented transform or
    # Monte Carlo samples to transform the 6x6 covariance will naturally turn
    # out different from the TSB formulation.

    # Size of input/output
    L = int(chi.shape[1])
    Y = np.zeros((6, L))
    
    # Input parameters
    rc_vect = inputs['rc_vect']
    vc_vect = inputs['vc_vect']

    for jj in range(L):

        # Pull out column of chi
        Xrel_ric = chi[:,jj]
        rho_ric = Xrel_ric[0:3].reshape(3,1)
        drho_ric = Xrel_ric[3:6].reshape(3,1)

        # Convert to ECI
        rho_eci = coord.ric2eci(rc_vect, vc_vect, rho_ric)
        drho_eci = coord.ric2eci_vel(rc_vect, vc_vect, rho_ric, drho_ric)        
        Xrel_eci = np.concatenate((rho_eci, drho_eci), axis=0)
        
        Y[:,jj] = Xrel_eci.flatten()
    
    
    return Y



#def lp_batch(state_dict, meas_dict, inputs, intfcn, meas_fcn, pnorm=2.):
#    '''
#    This function implements the linearized batch estimator for a minimum
#    p-norm distribution.
#
#    Parameters
#    ------
#    state_dict : dictionary
#        initial state and covariance for filter execution
#    meas_dict : dictionary
#        measurement data over time for the filter and parameters (noise, etc)
#    inputs : dictionary
#        input parameters
#    intfcn : function handle
#        function for dynamics model
#    meas_fcn : function handle
#        function for measurements
#    pnorm : float, optional
#        p-norm distribution parameter (default=2.)
#
#    Returns
#    ------
#    filter_output : dictionary
#        output state, covariance, and post-fit residuals over time
#    '''
#
#    # State information
#    state_ti = sorted(state_dict.keys())[-1]
#    Xo_ref = state_dict[state_ti]['X']
#    Po_bar = state_dict[state_ti]['P']
#
#    # Measurement information
#    meas_types = meas_dict['meas_types']
#    sigma_dict = meas_dict['sigma_dict']
#    p = len(meas_types)
#    Rk = np.zeros((p, p))
#    for ii in xrange(p):
#        mtype = meas_types[ii]
#        sig = sigma_dict[mtype]
#        Rk[ii,ii] = sig**2.   
#
#    # Rescale noise for pnorm distribution
#    scale = (gamma(3./pnorm)/gamma(1./pnorm)) * pnorm**(2./pnorm)
#    Rk = scale*Rk
#
#    # Setup
#    cholPo = np.linalg.inv(np.linalg.cholesky(Po_bar))
#    invPo_bar = np.dot(cholPo.T, cholPo)
#    cholRk = np.linalg.inv(np.linalg.cholesky(Rk))
#    invRk = np.dot(cholRk.T, cholRk)
#    #invRk_pover2 = np.diag(np.diag(Rk)**(pnorm/2.))
#    invRk_pover2 = invRk
#    n = len(Xo_ref)
#
#    # Integrator tolerance
#    int_tol = inputs['int_tol']
#
#    # Initialize output
#    filter_output = {}
#
#    # Measurement times
#    ti_list = sorted(meas_dict['meas'].keys())
#
#    # Number of epochs
#    L = len(ti_list)
#
#    # Initialize
#    maxiters = 20
#    newt_maxiters = 100
#    newt_conv = 1e-10
#    xo_bar = np.zeros((n, 1))
#    xo_hat = np.zeros((n, 1))
#    phi0 = np.identity(n)
#    phi0_v = np.reshape(phi0, (n**2, 1))
#
#    # Begin Loop
#    iters = 0
#    xo_hat_mag = 1
#    conv_crit = 1e-5
#    while xo_hat_mag > conv_crit:
#
#        # Increment loop counter and exit if necessary
#        iters += 1
#        if iters > maxiters:
#            iters -= 1
#            print 'Solution did not converge in ', iters, ' iterations'
#            print 'Last xo_hat magnitude: ', xo_hat_mag
#            break
#
#        # Initialze values for this iteration
#        Xref_list = []
#        phi_list = []
#        resids_list = []
#        Hi_list = []
#        phi_v = phi0_v.copy()
#        Xref = Xo_ref.copy()
#
#        # Loop over times
#        for ii in xrange(L):
#            if ii == 0:
#                ti_prior = copy.copy(state_ti)
#            else:
#                ti_prior = ti_list[ii-1]
#
#            ti = ti_list[ii]
#
#            # Read the next observation
#            Yi = meas_dict['meas'][ti]
#
#            # If Rk is different at each time epoch, include it here
#
#            # Initialize
#            Xref_prior = Xref.copy()
#
#            # Initial Conditions for Integration Routine
#            int0 = np.concatenate((Xref_prior, phi_v))
#
#            # Integrate Xref and STM
#            if ti_prior == ti:
#                intout = int0.T
#            else:
#                int0 = int0.flatten()
#                tin = [ti_prior, ti]
#                intout = odeint(intfcn, int0, tin, args=(inputs,),
#                                rtol=int_tol, atol=int_tol)
#
#            # Extract values for later calculations
#            xout = intout[-1,:]
#            Xref1 = xout[0:n]
#            Xref = np.reshape(Xref1, (n, 1))
#            phi_v = np.reshape(xout[n:], (n**2, 1))
#            phi = np.reshape(phi_v, (n, n))
#            phi_list.append(phi)
#
#            # Compute expected measurement and linearized observation
#            # sensitivity matrix
#            Hi_til, Gi = meas_fcn(Xref, inputs)
#            yi = Yi - Gi
#            Hi = np.dot(Hi_til, phi)
#            Hi_list.append(Hi)
#
#            # Save output
#            resids_list.append(yi)
#            Xref_list.append(Xref)
#
#        # Solve the Normal Equations
#        # Newton Raphson iteration to get best xo_hat
#        diff_mag = 1
#        xo_bar_newt = xo_bar.copy()
#
#        newt_iters = 0
#        while diff_mag > newt_conv:
#
#            newt_iters += 1
#            if newt_iters > newt_maxiters:
#                print 'difference magnitude:', diff_mag
#                print 'newton iteration #', newt_iters
#                break
#
#            Lambda = invPo_bar.copy()
#            N = np.dot(Lambda, xo_bar_newt)
#
#            # Loop over times
#            for ii in xrange(L):
#
#                # Retrieve values
#                yi = resids_list[ii]
#                Hi = Hi_list[ii]
#
#                # Compute weighting matrix
#                W_vect = abs(yi - np.dot(Hi, xo_hat))**(pnorm-2.)
#                W = np.diag(W_vect.flatten())
#
#                # Accumulate quantities of interest
#                Lambda += (pnorm-1.)*\
#                    np.dot(Hi.T, np.dot(W, np.dot(invRk_pover2, Hi)))
#                abs_vect = np.multiply(abs(yi-np.dot(Hi, xo_hat))**(pnorm-1.),
#                                       np.sign(yi-np.dot(Hi, xo_hat)))
#                N += np.dot(Hi.T, np.dot(invRk_pover2, abs_vect))
#
#            # Solve the normal equations
#            cholLam_inv = np.linalg.inv(np.linalg.cholesky(Lambda))
#            Po = np.dot(cholLam_inv.T, cholLam_inv)
#
#            if pnorm > 2.:
#                alpha = 1
#            else:
#                alpha = pnorm - 1.
#
#            xo_hat += alpha * np.dot(Po, N)
#            xo_bar_newt = xo_bar - xo_hat
#            xo_hat_mag = np.linalg.norm(xo_hat)
#            diff_mag = alpha * np.linalg.norm(np.dot(Po, N))
#
#        # Update for next batch iteration
#        Xo_ref = Xo_ref + xo_hat
#        xo_bar = xo_bar - xo_hat
#
#        print 'Iteration Number: ', iters
#        print 'xo_hat_mag = ', xo_hat_mag
#
#    # Form output
#    for ii in xrange(L):
#        ti = ti_list[ii]
#        X = Xref_list[ii]
#        resids = resids_list[ii]
#        phi = phi_list[ii]
#        P = np.dot(phi, np.dot(Po, phi.T))
#
#        filter_output[ti] = {}
#        filter_output[ti]['X'] = copy.copy(X)
#        filter_output[ti]['P'] = copy.copy(P)
#        filter_output[ti]['resids'] = copy.copy(resids)
#
#    return filter_output

#
#
#
#def unscented_batch(state_dict, meas_dict, inputs, intfcn, meas_fcn, alpha=1.,
#                    pnorm=2.):
#    '''
#    This function implements the unscented batch estimator for a minimum
#    p-norm distribution.
#
#    Parameters
#    ------
#    state_dict : dictionary
#        initial state and covariance for filter execution
#    meas_dict : dictionary
#        measurement data over time for the filter and parameters (noise, etc)
#    inputs : dictionary
#        input parameters
#    intfcn : function handle
#        function for dynamics model
#    meas_fcn : function handle
#        function for measurements
#    alpha : float, optional
#        sigma point distribution parameter (default=1.)
#    pnorm : float, optional
#        p-norm distribution parameter (default=2.)
#
#    Returns
#    ------
#    filter_output : dictionary
#        output state, covariance, and post-fit residuals over time
#    '''
#
#    # State information
#    state_ti = sorted(state_dict.keys())[-1]
#    Xo = state_dict[state_ti]['X']
#    Po = state_dict[state_ti]['P']
#
#    # Measurement information
#    meas_types = meas_dict['meas_types']
#    sigma_dict = meas_dict['sigma_dict']
#    p = len(meas_types)
#    Rk = np.zeros((p, p))
#    for ii in xrange(p):
#        mtype = meas_types[ii]
#        sig = sigma_dict[mtype]
#        Rk[ii,ii] = sig**2.   
#
#    # Rescale noise for pnorm distribution
#    scale = (gamma(3./pnorm)/gamma(1./pnorm)) * pnorm**(2./pnorm)
#    Rk = scale*Rk
#
#    # Setup
#    # cholPo = np.linalg.inv(np.linalg.cholesky(Po_bar))
#    # invPo_bar = np.dot(cholPo.T, cholPo)
#    # cholRk = np.linalg.inv(np.linalg.cholesky(Rk))
#    # invRk = np.dot(cholRk.T, cholRk)
#    #invRk_pover2 = np.diag(np.diag(Rk)**(pnorm/2.))
#    # invRk_pover2 = invRk
#    L = len(Xo)
#    
#    # Prior information about the distribution
#    kurt = gamma(5./pnorm)*gamma(1./pnorm)/(gamma(3./pnorm)**2.)
#    beta = kurt - 1.
#    kappa = kurt - float(L)
#    
#    print 'pnorm',pnorm
#    print 'kappa',kappa
#    print 'beta',beta
#
#    # Compute sigma point weights
#    lam = alpha**2.*(L + kappa) - L
#    gam = np.sqrt(L + lam)
#    Wm = 1./(2.*(L + lam)) * np.ones((1, 2*L))
#    Wm = list(Wm.flatten())
#    Wc = copy.copy(Wm)
#    Wm.insert(0, lam/(L + lam))
#    Wc.insert(0, lam/(L + lam) + (1 - alpha**2 + beta))
#    Wm = np.asarray(Wm)
#    diagWc = np.diag(Wc)
#    
#
#    # Integrator tolerance
#    int_tol = inputs['int_tol']
#
#    # Initialize output
#    filter_output = {}
#
#    # Measurement times
#    ti_list = sorted(meas_dict['meas'].keys())
#    N = len(ti_list)
#    
#    # Block diagonal Rk matrix
#    Rk_full = np.kron(np.eye(N), Rk)
#
#    # Initialize
#    maxiters = 10   
#    X = Xo.copy()
#    P = Po.copy()
#    
#    # Begin Loop
#    iters = 0
#    diff = 1
#    conv_crit = 1e-5
#    while diff > conv_crit:
#
#        # Increment loop counter and exit if necessary
#        iters += 1
#        if iters > maxiters:
#            iters -= 1
#            print 'Solution did not converge in ', iters, ' iterations'
#            print 'Last xo_hat magnitude: ', xo_hat_mag
#            break
#
#        # Reset P every iteration???
#        # P = Po.copy()
#        
#        # Compute Sigma Points
#        sqP = np.linalg.cholesky(P)
#        Xrep = np.tile(X, (1, L))
#        chi0 = np.concatenate((X, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
#        chi_v = np.reshape(chi0, (L*(2*L+1), 1), order='F')  
#        chi_diff0 = chi0 - np.dot(X, np.ones((1, 2*L+1)))
#
#        # Loop over times
#        meas_ind = 0
#        Y_bar = np.zeros((2*N, 1))
#        Y_til = np.zeros((2*N, 1))
#        gamma_til_mat = np.zeros((2*N, 2*L+1)) 
#        for ii in xrange(len(ti_list)):
#            if ii == 0:
#                ti_prior = copy.copy(state_ti)
#            else:
#                ti_prior = ti_list[ii-1]
#
#            ti = ti_list[ii]
#
#            # Read the next observation
#            Yi = meas_dict['meas'][ti]
#
#            # If Rk is different at each time epoch, include it here
#
#            # Integrate chi
#            if ti_prior == ti:
#                intout = chi_v.T
#            else:
#                int0 = chi_v.flatten()
#                tin = [ti_prior, ti]
#                intout = odeint(intfcn, int0, tin, args=(inputs,), rtol=int_tol,
#                                atol=int_tol)
#
#            # Extract values for later calculations
#            chi_v = intout[-1,:]
#            chi = np.reshape(chi_v, (L, 2*L+1), order='F')
#            
#            # Compute measurement for each sigma point
#            gamma_til_k = meas_fcn(chi, inputs)
#            ybar = np.dot(gamma_til_k, Wm.T)
#            ybar = np.reshape(ybar, (p,1))
#            
#            # Accumulate measurements and computed measurements
#            Y_til[meas_ind:meas_ind+p] = Yi
#            Y_bar[meas_ind:meas_ind+p] = ybar
#            gamma_til_mat[meas_ind:meas_ind+p, :] = gamma_til_k  
#            
#            # Increment measurement index
#            meas_ind += p
#
#        # Compute covariances
#        Y_diff = gamma_til_mat - np.dot(Y_bar, np.ones((1, 2*L+1)))
#        Pyy = np.dot(Y_diff, np.dot(diagWc, Y_diff.T)) + Rk_full
#        Pxy = np.dot(chi_diff0, np.dot(diagWc, Y_diff.T))        
#
#        # Compute Kalman Gain
#        K = np.dot(Pxy, np.linalg.inv(Pyy))
#
#        # Compute updated state and covariance    
#        X += np.dot(K, Y_til-Y_bar)
#        P = P - np.dot(K, np.dot(Pyy, K.T))
#        diff = np.linalg.norm(np.dot(K, Y_til-Y_bar))
#        
#        print 'Iteration Number: ', iters
#        print 'diff = ', diff
#        
#    
#    # Compute final output    
#    # Compute Sigma Points
#    sqP = np.linalg.cholesky(P)
#    Xrep = np.tile(X, (1, L))
#    chi0 = np.concatenate((X, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
#    chi_v = np.reshape(chi0, (L*(2*L+1), 1), order='F')  
#    chi_diff0 = chi0 - np.dot(X, np.ones((1, 2*L+1)))
#    
#    # Loop over times 
#    meas_ind = 0
#    for ii in xrange(len(ti_list)):
#        if ii == 0:
#            ti_prior = copy.copy(state_ti)
#        else:
#            ti_prior = ti_list[ii-1]
#
#        ti = ti_list[ii]
#        
#        # Integrate chi
#        if ti_prior == ti:
#            intout = chi_v.T
#        else:
#            int0 = chi_v.flatten()
#            tin = [ti_prior, ti]
#            intout = odeint(intfcn, int0, tin, args=(inputs,), rtol=int_tol,
#                            atol=int_tol)
#        
#        # Extract values for later calculations
#        chi_v = intout[-1,:]
#        chi = np.reshape(chi_v, (L, 2*L+1), order='F')  
#                
#        # Save data for this time step
#        Xbar = np.dot(chi, Wm.T)
#        Xbar = np.reshape(Xbar, (L, 1))
#        chi_diff = chi - np.dot(Xbar, np.ones((1, (2*L+1))))
#        Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T))
#        Pbar = 0.5 * (Pbar + Pbar.T)
#        
#        filter_output[ti] = {}
#        filter_output[ti]['X'] = Xbar.copy()
#        filter_output[ti]['P'] = Pbar.copy()
#        filter_output[ti]['resids'] = Y_til[meas_ind:meas_ind+p] -\
#                                      Y_bar[meas_ind:meas_ind+p]
#                                      
#        meas_ind += p
#
#    return filter_output







