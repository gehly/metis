import numpy as np
import scipy
import math
import sys
import os
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

import dynamics.dynamics_functions as dyn



###############################################################################
# Batch Estimation
###############################################################################

def ls_batch(state_dict, truth_dict, meas_dict, meas_fcn, state_params,
             sensor_params, int_params):
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

    # State information
    state_tk = sorted(state_dict.keys())[-1]
    Xo_ref = state_dict[state_tk]['X']
    Po_bar = state_dict[state_tk]['P']

    # Setup
    cholPo = np.linalg.inv(np.linalg.cholesky(Po_bar))
    invPo_bar = np.dot(cholPo.T, cholPo)

    n = len(Xo_ref)

    # Initialize output
    filter_output = {}

    # Measurement times
    tk_list = meas_dict['tk_list']
    Yk_list = meas_dict['Yk_list']
    sensor_id_list = meas_dict['sensor_id_list']

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
    conv_crit = 1e-5
    while xo_hat_mag > conv_crit:

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

        print('Iteration Number: ', iters)
        print('xo_hat_mag = ', xo_hat_mag)

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


def unscented_batch(state_dict, truth_dict, meas_dict, meas_fcn, state_params,
                    sensor_params, int_params):
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
    alpha = state_params['alpha']
    lam = alpha**2.*(n + kappa) - n
    gam = np.sqrt(n + lam)
    Wm = 1./(2.*(n + lam)) * np.ones(2*n,)
    Wc = Wm.copy()
    Wm = np.insert(Wm, 0, lam/(n + lam))
    Wc = np.insert(Wc, 0, lam/(n + lam) + (1 - alpha**2 + beta))
    diagWc = np.diag(Wc)
    unscented_params = {}
    unscented_params['gam'] = gam
    unscented_params['Wm'] = Wm
    unscented_params['diagWc'] = diagWc

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
    diff = 1
    conv_crit = 1e-5
    
    # Begin loop
    iters = 0
    while diff > conv_crit:

        # Increment loop counter and exit if necessary
        iters += 1
        if iters > maxiters:
            iters -= 1
            print('Solution did not converge in ', iters, ' iterations')
            print('Last diff magnitude: ', diff)
            break

        # Initialze values for this iteration
        # Reset P every iteration???
        # P = Po.copy()
        
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
            ybar = np.dot(gamma_til_k, Wm.T)
            ybar = np.reshape(ybar, (p, 1))
            resids = Yk - ybar
            
            # Accumulate measurements and computed measurements
            Y_til[meas_ind:meas_ind+p] = Yk
            Y_bar[meas_ind:meas_ind+p] = ybar
            gamma_til_mat[meas_ind:meas_ind+p, :] = gamma_til_k  
            Rk_list.append(Rk)
            
            # Store output
            resids_list.append(resids)
            
            
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

        # Compute Kalman Gain
        cholPyy_inv = np.linalg.inv(np.linalg.cholesky(Pyy))
        Pyy_inv = np.dot(cholPyy_inv.T, cholPyy_inv) 
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
        
        
        diff = np.linalg.norm(np.dot(K, Y_til-Y_bar))

        print('Iteration Number: ', iters)
        print('diff = ', diff)

    
    
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

def ls_ekf(state_dict, truth_dict, meas_dict, meas_fcn, state_params,
           sensor_params, int_params):    
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
    
    # State information
    state_tk = sorted(state_dict.keys())[-1]
    Xo_ref = state_dict[state_tk]['X']
    Po_bar = state_dict[state_tk]['P']
    Q = state_params['Q']
    gap_seconds = state_params['gap_seconds']
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
        

        # After filter convergence, update reference trajectory
        if conv_flag:
            Xref = Xk
            xhat = np.zeros((n, 1))
            
            
    
            
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


def ls_ukf(state_dict, truth_dict, meas_dict, meas_fcn, state_params,
           sensor_params, int_params):    
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
    
    # State information
    state_tk = sorted(state_dict.keys())[-1]
    Xk = state_dict[state_tk]['X']
    P = state_dict[state_tk]['P']
    Q = state_params['Q']
    gap_seconds = state_params['gap_seconds']
    time_format = int_params['time_format']

    n = len(Xk)
    q = int(Q.shape[0])
    
    # Prior information about the distribution
    pnorm = 2.
    kurt = math.gamma(5./pnorm)*math.gamma(1./pnorm)/(math.gamma(3./pnorm)**2.)
    beta = kurt - 1.
    kappa = kurt - float(n)
    
    # Compute sigma point weights
    alpha = state_params['alpha']
    lam = alpha**2.*(n + kappa) - n
    gam = np.sqrt(n + lam)
    Wm = 1./(2.*(n + lam)) * np.ones(2*n,)
    Wc = Wm.copy()
    Wm = np.insert(Wm, 0, lam/(n + lam))
    Wc = np.insert(Wc, 0, lam/(n + lam) + (1 - alpha**2 + beta))
    diagWc = np.diag(Wc)
    unscented_params = {}
    unscented_params['gam'] = gam
    unscented_params['Wm'] = Wm
    unscented_params['diagWc'] = diagWc

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







