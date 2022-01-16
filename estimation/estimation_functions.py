import numpy as np
import sys

sys.path.append('../')

from dynamics.dynamics_functions import general_dynamics



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
    meas_dict : dictionary
        measurement data over time for the filter and parameters (noise, etc)

    meas_fcn : function handle
        function for measurements


    Returns
    ------
    filter_output : dictionary
        output state, covariance, and post-fit residuals over time
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
    L = len(tk_list)

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
        N = np.dot(Lambda, xo_bar)

        # Loop over times
        for kk in range(L):
            
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
                
                tout, intout = general_dynamics(int0, tin, state_params, int_params)

            # Extract values for later calculations
            xout = intout[-1,:]
            Xref = xout[0:n].reshape(n, 1)
            phi_v = xout[n:].reshape(n**2, 1)
            phi = np.reshape(phi_v, (n, n))

            # Accumulate the normal equations
            Hk_til, Gk, Rk = meas_fcn(tk, Xref, state_params, sensor_params, sensor_id)
            yk = Yk - Gk
            Hk = np.dot(Hk_til, phi)
            cholRk = np.linalg.inv(np.linalg.cholesky(Rk))
            invRk = np.dot(cholRk.T, cholRk)
                        
            Lambda += np.dot(Hk.T, np.dot(invRk, Hk))
            N += np.dot(Hk.T, np.dot(invRk, yk))
            
            # Store output
            resids_list.append(yk)
            Xref_list.append(Xref)
            phi_list.append(phi)
            
#            print(kk)
#            print(tk)
#            print(int0)
#            print(Xref)
#            print(Yk)
#            print(Gk)
#            print(yk)
#            
#            if kk > 0:
#                mistake


        # Solve the normal equations
        cholLam_inv = np.linalg.inv(np.linalg.cholesky(Lambda))
        Po = np.dot(cholLam_inv.T, cholLam_inv)     
        xo_hat = np.dot(Po, N)
        xo_hat_mag = np.linalg.norm(xo_hat)

        # Update for next batch iteration
        Xo_ref = Xo_ref + xo_hat
        xo_bar = xo_bar - xo_hat

        print('Iteration Number: ', iters)
        print('xo_hat_mag = ', xo_hat_mag)

    # Form output
    for kk in range(L):
        tk = tk_list[kk]
        X = Xref_list[kk]
        resids = resids_list[kk]
        phi = phi_list[kk]
        P = np.dot(phi, np.dot(Po, phi.T))

        filter_output[tk] = {}
        filter_output[tk]['X'] = X
        filter_output[tk]['P'] = P
        filter_output[tk]['resids'] = resids
        
    
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
            
            tout, intout = general_dynamics(int0, tin, state_params, int_params)

        # Extract values for later calculations
        xout = intout[-1,:]
        Xref = xout[0:n].reshape(n, 1)
        phi_v = xout[n:].reshape(n**2, 1)
        phi = np.reshape(phi_v, (n, n))
        P = np.dot(phi, np.dot(Po, phi.T))
        
        full_state_output[tk] = {}
        full_state_output[tk]['X'] = Xref
        full_state_output[tk]['P'] = P
        
    

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







