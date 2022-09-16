import numpy as np
from math import floor

import sys
import os
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)


def interp_lagrange(X, Y, xx, p):
    '''
    This function interpolates data using Lagrange method of order P
    
    Parameters
    ------
    X : 1D numpy array
        x-values of data to interpolate
    Y : 2D numpy array
        y-values of data to interpolate
    xx : float
        single x value to interpolate at
    p : int
        order of interpolation
    
    Returns
    ------
    yy : 1D numpy array
        interpolated y-value(s)
        
    References
    ------
    [1] Kharab, A., An Introduction to Numerical Methods: A MATLAB 
        Approach, 2nd ed., 2005.
            
    '''
    
    # Number of data points to use for interpolation (e.g. 8,9,10...)
    N = p + 1

    if (len(X) < N):
        print('Not enough data points for desired Lagrange interpolation!')
        
    # Compute number of elements on either side of middle element to grab
    No2 = 0.5*N
    nn  = int(floor(No2))
    
    # Find index such that X[row0] < xx < X[row0+1]
    row0 = list(np.where(X < xx)[0])[-1]
    
    # Trim data set
    # N is even (p is odd)    
    if (No2-nn == 0): 
        
        # adjust row0 in case near data set endpoints
        if (N == len(X)) or (row0 < nn-1):
            row0 = nn-1
        elif (row0 > (len(X)-nn)):  # (row0 == length(X))            
            row0 = len(X) - nn - 1        
    
        # Trim to relevant data points
        X = X[row0-nn+1 : row0+nn+1]
        Y = Y[row0-nn+1 : row0+nn+1, :]


    # N is odd (p is even)
    else:
    
        # adjust row0 in case near data set endpoints
        if (N == len(X)) or (row0 < nn):
            row0 = nn
        elif (row0 > len(X)-nn):
            row0 = len(X) - nn - 1
        else:
            if (xx-X(row0) > 0.5) and (row0+1+nn < len(X)):
                row0 = row0 + 1
    
        # Trim to relevant data points
        X = X[row0-nn:row0+nn+1]
        Y = Y[row0-nn:row0+nn+1, :]
        
    # Compute coefficients
    Pj = np.ones((1,N))
    
    for jj in range(N):
        for ii in range(N):
            
            if jj != ii:
                Pj[0, jj] = Pj[0, jj] * (-xx+X[ii])/(-X[jj]+X[ii])
    
    
    yy = np.dot(Pj, Y)
    
    return yy


def single_shooting(Xo_init, bc_vect, tin, boundary_fcn, state_params, int_params,
                    finite_diff_step=1e-6, tol=1e-14, maxiters=100):
    '''
    This function implements the single shooting technique to solve two point
    boundary value problems. The method takes an input guess at the initial
    conditions, computes the difference to the final boundary condition and
    uses Newton iteration to update the initial state until the boundary
    condition is met.
    
    Parameters
    ------
    Xo_init : n0x1 numpy array
        initial guess at parameters that will be varied
    bc_vect : nfx1 numpy array
        boundary condition values
    tin : 1D array or list
        times to integrate over, must contain at least [t0, tf] but can 
        include intermediate values 
    boundary_fcn : function handle
        function to compute boundary values for given initial parameters 
        (e.g. using numerical integration)
    state_params : dictionary
        physical parameters related to dynamics or object
    int_params : dictionary
        numerical integration parameters
    finite_diff_step : float, optional
        unitless step size for central finite difference calculation
        (default=1e-6)
    tol : float, optional
        loop error tolerance (default=1e-14)
    maxiters : int, optional
        maximum number of iterations (default=100)
        
    Returns
    ------
    Xo : n0x1 numpy array
        solved initial parameters to achieve boundary condition
    fail_flag : boolean
        exit status (True = iteration did not converge)

    '''
    
    # Initial and final states
    Xo = Xo_init.copy()
    n = len(Xo)
    m = len(bc_vect)
    
    # Setup loop
    err = 10*tol
    iters = 0
    fail_flag = False
    while err > tol:
        
        # Compute penalty for current guess
        Xb_num = boundary_fcn(Xo, tin, state_params, int_params)
        c_vect = Xb_num - bc_vect
        
        # Loop over states and compute central finite differences to populate
        # Jacobian matrix
        cmat = np.zeros((m, n))
        for ii in range(n):
            
            # Step size for this state parameter
            dxi = max(Xo[ii]*finite_diff_step, finite_diff_step)
            
            # Compute minus side            
            Xo_minus = Xo.copy()
            Xo_minus[ii] -= dxi
            Xb_num = boundary_fcn(Xo_minus, tin, state_params, int_params)
            cm_vect = Xb_num - bc_vect
            
            # Compute plus side
            Xo_plus = Xo.copy()
            Xo_plus[ii] += dxi
            Xb_num = boundary_fcn(Xo_plus, tin, state_params, int_params)
            cp_vect = Xb_num - bc_vect
            
            dc_dxi = (1./(2.*dxi)) * (cp_vect - cm_vect)
            cmat[ii,:] = dc_dxi.flatten()

        # Compute updated initial state
        delta_vect = -np.dot(np.linalg.inv(cmat), c_vect)
        Xo += delta_vect
        
        denom = max([np.linalg.norm(bc_vect), np.linalg.norm(Xo), 1.])
        err = np.linalg.norm(c_vect)/denom
        
        print('')
        print('iters', iters)
        print('cmat', cmat)
        print('err', err)

        iters += 1
        if iters > maxiters:
            fail_flag = True
            break
    
    
    return Xo, fail_flag


def multiple_shooting(Xo_init, bc_vect, tin, cvect_fcn, state_params,
                      int_params, finite_diff_step=1e-6, tol=1e-14,
                      maxiters=100):
    '''
    This function implements the single shooting technique to solve two point
    boundary value problems. The method takes an input guess at the n variable
    parameters Xo_init, computes the difference to the m constraints 
    (e.g. boundary conditions from bc_vect) computed by cvect_function and uses 
    Newton iteration to update the variable parameters until the boundary 
    conditions and other constraints are met.
    
    Number of constraints m <= Number of variable parameters n
    
    Parameters
    ------
    Xo_init : nx1 numpy array
        initial guess at parameters that will be varied
    bc_vect : bx1 numpy array
        boundary condition values
    tin : 1D array or list
        times to integrate over, must contain at least [t0, tf] but can 
        include intermediate values 
    cvect_fcn : function handle
        function to compute defects/boundary penalties (desire c(x) = 0)
        (e.g. using numerical integration)
    state_params : dictionary
        physical parameters related to dynamics or object
    int_params : dictionary
        numerical integration parameters
    finite_diff_step : float, optional
        unitless step size for central finite difference calculation
        (default=1e-6)
    tol : float, optional
        loop error tolerance (default=1e-14)
    maxiters : int, optional
        maximum number of iterations (default=100)
        
    Returns
    ------
    Xo : nx1 numpy array
        solved parameters to achieve boundary conditions and other penalties
    fail_flag : boolean
        exit status (True = iteration did not converge)
        
        
    References
    ------
    [1] Betts, J.T., "Practical Method for Optimal Control and Estimation 
        Using Nonlinear Programming," 2nd ed. 2010.
    

    '''
    
    # Initial vector of parameters to vary
    Xo = Xo_init.copy()
    n = len(Xo)
    
    # Setup loop
    err = 10*tol
    iters = 0
    fail_flag = False
    while err > tol:
        
        # Compute penalty for current guess
        c_vect = cvect_fcn(Xo, bc_vect, tin, state_params, int_params)
        m = len(c_vect)
        
        # Loop over states and compute central finite differences to populate
        # Jacobian matrix
        G = np.zeros((m, n))
        for ii in range(n):
            
            # Step size for this state parameter
            dxi = max(Xo[ii]*finite_diff_step, finite_diff_step)
            
            # Compute minus side            
            Xo_minus = Xo.copy()
            Xo_minus[ii] -= dxi
            cm_vect = cvect_fcn(Xo_minus, bc_vect, tin, state_params, int_params)
            
            # Compute plus side
            Xo_plus = Xo.copy()
            Xo_plus[ii] += dxi
            cp_vect = cvect_fcn(Xo_plus, bc_vect, tin, state_params, int_params)
            
            dc_dxi = (1./(2.*dxi)) * (cp_vect - cm_vect)
            G[:,ii] = dc_dxi.flatten()
            
            print('')
            print('ii', ii)
            print('X[ii]', Xo[ii])
            print('dxi', dxi)
            print('dc_dxi', dc_dxi)
            

        # Compute updated initial state
        if m == n:
            delta_vect = -np.dot(np.linalg.inv(G), c_vect)
        elif m < n:
            Ginv = np.dot(G.T, np.linalg.inv(np.dot(G, G.T)))
            delta_vect = -np.dot(Ginv, c_vect)
            
        Xo += delta_vect
        
        denom = max([np.linalg.norm(bc_vect), np.linalg.norm(Xo), 1.])
        err = np.linalg.norm(c_vect)/denom
        
        print('')
        print('iters', iters)
        print('G', G)
        print('delta_vect', delta_vect)
        print('Xo', Xo)
        print('err', err)

        
        iters += 1
        if iters > maxiters:
            fail_flag = True
            break
    
    
    return Xo, fail_flag


