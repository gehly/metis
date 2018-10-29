import numpy as np
from math import ceil




def rk4(intfcn, tin, y0, params):
    '''
    This function implements the fixed-step, single-step, 4th order Runge-Kutta
    integrator.
    
    Parameters
    ------
    intfcn : function handle
        handle for function to integrate
    tin : 1D numpy array
        times to integrate over, [t0, tf]
    y0 : numpy array
        initial state vector
    params : dictionary
        parameters for integration including step size and any additional
        variables needed for the integration function
    
    Returns
    ------
    tvec : 1D numpy array
        times corresponding to output states from t0 to tf
    yvec : (N+1)xn numpy array
        output state vectors at each time, each row is 1xn vector of state
        at corresponding time
    
    '''
    
    # Start and end times
    h = params['step']
    t0 = tin[0]
    tf = tin[-1]
    N = int(ceil((tf-t0)/h))
    
    # Initial setup
    yn = y0
    tn = t0
    yvec = y0.reshape(1, len(y0))
    tvec = np.zeros(N+1,)
    tvec[0] = t0
    
    # Loop to end
    for ii in range(N):
        
        # Compute k values
        k1 = h * intfcn(tn,yn,params)
        k2 = h * intfcn(tn+h/2,yn+k1/2,params)
        k3 = h * intfcn(tn+h/2,yn+k2/2,params)
        k4 = h * intfcn(tn+h,yn+k3,params)
        
        # Compute solution
        yn1 = yn + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        
        # Increment time and yn
        tn = tn+h
        yn = yn1
        
        # Store output
        yvec = np.concatenate((yvec, yn1), axis=0)
        tvec[ii+1] = tn

    return tvec, yvec