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
    yn = y0.flatten()
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
        yvec = np.concatenate((yvec, yn1.reshape(1,len(y0))), axis=0)
        tvec[ii+1] = tn

    return tvec, yvec


def rkf78(intfcn, tin, y0, params):
    '''
    This function implements the variable step-size, single-step,
    RKF78 integrator.
    
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
        
    Reference
    ------
    [1] Erwin Fehlberg, "Classical Fifth-, Sixth-, Seventh-, and
          Eighth-Order Runge-Kutta Formulas with Stepsize Control," 
          George C. Marshall Space Flight Center, Huntsville, AL, 
          NASA TR R-287, 1968.
    
    '''
    
    # Input parameters
    h = params['step']
    rtol = params['rtol']
    atol = params['atol']
    #m = params['m']
    m = 0.9
    local_extrap = params['local_extrap']
    
    # Start and end times
    t0 = tin[0]
    tf = tin[-1]
    
#    tflag = False
#    if len(tin) > 2:
#        tflag = True
    
    # Initial setup
    yn = y0.flatten()
    tn = t0
    yvec = y0.reshape(1, len(y0))
    tvec = np.array([t0])
    fcalls = 0
    
    # Set up weights    
    # 11 stage
    b_7  = np.array([41./840., 0., 0., 0., 0., 34./105., 9./35., 9./35.,
                     9./280., 9./280., 41./840., 0., 0.])
    
    # 13 stage (bhat)
    b_8  = np.array([0., 0., 0., 0., 0., 34./105., 9./35., 9./35., 9./280.,
                     9./280., 0., 41./840., 41./840.])
 
    # Set up nodes
    c = np.array([0., 2./27., 1./9., 1./6., 5./12., 1./2., 5./6., 1./6.,
                  2./3., 1./3., 1., 0., 1.])

    # Set up Runge-Kutta matrix
    A = np.zeros((13,13))
    A[1,0] = 2./27.
    A[2,0:2] = [1./36., 1./12.]
    A[3,0:3] = [1./24., 0., 1./8.]
    A[4,0:4] = [5./12., 0., -25./16., 25./16.]
    A[5,0:5] = [1./20., 0., 0., 1./4., 1./5.]
    A[6,0:6] = [-25./108., 0., 0., 125./108., -65./27., 125./54.]
    A[7,0:7] = [31./300., 0., 0., 0., 61./225., -2./9., 13./900.]
    A[8,0:8] = [2., 0., 0., -53./6., 704./45., -107./9., 67./90., 3.]
    A[9,0:9] = [-91./108., 0., 0., 23./108., -976./135., 311./54., -19./60., 17./6., -1./12.]
    A[10,0:10] = [2383./4100., 0., 0., -341./164., 4496./1025., -301./82., 2133./4100., 45./82., 45./164., 18./41.]
    A[11,0:11] = [3./205., 0., 0., 0., 0., -6./41., -3./205., -3./41., 3./41., 6./41., 0.]
    A[12,0:12] = [-1777./4100., 0., 0., -341./164., 4496./1025., -289./82., 2193./4100., 51./82., 33./164., 12./41., 0., 1.]
    
    # Loop to end
    k8 = np.zeros((len(yn),13))
    while tn < tf:

        k8[:,0] = h * intfcn(tn,yn,params)
        k8[:,1] = h * intfcn(tn+c[1]*h,yn+(k8[:,0]*A[1,0]).flatten(),params)
        k8[:,2] = h * intfcn(tn+c[2]*h,yn+np.dot(k8[:,0:2],A[2,0:2].T).flatten(),params)
        k8[:,3] = h * intfcn(tn+c[3]*h,yn+np.dot(k8[:,0:3],A[3,0:3].T).flatten(),params)
        k8[:,4] = h * intfcn(tn+c[4]*h,yn+np.dot(k8[:,0:4],A[4,0:4].T).flatten(),params)
        k8[:,5] = h * intfcn(tn+c[5]*h,yn+np.dot(k8[:,0:5],A[5,0:5].T).flatten(),params)
        k8[:,6] = h * intfcn(tn+c[6]*h,yn+np.dot(k8[:,0:6],A[6,0:6].T).flatten(),params)
        k8[:,7] = h * intfcn(tn+c[7]*h,yn+np.dot(k8[:,0:7],A[7,0:7].T).flatten(),params)
        k8[:,8] = h * intfcn(tn+c[8]*h,yn+np.dot(k8[:,0:8],A[8,0:8].T).flatten(),params)
        k8[:,9] = h * intfcn(tn+c[9]*h,yn+np.dot(k8[:,0:9],A[9,0:9].T).flatten(),params)
        k8[:,10] = h * intfcn(tn+c[10]*h,yn+np.dot(k8[:,0:10],A[10,0:10].T).flatten(),params)
        k8[:,11] = h * intfcn(tn+c[11]*h,yn+np.dot(k8[:,0:11],A[11,0:11].T).flatten(),params)
        k8[:,12] = h * intfcn(tn+c[12]*h,yn+np.dot(k8[:,0:12],A[12,0:12].T).flatten(),params)
        
        # Increment counter
        fcalls += 13
        
        # Compute updated solution and embedded solution for step size control
        # 7th Order Solution (q)
        y_7 = yn + np.dot(k8,b_7.T).flatten()
    
        # 8th Order Solution (p)
        y_8 = yn + np.dot(k8,b_8.T).flatten()
    
        # Error Checks
        delta = max(abs(y_7 - y_8))
        if delta < 1e-15:
            delta = 5e-15
        
        epsilon = max(abs(yn))*rtol + atol
    
        # Updated step size
        hnew = h*m*(epsilon/delta)**(1./8.)
        
        # Check condition for appropriate step size
        # If condition met, solution is good, proceed 
        if delta <= epsilon:
                  
            # Standard 7th Order Solution
            yn = y_7
        
            # Local Extrapolation Using 8th Order Solution
            if local_extrap:
                yn = y_8            
        
            # Increment current time
            tn = tn + h            
            
            # Use new time step
            h = hnew
        
            # Store Output
            yvec = np.concatenate((yvec, yn.reshape(1,len(yn))), axis=0)
            tvec = np.append(tvec, tn)
        
            
        
        # Otherwise, solution is bad, repeat at current time with new h
        else:            
            h = hnew

    return tvec, yvec, fcalls




















