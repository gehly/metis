import numpy as np
from numba import jit



@jit(nopython=True)
def rk4(intfcn, tin, y0, step, GM):
    '''
    This function implements the fixed-step, single-step, 4th order Runge-Kutta
    integrator.
    
    Parameters
    ------
    intfcn : function handle
        handle for function to integrate
    tin : 1D numpy array
        times to integrate over, [t0, tf] or [t0, t1, t2, ... , tf]
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
    t0 = tin[0]
    tf = tin[-1]
    if len(tin) == 2:
        h = step
        tvec = np.arange(t0, tf, h)
        tvec = np.append(tvec, tf)
    else:
        tvec = tin

    # Initial setup
    h = step
    yn = y0.flatten()
    tn = t0
    yvec = y0.reshape(1, len(y0))
    
    # Loop to end
    while tn < tf:
                
        # Compute k values
        k1 = h * intfcn(tn,yn,GM)
        k2 = h * intfcn(tn+h/2.,yn+k1/2.,GM)
        k3 = h * intfcn(tn+h/2.,yn+k2/2.,GM)
        k4 = h * intfcn(tn+h,yn+k3,GM)
        
        # Compute solution
        yn += (1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)

        # Store output
        yvec = np.concatenate((yvec, yn.reshape(1,len(y0))), axis=0)
        
        # Increment time
        tn += h

    return tvec, yvec



@jit(nopython=True)
def jit_twobody(t, X, GM):
    '''
    This function works with ode to propagate object assuming
    simple two-body dynamics.  No perturbations included.

    Parameters
    ------
    X : 6 element array
      cartesian state vector (Inertial Frame)
    t : float 
      current time in seconds
    params : dictionary
        additional arguments

    Returns
    ------
    dX : 6 element array array
      state derivative vector
    
    '''

    # State Vector
    x = X[0]
    y = X[1]
    z = X[2]
    dx = X[3]
    dy = X[4]
    dz = X[5]

    # Compute radius
    r = np.sqrt(x**2. + y**2. + z**2.)

    # Derivative vector
    dX = np.zeros(6,)

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3
    dX[4] = -GM*y/r**3
    dX[5] = -GM*z/r**3
    
    return dX