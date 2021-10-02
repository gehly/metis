import numpy as np
from scipy.integrate import odeint, ode
import sys

sys.path.append('../')

from dynamics.numerical_integration import rk4, rkf78, dopri87



###############################################################################
# General Interface
###############################################################################

def general_dynamics(Xo, tvec, state_params, int_params):
    '''
    This function provides a general interface to numerical integration 
    routines.
    
    '''
    
    integrator = int_params['integrator']
    
    # Setup and run integrator depending on user selection
    if integrator == 'rk4':
        
        # Setup integrator parameters
        params = state_params
        intfcn = int_params['intfcn']        
        params['step'] = int_params['step']
        
        # Run integrator
        tout, Xout, fcalls = rk4(intfcn, tvec, Xo, params)
        
        
    if integrator == 'rkf78':
        
        # Setup integrator parameters
        params = state_params
        intfcn = int_params['intfcn']
        params['step'] = int_params['step']
        params['rtol'] = int_params['rtol']
        params['atol'] = int_params['atol']
        params['local_extrap'] = int_params['local_extrap']
        
        # Run integrator
        if len(tvec) == 2:
            tout, Xout, fcalls = rkf78(intfcn, tvec, Xo, params)
            
        else:
            
            Xout = np.zeros((len(tvec), len(Xo)))
            Xout[0] = Xo
            tin = tvec[0:2]
            
            # Run integrator
            k = 1
            while tin[0] < tvec[-1]:           
                dum, Xout_step, fcalls = rkf78(intfcn, tin, Xo, params)
                Xo = Xout_step[-1,:]
                tin = tvec[k:k+2]
                Xout[k] = Xo               
                k += 1
            
            tout = tvec
            
        
    if integrator == 'dopri87':
        
        # Setup integrator parameters
        params = state_params
        intfcn = int_params['intfcn']
        params['step'] = int_params['step']
        params['rtol'] = int_params['rtol']
        params['atol'] = int_params['atol']
        
        # Run integrator
        if len(tvec) == 2:
            tout, Xout, fcalls = dopri87(intfcn, tvec, Xo, params)
            
        else:
            
            Xout = np.zeros((len(tvec), len(Xo)))
            Xout[0] = Xo
            tin = tvec[0:2]
            
            # Run integrator
            k = 1
            while tin[0] < tvec[-1]:           
                dum, Xout_step, fcalls = dopri87(intfcn, tin, Xo, params)
                Xo = Xout_step[-1,:]
                tin = tvec[k:k+2]
                Xout[k] = Xo               
                k += 1
            
            tout = tvec
            
        
    if integrator == 'odeint':
        
        # Setup integrator parameters
        params = state_params
        intfcn = int_params['intfcn']
        rtol = int_params['rtol']
        atol = int_params['atol']
        
        # Run integrator
        tout = tvec
        Xout = odeint(intfcn,Xo,tvec,(params,),rtol=rtol,atol=atol)
        
        
    if integrator == 'ode':
        
        # Setup integrator parameters
        params = state_params
        intfcn = int_params['intfcn']
        ode_integrator = int_params['ode_integrator']
        rtol = int_params['rtol']
        atol = int_params['atol']
        
        solver = ode(intfcn)
        solver.set_integrator(ode_integrator, atol=atol, rtol=rtol)
        solver.set_f_params(params)
        
        solver.set_initial_value(Xo, tvec[0])
        Xout = np.zeros((len(tvec), len(Xo)))
        Xout[0] = Xo.flatten()
        
        # Run integrator
        k = 1
        while solver.successful() and solver.t < tvec[-1]:
            solver.integrate(tvec[k])
            Xout[k] = solver.y.flatten()
            k += 1
        
        tout = tvec
    
    
    
    return tout, Xout



###############################################################################
# Generic Dynamics Functions
###############################################################################

def ode_balldrop(t, X, params):
    '''
    This function works with ode to propagate an object moving under constant
    acceleration

    Parameters
    ------
    X : 2 element array
      cartesian state vector 
    t : float 
      current time in seconds
    params : dictionary
        additional arguments

    Returns
    ------
    dX : 2 element array array
      state derivative vector
    
    '''
    
    y = float(X[0])
    dy = float(X[1])
    
    dX = np.zeros(2,)
    dX[0] = dy
    dX[1] = params['acc']
    
    return dX


def ode_balldrop_stm(t, X, params):

    # Input data
    acc = params['acc']

    # Number of states
    n = 2

    # State Vector
    y = float(X[0])
    dy = float(X[1])

    # Generate A matrix
    A = np.zeros((n, n))
    A[0,1] = 1.

    # Compute STM components dphi = A*phi
    phi_mat = np.reshape(X[n:], (n, n))
    dphi_mat = np.dot(A, phi_mat)
    dphi_v = np.reshape(dphi_mat, (n**2, 1))

    # Derivative vector
    dX = np.zeros(n+n**2,)

    dX[0] = dy
    dX[1] = acc
    dX[n:] = dphi_v.flatten()

    return dX


###############################################################################
# Orbit Propagation Routines
###############################################################################

###############################################################################
# Two-Body Orbit Functions
###############################################################################

def int_twobody(X, t, params):
    '''
    This function works with odeint to propagate object assuming
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
    dX : 6 element array
      state derivative vector
    '''
    

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])
    
    # Additional arguments
    GM = params['GM']

    # Compute radius
    r = np.linalg.norm([x, y, z])

    # Derivative vector
    dX = np.zeros(6,)

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3
    dX[4] = -GM*y/r**3
    dX[5] = -GM*z/r**3

    return dX


def int_twobody_ukf(X, t, params):
    '''
    This function works with odeint to propagate object assuming
    simple two-body dynamics.  No perturbations included.  States for UKF
    sigma points included.

    Parameters
    ------
    X : (n*(2n+1)) element array
      initial condition vector of cartesian state and sigma points
    t : float 
      current time in seconds
    params : dictionary
        additional arguments

    Returns
    ------
    dX : (n*(2n+1)) element array
      derivative vector

    '''
    
    # Additional arguments
    GM = params['GM']

    # Initialize
    dX = np.zeros(len(X),)
    n = int((-1 + np.sqrt(1. + 8.*len(X)))/4.)

    for ind in range(0, 2*n+1):

        # Pull out relevant values from X
        x = float(X[ind*n])
        y = float(X[ind*n + 1])
        z = float(X[ind*n + 2])
        dx = float(X[ind*n + 3])
        dy = float(X[ind*n + 4])
        dz = float(X[ind*n + 5])

        # Compute radius
        r = np.linalg.norm([x, y, z])

        # Solve for components of dX
        dX[ind*n] = dx
        dX[ind*n + 1] = dy
        dX[ind*n + 2] = dz

        dX[ind*n + 3] = -GM*x/r**3
        dX[ind*n + 4] = -GM*y/r**3
        dX[ind*n + 5] = -GM*z/r**3

    return dX


def int_twobody_stm(X, t, params):
    '''
    This function works with odeint to propagate object assuming
    simple two-body dynamics.  No perturbations included.
    Partials for the STM dynamics are included.

    Parameters
    ------
    X : (n+n^2) element array
      initial condition vector of cartesian state and STM (Inertial Frame)
    t : float 
      current time in seconds
    params : dictionary
        additional arguments

    Returns
    ------
    dX : (n+n^2) element array
      derivative vector
      
    '''

    # Additional arguments
    GM = params['GM']

    # Compute number of states
    n = int((-1 + np.sqrt(1 + 4*len(X)))/2)

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])

    # Compute radius
    r = np.linalg.norm([x, y, z])

    # Find elements of A matrix
    xx_cf = -GM/r**3 + 3.*GM*x**2/r**5
    xy_cf = 3.*GM*x*y/r**5
    xz_cf = 3.*GM*x*z/r**5
    yy_cf = -GM/r**3 + 3.*GM*y**2/r**5
    yx_cf = xy_cf
    yz_cf = 3.*GM*y*z/r**5
    zz_cf = -GM/r**3 + 3.*GM*z**2/r**5
    zx_cf = xz_cf
    zy_cf = yz_cf

    # Generate A matrix
    A = np.zeros((n, n))

    A[0,3] = 1.
    A[1,4] = 1.
    A[2,5] = 1.

    A[3,0] = xx_cf
    A[3,1] = xy_cf
    A[3,2] = xz_cf

    A[4,0] = yx_cf
    A[4,1] = yy_cf
    A[4,2] = yz_cf

    A[5,0] = zx_cf
    A[5,1] = zy_cf
    A[5,2] = zz_cf

    # Compute STM components dphi = A*phi
    phi_mat = np.reshape(X[n:], (n, n))
    dphi_mat = np.dot(A, phi_mat)
    dphi_v = np.reshape(dphi_mat, (n**2, 1))

    # Derivative vector
    dX = np.zeros(n+n**2,)

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3
    dX[4] = -GM*y/r**3
    dX[5] = -GM*z/r**3

    dX[n:] = dphi_v

    return dX


def ode_twobody(t, X, params):
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
    
    # Additional arguments
    GM = params['GM']

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])

    # Compute radius
    r = np.linalg.norm([x, y, z])

    # Derivative vector
    dX = np.zeros(6,)

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3
    dX[4] = -GM*y/r**3
    dX[5] = -GM*z/r**3
    
    return dX


def ode_twobody_ukf(t, X, params):
    '''
    This function works with ode to propagate object assuming
    simple two-body dynamics.  No perturbations included.  States for UKF
    sigma points included.

    Parameters
    ------
    X : (n*(2n+1)) element list
      initial condition vector of cartesian state and sigma points
    t : float 
      current time in seconds
    params : dictionary
        additional arguments

    Returns
    ------
    dX : (n*(2n+1)) element list
      derivative vector

    '''
    
    # Additional arguments
    GM = params['GM']
    
    # Initialize
    dX = [0]*len(X)
    n = int((-1 + np.sqrt(1. + 8.*len(X)))/4.)

    for ind in range(0, 2*n+1):

        # Pull out relevant values from X
        x = float(X[ind*n])
        y = float(X[ind*n + 1])
        z = float(X[ind*n + 2])
        dx = float(X[ind*n + 3])
        dy = float(X[ind*n + 4])
        dz = float(X[ind*n + 5])

        # Compute radius
        r = np.linalg.norm([x, y, z])

        # Solve for components of dX
        dX[ind*n] = dx
        dX[ind*n + 1] = dy
        dX[ind*n + 2] = dz

        dX[ind*n + 3] = -GM*x/r**3
        dX[ind*n + 4] = -GM*y/r**3
        dX[ind*n + 5] = -GM*z/r**3

    return dX


def ode_twobody_stm(t, X, params):
    '''
    This function works with ode to propagate object assuming
    simple two-body dynamics.  No perturbations included.
    Partials for the STM dynamics are included.

    Parameters
    ------
    X : (n+n^2) element array
      initial condition vector of cartesian state and STM (Inertial Frame)
    t : float 
      current time in seconds
    params : dictionary
        additional arguments

    Returns
    ------
    dX : (n+n^2) element array
      derivative vector
      
    '''

    # Additional arguments
    GM = params['GM']

    # Compute number of states
    n = int((-1 + np.sqrt(1 + 4*len(X)))/2)

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])

    # Compute radius
    r = np.linalg.norm([x, y, z])

    # Find elements of A matrix
    xx_cf = -GM/r**3 + 3.*GM*x**2/r**5
    xy_cf = 3.*GM*x*y/r**5
    xz_cf = 3.*GM*x*z/r**5
    yy_cf = -GM/r**3 + 3.*GM*y**2/r**5
    yx_cf = xy_cf
    yz_cf = 3.*GM*y*z/r**5
    zz_cf = -GM/r**3 + 3.*GM*z**2/r**5
    zx_cf = xz_cf
    zy_cf = yz_cf

    # Generate A matrix
    A = np.zeros((n, n))

    A[0,3] = 1.
    A[1,4] = 1.
    A[2,5] = 1.

    A[3,0] = xx_cf
    A[3,1] = xy_cf
    A[3,2] = xz_cf

    A[4,0] = yx_cf
    A[4,1] = yy_cf
    A[4,2] = yz_cf

    A[5,0] = zx_cf
    A[5,1] = zy_cf
    A[5,2] = zz_cf

    # Compute STM components dphi = A*phi
    phi_mat = np.reshape(X[n:], (n, n))
    dphi_mat = np.dot(A, phi_mat)
    dphi_v = np.reshape(dphi_mat, (n**2, 1))

    # Derivative vector
    dX = np.zeros(n+n**2,)

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3
    dX[4] = -GM*y/r**3
    dX[5] = -GM*z/r**3

    dX[n:] = dphi_v

    return dX