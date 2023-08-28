import numpy as np
from math import exp
from scipy.integrate import odeint, ode, solve_ivp
import sys
import os
import inspect
import copy
from datetime import datetime, timedelta
from numba import types
from numba.typed import Dict

# Load tudatpy modules  
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array

# Load spice kernels
spice.load_standard_kernels()

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from dynamics import numerical_integration as numint
from dynamics import fast_integration as fastint
from utilities import astrodynamics as astro
from utilities import numerical_methods as num



###############################################################################
# General Interface
###############################################################################


def general_dynamics(Xo, tvec, state_params, int_params, bodies=None):
    '''
    This function provides a general interface to numerical integration 
    routines.
    
    '''
    
#    print(tvec)
    state_params = copy.deepcopy(state_params)
    int_params = copy.deepcopy(int_params)
    integrator = int_params['integrator']
    
    # Flatten input state
    Xo = Xo.flatten()
    
    # Convert time to seconds
    time_format = int_params['time_format']
    if time_format == 'datetime':
        t0 = tvec[0]
        tvec = [(ti - t0).total_seconds() for ti in tvec]
    if time_format == 'JD':
        t0 = tvec[0]
        tvec = [(ti - t0)*86400. for ti in tvec]
        
    
    # print('tvec', tvec)
        
    # Exit if no integration needed
    if tvec[0] == tvec[-1]:
        Xout = np.zeros((len(tvec), len(Xo)))
        Xout[0] = Xo.flatten()
        Xout[-1] = Xo.flatten()
        
        return tvec, Xout
    
    # Setup and run integrator depending on user selection
    if integrator == 'rk4':
        
        # Setup integrator parameters
        params = state_params
        intfcn = int_params['intfcn']        
        params['step'] = int_params['step']
        
        # Run integrator
        tout, Xout, fcalls = numint.rk4(intfcn, tvec, Xo, params)
        
        return tout, Xout
    
    
    if integrator == 'rk4_jit':
        
        # Setup integrator parameters
        intfcn = int_params['intfcn']
        params = Dict.empty(key_type=types.unicode_type, value_type=types.float64,)
        params['step'] = int_params['step']
        
        for key in state_params:
            params[key] = state_params[key]
            
        # Run integrator
        tvec = np.asarray(tvec)
        tout, Xout = fastint.rk4(intfcn, tvec, Xo, params)
        
        return tout, Xout
        
        
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
            tout, Xout, fcalls = numint.rkf78(intfcn, tvec, Xo, params)
            
        else:
            
            Xout = np.zeros((len(tvec), len(Xo)))
            Xout[0] = Xo
            tin = tvec[0:2]
            
            # Run integrator
            k = 1
            while tin[0] < tvec[-1]:           
                dum, Xout_step, fcalls = numint.rkf78(intfcn, tin, Xo, params)
                Xo = Xout_step[-1,:]
                tin = tvec[k:k+2]
                Xout[k] = Xo               
                k += 1
            
            tout = tvec
        
        return tout, Xout
            
        
    if integrator == 'dopri87':
        
        # Setup integrator parameters
        params = state_params
        intfcn = int_params['intfcn']
        params['step'] = int_params['step']
        params['rtol'] = int_params['rtol']
        params['atol'] = int_params['atol']
        
        # Run integrator
        if len(tvec) == 2:
            tout, Xout, fcalls = numint.dopri87(intfcn, tvec, Xo, params)
            
        else:
            
            Xout = np.zeros((len(tvec), len(Xo)))
            Xout[0] = Xo
            tin = tvec[0:2]
            
            # Run integrator
            k = 1
            while tin[0] < tvec[-1]:           
                dum, Xout_step, fcalls = numint.dopri87(intfcn, tin, Xo, params)
                Xo = Xout_step[-1,:]
                tin = tvec[k:k+2]
                Xout[k] = Xo               
                k += 1
            
            tout = tvec
        
        return tout, Xout
    
    
    if integrator == 'dopri87_jit':
        
        # Setup integrator parameters
        intfcn = int_params['intfcn']
        params = Dict.empty(key_type=types.unicode_type, value_type=types.float64,)
        params['step'] = int_params['step']
        params['rtol'] = int_params['rtol']
        params['atol'] = int_params['atol']
        
        for key in state_params:
            params[key] = state_params[key]
            
        # Run integrator
        tvec = np.asarray(tvec)
        if len(tvec) == 2:
            tout, Xout = fastint.dopri87(intfcn, tvec, Xo, params)
            
        else:
            
            Xout = np.zeros((len(tvec), len(Xo)))
            Xout[0] = Xo
            tin = tvec[0:2]
            
            # Run integrator
            k = 1
            while tin[0] < tvec[-1]:           
                dum, Xout_step = fastint.dopri87(intfcn, tin, Xo, params)
                Xo = Xout_step[-1,:]
                tin = tvec[k:k+2]
                Xout[k] = Xo               
                k += 1
            
            tout = tvec

        return tout, Xout
            
            
    if integrator == 'dopri87_aegis':
        
        # Setup integrator parameters
        params = state_params
        intfcn = int_params['intfcn']
        params['step'] = int_params['step']
        params['rtol'] = int_params['rtol']
        params['atol'] = int_params['atol']
        params['A_fcn'] = int_params['A_fcn']
        params['dyn_fcn'] = int_params['dyn_fcn']
        
        # Run integrator
        if len(tvec) == 2:
            tout, Xout, fcalls, split_flag = numint.dopri87_aegis(intfcn, tvec, Xo, params)
            
        else:
            mistake
            
        return tout, Xout, split_flag
    
    
    if integrator == 'dopri87_aegis_jit':
        
        # Setup integrator parameters
        intfcn = int_params['intfcn']
        params = Dict.empty(key_type=types.unicode_type, value_type=types.float64,)
        params['step'] = int_params['step']
        params['rtol'] = int_params['rtol']
        params['atol'] = int_params['atol']
        
        for key in state_params:
            params[key] = state_params[key]
            
        # Run integrator
        tvec = np.asarray(tvec)
        if len(tvec) == 2:
            tout, Xout, split_flag = fastint.dopri87_aegis(intfcn, tvec, Xo, params)
            
        else:
             mistake
        
        return tout, Xout, split_flag
        
            
        
    if integrator == 'odeint':
        
        # Setup integrator parameters
        params = state_params
        intfcn = int_params['intfcn']
        rtol = int_params['rtol']
        atol = int_params['atol']
        tfirst = int_params['tfirst']
        hmax = int_params['max_step']
        
        # Run integrator
        tout = tvec
        Xout = odeint(intfcn,Xo,tvec,args=(params,),rtol=rtol,atol=atol,hmax=hmax,tfirst=tfirst)
        
        return tout, Xout
        
        
    if integrator == 'solve_ivp':
        
        # Setup integrator parameters
        params = state_params
        params.update(int_params)
        intfcn = int_params['intfcn']
        method = int_params['ode_integrator']
        rtol = int_params['rtol']
        atol = int_params['atol']
                
        # Run integrator
        tin = (tvec[0], tvec[-1])
        tout = tvec
        output = solve_ivp(intfcn,tin,Xo,method=method,args=(params,),rtol=rtol,atol=atol,t_eval=tvec)
        
        Xout = output['y'].T
        
        return tout, Xout

        
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
        
        eps = 1e-12
        
        # Run integrator
        k = 1
        while solver.successful() and solver.t < (tvec[-1]-eps):
#            print('k', k)
#            print('tvec_k', tvec[k])
            solver.integrate(tvec[k])
            Xout[k] = solver.y.flatten()
            k += 1
        
        tout = tvec
        
        return tout, Xout
    
    
    if integrator == 'tudat':
        
        # Convert initial state vector from km to meters for TUDAT propagator
        initial_state = Xo.flatten()*1000.
        
        # Set simulation start and end epochs
        if time_format == 'datetime':
            simulation_start_epoch = (t0 - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
            
        if time_format == 'JD':
            simulation_start_epoch = (t0 - 2451545.0) * 86400.
            
        if time_format == 'seconds':
            simulation_start_epoch = t0
        
        simulation_end_epoch = simulation_start_epoch + tvec[-1]
        
        # Initialize bodies if needed and retrieve state parameter data
        if bodies is None:
            bodies = initialize_tudat(state_params)
            
        central_bodies = state_params['central_bodies']
        bodies_to_create = state_params['bodies_to_create']
        mass = state_params['mass']
        Cd = state_params['Cd']
        Cr = state_params['Cr']
        drag_area_m2 = state_params['drag_area_m2']
        srp_area_m2 = state_params['srp_area_m2']
        sph_deg = state_params['sph_deg']
        sph_ord = state_params['sph_ord']
        
        
        # Create the bodies to propagate
        # TUDAT always uses 6 element state vector
        N = int(len(initial_state)/6)
        central_bodies = central_bodies*N
        bodies_to_propagate = []
        for jj in range(N):
            jj_str = str(jj)
            bodies.create_empty_body(jj_str)
            bodies.get(jj_str).mass = mass
            bodies_to_propagate.append(jj_str)
            
            if Cd > 0.:
                aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
                    drag_area_m2, [Cd, 0, 0]
                )
                environment_setup.add_aerodynamic_coefficient_interface(
                    bodies, jj_str, aero_coefficient_settings)
                
            if Cr > 0.:
                occulting_bodies = ['Earth']
                radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
                    'Sun', srp_area_m2, Cr, occulting_bodies
                )
                environment_setup.add_radiation_pressure_interface(
                    bodies, jj_str, radiation_pressure_settings)
                
                

        acceleration_settings_setup = {}        
        if 'Earth' in bodies_to_create:
            
            # Gravity
            if sph_deg == 0 and sph_ord == 0:
                acceleration_settings_setup['Earth'] = [propagation_setup.acceleration.point_mass_gravity()]
            else:
                acceleration_settings_setup['Earth'] = [propagation_setup.acceleration.spherical_harmonic_gravity(sph_deg, sph_ord)]
            
            # Aerodynamic Drag
            if Cd > 0.:                
                acceleration_settings_setup['Earth'].append(propagation_setup.acceleration.aerodynamic())
            
        if 'Sun' in bodies_to_create:
            
            # Gravity
            acceleration_settings_setup['Sun'] = [propagation_setup.acceleration.point_mass_gravity()]
            
            # Solar Radiation Pressure
            if Cr > 0.:                
                acceleration_settings_setup['Sun'].append(propagation_setup.acceleration.cannonball_radiation_pressure())

        
        if 'Moon' in bodies_to_create:
            
            # Gravity
            acceleration_settings_setup['Moon'] = [propagation_setup.acceleration.point_mass_gravity()]
        

        acceleration_settings = {}
        for jj in range(N):
            acceleration_settings[str(jj)] = acceleration_settings_setup
            
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies
        )
        

        # Create termination settings
        termination_condition = propagation_setup.propagator.time_termination(
            simulation_end_epoch, terminate_exactly_on_final_condition=True
        )


        # Create numerical integrator settings
        if int_params['tudat_integrator'] == 'rk4':
            fixed_step_size = int_params['step']
            integrator_settings = propagation_setup.integrator.runge_kutta_4(
                fixed_step_size
            )
            
        elif int_params['tudat_integrator'] == 'rkf78':
            initial_step_size = int_params['step']
            maximum_step_size = int_params['max_step']
            minimum_step_size = int_params['min_step']
            rtol = int_params['rtol']
            atol = int_params['atol']
            integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
                initial_step_size,
                propagation_setup.integrator.CoefficientSets.rkf_78,
                minimum_step_size,
                maximum_step_size,
                rtol,
                atol)
            
        # Create propagation settings
        propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            initial_state,
            simulation_start_epoch,
            integrator_settings,
            termination_condition
        )
        
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies, propagator_settings )

        # Extract the resulting state history and convert it to an ndarray
        states = dynamics_simulator.state_history
        states_array = result2array(states)        
        
        # print('states_array', states_array.shape)
        
        tout = states_array[:,0] - simulation_start_epoch
        Xout = states_array[:,1:6*N+1]*1e-3
        

        
        return tout, Xout
    
    
def initialize_tudat(state_params):
    
    # Retrive state and propagator settings
    bodies_to_create = state_params['bodies_to_create']
    global_frame_origin = state_params['global_frame_origin']
    global_frame_orientation = state_params['global_frame_orientation']
    
    
    # Create bodies
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)
    bodies = environment_setup.create_system_of_bodies(body_settings)
    
    return bodies


###############################################################################
# Generic Dynamics Functions
###############################################################################

def ode_linear1d(t, X, params):
    '''
    This function works with ode to propagate an object moving with no 
    acceleration.

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
    
    x = float(X[0])
    dx = float(X[1])
    
    dX = np.zeros(2,)
    dX[0] = dx
    dX[1] = 0.
    
    return dX


def ode_linear1d_stm(t, X, params):

    # Number of states
    n = 2

    # State Vector
    x = float(X[0])
    dx = float(X[1])

    # Generate A matrix
    A = np.zeros((n, n))
    A[0,1] = 1.

    # Compute STM components dphi = A*phi
    phi_mat = np.reshape(X[n:], (n, n))
    dphi_mat = np.dot(A, phi_mat)
    dphi_v = np.reshape(dphi_mat, (n**2, 1))

    # Derivative vector
    dX = np.zeros(n+n**2,)

    dX[0] = dx
    dX[1] = 0.
    dX[n:] = dphi_v.flatten()

    return dX


def ode_linear1d_ukf(t, X, params):
    
    
    # Initialize
    dX = np.zeros(len(X),)
    n = int((-1 + np.sqrt(1. + 8.*len(X)))/4.)

    for ind in range(0, 2*n+1):

        # Pull out relevant values from X
        x = float(X[ind*n])
        dx = float(X[ind*n + 1])

        # Set components of dX
        dX[ind*n] = dx
        dX[ind*n + 1] = 0.
    
    return dX


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


def ode_balldrop_ukf(t, X, params):
    
    # Input data
    acc = params['acc']
    
    # Initialize
    dX = np.zeros(len(X),)
    n = int((-1 + np.sqrt(1. + 8.*len(X)))/4.)

    for ind in range(0, 2*n+1):

        # Pull out relevant values from X
        y = float(X[ind*n])
        dy = float(X[ind*n + 1])

        # Set components of dX
        dX[ind*n] = dy
        dX[ind*n + 1] = acc
    
    return dX


def ode_coordturn(t, X, params):
    '''
    This function works with ode to propagate an object moving and turning.

    Parameters
    ------
    X : 5 element array
      state vector 
    t : float 
      current time in seconds
    params : dictionary
        additional arguments

    Returns
    ------
    dX : 5 element array array
      state derivative vector
    
    '''
    
    x = float(X[0])
    dx = float(X[1])
    y = float(X[2])
    dy = float(X[3])
    w = float(X[4])
    
    dX = np.zeros(5,)
    dX[0] = dx   #dx*np.cos(w*t) - dy*np.sin(w*t)
    dX[1] = -w*dx*np.sin(w*t) - w*dy*np.cos(w*t)
    dX[2] = dy   #dx*np.sin(w*t) + dy*np.cos(w*t)
    dX[3] = w*dx*np.cos(w*t) - w*dy*np.sin(w*t)
    
    return dX


def ode_coordturn_ukf(t, X, params):
    '''
    This function works with ode to propagate an object moving and turning.

    Parameters
    ------
    X : 5 element array
      state vector 
    t : float 
      current time in seconds
    params : dictionary
        additional arguments

    Returns
    ------
    dX : 5 element array array
      state derivative vector
    
    '''
    
    n = 5
    dX = np.zeros(len(X),)
    for ind in range(0, 2*n+1):

        # Pull out relevant values from X
        x  = float(X[ind*n+0])
        dx = float(X[ind*n+1])
        y  = float(X[ind*n+2])
        dy = float(X[ind*n+3])
        w  = float(X[ind*n+4])

        # Set components of dX
        dX[ind*n+0] = dx   #dx*np.cos(w*t) - dy*np.sin(w*t)
        dX[ind*n+1] = -w*dx*np.sin(w*t) - w*dy*np.cos(w*t)
        dX[ind*n+2] = dy   #dx*np.sin(w*t) + dy*np.cos(w*t)
        dX[ind*n+3] = w*dx*np.cos(w*t) - w*dy*np.sin(w*t)
    
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


def ode_twobody_stm(t, X, params):
    '''
    This function works with ode to propagate object assuming
    simple two-body dynamics.  No perturbations included.
    Partials for the STM dynamics are included.

    Parameters
    ------
    t : float 
      current time in seconds
    X : (n+n^2) element array
      initial condition vector of cartesian state and STM (Inertial Frame)    
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

    dX[n:] = dphi_v.flatten()

    return dX


def A_twobody(t, X, params):
    '''
    This function computes the dynamics model Jacobian corresponding to the
    first order Taylor Series expansion, for use with standard batch and
    Kalman filter implementations.
    
    Two-Body dynamics, no perturbing forces included.
    
    Parameters
    ------
    t : float 
      current time in seconds
    X : nx1 numpy array
      initial condition vector of cartesian state and STM (Inertial Frame)
    params : dictionary
        additional arguments
        
    Returns
    ------
    A : nxn numpy array
        Jacobian matrix    
    '''
    
    # Additional arguments
    GM = params['GM']

    # Compute number of states
    n = len(X)

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])

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
    
    return A






###############################################################################
# Orbit Dynamics with Perturbations
###############################################################################
   

def ode_twobody_j2_drag(t, X, params):
    '''
    This function works with ode to propagate object assuming
    two-body dynamics with J2 and drag perturbations included.
    This function uses a low fidelity drag model using the 
    standard atmospheric model, a co-rotating atmosphere to 
    compute winds, and a spherical Earth to compute orbit height.
    
    It should NOT be used for high fidelity orbit prediction.

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
    dX : n element array array
      state derivative vector
    
    '''
    
    # Additional arguments
    GM = params['GM']
    J2 = params['J2']
    Cd = params['Cd']
    R = params['R']
    dtheta = params['dtheta']
    A_m = params['A_m']
#    UTC0 = params['UTC0']
#    eop_alldata = params['eop_alldata']
#    XYs_df = params['XYs_df']
    
    # Number of states
    n = len(X)

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])
    
    if n > 6:
        beta = float(X[6])
    else:
        beta = Cd*A_m

    # Compute radius
    r_vect = np.array([[x], [y], [z]])
    r = np.linalg.norm(r_vect)

    # Compute drag component
    if beta == 0.:
        x_drag = 0.
        y_drag = 0.
        z_drag = 0.
    
    else:    
        # Find vector va of spacecraft relative to atmosphere
        v_vect = np.array([[dx], [dy], [dz]])
        w_vect = np.array([[0.], [0.], [dtheta]])
        va_vect = v_vect - np.cross(w_vect, r_vect, axis=0)
        va = np.linalg.norm(va_vect)
        va_x = float(va_vect[0])
        va_y = float(va_vect[1])
        va_z = float(va_vect[2])
        
        # Atmosphere lookup
#        UTC = UTC0 + timedelta(seconds=t)
#        EOP_data = eop.get_eop_data(eop_alldata, UTC)
#        r_ecef, dum = coord.gcrf2itrf(r_vect, v_vect, UTC, EOP_data, XYs_df)
#        lat, lon, ht = coord.ecef2latlonht(r_ecef)
        
        ht = r - R
        rho0, h0, H = astro.atmosphere_lookup(ht)
        
        # Calculate drag
        drag = -0.5*beta*rho0*exp(-(ht - h0)/H)
        x_drag = drag*va*va_x
        y_drag = drag*va*va_y
        z_drag = drag*va*va_z
    

    # Derivative vector
    dX = np.zeros(n,)

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3. + x_drag - 1.5*J2*R**2.*GM*((x/r**5.) - (5.*x*z**2./r**7.))
    dX[4] = -GM*y/r**3. + y_drag - 1.5*J2*R**2.*GM*((y/r**5.) - (5.*y*z**2./r**7.))
    dX[5] = -GM*z/r**3. + z_drag - 1.5*J2*R**2.*GM*((3.*z/r**5.) - (5.*z**3./r**7.))
    
    # If additional states such as beta are included their first derivative
    # is initialized to zero above
    
    return dX


def ode_twobody_j2_drag_stm(t, X, params):
    '''
    This function works with ode to propagate object assuming
    two-body dynamics with J2 and drag perturbations included.
    Partials for the STM dynamics are included.
    
    This function uses a low fidelity drag model using the 
    standard atmospheric model, a co-rotating atmosphere to 
    compute winds, and a spherical Earth to compute orbit height.
    
    It should NOT be used for high fidelity orbit prediction.
    

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
    J2 = params['J2']
    Cd = params['Cd']
    R = params['R']
    dtheta = params['dtheta']
    A_m = params['A_m']

    # Compute number of states
    n = int((-1 + np.sqrt(1 + 4*len(X)))/2)

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])
    
    # Compute ballistic coefficient from state vector or params
    if n > 6:
        beta = float(X[6])
    else:
        beta = Cd*A_m

    # Compute radius
    r_vect = np.array([[x], [y], [z]])
    r = np.linalg.norm(r_vect)
    
    # Find vector va of spacecraft relative to atmosphere
    v_vect = np.array([[dx], [dy], [dz]])
    w_vect = np.array([[0.], [0.], [dtheta]])
    va_vect = v_vect - np.cross(w_vect, r_vect, axis=0)
    va = np.linalg.norm(va_vect)
    va_x = float(va_vect[0])
    va_y = float(va_vect[1])
    va_z = float(va_vect[2])
    
    # Atmosphere lookup
#        UTC = UTC0 + timedelta(seconds=t)
#        EOP_data = eop.get_eop_data(eop_alldata, UTC)
#        r_ecef, dum = coord.gcrf2itrf(r_vect, v_vect, UTC, EOP_data, XYs_df)
#        lat, lon, ht = coord.ecef2latlonht(r_ecef)
    
    ht = r - R
    rho0, h0, H = astro.atmosphere_lookup(ht)
    
    # Calculate drag
    drag = -0.5*beta*rho0*exp(-(ht - h0)/H)
    x_drag = drag*va*va_x
    y_drag = drag*va*va_y
    z_drag = drag*va*va_z
    
    
    
    # Find elements of A matrix
    xx_cf = -GM/r**3. + 3.*GM*x**2./r**5.
    xx_drag = drag*((-x*va*va_x/(H*r)) - dtheta*va_y*va_x/va)
    xx_J2 = -1.5*J2*GM*R**2./r**5. - 7.5*J2*GM*R**2./r**7.*(-x**2. - z**2. + 7.*x**2.*z**2./r**2.)

    xy_cf = 3.*GM*x*y/r**5.
    xy_drag = drag*((-y*va*va_x/(H*r)) + dtheta*va_x**2./va + va*dtheta)
    xy_J2 = -7.5*x*y/r**7. * J2*R**2.*GM*(-1. + 7.*z**2./r**2.)

    xz_cf = 3.*GM*x*z/r**5.
    xz_drag = drag*(-z*va*va_x/(H*r))
    xz_J2 = -7.5*x*z/r**7. * J2*R**2.*GM*(-3. + 7.*z**2./r**2.)

    yy_cf = -GM/r**3. + 3.*GM*y**2./r**5.
    yy_drag = drag*((-y*va*va_y/(H*r)) + dtheta*va_x*va_y/va)
    yy_J2 = -1.5*J2*GM*R**2./r**5. - 7.5*J2*GM*R**2./r**7.*(-y**2. - z**2. + 7.*y**2.*z**2./r**2.)

    yx_cf = xy_cf
    yx_drag = drag*((-x*va*va_y/(H*r)) - dtheta*va_y**2./va - va*dtheta)
    yx_J2 = xy_J2

    yz_cf = 3.*GM*y*z/r**5.
    yz_drag = drag*(-z*va*va_y/(H*r))
    yz_J2 = -7.5*y*z/r**7. * J2*R**2.*GM*(-3. + 7.*z**2./r**2.)

    zz_cf = -GM/r**3. + 3.*GM*z**2./r**5.
    zz_drag = drag*(-z*va*va_z/(H*r))
    zz_J2 = -4.5*J2*R**2.*GM/r**5. - 7.5*J2*R**2.*GM/r**7.*(-6.*z**2. + 7.*z**4./r**2.)

    zx_cf = xz_cf
    zx_drag = drag*((-x*va*va_z/(H*r)) - dtheta*va_y*va_z/va)
    zx_J2 = xz_J2

    zy_cf = yz_cf
    zy_drag = drag*((-y*va*va_z/(H*r)) + dtheta*va_x*va_z/va)
    zy_J2 = yz_J2
    
    
    
    
    # Generate A matrix using partials from above
    A = np.zeros((n,n))

    A[0,3] = 1. 
    A[1,4] = 1. 
    A[2,5] = 1.

    A[3,0] = xx_cf + xx_drag + xx_J2
    A[3,1] = xy_cf + xy_drag + xy_J2
    A[3,2] = xz_cf + xz_drag + xz_J2
    A[3,3] = drag*(va_x**2./va + va)
    A[3,4] = drag*(va_y*va_x/va)
    A[3,5] = drag*(va_z*va_x/va)      # Note, va_z = dz

    A[4,0] = yx_cf + yx_drag + yx_J2
    A[4,1] = yy_cf + yy_drag + yy_J2
    A[4,2] = yz_cf + yz_drag + yz_J2
    A[4,3] = drag*(va_y*va_x/va)
    A[4,4] = drag*(va_y**2./va + va)
    A[4,5] = drag*(va_y*va_z/va)       # Note, va_z = dz

    A[5,0] = zx_cf + zx_drag + zx_J2
    A[5,1] = zy_cf + zy_drag + zy_J2
    A[5,2] = zz_cf + zz_drag + zz_J2
    A[5,3] = drag*(va_x*va_z/va)
    A[5,4] = drag*(va_y*va_z/va)
    A[5,5] = drag*(va_z**2./va + va)
    
    if n > 6:
        A[3,6] = x_drag/beta
        A[4,6] = y_drag/beta
        A[5,6] = z_drag/beta
    

    # Compute STM components dphi = A*phi
    phi_mat = np.reshape(X[n:], (n, n))
    dphi_mat = np.dot(A, phi_mat)
    dphi_v = np.reshape(dphi_mat, (n**2, 1))

    # Derivative vector
    dX = np.zeros(n+n**2,)

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3. + x_drag - 1.5*J2*R**2.*GM*((x/r**5.) - (5.*x*z**2./r**7.))
    dX[4] = -GM*y/r**3. + y_drag - 1.5*J2*R**2.*GM*((y/r**5.) - (5.*y*z**2./r**7.))
    dX[5] = -GM*z/r**3. + z_drag - 1.5*J2*R**2.*GM*((3.*z/r**5.) - (5.*z**3./r**7.))
    
    # If additional states such as beta are included their first derivative
    # is initialized to zero above

    dX[n:] = dphi_v.flatten()

    return dX




#def ode_twobody_j2_drag_ukf(t, X, params):
#    '''
#    This function works with ode to propagate object assuming
#    simple two-body dynamics.  No perturbations included.  States for UKF
#    sigma points included.
#
#    Parameters
#    ------
#    X : (n*(2n+1)) element list
#      initial condition vector of cartesian state and sigma points
#    t : float 
#      current time in seconds
#    params : dictionary
#        additional arguments
#
#    Returns
#    ------
#    dX : (n*(2n+1)) element list
#      derivative vector
#
#    '''
#    
#    # Additional arguments
#    GM = params['GM']
#    
#    # Initialize
#    dX = [0]*len(X)
#    n = int((-1 + np.sqrt(1. + 8.*len(X)))/4.)
#
#    for ind in range(0, 2*n+1):
#
#        # Pull out relevant values from X
#        x = float(X[ind*n])
#        y = float(X[ind*n + 1])
#        z = float(X[ind*n + 2])
#        dx = float(X[ind*n + 3])
#        dy = float(X[ind*n + 4])
#        dz = float(X[ind*n + 5])
#
#        # Compute radius
#        r = np.linalg.norm([x, y, z])
#
#        # Solve for components of dX
#        dX[ind*n] = dx
#        dX[ind*n + 1] = dy
#        dX[ind*n + 2] = dz
#
#        dX[ind*n + 3] = -GM*x/r**3
#        dX[ind*n + 4] = -GM*y/r**3
#        dX[ind*n + 5] = -GM*z/r**3
#
#    return dX
#
#



###############################################################################
# Non-Gaussian Uncertainty Functions
###############################################################################

def ode_aegis(t, X, params):
    '''
    This function propagates the sigma points and entropy of a Gaussian Mixture
    Model per the dynamics model specificied in the input params.
    
    Parameters
    ------
    X : numpy array
        initial condition vector of entropies and cartesian state vectors 
        corresponding to sigma points
    t : float 
        current time in seconds
    params : dictionary
        additional arguments

    Returns
    ------
    dX : numpy array
        derivative vector
      
    Reference
    ------
    DeMars, K.J., Bishop, R.H., Jah, M.K., "Entropy-Based Approach for 
        Uncertainty Propagation of Nonlinear Dynamical Systems," JGCD 2013.
        
    '''
    
    # Function handles
    A_fcn = params['A_fcn']
    dyn_fcn = params['dyn_fcn']
    
    # Retrieve number of states, sigma points, entropies
    nstates = params['nstates']
    npoints = params['npoints']
    ncomp = params['ncomp']
    
    # For each GMM component, there should be 1 entropy, n states, and 2n+1
    # sigma points. Loop over components to compute derivative values
    dX = np.zeros(len(X),)
    for jj in range(ncomp):
        
        # Indices
        nn = npoints*nstates 
        entropy_ind = jj*(nn + 1)
        mean_ind = entropy_ind + 1
        
        # Mean state
        mj = X[mean_ind:mean_ind+nstates]
        
        # Compute A matrix
        A = A_fcn(t, mj, params)
        
        # Compute derivative of entropy (DeMars Eq. 13)
        dX[entropy_ind] = np.trace(A)
        
        # Compute derivatives of states
        Xj = X[mean_ind:mean_ind+nn]
        dXj = dyn_fcn(t, Xj, params)
        dX[mean_ind:mean_ind+nn] = dXj.flatten()
    
    
    return dX


###############################################################################
# Relative Motion Functions
###############################################################################
    

def ode_nonlin_cw(t, X, params):
    
    # Additional arguments
    GM = params['GM']

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])
    dtheta = float(X[6])
    rc = float(X[7])
    drc = float(X[8])
    
    # Deputy orbit radius
    rd = np.sqrt((rc + x)**2. + y**2. + z**2.)

    # Derivative vector
    dX = np.zeros(9,)

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = 2.*dtheta*dy - 2.*dtheta*y*drc/rc + x*dtheta**2. + GM/rc**2. - (GM/rd**3.)*(rc + x)
    dX[4] = -2.*dtheta*dx + 2.*dtheta*x*drc/rc + y*dtheta**2. - (GM/rd**3.)*y
    dX[5] = -(GM/rd**3.)*z
    
    dX[6] = -2.*drc/rc*dtheta
    
    dX[7] = drc
    dX[8] = rc*dtheta**2. - GM/rc**2.
    
    return dX


def ode_nonlin_cw_stm(t, X, params):
    
    # Additional arguments
    GM = params['GM']

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])
    dtheta = float(X[6])
    rc = float(X[7])
    drc = float(X[8])
    
    # Deputy orbit radius
    rd = np.sqrt((rc + x)**2. + y**2. + z**2.)
    
    # A matrix partials
    drd_dx = (rc + x)/rd
    drd_dy = y/rd
    drd_dz = z/rd
    drd_drc = (rc + x)/rd
    
    mu_term = 3.*GM/rd**4.
    
    ddx_dx = dtheta**2. - GM/rd**3. + mu_term*drd_dx*(rc + x)
    ddx_dy = -2.*dtheta*y*drc/rc + mu_term*drd_dy*(rc + x)
    ddx_dz = mu_term*drd_dz*(rc + x)
    ddx_ddx = 0.
    ddx_ddy = 2.*dtheta
    ddx_ddz = 0.
    ddx_ddtheta = 2.*dy - 2.*y*drc/rc + 2.*x*dtheta
    ddx_drc = 2.*dtheta*y*drc/rc**2. - 2.*GM/rc**3. + mu_term*drd_drc*(rc + x)
    ddx_ddrc = -2.*dtheta*y/rc
    
    ddy_dx = 2.*dtheta*drc/rc + mu_term*drd_dx*y
    ddy_dy = dtheta**2. - GM/rd**3. + mu_term*drd_dy*y
    ddy_dz = mu_term*drd_dz*y
    ddy_ddx = -2.*dtheta
    ddy_ddy = 0.
    ddy_ddz = 0.
    ddy_ddtheta = -2.*dx + 2.*x*drc/rc + 2.*y*dtheta
    ddy_drc = -2.*dtheta*x*drc/rc**2. + mu_term*drd_drc*y
    ddy_ddrc = 2.*dtheta*x/rc
    
    ddz_dx = mu_term*drd_dx*z
    ddz_dy = mu_term*drd_dy*z
    ddz_dz = -GM/rd**3. + mu_term*drd_dz*z
    ddz_drc = mu_term*drd_drc*z
    
    
    # Generate A matrix
    A = np.zeros((9, 9))

    A[0,3] = 1.
    A[1,4] = 1.
    A[2,5] = 1.

    A[3,0] = ddx_dx
    A[3,1] = ddx_dy
    A[3,2] = ddx_dz
    A[3,3] = ddx_ddx
    A[3,4] = ddx_ddy
    A[3,5] = ddx_ddz
    A[3,6] = ddx_ddtheta
    A[3,7] = ddx_drc
    A[3,8] = ddx_ddrc
    
    A[4,0] = ddy_dx
    A[4,1] = ddy_dy
    A[4,2] = ddy_dz
    A[4,3] = ddy_ddx
    A[4,4] = ddy_ddy
    A[4,5] = ddy_ddz
    A[4,6] = ddy_ddtheta
    A[4,7] = ddy_drc
    A[4,8] = ddy_ddrc
    
    A[5,0] = ddz_dx
    A[5,1] = ddz_dy
    A[5,2] = ddz_dz
    A[5,7] = ddz_drc
    
    A[6,6] = -2.*drc/rc
    A[6,7] = 2.*drc*dtheta/rc**2.
    A[6,8] = -2.*dtheta/rc
    
    A[7,8] = 1.
    
    A[8,6] = 2.*rc*dtheta
    A[8,7] = dtheta**2. + 2.*GM/rc

    # Compute STM components dphi = A*phi
    phi_mat = np.reshape(X[9:], (9,9))
    dphi_mat = np.dot(A, phi_mat)
    dphi_v = np.reshape(dphi_mat, (9**2, 1))
    

    # Derivative vector
    dX = np.zeros(90,)

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = 2.*dtheta*dy - 2.*dtheta*y*drc/rc + x*dtheta**2. + GM/rc**2. - (GM/rd**3.)*(rc + x)
    dX[4] = -2.*dtheta*dx + 2.*dtheta*x*drc/rc + y*dtheta**2. - (GM/rd**3.)*y
    dX[5] = -(GM/rd**3.)*z
    
    dX[6] = -2.*drc/rc*dtheta
    
    dX[7] = drc
    dX[8] = rc*dtheta**2. - GM/rc**2.
    
    dX[9:] = dphi_v.flatten()
    
    return dX



def ode_lincw(t, X, params):
    
    # Additional arguments
    n = params['mean_motion']

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])
    
    # Derivative vector
    dX = np.zeros(6,)

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = 2.*n*dy + 3.*n**2.*x
    dX[4] = -2.*n*dx
    dX[5] = -n**2.*z
    
    return dX


    
def ode_lincw_stm(t, X, params):
    '''
    This function works with ode to propagate a relative orbit using the 
    linear Clohessy-Wiltshire Equations, assuming simple two-body dynamics.  
    No perturbations included.
    Partials for the STM dynamics are included.

    Parameters
    ------
    X : 42 element array
      initial condition vector of relative orbit state and STM (Hill Frame)
    t : float 
      current time in seconds
    params : dictionary
        additional arguments

    Returns
    ------
    dX : 42 element array
      derivative vector
      
    '''
    
#    print('\nODE function')
#    print('X', X)
    
    # Additional arguments
    n = params['mean_motion']

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])

    # Generate A matrix
    A = np.zeros((6, 6))

    A[0,3] = 1.
    A[1,4] = 1.
    A[2,5] = 1.

    A[3,0] = 3.*n**2.
    A[3,4] = 2.*n

    A[4,0] = -2.*n

    A[5,2] = n**2.

    # Compute STM components dphi = A*phi
    phi_mat = np.reshape(X[6:], (6, 6))
    dphi_mat = np.dot(A, phi_mat)
    dphi_v = np.reshape(dphi_mat, (6**2, 1))

    # Derivative vector
    dX = np.zeros(42,)

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = 2.*n*dy + 3.*n**2.*x
    dX[4] = -2.*n*dx
    dX[5] = -n**2.*z

    dX[6:] = dphi_v.flatten()
    
    
    return dX








