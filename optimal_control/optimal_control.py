import numpy as np
import math
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import inspect
import os

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from dynamics import dynamics_functions as dyn
from utilities import astrodynamics as astro
from utilities import numerical_methods as nummeth


from utilities.constants import GME


def ocp_linear_cw():
    
    GM = GME
    n = 6
    
    # Initial orbital state
    elem0 = [7000., 0.001, 55., 0., 0., 0.]
    sma = elem0[0]
    mean_motion = np.sqrt(GM/sma**3.)    
    
    # Initial and final relative orbit
    x0 = 0.1
    y0 = 1.5
    z0 = 0.1
    dx0 = 1e-3
    dy0 = 1e-3
    dz0 = 1e-3
    
    xf = 0.
    yf = 0.
    zf = 0.
    dxf = 0.
    dyf = 0.
    dzf = 0.
    
    Xo = np.reshape([x0, y0, z0, dx0, dy0, dz0], (6,1))
    Xf = np.reshape([xf, yf, zf, dxf, dyf, dzf], (6,1))
    
    # Final time
    t0 = datetime(2022, 8, 31, 12, 0, 0)
    tf = datetime(2022, 8, 31, 12, 30, 0)
    
    # Compute B matrix
    B = np.zeros((n,3))
    B[3:6,:] = np.eye(3)    

    # Numerically integrate to get STM from t0 to tf
    phi0 = np.eye(2*n)
    phi0_v = np.reshape(phi0, ((2*n)**2,1))
    
    # Define state parameters
    state_params = {}
    state_params['mean_motion'] = mean_motion
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = ode_lincw_ocp_stm
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'
    
    # Run numerical integration for STM      
    int0 = phi0_v.flatten()
    tin = [t0, tf]
    
    tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)

    # Extract values for later calculations
    xout = intout[-1,:]
    phi_v = xout.reshape((2*n)**2, 1)
    phi = np.reshape(phi_v, (2*n, 2*n))
    
    # Partition STM
    phi_xx = phi[0:6,0:6]
    phi_xp = phi[0:6,6:12]
    phi_px = phi[6:12,0:6]
    phi_pp = phi[6:12,6:12]
    
    # Solve for initial costate vector 
    phi_xp_inv = np.linalg.inv(phi_xp)
    p0 = np.dot(phi_xp_inv, (Xf - np.dot(phi_xx, Xo)))
    
    print(p0)
    
    # Run numerical integration for full system
    int_params['intfcn'] = ode_lincw_ocp
    int0 = np.concatenate((Xo, p0), axis=0)    
    
    dt_total = (tf - t0).total_seconds()
    tk_list = [t0 + timedelta(seconds=ti) for ti in np.linspace(0., dt_total, 1000)]

    tout, intout = dyn.general_dynamics(int0, tk_list, state_params, int_params)

    # Extract states and plot
    xt = intout[:,0]
    yt = intout[:,1]
    zt = intout[:,2]
    dxt = intout[:,3]
    dyt = intout[:,4]
    dzt = intout[:,5]
    
    u_mat = -np.dot(intout[:,6:12], B)
    ux = u_mat[:,0]
    uy = u_mat[:,1]
    uz = u_mat[:,2]
    
    print('Final State', intout[-1,0:6])
    
    
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tout, xt*1000., 'k.')
    plt.ylabel('Radial [m]')
    plt.title('Relative Position')
    plt.subplot(3,1,2)
    plt.plot(tout, yt*1000., 'k.')
    plt.ylabel('In-Track [m]')
    plt.subplot(3,1,3)
    plt.plot(tout, zt*1000., 'k.')
    plt.ylabel('Cross-Track [m]')
    plt.xlabel('Time [sec]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tout, dxt*1000., 'k.')
    plt.ylabel('Radial [m/s]')
    plt.title('Relative Velocity')
    plt.subplot(3,1,2)
    plt.plot(tout, dyt*1000., 'k.')
    plt.ylabel('In-Track [m/s]')
    plt.subplot(3,1,3)
    plt.plot(tout, dzt*1000., 'k.')
    plt.ylabel('Cross-Track [m/s]')
    plt.xlabel('Time [sec]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tout, ux*1000., 'k.')
    plt.ylabel('Radial [m/s^2]')
    plt.title('Control Accelerations')
    plt.subplot(3,1,2)
    plt.plot(tout, uy*1000., 'k.')
    plt.ylabel('In-Track [m/s^2]')
    plt.subplot(3,1,3)
    plt.plot(tout, uz*1000., 'k.')
    plt.ylabel('Cross-Track [m/s^2]')
    plt.xlabel('Time [sec]')
    
    
    plt.show()
    
    
    
    return


def ode_lincw_ocp(t, X, params):

    mean_motion = params['mean_motion']
    
    # Number of states
    n = 6
    
    # Retrieve states
    x_vect = X[0:n].reshape(n,1)
    p_vect = X[n:2*n].reshape(n,1)

    # Generate A matrix
    A = np.zeros((n, n))
    A[0,3] = 1.
    A[1,4] = 1.
    A[2,5] = 1.
    
    A[3,0] = 3.*mean_motion**2.
    A[3,4] = 2.*mean_motion
    A[4,3] = -2.*mean_motion
    A[5,2] = -mean_motion**2.
    
    # Compute B matrix
    B = np.zeros((n,3))
    B[3:6,:] = np.eye(3)
    
    # Compute controls
    u_vect = -np.dot(B.T, p_vect)
    
    # State and costate derivatives
    dx_vect = np.dot(A, x_vect) + np.dot(B, u_vect)
    dp_vect = -np.dot(A.T, p_vect)

    # Compute derivative vector
    dX = np.zeros(2*n,)
    dX[0:n] = dx_vect.flatten()
    dX[n:2*n] = dp_vect.flatten()

    return dX


def ode_lincw_ocp_stm(t, X, params):

    mean_motion = params['mean_motion']
    
    # Number of states
    n = 6

    # Generate A matrix
    A = np.zeros((n, n))
    A[0,3] = 1.
    A[1,4] = 1.
    A[2,5] = 1.
    
    A[3,0] = 3.*mean_motion**2.
    A[3,4] = 2.*mean_motion
    A[4,3] = -2.*mean_motion
    A[5,2] = -mean_motion**2.
    
    # Compute B matrix
    B = np.zeros((n,3))
    B[3:6,:] = np.eye(3)
    
    # Compute augmented system A matrix
    A_til = np.zeros((2*n,2*n))
    A_til[0:6,0:6] = A
    A_til[0:6,6:12] = -np.dot(B, B.T)
    A_til[6:12,6:12] = -A.T


    # Compute STM components dphi = A_til*phi
    phi_mat = np.reshape(X, (2*n, 2*n))
    dphi_mat = np.dot(A_til, phi_mat)
    dphi_v = np.reshape(dphi_mat, ((2*n)**2, 1))

    # Derivative vector
    dX = dphi_v.flatten()

    return dX


def ocp_twobody():
    
    GM = GME
    n = 6
    
    # Times
    t0 = datetime(2022, 8, 30, 12, 0, 0)
    tf = datetime(2022, 8, 31, 0, 0, 0)
    dt_sec = (tf - t0).total_seconds()
    
    
    # GEO Test Case - North-South stationkeeping, fix inclination    
    # Initial orbital state
    elem0 = np.reshape([42164.1, 0.001, 1., 0., 0., 0.], (6,1))
    Xo = astro.kep2cart(elem0)
    
    # Propagate and reset inclination to zero
    elemf = astro.element_conversion(elem0, 0, 0, dt=dt_sec)
    elemf[2] = 0.001
    Xf = astro.kep2cart(elemf)
    
    print(elem0)
    print(elemf)

    
    
    
    
    return


#def single_shooting_cannon():
#    
#    
#    # Parameters
#    state_params = {}
#    state_params['g'] = 9.81
#    
#    # Integration function and additional settings
#    int_params = {}
#    int_params['integrator'] = 'solve_ivp'
#    int_params['ode_integrator'] = 'DOP853'
#    int_params['intfcn'] = ode_cannon
#    int_params['rtol'] = 1e-12
#    int_params['atol'] = 1e-12
#    int_params['time_format'] = 'seconds'
#    
#    # Times
#    t0 = 0.
#    tf = 10.
#    tin = [t0, tf]
#    
#    # Final state (boundary condition)
#    xf = 1000.
#    yf = 0.
#    Xf = np.array([[xf], [yf]])
#    
#    # Guess initial condition
#    vx0 = 10.
#    vy0 = 10.
#    
#    
#    
#    
#    # Setup loop
#    finite_diff_step = 1e-6
#    err = 1.
#    tol = 1e-8
#    iters = 0
#    maxiters = 100
#    while err > tol:
#        
#        # Compute differentials for finite difference method
#        Xf_num = compute_shooting_error(vx0, vy0, tin, state_params, int_params)
#        c_vect = Xf_num - Xf        
#        
#        vx0_minus = vx0 - vx0*finite_diff_step
#        Xf_num = compute_shooting_error(vx0_minus, vy0, tin, state_params, int_params)
#        cm1 = Xf_num - Xf
#        
#        vx0_plus = vx0 + vx0*finite_diff_step
#        Xf_num = compute_shooting_error(vx0_plus, vy0, tin, state_params, int_params)
#        cp1 = Xf_num - Xf
#        
#        vy0_minus = vy0 - vy0*finite_diff_step
#        Xf_num = compute_shooting_error(vx0, vy0_minus, tin, state_params, int_params)
#        cm2 = Xf_num - Xf
#        
#        vy0_plus = vy0 + vy0*finite_diff_step
#        Xf_num = compute_shooting_error(vy0, vy0_plus, tin, state_params, int_params)
#        cp2 = Xf_num - Xf
#            
#        # Compute first derivative matrix
#        dc1_dx = (float(cp1[0]) - float(cm1[0]))/(2.*vx0*finite_diff_step)
#        dc1_dy = (float(cp1[1]) - float(cm1[1]))/(2.*vy0*finite_diff_step)
#        dc2_dx = (float(cp2[0]) - float(cm2[0]))/(2.*vx0*finite_diff_step)
#        dc2_dy = (float(cp2[1]) - float(cm2[1]))/(2.*vy0*finite_diff_step)
#        
#        mat = np.array([[dc1_dx, dc1_dy],
#                        [dc2_dx, dc2_dy]])
#    
#        delta_vect = -np.dot(np.linalg.inv(mat), c_vect)
#        
#        vx0 += float(delta_vect[0])
#        vy0 += float(delta_vect[1])
#        
#        err = np.linalg.norm(c_vect)
#        
#        print('')
#        print('iters', iters)
#        print('c_vect', c_vect)
#        print('delta_vect', delta_vect)
#        
#        print('err', err)
#        
#        iters += 1
#        if iters > maxiters:
#            break
#        
#        
#        
#    print('\nFinal Values')
#    print('vx0 [m/s]', vx0)
#    print('vy0 [m/s]', vy0)
#    print('v0 [m/s]', np.sqrt(vx0**2. + vy0**2.))
#    print('theta [deg]', math.atan2(vy0, vx0)*180./math.pi)
#        
#    # Propagate final parameters
#    tin = np.linspace(t0, tf, 1000)
#    int0 = np.array([0., 0., vx0, vy0])
#    
#    tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)
#        
#    xt = intout[:,0]
#    yt = intout[:,1]
#    dxt = intout[:,2]
#    dyt = intout[:,3]
#
#
#    plt.figure()
#    plt.subplot(2,1,1)
#    plt.plot(tout, xt, 'k.')
#    plt.ylabel('x [m]')
#    plt.subplot(2,1,2)
#    plt.plot(tout, yt, 'k.')
#    plt.ylabel('y [m]')
#    plt.xlabel('Time [sec]')
#    
#    plt.figure()
#    plt.subplot(2,1,1)
#    plt.plot(tout, dxt, 'k.')
#    plt.ylabel('dx [m/s]')
#    plt.subplot(2,1,2)
#    plt.plot(tout, dyt, 'k.')
#    plt.ylabel('dy [m/s]')
#    plt.xlabel('Time [sec]')
#    
#    plt.show()
#    
#    
#    return


def single_shooting_cannon():
    
    # Parameters
    state_params = {}
    state_params['g'] = 9.81
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = ode_cannon
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'seconds'
    
    # Error calculation
    boundary_fcn = compute_cannon_error
    
    # Times
    t0 = 0.
    tf = 10.
    tin = [t0, tf]
    
    # Final state (boundary condition)
    xf = 1000.
    yf = 0.
    Xf = np.array([[xf], [yf]])
    
    # Guess initial condition
    vx0 = 10.
    vy0 = 10.
    Xo_init = np.array([[vx0], [vy0]])
    
    
    Xo, fail_flag = \
        nummeth.single_shooting(Xo_init, Xf, tin, boundary_fcn, state_params, int_params)
        
        
    vx0 = float(Xo[0])
    vy0 = float(Xo[1])
    
    
    print('\nFinal Values')
    print('fail_flag', fail_flag)
    print('vx0 [m/s]', vx0)
    print('vy0 [m/s]', vy0)
    print('v0 [m/s]', np.sqrt(vx0**2. + vy0**2.))
    print('theta [deg]', math.atan2(vy0, vx0)*180./math.pi)
        
    # Propagate final parameters
    tin = np.linspace(t0, tf, 1000)
    int0 = np.array([0., 0., vx0, vy0])
    
    tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)
        
    xt = intout[:,0]
    yt = intout[:,1]
    dxt = intout[:,2]
    dyt = intout[:,3]


    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tout, xt, 'k.')
    plt.ylabel('x [m]')
    plt.subplot(2,1,2)
    plt.plot(tout, yt, 'k.')
    plt.ylabel('y [m]')
    plt.xlabel('Time [sec]')
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tout, dxt, 'k.')
    plt.ylabel('dx [m/s]')
    plt.subplot(2,1,2)
    plt.plot(tout, dyt, 'k.')
    plt.ylabel('dy [m/s]')
    plt.xlabel('Time [sec]')
    
    plt.show()
    
    return


def single_shooting_cannon2():
    
    # Parameters
    state_params = {}
    state_params['g'] = 9.81
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = ode_cannon
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'seconds'
    
    # Error calculation
    boundary_fcn = compute_cannon_error2
    
    # Times
    t0 = 0.
    tf = 10.
    tin = [t0, tf]
    
    # Initial and final state (boundary condition)
    x0 = 0.
    y0 = 0.
    xf = 1000.
    yf = 0.
    bc_vect = np.reshape([x0, y0, xf, yf], (4,1))
    
    # Guess initial condition
    Xo_init = np.reshape([0., 0., 10., 10.], (4,1))
    
    
    Xo, fail_flag = \
        nummeth.single_shooting(Xo_init, bc_vect, tin, boundary_fcn, state_params, int_params)
        
        
    x0 = float(Xo[0,0])
    y0 = float(Xo[1,0])
    vx0 = float(Xo[2,0])
    vy0 = float(Xo[3,0])
    
    
    print('\nFinal Values')
    print('fail_flag', fail_flag)
    print('vx0 [m/s]', vx0)
    print('vy0 [m/s]', vy0)
    print('v0 [m/s]', np.sqrt(vx0**2. + vy0**2.))
    print('theta [deg]', math.atan2(vy0, vx0)*180./math.pi)
        
    # Propagate final parameters
    tin = np.linspace(t0, tf, 1000)
    int0 = np.array([x0, y0, vx0, vy0])
    
    tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)
        
    xt = intout[:,0]
    yt = intout[:,1]
    dxt = intout[:,2]
    dyt = intout[:,3]


    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tout, xt, 'k.')
    plt.ylabel('x [m]')
    plt.subplot(2,1,2)
    plt.plot(tout, yt, 'k.')
    plt.ylabel('y [m]')
    plt.xlabel('Time [sec]')
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tout, dxt, 'k.')
    plt.ylabel('dx [m/s]')
    plt.subplot(2,1,2)
    plt.plot(tout, dyt, 'k.')
    plt.ylabel('dy [m/s]')
    plt.xlabel('Time [sec]')
    
    plt.show()
    
    return



def multiple_shooting_cannon():
    
    # Parameters
    state_params = {}
    state_params['g'] = 9.81
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = ode_cannon
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'seconds'
    
    # Error calculation
    cvect_fcn = compute_cannon_cvect2
    
    # Times
    t0 = 0.
    tf = 10.
#    tin = [t0, tf]
    tin = np.array([t0, (tf+t0)/2., tf])
    
    # Initial and final state (boundary condition)
    x0 = 0.
    y0 = 0.
    xf = 1000.
    yf = 0.
    bc_vect = np.reshape([x0, y0, xf, yf], (4,1))
    
    # Guess initial condition
#    Xo_init = np.reshape([0.1, 0.1, 100.1, 40., 501., 101., 100.1, 0.1], (8,1))
    Xo_init = np.ones((8,1))
    
    
    Xo, fail_flag = \
        nummeth.multiple_shooting(Xo_init, bc_vect, tin, cvect_fcn, state_params, int_params)
        
        
    x0 = float(Xo[0,0])
    y0 = float(Xo[1,0])
    vx0 = float(Xo[2,0])
    vy0 = float(Xo[3,0])
    
    
    print('\nFinal Values')
    print('fail_flag', fail_flag)
    print('vx0 [m/s]', vx0)
    print('vy0 [m/s]', vy0)
    print('v0 [m/s]', np.sqrt(vx0**2. + vy0**2.))
    print('theta [deg]', math.atan2(vy0, vx0)*180./math.pi)
        
    # Propagate final parameters
    tin = np.linspace(t0, tf, 1000)
    int0 = np.array([x0, y0, vx0, vy0])
    
    tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)
        
    xt = intout[:,0]
    yt = intout[:,1]
    dxt = intout[:,2]
    dyt = intout[:,3]


    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tout, xt, 'k.')
    plt.ylabel('x [m]')
    plt.subplot(2,1,2)
    plt.plot(tout, yt, 'k.')
    plt.ylabel('y [m]')
    plt.xlabel('Time [sec]')
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tout, dxt, 'k.')
    plt.ylabel('dx [m/s]')
    plt.subplot(2,1,2)
    plt.plot(tout, dyt, 'k.')
    plt.ylabel('dy [m/s]')
    plt.xlabel('Time [sec]')
    
    plt.show()
    
    return


def multiple_shooting_yt():
    
    # Parameters
    state_params = {}
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = ode_yt
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'seconds'
    
    # Error calculation
    cvect_fcn = compute_yt_cvect2
    
    # Times
    t0 = 0.
    tf = 2.
#    tin = [t0, tf]
    tin = np.array([t0, (tf+t0)/2., tf])
    
    # Initial and final state (boundary condition)
    yf = 10.
    bc_vect = np.array([yf])
    
    # Guess initial condition
    Xo_init = np.reshape([0.1, 0.1], (2,1))
    
    
    Xo, fail_flag = \
        nummeth.multiple_shooting(Xo_init, bc_vect, tin, cvect_fcn, state_params, int_params)
        
        
    
    y0 = float(Xo[0])
    
    
    print('\nFinal Values')
    print('fail_flag', fail_flag)
    print('y0', y0)
        
    # Propagate final parameters
    tin = np.linspace(t0, tf, 100)
    int0 = np.array([y0])
    
    tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)
        
    yt = intout[:,0]


    plt.figure()
    plt.plot(tout, yt, 'k.')
    plt.ylabel('y')
    plt.xlabel('Time [sec]')
    
    plt.show()
    
    return


def multiple_shooting_hohmann_transfer():
    
    # Parameters
    state_params = {}
    state_params['GM'] = GME
    
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = dyn.ode_twobody
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'seconds'
    
    # Error calculation
    cvect_fcn = cvect_hohmann_transfer
    
    # Initial and final state (boundary condition)
    # Initial condition corresponds to object at the ascending node with
    # RAAN = 180 deg, AoP = 0, TA = 0 in a circular LEO orbit at 28.5 deg
    # inclination.
    p0 = 6655.94200010410805    # km
    f0 = 0.
    g0 = 0.
    h0 = -0.25396764647494
    k0 = 0.
    L0 = 180.    # deg
    
    # Final orbit is Geostationary, location unspecified
    pf = 42164.1    # km
    ff = 0.
    gf = 0.
    hf = 0.
    kf = 0.
    

    bc_vect = np.reshape([p0, f0, g0, h0, k0, L0, pf, ff, gf, hf, kf], (11,1))
    
    # Guess initial condition
#    Xo_init = np.ones((8,1))
    Xo_init = np.reshape([1000., 1., 1., 1., 2000., 1., 1., 1.], (8,1))
    tin = [0.]
    
    Xo, fail_flag = \
        nummeth.multiple_shooting(Xo_init, bc_vect, tin, cvect_fcn, state_params, int_params)
        
    
    # Retrieve values   
    t1 = float(Xo[0])
    dv1x = float(Xo[1])
    dv1y = float(Xo[2])
    dv1z = float(Xo[3])
    t2 = float(Xo[4])
    dv2x = float(Xo[5])
    dv2y = float(Xo[6])
    dv2z = float(Xo[7])
    
    
    print('\nFinal Values')
    print('fail_flag', fail_flag)
    print('t1 [sec]', t1)
    print('dv1x', dv1x)
    print('dv1y', dv1y)
    print('dv1z', dv1z)
    print('dv1 [km/s]', np.linalg.norm([dv1x, dv1y, dv1z]))
    print('t2 [sec]', t2)
    print('dv2x', dv2x)
    print('dv2y', dv2y)
    print('dv2z', dv2z)
    print('dv2 [km/s]', np.linalg.norm([dv2x, dv2y, dv2z]))
        
    
    
    return


def cvect_hohmann_transfer(Xo, bc_vect, tin, state_params, int_params):
    
    # Retrieve initial boundary condition
    modeqn0 = bc_vect[0:6]
    cart0 = astro.modeqn2cart(modeqn0)
        
    # Retrieve variable parameters  
    t1 = float(Xo[0])
    dv1x = float(Xo[1])
    dv1y = float(Xo[2])
    dv1z = float(Xo[3])
    t2 = float(Xo[4])
    dv2x = float(Xo[5])
    dv2y = float(Xo[6])
    dv2z = float(Xo[7])
    
    # Set up first numerical integration - coast    
    t0 = tin[0]
    tin1 = [t0, t1]
    int0 = cart0.flatten()
    tout, intout = dyn.general_dynamics(int0, tin1, state_params, int_params)
    
    # Retrieve cartesian state at t1 and add delta-v vector
    cart1 = intout[-1,:].reshape(6,1)
    cart1[3] += dv1x
    cart1[4] += dv1y
    cart1[5] += dv1z
    
    # Run second numerical integration - transfer orbit    
    tin2 = [t1, t2]
    int0 = cart1.flatten()
    tout, intout = dyn.general_dynamics(int0, tin2, state_params, int_params)
    
    # Retrieve cartesian state at t2 and add delta-v vector
    cart2 = intout[-1,:].reshape(6,1)
    cart2[3] += dv2x
    cart2[4] += dv2y
    cart2[5] += dv2z
    
    # Compute modeqn and compare against final boundary conditions
    kep = astro.cart2kep(cart2)
    modeqn2 = astro.cart2modeqn(cart2)

    # Extract values for output
    c_vect = modeqn2[0:5] - bc_vect[6:11]
    
    print('bc_vect', bc_vect)
    print('c_vect', c_vect)
    print('kep', kep)
    
    dv1 = np.linalg.norm([dv1x, dv1y, dv1z])
    dv2 = np.linalg.norm([dv2x, dv2y, dv2z])
    F = dv1 + dv2
    
    # Compute g = dF/dx
    
    
    
    

    return c_vect



def compute_cannon_error(Xo, tin, state_params, int_params):
    
    # Run numerical integration   
#    vx0 = v0*np.cos(theta*math.pi/180.)
#    vy0 = v0*np.sin(theta*math.pi/180.)
    vx0 = float(Xo[0])
    vy0 = float(Xo[1])
    int0 = np.array([0., 0., vx0, vy0])
    
    tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)

    # Extract values for later calculations
    Xf_num = intout[-1,0:2].reshape(2,1)
    
    return Xf_num


def compute_cannon_error2(Xo, tin, state_params, int_params):
    
    # Run numerical integration   
    x0 = float(Xo[0])
    y0 = float(Xo[1])
    vx0 = float(Xo[2])
    vy0 = float(Xo[3])
    int0 = np.array([x0, y0, vx0, vy0])
    
    tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)

    # Extract values for output
    bc_num = np.zeros((4,1))
    bc_num[0] = x0
    bc_num[1] = y0
    bc_num[2] = intout[-1,0]
    bc_num[3] = intout[-1,1]
    
    return bc_num



def compute_cannon_cvect(Xo, bc_vect, tin, state_params, int_params):
    
    # Run numerical integration   
    x0 = float(Xo[0])
    y0 = float(Xo[1])
    vx0 = float(Xo[2])
    vy0 = float(Xo[3])

    int0 = np.array([x0, y0, vx0, vy0])
    
    tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)

    
    xf = float(intout[-1,0])
    yf = float(intout[-1,1])

    # Extract values for output
    c_vect = np.zeros((4,1))

    c_vect[0] = x0 - float(bc_vect[0])
    c_vect[1] = y0 - float(bc_vect[1])
    c_vect[2] = xf - float(bc_vect[2])
    c_vect[3] = yf - float(bc_vect[3])
    

    return c_vect


def compute_cannon_cvect2(Xo, bc_vect, tin, state_params, int_params):
    
    # Run numerical integration   
    x0 = float(Xo[0])
    y0 = float(Xo[1])
    vx0 = float(Xo[2])
    vy0 = float(Xo[3])
    x1 = float(Xo[4])
    y1 = float(Xo[5])
    vx1 = float(Xo[6])
    vy1 = float(Xo[7])
    
    
    tin1 = [tin[0], tin[1]]
    int0 = np.array([x0, y0, vx0, vy0])
    tout, intout = dyn.general_dynamics(int0, tin1, state_params, int_params)
    
    x1_bar = float(intout[-1,0])
    y1_bar = float(intout[-1,1])
    vx1_bar = float(intout[-1,2])
    vy1_bar = float(intout[-1,3])
    
    print('x1_bar', x1_bar)
    print('y1_bar', y1_bar)
    print('vx1_bar', vx1_bar)
    print('vy1_bar', vy1_bar)
    
    tin2 = [tin[1], tin[2]]
    int0 = np.array([x1, y1, vx1, vy1])
    tout, intout = dyn.general_dynamics(int0, tin2, state_params, int_params)
    
    xf = float(intout[-1,0])
    yf = float(intout[-1,1])
    
    print('xf', xf)
    print('yf', yf)

    # Extract values for output
    c_vect = np.zeros((8,1))
    c_vect[0] = x1 - x1_bar
    c_vect[1] = y1 - y1_bar
    c_vect[2] = vx1 - vx1_bar
    c_vect[3] = vy1 - vy1_bar
    c_vect[4] = x0 - float(bc_vect[0])
    c_vect[5] = y0 - float(bc_vect[1])
    c_vect[6] = xf - float(bc_vect[2])
    c_vect[7] = yf - float(bc_vect[3])
    
    print('bc_vect', bc_vect)
    print('c_vect', c_vect)
    

    return c_vect


def compute_yt_cvect(Xo, bc_vect, tin, state_params, int_params):
    
    # Run numerical integration   
    y0 = float(Xo[0])
    
    int0 = np.array([y0])
    tout, intout = dyn.general_dynamics(int0, tin, state_params, int_params)
    
    yf = float(intout[-1,0])
    
    c_vect = np.zeros((1,1))
    c_vect[0] = yf - float(bc_vect[0])
    
    
    return c_vect


def compute_yt_cvect2(Xo, bc_vect, tin, state_params, int_params):
    
    # Run numerical integration   
    y0 = float(Xo[0])
    y1 = float(Xo[1])
    
    int0 = np.array([y0])
    tin1 = [tin[0], tin[1]]
    tout, intout = dyn.general_dynamics(int0, tin1, state_params, int_params)
    
    y1_bar = float(intout[-1,0])
    
    int0 = np.array([y1])
    tin2 = [tin[1], tin[2]]
    tout, intout = dyn.general_dynamics(int0, tin2, state_params, int_params)
    
    
    yf = float(intout[-1,0])
    
    c_vect = np.zeros((2,1))
    c_vect[0] = y1 - y1_bar
    c_vect[1] = yf - float(bc_vect[0])
    
    
    return c_vect


def ode_cannon(t, X, params):
    
    g = params['g']
    
    N = int(len(X)/4)
    dX = np.zeros(len(X),)
    
    for ii in range(N):
        x = X[4*ii + 0]
        y = X[4*ii + 1]
        dx = X[4*ii + 2]
        dy = X[4*ii + 3]
        
        dX[4*ii + 0] = dx
        dX[4*ii + 1] = dy
        dX[4*ii + 2] = 0.
        dX[4*ii + 3] = -g
    
    return dX


def ode_yt(t, X, params):
    
    dX = X
    
    return dX



if __name__ == '__main__':
    
    plt.close('all')
    
#    ocp_linear_cw()
    
#    ocp_twobody()
    
    # single_shooting_cannon2()
    
    multiple_shooting_cannon()
    
#    multiple_shooting_yt()
    
    # multiple_shooting_hohmann_transfer()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    