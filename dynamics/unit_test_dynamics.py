import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
import time
import inspect
import os
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

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from dynamics import dynamics_functions as dyn
from dynamics import numerical_integration as numint
from dynamics import fast_integration as fastint
from utilities import astrodynamics as astro
from utilities.constants import GME, J2E, wE, Re


def unit_test_orbit():
    
    
    # Orbit Parameter Setup
    params = {}
    params['GM'] = GME
    params['J2'] = J2E
    params['dtheta'] = wE  # rad/s
    params['R'] = Re  # km
    params['Cd'] = 2.2
    params['A_m'] = 1e-8    # km^2/kg

    # Integration times
#    tin = np.array([0., 86400.*2.])   
    tin = np.arange(0., 86400.*2+1., 10.)
    
    # Initial orbit - Molniya     
#    Xo = np.array([2.88824880e3, -7.73903934e2, -5.97116199e3, 2.64414431,
#                   9.86808092, 0.0])
    
    # Initial orbit - sun-synchronous
    elem0 = [6978.1363, 0.01, 97.79, 30., 30., 0.]
    Xo = astro.kep2cart(elem0, GM=params['GM'])
    
    # Integration function and additional settings
    int_params = {}
#    int_params['integrator'] = 'ode'
#    int_params['ode_integrator'] = 'dop853'
#    int_params['intfcn'] = dyn.ode_twobody_j2_drag 
    
    
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = dyn.ode_twobody_j2_drag
    
    int_params['step'] = 10.
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['local_extrap'] = True
    int_params['time_format'] = 'sec'
    
    # Run integrator
    tout, Xout = dyn.general_dynamics(Xo, tin, params, int_params)
    
    print(len(tout))
    print(tout[-1])
    
    # Analytic TwoBody solution
    elem = astro.cart2kep(Xo, GM=params['GM'])
    a = elem[0]
    e = elem[1]
    i = elem[2]
    RAAN = elem[3]
    w = elem[4]
    theta0 = elem[5]*math.pi/180.
    E0 = astro.true2ecc(theta0, e)
    M0 = astro.ecc2mean(E0, e)
    n = np.sqrt(params['GM']/a**3.)
    
    a_diff = []
    e_diff = []
    i_diff = []
    RAAN_diff = []
    w_diff = []
    theta_diff = []
    energy_diff = []
    pos_diff = []
    vel_diff = []
    
    kk = 0
    for t in tout:
        
        # Compute new mean anomaly [rad]
        M = M0 + n*(t-tout[0])
        while M > 2*math.pi:
            M -= 2*math.pi
        
        # Convert to true anomaly [rad]
        E = astro.mean2ecc(M,e)
        theta = astro.ecc2true(E,e)  
        
        # Convert anomaly angles to deg
        M *= 180./math.pi
        E *= 180./math.pi
        theta *= 180./math.pi
        
        X_true = astro.kep2cart([a,e,i,RAAN,w,theta], GM=params['GM'])
        elem_true = [a,e,i,RAAN,w,theta]
        
        # Convert numeric to elements
        elem_num = astro.cart2kep(Xout[kk,:], GM=params['GM'])
        
        a_diff.append(elem_num[0] - elem_true[0])
        e_diff.append(elem_num[1] - elem_true[1])
        i_diff.append(elem_num[2] - elem_true[2])
        RAAN_diff.append(elem_num[3] - elem_true[3])
        w_diff.append(elem_num[4]-elem_true[4])
        theta_diff.append(elem_num[5] - elem_true[5])
        pos_diff.append(np.linalg.norm(X_true[0:3].flatten() - Xout[kk,0:3].flatten()))
        vel_diff.append(np.linalg.norm(X_true[3:6].flatten() - Xout[kk,3:6].flatten()))
        
        if RAAN_diff[kk] < -180:
            RAAN_diff[kk] += 360.
        if RAAN_diff[kk] > 180:
            RAAN_diff[kk] -= 360.
        

        energy_diff.append(params['GM']/(2*elem_true[0]) - params['GM']/(2*elem_num[0]))
        
        kk += 1
        
        
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tout/3600., energy_diff, 'k.')
    plt.ylabel('Energy [km^2/s^2]')
    plt.title('Size and Shape Parameters')
    plt.subplot(3,1,2)
    plt.plot(tout/3600., np.asarray(a_diff), 'k.')
    plt.ylabel('SMA [km]')
    plt.subplot(3,1,3)
    plt.plot(tout/3600., e_diff, 'k.')
    plt.ylabel('Eccentricity')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tout/3600., i_diff, 'k.')
    plt.ylabel('Inclination [deg]')
    plt.title('Orientation Parameters')
    plt.subplot(3,1,2)
    plt.plot(tout/3600., RAAN_diff, 'k.')
    plt.ylabel('RAAN [deg]')
    plt.subplot(3,1,3)
    plt.plot(tout/3600., w_diff, 'k.')
    plt.ylabel('AoP [deg]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tout/3600., np.asarray(pos_diff), 'k.')
    plt.ylabel('3D Pos [km]')
    plt.title('Position and Velocity')
    plt.subplot(2,1,2)
    plt.plot(tout/3600., np.asarray(vel_diff), 'k.')
    plt.ylabel('3D Vel [km/s]')
    plt.xlabel('Time [hours]')

    
    plt.show()
    
    
    return


def test_hyperbolic_prop():
    
    # Orbit Parameter Setup
    params = {}
    params['GM'] = GME
    params['J2'] = J2E*0
    params['dtheta'] = wE  # rad/s
    params['R'] = Re  # km
    params['Cd'] = 2.2*0
    params['A_m'] = 1e-8*0    # km^2/kg

    # Integration times
#    tin = np.array([0., 86400.*2.])   
    tin = np.arange(0., 86400.*100+1., 10000.)
    
    # Initial orbit - hyperbolic escape
    rp = Re + 300.
    vinf = 1.0
    a = -GME/vinf**2.
    e = 1. + rp*vinf**2./GME
    
    elem0 = [a, e, 10., 20., 30., -10.]    
    Xo = astro.kep2cart(elem0, GM=params['GM'])
    
    # Integration function and additional settings
    int_params = {}    
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = dyn.ode_twobody
    
    int_params['step'] = 10.
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['local_extrap'] = True
    int_params['time_format'] = 'sec'
    
    # Run integrator
    tout, Xout = dyn.general_dynamics(Xo, tin, params, int_params)
    
    print(len(tout))
    print(tout[-1])
    
    # Analytic TwoBody solution
    elem = astro.cart2kep(Xo, GM=params['GM'])
    a = elem[0]
    e = elem[1]
    i = elem[2]
    RAAN = elem[3]
    w = elem[4]
    theta0 = elem[5]*math.pi/180.
    H0 = astro.true2hyp(theta0, e)
    M0 = astro.hyp2mean(H0, e)
    n = np.sqrt(params['GM']/-a**3.)
    
    elem0_M0 = elem0
    elem0_M0[5] = M0*180./math.pi
    
    print('Xo', Xo)
    print('Xo_elemconv', astro.element_conversion(elem0_M0,0,1))
    
    
    a_diff = []
    e_diff = []
    i_diff = []
    RAAN_diff = []
    w_diff = []
    theta_diff = []
    energy_diff = []
    pos_diff = []
    vel_diff = []
    M_plot = []
    H_plot = []
    theta_plot = []
    M_diff = []
    pos_diff2 = []
    vel_diff2 = []
    
    kk = 0
    for t in tout:
        
        # Compute new mean anomaly [rad]
        M = M0 + n*(t-tout[0])
        
        # Convert to true anomaly [rad]
        H = astro.mean2hyp(M,e)
        theta = astro.hyp2true(H,e)  
        
        # Convert anomaly angles to deg
        M *= 180./math.pi
        H *= 180./math.pi
        theta *= 180./math.pi
        
        M_plot.append(M)
        H_plot.append(H)
        theta_plot.append(theta)
        
        X_true = astro.kep2cart([a,e,i,RAAN,w,theta], GM=params['GM'])
        elem_true = [a,e,i,RAAN,w,theta]
        
        # Convert numeric to elements
        elem_num = astro.cart2kep(Xout[kk,:], GM=params['GM'])
        
        a_diff.append(elem_num[0] - elem_true[0])
        e_diff.append(elem_num[1] - elem_true[1])
        i_diff.append(elem_num[2] - elem_true[2])
        RAAN_diff.append(elem_num[3] - elem_true[3])
        w_diff.append(elem_num[4]-elem_true[4])
        theta_diff.append(elem_num[5] - elem_true[5])
        pos_diff.append(np.linalg.norm(X_true[0:3].flatten() - Xout[kk,0:3].flatten()))
        vel_diff.append(np.linalg.norm(X_true[3:6].flatten() - Xout[kk,3:6].flatten()))
        
        if RAAN_diff[kk] < -180:
            RAAN_diff[kk] += 360.
        if RAAN_diff[kk] > 180:
            RAAN_diff[kk] -= 360.
        

        energy_diff.append(params['GM']/(2*elem_true[0]) - params['GM']/(2*elem_num[0]))

        # Use Element Conversion function
#        elem_out = astro.element_conversion(elem0_M0, 0, 0, dt=(t-tout[0]))
#        cart_out = astro.element_conversion(elem0_M0, 0, 1, dt=(t-tout[0]))
        elem_out = astro.element_conversion(Xo, 1, 0, dt=(t-tout[0]))
        cart_out = astro.element_conversion(Xo, 1, 1, dt=(t-tout[0]))
        
        M_diff.append(elem_out[5] - M)
        pos_diff2.append(np.linalg.norm(cart_out[0:3].flatten() - Xout[kk,0:3].flatten()))
        vel_diff2.append(np.linalg.norm(cart_out[3:6].flatten() - Xout[kk,3:6].flatten()))
        
        
        kk += 1
        
        
        
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tout/3600., energy_diff, 'k.')
    plt.ylabel('Energy [km^2/s^2]')
    plt.title('Size and Shape Parameters')
    plt.subplot(3,1,2)
    plt.plot(tout/3600., np.asarray(a_diff), 'k.')
    plt.ylabel('SMA [km]')
    plt.subplot(3,1,3)
    plt.plot(tout/3600., e_diff, 'k.')
    plt.ylabel('Eccentricity')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tout/3600., i_diff, 'k.')
    plt.ylabel('Inclination [deg]')
    plt.title('Orientation Parameters')
    plt.subplot(3,1,2)
    plt.plot(tout/3600., RAAN_diff, 'k.')
    plt.ylabel('RAAN [deg]')
    plt.subplot(3,1,3)
    plt.plot(tout/3600., w_diff, 'k.')
    plt.ylabel('AoP [deg]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tout/3600., M_plot, 'k.')
    plt.ylabel('M [deg]')
    plt.title('Location Parameters')
    plt.subplot(3,1,2)
    plt.plot(tout/3600., H_plot, 'k.')
    plt.ylabel('H [deg]')
    plt.subplot(3,1,3)
    plt.plot(tout/3600., theta_plot, 'k.')
    plt.ylabel('TA [deg]')
    plt.xlabel('Time [hours]')
    
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tout/3600., np.asarray(pos_diff), 'k.')
    plt.ylabel('3D Pos [km]')
    plt.title('Position and Velocity using cart2kep')
    plt.subplot(2,1,2)
    plt.plot(tout/3600., np.asarray(vel_diff), 'k.')
    plt.ylabel('3D Vel [km/s]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tout/3600., np.asarray(pos_diff2), 'k.')
    plt.ylabel('3D Pos [km]')
    plt.title('Position and Velocity using element_conversion')
    plt.subplot(3,1,2)
    plt.plot(tout/3600., np.asarray(vel_diff2), 'k.')
    plt.ylabel('3D Vel [km/s]')
    plt.subplot(3,1,3)
    plt.plot(tout/3600., np.asarray(M_diff), 'k.')
    plt.ylabel('M diff [deg]')    
    plt.xlabel('Time [hours]')

    
    plt.show()
    
    
    return


def test_orbit_timestep():
    
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
    GM = state_params['GM']

    
    # Integration function and additional settings
    int_params = {}
#    int_params['integrator'] = 'ode'
#    int_params['ode_integrator'] = 'dop853'
#    int_params['intfcn'] = dyn.ode_twobody
    
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = dyn.ode_twobody
    
#    int_params['tfirst'] = True
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
#    int_params['step'] = 10.
#    int_params['max_step'] = 1000.
    int_params['time_format'] = 'datetime'
    
    # Initial object state vector
    # Sun-Synch Orbit
#    Xo = np.reshape([757.700301, 5222.606566, 4851.49977,
#                     2.213250611, 4.678372741, -5.371314404], (6,1))
#    elem0 = astro.cart2kep(Xo)
    
#    a = float(elem0[0])
#    P = 2.*math.pi*np.sqrt(a**3./GM)
##    fraction_list = [0., 0.2, 0.8, 1.2, 1.8, 10.2, 10.8]
#    
#    fraction_list = [0., 10.2]
#    
#    
#    tvec = np.asarray([frac*P for frac in fraction_list])
#    UTC0 = datetime(1999, 10, 4, 1, 45, 0)
#    tin = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    
    # QZS-1R
    elem0 = [4.21639888e+04, 7.47880515e-02, 3.48399170e+01, 9.92089475e+01,
            2.70695246e+02, 3.33331109e+02]
    
    Xo = astro.kep2cart(elem0)
    
    UTC0 = datetime(2022, 11, 7, 11, 0, 0)
    UTC1 = datetime(2022, 11, 8, 14, 10, 0)

    tin = [UTC0, UTC1]
    
    
    
    # Run integrator
    tout, Xout = dyn.general_dynamics(Xo, tin, state_params, int_params)

    print(tout)
    print(Xout)
    
    Xnum = Xout[-1,:].reshape(6,1)
    
    # Analytic solution
    dt_sec = (tin[-1] - tin[0]).total_seconds()
    Xtrue = astro.element_conversion(Xo, 1, 1, GME, dt_sec)
    
    print(Xnum)
    print(Xtrue)
    
    err = np.linalg.norm(Xnum - Xtrue)
    print('err', err)
    
    
    return


def test_dopri_computation():
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
    
    # Integration function and additional settings
    int_params = {}    
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = dyn.ode_twobody
    
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['step'] = 10.
    int_params['time_format'] = 'datetime'
    
    # Initial object state vector
    # Sun-Synch Orbit
#    Xo = np.reshape([757.700301, 5222.606566, 4851.49977,
#                     2.213250611, 4.678372741, -5.371314404], (6,1))
    
    # Molniya
    elem0 = [26600., 0.74, 63.4, 90., 270., 10.]
    Xo = astro.kep2cart(elem0)
    
    UTC0 = datetime(1999, 10, 4, 1, 45, 0)
    UTC1 = UTC0 + timedelta(days=2.)
    tin = [UTC0, UTC1]
    
    # Run integrator
    start = time.time()
    tout, Xout = dyn.general_dynamics(Xo, tin, state_params, int_params)
    solve_ivp_time = time.time() - start


    print('\nSolve IVP Results')
    print(tout)
    print(Xout)
    
    Xnum = Xout[-1,:].reshape(6,1)
    
    # Analytic solution
    dt_sec = (tin[-1] - tin[0]).total_seconds()
    Xtrue = astro.element_conversion(Xo, 1, 1, GME, dt_sec)
    
    print(Xnum)
    print(Xtrue)
    
    err = np.linalg.norm(Xnum - Xtrue)
    print('err', err)
    
    
    
    # Setup and run DOPRI87
    int_params['integrator'] = 'dopri87'
    start = time.time()
    tout, Xout = dyn.general_dynamics(Xo, tin, state_params, int_params)
    dopri_time = time.time() - start
    
    
    print('\nDOPRI87 Results')
    print(tout)
    print(Xout)
    
    Xnum = Xout[-1,:].reshape(6,1)
    
    # Analytic solution
    dt_sec = (tin[-1] - tin[0]).total_seconds()
    Xtrue = astro.element_conversion(Xo, 1, 1, GME, dt_sec)
    
    print(Xnum)
    print(Xtrue)
    
    err = np.linalg.norm(Xnum - Xtrue)
    print('err', err)
    
    
    # Setup and run RKF78
    int_params['integrator'] = 'rkf78'
    int_params['local_extrap'] = True
    start = time.time()
    tout, Xout = dyn.general_dynamics(Xo, tin, state_params, int_params)
    rkf_time = time.time() - start
    
    
    print('\nRKF78 Results')
    print(tout)
    print(Xout)
    
    Xnum = Xout[-1,:].reshape(6,1)
    
    # Analytic solution
    dt_sec = (tin[-1] - tin[0]).total_seconds()
    Xtrue = astro.element_conversion(Xo, 1, 1, GME, dt_sec)
    
    print(Xnum)
    print(Xtrue)
    
    err = np.linalg.norm(Xnum - Xtrue)
    print('err', err)
    
    
    
    print('')
    print('solve_ivp_time', solve_ivp_time)
    print('dopri_time', dopri_time)
    print('rkf78_time', rkf_time)
    
    
    
    return


def test_jit_twobody():
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
    
    
    
    # Xo = np.reshape([ 7.03748400133e+06,  3.23805901792e+06,  2.1507241875e+06, -1.46565763e+03,
    #                  -4.09583949e+01,  6.62279761e+03], (6,1)) * 1e-3
    
    # print(Xo)
    
    # Create default body settings for "Earth"
    bodies_to_create = ["Earth"]

    # Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)

    # Create system of bodies (in this case only Earth)
    bodies = environment_setup.create_system_of_bodies(body_settings)
    
    earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
    initial_state = element_conversion.keplerian_to_cartesian_elementwise(
        gravitational_parameter=earth_gravitational_parameter,
        semi_major_axis=7500.0e3,
        eccentricity=0.1,
        inclination=np.deg2rad(85.3),
        argument_of_periapsis=np.deg2rad(235.7),
        longitude_of_ascending_node=np.deg2rad(23.4),
        true_anomaly=np.deg2rad(139.87),
    )
    
    # print(initial_state)
    
    # Xo = np.reshape(initial_state, (6,1))*1e-3
    
    # GEO orbit
    elem = [42164.1, 0.001, 0.1, 90., 0., 0.]
    Xo = np.reshape(astro.kep2cart(elem), (6,1))

    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'rk4'
    int_params['intfcn'] = dyn.ode_twobody
    int_params['step'] = 10.
    int_params['time_format'] = 'datetime'
    
    # Initial object state vector
    # Sun-Synch Orbit
#    Xo = np.reshape([757.700301, 5222.606566, 4851.49977,
#                     2.213250611, 4.678372741, -5.371314404], (6,1))
    
    # Molniya
    # elem0 = [26600., 0.74, 63.4, 90., 270., 10.]
    # Xo = astro.kep2cart(elem0)
    
    # # Time vector
    # UTC1 = datetime(2022, 10, 20, 0, 0, 0)
    # UTC2 = datetime(2022, 10, 22, 0, 0, 0)
    # tvec = [UTC1, UTC2]
    
    UTC0 = datetime(2000, 1, 1, 12, 0, 0)
    UTC1 = datetime(2000, 1, 2, 12, 0, 0)
    tvec = [UTC0, UTC1]
    

    # Run integrator
    start = time.time()
    tout, Xout = dyn.general_dynamics(Xo, tvec, state_params, int_params)
    rk4_time = time.time() - start
    
    print(tout[-1])
    print(Xout[-1])

        
    # Setup for JIT execution
    int_params['integrator'] = 'rk4_jit'
    int_params['intfcn'] = fastint.jit_twobody
    
    start = time.time()
    tout, Xout = dyn.general_dynamics(Xo, tvec, state_params, int_params)
    rk4_jit_time = time.time() - start
    
    print('rk4 time1', rk4_jit_time)
    
    start = time.time()
    tout, Xout = dyn.general_dynamics(Xo, tvec, state_params, int_params)
    rk4_jit_time = time.time() - start
    
    print(tout[-1])
    print(Xout[-1])

    
    # Compute and plot errors
    Xerr = np.zeros(Xout.shape)
    for ii in range(len(tout)):
        X_true = astro.element_conversion(Xo, 1, 1, dt=tout[ii])        
        Xerr[ii,:] = (Xout[ii,:].reshape(6,1) - X_true).flatten()
        
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tout/3600., Xerr[:,0], 'k.')
    plt.ylabel('X Err [km]')
    plt.title('RK4 Errors')
    plt.subplot(3,1,2)
    plt.plot(tout/3600., Xerr[:,1], 'k.')
    plt.ylabel('Y Err [km]')
    plt.subplot(3,1,3)
    plt.plot(tout/3600., Xerr[:,2], 'k.')
    plt.ylabel('Z Err [km]')
    plt.xlabel('Time [hours]')
    
    
    # DOPRI87 Testing
    int_params['integrator'] = 'dopri87'
    int_params['intfcn'] = dyn.ode_twobody
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    
    start = time.time()
    tout, Xout = dyn.general_dynamics(Xo, tvec, state_params, int_params)
    dp87_time = time.time() - start
    
    print(tout[-1])
    print(Xout[-1])
    
    # JIT execution
    int_params['integrator'] = 'dopri87_jit'
    int_params['intfcn'] = fastint.jit_twobody

    start = time.time()
    tout, Xout = dyn.general_dynamics(Xo, tvec, state_params, int_params)
    dp87_jit_time = time.time() - start
    
    print('dp87 time1', dp87_jit_time)
    
    start = time.time()
    tout, Xout = dyn.general_dynamics(Xo, tvec, state_params, int_params)
    dp87_jit_time = time.time() - start
    
    print(tout[-1])
    print(Xout[-1])
    
    # Compute and plot errors
    Xerr = np.zeros(Xout.shape)
    for ii in range(len(tout)):
        X_true = astro.element_conversion(Xo, 1, 1, dt=tout[ii])
        Xerr[ii,:] = (Xout[ii,:].reshape(6,1) - X_true).flatten()
        
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tout/3600., Xerr[:,0], 'k.')
    plt.ylabel('X Err [km]')
    plt.title('DOPRI87 Errors')
    plt.subplot(3,1,2)
    plt.plot(tout/3600., Xerr[:,1], 'k.')
    plt.ylabel('Y Err [km]')
    plt.subplot(3,1,3)
    plt.plot(tout/3600., Xerr[:,2], 'k.')
    plt.ylabel('Z Err [km]')
    plt.xlabel('Time [hours]')
    
    
    print('rk4 time', rk4_time)
    print('rk4 jit time', rk4_jit_time)
    print('dp87 time', dp87_time)
    print('dp87 jit time', dp87_jit_time)
    
    plt.show()
    
    return


def test_tudat_prop():
    
    
    # UTC0 = datetime(2000, 1, 1, 12, 0, 0)
    # UTC1 = datetime(2000, 1, 2, 12, 0, 0)
    # tvec = [UTC0, UTC1]
    
    # Xo = np.reshape([ 7.03748400133e+06,  3.23805901792e+06,  2.1507241875e+06, -1.46565763e+03,
    #                  -4.09583949e+01,  6.62279761e+03], (6,1)) * 1e-3
    
    # print(Xo)
    
    # Create default body settings for "Earth"
    bodies_to_create = ["Earth"]

    # Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)

    # Create system of bodies (in this case only Earth)
    bodies = environment_setup.create_system_of_bodies(body_settings)
    
    earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
    initial_state = element_conversion.keplerian_to_cartesian_elementwise(
        gravitational_parameter=earth_gravitational_parameter,
        semi_major_axis=7500.0e3,
        eccentricity=0.1,
        inclination=np.deg2rad(85.3),
        argument_of_periapsis=np.deg2rad(235.7),
        longitude_of_ascending_node=np.deg2rad(23.4),
        true_anomaly=np.deg2rad(139.87),
    )
    
    # print(initial_state)
    # print(earth_gravitational_parameter)

    # initial_states = np.concatenate((initial_state, initial_state))
    
    # print(initial_states)
    
    # print(initial_states.shape)
    

    
    # Xo = np.reshape(initial_state, (6,1))*1e-3
    
    # print(Xo)
    
    # Xo = np.reshape(initial_states, (12,1))*1e-3
    
    # GEO orbit
    elem = [42164.1, 0.001, 0.1, 90., 0., 0.]
    Xo = np.reshape(astro.kep2cart(elem), (6,1))
    
    
    
    
    # Setup dynamics and coordinate frame models    
    state_params = {}
    state_params['bodies_to_create'] = ['Earth']
    state_params['global_frame_origin'] = 'Earth'
    state_params['global_frame_orientation'] = 'J2000'
    state_params['central_bodies'] = ['Earth']
    state_params['sph_deg'] = 0
    state_params['sph_ord'] = 0
    state_params['mass'] = 400.
    state_params['Cd'] = 0.
    state_params['Cr'] = 0.
    state_params['drag_area_m2'] = 4.
    state_params['srp_area_m2'] = 4.
    
    # Initialize Tudat bodies once to avoid excess memory allocation
    bodies = dyn.initialize_tudat(state_params)
    

    int_params = {}
    int_params['integrator'] = 'tudat'
    int_params['tudat_integrator'] = 'rkf78'
    int_params['step'] = 10.
    int_params['max_step'] = 100.
    int_params['min_step'] = 1.
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'
    
    # Time vector
    tk_list = []
    for hr in range(24):
        UTC = datetime(2021, 6, 21, hr, 0, 0)
        tvec = np.arange(0., 601., 60.)
        tk_list.extend([UTC + timedelta(seconds=ti) for ti in tvec])
    
    X = Xo
    tout = np.zeros(len(tk_list),)
    Xout = np.zeros((len(tk_list), 6))
    for kk in range(len(tk_list)):

        
        if kk > 0:
            tin = [tk_list[kk-1], tk_list[kk]]
            tout2, Xout2 = dyn.general_dynamics(X, tin, state_params, int_params, bodies)
            X = Xout2[-1,:].reshape(6, 1)
        
            tout[kk] = tout2[-1] + tout[kk-1]
        
        Xout[kk,:] = X.flatten()
    
    # tout, Xout = dyn.general_dynamics(Xo, tvec, state_params, int_params)
    
    print(tout)
    print(Xout)
    
    print(Xout[-1])
    
    
    # Compute and plot errors
    Xerr = np.zeros(Xout[:,0:6].shape)
    for ii in range(len(tout)):
        X_true = astro.element_conversion(Xo[0:6], 1, 1, dt=tout[ii])
        Xerr[ii,:] = (Xout[ii,0:6].reshape(6,1) - X_true).flatten()
        
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tout/3600., Xerr[:,0], 'k.')
    plt.ylabel('X Err [km]')
    plt.title('Position Errors')
    plt.subplot(3,1,2)
    plt.plot(tout/3600., Xerr[:,1], 'k.')
    plt.ylabel('Y Err [km]')
    plt.subplot(3,1,3)
    plt.plot(tout/3600., Xerr[:,2], 'k.')
    plt.ylabel('Z Err [km]')
    plt.xlabel('Time [hours]')
    
    
    
    plt.show()
    
    
    return


def test_coord_turn():
    
    wturn = 2.*math.pi/180.
    Xo = np.reshape([ 1000+3.8676, -10, 1500-11.7457, -10, wturn/8 ], (5,1))
    
    state_params = {}
    
    int_params = {}
    int_params['integrator'] = 'rk4'
    int_params['step'] = 0.1
    int_params['intfcn'] = dyn.ode_coordturn
    int_params['time_format'] = 'seconds'
    
    
    print(Xo)
    X = Xo.copy()
    X_num = Xo.copy()
    tk_list = list(range(1,101))
    X_plot = np.zeros((5,len(tk_list)))
    Xnum_plot = np.zeros((5,len(tk_list)))
    for kk in range(len(tk_list)):
        
        t = 1.
        w = X[4]
        
        F = np.zeros((5,5))
        F[0,0] = 1.
        F[0,1] = np.sin(w*t)/w
        F[0,3] = -(1. - np.cos(w*t))/w
        F[1,1] = np.cos(w*t)
        F[1,3] = -np.sin(w*t)
        F[2,1] = (1. - np.cos(w*t))/w
        F[2,2] = 1.
        F[2,3] = np.sin(w*t)/w
        F[3,1] = np.sin(w*t)
        F[3,3] = np.cos(w*t)
        F[4,4] = 1.
        
        X = np.dot(F, X)        
        X_plot[:,kk] = X.flatten()
        
        tout, Xout = dyn.general_dynamics(X_num, [0., 1.], state_params, int_params)
        
        X_num = Xout[-1,:].reshape(5,1)
        Xnum_plot[:,kk] = X_num.flatten()
        
        
        # print(X)
        # print(X_num)
        # print(X_num - X)
        
        # mistake
        
        
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tk_list, X_plot[0,:], 'k.')
    plt.plot(tk_list, Xnum_plot[0,:], 'b.')
    plt.ylabel('X [m]')
    plt.subplot(2,1,2)
    plt.plot(tk_list, X_plot[2,:], 'k.')
    plt.plot(tk_list, Xnum_plot[2,:], 'b.')
    plt.ylabel('Y [m]')
    plt.xlabel('Time [sec]')
    
    plt.figure()
    plt.plot(X_plot[0,:], X_plot[2,:], 'k.')
    plt.plot(Xnum_plot[0,:], Xnum_plot[2,:], 'b.')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.xlim([-2000., 2000])
    plt.ylim([0., 2000.])
    
        
        
    plt.show()
                   
        
    
    
    return


if __name__ == '__main__':
    
    plt.close('all')
    
    # unit_test_orbit()
    
    # test_orbit_timestep()
    
    # test_hyperbolic_prop()
    
    # test_dopri_computation()
    
    # test_jit_twobody()
    
    test_tudat_prop()
    
    # test_coord_turn()
    

#    test, test2, test3, test4, test5 = fastint.test_jit()
#    
#    print(test)
#    print(test2)
#    print(test3)
#    print(test4)
#    print(test5)
    
    
#    # Set up Runge-Kutta matrix
#    A = np.zeros((13,13))
#    A[1,0] = 1./18.
#    A[2,0:2] = [1./48., 1./16.]
#    A[3,0:3] = [1./32., 0., 3./32.]
#    A[4,0:4] = [5./16., 0., -75./64., 75./64.]
#    A[5,0:5] = [3./80., 0., 0., 3./16., 3./20.]
#    A[6,0:6] = [29443841./614563906., 0., 0., 77736538./692538347., 
#               -28693883./1125000000., 23124283./1800000000.]
#    A[7,0:7] = [16016141./946692911., 0., 0., 61564180./158732637., 
#                22789713./633445777., 545815736./2771057229., 
#               -180193667./1043307555.]
#    A[8,0:8] = [39632708./573591083., 0., 0., -433636366./683701615., 
#               -421739975./2616292301., 100302831./723423059., 
#                790204164./839813087., 800635310./3783071287.]
#    A[9,0:9] = [246121993./1340847787., 0., 0., -37695042795./15268766246., 
#               -309121744./1061227803., -12992083./490766935., 
#                6005943493./2108947869., 393006217./1396673457., 
#                123872331./1001029789.]
#    A[10,0:10] = [-1028468189./846180014., 0., 0., 8478235783./508512852., 
#                   1311729495./1432422823., -10304129995./1701304382., 
#                  -48777925059./3047939560., 15336726248./1032824649., 
#                  -45442868181./3398467696., 3065993473./597172653.]
#    A[11,0:11] = [185892177./718116043., 0., 0., -3185094517./667107341., 
#                 -477755414./1098053517., -703635378./230739211., 
#                  5731566787./1027545527., 5232866602./850066563., 
#                 -4093664535./808688257., 3962137247./1805957418., 
#                  65686358./487910083.]
#    A[12,0:12] = [403863854./491063109., 0., 0., -5068492393./434740067., 
#                 -411421997./543043805., 652783627./914296604., 
#                  11173962825./925320556., -13158990841./6184727034., 
#                  3936647629./1978049680., -160528059./685178525., 
#                  248638103./1413531060., 0.]
#    
#    
#    k8 = np.zeros((4,13))
#    
#    
#    k8[:,0] = 2.
#    k8[:,1] = (k8[:,0]*A[1,0]).flatten()
#    k8[:,2] = np.dot(k8[:,0:2],A[2,0:2].T).flatten()
#    k8[:,3] = np.dot(k8[:,0:3],A[3,0:3].T).flatten()
#    
#    print(k8)
#    
#    ktest = np.zeros((4,13))
#    
#    
#    ktest[:,0] = 2.
#    ktest[:,1] = (ktest[:,0]*A[1,0]).flatten()
#    ktest[:,2] = np.dot(ktest.T[0:2,:],A[2,0:2].T).flatten()
#    ktest[:,3] = np.dot(ktest.T[0:3,:],A[3,0:3].T).flatten()
#    
#    print(ktest-k8)
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    