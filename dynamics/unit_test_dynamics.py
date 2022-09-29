import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta

sys.path.append('../')

import dynamics.dynamics_functions as dyn
import utilities.astrodynamics as astro
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
    tin = np.arange(0., 86400.*2+1., 10.)
    
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
    
    int_params['tfirst'] = True
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['step'] = 10.
    int_params['max_step'] = 1000.
    int_params['time_format'] = 'datetime'
    
    # Initial object state vector
    # Sun-Synch Orbit
    Xo = np.reshape([757.700301, 5222.606566, 4851.49977,
                     2.213250611, 4.678372741, -5.371314404], (6,1))
    

    # Propagate several orbit fractions
    elem0 = astro.cart2kep(Xo)
    a = float(elem0[0])
    P = 2.*math.pi*np.sqrt(a**3./GM)
#    fraction_list = [0., 0.2, 0.8, 1.2, 1.8, 10.2, 10.8]
    
    fraction_list = [0., 10.2]
    
    
    tvec = np.asarray([frac*P for frac in fraction_list])
    UTC0 = datetime(1999, 10, 4, 1, 45, 0)
    tin = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    
    
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


if __name__ == '__main__':
    
    plt.close('all')
    
#    unit_test_orbit()
    
#    test_orbit_timestep()
    
    test_hyperbolic_prop()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    