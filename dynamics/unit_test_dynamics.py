import numpy as np
from math import pi
import matplotlib.pyplot as plt
import sys

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
    int_params['integrator'] = 'ode'
    int_params['ode_integrator'] = 'dop853'
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
    theta0 = elem[5]*pi/180.
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
        while M > 2*pi:
            M -= 2*pi
        
        # Convert to true anomaly [rad]
        E = astro.mean2ecc(M,e)
        theta = astro.ecc2true(E,e)  
        
        # Convert anomaly angles to deg
        M *= 180./pi
        E *= 180./pi
        theta *= 180./pi
        
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


if __name__ == '__main__':
    
    plt.close('all')
    
    unit_test_orbit()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    