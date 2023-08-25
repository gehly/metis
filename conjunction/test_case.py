import numpy as np
import math
import matplotlib.pyplot as plt
# import os
# import sys
# import inspect


# filename = inspect.getframeinfo(inspect.currentframe()).filename
# current_dir = os.path.dirname(os.path.abspath(filename))

# ind = current_dir.find('metis')
# metis_dir = current_dir[0:ind+5]
# sys.path.append(metis_dir)

from utilities import astrodynamics as astro
from utilities import coordinate_systems as coord
from utilities.constants import Re, GME

def compute_initial_conditions(elem_chief, dt_vect, GM=GME):
    
    # Compute mean motion of chief
    a = float(elem_chief[0])
    n = astro.sma2meanmot(a, GM)
    
    # Set up initial conditions in RIC frame
    x0 = 0.
    y0 = -1.
    z0 = 0.
    dx0 = -2.0
    dy0 = 0.500
    dz0 = 1.0
    rho_ric = np.reshape([x0, y0, z0], (3,1))
    drho_ric = np.reshape([dx0, dy0, dz0], (3,1))
    
    x_off = 2*dy0/n
    d = dy0 + 2*n*x0
    rho0 = np.sqrt(x0**2 + y0**2 + z0**2)
    
    print('d', d)
    print('x_off', x_off)
    print('rho0', rho0)
    
    # Convert to ECI and compute deputy orbit    
    Xo_chief = astro.kep2cart(elem_chief, GM)
    rc_vect = Xo_chief[0:3].reshape(3,1)
    vc_vect = Xo_chief[3:6].reshape(3,1)
    rho_eci = coord.ric2eci(rc_vect, vc_vect, rho_ric)
    drho_eci = coord.ric2eci_vel(rc_vect, vc_vect, rho_ric, drho_ric)
    
    print(rho_eci)
    print(drho_eci)
    print(np.linalg.norm(rho_eci))
    print(np.linalg.norm(drho_eci))
    
    rd_vect = rc_vect + rho_eci
    vd_vect = vc_vect + drho_eci
    
    Xo_deputy = np.concatenate((rd_vect, vd_vect), axis=0)
    elem_deputy = astro.cart2kep(Xo_deputy, GM)
    
    print(Xo_deputy)
    print(elem_deputy)
    
    # Back and forward propagate, compute differences, and plot
    rho_plot = np.zeros(dt_vect.shape)
    r_plot = np.zeros(dt_vect.shape)
    i_plot = np.zeros(dt_vect.shape)
    c_plot = np.zeros(dt_vect.shape)
    ii = 0
    for dt in dt_vect:
        Xt_chief = astro.element_conversion(Xo_chief, 1, 1, GM, dt)
        Xt_deputy = astro.element_conversion(Xo_deputy, 1, 1, GM, dt)
        
        if ii == 0:
            Xc_output = Xt_chief.copy()
            Xd_output = Xt_deputy.copy()
        
        rc_t = Xt_chief[0:3].reshape(3,1)
        vc_t = Xt_chief[3:6].reshape(3,1)
        rd_t = Xt_deputy[0:3].reshape(3,1)
        
        rho_eci = rd_t - rc_t
        rho_ric = coord.eci2ric(rc_t, vc_t, rho_eci)
        
        rho_plot[ii] = np.linalg.norm(rho_eci)
        r_plot[ii] = float(rho_ric[0])
        i_plot[ii] = float(rho_ric[1])
        c_plot[ii] = float(rho_ric[2])
        
        ii += 1
        
    rho_min = min(rho_plot)
    ind = list(rho_plot).index(rho_min)
    tmin = dt_vect[ind]
    
    print(rho_min)
    print(tmin)
    print(rho_plot[ind-3:ind+4])
        
        
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(dt_vect, rho_plot, 'k.')
    plt.plot(tmin, rho_min, 'ro')
    plt.ylabel('rho [km]')
    plt.subplot(4,1,2)
    plt.plot(dt_vect, r_plot, 'k.')
    plt.ylabel('r [km]')
    plt.subplot(4,1,3)
    plt.plot(dt_vect, i_plot, 'k.')
    plt.ylabel('i [km]')
    plt.subplot(4,1,4)
    plt.plot(dt_vect, c_plot, 'k.')
    plt.ylabel('c [km]')
    plt.xlabel('Seconds since epoch')
    
    plt.show()
    
    
    
    return Xc_output, Xd_output, tmin, rho_min


def TCA_test_setup():
    
    
    elem_chief = np.array([Re+550., 1e-4, 98.6, 30., 40., 50.])
    dt_vect = np.arange(-100., 100.1, 0.01)
    
    Xc_output, Xd_output, tmin, rho_min = \
        compute_initial_conditions(elem_chief, dt_vect)
    
    
    
    
    return


if __name__ == '__main__':
    
    plt.close('all')
    
    TCA_test_setup()

