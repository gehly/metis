import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sys
import inspect
import pickle
import time
from datetime import datetime, timedelta


filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from conjunction import conjunction_analysis as ca
from utilities import astrodynamics as astro
from utilities import coordinate_systems as coord
from utilities import tle_functions as tle
from utilities.constants import Re, GME



###############################################################################
# Time of Closest Approach (TCA) Test Cases
###############################################################################



def compute_initial_conditions_twobody(elem_chief, rho_ric, drho_ric, dt_vect,
                                       GM=GME):
    
    
    
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
    
    # Adjust to correspond to Xc_output and Xd_output
    tout = dt_vect - dt_vect[0]
    tmin = dt_vect[ind] - dt_vect[0]
    
    print(rho_min)
    print(tmin)
    print(rho_plot[ind-3:ind+4])
        
        
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(tout, rho_plot, 'k.')
    plt.plot(tmin, rho_min, 'ro')
    plt.ylabel('rho [km]')
    plt.subplot(4,1,2)
    plt.plot(tout, r_plot, 'k.')
    plt.ylabel('r [km]')
    plt.subplot(4,1,3)
    plt.plot(tout, i_plot, 'k.')
    plt.ylabel('i [km]')
    plt.subplot(4,1,4)
    plt.plot(tout, c_plot, 'k.')
    plt.ylabel('c [km]')
    plt.xlabel('Seconds since epoch')
    
    plt.show()
    
    
    
    return Xc_output, Xd_output, tout, tmin, rho_min


def setup_leo_case1():
    
    # Setup initial states and compute truth (Two-Body Dynamics)
    GM = GME
    elem_chief = np.array([Re+550., 1e-4, 98.6, 30., 40., 50.])
    
    # Time vector
    dt_vect = np.arange(-72*3600., 72*3600., 10.)
    dt_vect2 = np.arange(-1., 1., 1e-4)
    dt_vect3 = np.concatenate((dt_vect, dt_vect2))
    dt_vect = np.asarray(sorted(list(set(dt_vect3))))
    
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
    
    X1_0, X2_0, tout, tmin, rho_min = \
        compute_initial_conditions_twobody(elem_chief, rho_ric, drho_ric,
                                           dt_vect, GM)
        
    # Convert tout to UTC for use with tudat
    UTC0 = datetime(2000, 1, 1, 12, 0, 0)
    tout = [UTC0 + timedelta(seconds=ti) for ti in tout]
    
    setup_file = os.path.join('unit_test', 'tca_twobody_leo_case1c.pkl')
    pklFile = open( setup_file, 'wb' )
    pickle.dump([X1_0, X2_0, tout, tmin, rho_min, elem_chief, rho_ric, drho_ric],
                pklFile, -1 )
    pklFile.close()
    
    return


def setup_leo_case2():
    
    
    # This test case is based on Denenberg (2020) COSMOS 1607 - Fenyung 1C DEB
    line1 = '1 15378U 84112A   15194.50416942 -.00000089  00000-0 -28818-6 0  9993'
    line2 = '2 15378  64.9934 313.9757 0056000 236.4761 194.3519 13.83536818552983'
    UTC = tle.tletime2datetime(line1)
    t0 = UTC
    
    tle_dict = {}    
    tle_dict[15378] = {}
    tle_dict[15378]['UTC_list'] = [UTC]
    tle_dict[15378]['line1_list'] = [line1]
    tle_dict[15378]['line2_list'] = [line2]
    
            
    line1 = '1 31570U 99025BZM 15193.80658714  .00004278  00000-0  28637-2 0  9998'
    line2 = '2 31570 102.6018 188.9155 0192610  30.1541  89.8377 13.93148752415999'
    UTC = tle.tletime2datetime(line1)
  
    tle_dict[31570] = {}
    tle_dict[31570]['UTC_list'] = [UTC]
    tle_dict[31570]['line1_list'] = [line1]
    tle_dict[31570]['line2_list'] = [line2]
    
    # Convert object states to ECI
    obj_id_list = [15378, 31570]
    UTC_list = [t0]
    output_state = tle.propagate_TLE(obj_id_list, UTC_list, tle_dict)
    
    obj_id = obj_id_list[0]
    r_GCRF = output_state[obj_id]['r_GCRF'][0]
    v_GCRF = output_state[obj_id]['v_GCRF'][0]
    X1_0 = np.concatenate((r_GCRF, v_GCRF), axis=0)
    
    obj_id = obj_id_list[1]
    r_GCRF = output_state[obj_id]['r_GCRF'][0]
    v_GCRF = output_state[obj_id]['v_GCRF'][0]
    X2_0 = np.concatenate((r_GCRF, v_GCRF), axis=0)
    
    # Time vector
    dt_vect = np.arange(0., 5*86400., 10.)
    # dt_vect2 = np.arange(-1., 1., 1e-4)
    # dt_vect3 = np.concatenate((dt_vect, dt_vect2))
    # dt_vect = np.asarray(sorted(list(set(dt_vect3))))
    tout = dt_vect
    
    rho_plot = np.zeros(dt_vect.shape)
    r_plot = np.zeros(dt_vect.shape)
    i_plot = np.zeros(dt_vect.shape)
    c_plot = np.zeros(dt_vect.shape)
    ii = 0
    for dt in dt_vect:
        Xt_chief = astro.element_conversion(X1_0, 1, 1, GME, dt)
        Xt_deputy = astro.element_conversion(X2_0, 1, 1, GME, dt)
        
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
    
    # Adjust to correspond to Xc_output and Xd_output
    tmin = dt_vect[ind]
    
    print(rho_min)
    print(tmin)
    print(rho_plot[ind-3:ind+4])
        
        
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(tout, rho_plot, 'k.')
    plt.plot(tmin, rho_min, 'ro')
    plt.ylabel('rho [km]')
    plt.subplot(4,1,2)
    plt.plot(tout, r_plot, 'k.')
    plt.ylabel('r [km]')
    plt.subplot(4,1,3)
    plt.plot(tout, i_plot, 'k.')
    plt.ylabel('i [km]')
    plt.subplot(4,1,4)
    plt.plot(tout, c_plot, 'k.')
    plt.ylabel('c [km]')
    plt.xlabel('Seconds since epoch')
    
    plt.show()
    
    

    
    elem_chief = 0.
    rho_ric = 0.
    drho_ric = 0.
    
    
    
    setup_file = os.path.join('unit_test', 'tca_twobody_leo_case2.pkl')
    pklFile = open( setup_file, 'wb' )
    pickle.dump([X1_0, X2_0, tout, tmin, rho_min, elem_chief, rho_ric, drho_ric],
                pklFile, -1 )
    pklFile.close()
    
    return


def run_tca_test(setup_file):
    
    pklFile = open(setup_file, 'rb' )
    data = pickle.load( pklFile )
    X1_0 = data[0]
    X2_0 = data[1]
    tout = data[2]
    tmin = data[3]
    rho_min = data[4]
    elem_chief = data[5]
    rho_ric = data[6]
    drho_ric = data[7]
    pklFile.close()
    
    trange = [tout[0], tout[-1]]
    gvec_fcn = ca.gvec_twobody_analytic
    params = {}
    params['GM'] = GME
    
    
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

    int_params = {}
    int_params['integrator'] = 'tudat'
    int_params['tudat_integrator'] = 'rkf78'
    int_params['step'] = 10.
    int_params['max_step'] = 100.
    int_params['min_step'] = 1.
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'
    
    params['state_params'] = state_params
    params['int_params'] = int_params
    
    
    start = time.time()
    T_list, rho_list = ca.compute_TCA(X1_0, X2_0, trange, gvec_fcn, params,
                                      rho_min_crit=0., N=32, tudat_flag=True)
    runtime = time.time() - start
    
    print('\n')
    print('Results from Setup')
    print('tmin', tmin)
    print('rho_min', rho_min)
    
    print('\nResults from compute_TCA')
    print('T_list', T_list)
    print('rho_list', rho_list)
    
    print('\nruntime', runtime)
    
    
    return


if __name__ == '__main__':
    
    # plt.close('all')
    
    # setup_leo_case1()

    fname = os.path.join('unit_test', 'tca_twobody_leo_case1c.pkl')
    run_tca_test(fname)

