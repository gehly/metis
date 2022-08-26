import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
import pickle
import csv
import os
import inspect
import time
from astroquery.jplhorizons import Horizons

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

import dynamics.dynamics_functions as dyn
import estimation.analysis_functions as analysis
import estimation.estimation_functions as est
import iod.iod_functions as iod
import sensors.measurement_functions as mfunc
import sensors.sensors as sens
import sensors.visibility_functions as visfunc
import utilities.astrodynamics as astro
import utilities.coordinate_systems as coord
import utilities.eop_functions as eop
import utilities.time_systems as timesys

from utilities.constants import GME, Re, arcsec2rad




def lambert_test():
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
    GM = state_params['GM']

    
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
    Xo = np.reshape([757.700301, 5222.606566, 4851.49977,
                     2.213250611, 4.678372741, -5.371314404], (6,1))
    
    
    # MEO Orbit
#    elem0 = [26560., 0.001,55, 0., 0., 0.]
#    Xo = astro.kep2cart(elem0, GM)
    
    # HEO Molniya Orbit
#    Xo = np.reshape([2.88824880e3, -7.73903934e2, -5.97116199e3, 2.64414431,
#                     9.86808092, 0.0], (6,1))
    
    
    # GEO Orbit
#    elem0 = [42164.2, 0.001, 0.01, 0., 0., 0.]
#    Xo = astro.kep2cart(elem0, GM)
    
    
    
    results_flag = 'all'
    periapsis_check = True
    
    # Propagate several orbit fractions
    elem0 = astro.cart2kep(Xo)
    a = float(elem0[0])
    print('a', a)
    theta0 = float(elem0[5])
    P = 2.*math.pi*np.sqrt(a**3./GM)
    fraction_list = [0., 0.2, 0.8, 1.2, 1.8, 5.8, 10.8, 50.2]
    
    
    tvec = np.asarray([frac*P for frac in fraction_list])
    UTC0 = datetime(1999, 10, 4, 1, 45, 0)
    tk_list = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    
    
    # Generate truth fata
    truth_dict = {}
    X = Xo.copy()
    for kk in range(len(tk_list)):
        
        if kk > 0:
            tin = [tk_list[kk-1], tk_list[kk]]
            tout, Xout = dyn.general_dynamics(X, tin, state_params, int_params)
            X = Xout[-1,:].reshape(6, 1)
        
        truth_dict[tk_list[kk]] = X
        
        
    print(truth_dict)
    
    # Setup and run Lambert Solvers
    t0 = tk_list[0]    
#    for kk in range(1,len(fraction_list)):
    for kk in range(5,6):
        
        frac = fraction_list[kk]        
        tf = tk_list[kk]
        tof = (tf - t0).total_seconds()
        r0_true = truth_dict[t0][0:3]
        v0_true = truth_dict[t0][3:6]
        rf_true = truth_dict[tf][0:3]
        vf_true = truth_dict[tf][3:6]
        
        elemf = astro.cart2kep(truth_dict[tf])
        thetaf = float(elemf[5])
        
        # Get m, transfer type, and branch
        d, m = math.modf(frac)        
        m = int(m)
        
        dtheta = thetaf - theta0
        if dtheta < 0.:
            dtheta += 360.
            
        if dtheta < 180.:
            transfer_type = 1
            branch = 'right'
        else:
            transfer_type = 2
            branch = 'left'
            
        print('')
        print(kk)
        print('tof [hours]', tof/3600.)
        print('dtheta', dtheta)
        print('transfer_type', transfer_type)
        print('branch', branch)
        
        print('\n')
        
        # Compute truth data
        # Compute RIC direction and velocity components at t0
        print('Initial Vectors and Velocity')
        v0_ric = eci2ric(r0_true, v0_true, v0_true)
        print('v0_ric', v0_ric)
        print('V1r', float(v0_ric[0]))
        print('V1t', float(v0_ric[1]))
        
        print('Final Vectors and Velocity')
        vf_ric = eci2ric(rf_true, vf_true, vf_true)
        print('vf_ric', vf_ric)
        print('V2r', float(vf_ric[0]))
        print('V2t', float(vf_ric[1]))
        print('\n')
        
        start_time = time.time()
        
        v0_list, vf_list, M_list, type_list = \
            iod.izzo_lambert(r0_true, rf_true, tof, GM, Re, results_flag,
                             periapsis_check)
        
        izzo_time = time.time() - start_time
        
        print('\n')
        print('izzo time', izzo_time)

        print('v0_list', v0_list)
        print('vf_list', vf_list)
        print('v0_true', v0_true)
        print('vf_true', vf_true)
        
        print('')
        print('M_list', M_list)
        print('type_list', type_list)
        print('len M_list', len(M_list))
        
        # Propagate output to ensure it achieves the right final position
        
        rf_err_list = []
        v0_err_list = []
        vf_err_list = []
        for ii in range(len(M_list)):
            
            v0_ii = v0_list[ii]
            vf_ii = vf_list[ii]
#            M_ii = M_list[ii]
                        
            X_test = np.concatenate((r0_true, v0_ii), axis=0)
            elem_test = astro.cart2kep(X_test, GM)
            
            tin = [t0, tf]
#            print('tin', tin)
#            print('tof [hours]', tof/3600.)
            
#            tin = [t0 + timedelta(seconds=tii) for tii in np.arange(0, tof, 10)]
#            tin.append(tf)
#            print('tf', tf)
#            print('tin[-1]', tin[-1])
#            mistake
            
            tout, Xout = dyn.general_dynamics(X_test, tin, state_params, int_params)
            X = Xout[-1,:].reshape(6, 1)
            
#            print('tof diff', tout[-1] - tof)
            print('tout', tout)
            print('X', X)
            
            rf_test = X[0:3].reshape(3,1)
            vf_test = X[3:6].reshape(3,1)
            
            # Check analytic orbit prediction
            Xout2 = astro.element_conversion(X_test, 1, 1, GM, tof)
            print('Xout2', Xout2)
            
            
            print('')
            print('ii', ii)
            print('Mi', M_list[ii])
            print('type', type_list[ii])
            print('rf_test', rf_test)
            print('rf_true', rf_true)
            print('vf_test', vf_test)
            print('vf_true', vf_true)
            
            print('')
            print('X_test', X_test)
            print('elem_test', elem_test)
            print('v0_ii', v0_ii)
            print('vf_ii', vf_ii)
            
            if np.linalg.norm(vf_test - vf_ii) > 1e-6:
                print(vf_test)
                print(vf_ii)
                print(np.linalg.norm(vf_test - vf_ii))
                mistake
            
            
            
            
            rf_err = np.linalg.norm(rf_test - rf_true)
            v0_err = np.linalg.norm(v0_ii - v0_true)
            vf_err = np.linalg.norm(vf_ii - vf_true)
            
            rf_err_list.append(rf_err)
            v0_err_list.append(v0_err)
            vf_err_list.append(vf_err)
            
            
#            rbg = (colors[ii], colors[ii], colors[ii])
#            
#            plt.subplot(3,1,1)
#            plt.plot(M_ii, rf_err, 'o', color=rbg)
#            
#            plt.subplot(3,1,2)
#            plt.plot(M_ii, v0_err, 'o', color=rbg)
#            
#            plt.subplot(3,1,3)
#            plt.plot(M_ii, vf_err, 'o', color=rbg)
            
            
#        colors = np.linspace(0, 1, len(M_list))
        colors = np.random.rand(len(M_list),3)
        
        plt.figure()
        
        plt.subplot(3,1,1)
        for ii in range(len(M_list)):            
            rbg = (colors[ii,0], colors[ii,1], colors[ii,2])
            plt.plot(M_list[ii], rf_err_list[ii], 'o', color=rbg)
        plt.ylabel('Final Pos [km]')
        plt.title('Lambert Pos/Vel Errors')
        if max(rf_err_list) < 1.:
            plt.ylim([0., 1])
        
        plt.subplot(3,1,2)
        for ii in range(len(M_list)):            
            rbg = (colors[ii,0], colors[ii,1], colors[ii,2])
            plt.plot(M_list[ii], v0_err_list[ii], 'o', color=rbg)
        plt.ylabel('Initial Vel [km/s]')
        
        plt.subplot(3,1,3)
        for ii in range(len(M_list)):            
            rbg = (colors[ii,0], colors[ii,1], colors[ii,2])
            plt.plot(M_list[ii], vf_err_list[ii], 'o', color=rbg)
        plt.ylabel('Final Vel [km/s]')



        plt.show()

      
#        start_time = time.time()
#        
#        v0_vect, vf_vect, extremal_distances, exit_flag = \
#            iod.fast_lambert(r0_true, rf_true, tof, m, GM, transfer_type, branch)
#            
#        fast_lambert_time = time.time() - start_time
#        
#        v0_err = v0_vect - v0_true
#        vf_err = vf_vect - vf_true
#        
#        print(v0_err)
#        print(vf_err)
#        
#        start_time = time.time()
#        
#        v0_vect, vf_vect, extremal_distances, exit_flag = \
#            iod.robust_lambert(r0_true, rf_true, tof, m, GM, transfer_type, branch)
#            
#        robust_lambert_time = time.time() - start_time
#        
#        v0_err = v0_vect - v0_true
#        vf_err = vf_vect - vf_true
#        
#        print(v0_err)
#        print(vf_err)
#        

    
    
    return


def lambert_test_hyperbolic():
    
     # Define state parameters
    state_params = {}
    state_params['GM'] = GME
    GM = state_params['GM']

    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = dyn.ode_twobody
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['step'] = 10.
    int_params['time_format'] = 'datetime'
    
    # Hyperbolic Orbit
    vinf = 1.
    rp = Re + 500.
    a = -GM/vinf**2.
    e = 1. + rp*vinf**2./GM
    elem0 = [a, e, 10., 10., 10., 10.]
    theta0 = elem0[5]
    Xo = astro.kep2cart(elem0, GM)
    
    
    results_flag = 'all'
    periapsis_check = True
    
    
    tvec = [0., 1000.]
    UTC0 = datetime(1999, 10, 4, 1, 45, 0)
    tk_list = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    
    
    # Generate truth fata
    truth_dict = {}
    X = Xo.copy()
    for kk in range(len(tk_list)):
        
        if kk > 0:
            tin = [tk_list[kk-1], tk_list[kk]]
            tout, Xout = dyn.general_dynamics(X, tin, state_params, int_params)
            X = Xout[-1,:].reshape(6, 1)
        
        truth_dict[tk_list[kk]] = X
        
        
    print(truth_dict)
    
    # Setup and run Lambert Solvers
    t0 = tk_list[0]    
    tf = tk_list[-1]

            
    tof = (tf - t0).total_seconds()
    r0_true = truth_dict[t0][0:3]
    v0_true = truth_dict[t0][3:6]
    rf_true = truth_dict[tf][0:3]
    vf_true = truth_dict[tf][3:6]
    
    elemf = astro.cart2kep(truth_dict[tf])
    thetaf = float(elemf[5])
    
           
    print('')
    print(kk)
    print('tof [hours]', tof/3600.)
    print('dtheta [deg]', thetaf - theta0)
    print('\n')
    
    # Compute truth data
    # Compute RIC direction and velocity components at t0
    print('Initial Vectors and Velocity')
    v0_ric = eci2ric(r0_true, v0_true, v0_true)
    print('v0_ric', v0_ric)
    print('V1r', float(v0_ric[0]))
    print('V1t', float(v0_ric[1]))
    
    print('Final Vectors and Velocity')
    vf_ric = eci2ric(rf_true, vf_true, vf_true)
    print('vf_ric', vf_ric)
    print('V2r', float(vf_ric[0]))
    print('V2t', float(vf_ric[1]))
    print('\n')
    
    start_time = time.time()
    
    v0_list, vf_list, M_list = iod.izzo_lambert(r0_true, rf_true, tof, GM, Re, results_flag, periapsis_check)
    
    izzo_time = time.time() - start_time
    
    print('\n')
    print('izzo time', izzo_time)
    print('v0_list', v0_list)
    print('vf_list', vf_list)
    print('v0_true', v0_true)
    print('vf_true', vf_true)
    
    print('')
    print('M_list', M_list)
    print('len M_list', len(M_list))
    
    # Propagate output to ensure it achieves the right final position
    
    rf_err_list = []
    v0_err_list = []
    vf_err_list = []
    for ii in range(len(M_list)):
        
        v0_ii = v0_list[ii]
        vf_ii = vf_list[ii]
#            M_ii = M_list[ii]
                    
        X_test = np.concatenate((r0_true, v0_ii), axis=0)
        elem_test = astro.cart2kep(X_test, GM)
        
        tin = [t0, tf]
#            print('tin', tin)
#            print('tof [hours]', tof/3600.)
        
#            tin = [t0 + timedelta(seconds=tii) for tii in np.arange(0, tof, 10)]
#            tin.append(tf)
#            print('tf', tf)
#            print('tin[-1]', tin[-1])
#            mistake
        
        tout, Xout = dyn.general_dynamics(X_test, tin, state_params, int_params)
        X = Xout[-1,:].reshape(6, 1)
        
#            print('tof diff', tout[-1] - tof)
        print('tout', tout)
        print('X', X)
        
        rf_test = X[0:3].reshape(3,1)
        vf_test = X[3:6].reshape(3,1)
        
        # Check analytic orbit prediction
        Xout2 = astro.element_conversion(X_test, 1, 1, GM, tof)
        print('Xout2', Xout2)
        
        
        print('')
        print('ii', ii)
        print('Mi', M_list[ii])
        print('rf_test', rf_test)
        print('rf_true', rf_true)
        print('vf_test', vf_test)
        print('vf_true', vf_true)
        
        print('')
        print('X_test', X_test)
        print('elem_test', elem_test)
        print('v0_ii', v0_ii)
        print('vf_ii', vf_ii)
        
        if np.linalg.norm(vf_test - vf_ii) > 1e-6:
            print(vf_test)
            print(vf_ii)
            print(np.linalg.norm(vf_test - vf_ii))
            mistake
        
        
        
        
        rf_err = np.linalg.norm(rf_test - rf_true)
        v0_err = np.linalg.norm(v0_ii - v0_true)
        vf_err = np.linalg.norm(vf_ii - vf_true)
        
        rf_err_list.append(rf_err)
        v0_err_list.append(v0_err)
        vf_err_list.append(vf_err)
        
        
#            rbg = (colors[ii], colors[ii], colors[ii])
#            
#            plt.subplot(3,1,1)
#            plt.plot(M_ii, rf_err, 'o', color=rbg)
#            
#            plt.subplot(3,1,2)
#            plt.plot(M_ii, v0_err, 'o', color=rbg)
#            
#            plt.subplot(3,1,3)
#            plt.plot(M_ii, vf_err, 'o', color=rbg)
        
        
#        colors = np.linspace(0, 1, len(M_list))
    colors = np.random.rand(len(M_list),3)
    
    plt.figure()
    
    plt.subplot(3,1,1)
    for ii in range(len(M_list)):            
        rbg = (colors[ii,0], colors[ii,1], colors[ii,2])
        plt.plot(M_list[ii], rf_err_list[ii], 'o', color=rbg)
    plt.ylabel('Final Pos [km]')
    plt.title('Lambert Pos/Vel Errors')
    if max(rf_err_list) < 1.:
        plt.ylim([0., 1])
    
    plt.subplot(3,1,2)
    for ii in range(len(M_list)):            
        rbg = (colors[ii,0], colors[ii,1], colors[ii,2])
        plt.plot(M_list[ii], v0_err_list[ii], 'o', color=rbg)
    plt.ylabel('Initial Vel [km/s]')
    
    plt.subplot(3,1,3)
    for ii in range(len(M_list)):            
        rbg = (colors[ii,0], colors[ii,1], colors[ii,2])
        plt.plot(M_list[ii], vf_err_list[ii], 'o', color=rbg)
    plt.ylabel('Final Vel [km/s]')



    plt.show()
    
    
    
    
    return


def eci2ric(rc_vect, vc_vect, Q_eci=[]):
    '''
    This function computes the rotation from ECI to RIC and rotates input
    Q_eci (vector or matrix) to RIC.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    Q_eci : 3x1 or 3x3 numpy array
      vector or matrix in ECI

    Returns
    ------
    Q_ric : 3x1 or 3x3 numpy array
      vector or matrix in RIC
    '''
    
    # Reshape inputs
    rc_vect = rc_vect.reshape(3,1)
    vc_vect = vc_vect.reshape(3,1)

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)
    
    print('OR', OR)
    print('OH', OH)
    print('OT', OT)

    ON = np.concatenate((OR.T, OT.T, OH.T))

    # Rotate Q_eci as appropriate for vector or matrix
    if len(Q_eci) == 0:
        Q_ric = ON
    elif np.size(Q_eci) == 3:
        Q_eci = Q_eci.reshape(3,1)
        Q_ric = np.dot(ON, Q_eci)
    else:
        Q_ric = np.dot(np.dot(ON, Q_eci), ON.T)

    return Q_ric



def test_sigmax():
    
    y = -1.5
    sig1, dsigdx1, d2sigdx21, d3sigdx31 = iod.compute_sigmax(y)
    
    print(sig1, dsigdx1, d2sigdx21, d3sigdx31)
    
    return


def test_LancasterBlanchard():
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME

    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'ode'
    int_params['ode_integrator'] = 'dop853'
    int_params['intfcn'] = dyn.ode_twobody
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'
    
    # Initial object state vector
    # Sun-Synch Orbit
    Xo = np.reshape([757.700301, 5222.606566, 4851.49977,
                     2.213250611, 4.678372741, -5.371314404], (6,1))
    
    # Propagate several orbit fractions
    tvec = np.asarray([0., 20.*60., 70.*60.])
    UTC0 = datetime(1999, 10, 4, 1, 45, 0)
    tk_list = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    
    
    # Generate truth and measurements
    truth_dict = {}
    X = Xo.copy()
    for kk in range(len(tk_list)):
        
        if kk > 0:
            tin = [tk_list[kk-1], tk_list[kk]]
            tout, Xout = dyn.general_dynamics(X, tin, state_params, int_params)
            X = Xout[-1,:].reshape(6, 1)
        
        truth_dict[tk_list[kk]] = X
        
        
#    print(truth_dict)
    
    # Setup and run Lambert Solvers
    GM = state_params['GM']
    t0 = tk_list[0]
    tf = tk_list[1]
    tof = (tf - t0).total_seconds()
    r0_vect = truth_dict[t0][0:3]
    rf_vect = truth_dict[tf][0:3]
    
    r0 = np.linalg.norm(r0_vect)
    rf = np.linalg.norm(rf_vect)
    r0_hat = r0_vect/r0                        
    rf_hat = rf_vect/rf 
    dtheta = math.acos(max(-1, min(1, np.dot(r0_hat.T, rf_hat))))
    
    c = np.sqrt(r0**2. + rf**2. - 2.*r0*rf*math.cos(dtheta))
    s = (r0 + rf + c)/2.
    T = np.sqrt(8.*GM/s**3.) * tof
    q = np.sqrt(r0*rf)/s * math.cos(dtheta/2.)
    
    
    print('q', q)
    
    x = 0.5
    m = 0
    
    
    T, Tp, Tpp, Tppp = iod.LancasterBlanchard(x, q, m)
    print(T, Tp, Tpp, Tppp)
    
    return


def retrieve_jpl_vector_table(output_file, obj_id, start_dt, stop_dt, step, location='@0', id_type='majorbody'):
    '''
    This function retrieves ephemeris data from the JPL Horizons database using
    the astroquery module of astropy in a vector table format.
    
    Parameters
    ------
    obj_id : string
        unique identifier for target object
    start_dt : string
        date and time of start epoch formatted 'YYYY-MM-DD [HH:MM:SS]'
        time system [CT = TDB]
    stop_dt : string
        date and time of stop epoch formatted 'YYYY-MM-DD [HH:MM:SS]'
        time system [CT = TDB]
    step : string
        interval for retrieval with units, e.g. '10m' or '3h' or '1d'
    location : string, optional
        origin of coordinate system from which position/velocity measured
        (default = '@0' Solar System Barycenter)
        (use '500' for geocentric)
    id_type : string, optional
        further specification of identity to aid in disambiguation
        (default = 'majorbody')  
    
    Object ID and Type
    ------    
    The use of id_type in combination with obj_id is important.  For major
    bodies, e.g. planets and moons, the following id list applies:
        
    10 : Sun
            
    199 : Mercury
    
    299 : Venus
    
    399 : Earth [Geocenter]
    
    499 : Mars
    
    599 : Jupiter
    
    699 : Saturn
    
    799 : Uranus
    
    899 : Neptune
    
    999 : Pluto
    
    If you use these obj_id values without specifying id_type='majorbody' it
    will return values for a different object!!
    
    Returns
    ------
    vec_table : astropy vector table
        date, time, position, velocity and other values for the requested
        object at each requested time [CT=TDB, km, km/s]    
    
    References
    ------
    https://ssd.jpl.nasa.gov/?horizons_tutorial
    https://astroquery.readthedocs.io/en/latest/jplhorizons/jplhorizons.html
    
    '''
    
    # Run query
    obj = Horizons(id=obj_id, id_type=id_type, location=location, epochs={'start':start_dt, 'stop':stop_dt, 'step':step})
    vec_table = obj.vectors()

    # Convert to km, km/s    
    vec_table['x'].convert_unit_to('km')
    vec_table['y'].convert_unit_to('km')
    vec_table['z'].convert_unit_to('km')
    vec_table['vx'].convert_unit_to('km/s')
    vec_table['vy'].convert_unit_to('km/s')
    vec_table['vz'].convert_unit_to('km/s')
    
    # Generate output file    
    with open(output_file, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerow(['Julian Date', 'x [km]', 'y [km]', 'z [km]', 'vx [km/s]', 'vy [km/s]', 'vz [km/s]'])
    
        for ii in range(len(list(vec_table['x']))):
                
                date = vec_table['datetime_jd'][ii]
                x = vec_table['x'][ii]
                y = vec_table['y'][ii]
                z = vec_table['z'][ii]
                vx = vec_table['vx'][ii]
                vy = vec_table['vy'][ii]
                vz = vec_table['vz'][ii]
                                
                output_row = [date, x, y, z, vx, vy, vz]                
                filewriter.writerow(output_row)
    
    return vec_table


def generate_porkchop_data(launch_file, arrival_file, GM, R, min_tof, max_tof, output_file):
    '''
    This function generates a porkchop plot of the total v_infinity cost of
    transfer between two celestial objects for a range of launch and arrival
    times.  The function takes CSV files as input, containing the dates and
    position/velocity data of the celestial objects.
    
    Parameters
    ------
    launch_file : string
        file path and name containing data for launch object
    arrival_file : string
        file path and name containing data for arrival object
    GM : float
        gravitational parameter of central body [km^3/s^2]
    min_tof : float
        minimum time of flight to consider [days]
    max_tof : float
        maximum time of flight to consider [days]
    output_File : string
        file path and name to save porkchop data for plotting
        
    '''
    
    # Read input CSV data
    with open(launch_file, newline='') as csvfile:
        filereader = csv.reader(csvfile)
        ii = 0
        launch_JD = []
        launch_state = []
        for row in filereader:
            if ii == 0:
                ii += 1
                continue
            
            launch_JD.append(float(row[0]))
            state_vect = np.reshape([float(row[1]), float(row[2]),
                                     float(row[3]), float(row[4]),
                                     float(row[5]), float(row[6])], (6,1))
            launch_state.append(state_vect)

            
            
    with open(arrival_file, newline='') as csvfile:
        filereader = csv.reader(csvfile)
        ii = 0
        arrival_JD = []
        arrival_state = []
        for row in filereader:
            if ii == 0:
                ii += 1
                continue
            
            arrival_JD.append(float(row[0]))
            state_vect = np.reshape([float(row[1]), float(row[2]),
                                     float(row[3]), float(row[4]),
                                     float(row[5]), float(row[6])], (6,1))
            arrival_state.append(state_vect)
            
    
    # Loop over launch dates
    launch_vinf = np.zeros((len(launch_JD), len(arrival_JD)))
    arrival_vinf = np.zeros((len(launch_JD), len(arrival_JD)))
    for ii in range(len(launch_JD)):
        JD_ii = launch_JD[ii]
        state_ii = launch_state[ii]
        
        # Loop over arrival dates
        for jj in range(len(arrival_JD)):
            JD_jj = arrival_JD[jj]
            state_jj = arrival_state[jj]
            
            # Skip cases where transfer time exceeds user input boundaries
            if (JD_jj - JD_ii) < min_tof or (JD_jj - JD_ii) > max_tof:
                continue
            
            # Compute difference in anomaly angle to check Type I or Type II
            launch_elem = astro.cart2kep(state_ii, GM)
            arrival_elem = astro.cart2kep(state_jj, GM)
            
            launch_anomaly = sum(launch_elem[3:6]) % 360
            arrival_anomaly = sum(arrival_elem[3:6]) % 360
            diff = (arrival_anomaly - launch_anomaly) % 360
            
            print('\n')
            print(ii, jj)
            print(state_ii)
            print(launch_elem)
            print(launch_anomaly, arrival_anomaly, diff)
            print(JD_jj - JD_ii)
            

                
            # Call the Lambert solver
            r1 = state_ii[0:3]
            r2 = state_jj[0:3]
            tof = (JD_jj - JD_ii)*86400.
            v1_list, v2_list, M_list = iod.izzo_lambert(r1, r2, tof, GM, R, 
                                                        results_flag='prograde',
                                                        periapsis_check=True, 
                                                        maxiters=35, rtol=1e-8)
            
            if 0 not in M_list:
                continue
            
            ind = M_list.index(0)
            v1 = v1_list[ind]
            v2 = v2_list[ind]
            
            # Compute the v_infinity values and store
            launch_vinf[ii,jj] = np.linalg.norm(v1 - state_ii[3:6])
            arrival_vinf[ii,jj] = np.linalg.norm(v2 - state_jj[3:6])
            
            print(v1, v2)
            print(launch_vinf[ii,jj], arrival_vinf[ii,jj])
                
            
    # Generate plot
    total_vinf = launch_vinf + arrival_vinf
    
    pklFile = open( output_file , 'wb' )
    pickle.dump( [launch_JD, arrival_JD, launch_vinf, arrival_vinf, total_vinf], pklFile, -1 )
    pklFile.close()
    
    
    
    return


def plot_porkchop_data(vinf_file, min_tof, max_tof):
    
    plt.close('all')
    
    pklFile = open(vinf_file, 'rb')
    data = pickle.load(pklFile)
    launch_JD = data[0]
    arrival_JD = data[1]
    launch_vinf = data[2]
    arrival_vinf = data[3]
    total_vinf = data[4]
    pklFile.close()
    
    launch_dt = [timesys.jd2dt(jd) for jd in launch_JD]
    
    # Compute TOF for x-axis
    nrow = total_vinf.shape[0]
    ncol = total_vinf.shape[1]
    plot_tof = np.arange(min_tof, max_tof+0.1, 1.)
    plot_vinf = np.zeros((nrow, len(plot_tof)))
    for ii in range(nrow):
        for jj in range(ncol):
            
            if total_vinf[ii,jj] > 0. and total_vinf[ii,jj] < 30.:
                tof = arrival_JD[jj] - launch_JD[ii]
                tof_ind = list(plot_tof).index(tof)
                plot_vinf[ii, tof_ind] = total_vinf[ii,jj]
                
                
    
    fig1, ax1 = plt.subplots()
    levels = list(np.arange(3,31,3))
    cs = ax1.contourf(plot_tof, launch_dt, plot_vinf, levels)
#    ax1.set_ylim([2455375, 2454930])
    ax1.set_ylim([datetime(2029, 12, 31), datetime(2020, 1, 1)])
    
    ax1.set_xlabel('Time of Flight [days]')
    ax1.set_ylabel('Launch Date')

    cbar = fig1.colorbar(cs)
    cbar.ax.set_ylabel('Total Vinf [km/s]')
    
    plt.show()
    
    
    return


def porkchop_plot_demo():
    
    # Retrieve data for Earth
    earth_data = retrieve_jpl_vector_table('earth_2020_2030.csv', '399', '2020-01-01 00:00:00', '2029-12-31 00:00:00', '1d', location='@0', id_type='majorbody')
  
    print(earth_data)
    
    # Retrieve data for Mars
    mars_data = retrieve_jpl_vector_table('mars_2020_2030.csv', '499', '2020-01-01 00:00:00', '2029-12-31 00:00:00', '1d', location='@0', id_type='majorbody')
  
    print(mars_data)
    
    
    launch_file = 'earth_2020_2030.csv'
    arrival_file = 'mars_2020_2030.csv'
    vinf_file = 'earth_mars.pkl'
    GM = 1.32712440018e11
    R = 1e6
    min_tof = 60.
    max_tof = 500.
    generate_porkchop_data(launch_file, arrival_file, GM, R, min_tof, max_tof, vinf_file)
    
    
    plot_porkchop_data(vinf_file, min_tof, max_tof)
    
    
    
    return


def unit_test_gibbs_iod():
    
    # Vallado Test Case (Example 7-3)
    r1_vect = np.reshape([0., 0., 6378.137], (3,1))
    r2_vect = np.reshape([0., -4464.696, -5102.509], (3,1))
    r3_vect = np.reshape([0., 5740.323, 3189.068], (3,1))
    
    GM = GME
    v2_vect = iod.gibbs_iod(r1_vect, r2_vect, r3_vect, GM)
    
    print(v2_vect)
    
    return


def unit_test_herrick_gibbs_iod():
    
    # Vallado Test Case (Example 7-4)
    r1_vect = np.reshape([3419.85564, 6019.82602, 2784.60022], (3,1))
    r2_vect = np.reshape([2935.91195, 6326.18324, 2660.59584], (3,1))
    r3_vect = np.reshape([2434.95202, 6597.38674, 2521.52311], (3,1))
    
    UTC0 = datetime(2022, 1, 1, 12, 0, 0)
    UTC1 = UTC0 + timedelta(seconds=76.48)
    UTC2 = UTC0 + timedelta(seconds=153.04)
    UTC_list = [UTC0, UTC1, UTC2]
    
    GM = GME
    v2_vect = iod.herrick_gibbs_iod(r1_vect, r2_vect, r3_vect, UTC_list, GM)
    
    print(v2_vect)
    
    return


def unit_test_gauss_iod():
    
    # Vallado Test Case (Example 7-2)
    UTC2 = datetime(2012, 8, 20, 11, 48, 28)
    r2_true = np.reshape([6356.486034, 5290.5322578, 6511.396979], (3,1))
    v2_true = np.reshape([-4.172948, 4.776550, 1.720271], (3,1))
    
    # Observations
    UTC1 = datetime(2012, 8, 20, 11, 40, 28)
    UTC3 = datetime(2012, 8, 20, 11, 52, 28)
    
    Y1 = np.array([[0.939913*math.pi/180.],
                   [18.667717*math.pi/180.]])
    
    Y2 = np.array([[45.025748*math.pi/180.],
                   [35.664741*math.pi/180.]])
    
    Y3 = np.array([[67.886655*math.pi/180.],
                   [36.996583*math.pi/180.]])
    
    # Sensor parameters
    sensor_id = 'Vallado 7-2'
    lat = 40.
    lon = -110.
    ht = 2.
    
    site_ecef = coord.latlonht2ecef(lat, lon, ht)
    
    sensor_params = {}
    sensor_params[sensor_id] = {}
    sensor_params[sensor_id]['site_ecef'] = site_ecef
    sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
    
    # Form inputs
    UTC_list = [UTC1, UTC2, UTC3]
    Yk_list = [Y1, Y2, Y3]
    sensor_id_list = [sensor_id]*3
    
    # Execute function
    UTC2, r2_vect, v2_vect, exit_flag = \
        iod.gauss_angles_iod(UTC_list, Yk_list, sensor_id_list, sensor_params)
        
    print('')
    print('r2_vect', r2_vect)
    print('r2_true', r2_true)
    print('v2_vect', v2_vect)
    print('v2_true', v2_true)
                                                             
    
    return


if __name__ == '__main__':
    
    plt.close('all')
    
#    unit_test_gibbs_iod()
    
#    unit_test_herrick_gibbs_iod()
    
#    unit_test_gauss_iod()
    
    
    lambert_test()
    
#    lambert_test_hyperbolic()
    
#    porkchop_plot_demo()
    
#    test_sigmax()
    
#    test_LancasterBlanchard()
    

    
    
    
    
    