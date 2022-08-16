import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
import pickle
import os
import inspect
import time

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
from utilities.constants import GME, arcsec2rad




def lambert_test():
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
    GM = state_params['GM']

    
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
    
    retrograde_flag = True
    
    # Propagate several orbit fractions
    elem0 = astro.cart2kep(Xo)
    a = float(elem0[0])
    print('a', a)
    theta0 = float(elem0[5])
    P = 2.*math.pi*np.sqrt(a**3./GM)
    fraction_list = [0., 0.2, 0.8, 1.2, 1.8, 10.2, 10.8]
    
    
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
    for kk in range(2,3):
        
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
        
        v0_list, vf_list, M_list = iod.izzo_lambert(r0_true, rf_true, tof, GM, retrograde_flag)
        
        izzo_time = time.time() - start_time
        
        print('\n')
        print('izzo time', izzo_time)
        print('v0_list', v0_list)
        print('vf_list', vf_list)
        print('v0_true', v0_true)
        print('vf_true', vf_true)
        
      
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


if __name__ == '__main__':
    
    
    lambert_test()
    
#    test_sigmax()
    
#    test_LancasterBlanchard()
    
    
    
    
    