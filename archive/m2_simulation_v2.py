import numpy as np
from datetime import datetime, timedelta
import sys
import os
import copy
from math import pi, sin, cos, tan, asin, acos, atan2
import matplotlib.pyplot as plt
import pickle

sys.path.append(r'C:\Users\Steve\Documents\code\metis')


from estimation.analysis_functions import compute_orbit_errors
from estimation.estimation_functions import ls_batch, H_radec, H_rgradec, H_cwrho, H_cwxyz, H_nonlincw_full
from dynamics.dynamics_functions import general_dynamics
from dynamics.dynamics_functions import ode_twobody, ode_twobody_stm, ode_lincw_stm, ode_nonlin_cw, ode_nonlin_cw_stm, ode_lincw
from sensors.sensors import define_sensors
from sensors.visibility_functions import check_visibility
from sensors.measurements import compute_measurement
from utilities.astrodynamics import kep2cart, cart2kep
from utilities.attitude import dcm_principal_axis
from utilities.constants import Re, GME, arcsec2rad
from utilities.coordinate_systems import itrf2gcrf, gcrf2itrf, ecef2enu, ric2eci, latlonht2ecef, eci2ric
from utilities.coordinate_systems import eci2ric_vel, ric2eci_vel
from utilities.eop_functions import get_celestrak_eop_alldata, get_eop_data
from utilities.eop_functions import get_XYs2006_alldata
from utilities.tle_functions import propagate_TLE


###############################################################################
# Utility Functions
###############################################################################

def el2range(sensor_ecef, el, r):
    '''
    This function computes the range from observer to object given the sensor
    location in ECEF, elevation angle, and orbit radius of the object.
    
    Parameters
    ------
    sensor_ecef : 3x1 numpy array
        location of sensor in ECEF [km]
    el : float
        elevation angle [rad]
    r : float
        orbit radius [km]
    
    Returns
    ------
    rho_los : float
        range from observer to object [km]
    '''
    
    q = np.linalg.norm(sensor_ecef)
    a = 1.
    b = -2.*q*cos(el+pi/2.)
    c = q**2. - r**2.
    
    rho_los = (-b + np.sqrt(b**2. - 4.*a*c))/(2.*a)    
    
    return rho_los


def compute_psi(uhat_los, rhat):
    '''
    This function computes the tilt angle psi between the focal plane and
    IC plane of the RIC frame.
    
    Parameters
    ------
    uhat_los : 3x1 numpy array
        unit vector pointing from sensor to chief spacecraft
    rhat : 3x1 numpy array
         unit vector pointing from center of earth to chief spacecraft
         
    Returns
    ------
    psi : float
        angle between FP and IC plane [rad]
    
    '''
    
    psi = acos(np.dot(uhat_los.flatten(), rhat.flatten()))    
    
    return psi


def compute_uhat_psi(uhat_los, rhat):
    '''
    This function returns the unit vector defining the tilt axis of the focal
    plane relative to the IC plane. Equivalently this vector is formed by 
    the intersection of the focal plane and IC plane.
    
    Parameters
    ------
    uhat_los : 3x1 numpy array
        unit vector pointing from sensor to chief spacecraft
    rhat : 3x1 numpy array
         unit vector pointing from center of earth to chief spacecraft
         
    Returns
    ------
    uhat_psi : 3x1 numpy array
        unit vector of tilt axis (intersection of focal plane and IC plane)
    
    '''
    
    uhat_psi = np.cross(uhat_los.flatten(), rhat.flatten())
    uhat_psi = np.reshape(uhat_psi, (3,1))
    
    return uhat_psi


def compute_phi(uhat_psi_ric):
    '''
    This function returns the clock angle phi by which the RIC has to rotate
    to align the I axis with the uhat_psi tilt axis.
    
    uhat_psi is the unit vector defined by the intersection of the focal plane
    and IC plane.
    
    Parameters
    ------
    uhat_psi_ric : 3x1 numpy array
        unit vector of tilt axis (intersection of focal plane and IC plane)
        defined in RIC coordinates
        
    Returns
    ------
    phi : float
        angle to rotate about R to align the I axis with uhat_psi tilt axis
        
    '''
    
    phi = atan2(float(uhat_psi_ric[2]), float(uhat_psi_ric[1]))
    
    return phi


def radec2losvec(ra, dec):
    '''
    This function computes the LOS unit vector in ECI frame given topocentric
    RA and DEC in radians.
    
    Parameters
    ------
    ra : float
        topocentric right ascension [rad]
    dec : float
        topocentric declination [rad]
        
    Returns
    ------
    rho_hat_eci : 3x1 numpy array
        LOS unit vector in ECI
    
    '''
    
    rho_hat_eci = np.array([[cos(ra)*cos(dec)],
                            [sin(ra)*cos(dec)],
                            [sin(dec)]])
    
    return rho_hat_eci


def compute_delta(rho_hat1, rho_hat2):
    '''
    This function computes the angle in radians between two unit vectors
    
    Parameters
    ------
    rho_hat1 : 3x1 numpy array
        unit vector
    rho_hat2 : 3x1 numpy array
        unit vector
    
    Returns
    ------
    delta : float
        angle between the input unit vectors [rad]
    
    '''
    
    delta = acos(np.dot(rho_hat1.T, rho_hat2))
    
    return delta


def ric2fp(psi, phi, r_ric):
    '''
    This function rotates a 3x1 position vector from RIC frame to FP frame.
    
    Parameters
    ------
    psi : float
        tilt angle between focal plane (FP) and IC-plane of RIC [rad]
    phi : float
        clock angle rotation between I and uhat_psi tilt axis [rad]
        uhat_psi is equivalently defined by the intersection of FP and IC
        planes
    r_ric : 3x1 numpy array
        position vector in RIC coordinates
        
    Returns
    ------
    r_fp : 3x1 numpy array
        position vector in FP coordinates
    
    '''
    
    r_ric = np.reshape(r_ric, (3,1))
    
    R2_psi = dcm_principal_axis(2, -psi)
    R1_phi = dcm_principal_axis(1, phi)
    
    r_fp = np.dot(R2_psi, np.dot(R1_phi, r_ric))
    
    return r_fp


###############################################################################
# Unit Test
###############################################################################
    

def unit_test_relative_geometry():
    
    # Initial time and chief orbit
    UTC = datetime(2000, 3, 21, 11, 0, 0)
    
    rc_GCRF = np.reshape([Re+1000., 0., 1000.], (3,1))
    rc = np.linalg.norm(rc_GCRF)
    vc = np.sqrt(GME/rc)
    vc_GCRF = np.reshape([0., vc, 0.], (3,1))
    
    # Sensor params
    lat_gs = 0.       # deg
    lon_gs = 0.   # deg
    ht_gs = 0.   # km
    site_ecef = latlonht2ecef(lat_gs, lon_gs, ht_gs)
    sensor = {}
    sensor['site_ecef'] = site_ecef
    
    # EOP Data
    eop_alldata = get_celestrak_eop_alldata()
    XYs_df = get_XYs2006_alldata()
    EOP_data = get_eop_data(eop_alldata, UTC)
    
    # Compute fixed parameters like el, beta, rho_obs
    Xc = np.concatenate((rc_GCRF, vc_GCRF))
    state_params = {}
    
    
    meas_types = ['az', 'el', 'rg']
    meas = compute_measurement(Xc, state_params, sensor, UTC, EOP_data, XYs_df, meas_types)
    
    el = float(meas[1])    
    rho_los = el2range(sensor['site_ecef'], el, rc)
    rho_c = float(meas[2])
    
    print('\n\n')
    print('rc', rc)
    print('Xc', Xc)
    print('el [deg]', el*180./pi)
    print('rho_los', rho_los)
    print('rho_c', rho_c)
    
    
    # Apply different relative position vectors to check results
    x = 3.
    y = 8.
    z = -1.
    rho_rel_ric = np.reshape([x, y, z], (3,1))
    
    # Compute Deputy location in ECI
    rho_rel_eci = ric2eci(rc_GCRF, vc_GCRF, rho_rel_ric)
    rd_GCRF = rc_GCRF + rho_rel_eci
    rd = np.linalg.norm(rd_GCRF)
    
    print('dist check', rd**2 - rc**2 - y**2. - z**2.)
    
    # Compute RA/DEC of Chief and Deputy
    meas_types = ['ra', 'dec']
    Xd = np.concatenate((rd_GCRF, vc_GCRF))   # note velocity won't matter
    
    radec_c = compute_measurement(Xc, state_params, sensor, UTC, EOP_data, XYs_df, meas_types)
    radec_d = compute_measurement(Xd, state_params, sensor, UTC, EOP_data, XYs_df, meas_types)
    
    # Compute Delta angle between Chief and Deputy - Truth
    rho_hat_eci_c = radec2losvec(radec_c[0], radec_c[1])
    rho_hat_eci_d = radec2losvec(radec_d[0], radec_d[1])
    delta_true = compute_delta(rho_hat_eci_c, rho_hat_eci_d)
    rho_fp_true = delta_true*rho_c

    # Compute checks
    sensor_itrf = site_ecef
    sensor_gcrf, dum = itrf2gcrf(sensor_itrf, np.zeros((3,1)), UTC, EOP_data,
                                 XYs_df)
    rho_los_vect = rc_GCRF - sensor_gcrf
    
    
    rc_hat = rc_GCRF/rc
    uhat_los = rho_los_vect/np.linalg.norm(rho_los_vect)
    
    uhat_psi_eci = compute_uhat_psi(uhat_los, rc_hat)
    uhat_psi_ric = eci2ric(rc_GCRF, vc_GCRF, uhat_psi_eci)
    
    psi = compute_psi(uhat_los, rc_hat)
    phi = compute_phi(uhat_psi_ric)
    
    R2_psi = dcm_principal_axis(2, -psi)
    R1_phi = dcm_principal_axis(1, phi)
    
    rho_rel_fp = np.dot(R2_psi, np.dot(R1_phi, rho_rel_ric))
    rho_fp_check = np.linalg.norm(rho_rel_fp[1:3])
    delta_check = rho_fp_check/rho_los
    
    print('\n\n')
    print('rho_rel_ric', rho_rel_ric)
    print('rho_rel_fp', rho_rel_fp)
    
    print('delta_true [deg]', delta_true*180/pi)
    print('delta_check [deg]', delta_check*180/pi)
    print('rho_fp_true', rho_fp_true)    
    print('rho_fp_check', rho_fp_check)
    
    
    return


def unit_test_cw_propagation():
    
    # Initial Time
    t0 = '2021-09-10T04:55:00.000'
    UTC0 = datetime.strptime(t0, '%Y-%m-%dT%H:%M:%S.%f')
    tvec = np.arange(0., 3600.*36. + 1., 10.)
    thrs = tvec/3600.
    
    # Initial State
    obj_id = 47967
    obj_id_list = [obj_id]
    UTC_list = [UTC0]
    tle_dict = {}
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict, prev_flag=True,
                                 offline_flag=False, frame_flag=True,
                                 username='steve.gehly@gmail.com',
                                 password='SpaceTrackPword!')

    
    r_GCRF = output_state[obj_id]['r_GCRF'][0]
    v_GCRF = output_state[obj_id]['v_GCRF'][0]
    
    X1_true = np.concatenate((r_GCRF, v_GCRF))
    elem = cart2kep(X1_true)
    a = float(elem[0])    
    mean_motion = np.sqrt(GME/a**3.)
    state_params = {}
    state_params['mean_motion'] = mean_motion
    
    # Initial Conditions
    dx = 0.02e-3
    dy = 0.01e-3
    dz = 0.005e-3
    Xo = np.zeros((6,1))
    Xo[3] = dx
    Xo[4] = dy
    Xo[5] = dz
    
    # Analytic Calculations
    beta = pi/2.
    Bo = -dz/mean_motion
    xoff = 2.*dy/mean_motion
    yoff = -2.*dx/mean_motion
    alpha = atan2(dx, 2.*dy)
    
    Ao = 0.
    if dx != 0.:
        Ao = -dx/(mean_motion*sin(alpha))
    if dy != 0.:
        Ao = -2.*dy/(mean_motion*cos(alpha))
    
    # Analtyic results
    xt = Ao*np.cos(mean_motion*tvec + alpha) + xoff
    yt = -2.*Ao*np.sin(mean_motion*tvec + alpha) - 1.5*mean_motion*tvec*xoff + yoff
    zt = Bo*np.cos(mean_motion*tvec + beta)
    
    dxt = -mean_motion*Ao*np.sin(mean_motion*tvec + alpha)
    dyt = -2.*mean_motion*Ao*np.cos(mean_motion*tvec + alpha) - 1.5*mean_motion*xoff
    dzt = -mean_motion*Bo*np.sin(mean_motion*tvec + beta)
    
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'ode'
    int_params['ode_integrator'] = 'dop853'
    int_params['intfcn'] = ode_lincw
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'sec'
    
    
    # Generate numerical output
    X_numerical = np.zeros((len(tvec), 6))
    X = Xo.copy()
    for kk in range(len(tvec)):
        
        if kk > 0:
            tin = [tvec[kk-1], tvec[kk]]
            tout, Xout = general_dynamics(X, tin, state_params, int_params)
            X = Xout[-1,:].reshape(6, 1)
        
        X_numerical[kk,:] = X.flatten()
        
    # Generate Plots
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, X_numerical[:,0], 'b.')
    plt.plot(thrs, xt, 'r.')
    plt.ylabel('x [km]')
    plt.title('RIC Positions')
    plt.subplot(3,1,2)
    plt.plot(thrs, X_numerical[:,1], 'b.')
    plt.plot(thrs, yt, 'r.')
    plt.ylabel('y [km]')
    plt.subplot(3,1,3)
    plt.plot(thrs, X_numerical[:,2], 'b.')
    plt.plot(thrs, zt, 'r.')
    plt.ylabel('z [km]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, X_numerical[:,3], 'b.')
    plt.plot(thrs, dxt, 'r.')
    plt.ylabel('x [km/s]')
    plt.title('RIC Velocities')
    plt.subplot(3,1,2)
    plt.plot(thrs, X_numerical[:,4], 'b.')
    plt.plot(thrs, dyt, 'r.')
    plt.ylabel('y [km/s]')
    plt.subplot(3,1,3)
    plt.plot(thrs, X_numerical[:,5], 'b.')
    plt.plot(thrs, dzt, 'r.')
    plt.ylabel('z [km/s]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, xt - X_numerical[:,0], 'k.')
    plt.ylabel('x [km]')
    plt.title('Analytic - Numeric CW Errors')
    plt.subplot(3,1,2)
    plt.plot(thrs, yt - X_numerical[:,1], 'k.')
    plt.ylabel('y [km]')
    plt.subplot(3,1,3)
    plt.plot(thrs, zt - X_numerical[:,2], 'k.')
    plt.ylabel('z [km]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, dxt - X_numerical[:,3], 'k.')
    plt.ylabel('x [km/s]')
    plt.title('Analytic - Numeric CW Errors')
    plt.subplot(3,1,2)
    plt.plot(thrs, dyt - X_numerical[:,4], 'k.')
    plt.ylabel('y [km/s]')
    plt.subplot(3,1,3)
    plt.plot(thrs, dzt - X_numerical[:,5], 'k.')
    plt.ylabel('z [km/s]')
    plt.xlabel('Time [hours]')
        
    
    plt.show()
    
    
    return


def unit_test_dy2rho():
    
    # State and Measurement Times
    
    # Initial Time
    t0 = '2021-09-10T04:55:00.000'
    UTC0 = datetime.strptime(t0, '%Y-%m-%dT%H:%M:%S.%f')
    
    # Measurement Times
    time_list = ['2021-09-10T10:30:06.160', '2021-09-10T10:30:11.490',
                 '2021-09-10T10:30:16.812', '2021-09-10T10:30:22.178',
                 '2021-09-10T10:30:27.492', '2021-09-10T10:30:32.805',
                 '2021-09-10T10:30:38.130', '2021-09-10T10:30:43.478',
                 '2021-09-10T10:30:48.804', '2021-09-10T10:30:54.119',
                 '2021-09-10T10:30:59.435', '2021-09-10T10:31:04.745',
                 '2021-09-10T10:31:10.048', '2021-09-10T10:31:15.362',
                 '2021-09-10T10:31:20.727', '2021-09-10T10:31:26.070',
                 '2021-09-10T10:31:31.393', '2021-09-10T10:31:36.735',
                 '2021-09-10T10:31:42.057', '2021-09-10T10:31:47.377',
                 '2021-09-10T10:31:52.697', '2021-09-10T10:31:58.005',
                 '2021-09-10T10:32:03.351', '2021-09-10T10:32:08.653',
                 '2021-09-10T10:32:13.973', '2021-09-10T10:32:19.335',
                 '2021-09-10T10:32:24.707', '2021-09-10T10:32:30.028',
                 '2021-09-11T10:25:39.709', '2021-09-11T10:25:47.024',
                 '2021-09-11T10:25:54.332', '2021-09-11T10:26:01.647',
                 '2021-09-11T10:26:08.990', '2021-09-11T10:26:16.339',
                 '2021-09-11T10:26:23.647', '2021-09-11T10:26:30.961',
                 '2021-09-11T10:26:38.276', '2021-09-11T10:26:45.603',
                 '2021-09-11T10:26:52.899', '2021-09-11T10:27:00.220',
                 '2021-09-11T10:27:07.522', '2021-09-11T10:27:14.833',
                 '2021-09-11T10:27:22.158', '2021-09-11T10:27:29.460',
                 '2021-09-11T10:27:36.771', '2021-09-11T10:27:44.076']
    
    
    
    tvec = np.arange(0., 3600.*36. + 1., 10.)
    tk_list = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    
    time_list_dt = []
    for ii in range(len(time_list)):
        
        ti = time_list[ii]
        ti_dt = datetime.strptime(ti, '%Y-%m-%dT%H:%M:%S.%f')
        tk_list.append(ti_dt)
        time_list_dt.append(ti_dt)
        
    tk_list = sorted(tk_list)
    
    meas_inds = [tk_list.index(ti) for ti in time_list_dt]
        
    
    
    # Initial State
    obj_id = 47967
    obj_id_list = [obj_id]
    UTC_list = [UTC0]
    tle_dict = {}
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict, prev_flag=True,
                                 offline_flag=False, frame_flag=True,
                                 username='steve.gehly@gmail.com',
                                 password='SpaceTrackPword!')

    
    r_GCRF = output_state[obj_id]['r_GCRF'][0]
    v_GCRF = output_state[obj_id]['v_GCRF'][0]
    
    X1_true = np.concatenate((r_GCRF, v_GCRF))
    elem = cart2kep(X1_true)
    a = float(elem[0])    
    mean_motion = np.sqrt(GME/a**3.)
    

    # Delta-V vector for object 2 [km/s]
    dx = 0.
    dy = 0.02e-3
    dz = 0.
    rho_ric = np.zeros((3,1))
    drho_ric = np.reshape([dx, dy, dz], (3,1))
    drho_eci = ric2eci_vel(r_GCRF, v_GCRF, rho_ric, drho_ric)
    v2_GCRF = v_GCRF + drho_eci
    
    X2_true = np.concatenate((r_GCRF, v2_GCRF))
    
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = get_XYs2006_alldata()  

    # Sensor Data    
    sigma_dict = {}
    sigma_dict['rho'] = 0.001
    sigma_dict['delta'] = 1.*arcsec2rad
    
    sensor_params = define_sensors(['CMU Falcon'])
    sensor_params['eop_alldata'] = eop_alldata
    sensor_params['XYs_df'] = XYs_df
    
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
    state_params['mean_motion'] = mean_motion
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'ode'
    int_params['ode_integrator'] = 'dop853'
    int_params['intfcn'] = ode_twobody
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'
    
    
    # Generate truth and measurements
    X1 = X1_true.copy()
    X2 = X2_true.copy()
    numeric_delta = []
    numeric_rhofp = []
    analytic_delta = []
    analytic_rhofp = []
    check_delta = []
    check_rhofp = []
    dy_calc = []
    for kk in range(len(tk_list)):
        
        if kk > 0:
            tin = [tk_list[kk-1], tk_list[kk]]
            tout, X1out = general_dynamics(X1, tin, state_params, int_params)
            X1 = X1out[-1,:].reshape(6, 1)
            
            tout, X2out = general_dynamics(X2, tin, state_params, int_params)
            X2 = X2out[-1,:].reshape(6, 1)
            
        if kk not in meas_inds:
            continue
        
        rc_vect = X1[0:3]
        vc_vect = X1[3:6]
        rho_eci = X2[0:3] - X1[0:3]
        drho_eci = X2[3:6] - X1[3:6]
        rho_ric = eci2ric(rc_vect, vc_vect, rho_eci)
        drho_ric = eci2ric_vel(rc_vect, vc_vect, rho_ric, drho_eci)
        
        X = np.concatenate((rho_ric, drho_ric))
        
        UTC = tk_list[kk]
        EOP_data = get_eop_data(eop_alldata, UTC)
          
        # Compute RA/DEC of both objects
        sensor = sensor_params['CMU Falcon']
        meas_types = ['ra', 'dec', 'rg']
        radecrg1 = compute_measurement(X1, state_params, sensor, UTC,
                                     EOP_data, XYs_df, meas_types)
        
        radecrg2 = compute_measurement(X2, state_params, sensor, UTC,
                                     EOP_data, XYs_df, meas_types)
        
        meas_types = ['az', 'el']
        azel1 = compute_measurement(X1, state_params, sensor, UTC,
                                    EOP_data, XYs_df, meas_types)
        
        # Compute unit vectors, delta angle, rho_fp
        print('\n\nUTC: ', UTC)
        print('X1', X1)
        print('X2', X2)
        print('rho_ric', rho_ric)
        print('az deg', azel1[0]*180/pi)
        print('el deg', azel1[1]*180/pi)
        
        
        rho_hat1 = radec2losvec(radecrg1[0], radecrg1[1])
        rho_hat2 = radec2losvec(radecrg2[0], radecrg2[1])
        rho_los = float(radecrg1[2])
        
        print('rho_hat1', rho_hat1)
        print('rho_hat2', rho_hat2)
        print('rho_los', rho_los)
        
        delta_true = compute_delta(rho_hat1, rho_hat2)
        rho_fp = delta_true*rho_los
        
        numeric_delta.append(delta_true)
        numeric_rhofp.append(rho_fp)
        
        # Use analytic formulas to compute x,y,z        
        tk_sec = (UTC - UTC0).total_seconds()        

        beta = pi/2.
        Bo = -dz/mean_motion
        xoff = 2.*dy/mean_motion
        yoff = -2.*dx/mean_motion
        alpha = atan2(dx, 2.*dy)
        
        Ao = 0.
        if dx != 0.:
            Ao = -dx/(mean_motion*sin(alpha))
        if dy != 0.:
            Ao = -2.*dy/(mean_motion*cos(alpha))
        
        # Analtyic results
        xt = Ao*np.cos(mean_motion*tk_sec + alpha) + xoff
        yt = -2.*Ao*np.sin(mean_motion*tk_sec + alpha) - 1.5*mean_motion*tk_sec*xoff + yoff
        zt = Bo*np.cos(mean_motion*tk_sec + beta)
        
        print('\n\n')
        print('numeric x y z')
        print(rho_ric[0], rho_ric[1], rho_ric[2])
        print('analytic x y z')
        print(xt, yt, zt)
        
        uhat_los = rho_hat1  
        rc = np.linalg.norm(rc_vect)
        rhat = rc_vect/rc
        
        psi = compute_psi(uhat_los, rhat)
        uhat_psi_eci = compute_uhat_psi(uhat_los, rhat)
        uhat_psi_ric = eci2ric(rc_vect, vc_vect, uhat_psi_eci)
        phi = compute_phi(uhat_psi_ric)
        
        r_ric = np.reshape([xt, yt, zt], (3,1))
        r_fp = ric2fp(psi, phi, r_ric)
        rho_fp = np.linalg.norm(r_fp[1:3])
        delta = rho_fp/rho_los
        
        analytic_delta.append(delta)
        analytic_rhofp.append(rho_fp)
        
        # Compute rho_fp using only dy
        xt = -2.*dy/mean_motion*cos(mean_motion*tk_sec) + 2.*dy/mean_motion
        yt = 4.*dy/mean_motion*sin(mean_motion*tk_sec) - 3.*dy*tk_sec
        print('check x y')
        print(xt, yt)
        
        r_fp_check = np.array([[xt*cos(psi) - yt*sin(psi)*sin(phi)],
                               [yt*cos(phi)],
                               [-xt*sin(psi) - yt*cos(psi)*sin(phi)]])
        
        print('\n')
        print('r_fp', r_fp)
        print('r_fp_check', r_fp_check)
        
        rho_fp_check = np.sqrt(yt**2.*cos(phi)**2. + xt**2.*sin(psi)**2. + yt**2.*cos(psi)**2.*sin(phi)**2. + 2.*xt*yt*cos(psi)*sin(psi)*sin(phi))
        print('rho_fp_check', rho_fp_check)
        
        
        xcoeff = 2./mean_motion*(1. - cos(mean_motion*tk_sec))
        ycoeff = 4./mean_motion*sin(mean_motion*tk_sec) - 3.*tk_sec
        
        print('x', xt, dy*xcoeff)
        print('y', yt, dy*ycoeff)
        
        
        
        a1 = cos(phi)**2.*ycoeff**2.
        a2 = sin(psi)**2.*xcoeff**2.
        a3 = cos(psi)**2.*sin(phi)**2.*ycoeff**2.
        a4 = 2.*xcoeff*ycoeff*cos(psi)*sin(psi)*sin(phi)
        
        rho_fp = np.sqrt(dy**2.*(a1 + a2 + a3 + a4))
        delta = rho_fp/rho_los
        
        check_delta.append(delta)
        check_rhofp.append(rho_fp)
        
#        print('\n')
#        print('true rho fp', numeric_rhofp[0])
#        print('analytic rho fp', analytic_rhofp[0])
#        print('check rho fp', check_rhofp[0])
        
        rho_fp_calc = delta_true*rho_los
        dy_calc.append(np.sqrt(rho_fp_calc**2./(a1 + a2 + a3 + a4)))
        
    
    
    print('\n\nrho fp values')
    print('numeric rho fp', numeric_rhofp)
    print('analytic rho fp', analytic_rhofp)
    print('check rho fp', check_rhofp)
    
    print('\n\ndy calculations')
    print(dy_calc)
    print('dy mean [m/s]: ', np.mean(dy_calc)*1000.)
    print('dy STD [m/s]: ', np.std(dy_calc)*1000.)

    return


###############################################################################
# Estimation
###############################################################################


def m2_formation_setup(meas_file, meas_output='rho'):
    
    # Initial Time
    t0 = '2021-09-10T04:55:00.000'
    UTC0 = datetime.strptime(t0, '%Y-%m-%dT%H:%M:%S.%f')
    
    # Simplified measurement times
    tvec = np.arange(0., 3600.*36. + 1., 10.)
    tk_list = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    
#    meas_inds = list(range(len(tk_list)))
    meas_inds = []
    for nn in range(3):
        inds = list(nn*360*12 + np.arange(0,12,1))
        meas_inds.extend(inds)
    
    
    # Initial State
    obj_id = 47967
    obj_id_list = [obj_id]
    UTC_list = [UTC0]
    tle_dict = {}
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict, prev_flag=True,
                                 offline_flag=False, frame_flag=True,
                                 username='steve.gehly@gmail.com',
                                 password='SpaceTrackPword!')
    
    print(output_state)
    
    r_GCRF = output_state[obj_id]['r_GCRF'][0]
    v_GCRF = output_state[obj_id]['v_GCRF'][0]
    
    X1_true = np.concatenate((r_GCRF, v_GCRF))
    elem = cart2kep(X1_true)
    a = float(elem[0])
    mean_motion = np.sqrt(GME/a**3.)
    

    # Delta-V vector for object 2 [km/s]
    dx = 0.
    dy = 0.02e-3
    dz = 0.
    rho_ric = np.zeros((3,1))
    drho_ric = np.reshape([dx, dy, dz], (3,1))
    drho_eci = ric2eci_vel(r_GCRF, v_GCRF, rho_ric, drho_ric)
    v2_GCRF = v_GCRF + drho_eci
    
    X2_true = np.concatenate((r_GCRF, v2_GCRF))
    
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = get_XYs2006_alldata()  

    # Sensor Data    
    sigma_dict = {}
    sigma_dict['rho'] = 0.001
    sigma_dict['delta'] = 1.*arcsec2rad
    
    sensor_params = define_sensors(['CMU Falcon'])
    sensor_params['eop_alldata'] = eop_alldata
    sensor_params['XYs_df'] = XYs_df
        
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
    state_params['mean_motion'] = mean_motion
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'ode'
    int_params['ode_integrator'] = 'dop853'
    int_params['intfcn'] = ode_twobody
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'
    
    
    # Generate truth and measurements
    truth_dict = {}
    meas_dict = {}
    meas_dict['tk_list'] = []
    meas_dict['Yk_list'] = []
    meas_dict['sensor_id_list'] = []
    X1 = X1_true.copy()
    X2 = X2_true.copy()
    for kk in range(len(tk_list)):
        
        if kk > 0:
            tin = [tk_list[kk-1], tk_list[kk]]
#            print('tin', tin)
            tout, X1out = general_dynamics(X1, tin, state_params, int_params)
            X1 = X1out[-1,:].reshape(6, 1)
            
            tout, X2out = general_dynamics(X2, tin, state_params, int_params)
            X2 = X2out[-1,:].reshape(6, 1)
        
        rc_vect = X1[0:3]
        vc_vect = X1[3:6]
        rho_eci = X2[0:3] - X1[0:3]
        drho_eci = X2[3:6] - X1[3:6]
        rho_ric = eci2ric(rc_vect, vc_vect, rho_eci)
        drho_ric = eci2ric_vel(rc_vect, vc_vect, rho_ric, drho_eci)
        
        X = np.concatenate((rho_ric, drho_ric))
        truth_dict[tk_list[kk]] = X
        
        if kk not in meas_inds:
            continue
        
        UTC = tk_list[kk]
        EOP_data = get_eop_data(eop_alldata, UTC)
        
        if meas_output == 'rho':
            Yk = np.zeros((1,1))
            Yk[0] = np.linalg.norm(rho_ric) + np.random.randn()*sigma_dict['rho']
        
        if meas_output == 'delta':
            
            # Compute RA/DEC of both objects
            sensor = sensor_params['CMU Falcon']
            meas_types = ['ra', 'dec']
            radec1 = compute_measurement(X1, state_params, sensor, UTC,
                                         EOP_data, XYs_df, meas_types)
            
            radec2 = compute_measurement(X2, state_params, sensor, UTC,
                                         EOP_data, XYs_df, meas_types)
            
            # Compute unit vectors and delta angle
            rho_hat1 = radec2losvec(radec1[0], radec1[1])
            rho_hat2 = radec2losvec(radec2[0], radec2[1])
            delta = compute_delta(rho_hat1, rho_hat2)
            
            Yk = np.zeros((1,1))
            Yk[0] = delta + np.random.randn()*sigma_dict['delta']
            if Yk[0] < 0.:
                Yk[0] = delta + abs(np.random.randn()*sigma_dict['delta'])
        
        meas_dict['tk_list'].append(UTC)
        meas_dict['Yk_list'].append(Yk)
        meas_dict['sensor_id_list'].append(1)
    
    
    print(meas_dict)
    
    # Plot data
    tplot = [(tk - tk_list[0]).total_seconds()/3600. for tk in tk_list]
    xplot = []
    yplot = []
    zplot = []
    dxplot = []
    dyplot = []
    dzplot = []
    for tk in tk_list:
        X = truth_dict[tk]
        xplot.append(X[0]*1000.)
        yplot.append(X[1]*1000.)
        zplot.append(X[2]*1000.)
        dxplot.append(X[3]*1000.)
        dyplot.append(X[4]*1000.)
        dzplot.append(X[5]*1000.)
        
    meas_tk = meas_dict['tk_list']
    meas_tplot = [(tk - tk_list[0]).total_seconds()/3600. for tk in meas_tk]
#    meas_sensor_id = meas_dict['sensor_id_list']
#    meas_sensor_index = [sensor_id_list.index(sensor_id) for sensor_id in meas_sensor_id]
    meas_plot0 = np.asarray([float(Yk[0]) for Yk in meas_dict['Yk_list']])
    

    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tplot, xplot, 'k.')
    plt.ylabel('Radial [m]')
    plt.title('True RIC Positions')
    plt.subplot(3,1,2)
    plt.plot(tplot, yplot, 'k.')
    plt.ylabel('In-Track [m]')
    plt.subplot(3,1,3)
    plt.plot(tplot, zplot, 'k.')
    plt.ylabel('Cross-Track [m]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tplot, dxplot, 'k.')
    plt.ylabel('Radial [m/s]')
    plt.title('True RIC Velocities')
    plt.subplot(3,1,2)
    plt.plot(tplot, dyplot, 'k.')
    plt.ylabel('In-Track [m/s]')
    plt.subplot(3,1,3)
    plt.plot(tplot, dzplot, 'k.')
    plt.ylabel('Cross-Track [m/s]')
    plt.xlabel('Time [hours]')
    
    
    
    plt.figure()
    
    if meas_output == 'rho':
        plt.plot(meas_tplot, meas_plot0*1000., 'k.')
        plt.ylabel('rho [m]')
    
    if meas_output == 'delta':
        plt.plot(meas_tplot, meas_plot0*(1./arcsec2rad), 'k.')
        plt.ylabel('Delta [arcsec]')
    
    plt.title('Measurements')
    plt.xlabel('Time [hours]')
    
                
    plt.show()   
    
    state_dict = []
    meas_fcn = []
    
    
    pklFile = open(meas_file , 'wb')
    pickle.dump( [state_dict, state_params, int_params, meas_fcn, meas_dict, sensor_params, truth_dict], pklFile, -1 )
    pklFile.close()
    
    
    return


def compute_gamma_til(chi0, tk, state_params, output='rho'):
    
    mean_motion = state_params['mean_motion']
    
    L = chi0.shape[0]
    
    if output == 'rho':
        gamma_til_k = np.zeros((1, 2*L+1))
        
    if output == 'delta':
        gamma_til_k = np.zeros((1, 2*L+1))
        
    if output == 'pos':
        gamma_til_k = np.zeros((3, 2*L+1))
        
    if output == 'vel':
        gamma_til_k = np.zeros((3, 2*L+1))
        
    
    for ii in range(2*L+1):
        
        # Retrieve states
        Xi = chi0[:,ii]
        dx = float(Xi[0])
        dy = float(Xi[1])
        dz = float(Xi[2])
        
#        print('ii', ii)
#        print(Xi)
#        print(dx, dy, dz)
        
        # Compute constants
        beta = pi/2.
        Bo = -dz/mean_motion
        xoff = 2.*dy/mean_motion
        yoff = -2.*dx/mean_motion
        alpha = atan2(dx, 2.*dy)
        
        Ao = 0.
        if dx != 0.:
            Ao = -dx/(mean_motion*sin(alpha))
        if dy != 0.:
            Ao = -2.*dy/(mean_motion*cos(alpha))
            
        # Compute states at current time
        xt = Ao*np.cos(mean_motion*tk + alpha) + xoff
        yt = -2.*Ao*sin(mean_motion*tk + alpha) - 1.5*mean_motion*tk*xoff + yoff
        zt = Bo*cos(mean_motion*tk + beta)
        
        dxt = -mean_motion*Ao*np.sin(mean_motion*tk + alpha)
        dyt = -2.*mean_motion*Ao*np.cos(mean_motion*tk + alpha) - 1.5*mean_motion*xoff
        dzt = -mean_motion*Bo*np.sin(mean_motion*tk + beta)
        
#        print(xt, yt, zt)
        
        # Compute output and store
        if output == 'rho':
            rho_rel = np.linalg.norm([xt, yt, zt])
            gamma_til_k[:,ii] = float(rho_rel)
            
        if output == 'pos':
            gamma_til_k[0,ii] = float(xt)
            gamma_til_k[1,ii] = float(yt)
            gamma_til_k[2,ii] = float(zt)
            
        if output == 'vel':
            gamma_til_k[0,ii] = float(dxt)
            gamma_til_k[1,ii] = float(dyt)
            gamma_til_k[2,ii] = float(dzt)
            
        if output == 'delta':
            X_chief = state_params['X_chief']
            rc_GCRF = X_chief[0:3].reshape(3,1)
            vc_GCRF = X_chief[3:6].reshape(3,1)
            sensor_gcrf = state_params['sensor_gcrf']
            
            rho_los_gcrf = rc_GCRF - sensor_gcrf
            rho_los = np.linalg.norm(rho_los_gcrf)
            rc = np.linalg.norm(rc_GCRF)
            
            uhat_los = rho_los_gcrf/rho_los
            rhat = rc_GCRF/rc
            
            psi = compute_psi(uhat_los, rhat)
            uhat_psi_eci = compute_uhat_psi(uhat_los, rhat)
            uhat_psi_ric = eci2ric(rc_GCRF, vc_GCRF, uhat_psi_eci)
            phi = compute_phi(uhat_psi_ric)
            
            r_ric = np.reshape([xt, yt, zt], (3,1))
            r_fp = ric2fp(psi, phi, r_ric)
            rho_fp = np.linalg.norm(r_fp[1:3])
            delta = rho_fp/rho_los
            
            gamma_til_k[:,ii] = float(delta)


    return gamma_til_k


def compute_gamma_til_dy(chi0, tk, state_params, output='rho'):
    
    mean_motion = state_params['mean_motion']
    
    L = chi0.shape[0]
    
    if output == 'rho':
        gamma_til_k = np.zeros((1, 2*L+1))
        
    if output == 'delta':
        gamma_til_k = np.zeros((1, 2*L+1))
        
    if output == 'pos':
        gamma_til_k = np.zeros((3, 2*L+1))
        
    if output == 'vel':
        gamma_til_k = np.zeros((3, 2*L+1))
        
    
    for ii in range(2*L+1):
        
        # Retrieve states
        Xi = chi0[:,ii]
        dy = float(Xi[0])
        
        dx = 0.
        dz = 0.
        
#        print('ii', ii)
#        print(Xi)
#        print(dx, dy, dz)
        
        # Compute constants
        beta = pi/2.
        Bo = -dz/mean_motion
        xoff = 2.*dy/mean_motion
        yoff = -2.*dx/mean_motion
        alpha = atan2(dx, 2.*dy)
        
        Ao = 0.
        if dx != 0.:
            Ao = -dx/(mean_motion*sin(alpha))
        if dy != 0.:
            Ao = -2.*dy/(mean_motion*cos(alpha))
            
        # Compute states at current time
        xt = Ao*np.cos(mean_motion*tk + alpha) + xoff
        yt = -2.*Ao*sin(mean_motion*tk + alpha) - 1.5*mean_motion*tk*xoff + yoff
        zt = Bo*cos(mean_motion*tk + beta)
        
        dxt = -mean_motion*Ao*np.sin(mean_motion*tk + alpha)
        dyt = -2.*mean_motion*Ao*np.cos(mean_motion*tk + alpha) - 1.5*mean_motion*xoff
        dzt = -mean_motion*Bo*np.sin(mean_motion*tk + beta)
        
#        print(xt, yt, zt)
        
        # Compute output and store
        if output == 'rho':
            rho_rel = np.linalg.norm([xt, yt, zt])
            gamma_til_k[:,ii] = float(rho_rel)
            
        if output == 'pos':
            gamma_til_k[0,ii] = float(xt)
            gamma_til_k[1,ii] = float(yt)
            gamma_til_k[2,ii] = float(zt)
            
        if output == 'vel':
            gamma_til_k[0,ii] = float(dxt)
            gamma_til_k[1,ii] = float(dyt)
            gamma_til_k[2,ii] = float(dzt)
            
        if output == 'delta':
            X_chief = state_params['X_chief']
            rc_GCRF = X_chief[0:3].reshape(3,1)
            vc_GCRF = X_chief[3:6].reshape(3,1)
            sensor_gcrf = state_params['sensor_gcrf']
            
            rho_los_gcrf = rc_GCRF - sensor_gcrf
            rho_los = np.linalg.norm(rho_los_gcrf)
            rc = np.linalg.norm(rc_GCRF)
            
            uhat_los = rho_los_gcrf/rho_los
            rhat = rc_GCRF/rc
            
            psi = compute_psi(uhat_los, rhat)
            uhat_psi_eci = compute_uhat_psi(uhat_los, rhat)
            uhat_psi_ric = eci2ric(rc_GCRF, vc_GCRF, uhat_psi_eci)
            phi = compute_phi(uhat_psi_ric)
            
            r_ric = np.reshape([xt, yt, zt], (3,1))
            r_fp = ric2fp(psi, phi, r_ric)
            rho_fp = np.linalg.norm(r_fp[1:3])
            delta = rho_fp/rho_los
            
            gamma_til_k[:,ii] = float(delta)


    return gamma_til_k


def unscented_batch_cw(meas_file, meas_input='rho'):
    
    
    # Initial Time
    t0 = '2021-09-10T04:55:00.000'
    UTC0 = datetime.strptime(t0, '%Y-%m-%dT%H:%M:%S.%f')
    
    # Initial State
    obj_id = 47967
    obj_id_list = [obj_id]
    UTC_list = [UTC0]
    tle_dict = {}
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict, prev_flag=True,
                                 offline_flag=False, frame_flag=True,
                                 username='steve.gehly@gmail.com',
                                 password='SpaceTrackPword!')
    
    print(output_state)
    
    r_GCRF = output_state[obj_id]['r_GCRF'][0]
    v_GCRF = output_state[obj_id]['v_GCRF'][0]
    
    X_chief = np.concatenate((r_GCRF, v_GCRF))
    
    
    
    # Load Data
    fname = os.path.join(meas_file)
    pklFile = open(fname, 'rb' )
    data = pickle.load( pklFile )
    state_dict = data[0]
    state_params = data[1]
    int_params = data[2]
    meas_fcn = data[3]
    meas_dict = data[4]
    sensor_params = data[5]
    truth_dict = data[6]
    pklFile.close()
    
    # Retrieve and setup data
    eop_alldata = sensor_params['eop_alldata']
    XYs_df = sensor_params['XYs_df']
    relative_state_params = state_params
    
    # Initial State
    tk_truth = sorted(truth_dict.keys())
    t0 = tk_truth[0]                    
    Xo = np.reshape([0.01e-4, 0.01e-4, 0.01e-4], (3,1))
    Po = np.diag(np.ones(3,)*1e-12)
    L = len(Xo)

    # Unscented Parameters
    alpha = 1e-4
    beta = 2.
    kappa = 3. - L
    lam = alpha**2.*(L + kappa) - L
    gam = np.sqrt(L + lam)
    Wm = 1./(2.*(L + lam)) * np.ones((1, 2*L))
    Wm = list(Wm.flatten())
    Wc = copy.copy(Wm)
    Wm.insert(0, lam/(L + lam))
    Wc.insert(0, lam/(L + lam) + (1 - alpha**2 + beta))
    Wm = np.asarray(Wm)
    diagWc = np.diag(Wc)
    
    
    # Measurements
    tk_list = meas_dict['tk_list']
    Yk_list = meas_dict['Yk_list']
    N = len(tk_list)
    
    # Block diagonal Rk matrix
    
    if meas_input == 'rho':
        Rk = np.array([[1e-6]])
        p = 1
        
    if meas_input == 'delta':
        Rk = np.array([[ (1.*arcsec2rad)**2. ]])
        p = 1
    
    Rk_full = np.kron(np.eye(N), Rk)
#    invRk_full = np.linalg.inv(Rk_full)
    
    # Initialize
    maxiters = 10   
    X = Xo.copy()
    P = Po.copy()
    invPo = np.linalg.inv(Po)
    
    # Begin Loop
    iters = 0
    diff = 1
    conv_crit = 1e-8
    while diff > conv_crit:

        # Increment loop counter and exit if necessary
        iters += 1
        if iters > maxiters:
            iters -= 1
            print('Solution did not converge in ', iters, ' iterations')
            print('Last xo_hat magnitude: ', diff)
            break

        # Reset P every iteration???
        P = Po.copy()        
        
        # Compute Sigma Points
        sqP = np.linalg.cholesky(P)
        Xrep = np.tile(X, (1, L))
        chi0 = np.concatenate((X, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
        chi_diff0 = chi0 - np.dot(X, np.ones((1, 2*L+1)))
        
#        print(chi0)
        
        # Loop over times
        meas_ind = 0
        Y_bar = np.zeros((p*N, 1))
        Y_til = np.zeros((p*N, 1))
        gamma_til_mat = np.zeros((p*N, 2*L+1)) 
        
        for kk in range(len(tk_list)):
            
            # Current time
            tk = tk_list[kk]
            tk_sec = (tk - t0).total_seconds()
            
            # Compute current chief object state vector
            if kk > 0:
                tin = [tk_list[kk-1], tk_list[kk]]
                tout, Xout = general_dynamics(X_chief, tin, state_params, int_params)
                X_chief = Xout[-1,:].reshape(6, 1)
                
            # Compute current sensor location
            EOP_data = get_eop_data(eop_alldata, tk)
            sensor_itrf = sensor_params['CMU Falcon']['site_ecef']
            sensor_gcrf, dum = itrf2gcrf(sensor_itrf, np.zeros((3,1)), tk, EOP_data,
                                         XYs_df)
            
            # Computed measurement sigma points
            relative_state_params['X_chief'] = X_chief
            relative_state_params['sensor_gcrf'] = sensor_gcrf
            gamma_til_k = compute_gamma_til(chi0, tk_sec, relative_state_params, meas_input)
            ybar_k = np.dot(gamma_til_k, Wm.T)
            ybar_k = np.reshape(ybar_k, (p,1))
            
#            print(gamma_til_k)
#            print('tk_sec', tk_sec)
#            print(gamma_til_k)
#            print(ybar_k)
#            print(Yk_list[kk])
            
#            if kk > 2:
#                mistake
            
            # Accumulate measurements and computed measurements
            Y_til[meas_ind:meas_ind+p] = Yk_list[kk]
            Y_bar[meas_ind:meas_ind+p] = ybar_k
            gamma_til_mat[meas_ind:meas_ind+p, :] = gamma_til_k
            
            # Increment measurement index
            meas_ind += p
            
        # Compute covariances
        Y_diff = gamma_til_mat - np.dot(Y_bar, np.ones((1, 2*L+1)))
        Pyy = np.dot(Y_diff, np.dot(diagWc, Y_diff.T)) + Rk_full
        Pxy = np.dot(chi_diff0, np.dot(diagWc, Y_diff.T))        

        # Compute Kalman Gain
        cholPyy = np.linalg.inv(np.linalg.cholesky(Pyy))
        invPyy = np.dot(cholPyy.T, cholPyy)
        K = np.dot(Pxy, invPyy)
           
        # Regular
        #P = P - np.dot(K, np.dot(Pyy, K.T))
        
        # Joseph Form
        P1 = (np.identity(L) - np.dot(np.dot(K, np.dot(Pyy, K.T)), invPo))
        P2 = np.dot(P1, np.dot(Po, P1.T)) + np.dot(K, np.dot(Rk_full, K.T))
        
        P = P2
        
#        # Compute Kalman Gain
#        Hi = np.dot(invPo, Pxy).T
#        K1 = np.dot(P, np.dot(Hi.T, invRk_full))
#        
#        check = K1 - K
#        #print check
#        print 'K check max diff'
#        print np.max(check)
#        #mistake
        
        # Compute updated state and covariance    
        X += np.dot(K, Y_til-Y_bar)
        resids = Y_til - Y_bar
        
        diff = np.linalg.norm(np.dot(K, Y_til-Y_bar))
        
        print('Iteration Number: ', iters)
        print('X', X)
        print('diff = ', diff)
        
        
        
    print(X)
    print(P)
    print('resids', np.mean(resids), np.std(resids))
    
    
    
    # Compute position and velocity truth values over time with covar
    tk_truth = sorted(truth_dict.keys())
    
    sqP = np.linalg.cholesky(P)
    Xrep = np.tile(X, (1, L))
    chi0 = np.concatenate((X, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
    
    xerr = np.zeros((len(tk_truth),))
    yerr = np.zeros((len(tk_truth),))
    zerr = np.zeros((len(tk_truth),))
    dxerr = np.zeros((len(tk_truth),))
    dyerr = np.zeros((len(tk_truth),))
    dzerr = np.zeros((len(tk_truth),))
    
    sig_x = np.zeros((len(tk_truth),))
    sig_y = np.zeros((len(tk_truth),))
    sig_z = np.zeros((len(tk_truth),))
    sig_dx = np.zeros((len(tk_truth),))
    sig_dy = np.zeros((len(tk_truth),))
    sig_dz = np.zeros((len(tk_truth),))
    
    xerr_meas = np.zeros((len(tk_list),))
    yerr_meas = np.zeros((len(tk_list),))
    zerr_meas = np.zeros((len(tk_list),))
    dxerr_meas = np.zeros((len(tk_list),))
    dyerr_meas = np.zeros((len(tk_list),))
    dzerr_meas = np.zeros((len(tk_list),))
    
    for kk in range(len(tk_truth)):
            
        # Current time
        tk = tk_truth[kk]
        tk_sec = (tk - t0).total_seconds()
        X_truth = truth_dict[tk]
        
        # Computed position sigma points and mean
        output = 'pos'
        gamma_til_k = compute_gamma_til(chi0, tk_sec, state_params, output)
        r_ric = np.dot(gamma_til_k, Wm.T)
        r_ric = np.reshape(r_ric, (3,1))
        
#        print(gamma_til_k.shape)
#        print(r_ric)
#        print(L)
        
        r_diff = gamma_til_k - np.dot(r_ric, np.ones((1, 2*L+1)))
        Pr_ric = np.dot(r_diff, np.dot(diagWc, r_diff.T))
        
        # Computed velocity sigma points and mean
        output = 'vel'
        gamma_til_k = compute_gamma_til(chi0, tk_sec, state_params, output)
        v_ric = np.dot(gamma_til_k, Wm.T)
        v_ric = np.reshape(v_ric, (3,1))
        v_diff = gamma_til_k - np.dot(v_ric, np.ones((1, 2*L+1)))
        Pv_ric = np.dot(v_diff, np.dot(diagWc, v_diff.T))
        
#        print(kk)
#        print(r_ric)
#        print(v_ric)
#        print(X_truth)
#        
#        if kk > 2:
#            mistake
        
        # Store Errors
        xerr[kk] = float(r_ric[0]) - float(X_truth[0])
        yerr[kk] = float(r_ric[1]) - float(X_truth[1])
        zerr[kk] = float(r_ric[2]) - float(X_truth[2])
        dxerr[kk] = float(v_ric[0]) - float(X_truth[3])
        dyerr[kk] = float(v_ric[1]) - float(X_truth[4])
        dzerr[kk] = float(v_ric[2]) - float(X_truth[5])
        
        sig_x[kk] = float(np.sqrt(Pr_ric[0,0]))
        sig_y[kk] = float(np.sqrt(Pr_ric[1,1]))
        sig_z[kk] = float(np.sqrt(Pr_ric[2,2]))
        
        sig_dx[kk] = float(np.sqrt(Pv_ric[0,0]))
        sig_dy[kk] = float(np.sqrt(Pv_ric[1,1]))
        sig_dz[kk] = float(np.sqrt(Pv_ric[2,2]))
        
        
        if tk in tk_list:
            ind = tk_list.index(tk)
            xerr_meas[ind] = float(r_ric[0]) - float(X_truth[0])
            yerr_meas[ind] = float(r_ric[1]) - float(X_truth[1])
            zerr_meas[ind] = float(r_ric[2]) - float(X_truth[2])
            dxerr_meas[ind] = float(v_ric[0]) - float(X_truth[3])
            dyerr_meas[ind] = float(v_ric[1]) - float(X_truth[4])
            dzerr_meas[ind] = float(v_ric[2]) - float(X_truth[5])
            
        
    
    
    # Generate Plots
    Xt0 = truth_dict[t0]
    tvec = [(tk - t0).total_seconds() for tk in tk_truth]
    t_hrs = [(tk - t0).total_seconds()/3600. for tk in tk_truth]
    tmeas_hrs = [(tk - t0).total_seconds()/3600. for tk in tk_list]
    
#    Xk_time, Pk_time = monte_carlo_prop(X, P, tvec)
    
    
    
    if meas_input == 'rho':
        plt.figure()
        plt.plot(tmeas_hrs, resids*1000., 'k.')
        plt.ylabel('Range Resids [m]')
        plt.xlabel('Time [hours]')
    
    if meas_input == 'delta':
        plt.figure()
        plt.plot(tmeas_hrs, resids*(1./arcsec2rad), 'k.')
        plt.ylabel('Delta Resids [arcsec]')
        plt.xlabel('Time [hours]')
        
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, xerr*1000., 'k.')
    plt.plot(tmeas_hrs, xerr_meas*1000., 'b.')
    plt.plot(t_hrs, 3.*sig_x*1000., 'k--')
    plt.plot(t_hrs, -3.*sig_x*1000., 'k--')
    plt.ylabel('x Err [m]')    
    plt.subplot(3,1,2)
    plt.plot(t_hrs, yerr*1000., 'k.')
    plt.plot(t_hrs, 3.*sig_y*1000., 'k--')
    plt.plot(t_hrs, -3.*sig_y*1000., 'k--')
    plt.plot(tmeas_hrs, yerr_meas*1000., 'b.')
    plt.ylabel('y Err [m]')     
    plt.subplot(3,1,3)
    plt.plot(t_hrs, zerr*1000., 'k.')
    plt.plot(t_hrs, 3.*sig_z*1000., 'k--')
    plt.plot(t_hrs, -3.*sig_z*1000., 'k--')
    plt.plot(tmeas_hrs, zerr_meas*1000., 'b.')
    plt.ylabel('z Err [m]') 
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, dxerr*1e6, 'k.')
    plt.plot(tmeas_hrs, dxerr_meas*1e6, 'b.')
    plt.plot(t_hrs, 3.*sig_dx*1e6, 'k--')
    plt.plot(t_hrs, -3.*sig_dx*1e6, 'k--')
    plt.ylabel('dx Err [mm/s]')    
    plt.subplot(3,1,2)
    plt.plot(t_hrs, dyerr*1e6, 'k.')
    plt.plot(t_hrs, 3.*sig_dy*1e6, 'k--')
    plt.plot(t_hrs, -3.*sig_dy*1e6, 'k--')
    plt.plot(tmeas_hrs, dyerr_meas*1e6, 'b.')
    plt.ylabel('dy Err [mm/s]')     
    plt.subplot(3,1,3)
    plt.plot(t_hrs, dzerr*1e6, 'k.')
    plt.plot(t_hrs, 3.*sig_dz*1e6, 'k--')
    plt.plot(t_hrs, -3.*sig_dz*1e6, 'k--')
    plt.plot(tmeas_hrs, dzerr_meas*1e6, 'b.')
    plt.ylabel('dz Err [mm/s]') 
    plt.xlabel('Time [hours]')
    
   
    
    plt.show()
    
    
    
    print('\n\nSeparation Velocity Estimates and Truth')
    print('Radial Est [m/s]:\t\t', '{0:0.2E}'.format(float(X[0]) * 1000.))
    print('Radial STD [m/s]:\t\t', '{0:0.2E}'.format(float(np.sqrt(P[0,0])) * 1000.))
    print('Radial True [m/s]:\t\t', '{0:0.2E}'.format(float(Xt0[3]) * 1000.))
    
    print('In-track Est [m/s]:\t\t', '{0:0.2E}'.format(float(X[1]) * 1000.))
    print('In-track STD [m/s]:\t\t', '{0:0.2E}'.format(float(np.sqrt(P[1,1])) * 1000.))
    print('In-track True [m/s]:\t', '{0:0.2E}'.format(float(Xt0[4]) * 1000.))
    
    print('Cross-track Est [m/s]:\t ', '{0:0.2E}'.format(float(X[2]) * 1000.))
    print('Cross-track STD [m/s]:\t', '{0:0.2E}'.format(float(np.sqrt(P[2,2])) * 1000.))
    print('Cross-track True [m/s]:\t', '{0:0.2E}'.format(float(Xt0[5]) * 1000.))
    

    
    return 


def unscented_batch_dy(meas_file, meas_input='rho'):
    
    
    # Initial Time
    t0 = '2021-09-10T04:55:00.000'
    UTC0 = datetime.strptime(t0, '%Y-%m-%dT%H:%M:%S.%f')
    
    # Initial State
    obj_id = 47967
    obj_id_list = [obj_id]
    UTC_list = [UTC0]
    tle_dict = {}
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict, prev_flag=True,
                                 offline_flag=False, frame_flag=True,
                                 username='steve.gehly@gmail.com',
                                 password='SpaceTrackPword!')
    
    print(output_state)
    
    r_GCRF = output_state[obj_id]['r_GCRF'][0]
    v_GCRF = output_state[obj_id]['v_GCRF'][0]
    
    X_chief = np.concatenate((r_GCRF, v_GCRF))
    
    
    
    # Load Data
    fname = os.path.join(meas_file)
    pklFile = open(fname, 'rb' )
    data = pickle.load( pklFile )
    state_dict = data[0]
    state_params = data[1]
    int_params = data[2]
    meas_fcn = data[3]
    meas_dict = data[4]
    sensor_params = data[5]
    truth_dict = data[6]
    pklFile.close()
    
    # Retrieve and setup data
    eop_alldata = sensor_params['eop_alldata']
    XYs_df = sensor_params['XYs_df']
    relative_state_params = state_params
    
    # Initial State
    tk_truth = sorted(truth_dict.keys())
    t0 = tk_truth[0]                    
    Xo = np.array([[0.02e-3]])
    Po = np.array([[1e-12]])
    L = len(Xo)

    # Unscented Parameters
    alpha = 1.
    beta = 2.
    kappa = 3. - L
    lam = alpha**2.*(L + kappa) - L
    gam = np.sqrt(L + lam)
    Wm = 1./(2.*(L + lam)) * np.ones((1, 2*L))
    Wm = list(Wm.flatten())
    Wc = copy.copy(Wm)
    Wm.insert(0, lam/(L + lam))
    Wc.insert(0, lam/(L + lam) + (1 - alpha**2 + beta))
    Wm = np.asarray(Wm)
    diagWc = np.diag(Wc)
    
    
    # Measurements
    tk_list = meas_dict['tk_list']
    Yk_list = meas_dict['Yk_list']
    N = len(tk_list)
    
    # Block diagonal Rk matrix
    
    if meas_input == 'rho':
        Rk = np.array([[1e-6]])
        p = 1
        
    if meas_input == 'delta':
        Rk = np.array([[ (1.*arcsec2rad)**2. ]])
        p = 1
    
    Rk_full = np.kron(np.eye(N), Rk)
#    invRk_full = np.linalg.inv(Rk_full)
    
    # Initialize
    maxiters = 10   
    X = Xo.copy()
    P = Po.copy()
    invPo = np.linalg.inv(Po)
    
    # Begin Loop
    iters = 0
    diff = 1
    conv_crit = 1e-8
    while diff > conv_crit:

        # Increment loop counter and exit if necessary
        iters += 1
        if iters > maxiters:
            iters -= 1
            print('Solution did not converge in ', iters, ' iterations')
            print('Last xo_hat magnitude: ', diff)
            break

        # Reset P every iteration???
        P = Po.copy()        
        
        # Compute Sigma Points
        sqP = np.linalg.cholesky(P)
        Xrep = np.tile(X, (1, L))
        chi0 = np.concatenate((X, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
        chi_diff0 = chi0 - np.dot(X, np.ones((1, 2*L+1)))
        
#        print(chi0)
        
        # Loop over times
        meas_ind = 0
        Y_bar = np.zeros((p*N, 1))
        Y_til = np.zeros((p*N, 1))
        gamma_til_mat = np.zeros((p*N, 2*L+1)) 
        
        for kk in range(len(tk_list)):
            
            # Current time
            tk = tk_list[kk]
            tk_sec = (tk - t0).total_seconds()
            
            # Compute current chief object state vector
            if kk > 0:
                tin = [tk_list[kk-1], tk_list[kk]]
                tout, Xout = general_dynamics(X_chief, tin, state_params, int_params)
                X_chief = Xout[-1,:].reshape(6, 1)
                
            # Compute current sensor location
            EOP_data = get_eop_data(eop_alldata, tk)
            sensor_itrf = sensor_params['CMU Falcon']['site_ecef']
            sensor_gcrf, dum = itrf2gcrf(sensor_itrf, np.zeros((3,1)), tk, EOP_data,
                                         XYs_df)
            
            # Computed measurement sigma points
            relative_state_params['X_chief'] = X_chief
            relative_state_params['sensor_gcrf'] = sensor_gcrf
            gamma_til_k = compute_gamma_til_dy(chi0, tk_sec, relative_state_params, meas_input)
            ybar_k = np.dot(gamma_til_k, Wm.T)
            ybar_k = np.reshape(ybar_k, (p,1))
            
#            print(gamma_til_k)
#            print('tk_sec', tk_sec)
#            print(gamma_til_k)
#            print(ybar_k)
#            print(Yk_list[kk])
            
#            if kk > 2:
#                mistake
            
            # Accumulate measurements and computed measurements
            Y_til[meas_ind:meas_ind+p] = Yk_list[kk]
            Y_bar[meas_ind:meas_ind+p] = ybar_k
            gamma_til_mat[meas_ind:meas_ind+p, :] = gamma_til_k
            
            # Increment measurement index
            meas_ind += p
            
        # Compute covariances
        Y_diff = gamma_til_mat - np.dot(Y_bar, np.ones((1, 2*L+1)))
        Pyy = np.dot(Y_diff, np.dot(diagWc, Y_diff.T)) + Rk_full
        Pxy = np.dot(chi_diff0, np.dot(diagWc, Y_diff.T))        

        # Compute Kalman Gain
        cholPyy = np.linalg.inv(np.linalg.cholesky(Pyy))
        invPyy = np.dot(cholPyy.T, cholPyy)
        K = np.dot(Pxy, invPyy)
           
        # Regular
        #P = P - np.dot(K, np.dot(Pyy, K.T))
        
        # Joseph Form
        P1 = (np.identity(L) - np.dot(np.dot(K, np.dot(Pyy, K.T)), invPo))
        P2 = np.dot(P1, np.dot(Po, P1.T)) + np.dot(K, np.dot(Rk_full, K.T))
        
        P = P2
        
#        # Compute Kalman Gain
#        Hi = np.dot(invPo, Pxy).T
#        K1 = np.dot(P, np.dot(Hi.T, invRk_full))
#        
#        check = K1 - K
#        #print check
#        print 'K check max diff'
#        print np.max(check)
#        #mistake
        
        # Compute updated state and covariance    
        X += np.dot(K, Y_til-Y_bar)
        resids = Y_til - Y_bar
        
        diff = np.linalg.norm(np.dot(K, Y_til-Y_bar))
        
        print('Iteration Number: ', iters)
        print('X', X)
        print('diff = ', diff)
        
        
        
    print(X)
    print(P)
    print('resids', np.mean(resids), np.std(resids))
    
    
    
    # Compute position and velocity truth values over time with covar
    tk_truth = sorted(truth_dict.keys())
    
    sqP = np.linalg.cholesky(P)
    Xrep = np.tile(X, (1, L))
    chi0 = np.concatenate((X, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
    
    xerr = np.zeros((len(tk_truth),))
    yerr = np.zeros((len(tk_truth),))
    zerr = np.zeros((len(tk_truth),))
    dxerr = np.zeros((len(tk_truth),))
    dyerr = np.zeros((len(tk_truth),))
    dzerr = np.zeros((len(tk_truth),))
    
    sig_x = np.zeros((len(tk_truth),))
    sig_y = np.zeros((len(tk_truth),))
    sig_z = np.zeros((len(tk_truth),))
    sig_dx = np.zeros((len(tk_truth),))
    sig_dy = np.zeros((len(tk_truth),))
    sig_dz = np.zeros((len(tk_truth),))
    
    xerr_meas = np.zeros((len(tk_list),))
    yerr_meas = np.zeros((len(tk_list),))
    zerr_meas = np.zeros((len(tk_list),))
    dxerr_meas = np.zeros((len(tk_list),))
    dyerr_meas = np.zeros((len(tk_list),))
    dzerr_meas = np.zeros((len(tk_list),))
    
    for kk in range(len(tk_truth)):
            
        # Current time
        tk = tk_truth[kk]
        tk_sec = (tk - t0).total_seconds()
        X_truth = truth_dict[tk]
        
        # Computed position sigma points and mean
        output = 'pos'
        gamma_til_k = compute_gamma_til_dy(chi0, tk_sec, state_params, output)
        r_ric = np.dot(gamma_til_k, Wm.T)
        r_ric = np.reshape(r_ric, (3,1))
        
#        print(gamma_til_k.shape)
#        print(r_ric)
#        print(L)
        
        r_diff = gamma_til_k - np.dot(r_ric, np.ones((1, 2*L+1)))
        Pr_ric = np.dot(r_diff, np.dot(diagWc, r_diff.T))
        
        # Computed velocity sigma points and mean
        output = 'vel'
        gamma_til_k = compute_gamma_til_dy(chi0, tk_sec, state_params, output)
        v_ric = np.dot(gamma_til_k, Wm.T)
        v_ric = np.reshape(v_ric, (3,1))
        v_diff = gamma_til_k - np.dot(v_ric, np.ones((1, 2*L+1)))
        Pv_ric = np.dot(v_diff, np.dot(diagWc, v_diff.T))
        
#        print(kk)
#        print(r_ric)
#        print(v_ric)
#        print(X_truth)
#        
#        if kk > 2:
#            mistake
        
        # Store Errors
        xerr[kk] = float(r_ric[0]) - float(X_truth[0])
        yerr[kk] = float(r_ric[1]) - float(X_truth[1])
        zerr[kk] = float(r_ric[2]) - float(X_truth[2])
        dxerr[kk] = float(v_ric[0]) - float(X_truth[3])
        dyerr[kk] = float(v_ric[1]) - float(X_truth[4])
        dzerr[kk] = float(v_ric[2]) - float(X_truth[5])
        
        sig_x[kk] = float(np.sqrt(Pr_ric[0,0]))
        sig_y[kk] = float(np.sqrt(Pr_ric[1,1]))
        sig_z[kk] = float(np.sqrt(Pr_ric[2,2]))
        
        sig_dx[kk] = float(np.sqrt(Pv_ric[0,0]))
        sig_dy[kk] = float(np.sqrt(Pv_ric[1,1]))
        sig_dz[kk] = float(np.sqrt(Pv_ric[2,2]))
        
        
        if tk in tk_list:
            ind = tk_list.index(tk)
            xerr_meas[ind] = float(r_ric[0]) - float(X_truth[0])
            yerr_meas[ind] = float(r_ric[1]) - float(X_truth[1])
            zerr_meas[ind] = float(r_ric[2]) - float(X_truth[2])
            dxerr_meas[ind] = float(v_ric[0]) - float(X_truth[3])
            dyerr_meas[ind] = float(v_ric[1]) - float(X_truth[4])
            dzerr_meas[ind] = float(v_ric[2]) - float(X_truth[5])
            
        
    
    
    # Generate Plots
    Xt0 = truth_dict[t0]
    tvec = [(tk - t0).total_seconds() for tk in tk_truth]
    t_hrs = [(tk - t0).total_seconds()/3600. for tk in tk_truth]
    tmeas_hrs = [(tk - t0).total_seconds()/3600. for tk in tk_list]
    
#    Xk_time, Pk_time = monte_carlo_prop(X, P, tvec)
    
    
    
    if meas_input == 'rho':
        plt.figure()
        plt.plot(tmeas_hrs, resids*1000., 'k.')
        plt.ylabel('Range Resids [m]')
        plt.xlabel('Time [hours]')
    
    if meas_input == 'delta':
        plt.figure()
        plt.plot(tmeas_hrs, resids*(1./arcsec2rad), 'k.')
        plt.ylabel('Delta Resids [arcsec]')
        plt.xlabel('Time [hours]')
        
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, xerr*1000., 'k.')
    plt.plot(tmeas_hrs, xerr_meas*1000., 'b.')
    plt.plot(t_hrs, 3.*sig_x*1000., 'k--')
    plt.plot(t_hrs, -3.*sig_x*1000., 'k--')
    plt.ylabel('x Err [m]')    
    plt.subplot(3,1,2)
    plt.plot(t_hrs, yerr*1000., 'k.')
    plt.plot(t_hrs, 3.*sig_y*1000., 'k--')
    plt.plot(t_hrs, -3.*sig_y*1000., 'k--')
    plt.plot(tmeas_hrs, yerr_meas*1000., 'b.')
    plt.ylabel('y Err [m]')     
    plt.subplot(3,1,3)
    plt.plot(t_hrs, zerr*1000., 'k.')
    plt.plot(t_hrs, 3.*sig_z*1000., 'k--')
    plt.plot(t_hrs, -3.*sig_z*1000., 'k--')
    plt.plot(tmeas_hrs, zerr_meas*1000., 'b.')
    plt.ylabel('z Err [m]') 
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, dxerr*1e6, 'k.')
    plt.plot(tmeas_hrs, dxerr_meas*1e6, 'b.')
    plt.plot(t_hrs, 3.*sig_dx*1e6, 'k--')
    plt.plot(t_hrs, -3.*sig_dx*1e6, 'k--')
    plt.ylabel('dx Err [mm/s]')    
    plt.subplot(3,1,2)
    plt.plot(t_hrs, dyerr*1e6, 'k.')
    plt.plot(t_hrs, 3.*sig_dy*1e6, 'k--')
    plt.plot(t_hrs, -3.*sig_dy*1e6, 'k--')
    plt.plot(tmeas_hrs, dyerr_meas*1e6, 'b.')
    plt.ylabel('dy Err [mm/s]')     
    plt.subplot(3,1,3)
    plt.plot(t_hrs, dzerr*1e6, 'k.')
    plt.plot(t_hrs, 3.*sig_dz*1e6, 'k--')
    plt.plot(t_hrs, -3.*sig_dz*1e6, 'k--')
    plt.plot(tmeas_hrs, dzerr_meas*1e6, 'b.')
    plt.ylabel('dz Err [mm/s]') 
    plt.xlabel('Time [hours]')
    
   
    
    plt.show()
    
    
    
    print('\n\nSeparation Velocity Estimates and Truth')    
    print('In-track Est [m/s]:\t\t', '{0:0.2E}'.format(float(X[0]) * 1000.))
    print('In-track STD [m/s]:\t\t', '{0:0.2E}'.format(float(np.sqrt(P[0,0])) * 1000.))
    print('In-track True [m/s]:\t', '{0:0.2E}'.format(float(Xt0[4]) * 1000.))

    
    return 


if __name__ == '__main__':
    
    plt.close('all')
    
    # Unit tests
#    unit_test_relative_geometry()
#    unit_test_cw_propagation()
    unit_test_dy2rho()
    
    
    # Estimation test cases
    outdir = r'D:\documents\research\cubesats\M2\analysis\simulation\unscented_formation'
    meas_file = os.path.join(outdir, 'm2_formation_twobody_delta_36hrs_2min.pkl')
    
#    m2_formation_setup(meas_file, meas_output='delta')
#    unscented_batch_cw(meas_file, meas_input='delta')
#    unscented_batch_dy(meas_file, meas_input='delta')
    
    
    
    



