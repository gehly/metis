import numpy as np
from math import pi
from matplotlib import pyplot as plt
from datetime import datetime
import pickle
import copy
import time
import sys
from scipy.integrate import odeint
from scipy.integrate import ode

from skyfield.api import utc

sys.path.append('../')

from filter_functions import compute_gaussian
from sensors.measurements import compute_measurement
from utilities.eop_functions import get_eop_data
import utilities.attitude as att



def unscented_kalman_filter(model_params_file, sensor_file, meas_file,
                            ephemeris, ts, alpha=1.):
    '''
    
    '''
    
    # Load model parameters
    pklFile = open(model_params_file, 'rb')
    data = pickle.load(pklFile)
    spacecraftConfig = data[0]
    forcesCoeff = data[1]
    surfaces = data[2]
    eop_alldata = data[3]
    XYs_df = data[4]
    pklFile.close()
    
    # Load sensor data
    pklFile = open(sensor_file, 'rb')
    data = pickle.load(pklFile)
    sensor_dict = data[0]
    pklFile.close()    
    
    # Load measurement data
    pklFile = open(meas_file, 'rb')
    data = pickle.load(pklFile)
    meas_dict = data[0]
    pklFile.close()
    
    meas_times = sorted(list(meas_dict.keys()))
    
#    # Force model parameters
#    dragCoeff = forcesCoeff['dragCoeff']
#    order = forcesCoeff['order']
#    degree = forcesCoeff['degree']
#    emissivity = forcesCoeff['emissivity']
    
    # Sun and earth data
    earth = ephemeris['earth']
    sun = ephemeris['sun']

    #Number of states and observations per epoch
    n = len(spacecraftConfig['X'])    
    
    # Initial state parameters
    X = spacecraftConfig['X'].reshape(n,1)
    P = spacecraftConfig['covar']    
       
    print(spacecraftConfig)
    print(X)
    
    
    # Loop over times
#    beta_list = []
    filter_output = {}
    filter_output['time'] = []
    filter_output['X'] = []
    filter_output['P'] = []
    filter_output['resids'] = []
    for ii in range(len(meas_times)):
        
        # Retrieve current and previous times
        ti = meas_times[ii]
        print('Current time: ', ti)
        
        ti_prior = spacecraftConfig['time']
        print(ti_prior)
        
        delta_t = (ti - ti_prior).total_seconds()        
        print('delta_t', delta_t)
               
        # Predictor
        if spacecraftConfig['type'] == '3DoF':
            Xbar, Pbar = \
                ukf_3dof_predictor(X, P, delta_t, n, alpha, 
                                   spacecraftConfig, forcesCoeff, surfaces)
                
        elif spacecraftConfig['type'] == '3att':
            Xbar, Pbar = \
                ukf_3att_predictor(X, P, delta_t, n, alpha, 
                                   spacecraftConfig, forcesCoeff, surfaces)
        
        elif spacecraftConfig['type'] == '6DoF':
            Xbar, Pbar, qmean = \
                ukf_6dof_predictor(X, P, delta_t, n, alpha, 
                                   spacecraftConfig, forcesCoeff, surfaces)
                
        else:
            print('Spacecraft Type Error')
            print(spacecraftConfig)
            break
        
        print('\n\n Predictor Step')
        print(ti)
        print(Xbar)
        print(Pbar)
        

        # Skyfield time and sun position
        UTC_skyfield = ts.utc(ti.replace(tzinfo=utc))
        sun_gcrf = earth.at(UTC_skyfield).observe(sun).position.km
        sun_gcrf = np.reshape(sun_gcrf, (3,1))
        
        # EOPs at current time
        EOP_data = get_eop_data(eop_alldata, ti)
        
        # Loop over sensors 
#        Yi = meas[ii,:].reshape(len(meas[ii,:]),1)
        sensor_id_list = list(meas_dict[ti].keys())
        for sensor_id in sensor_id_list:
            sensor = sensor_dict[sensor_id]
            Yi = meas_dict[ti][sensor_id]
            
            print('Yi', Yi)
            
            # Corrector
            if spacecraftConfig['type'] == '3DoF':
                X, P, beta = ukf_3dof_corrector(Xbar, Pbar, Yi, ti, n, alpha,
                                                sun_gcrf, sensor, EOP_data, XYs_df,
                                                spacecraftConfig, surfaces)
                
            elif spacecraftConfig['type'] == '3att':
                X, P, beta = ukf_3att_corrector(Xbar, Pbar, Yi, ti, n, alpha,
                                                sun_gcrf, sensor, EOP_data, XYs_df,
                                                spacecraftConfig, surfaces)
                
            elif spacecraftConfig['type'] == '6DoF':
                X, P, beta = ukf_6dof_corrector(Xbar, Pbar, qmean, Yi, ti, n, alpha,
                                                sun_gcrf, sensor, EOP_data, XYs_df,
                                                spacecraftConfig, surfaces)
            
            else:
                print('Spacecraft Type Error')
                print(spacecraftConfig)
                break
            
            print('\n\n Corrector Step')
            print(ti)
            print(sensor_id)
            print(X)
            print(P)
            print(beta)
        
        # Update with post-fit solution
        spacecraftConfig['time'] = ti     
        
        # Compute post-fit residuals
        Ybar_post = compute_measurement(X, sun_gcrf, sensor, spacecraftConfig,
                                        surfaces, ti, EOP_data,
                                        sensor['meas_types'], XYs_df)
        resids = Yi - Ybar_post
        
        print('post')
        print('Ybar_post', Ybar_post)
        print('resids', resids)
        
#        if ii > 3:
#            mistake
        
        # Append data to output
        filter_output['time'].append(ti)
        filter_output['X'].append(X)
        filter_output['P'].append(P)
        filter_output['resids'].append(resids)
    

    return filter_output


def ukf_3dof_predictor(X, P, delta_t, n, alpha, 
                       spacecraftConfig, forcesCoeff, surfaces):
    
    # Integration parameters
    int_tol = 1e-12
    int_dt = 10.
    intfcn = spacecraftConfig['intfcn']
    integrator = spacecraftConfig['integrator']   
    Q = forcesCoeff['Q']
    
    #Compute Weights
    beta = 2.
    kappa = 3. - n
    lam = alpha**2 * (n + kappa) - n
    gam = np.sqrt(n + lam)

    Wm = 1./(2.*(n + lam)) * np.ones((1,2*n))
    Wm = list(Wm.flatten())
    Wc = copy.copy(Wm)
    Wm.insert(0,lam/(n + lam))
    Wc.insert(0,lam/(n + lam) + (1 - alpha**2 + beta))
    Wm = np.asarray(Wm)
    diagWc = np.diag(Wc)    
    
    # Predictor step
    # Compute sigma points for propagation
    sqP = np.linalg.cholesky(P)
    Xrep = np.tile(X, (1, n))
    chi = np.concatenate((X, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
    chi_v = np.reshape(chi, (n*(2*n+1), 1), order='F')

    # Integrate chi
    if delta_t == 0.:
        intout = chi_v.T
    else:
        y0 = chi_v.flatten()
        tvec = np.arange(0., delta_t+(0.1*int_dt), int_dt)
        solver = ode(intfcn)
        solver.set_integrator(integrator, atol=int_tol, rtol=int_tol)
        solver.set_f_params([spacecraftConfig, forcesCoeff, surfaces])
        
        solver.set_initial_value(y0, tvec[0])
        intout = np.zeros((len(tvec), len(y0)))
        intout[0] = y0
        
        k = 1
        while solver.successful() and solver.t < tvec[-1]:
            solver.integrate(tvec[k])
            intout[k] = solver.y
            k += 1

    # Extract values for later calculations
    chi_v = intout[-1,:]
    chi_bar = np.reshape(chi_v, (n, 2*n+1), order='F')

    # Add process noise
    Xbar = np.dot(chi_bar, Wm.T)
    Xbar = np.reshape(Xbar, (n, 1))
    chi_diff = chi_bar - np.dot(Xbar, np.ones((1, (2*n+1))))
    if delta_t > 100.:
        Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T))
    else:
        print('\n Process Noise')
        Gamma1 = np.eye(3) * 0.5*delta_t**2.
        Gamma2 = np.eye(3) * delta_t
        Gamma = np.concatenate((Gamma1, Gamma2), axis=0)        
        Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T)) + \
            np.dot(Gamma, np.dot(Q, Gamma.T))

    # Re-symmetric pos def
    Pbar = 0.5 * (Pbar + Pbar.T)
    
#    print(Pbar)
#    print(np.linalg.eig(Pbar))
    
    
    return Xbar, Pbar


def ukf_3dof_corrector(Xbar, Pbar, Yi, ti, n, alpha, sun_gcrf,
                       sensor, EOP_data, XYs_df, spacecraftConfig, surfaces):
    
    
    #Compute Weights
    beta = 2.
    kappa = 3. - n
    lam = alpha**2 * (n + kappa) - n
    gam = np.sqrt(n + lam)

    Wm = 1./(2.*(n + lam)) * np.ones((1,2*n))
    Wm = list(Wm.flatten())
    Wc = copy.copy(Wm)
    Wm.insert(0,lam/(n + lam))
    Wc.insert(0,lam/(n + lam) + (1 - alpha**2 + beta))
    Wm = np.asarray(Wm)
    diagWc = np.diag(Wc)  
    
    # Sensor parameters
    meas_types = sensor['meas_types']
    sigma_dict = sensor['sigma_dict']
    p = len(meas_types)
    
    # Measurement noise
    var = []
    for mt in meas_types:
        var.append(sigma_dict[mt]**2.)
    Rk = np.diag(var)
    
    # Recompute sigma points to incorporate process noise
    sqP = np.linalg.cholesky(Pbar)
    Xrep = np.tile(Xbar, (1, n))
    chi_bar = np.concatenate((Xbar, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1) 
    chi_diff = chi_bar - np.dot(Xbar, np.ones((1, (2*n+1))))
    
    
    # Computed measurements    
    meas_bar = np.zeros((p, 2*n+1))
    for jj in range(chi_bar.shape[1]):
        Xj = chi_bar[:,jj]
        Yj = compute_measurement(Xj, sun_gcrf, sensor, spacecraftConfig,
                                 surfaces, ti, EOP_data,
                                 sensor['meas_types'], XYs_df)
        meas_bar[:,jj] = Yj.flatten()
    
    Ybar = np.dot(meas_bar, Wm.T)
    Ybar = np.reshape(Ybar, (p, 1))
    Y_diff = meas_bar - np.dot(Ybar, np.ones((1, (2*n+1))))
    Pyy = np.dot(Y_diff, np.dot(diagWc, Y_diff.T))
    Pxy = np.dot(chi_diff,  np.dot(diagWc, Y_diff.T))

    Pyy += Rk

    # Measurement Update
    K = np.dot(Pxy, np.linalg.inv(Pyy))
    X = Xbar + np.dot(K, Yi-Ybar)
    
#        # Regular update
#        P = Pbar - np.dot(K, np.dot(Pyy, K.T))
#
#        # Re-symmetric pos def
#        P = 0.5 * (P + P.T)
    
    # Joseph Form
    cholPbar = np.linalg.inv(np.linalg.cholesky(Pbar))
    invPbar = np.dot(cholPbar.T, cholPbar)
    P1 = (np.identity(6) - np.dot(np.dot(K, np.dot(Pyy, K.T)), invPbar))
    P = np.dot(P1, np.dot(Pbar, P1.T)) + np.dot(K, np.dot(Rk, K.T))
    
    print('posterior')
    print(Xbar)
    print(X)
    print(P)
    print(Ybar)
    print(Yi - Ybar)
#        print(Pyy)
#        print(Rk)
#        print(Pxy)
    

#    # Gaussian Likelihood
    beta = compute_gaussian(Yi, Ybar, Pyy)
#    beta_list.append(beta)
    
    
    return X, P, beta





def ukf_3att_predictor(X, P, delta_t, n, alpha, 
                       spacecraftConfig, forcesCoeff, surfaces):
    
    # Integration parameters
    int_tol = 1e-12
    int_dt = 10.
    intfcn = spacecraftConfig['intfcn']
    integrator = spacecraftConfig['integrator']   
    Q = forcesCoeff['Q']
    
    #Compute Weights
    n = 6
    beta = 2.
    kappa = 3. - n
    lam = alpha**2 * (n + kappa) - n
    gam = np.sqrt(n + lam)

    Wm = 1./(2.*(n + lam)) * np.ones((1,2*n))
    Wm = list(Wm.flatten())
    Wc = copy.copy(Wm)
    Wm.insert(0,lam/(n + lam))
    Wc.insert(0,lam/(n + lam) + (1 - alpha**2 + beta))
    Wm = np.asarray(Wm)
    diagWc = np.diag(Wc)    
    
    # Predictor step
    # Compute sigma points for propagation
    sqP = np.linalg.cholesky(P)
    Xpv = X[0:6].reshape(6,1)
    Xrep = np.tile(Xpv, (1, n))
    chi = np.concatenate((Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
    chi_v = np.reshape(chi, (n*(2*n), 1), order='F')    
    chi_v = np.concatenate((X, chi_v), axis=0)

    # Integrate chi
    if delta_t == 0.:
        intout = chi_v.T
    else:
        y0 = chi_v.flatten()
        tvec = np.arange(0., delta_t+(0.1*int_dt), int_dt)
        solver = ode(intfcn)
        solver.set_integrator(integrator, atol=int_tol, rtol=int_tol)
        solver.set_f_params([spacecraftConfig, forcesCoeff, surfaces])
        
        solver.set_initial_value(y0, tvec[0])
        intout = np.zeros((len(tvec), len(y0)))
        intout[0] = y0
        
        k = 1
        while solver.successful() and solver.t < tvec[-1]:
            solver.integrate(tvec[k])
            intout[k] = solver.y
            k += 1

    # Extract values for later calculations
    chi_v = intout[-1,:]
    Xatt = chi_v[6:13].reshape(7,1)
    chi_rep = chi_v[13:]
    chi_bar = np.reshape(chi_rep, (n, 2*n), order='F')
    pv_bar = chi_v[0:6].reshape(6,1)
    chi_bar = np.concatenate((pv_bar, chi_bar), axis=1)

    # Add process noise
    Xbar = np.dot(chi_bar, Wm.T)
    Xbar = np.reshape(Xbar, (6, 1))
    chi_diff = chi_bar - np.dot(Xbar, np.ones((1, (2*n+1))))
    if delta_t > 100.:
        Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T))
    else:
        print('\n Process Noise')
        Gamma1 = np.eye(3) * 0.5*delta_t**2.
        Gamma2 = np.eye(3) * delta_t
        Gamma = np.concatenate((Gamma1, Gamma2), axis=0)        
        Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T)) + \
            np.dot(Gamma, np.dot(Q, Gamma.T))

    # Re-symmetric pos def
    Pbar = 0.5 * (Pbar + Pbar.T)
    
#    print(Pbar)
#    print(np.linalg.eig(Pbar))
    
    Xbar = np.concatenate((Xbar, Xatt), axis=0)
    
    
    return Xbar, Pbar


def ukf_3att_corrector(Xbar, Pbar, Yi, ti, n, alpha, sun_gcrf,
                       sensor, EOP_data, XYs_df, spacecraftConfig, surfaces):
    
    
    #Compute Weights
    n = 6
    beta = 2.
    kappa = 3. - n
    lam = alpha**2 * (n + kappa) - n
    gam = np.sqrt(n + lam)

    Wm = 1./(2.*(n + lam)) * np.ones((1,2*n))
    Wm = list(Wm.flatten())
    Wc = copy.copy(Wm)
    Wm.insert(0,lam/(n + lam))
    Wc.insert(0,lam/(n + lam) + (1 - alpha**2 + beta))
    Wm = np.asarray(Wm)
    diagWc = np.diag(Wc)  
    
    # Sensor parameters
    meas_types = sensor['meas_types']
    sigma_dict = sensor['sigma_dict']
    p = len(meas_types)
    
    # Measurement noise
    var = []
    for mt in meas_types:
        var.append(sigma_dict[mt]**2.)
    Rk = np.diag(var)
    
    # Recompute sigma points to incorporate process noise
    sqP = np.linalg.cholesky(Pbar)
    Xpv = Xbar[0:6].reshape(6,1)
    Xatt = Xbar[6:13].reshape(7,1)
    Xrep = np.tile(Xpv, (1, n))
    chi_bar = np.concatenate((Xpv, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1) 
    chi_diff = chi_bar - np.dot(Xpv, np.ones((1, (2*n+1))))
    
    
    # Computed measurements    
    meas_bar = np.zeros((p, 2*n+1))
    for jj in range(chi_bar.shape[1]):
        Xj = chi_bar[:,jj].reshape(6,1)
        Xj_full = np.concatenate((Xj, Xatt), axis=0)
        Yj = compute_measurement(Xj_full, sun_gcrf, sensor, spacecraftConfig,
                                 surfaces, ti, EOP_data,
                                 sensor['meas_types'], XYs_df)
        meas_bar[:,jj] = Yj.flatten()
    
    Ybar = np.dot(meas_bar, Wm.T)
    Ybar = np.reshape(Ybar, (p, 1))
    Y_diff = meas_bar - np.dot(Ybar, np.ones((1, (2*n+1))))
    Pyy = np.dot(Y_diff, np.dot(diagWc, Y_diff.T))
    Pxy = np.dot(chi_diff,  np.dot(diagWc, Y_diff.T))

    Pyy += Rk

    # Measurement Update
    K = np.dot(Pxy, np.linalg.inv(Pyy))
    X = Xpv + np.dot(K, Yi-Ybar)
    X = np.concatenate((X, Xatt), axis=0)
    
#        # Regular update
#        P = Pbar - np.dot(K, np.dot(Pyy, K.T))
#
#        # Re-symmetric pos def
#        P = 0.5 * (P + P.T)
    
    # Joseph Form
    cholPbar = np.linalg.inv(np.linalg.cholesky(Pbar))
    invPbar = np.dot(cholPbar.T, cholPbar)
    P1 = (np.identity(6) - np.dot(np.dot(K, np.dot(Pyy, K.T)), invPbar))
    P = np.dot(P1, np.dot(Pbar, P1.T)) + np.dot(K, np.dot(Rk, K.T))
    
    print('posterior')
    print(Xbar)
    print(X)
    print(P)
    print(Ybar)
    print(Yi - Ybar)
#        print(Pyy)
#        print(Rk)
#        print(Pxy)
    

#    # Gaussian Likelihood
    beta = compute_gaussian(Yi, Ybar, Pyy)
#    beta_list.append(beta)
    
    
    return X, P, beta



def ukf_6dof_predictor(X, P, delta_t, n, alpha, 
                       spacecraftConfig, forcesCoeff, surfaces):
    
    # Integration parameters
    int_tol = 1e-12
    int_dt = 10.
    intfcn = spacecraftConfig['intfcn']
    integrator = spacecraftConfig['integrator']   
    Q = forcesCoeff['Q']
    sig_u = forcesCoeff['sig_u']
    sig_v = forcesCoeff['sig_v']
    
    #Compute Weights
    beta = 2.
    kappa = 3. - 12.
    lam = alpha**2 * (12. + kappa) - 12.
    gam = np.sqrt(12. + lam)

    Wm = 1./(2.*(12. + lam)) * np.ones((1,2*12))
    Wm = list(Wm.flatten())
    Wc = copy.copy(Wm)
    Wm.insert(0,lam/(12. + lam))
    Wc.insert(0,lam/(12. + lam) + (1 - alpha**2 + beta))
    Wm = np.asarray(Wm)
    diagWc = np.diag(Wc)    
    
    # Initialize the GRP error sigma points
    sqP = np.linalg.cholesky(P)
    Xgrp = np.zeros((12,1))
    Xgrp[0:6] = X[0:6]
    Xgrp[9:12] = X[10:13]
    Xrep = np.tile(Xgrp, (1, 12))
    chi_grp = np.concatenate((Xgrp, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)

    # Form sigma point matrix for quaternions
    chi_quat = np.zeros((13, 25))
    chi_quat[:, 0] = X.flatten()
    chi_quat[0:6, 1:25] = chi_grp[0:6, 1:25]
    chi_quat[10:13, 1:25] = chi_grp[9:12, 1:25]
    
    # Compute the delta quaternion sigma points
    qmean = X[6:10].reshape(4,1)
    for jj in range(24):
        dp = chi_grp[6:9, jj+1].reshape(3,1)
        dq = att.grp2quat(dp, 1)
        qj = att.quat_composition(dq, qmean)
        chi_quat[6:10, jj+1] = qj.flatten()   
    
    # Vector for integration function
    chi_v = np.reshape(chi_quat, (13*25, 1), order='F')

    # Integrate chi
    if delta_t == 0.:
        intout = chi_v.T
    else:
        y0 = chi_v.flatten()
        tvec = np.arange(0., delta_t+(0.1*int_dt), int_dt)
        solver = ode(intfcn)
        solver.set_integrator(integrator, atol=int_tol, rtol=int_tol)
        solver.set_f_params([spacecraftConfig, forcesCoeff, surfaces])
        
        solver.set_initial_value(y0, tvec[0])
        intout = np.zeros((len(tvec), len(y0)))
        intout[0] = y0
        
        k = 1
        while solver.successful() and solver.t < tvec[-1]:
            solver.integrate(tvec[k])
            intout[k] = solver.y
            k += 1

    # Extract values for later calculations
    chi_v = intout[-1,:]
    chi_bar = np.reshape(chi_v, (13, 25), order='F')
    
    # Form sigma point matrix for GRPs
    chi_grp = np.zeros((12, 25))
    chi_grp[0:6, :] = chi_bar[0:6, :]
    chi_grp[9:12, :] = chi_bar[9:12, :]
    
    # Compute the delta quaternion sigma points
    qmean = chi_bar[6:10, 0].reshape(4,1)
    qinv = att.quat_inverse(qmean)    
    for jj in range(1, 25):
        qj = chi_bar[6:10, jj].reshape(4,1)
        dq = att.quat_composition(qj, qinv)
        dp = att.quat2grp(dq, 1)
        chi_grp[6:9, jj] = dp.flatten()
    

    Xgrp = np.dot(chi_grp, Wm.T)
    Xgrp = np.reshape(Xgrp, (12, 1))
    chi_diff = chi_grp - np.dot(Xgrp, np.ones((1, 25)))
    
    if delta_t > 100.:
        Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T))
    else:
        print('\n Process Noise')
        Gamma1 = np.eye(3) * 0.5*delta_t**2.
        Gamma2 = np.eye(3) * delta_t
        Gamma = np.concatenate((Gamma1, Gamma2), axis=0)  
        
        Qatt1 = np.eye(3) * (sig_v**2 - (1./6.)*sig_u**2.*delta_t**2.)
        Qatt2 = np.eye(3) * sig_u**2.
        Qatt = np.zeros((6,6))
        Qatt[0:3, 0:3] = Qatt1
        Qatt[3:6, 3:6] = Qatt2
        Qatt *= delta_t/2.
        
        Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T))
        Pbar[0:6, 0:6] += np.dot(Gamma, np.dot(Q, Gamma.T))
        Pbar[6:12, 6:12] += Qatt

    # Re-symmetric pos def
    Pbar = 0.5 * (Pbar + Pbar.T)
    
#    print(Pbar)
#    print(np.linalg.eig(Pbar))
#    
#    print(Xgrp)
#    print(qmean)
#    mistake
    
    
    return Xgrp, Pbar, qmean


def ukf_6dof_corrector(Xgrp, Pbar, qmean, Yi, ti, n, alpha, sun_gcrf,
                       sensor, EOP_data, XYs_df, spacecraftConfig, surfaces):
    
    
    #Compute Weights
    beta = 2.
    kappa = 3. - 12.
    lam = alpha**2 * (12. + kappa) - 12.
    gam = np.sqrt(12. + lam)

    Wm = 1./(2.*(12. + lam)) * np.ones((1,2*12))
    Wm = list(Wm.flatten())
    Wc = copy.copy(Wm)
    Wm.insert(0,lam/(12. + lam))
    Wc.insert(0,lam/(12. + lam) + (1 - alpha**2 + beta))
    Wm = np.asarray(Wm)
    diagWc = np.diag(Wc)  
    
    # Sensor parameters
    meas_types = sensor['meas_types']
    sigma_dict = sensor['sigma_dict']
    p = len(meas_types)
    
    # Measurement noise
    var = []
    for mt in meas_types:
        var.append(sigma_dict[mt]**2.)
    Rk = np.diag(var)
    
    
    # Recompute sigma points to incorporate process noise
    sqP = np.linalg.cholesky(Pbar)
    Xrep = np.tile(Xgrp, (1, 12))
    chi_grp = np.concatenate((Xgrp, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1) 
    chi_diff = chi_grp - np.dot(Xgrp, np.ones((1, 25)))
    
    # Compute sigma points with quaternions
    chi_quat = np.zeros((13, 25))
    chi_quat[0:6, :] = chi_grp[0:6, :]
    chi_quat[10:13, :] = chi_grp[9:12, :]
    chi_quat[6:10, 0] = qmean.flatten()
    for jj in range(24):
        dp = chi_grp[6:9, jj+1].reshape(3,1)
        dq = att.grp2quat(dp, 1)
        qj = att.quat_composition(dq, qmean)
        chi_quat[6:10, jj+1] = qj.flatten() 
    
    # Computed measurements    
    meas_bar = np.zeros((p, 25))
    for jj in range(chi_quat.shape[1]):
        Xj = chi_quat[:,jj]
        Yj = compute_measurement(Xj, sun_gcrf, sensor, spacecraftConfig,
                                 surfaces, ti, EOP_data,
                                 sensor['meas_types'], XYs_df)
        meas_bar[:,jj] = Yj.flatten()
        
#        print(jj)
#        print(Yj)
    
#    print(Wm)
    Ybar = np.dot(meas_bar, Wm.T)
    Ybar = np.reshape(Ybar, (p, 1))
    Y_diff = meas_bar - np.dot(Ybar, np.ones((1, 25)))
    Pyy = np.dot(Y_diff, np.dot(diagWc, Y_diff.T))
    Pxy = np.dot(chi_diff,  np.dot(diagWc, Y_diff.T))

    Pyy += Rk
    
#    print(meas_bar)
#    print(Yi)
#    print(Ybar)
#    print(Pyy)

    # Measurement Update
    K = np.dot(Pxy, np.linalg.inv(Pyy))
    Xgrp = Xgrp + np.dot(K, Yi-Ybar)
    
    # Compute updated quaternion and full state vector
    dp = Xgrp[6:9].reshape(3,1)
    dq = att.grp2quat(dp, 1)
    q = att.quat_composition(dq, qmean)
    X = np.zeros((13, 1))
    X[0:6] = Xgrp[0:6]
    X[6:10] = q
    X[10:13] = Xgrp[9:12]
    
#        # Regular update
#        P = Pbar - np.dot(K, np.dot(Pyy, K.T))
#
#        # Re-symmetric pos def
#        P = 0.5 * (P + P.T)
    
    # Joseph Form
    cholPbar = np.linalg.inv(np.linalg.cholesky(Pbar))
    invPbar = np.dot(cholPbar.T, cholPbar)
    P1 = (np.identity(12) - np.dot(np.dot(K, np.dot(Pyy, K.T)), invPbar))
    P = np.dot(P1, np.dot(Pbar, P1.T)) + np.dot(K, np.dot(Rk, K.T))
    
#    print('posterior')
#    print(X)
#    print(P)
#    print(Ybar)
#    print(Yi - Ybar)
#        print(Pyy)
#        print(Rk)
#        print(Pxy)
    

#    # Gaussian Likelihood
    beta = compute_gaussian(Yi, Ybar, Pyy)
#    beta_list.append(beta)
    
    return X, P, beta


###############################################################################
# Stand-alone execution
###############################################################################


#if __name__ == '__main__':

    