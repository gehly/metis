import numpy as np
import math
import sys
import os
import inspect
import copy


filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from dynamics import dynamics_functions as dyn
from estimation import estimation_functions as est
from sensors import measurement_functions as mfunc
from utilities import time_systems as timesys




###############################################################################
# This file contains a number of basic functions useful for data association
# and multitarget estimation problems.
#
#
# References:
#  [1] Blackman and Popoli, "Desing and Analysis of Modern Tracking Systems,"
#     1999.
#
#  [2] Cox and Hingorani, "An Efficient Implementation of Reid's Multiple 
#     Hypothesis Tracking Algorithm and Its Evaluation for the Purpose of 
#     Visual Tracking," IEEE TPAMI 1996.
#
###############################################################################



###############################################################################
# PHD Filter
###############################################################################

def phd_filter(state_dict, truth_dict, meas_dict, meas_fcn, params_dict):
    
    # Break out inputs
    state_params = params_dict['state_params']
    filter_params = params_dict['filter_params']
    
    # State information
    state_tk = sorted(state_dict.keys())[-1]
    weights = state_dict[state_tk]['weights']
    means = state_dict[state_tk]['means']
    covars = state_dict[state_tk]['covars']
    GMM_dict = {}
    GMM_dict['weights'] = weights
    GMM_dict['means'] = means
    GMM_dict['covars'] = covars
    nstates = len(means[0])    
    
    # Prior information about the distribution
    pnorm = 2.
    kurt = math.gamma(5./pnorm)*math.gamma(1./pnorm)/(math.gamma(3./pnorm)**2.)
    beta = kurt - 1.
    kappa = kurt - float(nstates)

    # Compute sigma point weights
    alpha = filter_params['alpha']
    lam = alpha**2.*(nstates + kappa) - nstates
    gam = np.sqrt(nstates + lam)
    Wm = 1./(2.*(nstates + lam)) * np.ones(2*nstates,)
    Wc = Wm.copy()
    Wm = np.insert(Wm, 0, lam/(nstates + lam))
    Wc = np.insert(Wc, 0, lam/(nstates + lam) + (1 - alpha**2 + beta))
    diagWc = np.diag(Wc)
    filter_params['gam'] = gam
    filter_params['Wm'] = Wm
    filter_params['diagWc'] = diagWc

    # Initialize output
    filter_output = {}

    # Measurement times
    tk_list = sorted(meas_dict.keys())
    
    # Number of epochs
    N = len(tk_list)
  
    # Loop over times
    for kk in range(N):
    
        # Current and previous time
        if kk == 0:
            tk_prior = state_tk
        else:
            tk_prior = tk_list[kk-1]

        tk = tk_list[kk]
        
        print('')
        print(tk)
        print('ncomps', len(GMM_dict['weights']))
        print('Nk est', sum(GMM_dict['weights']))

        # Predictor Step
        tin = [tk_prior, tk]
        GMM_bar = phd_predictor(GMM_dict, tin, params_dict)
        
        print('predictor')
        print('ncomps', len(GMM_bar['weights']))
        print('Nk est', sum(GMM_bar['weights']))
        
        # Corrector Step
        Zk = meas_dict[tk]['Zk_list']
        sensor_id_list = meas_dict[tk]['sensor_id_list']
        GMM_dict = phd_corrector(GMM_bar, tk, Zk, sensor_id_list, meas_fcn,
                                 params_dict)
        
        print('corrector')
        print('ncomps', len(GMM_dict['weights']))
        print('Nk est', sum(GMM_dict['weights']))
        
        # Prune/Merge Step
        GMM_dict = est.merge_GMM(GMM_dict, filter_params)
        
        print('merge')
        print('ncomps', len(GMM_dict['weights']))
        print('Nk est', sum(GMM_dict['weights']))
        
        
        # State extraction and residuals calculation
        wk_list, Xk_list, Pk_list, resids_k = \
            phd_state_extraction(GMM_dict, tk, Zk, sensor_id_list, meas_fcn,
                                 params_dict)
            
            
        
        # print('wk_list', wk_list)
        
        
        # Store output
        filter_output[tk] = {}
        filter_output[tk]['weights'] = GMM_dict['weights']
        filter_output[tk]['means'] = GMM_dict['means']
        filter_output[tk]['covars'] = GMM_dict['covars']
        filter_output[tk]['wk_list'] = wk_list
        filter_output[tk]['Xk_list'] = Xk_list
        filter_output[tk]['Pk_list'] = Pk_list
        filter_output[tk]['resids'] = resids_k
        
        
    # TODO Generation of full_state_output not working correctly
    # Use filter_output for error analysis
    full_state_output = {}
    
    return filter_output, full_state_output
    



def phd_predictor(GMM_dict, tin, params_dict):
    '''
    
    
    '''
    
    
    # Break out inputs
    filter_params = params_dict['filter_params']
    state_params = params_dict['state_params']
    int_params = params_dict['int_params']
    
    # Copy input to ensure pass by value
    GMM_dict = copy.deepcopy(GMM_dict)
    filter_params = copy.deepcopy(filter_params)
    state_params = copy.deepcopy(state_params)
    int_params = copy.deepcopy(int_params)
    
    # Retrieve parameters
    p_surv = filter_params['p_surv']
    Q = filter_params['Q']
    gam = filter_params['gam']
    Wm = filter_params['Wm']
    diagWc = filter_params['diagWc']    
    gap_seconds = filter_params['gap_seconds']        
    time_format = int_params['time_format']     
    
    # Fudge to work with general_dynamics
    state_params['alpha'] = filter_params['alpha']
    
    q = int(Q.shape[0])
    
    tk_prior = tin[0]
    tk = tin[1]
    
    if time_format == 'seconds':
        delta_t = tk - tk_prior
    elif time_format == 'JD':
        delta_t = (tk - tk_prior)*86400.
    elif time_format == 'datetime':
        delta_t = (tk - tk_prior).total_seconds()
    
    # Check if propagation is needed
    if delta_t == 0.:
        return GMM_dict
    
    # Initialize for integrator
    # Retrieve current GMM
    weights = GMM_dict['weights']
    means = GMM_dict['means']
    covars = GMM_dict['covars']    
    ncomp = len(weights)
    nstates = len(means[0])
    npoints = 2*nstates + 1

    # Loop over components
    for jj in range(ncomp):
        
#        print('\nstart loop')
#        print('jj', jj)
#        print('ncomp', len(weights))
#        print('t0', t0_list[jj])
        
        # Retrieve component values
        wj = weights[jj]
        mj = means[jj]
        Pj = covars[jj]
        tin = [tk_prior, tk]
            
        # Compute sigma points
        sqP = np.linalg.cholesky(Pj)
        Xrep = np.tile(mj, (1, nstates))
        chi = np.concatenate((mj, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
        chi_v = np.reshape(chi, (nstates*npoints, 1), order='F')
        
        # Integrate sigma points
        int0 = chi_v.flatten()
        tout, intout = \
            dyn.general_dynamics(int0, tin, state_params, int_params)

        # Retrieve output state        
        chi_v = intout[-1,:]
        chi = np.reshape(chi_v, (nstates, npoints), order='F')

        # State Noise Compensation
        # Zero out SNC for long time gaps
        Gamma = np.zeros((nstates,q))
        if delta_t < gap_seconds:   
            Gamma[0:q,:] = (delta_t**2./2) * np.eye(q)
            Gamma[q:2*q,:] = delta_t * np.eye(q)

        Xbar = np.dot(chi, Wm.T)
        Xbar = np.reshape(Xbar, (nstates, 1))
        chi_diff = chi - np.dot(Xbar, np.ones((1, npoints)))
        Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T)) + np.dot(Gamma, np.dot(Q, Gamma.T))

        # Store output
        weights[jj] *= p_surv
        means[jj] = Xbar
        covars[jj] = Pbar

    # Form output
    GMM_bar = {}
    GMM_bar['weights'] = weights
    GMM_bar['means'] = means
    GMM_bar['covars'] = covars   


    return GMM_bar



def phd_corrector(GMM_bar, tk, Zk, sensor_id_list, meas_fcn, params_dict):
    '''
    
    
    '''
    
    
    # Retrieve inputs
    filter_params = params_dict['filter_params']
    state_params = params_dict['state_params']
    sensor_params = params_dict['sensor_params']
    gam = filter_params['gam']
    Wm = filter_params['Wm']
    diagWc = filter_params['diagWc']
    p_det = filter_params['p_det']
    
    # Break out GMM
    weights0 = GMM_bar['weights']
    means0 = GMM_bar['means']
    covars0 = GMM_bar['covars']
    nstates = len(means0[0])
    npoints = 2*nstates + 1
    ncomp = len(weights0)
    nmeas = len(Zk)
    
    # Components for missed detection case
    weights = [(1. - p_det)*wj for wj in weights0]
    means = copy.copy(means0)
    covars = copy.copy(covars0)
    
    # Loop over each measurement and compute updates
    for ii in range(nmeas):
        
        # Retrieve measurement
        zi = Zk[ii]
        sensor_id = sensor_id_list[ii]
    
        # Loop over components   
        qk_list = []
        for jj in range(ncomp):        
            
            mj = means0[jj]
            Pj = covars0[jj]
            
            # Compute sigma points
            sqP = np.linalg.cholesky(Pj)
            Xrep = np.tile(mj, (1, nstates))
            chi = np.concatenate((mj, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
            chi_diff = chi - np.dot(mj, np.ones((1, npoints)))
    
            # Computed measurements and covariance
            gamma_til_k, Rk = meas_fcn(tk, chi, state_params, sensor_params, sensor_id)
            zbar = np.dot(gamma_til_k, Wm.T)
            zbar = np.reshape(zbar, (len(zbar), 1))
            z_diff = gamma_til_k - np.dot(zbar, np.ones((1, npoints)))
            Pyy = np.dot(z_diff, np.dot(diagWc, z_diff.T)) + Rk
            Pxy = np.dot(chi_diff,  np.dot(diagWc, z_diff.T))
            
            print('zi', zi)
            print('zbar', zbar)
            
            # Angle-rollover for RA
            if 'ra' in sensor_params[sensor_id]['meas_types']:
                ra_ind = sensor_params[sensor_id]['meas_types'].index('ra')
                
                if math.pi/2. < zbar[ra_ind] < math.pi:
                    if -math.pi < zi[ra_ind] < -math.pi/2.:
                        zi[ra_ind] += 2.*math.pi
                        
                if -math.pi < zbar[ra_ind] < -math.pi/2.:
                    if math.pi/2. < zi[ra_ind] < math.pi:
                        zi[ra_ind] -= 2.*math.pi
            
            # Kalman gain and measurement update
            Kk = np.dot(Pxy, np.linalg.inv(Pyy))
            mf = mj + np.dot(Kk, zi-zbar)
            
            # Joseph form
            cholPbar = np.linalg.inv(np.linalg.cholesky(Pj))
            invPbar = np.dot(cholPbar.T, cholPbar)
            P1 = (np.eye(nstates) - np.dot(np.dot(Kk, np.dot(Pyy, Kk.T)), invPbar))
            P2 = np.dot(Kk, np.dot(Rk, Kk.T))
            Pf = np.dot(P1, np.dot(Pj, P1.T)) + P2
            
            # Compute Gaussian likelihood
            qk_j = est.gaussian_likelihood(zi, zbar, Pyy)
    
            # Store output
            means.append(mf)
            covars.append(Pf)
            qk_list.append(qk_j)
        
        # Normalize updated weights
        denom = p_det*np.dot(qk_list, weights0) + clutter_intensity(zi, sensor_id, sensor_params)
        wf = [p_det*a1*a2/denom for a1,a2 in zip(weights0, qk_list)]
        weights.extend(wf)
        
        # print('clutter intensity', clutter_intensity(zi, sensor_id, sensor_params))
        
    # Form output  
    GMM_dict = {}
    GMM_dict['weights'] = weights
    GMM_dict['means'] = means
    GMM_dict['covars'] = covars
    
    
    return GMM_dict



def phd_state_extraction(GMM_dict, tk, Zk, sensor_id_list, meas_fcn, 
                         params_dict):
    '''
    
    
    '''
    
    # Retrieve inputs
    filter_params = params_dict['filter_params']
    state_params = params_dict['state_params']
    sensor_params = params_dict['sensor_params']
    int_params = params_dict['int_params']
    time_format = int_params['time_format']
    
    # Compute UTC
    if time_format == 'JD':
        UTC = timesys.jd2dt(tk)
    elif time_format == 'datetime':
        UTC = tk
    
    # Retrieve current GMM componets
    weights = GMM_dict['weights']
    means = GMM_dict['means']
    covars = GMM_dict['covars']

    
    # Compute cardinality
    Nk = int(round(sum(weights)))
    if Nk > len(weights):
        Nk = len(weights)
    
    # Select the Nk highest weighted components as the state estimate at 
    # current time
    sorted_inds = sorted(range(len(weights)), key=lambda k: weights[k], reverse=True)
    max_inds = sorted_inds[0:Nk]
    
    # Values for output
    wk_list = [weights[ii] for ii in max_inds]
    Xk_list = [means[ii] for ii in max_inds]
    Pk_list = [covars[ii] for ii in max_inds]
    
    # Calculate residuals
    resids_out = []
    
    if len(wk_list) > 0:
        for ii in range(len(Zk)):
            zi = Zk[ii]
            sensor_id = sensor_id_list[ii]
            
            resids_list = []
            for jj in range(len(Xk_list)):
                Xj = Xk_list[jj]            
                zbar = mfunc.compute_measurement(Xj, state_params, sensor_params,
                                                 sensor_id, UTC)
                resids = zi - zbar
                resids_list.append(resids)
                
            # Take smallest magnitude as residual for this measurement
            min_list = [np.linalg.norm(resid) for resid in resids_list]
            resids_k = resids_list[min_list.index(min(min_list))]
            resids_out.append(resids_k)
    
    
    return wk_list, Xk_list, Pk_list, resids_out











###############################################################################
# LMB Functions
###############################################################################


def lmb_filter(state_dict, truth_dict, meas_dict, meas_fcn, params_dict):
    
    # Break out inputs
    state_params = params_dict['state_params']
    filter_params = params_dict['filter_params']
    nstates = state_params['nstates']
    
    # State information
    state_tk = sorted(state_dict.keys())[-1]
    LMB_dict = state_dict[state_tk]['LMB_dict']

    # Prior information about the distribution
    pnorm = 2.
    kurt = math.gamma(5./pnorm)*math.gamma(1./pnorm)/(math.gamma(3./pnorm)**2.)
    beta = kurt - 1.
    kappa = kurt - float(nstates)

    # Compute sigma point weights
    alpha = filter_params['alpha']
    lam = alpha**2.*(nstates + kappa) - nstates
    gam = np.sqrt(nstates + lam)
    Wm = 1./(2.*(nstates + lam)) * np.ones(2*nstates,)
    Wc = Wm.copy()
    Wm = np.insert(Wm, 0, lam/(nstates + lam))
    Wc = np.insert(Wc, 0, lam/(nstates + lam) + (1 - alpha**2 + beta))
    diagWc = np.diag(Wc)
    filter_params['gam'] = gam
    filter_params['Wm'] = Wm
    filter_params['diagWc'] = diagWc

    # Initialize output
    filter_output = {}

    # Measurement times
    tk_list = sorted(meas_dict.keys())
    
    # Number of epochs
    N = len(tk_list)
    
    print(LMB_dict)
    print(state_tk)
    print(tk_list)
    print(N)

  
    # Loop over times
    for kk in range(N):
    
        # Current and previous time
        if kk == 0:
            tk_prior = state_tk
        else:
            tk_prior = tk_list[kk-1]

        tk = tk_list[kk]
        
        print('')
        print(tk)
        print(LMB_dict)
        print('ntracks', len(LMB_dict))

        # Predictor Step
        tin = [tk_prior, tk]
        LMB_birth, LMB_surv = lmb_predictor(LMB_dict, tin, params_dict)
        
        print('predictor')
        print(LMB_birth)
        print(LMB_surv)
        print('birth tracks', len(LMB_birth))
        print('surv tracks', len(LMB_surv))

        # Corrector Step
        Zk = meas_dict[tk]['Zk_list']
        sensor_id_list = meas_dict[tk]['sensor_id_list']
        LMB_dict = lmb_corrector(LMB_birth, LMB_surv, tk, Zk, sensor_id_list,
                                 meas_fcn, params_dict)
        
        print('corrector')
        # print('ncomps', len(GMM_dict['weights']))
        # print('Nk est', sum(GMM_dict['weights']))
        
        # Prune/Merge Step
        LMB_dict = lmb_cleanup(LMB_dict, params_dict)
        
        print('merge')
        # print('ncomps', len(GMM_dict['weights']))
        # print('Nk est', sum(GMM_dict['weights']))
        
        
        # State extraction and residuals calculation
        pk, Nk, labelk_list, rk_list, Xk_list, Pk_list, resids_k = \
            lmb_state_extraction(LMB_dict, tk, Zk, sensor_id_list, meas_fcn,
                                 params_dict)
            
            
        
        # print('wk_list', wk_list)
        
        
        # Store output
        filter_output[tk] = {}
        filter_output[tk]['LMB_dict'] = copy.deepcopy(LMB_dict)
        filter_output[tk]['card'] = pk
        filter_output[tk]['N'] = Nk
        filter_output[tk]['label_list'] = labelk_list
        filter_output[tk]['rk_list'] = rk_list
        filter_output[tk]['Xk_list'] = Xk_list
        filter_output[tk]['Pk_list'] = Pk_list
        filter_output[tk]['resids'] = resids_k

        
        
    # TODO Generation of full_state_output not working correctly
    # Use filter_output for error analysis
    full_state_output = {}
    
    return filter_output, full_state_output
    



def lmb_predictor(LMB_dict, tin, params_dict):
    '''
    
    
    '''
    
    
    # Break out inputs
    filter_params = params_dict['filter_params']
    state_params = params_dict['state_params']
    int_params = params_dict['int_params']
    
    # Copy input to ensure pass by value
    LMB_dict = copy.deepcopy(LMB_dict)
    filter_params = copy.deepcopy(filter_params)
    state_params = copy.deepcopy(state_params)
    int_params = copy.deepcopy(int_params)
    
    # Retrieve parameters
    p_surv = filter_params['p_surv']
    snc_flag = filter_params['snc_flag']
    Q = filter_params['Q']
    gam = filter_params['gam']
    Wm = filter_params['Wm']
    diagWc = filter_params['diagWc']    
    gap_seconds = filter_params['gap_seconds']        
    time_format = int_params['time_format']     
    
    # Fudge to work with general_dynamics
    state_params['alpha'] = filter_params['alpha']
    
    q = int(Q.shape[0])
    
    tk_prior = tin[0]
    tk = tin[1]
    
    if time_format == 'seconds':
        delta_t = tk - tk_prior
        tin = [0., delta_t]
    elif time_format == 'JD':
        delta_t = (tk - tk_prior)*86400.
    elif time_format == 'datetime':
        delta_t = (tk - tk_prior).total_seconds()


    # Birth Components
    birth_model = filter_params['birth_model']
    LMB_birth = {}
    for ii in birth_model.keys():
        label = (tk, ii)
        LMB_birth[label] = {}        
        LMB_birth[label]['r'] = birth_model[ii]['r_birth']
        LMB_birth[label]['weights'] = birth_model[ii]['weights']
        LMB_birth[label]['means'] = birth_model[ii]['means']
        LMB_birth[label]['covars'] = birth_model[ii]['covars']

    

    # Check if propagation is needed
    if delta_t == 0.:
        LMB_surv = LMB_dict
        return LMB_birth, LMB_surv
    
    # Surviving components
    # Initialize for integrator
    # Loop over labels
    label_list = list(LMB_dict.keys())
    LMB_surv = {}
    for label in label_list:
        
        # Retrieve current GMM
        r = LMB_dict[label]['r']
        weights = LMB_dict[label]['weights']
        means = LMB_dict[label]['means']
        covars = LMB_dict[label]['covars']    
        ncomp = len(weights)
        nstates = len(means[0])
        npoints = 2*nstates + 1
    
        # Loop over components
        for jj in range(ncomp):
            
    #        print('\nstart loop')
    #        print('jj', jj)
    #        print('ncomp', len(weights))
    #        print('t0', t0_list[jj])
            
            # Retrieve component values
            wj = weights[jj]
            mj = means[jj]
            Pj = covars[jj]
            tin = [tk_prior, tk]
                
            # Compute sigma points
            sqP = np.linalg.cholesky(Pj)
            Xrep = np.tile(mj, (1, nstates))
            chi = np.concatenate((mj, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
            chi_v = np.reshape(chi, (nstates*npoints, 1), order='F')
            
            # Integrate sigma points
            int0 = chi_v.flatten()
            tout, intout = \
                dyn.general_dynamics(int0, tin, state_params, int_params)
    
            # Retrieve output state        
            chi_v = intout[-1,:]
            chi = np.reshape(chi_v, (nstates, npoints), order='F')
    
            Xbar = np.dot(chi, Wm.T)
            Xbar = np.reshape(Xbar, (nstates, 1))
            chi_diff = chi - np.dot(Xbar, np.ones((1, npoints)))
    
            # State Noise Compensation            
            if snc_flag == 'gamma':
                
                # Zero out SNC for long time gaps
                Gamma = np.zeros((nstates,q))
                if delta_t < gap_seconds:   
                    Gamma[0:q,:] = (delta_t**2./2) * np.eye(q)
                    Gamma[q:2*q,:] = delta_t * np.eye(q)
                
                Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T)) + np.dot(Gamma, np.dot(Q, Gamma.T))
                
            elif snc_flag == 'qfull':
                
                # Zero out SNC for long time gaps
                if delta_t < gap_seconds:   
                    Q = np.zeros((q,q))
                
                Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T)) + Q
    
            # Store GMM components
            weights[jj] = wj
            means[jj] = Xbar
            covars[jj] = Pbar

        # Store LMB
        LMB_surv[label] = {}
        LMB_surv[label]['r'] = r*p_surv
        LMB_surv[label]['weights'] = weights
        LMB_surv[label]['means'] = means
        LMB_surv[label]['covars'] = covars


    return LMB_birth, LMB_surv



def lmb_corrector(LMB_birth, LMB_surv, tk, Zk, sensor_id_list, meas_fcn, params_dict):
    '''
    
    
    '''
    
    print('')
    print('lmb_corrector function')
    print('tk', tk)
    print('Zk', Zk)
    
    machine_eps = np.finfo(float).eps
    
    # Retrieve inputs
    filter_params = params_dict['filter_params']
    state_params = params_dict['state_params']
    sensor_params = params_dict['sensor_params']
    H_max = filter_params['H_max']
    H_max_birth = filter_params['H_max_birth']
    gam = filter_params['gam']
    Wm = filter_params['Wm']
    diagWc = filter_params['diagWc']
    p_det = filter_params['p_det']
    nstates = state_params['nstates']
    npoints = 2*nstates + 1
    nmeas = len(Zk)
    
    # Form GLMB from input LMB
    GLMB_birth = lmb2glmb(LMB_birth, H_max_birth)
    GLMB_surv = lmb2glmb(LMB_surv, H_max)
    
    print('GLMB_birth')
    print(GLMB_birth)
    print('GLMB_surv')
    print(GLMB_surv)

    
    # Combine and normalize weights
    GLMB_dict = combine_glmb(GLMB_birth, GLMB_surv)
    wsum = 0.
    for hyp in GLMB_dict:
        wsum += GLMB_dict[hyp]['hyp_weight']
    for hyp in GLMB_dict:
        GLMB_dict[hyp]['hyp_weight'] *= (1./wsum)
        
    
    print('GLMB_dict', GLMB_dict)
    print('wsum', wsum)
    total_weight = 0.
    for hyp in GLMB_dict:
        total_weight += GLMB_dict[hyp]['hyp_weight']
        
    print('total hyp weight', total_weight)

    # Get unique sensor id's
    unique_sensors = list(set(sensor_id_list))
    
    # Components for missed detection case
    LMB_bar = combine_lmb(LMB_birth, LMB_surv)
    full_label_list = list(LMB_bar.keys())
    track_update = {}
    tind = 0
    for label in full_label_list:
        track_update[tind] = {}
        track_update[tind]['label'] = label
        track_update[tind]['r'] = LMB_bar[label]['r']
        track_update[tind]['weights'] = LMB_bar[label]['weights']
        track_update[tind]['means'] = LMB_bar[label]['means']
        track_update[tind]['covars'] = LMB_bar[label]['covars']
        track_update[tind]['meas_ind'] = 'missed'
        tind += 1
        
    print(track_update)

    # Loop over each measurement and compute updates
    allcost_mat = np.zeros((len(full_label_list), nmeas))
    for ii in range(nmeas):
        
        # Retrieve measurement
        zi = Zk[ii]
        sensor_id = sensor_id_list[ii]
    
        # Loop over tracks
        for tt in range(len(full_label_list)):
            label = full_label_list[tt]
            weights0 = LMB_bar[label]['weights']
            means0 = LMB_bar[label]['means']
            covars0 = LMB_bar[label]['covars']
            
            # Loop over components
            ncomp = len(weights0)
            qk_list = []
            means = []
            covars = []
            for jj in range(ncomp):        
                
                mj = means0[jj]
                Pj = covars0[jj]
                
                # Compute sigma points
                sqP = np.linalg.cholesky(Pj)
                Xrep = np.tile(mj, (1, nstates))
                chi = np.concatenate((mj, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
                chi_diff = chi - np.dot(mj, np.ones((1, npoints)))
        
                # Computed measurements and covariance
                gamma_til_k, Rk = meas_fcn(tk, chi, state_params, sensor_params, sensor_id)
                zbar = np.dot(gamma_til_k, Wm.T)
                zbar = np.reshape(zbar, (len(zbar), 1))
                z_diff = gamma_til_k - np.dot(zbar, np.ones((1, npoints)))
                Pyy = np.dot(z_diff, np.dot(diagWc, z_diff.T)) + Rk
                Pxy = np.dot(chi_diff,  np.dot(diagWc, z_diff.T))
                
                print('')
                print('ii', ii)
                print('zi', zi)
                print('label', label)
                print('zbar', zbar)
                
                # Angle-rollover for RA
                if 'ra' in sensor_params[sensor_id]['meas_types']:
                    ra_ind = sensor_params[sensor_id]['meas_types'].index('ra')
                    
                    
                    # TODO it would probably be better not to update zi but
                    # instead zbar or just the diff used for state update
                    if math.pi/2. < zbar[ra_ind] < math.pi:
                        if -math.pi < zi[ra_ind] < -math.pi/2.:
                            zi[ra_ind] += 2.*math.pi
                            
                    if -math.pi < zbar[ra_ind] < -math.pi/2.:
                        if math.pi/2. < zi[ra_ind] < math.pi:
                            zi[ra_ind] -= 2.*math.pi
                
                # Kalman gain and measurement update
                Kk = np.dot(Pxy, np.linalg.inv(Pyy))
                mf = mj + np.dot(Kk, zi-zbar)
                
                # Joseph form
                cholPbar = np.linalg.inv(np.linalg.cholesky(Pj))
                invPbar = np.dot(cholPbar.T, cholPbar)
                P1 = (np.eye(nstates) - np.dot(np.dot(Kk, np.dot(Pyy, Kk.T)), invPbar))
                P2 = np.dot(Kk, np.dot(Rk, Kk.T))
                Pf = np.dot(P1, np.dot(Pj, P1.T)) + P2
                
                # Compute Gaussian likelihood
                qk_j = est.gaussian_likelihood(zi, zbar, Pyy)
        
                # Store output
                means.append(mf)
                covars.append(Pf)
                qk_list.append(qk_j)
            
            # Normalize updated weights (this will always sum to 1?)
            # Vo, Vo, Phung 2014 Eq 27-28
            factor = p_det/clutter_intensity(zi, sensor_id, sensor_params)            
            
            print('factor', factor)
            print('p_det', p_det)
            print('clutter', clutter_intensity(zi, sensor_id, sensor_params))
            print('weights0', weights0)
            print('qk_list', qk_list)
            print('machine_eps', machine_eps)
            
            
            weights = [a1*a2*factor + machine_eps for a1,a2 in zip(weights0, qk_list)]
            sum_weights = sum(weights)
            # weights = [wi/sum_weights for wi in weights]
                    
            # Update track dictionary
            track_update[tind] = {}
            track_update[tind]['label'] = label
            track_update[tind]['r'] = LMB_bar[label]['r']
            track_update[tind]['weights'] = weights
            track_update[tind]['means'] = means
            track_update[tind]['covars'] = covars
            track_update[tind]['qk_list'] = qk_list
            track_update[tind]['meas_ind'] = ii
            
            tind += 1
            
            # Update cost matrix (Vo, Vo, Phung 2014 Eq 26)
            # allcost_mat[tt,ii] = machine_eps + factor*sum_weights/(1.-p_det)
            
            weights2 = [a1*a2 + machine_eps for a1,a2 in zip(weights0, qk_list)]
            print('sum_weights', sum_weights)
            print('sum_weights2', sum(weights2))
            print('entry check', p_det/(1. - p_det)*sum(weights2)/clutter_intensity(zi, sensor_id, sensor_params))
            
            allcost_mat[tt,ii] = sum_weights/(1. - p_det)
            
        print('check column allcost_mat')
        print('meas_ind', ii)
        print('zi', zi)
        print(allcost_mat[:,ii])
        
        # mistake
          
    print(allcost_mat)

    
    # # TEST ONLY
    # # Get negative log cost (Vo, Vo, Phung 2014 Eq 26)
    # cost_mat = allcost_mat
    # neglog_mat = -np.log(cost_mat)
    
    # # Compute measurement to track assignments
    # kbest = 10
    # assign_lists = glmb_kbest_assignments(neglog_mat, kbest)
    
    # print(cost_mat)
    # print(neglog_mat)
    # print(assign_lists)
    
    # Reference Mahler (2014) Eq 15.191 for labeled RFS measurement likelihood
    # function

            
    # Measurement update
    # All missed detections
    hyp_list = sorted(list(GLMB_dict.keys()))
    new_hyp_ind = hyp_list[-1] + 1
    hyp_del_list = []
    if nmeas == 0:
        
        for hyp in hyp_list:
        
            hyp_weight = GLMB_dict[hyp]['hyp_weight']
            label_list = GLMB_dict[hyp]['label_list']
            nlabel = len(label_list)
            GLMB_dict[hyp]['hyp_weight'] = hyp_weight*(1-p_det)**nlabel
            
            for sensor_id in unique_sensors:
                lam_clutter = sensor_params[sensor_id]['lam_clutter']
                GLMB_dict[hyp]['hyp_weight'] *= np.exp(-lam_clutter)
            
            # No measurements to update tracks/state estimates
    else:
        
        # Loop over hypotheses
        for hyp in hyp_list:
            
            hyp_weight = GLMB_dict[hyp]['hyp_weight']
            label_list = GLMB_dict[hyp]['label_list']
            nlabel = len(label_list)
            
            print('')
            print('hyp', hyp)
            print('hyp_weight', hyp_weight)
            print('label_list', label_list)
            
            # No tracks means all measurements are clutter
            if nlabel == 0:
                
                # # Compute probability for this number of clutter measurements
                # p_meas = 1.
                # for sensor_id in unique_sensors:
                #     lam_clutter = sensor_params[sensor_id]['lam_clutter']
                #     nmeas_sensor = sensor_id_list.count(sensor_id)
                #     p_poiss = lam_clutter**nmeas_sensor*np.exp(-lam_clutter)/math.factorial(nmeas_sensor)
                    
                #     p_meas *= p_poiss
                    
                #     print('sensor_id', sensor_id)
                #     print('nmeas_sensor', nmeas_sensor)
                #     print('p_poiss', p_poiss)
                #     print('p_meas', p_poiss)
                
                # # Update hypothesis weight, no tracks to update
                # GLMB_dict[hyp]['hyp_weight'] = hyp_weight*p_meas
                
                # TODO - Work out how to handle multisensor case where each
                # could have a different clutter rate lam_clutter
                
                # Single sensor case
                sensor_id = sensor_id_list[0]
                lam_clutter = sensor_params[sensor_id]['lam_clutter']
                likelihood = np.exp(-lam_clutter)
                for ii in range(nmeas):
                    zi = Zk[ii]
                    sensor_id = sensor_id_list[ii]
                    likelihood *= clutter_intensity(zi, sensor_id, sensor_params)
                    
                    
                    
            
            # Compute measurement to track associations
            else:
                
                # Need to replace the current hypothesis with new ones spawned
                # by it
                hyp_del_list.append(hyp)
                
                # Retrieve costs for these labels and measurements
                label_inds = []
                for label in label_list:
                    label_inds.append(full_label_list.index(label))
                   
                # Get negative log cost (Vo, Vo, Phung 2014 Eq 26)
                cost_mat = allcost_mat[label_inds,:]
                neglog_mat = -np.log(cost_mat)
                
                print('')
                print('hyp', hyp)
                print('hyp_weight', hyp_weight)
                print('label_list', label_list)
                print('cost_mat', cost_mat)
                
                # Compute measurement to track assignments
                kbest = int(np.ceil(np.sqrt(H_max*hyp_weight)))
                assign_lists = glmb_kbest_assignments(neglog_mat, kbest)
                
                print('kbest', kbest)
                print('assign_lists', assign_lists)
                print('nmeas', nmeas)
                

                
                likelihood_list = []
                new_hyp_list = []
                for alist in assign_lists:
                    
                    # Create a new hypothesis for each new assignment
                    GLMB_dict[new_hyp_ind] = {}
                    GLMB_dict[new_hyp_ind]['hyp_weight'] = hyp_weight
                    GLMB_dict[new_hyp_ind]['label_list'] = label_list
                    
                    # TODO - Work out how to handle multisensor case where each
                    # could have a different clutter rate lam_clutter
                    
                    # Single sensor case
                    sensor_id = sensor_id_list[0]
                    lam_clutter = sensor_params[sensor_id]['lam_clutter']
                    likelihood = np.exp(-lam_clutter)
                    
                    # Loop over tracks
                    for jj in range(len(alist)):
                        
                        label = label_list[jj]
                        meas_ind = alist[jj]
                        
                        #######################################################
                        # TODO CHECK RETRIEVAL OF TIND
                        #######################################################
                        
                        print('alist', alist)
                        print('label_list', label_list)
                        print('jj', jj)
                        print('label', label)
                        print('meas_ind', meas_ind)
                        
                        # Missed detection                        
                        if meas_ind >= nmeas:
                            
                            # Compute update to track GMM
                            tind = full_label_list.index(label)
                            r = track_update[tind]['r']
                            weights = track_update[tind]['weights']
                            means = track_update[tind]['means']
                            covars = track_update[tind]['covars']
                            
                            print('missed det')
                            print('tind', tind)
                            print('label', track_update[tind]['label'])
                            print('meas_ind', track_update[tind]['meas_ind'])
                            
                            weights = [(1. - p_det)*wi for wi in weights]
                            
                            # Incorporate likelihood
                            eta = sum(weights)
                            assignment_likelihood *= eta
                            
                            # Normalize weights
                            weights = [wi/eta for wi in weights]
                            
                            # Store in GLMB_dict
                            GLMB_dict[new_hyp_ind][label] = {}
                            GLMB_dict[new_hyp_ind][label]['r'] = r
                            GLMB_dict[new_hyp_ind][label]['weights'] = weights
                            GLMB_dict[new_hyp_ind][label]['means'] = means
                            GLMB_dict[new_hyp_ind][label]['covars'] = covars

                        # Detection
                        else:
                            
                            zi = Zk[meas_ind]
                            sensor_id = sensor_id_list[meas_ind]
                            
                            # Compute update to track GMM
                            tind = (meas_ind+1)*len(full_label_list) + full_label_list.index(label)
                            r = track_update[tind]['r']
                            weights = track_update[tind]['weights']
                            means = track_update[tind]['means']
                            covars = track_update[tind]['covars']
                            qk_list = track_update[tind]['qk_list']
                            
                            print('det')
                            print('tind', tind)
                            print('label', track_update[tind]['label'])
                            print('meas_ind', track_update[tind]['meas_ind'])
                            
                            # Incorporate likelihood
                            eta = sum(weights)
                            assignment_likelihood *= eta
                            
                            # Normalize weights
                            weights = [wi/eta for wi in weights]
                            
                            # Store in GLMB_dict
                            GLMB_dict[new_hyp_ind][label] = {}
                            GLMB_dict[new_hyp_ind][label]['r'] = r
                            GLMB_dict[new_hyp_ind][label]['weights'] = weights
                            GLMB_dict[new_hyp_ind][label]['means'] = means
                            GLMB_dict[new_hyp_ind][label]['covars'] = covars
                            
                            
                    # Account for measurements not assigned to tracks as clutter
                    # Not needed, included in denominator of factor?
                    
                    
                    
                    
                            
                    # Store likelihood
                    new_hyp_list.append(new_hyp_ind)
                    likelihood_list.append(assignment_likelihood)
            
                    # Increment hypothesis index
                    new_hyp_ind += 1
                    
                # Normalize likelihood list to get updated hypothesis weights
                # prob_list = [eta/sum(likelihood_list) for eta in likelihood_list]
                
                
                print('new_hyp_list', new_hyp_list)
                print('likelihood_list', likelihood_list)
                # print('prob_list', prob_list)
                
                
                
                # Update hypothesis weights in GLMB_dict
                for hh in range(len(new_hyp_list)):
                    new_hyp_ind2 = new_hyp_list[hh]
                    # prob = prob_list[hh]
                    likelihood = likelihood_list[hh]
                    
                    GLMB_dict[new_hyp_ind2]['hyp_weight'] *= likelihood
                    
                
    # Delete old hypotheses
    for hyp in hyp_del_list:
        del GLMB_dict[hyp]
        
    print('')
    print('GLMB before normalize')
    print(GLMB_dict)
        
    # Renormalize hypothesis weights
    hyp_list = list(GLMB_dict.keys())
    prob_list = []
    for hyp in hyp_list:
        prob_list.append(GLMB_dict[hyp]['hyp_weight'])
    
    final_prob = [prob/sum(prob_list) for prob in prob_list]
    for hyp in hyp_list:
        GLMB_dict[hyp]['hyp_weight'] = final_prob[hyp_list.index(hyp)]
        
    print('')
    print('GLMB after normalize')
    print(GLMB_dict)
        
    # Convert GLMB to LMB
    LMB_dict = glmb2lmb(GLMB_dict, full_label_list)
    
    print('')
    print('LMB posterior')
    print(LMB_dict)
    
    mistake
    
    return LMB_dict






def lmb2glmb(LMB_dict, H_max=1000):
    
    # Get existence probability for each track
    label_list = list(LMB_dict.keys())
    r_list = [LMB_dict[label]['r'] for label in label_list]
    
    # # Compute negative log cost for K-shortest path algorithm
    # cost_vect = np.asarray([(ri/(1.-ri)) for ri in r_vect])
    # neglog_vect = -np.log(cost_vect)
    
    # Compute dictionary of hypothesis weights and labels
    hyp_dict = compute_hypothesis_dict(r_list, label_list, H_max)
    
    # Form GLMB dictionary
    GLMB_dict = {}
    for hyp in hyp_dict:
        GLMB_dict[hyp] = {}
        GLMB_dict[hyp]['hyp_weight'] = hyp_dict[hyp]['weight']
        GLMB_dict[hyp]['label_list'] = hyp_dict[hyp]['label_list']
        
        for label in hyp_dict[hyp]['label_list']:
            GLMB_dict[hyp][label] = LMB_dict[label]

    return GLMB_dict


def glmb2lmb(GLMB_dict, full_label_list=[]):
    
    # Reuter Eq 42-43
    
    
    if len(full_label_list) == 0:
        for hyp in GLMB_dict:
            label_list = GLMB_dict[hyp]['label_list']
            for label in label_list:
                if label not in full_label_list:
                    full_label_list.append(label)
    
    
    
    LMB_dict = {}
    hyp_list = list(GLMB_dict.keys())    
    for label in full_label_list:
        
        r = 0.
        label_weights = []
        label_means = []
        label_covars = []
        for hyp in hyp_list:
            hyp_labels = GLMB_dict[hyp]['label_list']
            if label in hyp_labels:
                
                # Retrieve data for this label in this hypothesis
                hyp_weight = GLMB_dict[hyp]['hyp_weight']
                w_list = GLMB_dict[hyp][label]['weights']
                m_list = GLMB_dict[hyp][label]['means']
                P_list = GLMB_dict[hyp][label]['covars']
                w_list = [hyp_weight*wi for wi in w_list]
                
                # Add to GMM lists representing the state
                label_weights.extend(w_list)
                label_means.extend(m_list)
                label_covars.extend(P_list)
                
                r += hyp_weight
                
        # Divide by r to get correct weights (should normalize to 1?)
        label_weights = [wi/r for wi in label_weights]
        
        # Store data in LMB
        LMB_dict[label] = {}
        LMB_dict[label]['r'] = r
        LMB_dict[label]['weights'] = label_weights
        LMB_dict[label]['means'] = label_means
        LMB_dict[label]['covars'] = label_covars
    
    return LMB_dict


def combine_glmb(GLMB_birth, GLMB_surv):
    
    
    hyp_list_birth = list(GLMB_birth.keys())
    hyp_list_surv = list(GLMB_surv.keys())
    
    GLMB_dict = {}
    hyp_ind = 0
    for shyp in hyp_list_surv:
        shyp_weight = GLMB_surv[shyp]['hyp_weight']
        shyp_labels = GLMB_surv[shyp]['label_list']        
        
        for bhyp in hyp_list_birth:
            bhyp_weight = GLMB_birth[bhyp]['hyp_weight']
            bhyp_labels = GLMB_birth[bhyp]['label_list']
            
            # Build entry for the output GLMB
            GLMB_dict[hyp_ind] = {}
            GLMB_dict[hyp_ind]['hyp_weight'] = shyp_weight*bhyp_weight
            
            # Combined labels for this hypothesis
            label_list = []
            label_list.extend(shyp_labels)
            label_list.extend(bhyp_labels)
            GLMB_dict[hyp_ind]['label_list'] = label_list
            
            
            # print('shyp_labels', shyp_labels)
            # print('bhyp_labels', bhyp_labels)
            # print('label_list', label_list)
            
            # Retrieve state represenation for this label
            for label in label_list:
                GLMB_dict[hyp_ind][label] = {}
                
                # print(label)
                
                if label in shyp_labels:
                    # print('shyp')
                    GLMB_dict[hyp_ind][label] = GLMB_surv[shyp][label]
                
                if label in bhyp_labels:
                    # print('bhyp')
                    GLMB_dict[hyp_ind][label] = GLMB_birth[bhyp][label]

            # Increment hypothesis index
            hyp_ind += 1
    
    
    return GLMB_dict


def combine_lmb(LMB_dict1, LMB_dict2):
    
    LMB_dict = {}
    LMB_dict.update(LMB_dict1)
    LMB_dict.update(LMB_dict2)
    
    return LMB_dict


def lmb_cleanup(LMB_in, params_dict):
    
    filter_params = params_dict['filter_params']
    T_max = filter_params['T_max']
    T_threshold = filter_params['T_threshold']
    
    # Get existence probabilities
    full_label_list = list(LMB_in.keys())
    r_list = []
    for label in full_label_list:
        r_list.append(LMB_in[label]['r'])
        
    # Compute minimum existence probability to keep
    if len(r_list) > T_max:
        sorted_r = sorted(r_list)
        rmin = min([T_threshold, sorted_r[T_max]])
    else:
        rmin = T_threshold
        
    # Only keep tracks with r greater than rmin
    LMB_dict = {}
    for ii in range(len(r_list)):
        ri = r_list[ii]
        if ri > rmin:
            label = full_label_list[ii]
            
            # Merge and prune GMM for this track label
            GMM_dict = {}
            GMM_dict['weights'] = LMB_in[label]['weights']
            GMM_dict['means'] = LMB_in[label]['means']
            GMM_dict['covars'] = LMB_in[label]['covars']

            GMM_dict = est.merge_GMM(GMM_dict, filter_params)
        
            # Store output
            LMB_dict[label] = {}
            LMB_dict[label]['r'] = ri
            LMB_dict[label]['weights'] = GMM_dict['weights']
            LMB_dict[label]['means'] = GMM_dict['means']
            LMB_dict[label]['covars'] = GMM_dict['covars']

    return LMB_dict


def lmb_state_extraction(LMB_dict, tk, Zk, sensor_id_list, meas_fcn,
                         params_dict):
    
    # Break out inputs
    state_params = params_dict['state_params']
    sensor_params = params_dict['sensor_params']
    
    label_list = list(LMB_dict.keys())
    r_list = []
    for label in label_list:
        r_list.append(LMB_dict[label]['r'])
        
    # Compute cardinality and MAP estimate of Nk
    pk = compute_multibern_card(r_list)
    Nk = np.argmax(pk)
    
    
    # r_sort = [r_list[ii] for ii in sorted_inds]
    # label_sort = [label_list[ii] for ii in sorted_inds]
    
    # Extract state estimates for Nk highest tracks
    sorted_inds = sorted(range(len(r_list)), key=lambda k: r_list[k], reverse=True)
    rk_list = []
    Xk_list = []
    Pk_list = []
    labelk_list = []
    params = {}
    params['prune_T'] = 1e-5
    params['merge_U'] = 1e6
    for nn in range(Nk):
        ii = sorted_inds[nn]
        ri = r_list[ii]
        label_i = label_list[ii]
        
        # Merge GMM to get single component output
        GMM_dict = {}
        GMM_dict['weights'] = LMB_dict[label_i]['weights']
        GMM_dict['means'] = LMB_dict[label_i]['means']
        GMM_dict['covars'] = LMB_dict[label_i]['covars']
        
        GMM_dict = est.merge_GMM(GMM_dict, params)
        
        weights = GMM_dict['weights']
        means = GMM_dict['means']
        covars = GMM_dict['covars']
        
        ind = np.argmax(weights)
        Xk = means[ind]
        Pk = covars[ind]
        
        # Store output
        rk_list.append(ri)
        labelk_list.append(label_i)
        Xk_list.append(Xk)
        Pk_list.append(Pk)
    
    
    # Calculate residuals
    resids_out = []
    if len(rk_list) > 0:
        for ii in range(len(Zk)):
            zi = Zk[ii]
            sensor_id = sensor_id_list[ii]
            
            resids_list = []
            for jj in range(len(Xk_list)):
                Xj = Xk_list[jj]            
                zbar = mfunc.compute_measurement(Xj, state_params, sensor_params,
                                                 sensor_id, tk)
                resids = zi - zbar
                resids_list.append(resids)
                
            # Take smallest magnitude as residual for this measurement
            min_list = [np.linalg.norm(resid) for resid in resids_list]
            resids_k = resids_list[min_list.index(min(min_list))]
            resids_out.append(resids_k)
    
    
    return pk, Nk, labelk_list, rk_list, Xk_list, Pk_list, resids_out


def compute_hypothesis_dict(r_list, label_list, H_max=1000):
    
    hyp_dict = {}
    
    # Generate all subsets of labels and existence probabilities
    N = len(label_list)
    ind_list = list(range(N))
    
    subset_list = compute_subsets(ind_list)
    
    # print('ind_list', ind_list)
    # print('subset_list', subset_list)
    
    weight_list = []
    for hyp in range(len(subset_list)):        
        
        # Indices to include for this hypothesis
        subset = subset_list[hyp]
        
        # Compute hypothesis weight and store labels
        hyp_weight = 1.
        hyp_label_list = []
        for ii in ind_list:
            if ii in subset:
                hyp_weight *= r_list[ii]
                hyp_label_list.append(label_list[ii])
            else:
                hyp_weight *= (1. - r_list[ii])
            
        # Store results
        hyp_dict[hyp] = {}
        hyp_dict[hyp]['weight'] = hyp_weight
        hyp_dict[hyp]['label_list'] = hyp_label_list
        weight_list.append(hyp_weight)
        
        
    # Reduce to only include H_max highest probability entries
    if len(weight_list) > H_max:
        sorted_weights = sorted(weight_list, reverse=True)
        wmin = sorted_weights[H_max-1]
        
        wsum = 0.
        hyp_list = list(hyp_dict.keys())
        for hyp in hyp_list:
            if hyp_dict[hyp]['weight'] < wmin:
                del hyp_dict[hyp]
            else:
                wsum += hyp_dict[hyp]['weight']
        
        # Renormalize weights
        for hyp in hyp_dict:
            hyp_dict[hyp]['weight'] *= (1./wsum)

    
    return hyp_dict



###############################################################################
# Utility Functions
###############################################################################


def clutter_intensity(zi, sensor_id, sensor_params):
    
    # Assume clutter is poisson-distributed in number and uniform in spatial
    # distribution
    sensor = sensor_params[sensor_id]
    lam_clutter = sensor['lam_clutter']
    V_sensor = sensor['V_sensor']
    
    kappa = lam_clutter/V_sensor    
    
    return kappa


def compute_subsets(input_list):
    
    N = len(input_list)
    subset_list = []
    for ii in range(1 << N):        
        subset_list.append([input_list[jj] for jj in range(N) if (ii & (1 << jj))])
        
    return subset_list


def compute_multibern_card(r_list):
    
    card = np.zeros(len(r_list)+1,)

    qprod = 1.
    qvec = []
    for j in range(len(r_list)):
        
        rj = r_list[j]
        if rj >= 1.:
            rj = 0.999
        if rj < 1e-3:
            rj = 1e-3

        qprod *= (1.-rj)
        qvec.append((rj/(1.-rj)))

    sigman_list = compute_sigmaj(qvec)
    for n in range(len(card)):
        card[n] = qprod * sigman_list[n]
    
    return card


def compute_sigmaj(qvec):
    '''
    This function computes the elementary symmetric function

    Parameters
    ------
    qvec : list

    Returns
    ------
    sigma_j : list

    Reference
    ------
    Mahler, 2007, P. 641

    '''

    # Length of list
    m = len(qvec)

    if m == 0:
        sigma_j = [1.]
    else:

        # Initialize list of lists
        sig_mat = [[]]*m

        # Build first list of just sums
        sig_i1 = []
        for i in range(1, m+1):
            sig_i1.append(sum(qvec[0:i]))

        sig_mat[0] = sig_i1

        # Build rest of lists
        for i in range(2, m+1):
            sig_mat[i-1] = [0.]*(m-i+1)
            sig_mat[i-1][0] = qvec[i-1] * sig_mat[i-2][0]
            for k in range(i+1, m+1):
                sig_ki = sig_mat[i-1][k-i-1] + qvec[k-1]*sig_mat[i-2][k-i]
                sig_mat[i-1][k-i] = sig_ki

        # Construct output vector
        sigma_j = [1.]
        for ind in range(0, m):
            sigma_j.append(sig_mat[ind][-1])

    return sigma_j


def initialize_kpath(G):
    
    n = int(G.shape[1])
    m = len(np.argwhere(G))
    tail = np.argwhere(G)[:,0]
    head = np.argwhere(G)[:,1]
    W = np.zeros(m,)
    for ii in range(m):
        W[ii] = G[tail[ii], head[ii]]
    
    p = -1*np.ones((n,1))
    D = np.inf*np.ones((n,1))
    
    return m, n, p, D, tail, head, W


def BFMSpathOT(G,r):
    
    m, n, p, D, tail, head, W = initialize_kpath(G)
    p[r] = 0.
    D[r] = 0.
    for iters in range(n-1):
        optimal = True
        for arc in range(m):
            u = int(tail[arc])
            v = int(head[arc])
            duv = W[arc]
            if D[v] > D[u] + duv:
                D[v] = D[u] + duv
                p[v] = u
                optimal = False
            
        if optimal:
            break

    return p, D, iters


def BFMSpathwrap(ncm, source, destination):
    
    p, D, iters = BFMSpathOT(ncm, source)
    dist = D[destination]
    pred = p.flatten()
    
    if np.isinf(dist):
        path = []
    else:
        path = [destination]
        while path[0] != source:
            path.insert(0, int(pred[path[0]]))
    
    
    return dist, path, pred


def glmb_kbest_assignments(C, kbest=1):
    
    
    # Assume input C has tracks on rows and measurements on columns
    
    # Need to minimize cost
    
    # Murty/Auction functions need more rows than columns and seek to maximize
    
    # Augment C with dummy variables
    n1 = int(C.shape[0])
    n2 = int(C.shape[1])
    dum = -np.log(np.ones((n1,n1)))
    C1 = np.concatenate((C, dum), axis=1)
    
    print('C', C)
    print('C1', C1)
    
    # Transpose and reformulate as maximization problem
    A = C1.T
    
    print('A', A)
    
    A = np.max(A) - A
    
    print('A', A)
    
    # Run Murty to get kbest solutions
    final_list = murty(A, kbest)
    
    
    
    
    
    # # Set assignments to dummy variables to -1 (missed detection)
    # final_list2 = []
    # for alist in final_list:
    #     for ii in range(len(alist)):
    #         if alist[ii] > n2:
    #             alist[ii] = -1
    #     final_list2.append(alist)
    
    return final_list



def auction(A) :
    '''
    This function computes a column order of assignments to maximize the score
    for the 2D assignment matrix A.

    Parameters
    ------
    A : NxM numpy array
        score table

    Returns
    ------
    row_indices : list
        each entry in list is assigned row index for the corresponding column
        e.g. row_index[0] = assigned row index for column index 0
    score : float
        total score of assignment
    eps: float
        parameter to increment prices to avoid repeated swapping of same
        assignment bids
    
    References
    ------
    [1] Blackman and Popoli, Section 6.5.1

    '''

    N = int(A.shape[0])
    M = int(A.shape[1])
    eps = 1./(2.*N)
    flag = 0

    #Check if A still has assignments possible
    Acheck = np.zeros((N,M))
    for ii in range(N):
        for jj in range(M):
            if A[ii,jj] > 0.:
                Acheck[ii,jj] = 1.
    sumA = sum(Acheck)

    for ii in range(len(sumA)):
        if sumA[ii] == 0.:
#            print('No more assignments available')
            flag = 1
                       
    if not flag:
        #Step 1: Initialize assignment matrix and track prices
        assign_mat = np.zeros((N,M))
        price = np.zeros((N,1))
        real_price = np.zeros((N,1))
        

#        eps = 0.5

        loop_count = 0

        #Repeat until all columns have been assigned a row
        while np.sum(assign_mat) < M:
            for jj in range(M):

                #print 'j',j
                
                #Step 2: Check if column j is unassigned
                if np.sum(assign_mat[:,jj]) == 0:

                    #Set cost for unallowed assignments
                    for row in range(N):
                        if A[row,jj] <= 0 and price[row] == 0:                            
                            price[row] = 1e15

                            #if row == 0 :
                            #    print 'unallowed cost set'

                    #Step 3: Find the best row i for column j                
                    jvec = np.reshape(A[:,jj],(N,1)) - price
                    ii = np.argmax(jvec)

                    #print 'best i',i

                    #Check if [i,j] is a valid assignment
                    if A[ii,jj] <= 0:
                        flag = 1
                        break

                    #Step 4: Assign row i to column j
                    assign_mat[ii,:] = np.zeros((1,M))
                    assign_mat[ii,jj] = 1.

                    #Step 5: Compute new price
                    jvec2 = np.sort(list(np.reshape(jvec,(1,N))))
                    yj = jvec2[0][-1] - jvec2[0][-2]                
                    real_price[ii] = real_price[ii] + yj + eps
                    price = copy.copy(real_price)

##                    print 'yj',yj
##                    print 'eps',eps
##                    print 'price',price[i]

##                    #Reset price for unallowed assignments
##                    for row in xrange(0,N) :
##                        if A[row,j] <= 0 :
##                            price[row] = 0.

                    #print 'assign_mat',assign_mat
                    #print 'price',price


##            for kk in xrange(0,M) :
##                x = np.nonzero(assign_mat[:,kk])
##                print kk
##                print x[0]

            loop_count += 1
#            print('loop', loop_count)
            if loop_count > 3*M:
                eps *= 2.
                loop_count = 0
                assign_mat = np.zeros((N,M))
                price = np.zeros((N,1))
                real_price = np.zeros((N,1))

            #mistake

            if flag :
                break            

    #Set the row indices to achieve assignment
    row_indices = []
    score = 0.
    #print 'eps',eps
    if not flag :
        for jj in range(M):
            x = np.nonzero(assign_mat[:,jj])       
            row_indices.append(int(x[0]))
            score += A[int(x[0]),jj]

    return row_indices, score, eps



def murty(A0, kbest=1):
    '''
    This function computes the k-best solutions to the 2D assignment problem
    by repeatedly running auction on reduced forms of the input score matrix.

    Parameters
    ------
    A0 : NxM numpy array
        score table
    kbest : int
        number of solutions to return (k highest scoring assignments)
    
    Returns
    ------
    final_list : list of lists
        each entry in list is a row_index list
        each entry in row_indices is assigned row index for the corresponding
        column, e.g. row_indices[0] = assigned row index for column index 0
    
    
    References
    ------
    [2] Cox and Hingorani

    '''

    #Form association table
    N = int(A0.shape[0])
    M = int(A0.shape[1])
    
    #Step 1: Solve for the best solution
    row_indices, score, eps = auction(A0)

    #print 'A',A
#    print(row_indices)
#    print(score)

    #Step 2: Initialize List of Problem/Solution Pairs
    candidate_A_list = [A0]
    candidate_solution_list = [row_indices]
    score_list = [score]

    #Step 3: Clear the list of solutions to be returned
    solution_list = []

    #Step 4: Loop to find kbest possible solutions
    for ind in range(kbest):

#        print('ind',ind)
#        print('A_list',candidate_A_list)
#        print('cand_list', candidate_solution_list)
#        print('scores',score_list)

        if not candidate_solution_list :
            # print('No more solutions available')
            break

        #Step 4.1: Find the best solution in PS_list
        best_ind = np.argmax(score_list)
        A1 = candidate_A_list[best_ind]
        S = candidate_solution_list[best_ind]

        #Step 4.2: Remove this entry from PS_list and score list
        del candidate_A_list[best_ind]
        del candidate_solution_list[best_ind]
        del score_list[best_ind]

        #Step 4.3: Add this solution to the final list
        solution_list.append(S)
#        print('solution_list', solution_list)

        #Step 4.4: Loop through all solution pairs in S
        for j in range(0,len(S)):
            
#            print('\n\n', j)

            #Step 4.4.1: Set A2 = A1
            A2 = copy.copy(A1)

            #Step 4.4.2: Remove solution [i,j] from A2
            i = S[j]
            A2[i,j] = 0.

#            print(A1)
#            print(A2)

            #Step 4.4.3: Solve for best remaining solution
            row_indices, score, eps = auction(A2)


            #Step 4.4.4: If solution exists, add to PS_list
            if row_indices:
                candidate_A_list.append(A2)
                candidate_solution_list.append(row_indices)
                score_list.append(score)
            else:
                continue

            #Step 4.4.5: Remove row/col from A1 except [i,j]           
            for i1 in range(N):
                if i1 != i:
                    A1[i1,j] = 0.
            
            for j1 in range(M):
                if j1 != j:
                    A1[i,j1] = 0.

#            print(row_indices)
#            print(score)
#            print('A1',A1)
                               

    #Remove duplicate solutions
    final_list = []    
    for i in solution_list:
        flag = 0
        for j in final_list:
            if i == j:
                flag = 1
        if not flag:
            final_list.append(i)
            
    return final_list








