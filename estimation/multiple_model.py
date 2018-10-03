import numpy as np
import pickle
import copy
import sys

from skyfield.api import utc

sys.path.append('../')

from estimation import ukf_3dof_predictor
from estimation import ukf_3att_predictor
from estimation import ukf_6dof_predictor
from estimation import ukf_3dof_corrector
from estimation import ukf_3att_corrector
from estimation import ukf_6dof_corrector

from utilities.time_systems import dt2jd
from utilities.eop_functions import get_eop_data
from sensors.measurements import compute_measurement


###############################################################################
#
# This file contains functions to implement Multiple Model Filtering
# algorithms, used to identify and estimate maneuvering target parameters.
#
# References
#   [1] Blackman, S. and Popoli, R., "Design and Analysis of Modern Tracking
#       Systems," Artech House Radar Library, 1999.
#
#
#
###############################################################################





def multiple_model_filter(model_params_file, sensor_file, meas_file,
                          filter_output_file, ephemeris, ts, method='mmae',
                          alpha=1.):
    '''
    
    '''
    
    # Load model parameters
    pklFile = open(model_params_file, 'rb')
    data = pickle.load(pklFile)
    model_bank = data[0]
    eop_alldata = data[1]
    XYs_df = data[2]
    pklFile.close()
    
    print(model_bank)
    
    # Load sensor data
    pklFile = open(sensor_file, 'rb')
    data = pickle.load(pklFile)
    sensor_dict = data[0]
    pklFile.close()    
    
    # Load measurement data
    pklFile = open(meas_file, 'rb')
    data = pickle.load(pklFile)
    meas_times = data[0]
    meas = data[1]
    pklFile.close()
    
    # Sun and earth data
    earth = ephemeris['earth']
    sun = ephemeris['sun']
    
    # Sensor and measurement parameters
    sensor_id = list(sensor_dict.keys())[0]
    sensor = sensor_dict[sensor_id]
    
    # Save initial state in output
    output_dict = {}
    model_id0 = list(model_bank.keys())[0]
    t0 = model_bank[model_id0]['spacecraftConfig']['time']
    X0 = model_bank[model_id0]['spacecraftConfig']['X']
    n = len(X0)
    X0 = np.reshape(X0, (n,1))
    P0 = model_bank[model_id0]['spacecraftConfig']['covar']
    extracted_model = {}
    extracted_model['est_weights'] = np.array([1.])
    extracted_model['est_means'] = np.reshape(X0[0:6], (6,1))
    extracted_model['est_covars'] = P0.copy()
    
    UTC_JD = dt2jd(t0)
    output_dict[UTC_JD] = {}
    output_dict[UTC_JD]['extracted_model'] = copy.deepcopy(extracted_model)
    output_dict[UTC_JD]['model_bank'] = copy.deepcopy(model_bank)

    # Loop over times    
    for ii in range(len(meas_times)):
        
        # Retrieve current and previous times
        ti = meas_times[ii]
        UTC_JD = dt2jd(ti)
        print('Current time: ', ti)
        
        # Predictor step
        model_bank = multiple_model_predictor(model_bank, ti, alpha)    
        
        print('predictor')
        print(model_bank)
        
        
        # Read the next observation
        Yi = meas[ii,:].reshape(len(meas[ii,:]),1)
        
        print('Yi', Yi)

        # Skyfield time and sun position
        UTC_skyfield = ts.utc(ti.replace(tzinfo=utc))
        sun_gcrf = earth.at(UTC_skyfield).observe(sun).position.km
        sun_gcrf = np.reshape(sun_gcrf, (3,1))
        
        # EOPs at current time
        EOP_data = get_eop_data(eop_alldata, ti)
        
        # Corrector step
        model_bank = multiple_model_corrector(model_bank, Yi, ti, sun_gcrf,
                                              sensor, EOP_data, XYs_df,
                                              method, alpha)
        
        print('corrector')
        
        # Estimate extractor
        extracted_model, model_bank = estimate_extractor(model_bank, method)
        print('extracted model')
        print(extracted_model)
        print('model bank')
        print(model_bank)
        
        # Output
        output_dict[UTC_JD] = {}
        output_dict[UTC_JD]['extracted_model'] = copy.deepcopy(extracted_model)
        output_dict[UTC_JD]['model_bank'] = copy.deepcopy(model_bank)
        
#        if ii > 3:
#            mistake
        
        # Save data
        pklFile = open( filter_output_file, 'wb' )
        pickle.dump( [output_dict], pklFile, -1 )
        pklFile.close()

        
    return output_dict


def imm_filter(model_params_file, sensor_file, meas_file, filter_output_file,
               ephemeris, ts, method='imm_mixcovar', alpha=1.):
    '''
    
    '''
    
    # Load model parameters
    pklFile = open(model_params_file, 'rb')
    data = pickle.load(pklFile)
    model_bank = data[0]
    eop_alldata = data[1]
    XYs_df = data[2]
    TPM0 = data[3]
    pklFile.close()
    
    print(model_bank)
    
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
    
    # Sun and earth data
    earth = ephemeris['earth']
    sun = ephemeris['sun']
    
    # Sensor and measurement parameters
    sensor_id = list(sensor_dict.keys())[0]
    sensor = sensor_dict[sensor_id]
    
    # Save initial state in output
    output_dict = {}
    model_id0 = list(model_bank.keys())[0]
    t0 = model_bank[model_id0]['spacecraftConfig']['time']
    X0 = model_bank[model_id0]['spacecraftConfig']['X']
    n = len(X0)
    X0 = np.reshape(X0, (n,1))
    P0 = model_bank[model_id0]['spacecraftConfig']['covar']
    extracted_model = {}
    extracted_model['est_weights'] = np.array([1.])
    extracted_model['est_means'] = np.reshape(X0[0:6], (6,1))
    extracted_model['est_covars'] = P0.copy()
    
    UTC_JD = dt2jd(t0)
    output_dict[UTC_JD] = {}
    output_dict[UTC_JD]['extracted_model'] = copy.deepcopy(extracted_model)
    output_dict[UTC_JD]['model_bank'] = copy.deepcopy(model_bank)

    # Loop over times
    TPM = TPM0
    for ii in range(len(meas_times)):
        
        # Retrieve current and previous times
        ti = meas_times[ii]
        UTC_JD = dt2jd(ti)
        print('Current time: ', ti)
        
        # Mixing Step
        if ii > 0:
            ti_prior = meas_times[ii-1]
            delta_t = (ti - ti_prior).total_seconds()
            if delta_t < 100.:
                TPM = np.eye(len(model_bank))
                print('TPM', TPM)
            else:
                TPM = TPM0
                print('TPM', TPM)
        model_bank = imm_mixing(model_bank, TPM, method)
        
        # Predictor step
        model_bank = multiple_model_predictor(model_bank, ti, alpha)    
        
        print('predictor')
        print(model_bank)

        # Skyfield time and sun position
        UTC_skyfield = ts.utc(ti.replace(tzinfo=utc))
        sun_gcrf = earth.at(UTC_skyfield).observe(sun).position.km
        sun_gcrf = np.reshape(sun_gcrf, (3,1))
        
        # EOPs at current time
        EOP_data = get_eop_data(eop_alldata, ti)
        
        sensor_id_list = list(meas_dict[ti].keys())
        for sensor_id in sensor_id_list:
            sensor = sensor_dict[sensor_id]
            Yi = meas_dict[ti][sensor_id]
            
            # Read the next observation
#            Yi = meas[ii,:].reshape(len(meas[ii,:]),1)
            
            print('Yi', Yi)
        
        # Corrector step
        model_bank = multiple_model_corrector(model_bank, Yi, ti, sun_gcrf,
                                              sensor, EOP_data, XYs_df,
                                              method, alpha)
        
        print('corrector')
        
        # Estimate extractor
        extracted_model, model_bank = estimate_extractor(model_bank, method)
        print('extracted model')
        print(extracted_model)
        print('model bank')
        print(model_bank)

        # Output
        output_dict[UTC_JD] = {}
        output_dict[UTC_JD]['extracted_model'] = copy.deepcopy(extracted_model)
        output_dict[UTC_JD]['model_bank'] = copy.deepcopy(model_bank)
        
#        if ii > 3:
#            mistake
        
        # Save data
        pklFile = open( filter_output_file, 'wb' )
        pickle.dump( [output_dict], pklFile, -1 )
        pklFile.close()

        
    return output_dict


def imm_mixing(model_bank_in, TPM, method='imm_mixcovar'):
    '''
    
    '''
    
    # Initialize output
    model_bank = copy.deepcopy(model_bank_in)
    model_id_list = list(sorted(model_bank.keys()))
    r = len(model_id_list)

    # Retrive current model probabilities (weights), states, and covars
    wi_list = []
    mi_list = []
    Pi_list = []
    for model_id in model_id_list:
        
        n = len(model_bank[model_id]['spacecraftConfig']['X'])
        
        wi_list.append(model_bank[model_id]['weight'])
        mi_list.append(model_bank[model_id]['spacecraftConfig']['X'].reshape(n,1))
        Pi_list.append(model_bank[model_id]['spacecraftConfig']['covar'])

    # Compute normalizing factors
    mu_prior = np.reshape(wi_list, (r, 1))
    C = np.dot(TPM.T, mu_prior)
    
    # Compute conditional probability matrix
    mu = np.zeros((r,r))
    for ii in range(r):
        for jj in range(r):
            mu[ii,jj] = TPM[ii,jj] * mu_prior[ii]/C[jj]
    
    # Compute mixed state and covariance for each model
    mj_list = []
    Pj_list = []
    for jj in range(r):        
        
        # Compute weighted mean state mj
        mj = np.zeros((n,1))
        for ii in range(r):
            mi = mi_list[ii]
            mj += mu[ii,jj] * mi
            
        # Compute weighted mean covar Pj
        # Option to mix covariances
        if method == 'imm_mixcovar':
            Pj = np.zeros((n,n))
            for ii in range(r):
                mi = mi_list[ii]
                Pi = Pi_list[ii]
                DP = np.dot((mi-mj), (mi-mj).T)
                Pj += mu[ii,jj] * (Pi + DP)
        
        # Option to retain model covariance - no mixing
        # Per Reference [1] P. 232, mixing covars may degrade performance
        else:
            Pj = Pi_list[jj]
            
        # Store output
        mj_list.append(mj)
        Pj_list.append(Pj)
        
    # Recompute model bank
    for jj in range(r):
        model_id = model_id_list[jj]
        model_bank[model_id]['weight'] = float(C[jj])
        model_bank[model_id]['spacecraftConfig']['X'] = mj_list[jj].reshape(n,1)
        model_bank[model_id]['spacecraftConfig']['covar'] = Pj_list[jj]    

    return model_bank
    
    
def multiple_model_predictor(model_bank0, ti, alpha=1.):
    
    
    # Initialize output
    model_bank = copy.deepcopy(model_bank0)
    
    # Loop over models
    model_id_list = sorted(model_bank0.keys())
    for model_id in model_id_list:
        
        print(model_id)
    
        # Extract model parameters
        spacecraftConfig = model_bank0[model_id]['spacecraftConfig']
        forcesCoeff = model_bank0[model_id]['forcesCoeff']
        surfaces = model_bank0[model_id]['surfaces']
        
        #Number of states and observations per epoch
        n = len(spacecraftConfig['X'])    
        
        # Initial state parameters
        X = spacecraftConfig['X'].reshape(n,1)
        P = spacecraftConfig['covar']    
           
        print(spacecraftConfig)
        print(X)
        
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
    
        # Update output
        model_bank[model_id]['spacecraftConfig']['X'] = Xbar.copy()
        model_bank[model_id]['spacecraftConfig']['covar'] = Pbar.copy()
    
    return model_bank


def multiple_model_corrector(model_bank_in, Yi, ti, sun_gcrf, sensor, EOP_data,
                             XYs_df, method='mmae', alpha=1e-4):
    
    # Initialize Output
    model_bank = copy.deepcopy(model_bank_in)

    # Compute update components
    wbar = []
    beta_list = []
    for model_id in sorted(model_bank_in.keys()):

        # Retrieve model parameters        
        spacecraftConfig = model_bank_in[model_id]['spacecraftConfig']
        surfaces = model_bank_in[model_id]['surfaces']
        Xbar = spacecraftConfig['X']
        Pbar = spacecraftConfig['covar']
        n = len(Xbar)
        
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
        print(X)
        print(P)
        print(beta)
        
        # Compute post-fit residuals
        Ybar_post = compute_measurement(X, sun_gcrf, sensor, spacecraftConfig,
                                        surfaces, ti, EOP_data,
                                        sensor['meas_types'], XYs_df)
        resids = Yi - Ybar_post
        
        print('post')
        print(model_id)
        print('Ybar_post', Ybar_post)
        print('resids', resids)
        
        # Store weights and likelihoods for weight updates
        wbar.append(model_bank_in[model_id]['weight'])
        beta_list.append(beta)
        
        # Update outputs for each model
        model_bank[model_id]['spacecraftConfig']['time'] = ti
        model_bank[model_id]['spacecraftConfig']['X'] = X
        model_bank[model_id]['spacecraftConfig']['covar'] = P
        model_bank[model_id]['resids'] = resids
    
    # Update weights per mulitple model method
    wf_list = multiple_model_weights(wbar, beta_list, method)
    
    print('Update Weights')
    print('wbar', wbar)
    print('beta_list', beta_list)
    print('wf_list', wf_list)
    
    # Update model bank
    ii = 0
    for model_id in sorted(model_bank_in.keys()):
        model_bank[model_id]['weight'] = np.array(wf_list[ii])
        ii += 1
    
    
    return model_bank


def multiple_model_weights(wbar, beta_list, method='mmae'):
    '''
    This function computes the weights for multiple model GMM components,
    according to different methods as specified in the function call.
    
    Parameters
    ------
    wbar : list
        list of multiple model GMM weights
    beta_list : list
        list of Gaussian likelihoods computed in corrector
    method : string, optional
        flag to specify method to compute weights (default = 'mmae')
    
    Returns
    ------
    wf_list : list
        list of normalized multiple model GMM weights
    
    References
    ------
    [1] R. Brown and P. Hwang, "Introduction to Random Signals and Applied
    Kalman Filtering," 4th ed, Section 6.5, 2012.
    
    [2] R. Linares, M. Jah, and J. Crassidis, "Space object area-to-mass ratio
    estimation using multiple model approaches," AAS Conf, 2012.
    
    '''
    
    
    # Multiple Model Adaptive Estimation
    # Reference 1
    if method == 'mmae' or 'imm' in method:
        
        # Normalize updated weights, muliply by previous
        denom = np.dot(beta_list, wbar)
        wf_list = [a1*a2/denom for a1, a2 in zip(wbar, beta_list)]
    
    # Adaptive Likelihood Mixture
    # Reference 2
    elif method == 'alm':
        
        # Normalize updated weights using current likelihood only
        denom = sum(beta_list)
        wf_list = [beta_list[ii]/denom for ii in range(len(beta_list))]
        
#        print
#        print 'alm detail'
#        print 'beta_list', beta_list
#        print 'denom', denom
#        print beta_list[0]/denom
#        print beta_list[1]/denom
#        print wf_list
    
    return wf_list


def estimate_extractor(model_bank, method='averaged'):
    '''
    This function computes estimated state to include in output according
    to the specified method.
    
    Parameters
    ------
    wbar : list
        list of multiple model GMM weights
    beta_list : list
        list of Gaussian likelihoods computed in corrector
    method : string, optional
        flag to specify method to compute weights (default = 'mmae')
    
    Returns
    ------
    wf_list : list
        list of normalized multiple model GMM weights
    
    References
    ------
    [1] R. Brown and P. Hwang, "Introduction to Random Signals and Applied
    Kalman Filtering," 4th ed, Section 6.5, 2012.
    
    [2] R. Linares, M. Jah, and J. Crassidis, "Space object area-to-mass ratio
    estimation using multiple model approaches," AAS Conf, 2012.
    
    '''
    
    extracted_model = {}
    
    
    if method == 'averaged':
        
        wbar, mbar, Pbar = merge_model_bank(model_bank)

        extracted_model = {}
        extracted_model['est_weights'] = np.array([wbar])
        extracted_model['est_means'] = mbar.copy()
        extracted_model['est_covars'] = Pbar.copy()
        
        # Reset pos/vel states to average values
        for model_id in model_bank:
            X = model_bank[model_id]['spacecraftConfig']['X']
            if len(X) > 6:
                Xatt = X[6:13].reshape(7,1)
                Xf = np.concatenate((mbar, Xatt), axis=0)
            else:
                Xf = mbar.copy()
            
            model_bank[model_id]['spacecraftConfig']['X'] = Xf.copy()
            model_bank[model_id]['spacecraftConfig']['covar'] = Pbar.copy()
            
    
    # Traditional MMAE uses weighted average as the best estimate
    # Reference 1
    elif method == 'mmae' or 'imm' in method:
            
        wbar, mbar, Pbar = merge_model_bank(model_bank)

        extracted_model = {}
        extracted_model['est_weights'] = np.array([wbar])
        extracted_model['est_means'] = mbar.copy()
        extracted_model['est_covars'] = Pbar.copy()
        
        
    
    # Adaptive Likelihood Mixtures resets individual models with weighted 
    # average, Reference 2
    elif method == 'alm':
        
        wbar, mbar, Pbar = merge_model_bank(model_bank)
            
        n = len(mbar)
        covars = np.zeros((n,n,1))
        covars[0:n,0:n,0] = Pbar
        
        extracted_model = {}
        extracted_model['est_weights'] = np.array([wbar])
        extracted_model['est_means'] = mbar.copy()
        extracted_model['est_covars'] = Pbar.copy()
        
        for model_id in model_bank:
            wsum = sum(model_bank[model_id]['weight'])
            model_bank[model_id]['weight'] = np.array([wsum])
            model_bank[model_id]['spacecraftConfig']['X'] = mbar.copy()
            model_bank[model_id]['spacecraftConfig']['covar'] = Pbar.copy()
            
            
#    # Max will output the highest weighted model as the best estimate
#    # This just seems like a good idea
#    elif method == 'max':
#        
#        model_weight_dict = sum_model_weights(model_bank)
#        
#        for obj_id in model_bank:
#            
#            model_list = model_weight_dict[obj_id]['model_list']
#            wsum_list = model_weight_dict[obj_id]['wsum_list']
#            
#            ind = wsum_list.index(max(wsum_list))
#            model_id = model_list[ind]
#            
#            extracted_model = {}
#            extracted_model[obj_id]['est_weights'] = \
#                model_bank[obj_id][model_id]['weights'].copy()
#            extracted_model[obj_id]['est_means'] = \
#                model_bank[obj_id][model_id]['means'].copy()
#            extracted_model[obj_id]['est_covars'] = \
#                model_bank[obj_id][model_id]['covars'].copy()

    return extracted_model, model_bank


def merge_model_bank(model_bank):
    
    
    w_list = []
    m_list = []
    P_list = []
    for model_id in sorted(model_bank.keys()):
        
        n = len(model_bank[model_id]['spacecraftConfig']['X'])
        
        w_list.append(model_bank[model_id]['weight'])
        m_list.append(model_bank[model_id]['spacecraftConfig']['X'].reshape(n,1))
        P_list.append(model_bank[model_id]['spacecraftConfig']['covar'])

        
    wbar, mbar, Pbar = merge_GMM_list(w_list, m_list, P_list)    
    
    return wbar, mbar, Pbar
    

def merge_GMM_list(w_list, m_list, P_list):
    
    wbar = sum(w_list)
    m_list2 = [np.reshape(m[0:6], (6,1)) for m in m_list]
    msum = sum([w_list[ii]*m_list2[ii][0:6] for ii in range(len(w_list))])            
    mbar = (1./wbar) * msum
    
    Psum = np.zeros(P_list[0].shape)
    for ii in range(len(w_list)):
        wi = w_list[ii]
        mi = m_list2[ii]
        Pi = P_list[ii]
        
        Psum += wi*(Pi + np.dot((mbar-mi), (mbar-mi).T))
    
    Pbar = (1./wbar) * Psum

    
    return wbar, mbar, Pbar

