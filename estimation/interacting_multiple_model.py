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
# This file contains functions to implement the Interacting Multiple Model
# algorithm, used to identify and estimate maneuvering target parameters.
#
# References
#   [1] Blackman, S. and Popoli, R., "Design and Analysis of Modern Tracking
#       Systems," Artech House Radar Library, 1999.
#
#
#
###############################################################################


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
    TPM = data[3]
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
        
        # Mixing Step
        model_bank = imm_mixing(model_bank, TPM, method)
        
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


def imm_mixing(model_bank_in, TPM, method='imm_mixcovar'):
    '''
    
    '''
    
    # Initialize output
    model_bank = copy.deepcopy(model_bank_in, TPM)
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
















