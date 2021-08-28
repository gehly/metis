import numpy as np
from math import pi
from matplotlib import pyplot as plt
from datetime import datetime
import pickle
import copy
import time

from state import State
from visibility import LambertianSphere
from visibility import ashikhminPremoze
from forces import NeutralDrag
from forces import HolmesFeatherstoneGravity
from forces import ThirdBodyForce
from forces import ClassicalRadiation
from forces import ImprovedRadiation
from torques import Torques
from torques import GravityGradient
from torques import MagneticDipole
from propagator import NumericalPropagator





def unscented_kalman_filter(model_params_file, sensor_file, meas_file, alpha=1e-4):
    '''
    
    '''
    
    # Load model parameters
    pklFile = open(model_params_file, 'rb')
    data = pickle.load(pklFile)
    spacecraftConfig = data[0]
    forcesCoeff = data[1]
    brdfCoeff = data[2]
    pklFile.close()
    
    # Load sensor data
    pklFile = open(sensor_file, 'rb')
    data = pickle.load(pklFile)
    sensor_dict = data[0]
    pklFile.close()    
    
    # Load measurement data
    pklFile = open(meas_file, 'rb')
    data = pickle.load(pklFile)
    meas_time = data[0]
    az = data[1]
    el = data[2]
    mapp = data[3]
    pklFile.close()
    
    # Force model parameters
    dragCoeff = forcesCoeff['dragCoeff']
    order = forcesCoeff['order']
    degree = forcesCoeff['degree']
    emissivity = forcesCoeff['emissivity']
    
    # Initial state parameters
    X = spacecraftConfig['orbit']
    P = spacecraftConfig['covar']
    Q = forcesCoeff['Q']
    
    # Sensor and measurement parameters
    sensor_id = list(sensor_dict.keys())[0]
    sensor = sensor_dict[sensor_id]
    meas_types = sensor['meas_types']
    sigma_dict = sensor['sigma_dict']
    sigma_dict['az'] *= 180./pi
    sigma_dict['el'] *= 180./pi
    gs = sensor['geodetic_latlonht']
    gs[2] *= 1000.  # convert to meters
    
    #Number of states and observations per epoch
    n = len(spacecraftConfig['orbit'])
    p = len(meas_types)
    
    print(spacecraftConfig)
    print(spacecraftConfig['orbit'])
    
    # Measurement noise
    var = []
    for mt in meas_types:
        var.append(sigma_dict[mt]**2.)
    Rk = np.diag(var)
    
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
    
    # Loop over times
    beta_list = []
    filter_output = {}
    filter_output['time'] = []
    filter_output['X'] = []
    filter_output['P'] = []
    filter_output['resids'] = []
    for ii in range(len(meas_time)):
        
        # Retrieve current and previous times
        ti = meas_time[ii]
        print('Current time: ', ti)
        
        date = spacecraftConfig['date']
        ti_prior = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f")
        print(ti_prior)
            

        delta_t = (ti - ti_prior).total_seconds()
        if delta_t > 11.:
            step = 10.
        else:
            step = 1.
        
        print('delta_t', delta_t)
        
        # Read the next observation
        Yi = np.array([[az[ii]], [el[ii]], [mapp[ii]]])   
        
        print('Yi', Yi)
        
        # Predictor step
        # Compute sigma points for propagation
        sqP = np.linalg.cholesky(P)
        Xrep = np.tile(X, (1, n))
        chi = np.concatenate((X, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
#        chi_v = np.reshape(chi, (L*(2*L+1), 1), order='F')
        
#        print(chi)
        
        # Integrate chi
        chi_bar = np.zeros(chi.shape)
        meas_bar = np.zeros((p, chi.shape[1]))
        for jj in range(chi.shape[1]):
            
            print(jj)
            
            spacecraftConfig['orbit'] = chi[:,jj].reshape(n,1)
            
            spacecraftState = State(spacecraftConfig)

            # Drag model
            dragNeutral = NeutralDrag(spacecraftState,dragCoeff)
        
            # Gravity Model
            gravity = HolmesFeatherstoneGravity(spacecraftState,order,degree)
        
            # 3-Body Model
            sunBody = spacecraftState.sun
            moonBody = spacecraftState.moon
            thirdBodies = ThirdBodyForce(spacecraftState,[sunBody,moonBody])
        
            sphereModel = LambertianSphere(spacecraftState,gs,brdfCoeff)
            SRPImproved = ImprovedRadiation(spacecraftState,brdfCoeff,emissivity)
        
            # Setup and run propagator
            prop = NumericalPropagator(delta_t,step,spacecraftState,'dop853',\
                                       [dragNeutral,gravity,thirdBodies,SRPImproved],\
                                        [],sphereModel)
        
        
            if delta_t == 0.:
                chi_bar[:,jj] = chi[:,jj]                
                
                vis = sphereModel.ComputeVisibilityState()
                appMagnitude = sphereModel.ComputeAppMagnitude()
                
                meas_bar[0,jj] = vis[1]
                meas_bar[1,jj] = vis[2]
                meas_bar[2,jj] = appMagnitude
                
            else :
                
                start = time.time()
                prop.Propagate()
                print('propagate time:', time.time() - start)
                
                # Output data
                sol_time = prop.sol['time']
                state = prop.sol['state']
                visibility = prop.sol['visibility']
                
#                print(sol_time)
#                print(state)            
            
                # Extract values for later calculations
                chi_bar[:,jj] = state[-1,:].flatten()
                meas_bar[0,jj] = visibility[-1,3]
                meas_bar[1,jj] = visibility[-1,4]
                meas_bar[2,jj] = visibility[-1,0]
            
        
        
        
#        print(chi_bar)
#        print(meas_bar)
    
        # Add process noise
        Xbar = np.dot(chi_bar, Wm.T)
        Xbar = np.reshape(Xbar, (n, 1))
        chi_diff = chi_bar - np.dot(Xbar, np.ones((1, (2*n+1))))
        if delta_t > 100.:
            Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T))
        else:
            Pbar = delta_t*Q + np.dot(chi_diff, np.dot(diagWc, chi_diff.T))
    
        # Re-symmetric pos def
        Pbar = 0.5 * (Pbar + Pbar.T)
        
#        print('a priori')
#        print(Xbar)
#        print(Pbar)
        
        # Computed measurements
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
        print(X)
        print(P)
        print(Ybar)
        print(Yi - Ybar)
#        print(Pyy)
#        print(Rk)
#        print(Pxy)

        # Gaussian Likelihood
        beta = compute_gaussian(Yi, Ybar, Pyy)
        beta_list.append(beta)
        
        # Update with post-fit solution
        spacecraftConfig['date'] = ti.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:-3]
        spacecraftConfig['orbit'] = X        
        spacecraftConfig['covar'] = P
        spacecraftState = State(spacecraftConfig)        
        
        # Compute post-fit residuals
        sphereModel = LambertianSphere(spacecraftState,gs,brdfCoeff)
        vis = sphereModel.ComputeVisibilityState()
        appMagnitude = sphereModel.ComputeAppMagnitude()
        ybar_post = np.zeros((3,1))
        ybar_post[0] = vis[1]
        ybar_post[1] = vis[2]
        ybar_post[2] = appMagnitude
            
        resids = Yi - ybar_post
        
        print('post')
        print('ybar_post', ybar_post)
        print('resids', resids)
        
#        if ii > 3:
#            mistake
        
        # Append data to output
        filter_output['time'].append(ti)
        filter_output['X'].append(X)
        filter_output['P'].append(P)
        filter_output['resids'].append(resids)
    

    return filter_output


def compute_gaussian(x, m, P):
    '''
    This function computes the likelihood of the multivariate gaussian pdf
    for a given state x, assuming mean m and covariance P.

    Parameters
    ------
    x : nx1 numpy array
        instance of a random vector
    m : nx1 numpy array
        mean
    P : nxn numpy array
        covariance

    Returns
    ------
    pg : float
        multivariate gaussian likelihood
    '''

    K1 = np.sqrt(np.linalg.det(2*pi*(P)))
    K2 = np.exp(-0.5 * np.dot((x-m).T, np.dot(np.linalg.inv(P), (x-m))))
    pg = (1/K1) * K2
    pg = float(pg)

    return pg




###############################################################################
# Stand-alone execution
###############################################################################


#if __name__ == '__main__':

    