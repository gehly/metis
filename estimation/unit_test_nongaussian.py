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

import estimation.analysis_functions as analysis
import estimation.estimation_functions as est
import dynamics.dynamics_functions as dyn
import sensors.measurement_functions as mfunc
import sensors.sensors as sens
import sensors.visibility_functions as visfunc
import utilities.astrodynamics as astro
import utilities.coordinate_systems as coord
import utilities.eop_functions as eop
from utilities.constants import GME, arcsec2rad



def twobody_geo_aegis_prop():
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
    state_params['Q'] = 1e-16 * np.diag([1, 1, 1])
    state_params['gap_seconds'] = 900.
    state_params['alpha'] = 1e-4
    state_params['A_fcn'] = dyn.A_twobody
    state_params['dyn_fcn'] = dyn.ode_twobody_ukf
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'solve_ivp'
    int_params['ode_integrator'] = 'DOP853'
    int_params['intfcn'] = dyn.ode_aegis
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'
    int_params['delta_s_sec'] = 600.
    int_params['split_T'] = 0.03

    # Time vector
    UTC0 = datetime(2021, 6, 21, 0, 0, 0)
    UTC1 = datetime(2021, 6, 21, 18, 0, 0)
    tk_list = [UTC0, UTC1]

    # Inital State
    elem = [42164.1, 0.001, 0., 90., 0., 0.]
    X_true = np.reshape(astro.kep2cart(elem), (6,1))
    P = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    
    state_dict = {}
    state_dict[tk_list[0]] = {}
    state_dict[tk_list[0]]['weights'] = [1.]
    state_dict[tk_list[0]]['means'] = [X_true]
    state_dict[tk_list[0]]['covars'] = [P]
    
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
    alpha = state_params['alpha']
    lam = alpha**2.*(nstates + kappa) - nstates
    gam = np.sqrt(nstates + lam)
    Wm = 1./(2.*(nstates + lam)) * np.ones(2*nstates,)
    Wc = Wm.copy()
    Wm = np.insert(Wm, 0, lam/(nstates + lam))
    Wc = np.insert(Wc, 0, lam/(nstates + lam) + (1 - alpha**2 + beta))
    diagWc = np.diag(Wc)
    state_params['gam'] = gam
    state_params['Wm'] = Wm
    state_params['diagWc'] = diagWc
    
    
    start = time.time()
    
    # Run AEGIS Propagation
    tin = tk_list
    GMM_final = est.aegis_predictor(GMM_dict, tin, state_params, int_params)
    
    weights = GMM_final['weights']
    print(len(weights))
    
    print('AEGIS run time', time.time() - start)
    
    N = 1000
    aegis_points = est.gmm_samples(GMM_final, N)
    
    start = time.time()
    
    
    # Run UKF Propagation
    int_params['split_T'] = 1e6
    ukf_final = est.aegis_predictor(GMM_dict, tin, state_params, int_params)
    ukf_points = est.gmm_samples(ukf_final, N)
    
    
    # Monte-Carlo Propagation
    # Generate samples
    mc_init = np.random.multivariate_normal(X_true.flatten(),P,int(N))
    mc_final = np.zeros(mc_init.shape)
    
    # Propagate samples
    int_params['intfcn'] = dyn.ode_twobody
    for jj in range(mc_init.shape[0]):
        
        int0 = mc_init[jj].flatten()
        tout, Xout = dyn.general_dynamics(int0, tk_list, state_params, int_params)
        mc_final[jj,:] = Xout[-1,:].flatten()
        
    print('Monte Carlo run time', time.time() - start)
    
    
    
    # Likelihood Agreement Measure
    aegis_lam = est.compute_LAM(GMM_final, mc_final)
    ukf_lam = est.compute_LAM(ukf_final, mc_final)
    
    print('AEGIS LAM: ', aegis_lam)
    print('UKF LAM: ', ukf_lam)
    
    
    # Generate plots
    plt.figure()
    plt.plot(mc_final[:,0], mc_final[:,1], 'k.')
    plt.plot(aegis_points[:,0], aegis_points[:,1], 'r.')
    plt.plot(ukf_points[:,0], ukf_points[:,1], 'b.')
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    
    plt.show()
    
    
    
    
    return






if __name__ == '__main__':
    
    
    plt.close('all')    
    
    twobody_geo_aegis_prop()

















