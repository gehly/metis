import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
import pickle
import os
import inspect
import time
from numba import types
from numba.typed import Dict

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from dynamics import fast_integration as fastint
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

    # Define filter parameters
    filter_params = {}
    filter_params['Q'] = 1e-16 * np.diag([1, 1, 1])
    filter_params['gap_seconds'] = 900.
    filter_params['alpha'] = 1e-4
    filter_params['delta_s_sec'] = 600.
    filter_params['split_T'] = 0.1
    
    # Integration function and additional settings
    int_params = {}   
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'

    # Time vector
    UTC0 = datetime(2021, 6, 21, 0, 0, 0)
    UTC1 = datetime(2021, 6, 23, 6, 0, 0)
    tk_list = [UTC0, UTC1]

    # Inital State
    elem = [42164.1, 0.001, 0.001, 90., 10., 10.]
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
    
    
    print('')
    print('state_params', state_params)
    
    # General setup
    tin = tk_list
    N = 5000

    
    # Run Variable Step AEGIS Propagation    
    int_params['integrator'] = 'dopri87_aegis'
    int_params['intfcn'] = dyn.ode_aegis
    int_params['A_fcn'] = dyn.A_twobody
    int_params['dyn_fcn'] = dyn.ode_twobody_ukf
    int_params['step'] = 10.
    params_dict = {}
    params_dict['filter_params'] = filter_params
    params_dict['state_params'] = state_params
    params_dict['int_params'] = int_params
    start = time.time()
    aegis_final3 = est.aegis_predictor(GMM_dict, tin, params_dict)
    
    aegis3_run_time = time.time() - start
    
    aegis_points3 = est.gmm_samples(aegis_final3, N)
    
    print('aegis3 run time', aegis3_run_time)
    
    print('')
    print('state_params', state_params)
    
    
    # Variable Step AEGIS Propagation with JIT
    int_params['integrator'] = 'dopri87_aegis_jit'
    int_params['intfcn'] = fastint.jit_twobody_aegis
    
    start = time.time()
    aegis_final4 = est.aegis_predictor(GMM_dict, tin, filter_params, 
                                        state_params, int_params)
    
    aegis4_run_time = time.time() - start
    
    aegis_points4 = est.gmm_samples(aegis_final4, N)
    
    print('aegis4 run time', aegis4_run_time)
    
    
    # Run UKF Propagation
    int_params['integrator'] = 'dopri87_aegis'
    int_params['intfcn'] = dyn.ode_aegis
    int_params['A_fcn'] = dyn.A_twobody
    int_params['dyn_fcn'] = dyn.ode_twobody_ukf
    filter_params['split_T'] = 1e6
    
    start = time.time()
    ukf_final = est.aegis_predictor(GMM_dict, tin, filter_params,
                                    state_params, int_params)
    ukf_points = est.gmm_samples(ukf_final, N)
    
    ukf_run_time = time.time() - start
    
    print('ukf run time', ukf_run_time)
    
    
    
    # Monte-Carlo Propagation
    # Generate samples
    mc_init = np.random.multivariate_normal(X_true.flatten(),P,int(N))
    mc_final = np.zeros(mc_init.shape)
    
    # Convert time to seconds
    time_format = int_params['time_format']
    if time_format == 'datetime':
        t0 = tk_list[0]
        tvec = np.asarray([(ti - t0).total_seconds() for ti in tk_list])
        
    params2 = Dict.empty(key_type=types.unicode_type, value_type=types.float64,)
    params2['step'] = 10.
    params2['GM'] = GME
    params2['rtol'] = 1e-12
    params2['atol'] = 1e-12
    
    # Propagate samples
    start = time.time()
    intfcn = fastint.jit_twobody
    for jj in range(mc_init.shape[0]):
        
        if jj % 100 == 0:
            print(jj)
            print('elapsed time', time.time() - start)
        
        int0 = mc_init[jj].flatten()
        tout, Xout = fastint.dopri87(intfcn, tvec, int0.flatten(), params2)
        mc_final[jj,:] = Xout[-1,:].flatten()
        
    mc_run_time = time.time() - start
    
    
    
#    # Likelihood Agreement Measure
#    aegis_lam2 = analysis.compute_LAM(aegis_final2, mc_final)
#    aegis_lam3 = analysis.compute_LAM(aegis_final3, mc_final)
#    aegis_lam4 = analysis.compute_LAM(aegis_final4, mc_final)
#    ukf_lam = analysis.compute_LAM(ukf_final, mc_final)
#    
#    print('AEGIS For Loop LAM: ', aegis_lam2)
#    print('AEGIS Variable LAM: ', aegis_lam3)
#    print('AEGIS JIT LAM: ', aegis_lam4)
#    print('UKF LAM: ', ukf_lam)
    
#    print('AEGIS For Loop Comps: ', len(aegis_final2['weights']))
    print('AEGIS Variable Comps: ', len(aegis_final3['weights']))
    print('AEGIS JIT Comps: ', len(aegis_final4['weights']))
    
#    print('AEGIS For Loop Time: ', aegis2_run_time)
    print('AEGIS Variable time: ', aegis3_run_time)
    print('AEGIS JIT time: ', aegis4_run_time)
    print('UKF time: ', ukf_run_time)
    print('MC time: ', mc_run_time)
    
    
    # Generate plots
    plt.figure()
    plt.plot(mc_final[:,0], mc_final[:,1], 'k.', alpha=0.2)    
    analysis.plot_pdf_contours(aegis_final4, axis1=0, axis2=1)
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('AEGIS Contours vs MC Points')
    
    plt.figure()
    plt.plot(mc_final[:,0], mc_final[:,1], 'k.', alpha=0.2)    
    analysis.plot_pdf_contours(ukf_final, axis1=0, axis2=1)
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
#    plt.title('UKF Contours vs MC Points')
    
    
    plt.figure()
    plt.plot(mc_final[:,0], mc_final[:,1], 'k.', alpha=0.2)
    plt.plot(aegis_points4[:,0], aegis_points4[:,1], 'r.', alpha=0.2)
    plt.plot(ukf_points[:,0], ukf_points[:,1], 'b.', alpha=0.2)
    plt.legend(['MC', 'AEGIS', 'UKF'])
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('MC Points vs AEGIS and UKF Samples')
    
    plt.figure()
    plt.plot(mc_final[:,0], mc_final[:,2], 'k.', alpha=0.2)
    plt.plot(aegis_points4[:,0], aegis_points4[:,2], 'r.', alpha=0.2)
    plt.plot(ukf_points[:,0], ukf_points[:,2], 'b.', alpha=0.2)
    plt.legend(['MC', 'AEGIS', 'UKF'])
    plt.xlabel('X [km]')
    plt.ylabel('Z [km]')
    plt.title('MC Points vs AEGIS and UKF Samples')
    
    plt.figure()
    plt.plot(mc_final[:,1], mc_final[:,2], 'k.', alpha=0.2)
    plt.plot(aegis_points4[:,1], aegis_points4[:,2], 'r.', alpha=0.2)
    plt.plot(ukf_points[:,1], ukf_points[:,2], 'b.', alpha=0.2)
    plt.legend(['MC', 'AEGIS', 'UKF'])
    plt.xlabel('Y [km]')
    plt.ylabel('Z [km]')
    plt.title('MC Points vs AEGIS and UKF Samples')
    
    plt.show()
    
    outfile = os.path.join('advanced_test', 'twobody_geo_aegis.pkl')
    pklFile = open( outfile, 'wb' )
    pickle.dump( [aegis_final3, aegis_final4, ukf_final, mc_final], pklFile, -1 )
    pklFile.close()
    
    
    
    return



def twobody_heo_aegis_prop():
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME

    # Define filter parameters
    filter_params = {}
    filter_params['Q'] = 1e-16 * np.diag([1, 1, 1])
    filter_params['gap_seconds'] = 900.
    filter_params['alpha'] = 1e-4
    filter_params['delta_s_sec'] = 600.
    filter_params['split_T'] = 0.1
    
    # Integration function and additional settings
    int_params = {}   
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'

    # Time vector
    UTC0 = datetime(2021, 6, 21, 0, 0, 0)
    UTC1 = datetime(2021, 6, 21, 12, 0, 0)
    tk_list = [UTC0, UTC1]

    # Inital State
    elem = [26600., 0.74, 63.4, 90., 270., 10.]
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
    
    
    print('')
    print('state_params', state_params)
    
    # General setup
    tin = tk_list
    N = 5000
    
    
    # Run AEGIS Propagation
#    start = time.time()
#    
#    aegis_final2 = est.aegis_predictor2(GMM_dict, tin, filter_params, 
#                                        state_params, int_params)
#
#    aegis2_run_time = time.time() - start
#    
#    weights = aegis_final2['weights']
#    print(len(weights))
#    
#    N = 5000
#    aegis_points2 = est.gmm_samples(aegis_final2, N)
#    
#    
#    print('aegis2 run time', aegis2_run_time)
#    
#    print('')
#    print('state_params', state_params)
    
    # Run Variable Step AEGIS Propagation    
    int_params['integrator'] = 'dopri87_aegis'
    int_params['intfcn'] = dyn.ode_aegis
    int_params['A_fcn'] = dyn.A_twobody
    int_params['dyn_fcn'] = dyn.ode_twobody_ukf
    int_params['step'] = 10.
    start = time.time()
    aegis_final3 = est.aegis_predictor(GMM_dict, tin, filter_params, 
                                       state_params, int_params)
    
    aegis3_run_time = time.time() - start
    
    aegis_points3 = est.gmm_samples(aegis_final3, N)
    
    print('aegis3 run time', aegis3_run_time)
    
    print('')
    print('state_params', state_params)
    
    
    # Variable Step AEGIS Propagation with JIT
    int_params['integrator'] = 'dopri87_aegis_jit'
    int_params['intfcn'] = fastint.jit_twobody_aegis
    
    start = time.time()
    aegis_final4 = est.aegis_predictor(GMM_dict, tin, filter_params, 
                                        state_params, int_params)
    
    aegis4_run_time = time.time() - start
    
    aegis_points4 = est.gmm_samples(aegis_final4, N)
    
    print('aegis4 run time', aegis4_run_time)
    
    
    # Run UKF Propagation
    int_params['integrator'] = 'dopri87_aegis'
    int_params['intfcn'] = dyn.ode_aegis
    int_params['A_fcn'] = dyn.A_twobody
    int_params['dyn_fcn'] = dyn.ode_twobody_ukf
    filter_params['split_T'] = 1e6
    
    start = time.time()
    ukf_final = est.aegis_predictor(GMM_dict, tin, filter_params,
                                    state_params, int_params)
    ukf_points = est.gmm_samples(ukf_final, N)
    
    ukf_run_time = time.time() - start
    
    print('ukf run time', ukf_run_time)
    
    
    
    # Monte-Carlo Propagation
    # Generate samples
    mc_init = np.random.multivariate_normal(X_true.flatten(),P,int(N))
    mc_final = np.zeros(mc_init.shape)
    
    # Convert time to seconds
    time_format = int_params['time_format']
    if time_format == 'datetime':
        t0 = tk_list[0]
        tvec = np.asarray([(ti - t0).total_seconds() for ti in tk_list])
        
    params2 = Dict.empty(key_type=types.unicode_type, value_type=types.float64,)
    params2['step'] = 10.
    params2['GM'] = GME
    params2['rtol'] = 1e-12
    params2['atol'] = 1e-12
    
    # Propagate samples
    start = time.time()
    intfcn = fastint.jit_twobody
    for jj in range(mc_init.shape[0]):
        
        if jj % 100 == 0:
            print(jj)
            print('elapsed time', time.time() - start)
        
        int0 = mc_init[jj].flatten()
        tout, Xout = fastint.dopri87(intfcn, tvec, int0.flatten(), params2)
        mc_final[jj,:] = Xout[-1,:].flatten()
        
    mc_run_time = time.time() - start
    
    
    
#    # Likelihood Agreement Measure
#    aegis_lam2 = analysis.compute_LAM(aegis_final2, mc_final)
#    aegis_lam3 = analysis.compute_LAM(aegis_final3, mc_final)
#    aegis_lam4 = analysis.compute_LAM(aegis_final4, mc_final)
#    ukf_lam = analysis.compute_LAM(ukf_final, mc_final)
#    
#    print('AEGIS For Loop LAM: ', aegis_lam2)
#    print('AEGIS Variable LAM: ', aegis_lam3)
#    print('AEGIS JIT LAM: ', aegis_lam4)
#    print('UKF LAM: ', ukf_lam)
    
#    print('AEGIS For Loop Comps: ', len(aegis_final2['weights']))
    print('AEGIS Variable Comps: ', len(aegis_final3['weights']))
    print('AEGIS JIT Comps: ', len(aegis_final4['weights']))
    
#    print('AEGIS For Loop Time: ', aegis2_run_time)
    print('AEGIS Variable time: ', aegis3_run_time)
    print('AEGIS JIT time: ', aegis4_run_time)
    print('UKF time: ', ukf_run_time)
    print('MC time: ', mc_run_time)
    
    
    # Generate plots
    plt.figure()
    plt.plot(mc_final[:,0], mc_final[:,1], 'k.', alpha=0.2)    
    analysis.plot_pdf_contours(aegis_final4, axis1=0, axis2=1)
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('AEGIS Contours vs MC Points')
    
    plt.figure()
    plt.plot(mc_final[:,0], mc_final[:,1], 'k.', alpha=0.2)    
    analysis.plot_pdf_contours(ukf_final, axis1=0, axis2=1)
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
#    plt.title('UKF Contours vs MC Points')
    
    
    plt.figure()
    plt.plot(mc_final[:,0], mc_final[:,1], 'k.', alpha=0.2)
    plt.plot(aegis_points4[:,0], aegis_points4[:,1], 'r.', alpha=0.2)
    plt.plot(ukf_points[:,0], ukf_points[:,1], 'b.', alpha=0.2)
    plt.legend(['MC', 'AEGIS', 'UKF'])
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('MC Points vs AEGIS and UKF Samples')
    
    plt.figure()
    plt.plot(mc_final[:,0], mc_final[:,2], 'k.', alpha=0.2)
    plt.plot(aegis_points4[:,0], aegis_points4[:,2], 'r.', alpha=0.2)
    plt.plot(ukf_points[:,0], ukf_points[:,2], 'b.', alpha=0.2)
    plt.legend(['MC', 'AEGIS', 'UKF'])
    plt.xlabel('X [km]')
    plt.ylabel('Z [km]')
    plt.title('MC Points vs AEGIS and UKF Samples')
    
    plt.figure()
    plt.plot(mc_final[:,1], mc_final[:,2], 'k.', alpha=0.2)
    plt.plot(aegis_points4[:,1], aegis_points4[:,2], 'r.', alpha=0.2)
    plt.plot(ukf_points[:,1], ukf_points[:,2], 'b.', alpha=0.2)
    plt.legend(['MC', 'AEGIS', 'UKF'])
    plt.xlabel('Y [km]')
    plt.ylabel('Z [km]')
    plt.title('MC Points vs AEGIS and UKF Samples')
    
    plt.show()
    
    outfile = os.path.join('advanced_test', 'twobody_heo_aegis.pkl')
    pklFile = open( outfile, 'wb' )
    pickle.dump( [aegis_final3, aegis_final4, ukf_final, mc_final], pklFile, -1 )
    pklFile.close()
    
    
    
    
    return


def demars_high_orbit():
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME

    # Define filter parameters
    filter_params = {}
    filter_params['Q'] = 1e-16 * np.diag([1, 1, 1])
    filter_params['gap_seconds'] = 900.
    filter_params['alpha'] = 1e-4
    filter_params['delta_s_sec'] = 600.
    filter_params['split_T'] = 0.1
    
    # Integration function and additional settings
    int_params = {}
    int_params['integrator'] = 'dopri87_aegis_jit'
    int_params['intfcn'] = fastint.jit_twobody_aegis
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['step'] = 10.
    int_params['time_format'] = 'datetime'

    # Inital State
    elem = [35000., 0.2, 0., 0., 0., 0.]
    X_true = np.reshape(astro.kep2cart(elem), (6,1))
    P = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    
    # Time vector
    period = 2.*np.pi*np.sqrt(elem[0]**3/GME)
    UTC0 = datetime(2021, 6, 21, 0, 0, 0)
    UTC1 = UTC0 + timedelta(seconds=2.*period)
#    UTC1 = datetime(2021, 6, 22, 0, 0, 0)
    tk_list = [UTC0, UTC1]
    tin = tk_list
    print(tk_list)
    
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
    
    
    
    # Run AEGIS Propagation
    start = time.time()
    aegis_final = est.aegis_predictor(GMM_dict, tin, filter_params, 
                                        state_params, int_params)
    
    aegis_run_time = time.time() - start
    
    
    # Run UKF Propagation
    filter_params['split_T'] = 1e6
    
    start = time.time()
    ukf_final = est.aegis_predictor(GMM_dict, tin, filter_params, 
                                     state_params, int_params)

    ukf_run_time = time.time() - start
    
    print('ukf run time', ukf_run_time)
    
    
    
    # Monte-Carlo Propagation
    # Generate samples
    N = 1000
    mc_init = np.random.multivariate_normal(X_true.flatten(),P,int(N))
    mc_final = np.zeros(mc_init.shape)
    
    # Convert time to seconds
    time_format = int_params['time_format']
    if time_format == 'datetime':
        t0 = tk_list[0]
        tvec = np.asarray([(ti - t0).total_seconds() for ti in tk_list])
        
    params2 = Dict.empty(key_type=types.unicode_type, value_type=types.float64,)
    params2['step'] = 10.
    params2['GM'] = GME
    params2['rtol'] = 1e-12
    params2['atol'] = 1e-12
    
    # Propagate samples
    start = time.time()
    intfcn = fastint.jit_twobody
    for jj in range(mc_init.shape[0]):
        
        if jj % 100 == 0:
            print(jj)
            print('elapsed time', time.time() - start)
        
        int0 = mc_init[jj].flatten()
        tout, Xout = fastint.dopri87(intfcn, tvec, int0.flatten(), params2)
        mc_final[jj,:] = Xout[-1,:].flatten()
        
    mc_run_time = time.time() - start
    
    # Likelihood Agreement Measure
    aegis_lam = analysis.compute_LAM(aegis_final, mc_final)
    ukf_lam = analysis.compute_LAM(ukf_final, mc_final)
    
    print('AEGIS JIT LAM: ', aegis_lam)
    print('UKF LAM: ', ukf_lam)   
    print('Normalized LAM', ukf_lam/aegis_lam)
    print('AEGIS JIT Comps: ', len(aegis_final['weights']))
    print('AEGIS JIT time: ', aegis_run_time)
    print('UKF time: ', ukf_run_time)
    print('MC time: ', mc_run_time)
    
    # Sample points
    ukf_points = est.gmm_samples(ukf_final, N)
    aegis_points = est.gmm_samples(aegis_final, N)
    
    
    
    # Generate plots
    plt.figure()
    plt.plot(mc_final[:,0], mc_final[:,1], 'k.', alpha=0.2)    
    analysis.plot_pdf_contours(aegis_final, axis1=0, axis2=1)
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('AEGIS Contours vs MC Points')
    
    plt.figure()
    plt.plot(mc_final[:,0], mc_final[:,1], 'k.', alpha=0.2)    
    analysis.plot_pdf_contours(ukf_final, axis1=0, axis2=1)
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('UKF Contours vs MC Points')
    
    
    plt.figure()
    plt.plot(mc_final[:,0], mc_final[:,1], 'k.', alpha=0.2)
    plt.plot(aegis_points[:,0], aegis_points[:,1], 'r.', alpha=0.2)
    plt.plot(ukf_points[:,0], ukf_points[:,1], 'b.', alpha=0.2)
    plt.legend(['MC', 'AEGIS', 'UKF'])
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('MC Points vs AEGIS and UKF Samples')
    
    plt.figure()
    plt.plot(mc_final[:,0], mc_final[:,2], 'k.', alpha=0.2)
    plt.plot(aegis_points[:,0], aegis_points[:,2], 'r.', alpha=0.2)
    plt.plot(ukf_points[:,0], ukf_points[:,2], 'b.', alpha=0.2)
    plt.legend(['MC', 'AEGIS', 'UKF'])
    plt.xlabel('X [km]')
    plt.ylabel('Z [km]')
    plt.title('MC Points vs AEGIS and UKF Samples')
    
    plt.figure()
    plt.plot(mc_final[:,1], mc_final[:,2], 'k.', alpha=0.2)
    plt.plot(aegis_points[:,1], aegis_points[:,2], 'r.', alpha=0.2)
    plt.plot(ukf_points[:,1], ukf_points[:,2], 'b.', alpha=0.2)
    plt.legend(['MC', 'AEGIS', 'UKF'])
    plt.xlabel('Y [km]')
    plt.ylabel('Z [km]')
    plt.title('MC Points vs AEGIS and UKF Samples')
    
    plt.show()
    
    outfile = os.path.join('advanced_test', 'demars_high_orbit.pkl')
    pklFile = open( outfile, 'wb' )
    pickle.dump( [aegis_final, ukf_final, mc_final], pklFile, -1 )
    pklFile.close()
    
    
    
    
    return


def pdf_contours():
    
    outfile = os.path.join('advanced_test', 'demars_high_orbit.pkl')
    
    pklFile = open(outfile, 'rb' )
    data = pickle.load( pklFile )
    aegis_final = data[0]
    ukf_final = data[1]
    mc_final = data[2]
    pklFile.close()
    
    
    # Generate plots
    plt.figure()
    plt.plot(mc_final[:,0], mc_final[:,1], 'k.', alpha=0.2)    
    analysis.plot_pdf_contours(aegis_final, axis1=0, axis2=1)
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    
    plt.figure()
    plt.plot(mc_final[:,0], mc_final[:,1], 'k.', alpha=0.2)    
    analysis.plot_pdf_contours(ukf_final, axis1=0, axis2=1)
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    
    plt.show()
    
    
    return


if __name__ == '__main__':
    
    
    plt.close('all')    
    
#    twobody_geo_aegis_prop()
    
#    twobody_heo_aegis_prop()
    
    demars_high_orbit()
    
#    pdf_contours()

















