import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
import pickle
import os
import inspect
import random
import scipy.stats as ss

# Load tudatpy modules  
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from estimation import analysis_functions as analysis
from estimation import estimation_functions as est
from estimation import multitarget_functions as mult
from dynamics import dynamics_functions as dyn
from sensors import measurement_functions as mfunc
from sensors import sensors as sens
from sensors import visibility_functions as visfunc
from utilities import astrodynamics as astro
from utilities import coordinate_systems as coord
from utilities import eop_functions as eop
from utilities import tle_functions as tle
from utilities.constants import GME, arcsec2rad









def unit_test_auction():
    '''
    Example assignment problem from [1] Blackman and Popoli 
    
    '''
    
    # C is cost matrix to minimize
    C = np.array([[10.,    5.,   8.,   9.],
                  [7.,   100.,  20., 100.],
                  [100.,  21., 100., 100.],
                  [100.,  15.,  17., 100.],
                  [100., 100.,  16.,  22.]])
    
    # A is score matrix to maximize
    A = 100.*np.ones((5,4)) - C
    
    # Compute assignment
    row_index, score, eps = auction(A)
    
    print(row_index, score, eps)
    
    truth = [7., 15., 16., 9.]
    test_sum = 0.
    for ii in range(4):
        print(C[row_index[ii],ii])
        test_sum += C[row_index[ii],ii] - truth[ii]
        
    if test_sum == 0.:
        print('pass')
    
    
    
    return


def unit_test_murty():
    '''
    Example assignment problem from [1] Blackman and Popoli
    '''
    
    # C is cost matrix to minimize
    C = np.array([[10.,    5.,   8.,   9.],
                  [7.,   100.,  20., 100.],
                  [100.,  21., 100., 100.],
                  [100.,  15.,  17., 100.],
                  [100., 100.,  16.,  22.]])
    
    # A is score matrix to maximize
    A = 100.*np.ones((5,4)) - C
    
    # Compute assignment
    kbest = 4
    final_list = murty(A, kbest)
    
    print(final_list)
    
    for row_index in final_list:
        for ii in range(4):
            print(C[row_index[ii], ii])
    
    
    
    return


def vo_2d_motion_setup():
    
    
    
    # Process Noise and State Covariance
    sig_w = 5.                 # m/s^2
    sig_u = 1.*np.pi/180.     # rad/s
    
    G = np.zeros((4,2))
    G[0,0] = 0.5
    G[1,0] = 1.
    G[2,1] = 0.5
    G[3,1] = 1.
    
    Q = np.zeros((5,5))
    Q[0:4,0:4] = sig_w**2.*np.dot(G, G.T)
    Q[4,4] = sig_u**2.
    
    P_birth = np.diag([50.**2., 50.**2., 50.**2., 50.**2., (6.*np.pi/180.)**2.])
    
    
    # Birth model
    birth_model = {}
    birth_model[1] = {}
    birth_model[1]['r_birth'] = 0.02
    birth_model[1]['weights'] = [1.]
    birth_model[1]['means'] = [np.reshape([-1500., 0., 250., 0., 0.], (5,1))]
    birth_model[1]['covars'] = [P_birth]
    birth_model[2] = {}
    birth_model[2]['r_birth'] = 0.02
    birth_model[2]['weights'] = [1.]
    birth_model[2]['means'] = [np.reshape([-250., 0., 1000., 0., 0.], (5,1))]
    birth_model[2]['covars'] = [P_birth]
    birth_model[3] = {}
    birth_model[3]['r_birth'] = 0.03
    birth_model[3]['weights'] = [1.]
    birth_model[3]['means'] = [np.reshape([250., 0., 750., 0., 0.], (5,1))]
    birth_model[3]['covars'] = [P_birth]
    birth_model[4] = {}
    birth_model[4]['r_birth'] = 0.03
    birth_model[4]['weights'] = [1.]
    birth_model[4]['means'] = [np.reshape([1000., 0., 1500., 0., 0.], (5,1))]
    birth_model[4]['covars'] = [P_birth]
    
    
    
    
    # Define state parameters
    wturn = 2.*np.pi/180.
    state_params = {}
    state_params['nstates'] = 5
    
    
    
    
    # Filter parameters
    filter_params = {}
    filter_params['Q'] = Q
    filter_params['snc_flag'] = 'qfull'
    filter_params['alpha'] = 1.
    filter_params['pnorm'] = 2.
    filter_params['prune_T'] = 1e-5
    filter_params['merge_U'] = 4.
    filter_params['p_surv'] = 0.99
    filter_params['p_det'] = 0.98
    filter_params['birth_model'] = birth_model
    
    # Integration function and additional settings    
    int_params = {}
    int_params['integrator'] = 'rk4'
    int_params['intfcn'] = dyn.ode_coordturn
    int_params['step'] = 0.1
    int_params['time_format'] = 'seconds'
    
    
    # Sensor parameters
    sensor_params = {}
    sensor_id = 1
    sensor_params[sensor_id] = {}
    sensor_params[sensor_id]['r_site'] = np.reshape([0., 0.], (2,1))
    sensor_params[sensor_id]['meas_types'] = ['az', 'rg']
    sensor_params[sensor_id]['sigma_dict'] = {}
    sensor_params[sensor_id]['sigma_dict']['az'] = (2.*np.pi/180.)
    sensor_params[sensor_id]['sigma_dict']['rg'] = 10.
    sensor_params[sensor_id]['lam_clutter'] = 15.
    sensor_params[sensor_id]['az_lim'] = [0., np.pi]
    sensor_params[sensor_id]['rg_lim'] = [0., 2000.]
    sensor_params[sensor_id]['V_sensor'] = np.pi*2000.   # rad*m
    

    # Time vector
    tk_list = list(range(1,101))
    
    # Initial state vectors
    object_dict = {}
    object_dict[1] = {}
    object_dict[1]['Xo'] = np.reshape([1000+3.8676, -10., 1500-11.7457, -10., wturn/8.], (5,1))
    object_dict[1]['birth_time'] = 1.
    object_dict[1]['death_time'] = 101.
    object_dict[2] = {}
    object_dict[2]['Xo'] = np.reshape([-250-5.8857,  20., 1000+11.4102, 3., -wturn/3.], (5,1))
    object_dict[2]['birth_time'] = 10.
    object_dict[2]['death_time'] = 101.
    object_dict[3] = {}
    object_dict[3]['Xo'] = np.reshape([-1500-7.3806, 11., 250+6.7993, 10., -wturn/2.], (5,1))
    object_dict[3]['birth_time'] = 10.
    object_dict[3]['death_time'] = 101.
    object_dict[4] = {}
    object_dict[4]['Xo'] = np.reshape([-1500., 43., 250., 0., 0.], (5,1))
    object_dict[4]['birth_time'] = 10.
    object_dict[4]['death_time'] = 66.
    object_dict[5] = {}
    object_dict[5]['Xo'] = np.reshape([250-3.8676, 11., 750-11.0747, 5., wturn/4.], (5,1))
    object_dict[5]['birth_time'] = 20.
    object_dict[5]['death_time'] = 80.
    object_dict[6] = {}
    object_dict[6]['Xo'] = np.reshape([-250+7.3806, -12., 1000-6.7993, -12., wturn/2.], (5,1))
    object_dict[6]['birth_time'] = 40.
    object_dict[6]['death_time'] = 101.
    object_dict[7] = {}
    object_dict[7]['Xo'] = np.reshape([1000., 0., 1500., -10., wturn/4.], (5,1))
    object_dict[7]['birth_time'] = 40.
    object_dict[7]['death_time'] = 101.
    object_dict[8] = {}
    object_dict[8]['Xo'] = np.reshape([250., -50., 750., 0., -wturn/4.], (5,1))
    object_dict[8]['birth_time'] = 40.
    object_dict[8]['death_time'] = 80.
    object_dict[9] = {}
    object_dict[9]['Xo'] = np.reshape([1000., -50., 1500., 0., -wturn/4.], (5,1))
    object_dict[9]['birth_time'] = 60.
    object_dict[9]['death_time'] = 101.
    object_dict[10] = {}
    object_dict[10]['Xo'] = np.reshape([250., -40., 750., 25., wturn/4.], (5,1))
    object_dict[10]['birth_time'] = 60.
    object_dict[10]['death_time'] = 101.

    # Generate truth and meas data
    truth_dict = {}
    meas_dict = {}
    sensor_id = 1
    for kk in range(len(tk_list)):
        
        tk = tk_list[kk]
        if kk > 0:
            tk_prior = tk_list[kk-1]
        
        truth_dict[tk] = {}
        Zk_list = []
        sensor_kk_list = []
        sensor = sensor_params[sensor_id]
        for obj_id in object_dict:
            
            if tk == object_dict[obj_id]['birth_time']:
                truth_dict[tk][obj_id] = object_dict[obj_id]['Xo']                                
            
            if tk > object_dict[obj_id]['birth_time'] and tk <= object_dict[obj_id]['death_time']:
                Xo = truth_dict[tk_prior][obj_id]
                tin = [0., (tk - tk_prior)]
                tout, Xout = dyn.general_dynamics(Xo, tin, state_params, int_params)
                X = Xout[-1,:].reshape(5,1)
                truth_dict[tk][obj_id] = X
                
            
            # If object exists at this time, compute measurement
            if obj_id in truth_dict[tk]:
                Xk = truth_dict[tk][obj_id]
                r_site = sensor_params[sensor_id]['r_site']
                xy = np.reshape([Xk[0], Xk[2]], (2,1))
                rho_vect = xy - r_site
                az = math.atan2(rho_vect[1], rho_vect[0])
                rg = np.linalg.norm(rho_vect)
                
                # Incorporate missed detections
                if np.random.rand() <= filter_params['p_det']:
                    
                    sigma_dict = sensor_params[sensor_id]['sigma_dict']
                    az += np.random.randn()*sigma_dict['az']
                    rg += np.random.randn()*sigma_dict['rg']
                    zi = np.reshape([az, rg], (2,1))
                    Zk_list.append(zi)
                    sensor_kk_list.append(sensor_id)
                    
            
        # Incorporate clutter
        n_clutter = ss.poisson.rvs(sensor['lam_clutter'])

        # Compute clutter meas in RA/DEC, uniform over FOV
        for c_ind in range(n_clutter):
            az_lim = sensor['az_lim']
            rg_lim = sensor['rg_lim']
            az = (az_lim[1]-az_lim[0])*np.random.rand() + az_lim[0]
            rg = (rg_lim[1]-rg_lim[0])*np.random.rand() + rg_lim[0]

            zclutter = np.reshape([az, rg], (2,1))
            Zk_list.append(zclutter)
            sensor_kk_list.append(sensor_id)
        
        # Shuffle order of measurements
        if len(Zk_list) > 0:
            
            inds = list(range(len(Zk_list)))
            random.shuffle(inds)
            
            meas_dict[tk] = {}
            meas_dict[tk]['Zk_list'] = [Zk_list[ii] for ii in inds]
            meas_dict[tk]['sensor_id_list'] = [sensor_kk_list[ii] for ii in inds]
                
    
    # Plot truth data
    plot_dict = {}
    for obj_id in object_dict:
        plot_dict[obj_id] = {}
        plot_dict[obj_id]['tk_list'] = []
        plot_dict[obj_id]['x_list'] = []
        plot_dict[obj_id]['y_list'] = []
        
        for tk in tk_list:
            if obj_id in truth_dict[tk]:
                plot_dict[obj_id]['tk_list'].append(tk)
                plot_dict[obj_id]['x_list'].append(truth_dict[tk][obj_id][0])
                plot_dict[obj_id]['y_list'].append(truth_dict[tk][obj_id][2])
                
    fig1 = plt.figure()
    fig2 = plt.figure()
    # color_list = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    cm = plt.get_cmap('hsv')
    num_colors = len(object_dict)
    ii = 0
    for obj_id in object_dict:
        tk_plot = plot_dict[obj_id]['tk_list']
        x_plot = plot_dict[obj_id]['x_list']
        y_plot = plot_dict[obj_id]['y_list']
        
        color_ii = cm(1*ii/num_colors)
        
        plt.figure(fig1)
        plt.subplot(2,1,1)
        plt.plot(tk_plot, x_plot, '-', color=color_ii)
        plt.ylabel('x [m]')
        plt.subplot(2,1,2)
        plt.plot(tk_plot, y_plot, '-', color=color_ii)
        plt.ylabel('y [m]')
        plt.xlabel('Time [sec]')
        
        plt.figure(fig2)
        plt.plot(x_plot, y_plot, '-', color=color_ii)
        plt.plot(x_plot[0], y_plot[0], 'o', color=color_ii)
        plt.plot(x_plot[-1], y_plot[-1], 'x', color=color_ii)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.xlim([-2000., 2000.])
        plt.ylim([0., 2000.])
        
        ii += 1
        
    plt.figure(fig1)
    for tk in meas_dict:
        
        print('tk', tk)        
        Zk_list = meas_dict[tk]['Zk_list']
        
        print('nmeas', len(Zk_list))
        
        for zi in Zk_list:
            az = float(zi[0])
            rg = float(zi[1])
            
            x = rg*np.cos(az)
            y = rg*np.sin(az)
            
            plt.subplot(2,1,1)
            plt.plot(tk, x, 'kx', alpha=0.5, ms=3)
            plt.subplot(2,1,2)
            plt.plot(tk, y, 'kx', alpha=0.5, ms=3)
        
        
    plt.show()
    
    

    
    return



def tudat_geo_2obj_setup(setup_file):
    
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Define state parameters
    state_params = {}
    state_params['GM'] = GME
    state_params['radius_m'] = 1.
    state_params['albedo'] = 0.1
    state_params['bodies_to_create'] = ['Earth']
    state_params['global_frame_origin'] = 'Earth'
    state_params['global_frame_orientation'] = 'J2000'
    state_params['central_bodies'] = ['Earth']
    state_params['sph_deg'] = 0
    state_params['sph_ord'] = 0
    state_params['mass'] = 400.
    state_params['Cd'] = 0.
    state_params['Cr'] = 0.
    state_params['drag_area_m2'] = 4.
    state_params['srp_area_m2'] = 4.
    
    # Filter parameters
    filter_params = {}
    filter_params['Q'] = 1e-16 * np.diag([1, 1, 1])
    filter_params['gap_seconds'] = 900.
    filter_params['alpha'] = 1e-4
    filter_params['pnorm'] = 2.
    filter_params['prune_T'] = 1e-3
    filter_params['merge_U'] = 36.
    filter_params['p_surv'] = 1.
    filter_params['p_det'] = 0.9
    
    # Integration function and additional settings    
    int_params = {}
    int_params['integrator'] = 'tudat'
    int_params['tudat_integrator'] = 'rkf78'
    int_params['step'] = 10.
    int_params['max_step'] = 1000.
    int_params['min_step'] = 1.
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    int_params['time_format'] = 'datetime'


    # Time vector
    tk_list = []
    for hr in [9, 10, 11, 12, 13, 14, 15]:
        UTC = datetime(2021, 6, 21, hr, 0, 0)
        tvec = np.arange(0., 601., 60.)
        tk_list.extend([UTC + timedelta(seconds=ti) for ti in tvec])

    # Inital State
    elem1 = [42164.1, 0.001, 0.1, 225., 0., 0.]
    X1_true = np.reshape(astro.kep2cart(elem1), (6,1))
    P1 = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    pert_vect1 = np.multiply(np.sqrt(np.diag(P1)), np.random.randn(6))
    X1_init = X1_true + np.reshape(pert_vect1, (6, 1))
    
    elem2 = [42164.1, 0.001, 0.1, 225., 0., 1.]
    X2_true = np.reshape(astro.kep2cart(elem2), (6,1))
    P2 = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
    pert_vect2 = np.multiply(np.sqrt(np.diag(P2)), np.random.randn(6))
    X2_init = X2_true + np.reshape(pert_vect2, (6, 1))
    
    
    state_dict = {}
    state_dict[tk_list[0]] = {}
    state_dict[tk_list[0]]['weights'] = [1., 1.]
    state_dict[tk_list[0]]['means'] = [X1_init, X2_init]
    state_dict[tk_list[0]]['covars'] = [P1, P2]
    
    
    # Sensor and measurement parameters
    sensor_id_list = ['RMIT ROO']
    sensor_params = sens.define_sensors(sensor_id_list)
    sensor_params['eop_alldata'] = eop_alldata
    sensor_params['XYs_df'] = XYs_df
    
    for sensor_id in sensor_id_list:
        sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
        sigma_dict = {}
        sigma_dict['ra'] = 5.*arcsec2rad   # rad
        sigma_dict['dec'] = 5.*arcsec2rad  # rad
        sensor_params[sensor_id]['sigma_dict'] = sigma_dict
        sensor_params[sensor_id]['lam_clutter'] = 5.
        FOV_hlim = sensor_params[sensor_id]['FOV_hlim']
        FOV_vlim = sensor_params[sensor_id]['FOV_vlim']        
        sensor_params[sensor_id]['V_sensor'] = (FOV_hlim[1] - FOV_hlim[0])*(FOV_vlim[1] - FOV_vlim[0])

        
#    print(sensor_params)
    
#    for sensor_id in sensor_id_list:
#        sensor_params[sensor_id]['meas_types'] = ['rg', 'ra', 'dec']
#        sigma_dict = {}
#        sigma_dict['rg'] = 0.001  # km
#        sigma_dict['ra'] = 5.*arcsec2rad   # rad
#        sigma_dict['dec'] = 5.*arcsec2rad  # rad
#        sensor_params[sensor_id]['sigma_dict'] = sigma_dict
#    print(sensor_params)

    # Generate truth and measurements
    truth_dict = {}
    meas_fcn = mfunc.unscented_radec
    meas_dict = {}
    X = np.concatenate((X1_true, X2_true), axis=0)
    for kk in range(len(tk_list)):
        
        if kk > 0:
            tin = [tk_list[kk-1], tk_list[kk]]
            tout, Xout = dyn.general_dynamics(X, tin, state_params, int_params)
            X = Xout[-1,:]
        
        X1_t = X[0:6].reshape(6,1)
        X2_t = X[6:12].reshape(6,1)
        truth_dict[tk_list[kk]] = {}
        truth_dict[tk_list[kk]]['Xt_list'] = [X1_t, X2_t]
        
        # Check visibility conditions and compute measurements
        UTC = tk_list[kk]
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        Zk_list = []
        sensor_kk_list = []
        for sensor_id in sensor_id_list:
            
            sensor = sensor_params[sensor_id]
            p_det = filter_params['p_det']
            center_flag = True            
            for Xj in truth_dict[tk_list[kk]]['Xt_list']:            
                          
                if visfunc.check_visibility(Xj, state_params, sensor_params,
                                            sensor_id, UTC, EOP_data, XYs_df):
                    
                    # Incorporate missed detection
                    if np.random.rand() > p_det:
                        continue
                    
                    # Compute measurements
                    zj = mfunc.compute_measurement(Xj, state_params, sensor_params,
                                                   sensor_id, UTC, EOP_data, XYs_df,
                                                   meas_types=sensor['meas_types'])
                    
                    # Store first measurement for each sensor as FOV center
                    if center_flag:
                        center = zj.copy()
                        center_flag = False
                    
                    # Add noise and store measurement data
                    zj[0] += np.random.randn()*sigma_dict['ra']
                    zj[1] += np.random.randn()*sigma_dict['dec']
                    
                    Zk_list.append(zj)
                    sensor_kk_list.append(sensor_id)
            
            # Incorporate clutter measurements
            n_clutter = ss.poisson.rvs(sensor['lam_clutter'])

            # Compute clutter meas in RA/DEC, uniform over FOV
            for c_ind in range(n_clutter):
                FOV_hlim = sensor['FOV_hlim']
                FOV_vlim = sensor['FOV_vlim']
                ra  = center[0] + (FOV_hlim[1]-FOV_hlim[0])*(np.random.rand() - 0.5)
                dec = center[1] + (FOV_vlim[1]-FOV_vlim[0])*(np.random.rand() - 0.5)

                zclutter = np.reshape([ra, dec], (2,1))
                Zk_list.append(zclutter)
                sensor_kk_list.append(sensor_id)

        # If measurements were collected, randomize order and store
        if len(Zk_list) > 0:
            
            inds = list(range(len(Zk_list)))
            random.shuffle(inds)
            
            meas_dict[UTC] = {}
            meas_dict[UTC]['Zk_list'] = [Zk_list[ii] for ii in inds]
            meas_dict[UTC]['sensor_id_list'] = [sensor_kk_list[ii] for ii in inds]
                

    # Plot data
    tplot = [(tk - tk_list[0]).total_seconds()/3600. for tk in tk_list]
    xplot1 = []
    yplot1 = []
    zplot1 = []
    xplot2 = []
    yplot2 = []
    zplot2 = []
    for tk in tk_list:
        Xt_list = truth_dict[tk]['Xt_list']
        X1_t = Xt_list[0]
        X2_t = Xt_list[1]
        
        xplot1.append(X1_t[0])
        yplot1.append(X1_t[1])
        zplot1.append(X1_t[2])
        
        xplot2.append(X2_t[0])
        yplot2.append(X2_t[1])
        zplot2.append(X2_t[2])
        
    
    meas_tk = sorted(meas_dict.keys())
    meas_tplot = [(tk - tk_list[0]).total_seconds()/3600. for tk in meas_tk]
    nmeas_plot = []
    for tk in meas_tk:
            
        Zk_list = meas_dict[tk]['Zk_list']
        # sensor_id_list = meas_dict[tk]['sensor_id_list']
        # meas_sensor_id = meas_dict[tk]['sensor_id_list']
        # meas_sensor_index = [sensor_id_list.index(sensor_id) for sensor_id in meas_sensor_id]
        
        nmeas_plot.append(len(Zk_list))
    
        
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tplot, xplot1, 'b.')
    plt.plot(tplot, xplot2, 'r.')
    plt.ylabel('X [km]')
    plt.subplot(3,1,2)
    plt.plot(tplot, yplot2, 'b.')
    plt.plot(tplot, yplot2, 'r.')
    plt.ylabel('Y [km]')
    plt.subplot(3,1,3)
    plt.plot(tplot, zplot1, 'b.')
    plt.plot(tplot, zplot2, 'r.')
    plt.ylabel('Z [km]')
    plt.xlabel('Time [hours]')
    
    # plt.figure()
    # plt.plot(meas_tplot, meas_sensor_index, 'k.')
    # plt.xlabel('Time [hours]')
    # plt.yticks([0], ['UNSW Falcon'])
    # plt.ylabel('Sensor ID')
    
    plt.figure()
    plt.plot(meas_tplot, nmeas_plot, 'k.')
    plt.ylabel('Number of Meas')
    plt.xlabel('Time [hours]')
    
                
    plt.show()   
    
    # print(meas_dict)
    
    params_dict = {}
    params_dict['state_params'] = state_params
    params_dict['filter_params'] = filter_params
    params_dict['int_params'] = int_params
    params_dict['sensor_params'] = sensor_params
                
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [state_dict, meas_fcn, meas_dict, params_dict, truth_dict], pklFile, -1 )
    pklFile.close()
                
    
    
    return


def run_multitarget_filter(setup_file, results_file):
    
    
    pklFile = open(setup_file, 'rb' )
    data = pickle.load( pklFile )
    state_dict = data[0]
    meas_fcn = data[1]
    meas_dict = data[2]
    params_dict = data[3]
    truth_dict = data[4]
    pklFile.close()
    
    
    filter_output, full_state_output = mult.phd_filter(state_dict, truth_dict, meas_dict, meas_fcn, params_dict)
    
    pklFile = open( results_file, 'wb' )
    pickle.dump( [filter_output, full_state_output, params_dict, truth_dict], pklFile, -1 )
    pklFile.close()
    
    # analysis.multitarget_orbit_errors(filter_output, filter_output, truth_dict)
    
    
    return


def multitarget_analysis(results_file):
    
    pklFile = open(results_file, 'rb' )
    data = pickle.load( pklFile )
    filter_output = data[0]
    full_state_output = data[1]
    params_dict = data[2]
    truth_dict = data[3]
    pklFile.close()
    
    analysis.multitarget_orbit_errors(filter_output, filter_output, truth_dict)
    
    return


if __name__ == '__main__':
    
    plt.close('all')
    
    fdir = r'D:\documents\research_projects\multitarget\data\sim\test\2022_12_01_geo_2obj'
    
    
    setup_file = os.path.join(fdir, 'tudat_geo_twobody_2obj_pd09_lam0_setup.pkl')
    results_file = os.path.join(fdir, 'tudat_geo_twobody_2obj_pd09_lam0_phd_results.pkl')
    
    
    # tudat_geo_2obj_setup(setup_file)    
    
    # run_multitarget_filter(setup_file, results_file)
    
    # multitarget_analysis(results_file)
    
    
    vo_2d_motion_setup()
    
    
    
#    unit_test_auction()
    
    
    # unit_test_murty()
    
    
    
    

#    # A is score matrix to maximize
#    A = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                  [2, 10, 3, 6, 2, 12, 6, 9, 6, 10],
#                  [3, 11, 1, 9, 4, 15, 5, 4, 9, 12],
#                  [4, 6, 5, 4, 0, 3, 4, 6, 10, 11],
#                  [5, 0, 6, 8, 1, 10, 3, 7, 8, 13],
#                  [6, 11, 0, 6, 5, 9, 2, 5, 3, 8],
#                  [7, 9, 2, 5, 6, 5, 1, 3, 6, 6],
#                  [8, 8, 6, 9, 4, 0, 8, 2, 1, 5],
#                  [10, 12, 11, 6, 5, 10, 9, 1, 6, 7],
#                  [9, 10, 4, 8, 0, 9, 1, 0, 5, 9]])
#
#    
#    # Compute assignment
#    row_index, score, eps = auction(A)
#    
#    print(row_index, score, eps)
#
#    for ii in range(10):
#        print(A[row_index[ii],ii])
#    