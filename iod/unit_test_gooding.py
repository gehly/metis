import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
import pickle
import csv
import os
import inspect
import copy
import time
# from astroquery.jplhorizons import Horizons
import cProfile

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

import dynamics.dynamics_functions as dyn
import estimation.analysis_functions as analysis
import estimation.estimation_functions as est
# import iod.iod_functions_jit as iod
import iod.iod_functions as iod
import sensors.measurement_functions as mfunc
import sensors.sensors as sens
import sensors.visibility_functions as visfunc
import utilities.astrodynamics as astro
import utilities.coordinate_systems as coord
import utilities.eop_functions as eop
import utilities.time_systems as timesys
from utilities import tle_functions as tle

from utilities.constants import GME, Re, arcsec2rad





def linh_test_case():
    
    # Generic GEO Orbit
    # elem = [42164.1, 0.01, 0.1, 90., 1., 1.]
    # Xo = np.reshape(astro.kep2cart(elem), (6,1))
    Xo = np.reshape([-4.016332655807315E4, -1.2814784554277074E4, 
                     120.47359695689153, 0.9342751231187755, 
                     -2.9297861717915916, -0.004559887207651069], (6,1))
    print('Xo true', Xo)
    
    UTC0 = datetime(2024, 8, 1, 20, 48, 6, 526944)
    
    #2024-08-02T04:58:39.00000000001125Z"; "2024-08-02T04:58:54.00000000001125Z"; "2024-08-02T04:59:09.00000000001125Z
    
    # Time vector
    UTC1 = datetime(2024, 8, 2, 4, 58, 39)
    UTC2 = datetime(2024, 8, 2, 4, 58, 54)
    UTC3 = datetime(2024, 8, 2, 4, 59, 9)
    UTC_list = [UTC1, UTC2, UTC3]
    tk_list = UTC_list
    
    # Sensor data
    sensor_id = 'Linh Optical'
    sensor_params = sens.define_sensors([sensor_id])
    sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
    sensor = sensor_params[sensor_id]
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Compute measurements
    Yk_list = []
    sensor_id_list = []
    rho_list = []
    for UTC in UTC_list:
        
        dt_sec = (UTC - UTC0).total_seconds()
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        Xk = astro.element_conversion(Xo, 1, 1, dt=dt_sec)
        all_meas = mfunc.compute_measurement(Xk, {}, sensor_params, sensor_id, UTC, EOP_data,
                                             XYs_df, meas_types=['ra', 'dec', 'rg', 'az', 'el'])
        
        
        print(all_meas)
        
        Yk = all_meas[0:2].reshape(2,1)
        Yk_list.append(Yk)
        sensor_id_list.append(sensor_id)
        rho_list.append(float(all_meas[2]))
        
    
    
    print(Yk_list)
    print(sensor_id_list)
    print(rho_list)
    

    # Execute function
    X_list, M_list = iod.gooding_angles_iod(UTC_list, Yk_list, sensor_id_list,
                                            sensor_params, orbit_regime='GEO',
                                            search_mode='middle_out',
                                            periapsis_check=True,
                                            rootfind='zeros')
    
    
    print('Final Answers')
    print('X_list', X_list)
    print('M_list', M_list)
    
    for ii in range(len(M_list)):
        
        X_err = X_list[ii] - Xo
        print('')
        print('ii', ii)
        print('M', M_list[ii])
        print('X err', np.linalg.norm(X_err))
        
        
    
    
    ii = 0
    while ii < len(X_list):
        
        Xi = X_list[ii]
        
        print('')
        print('ii', ii)
        
        del_list = []
        for jj in range(len(X_list)):
            
            if jj == ii:
                continue
            
            Xj = X_list[jj]
            if np.linalg.norm(Xi - Xj) < 1e-3:
                del_list.append(jj)
                
        print('del_list', del_list)
        del_list = sorted(del_list, reverse=True)
        for ind in del_list:
            del X_list[ind]
        
        ii += 1
    
    print(X_list)
    
    # Check final output states and angles
    sensor_id_time_list = sensor_id_list
    tof_2 = (tk_list[1] - tk_list[0]).total_seconds()
    tof_f = (tk_list[2] - tk_list[0]).total_seconds()
    sensor0 = sensor_id_time_list[0]
    sensor2 = sensor_id_time_list[1]
    sensorf = sensor_id_time_list[2]
    EOP_data0 = eop.get_eop_data(eop_alldata, tk_list[0])
    EOP_data2 = eop.get_eop_data(eop_alldata, tk_list[1])
    EOP_dataf = eop.get_eop_data(eop_alldata, tk_list[2])
    for ii in range(len(X_list)):
        
        Xi_0 = X_list[ii]
        Xi_2 = astro.element_conversion(Xi_0, 1, 1, dt=tof_2)
        Xi_f = astro.element_conversion(Xi_0, 1, 1, dt=tof_f)
        
        meas0 = mfunc.compute_measurement(Xi_0, {}, sensor_params, sensor0, tk_list[0], EOP_data0,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        meas2 = mfunc.compute_measurement(Xi_2, {}, sensor_params, sensor2, tk_list[1], EOP_data2,
                                          XYs_df, meas_types=['ra', 'dec'])
        measf = mfunc.compute_measurement(Xi_f, {}, sensor_params, sensorf, tk_list[2], EOP_dataf,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        resids0 = (meas0 - Yk_list[0])*(1./arcsec2rad)
        resids2 = (meas2 - Yk_list[1])*(1./arcsec2rad)
        residsf = (measf - Yk_list[2])*(1./arcsec2rad)
        
        # unit vectors
        uhat_meas2 = np.array([[math.cos(meas2[1])*math.cos(meas2[0])],
                               [math.cos(meas2[1])*math.sin(meas2[0])],
                               [math.sin(meas2[1])]])
    
        uhat_yk2 = np.array([[math.cos(Yk_list[1][1])*math.cos(Yk_list[1][0])],
                             [math.cos(Yk_list[1][1])*math.sin(Yk_list[1][0])],
                             [math.sin(Yk_list[1][1])]])
        
        
        print('')
        print('Xi', Xi)
        print('elem_i', astro.cart2kep(X_list[ii]))
        print('resids0', resids0)
        print('resids2', resids2)
        print('residsf', residsf)
        
        print(uhat_meas2)
        print(uhat_yk2)
        
        print(float(np.dot(uhat_meas2.T, uhat_yk2)))
    
    
    return


def single_rev_geo():
    
    # Generic GEO Orbit
    elem = [42164.1, 0.01, 0.1, 90., 1., 1.]
    Xo = np.reshape(astro.kep2cart(elem), (6,1))
    print('Xo true', Xo)
    
    # Time vector
    UTC0 = datetime(2021, 6, 21, 0, 0, 0)
    UTC1 = datetime(2021, 6, 21, 4, 0, 0)
    UTC2 = datetime(2021, 6, 21, 6, 0, 0)
    UTC_list = [UTC0, UTC1, UTC2]
    tk_list = UTC_list
    
    # Sensor data
    sensor_id = 'UNSW Falcon'
    sensor_params = sens.define_sensors([sensor_id])
    sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
    sensor = sensor_params[sensor_id]
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Compute measurements
    Yk_list = []
    sensor_id_list = []
    rho_list = []
    for UTC in UTC_list:
        
        dt_sec = (UTC - UTC0).total_seconds()
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        Xk = astro.element_conversion(Xo, 1, 1, dt=dt_sec)
        all_meas = mfunc.compute_measurement(Xk, {}, sensor_params, sensor_id, UTC, EOP_data,
                                             XYs_df, meas_types=['ra', 'dec', 'rg', 'az', 'el'])
        
        
        print(all_meas)
        
        Yk = all_meas[0:2].reshape(2,1)
        Yk_list.append(Yk)
        sensor_id_list.append(sensor_id)
        rho_list.append(float(all_meas[2]))
        
    
    
    print(Yk_list)
    print(sensor_id_list)
    print(rho_list)
    

    # Execute function
    X_list, M_list = iod.gooding_angles_iod(UTC_list, Yk_list, sensor_id_list,
                                            sensor_params, orbit_regime='GEO',
                                            search_mode='middle_out',
                                            periapsis_check=True,
                                            rootfind='zeros')
    
    
    print('Final Answers')
    print('X_list', X_list)
    print('M_list', M_list)
    
    for ii in range(len(M_list)):
        
        X_err = X_list[ii] - Xo
        print('')
        print('ii', ii)
        print('M', M_list[ii])
        print('X err', np.linalg.norm(X_err))
        
        
    
    
    ii = 0
    while ii < len(X_list):
        
        Xi = X_list[ii]
        
        print('')
        print('ii', ii)
        
        del_list = []
        for jj in range(len(X_list)):
            
            if jj == ii:
                continue
            
            Xj = X_list[jj]
            if np.linalg.norm(Xi - Xj) < 1e-3:
                del_list.append(jj)
                
        print('del_list', del_list)
        del_list = sorted(del_list, reverse=True)
        for ind in del_list:
            del X_list[ind]
        
        ii += 1
    
    print(X_list)
    
    # Check final output states and angles
    sensor_id_time_list = sensor_id_list
    tof_2 = (tk_list[1] - tk_list[0]).total_seconds()
    tof_f = (tk_list[2] - tk_list[0]).total_seconds()
    sensor0 = sensor_id_time_list[0]
    sensor2 = sensor_id_time_list[1]
    sensorf = sensor_id_time_list[2]
    EOP_data0 = eop.get_eop_data(eop_alldata, tk_list[0])
    EOP_data2 = eop.get_eop_data(eop_alldata, tk_list[1])
    EOP_dataf = eop.get_eop_data(eop_alldata, tk_list[2])
    for ii in range(len(X_list)):
        
        Xi_0 = X_list[ii]
        Xi_2 = astro.element_conversion(Xi_0, 1, 1, dt=tof_2)
        Xi_f = astro.element_conversion(Xi_0, 1, 1, dt=tof_f)
        
        meas0 = mfunc.compute_measurement(Xi_0, {}, sensor_params, sensor0, tk_list[0], EOP_data0,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        meas2 = mfunc.compute_measurement(Xi_2, {}, sensor_params, sensor2, tk_list[1], EOP_data2,
                                          XYs_df, meas_types=['ra', 'dec'])
        measf = mfunc.compute_measurement(Xi_f, {}, sensor_params, sensorf, tk_list[2], EOP_dataf,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        resids0 = (meas0 - Yk_list[0])*(1./arcsec2rad)
        resids2 = (meas2 - Yk_list[1])*(1./arcsec2rad)
        residsf = (measf - Yk_list[2])*(1./arcsec2rad)
        
        # unit vectors
        uhat_meas2 = np.array([[math.cos(meas2[1])*math.cos(meas2[0])],
                               [math.cos(meas2[1])*math.sin(meas2[0])],
                               [math.sin(meas2[1])]])
    
        uhat_yk2 = np.array([[math.cos(Yk_list[1][1])*math.cos(Yk_list[1][0])],
                             [math.cos(Yk_list[1][1])*math.sin(Yk_list[1][0])],
                             [math.sin(Yk_list[1][1])]])
        
        
        print('')
        print('Xi', Xi)
        print('elem_i', astro.cart2kep(X_list[ii]))
        print('resids0', resids0)
        print('resids2', resids2)
        print('residsf', residsf)
        
        print(uhat_meas2)
        print(uhat_yk2)
        
        print(float(np.dot(uhat_meas2.T, uhat_yk2)))
    
    
    return


def single_rev_leo():
    
    # Generic LEO Orbit
    elem = [6878., 0.01, 28.5, 90., 10., 10.]
    Xo = np.reshape(astro.kep2cart(elem), (6,1))
    
    
    # Time vector
    UTC0 = datetime(2021, 6, 21, 0, 0, 0)
    tvec = np.arange(0, 90*60., 10.)
    UTC_list = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    
    # Sensor data
    sensor_id_list = ['UNSW Falcon', 'CMU Falcon', 'PSU Falcon']
    sensor_params = sens.define_sensors(sensor_id_list)
    
    for sensor_id in sensor_id_list:
        sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Compute measurements
    Yk_list = []
    sensor_id_time_list = []
    rho_list = []
    tk_list = []
    Xk_truth = []
    for UTC in UTC_list:
        
        dt_sec = (UTC - UTC0).total_seconds()
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        Xk = astro.element_conversion(Xo, 1, 1, dt=dt_sec)
        
        for sensor_id in sensor_id_list:
            
            sensor = sensor_params[sensor_id]
            all_meas = mfunc.compute_measurement(Xk, {}, sensor_params, sensor_id, UTC, EOP_data,
                                                 XYs_df, meas_types=['ra', 'dec', 'rg', 'az', 'el'])
            
            el = float(all_meas[4])
            if el > 0.:
                
#                print('')
#                print(UTC)
#                print(sensor_id)
#                print(all_meas)
            
                tk_list.append(UTC)
                Xk_truth.append(Xk)
                Yk = all_meas[0:2].reshape(2,1)
                Yk_list.append(Yk)
                sensor_id_time_list.append(sensor_id)
                rho_list.append(float(all_meas[2]))

                
    # Select first, middle, last entries
    mid = int(len(Yk_list)/2)
    tk_list = [tk_list[ii] for ii in [0, mid, -1]]
    Yk_list = [Yk_list[ii] for ii in [0, mid, -1]]
    sensor_id_time_list = [sensor_id_time_list[ii] for ii in [0, mid, -1]]
    rho_list = [rho_list[ii] for ii in [0, mid, -1]]
    Xo_truth = Xk_truth[0]
    
    print(tk_list)
    print(Yk_list)
    print(sensor_id_time_list)
    print(rho_list)
    
    
    # Execute function
    X_list, M_list = iod.gooding_angles_iod(tk_list, Yk_list, sensor_id_time_list,
                                            sensor_params, orbit_regime='LEO',
                                            search_mode='middle_out',
                                            periapsis_check=True,
                                            rootfind='min')
    
    
    print('Final Answers')
    print('Xo_truth', Xo_truth)
    print('X_list', X_list)
    print('M_list', M_list)
    
    for ii in range(len(M_list)):
        
        X_err = X_list[ii] - Xo_truth
        print('')
        print('ii', ii)
        print('M', M_list[ii])
        print('X err', np.linalg.norm(X_err))
        
        
    
    
    ii = 0
    while ii < len(X_list):
        
        Xi = X_list[ii]
        
        print('')
        print('ii', ii)
        
        del_list = []
        for jj in range(len(X_list)):
            
            if jj == ii:
                continue
            
            Xj = X_list[jj]
            if np.linalg.norm(Xi - Xj) < 1e-3:
                del_list.append(jj)
                
        print('del_list', del_list)
        del_list = sorted(del_list, reverse=True)
        for ind in del_list:
            del X_list[ind]
        
        ii += 1
    
    print(X_list)  
        
    
    
    return


def single_rev_leo_retro():
    
    # SSO Orbit
    Xo = np.reshape([757.700301, 5222.606566, 4851.49977,
                         2.213250611, 4.678372741, -5.371314404], (6,1))
    
    
    # Time vector
    UTC0 = datetime(1999, 10, 4, 1, 45, 0)
    tvec = np.arange(0, 1.2*3600., 20.)
    UTC_list = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    
    # Sensor data
    sensor_id_list = ['Born s101', 'Born s337', 'Born s394']
    sensor_params = sens.define_sensors(sensor_id_list)
    
    sensor_id_list = list(sensor_params.keys())
    for sensor_id in sensor_id_list:
        sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Compute measurements
    Yk_list = []
    sensor_id_time_list = []
    rho_list = []
    tk_list = []
    Xk_truth = []
    for UTC in UTC_list:
        
        dt_sec = (UTC - UTC0).total_seconds()
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        Xk = astro.element_conversion(Xo, 1, 1, dt=dt_sec)
        
        for sensor_id in sensor_id_list:
            
            sensor = sensor_params[sensor_id]
            all_meas = mfunc.compute_measurement(Xk, {}, sensor_params, sensor_id, UTC, EOP_data,
                                                 XYs_df, meas_types=['ra', 'dec', 'rg', 'az', 'el'])
            
            el = float(all_meas[4])
            if el > 0.:
                
                print('')
                print(UTC)
                print(sensor_id)
                print(all_meas)
            
                tk_list.append(UTC)
                Xk_truth.append(Xk)
                Yk = all_meas[0:2].reshape(2,1)
                Yk_list.append(Yk)
                sensor_id_time_list.append(sensor_id)
                rho_list.append(float(all_meas[2]))
                
                
    # Select first, middle, last entries
    mid = int(len(Yk_list)/2)
    tk_list = [tk_list[ii] for ii in [0, mid, -1]]
    Yk_list = [Yk_list[ii] for ii in [0, mid, -1]]
    sensor_id_time_list = [sensor_id_time_list[ii] for ii in [0, mid, -1]]
    rho_list = [rho_list[ii] for ii in [0, mid, -1]]
    Xo_truth = Xk_truth[0]
    
    print(tk_list)
    print(Yk_list)
    print(sensor_id_time_list)
    print(rho_list)
    

    
    
    # Execute function
    X_list, M_list = iod.gooding_angles_iod(tk_list, Yk_list, sensor_id_time_list,
                                            sensor_params, orbit_regime='LEO',
                                            search_mode='middle_out',
                                            periapsis_check=True,
                                            rootfind='min')
    
    
    print('Final Answers')
    print('Xo_truth', Xo_truth)
    print('X_list', X_list)
    print('M_list', M_list)
    
    for ii in range(len(M_list)):
        
        X_err = X_list[ii] - Xo_truth
        print('')
        print('ii', ii)
        print('M', M_list[ii])
        print('X err', np.linalg.norm(X_err))
        
        
    
    
    ii = 0
    while ii < len(X_list):
        
        Xi = X_list[ii]
        
        print('')
        print('ii', ii)
        
        del_list = []
        for jj in range(len(X_list)):
            
            if jj == ii:
                continue
            
            Xj = X_list[jj]
            if np.linalg.norm(Xi - Xj) < 1e-3:
                del_list.append(jj)
                
        print('del_list', del_list)
        del_list = sorted(del_list, reverse=True)
        for ind in del_list:
            del X_list[ind]
        
        ii += 1
    
    print(X_list)  
        
    
    
    return


def single_rev_meo():
    
    # GPS Orbit
    elem0 = [26560., 0.01, 55., 90., 10., 10.]
    Xo = astro.kep2cart(elem0)    
    
    # Time vector
    UTC0 = datetime(1999, 10, 4, 1, 45, 0)
    tvec = np.arange(0, 10*3600., 60.)
    UTC_list = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    
    # Sensor data
    sensor_id_list = ['Born s101', 'Born s337', 'Born s394']
    sensor_params = sens.define_sensors(sensor_id_list)
    
    sensor_id_list = list(sensor_params.keys())
    for sensor_id in sensor_id_list:
        sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Compute measurements
    Yk_list = []
    sensor_id_time_list = []
    rho_list = []
    tk_list = []
    Xk_truth = []
    for UTC in UTC_list:
        
        dt_sec = (UTC - UTC0).total_seconds()
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        Xk = astro.element_conversion(Xo, 1, 1, dt=dt_sec)
        
        for sensor_id in sensor_id_list:
            
            sensor = sensor_params[sensor_id]
            all_meas = mfunc.compute_measurement(Xk, {}, sensor_params, sensor_id, UTC, EOP_data,
                                                 XYs_df, meas_types=['ra', 'dec', 'rg', 'az', 'el'])
            
            el = float(all_meas[4])
            if el > 0.:
                
                print('')
                print(UTC)
                print(sensor_id)
                print(all_meas)
            
                tk_list.append(UTC)
                Xk_truth.append(Xk)
                Yk = all_meas[0:2].reshape(2,1)
                Yk_list.append(Yk)
                sensor_id_time_list.append(sensor_id)
                rho_list.append(float(all_meas[2]))
                
                
    # Select first, middle, last entries
    mid = int(len(Yk_list)/2)
    tk_list = [tk_list[ii] for ii in [0, mid, -1]]
    Yk_list = [Yk_list[ii] for ii in [0, mid, -1]]
    sensor_id_time_list = [sensor_id_time_list[ii] for ii in [0, mid, -1]]
    rho_list = [rho_list[ii] for ii in [0, mid, -1]]
    Xo_truth = Xk_truth[0]
    
    print(tk_list)
    print(Yk_list)
    print(sensor_id_time_list)
    print(rho_list)
    
    
    # Execute function
    X_list, M_list = iod.gooding_angles_iod(tk_list, Yk_list, sensor_id_time_list,
                                            sensor_params, orbit_regime='MEO',
                                            search_mode='middle_out',
                                            periapsis_check=True,
                                            rootfind='zeros')
    
    
    print('Final Answers')
    print('Xo_truth', Xo_truth)
    print('X_list', X_list)
    print('M_list', M_list)
    
    for ii in range(len(M_list)):
        
        X_err = X_list[ii] - Xo_truth
        print('')
        print('ii', ii)
        print('M', M_list[ii])
        print('X err', np.linalg.norm(X_err))
        
        
    
    
    ii = 0
    while ii < len(X_list):
        
        Xi = X_list[ii]
        
        print('')
        print('ii', ii)
        
        del_list = []
        for jj in range(len(X_list)):
            
            if jj == ii:
                continue
            
            Xj = X_list[jj]
            if np.linalg.norm(Xi - Xj) < 1e-3:
                del_list.append(jj)
                
        print('del_list', del_list)
        del_list = sorted(del_list, reverse=True)
        for ind in del_list:
            del X_list[ind]
        
        ii += 1
    
    
    # Check final output states and angles
    tof_2 = (tk_list[1] - tk_list[0]).total_seconds()
    tof_f = (tk_list[2] - tk_list[0]).total_seconds()
    sensor0 = sensor_id_time_list[0]
    sensor2 = sensor_id_time_list[1]
    sensorf = sensor_id_time_list[2]
    EOP_data0 = eop.get_eop_data(eop_alldata, tk_list[0])
    EOP_data2 = eop.get_eop_data(eop_alldata, tk_list[1])
    EOP_dataf = eop.get_eop_data(eop_alldata, tk_list[2])
    for ii in range(len(X_list)):
        
        Xi_0 = X_list[ii]
        Xi_2 = astro.element_conversion(Xi_0, 1, 1, dt=tof_2)
        Xi_f = astro.element_conversion(Xi_0, 1, 1, dt=tof_f)
        
        meas0 = mfunc.compute_measurement(Xi_0, {}, sensor_params, sensor0, tk_list[0], EOP_data0,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        meas2 = mfunc.compute_measurement(Xi_2, {}, sensor_params, sensor2, tk_list[1], EOP_data2,
                                          XYs_df, meas_types=['ra', 'dec'])
        measf = mfunc.compute_measurement(Xi_f, {}, sensor_params, sensorf, tk_list[2], EOP_dataf,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        resids0 = (meas0 - Yk_list[0])*(1./arcsec2rad)
        resids2 = (meas2 - Yk_list[1])*(1./arcsec2rad)
        residsf = (measf - Yk_list[2])*(1./arcsec2rad)
        
        # unit vectors
        uhat_meas2 = np.array([[math.cos(meas2[1])*math.cos(meas2[0])],
                               [math.cos(meas2[1])*math.sin(meas2[0])],
                               [math.sin(meas2[1])]])
    
        uhat_yk2 = np.array([[math.cos(Yk_list[1][1])*math.cos(Yk_list[1][0])],
                             [math.cos(Yk_list[1][1])*math.sin(Yk_list[1][0])],
                             [math.sin(Yk_list[1][1])]])
        
        
        print('')
        print('Xi_0', Xi_0)
        print('elem_i', astro.cart2kep(X_list[ii]))
        print('resids0', resids0)
        print('resids2', resids2)
        print('residsf', residsf)

        print(uhat_meas2)
        print(uhat_yk2)
        
        print(float(np.dot(uhat_meas2.T, uhat_yk2)))
        
    
    
    return


def single_rev_heo():
    
    # Molniya
    elem0 = [26600., 0.74, 63.4, 90., 270., 10.]
    Xo = astro.kep2cart(elem0)
    
    # Time vector
    UTC0 = datetime(1999, 10, 4, 1, 45, 0)
    tvec = np.arange(0, 10*3600., 60.)
    UTC_list = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    
    # Sensor data
    sensor_id_list = ['Born s101', 'Born s337', 'Born s394']
    sensor_params = sens.define_sensors(sensor_id_list)
    
    sensor_id_list = list(sensor_params.keys())
    for sensor_id in sensor_id_list:
        sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Compute measurements
    Yk_list = []
    sensor_id_time_list = []
    rho_list = []
    tk_list = []
    Xk_truth = []
    for UTC in UTC_list:
        
        dt_sec = (UTC - UTC0).total_seconds()
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        Xk = astro.element_conversion(Xo, 1, 1, dt=dt_sec)
        
        for sensor_id in sensor_id_list:
            
            sensor = sensor_params[sensor_id]
            all_meas = mfunc.compute_measurement(Xk, {}, sensor_params, sensor_id, UTC, EOP_data,
                                                 XYs_df, meas_types=['ra', 'dec', 'rg', 'az', 'el'])
            
            el = float(all_meas[4])
            if el > 0.:
                
                print('')
                print(UTC)
                print(sensor_id)
                print(all_meas)
            
                tk_list.append(UTC)
                Xk_truth.append(Xk)
                Yk = all_meas[0:2].reshape(2,1)
                Yk_list.append(Yk)
                sensor_id_time_list.append(sensor_id)
                rho_list.append(float(all_meas[2]))
                
                
    # Select first, middle, last entries
    mid = int(len(Yk_list)/2)
    tk_list = [tk_list[ii] for ii in [0, mid, -1]]
    Yk_list = [Yk_list[ii] for ii in [0, mid, -1]]
    sensor_id_time_list = [sensor_id_time_list[ii] for ii in [0, mid, -1]]
    rho_list = [rho_list[ii] for ii in [0, mid, -1]]
    Xo_truth = Xk_truth[0]
    
    print(tk_list)
    print(Yk_list)
    print(sensor_id_time_list)
    print(rho_list)
    
    
    # Execute function
    X_list, M_list = iod.gooding_angles_iod(tk_list, Yk_list, sensor_id_time_list,
                                            sensor_params, orbit_regime='HEO',
                                            search_mode='middle_out',
                                            periapsis_check=True,
                                            rootfind='min')
    
    
    print('Final Answers')
    print('Xo_truth', Xo_truth)
    print('X_list', X_list)
    print('M_list', M_list)
    
    for ii in range(len(M_list)):
        
        X_err = X_list[ii] - Xo_truth
        print('')
        print('ii', ii)
        print('M', M_list[ii])
        print('X err', np.linalg.norm(X_err))
        
        
    
    
    ii = 0
    while ii < len(X_list):
        
        Xi = X_list[ii]
        
        print('')
        print('ii', ii)
        
        del_list = []
        for jj in range(len(X_list)):
            
            if jj == ii:
                continue
            
            Xj = X_list[jj]
            if np.linalg.norm(Xi - Xj) < 1e-3:
                del_list.append(jj)
                
        print('del_list', del_list)
        del_list = sorted(del_list, reverse=True)
        for ind in del_list:
            del X_list[ind]
        
        ii += 1
    
    
    # Check final output states and angles
    tof_2 = (tk_list[1] - tk_list[0]).total_seconds()
    tof_f = (tk_list[2] - tk_list[0]).total_seconds()
    sensor0 = sensor_id_time_list[0]
    sensor2 = sensor_id_time_list[1]
    sensorf = sensor_id_time_list[2]
    EOP_data0 = eop.get_eop_data(eop_alldata, tk_list[0])
    EOP_data2 = eop.get_eop_data(eop_alldata, tk_list[1])
    EOP_dataf = eop.get_eop_data(eop_alldata, tk_list[2])
    for ii in range(len(X_list)):
        
        Xi_0 = X_list[ii]
        Xi_2 = astro.element_conversion(Xi_0, 1, 1, dt=tof_2)
        Xi_f = astro.element_conversion(Xi_0, 1, 1, dt=tof_f)
        
        meas0 = mfunc.compute_measurement(Xi_0, {}, sensor_params, sensor0, tk_list[0], EOP_data0,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        meas2 = mfunc.compute_measurement(Xi_2, {}, sensor_params, sensor2, tk_list[1], EOP_data2,
                                          XYs_df, meas_types=['ra', 'dec'])
        measf = mfunc.compute_measurement(Xi_f, {}, sensor_params, sensorf, tk_list[2], EOP_dataf,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        resids0 = (meas0 - Yk_list[0])*(1./arcsec2rad)
        resids2 = (meas2 - Yk_list[1])*(1./arcsec2rad)
        residsf = (measf - Yk_list[2])*(1./arcsec2rad)
        
        # unit vectors
        uhat_meas2 = np.array([[math.cos(meas2[1])*math.cos(meas2[0])],
                               [math.cos(meas2[1])*math.sin(meas2[0])],
                               [math.sin(meas2[1])]])
    
        uhat_yk2 = np.array([[math.cos(Yk_list[1][1])*math.cos(Yk_list[1][0])],
                             [math.cos(Yk_list[1][1])*math.sin(Yk_list[1][0])],
                             [math.sin(Yk_list[1][1])]])
        
        
        print('')
        print('Xi', Xi)
        print('elem_i', astro.cart2kep(X_list[ii]))
        print('resids0', resids0)
        print('resids2', resids2)
        print('residsf', residsf)
        
        print(uhat_meas2)
        print(uhat_yk2)
        
        print(float(np.dot(uhat_meas2.T, uhat_yk2)))

        
    
    
    return


def single_rev_gto():
    
    # GTO Orbit
    rp = 6678.
    ra = 42164.1
    a = (ra + rp)/2.
    e = 1. - (rp/a)
    
    
    elem0 = [a, e, 28.5, 10., 5., 15.]
    Xo = astro.kep2cart(elem0)
    
    # Time vector
    UTC0 = datetime(1999, 10, 4, 1, 45, 0)
    tvec = np.arange(0, 10*3600., 60.)
    UTC_list = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    
    # Sensor data
    sensor_id_list = ['Born s101', 'Born s337', 'Born s394']
    sensor_params = sens.define_sensors(sensor_id_list)
    
    sensor_id_list = list(sensor_params.keys())
    for sensor_id in sensor_id_list:
        sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Compute measurements
    Yk_list = []
    sensor_id_time_list = []
    rho_list = []
    tk_list = []
    Xk_truth = []
    for UTC in UTC_list:
        
        dt_sec = (UTC - UTC0).total_seconds()
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        Xk = astro.element_conversion(Xo, 1, 1, dt=dt_sec)
        
        for sensor_id in sensor_id_list:
            
            sensor = sensor_params[sensor_id]
            all_meas = mfunc.compute_measurement(Xk, {}, sensor_params, sensor_id, UTC, EOP_data,
                                                 XYs_df, meas_types=['ra', 'dec', 'rg', 'az', 'el'])
            
            el = float(all_meas[4])
            if el > 0.:
                
#                print('')
#                print(UTC)
#                print(sensor_id)
#                print(all_meas)
            
                tk_list.append(UTC)
                Xk_truth.append(Xk)
                Yk = all_meas[0:2].reshape(2,1)
                Yk_list.append(Yk)
                sensor_id_time_list.append(sensor_id)
                rho_list.append(float(all_meas[2]))
                
                
    # Select first, middle, last entries
    mid = int(len(Yk_list)/2)
    tk_list = [tk_list[ii] for ii in [0, mid, -1]]
    Yk_list = [Yk_list[ii] for ii in [0, mid, -1]]
    sensor_id_time_list = [sensor_id_time_list[ii] for ii in [0, mid, -1]]
    rho_list = [rho_list[ii] for ii in [0, mid, -1]]
    Xo_truth = Xk_truth[0]
    
    print(tk_list)
    print(Yk_list)
    print(sensor_id_time_list)
    print(rho_list)

    
    # Execute function
    X_list, M_list = iod.gooding_angles_iod(tk_list, Yk_list, sensor_id_time_list,
                                            sensor_params, orbit_regime='HEO',
                                            search_mode='middle_out',
                                            periapsis_check=True,
                                            rootfind='min')
    
    
    print('Final Answers')
    print('Xo_truth', Xo_truth)
    print('X_list', X_list)
    print('M_list', M_list)
    
    for ii in range(len(M_list)):
        
        X_err = X_list[ii] - Xo_truth
        print('')
        print('ii', ii)
        print('M', M_list[ii])
        print('X err', np.linalg.norm(X_err))
        
        
    
    
    ii = 0
    while ii < len(X_list):
        
        Xi = X_list[ii]
        
        print('')
        print('ii', ii)
        
        del_list = []
        for jj in range(len(X_list)):
            
            if jj == ii:
                continue
            
            Xj = X_list[jj]
            if np.linalg.norm(Xi - Xj) < 1e-3:
                del_list.append(jj)
                
        print('del_list', del_list)
        del_list = sorted(del_list, reverse=True)
        for ind in del_list:
            del X_list[ind]
        
        ii += 1
    
    
    # Check final output states and angles
    tof_2 = (tk_list[1] - tk_list[0]).total_seconds()
    tof_f = (tk_list[2] - tk_list[0]).total_seconds()
    sensor0 = sensor_id_time_list[0]
    sensor2 = sensor_id_time_list[1]
    sensorf = sensor_id_time_list[2]
    EOP_data0 = eop.get_eop_data(eop_alldata, tk_list[0])
    EOP_data2 = eop.get_eop_data(eop_alldata, tk_list[1])
    EOP_dataf = eop.get_eop_data(eop_alldata, tk_list[2])
    for ii in range(len(X_list)):
        
        Xi_0 = X_list[ii]
        Xi_2 = astro.element_conversion(Xi_0, 1, 1, dt=tof_2)
        Xi_f = astro.element_conversion(Xi_0, 1, 1, dt=tof_f)
        
        meas0 = mfunc.compute_measurement(Xi_0, {}, sensor_params, sensor0, tk_list[0], EOP_data0,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        meas2 = mfunc.compute_measurement(Xi_2, {}, sensor_params, sensor2, tk_list[1], EOP_data2,
                                          XYs_df, meas_types=['ra', 'dec'])
        measf = mfunc.compute_measurement(Xi_f, {}, sensor_params, sensorf, tk_list[2], EOP_dataf,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        resids0 = (meas0 - Yk_list[0])*(1./arcsec2rad)
        resids2 = (meas2 - Yk_list[1])*(1./arcsec2rad)
        residsf = (measf - Yk_list[2])*(1./arcsec2rad)
        
        # unit vectors
        uhat_meas2 = np.array([[math.cos(meas2[1])*math.cos(meas2[0])],
                               [math.cos(meas2[1])*math.sin(meas2[0])],
                               [math.sin(meas2[1])]])
    
        uhat_yk2 = np.array([[math.cos(Yk_list[1][1])*math.cos(Yk_list[1][0])],
                             [math.cos(Yk_list[1][1])*math.sin(Yk_list[1][0])],
                             [math.sin(Yk_list[1][1])]])
        
        
        print('')
        print('Xi', Xi)
        print('elem_i', astro.cart2kep(X_list[ii]))
        print('resids0', resids0)
        print('resids2', resids2)
        print('residsf', residsf)
        
        print(uhat_meas2)
        print(uhat_yk2)
        
        print(float(np.dot(uhat_meas2.T, uhat_yk2)))

        
    
    
    return


def single_rev_hyperbola():
    
    # Escape orbit
    rp = 6678.
    vinf = 2.
    a = -GME/vinf**2.
    e = 1. + rp*vinf**2./GME
    
    
    elem0 = [a, e, 28.5, 10., 5., 15.]
    Xo = astro.kep2cart(elem0)
    
    # Time vector
    UTC0 = datetime(1999, 10, 4, 1, 45, 0)
    tvec = np.arange(0, 4*3600., 60.)
    UTC_list = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    
    # Sensor data
    sensor_id_list = ['Born s101', 'Born s337', 'Born s394']
    sensor_params = sens.define_sensors(sensor_id_list)
    
    sensor_id_list = list(sensor_params.keys())
    for sensor_id in sensor_id_list:
        sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Compute measurements
    Yk_list = []
    sensor_id_time_list = []
    rho_list = []
    tk_list = []
    Xk_truth = []
    for UTC in UTC_list:
        
        dt_sec = (UTC - UTC0).total_seconds()
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        Xk = astro.element_conversion(Xo, 1, 1, dt=dt_sec)
        
        for sensor_id in sensor_id_list:
            
            sensor = sensor_params[sensor_id]
            all_meas = mfunc.compute_measurement(Xk, {}, sensor_params, sensor_id, UTC, EOP_data,
                                                 XYs_df, meas_types=['ra', 'dec', 'rg', 'az', 'el'])
            
            el = float(all_meas[4])
            if el > 0.:
                
#                print('')
#                print(UTC)
#                print(sensor_id)
#                print(all_meas)
            
                tk_list.append(UTC)
                Xk_truth.append(Xk)
                Yk = all_meas[0:2].reshape(2,1)
                Yk_list.append(Yk)
                sensor_id_time_list.append(sensor_id)
                rho_list.append(float(all_meas[2]))
                
                
    # Select first, middle, last entries
    mid = int(len(Yk_list)/2)
    tk_list = [tk_list[ii] for ii in [0, mid, -1]]
    Yk_list = [Yk_list[ii] for ii in [0, mid, -1]]
    sensor_id_time_list = [sensor_id_time_list[ii] for ii in [0, mid, -1]]
    rho_list = [rho_list[ii] for ii in [0, mid, -1]]
    Xo_truth = Xk_truth[0]
    
    print(tk_list)
    print(Yk_list)
    print(sensor_id_time_list)
    print(rho_list)

    
    # Execute function
    X_list, M_list = iod.gooding_angles_iod(tk_list, Yk_list, sensor_id_time_list,
                                            sensor_params, orbit_regime='none',
                                            search_mode='middle_out',
                                            periapsis_check=True,
                                            rootfind='min')
    
    
    print('Final Answers')
    print('Xo_truth', Xo_truth)
    print('X_list', X_list)
    print('M_list', M_list)
    
    for ii in range(len(M_list)):
        
        X_err = X_list[ii] - Xo_truth
        print('')
        print('ii', ii)
        print('M', M_list[ii])
        print('X err', np.linalg.norm(X_err))
        
        
    
    
    ii = 0
    while ii < len(X_list):
        
        Xi = X_list[ii]
        
        print('')
        print('ii', ii)
        
        del_list = []
        for jj in range(len(X_list)):
            
            if jj == ii:
                continue
            
            Xj = X_list[jj]
            if np.linalg.norm(Xi - Xj) < 1e-3:
                del_list.append(jj)
                
        print('del_list', del_list)
        del_list = sorted(del_list, reverse=True)
        for ind in del_list:
            del X_list[ind]
        
        ii += 1
    
    
    # Check final output states and angles
    tof_2 = (tk_list[1] - tk_list[0]).total_seconds()
    tof_f = (tk_list[2] - tk_list[0]).total_seconds()
    sensor0 = sensor_id_time_list[0]
    sensor2 = sensor_id_time_list[1]
    sensorf = sensor_id_time_list[2]
    EOP_data0 = eop.get_eop_data(eop_alldata, tk_list[0])
    EOP_data2 = eop.get_eop_data(eop_alldata, tk_list[1])
    EOP_dataf = eop.get_eop_data(eop_alldata, tk_list[2])
    for ii in range(len(X_list)):
        
        Xi_0 = X_list[ii]
        Xi_2 = astro.element_conversion(Xi_0, 1, 1, dt=tof_2)
        Xi_f = astro.element_conversion(Xi_0, 1, 1, dt=tof_f)
        
        meas0 = mfunc.compute_measurement(Xi_0, {}, sensor_params, sensor0, tk_list[0], EOP_data0,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        meas2 = mfunc.compute_measurement(Xi_2, {}, sensor_params, sensor2, tk_list[1], EOP_data2,
                                          XYs_df, meas_types=['ra', 'dec'])
        measf = mfunc.compute_measurement(Xi_f, {}, sensor_params, sensorf, tk_list[2], EOP_dataf,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        resids0 = (meas0 - Yk_list[0])*(1./arcsec2rad)
        resids2 = (meas2 - Yk_list[1])*(1./arcsec2rad)
        residsf = (measf - Yk_list[2])*(1./arcsec2rad)
        
        # unit vectors
        uhat_meas2 = np.array([[math.cos(meas2[1])*math.cos(meas2[0])],
                               [math.cos(meas2[1])*math.sin(meas2[0])],
                               [math.sin(meas2[1])]])
    
        uhat_yk2 = np.array([[math.cos(Yk_list[1][1])*math.cos(Yk_list[1][0])],
                             [math.cos(Yk_list[1][1])*math.sin(Yk_list[1][0])],
                             [math.sin(Yk_list[1][1])]])
        
        
        print('')
        print('Xi', Xi)
        print('elem_i', astro.cart2kep(X_list[ii]))
        print('resids0', resids0)
        print('resids2', resids2)
        print('residsf', residsf)
        
        print(uhat_meas2)
        print(uhat_yk2)
        
        print(float(np.dot(uhat_meas2.T, uhat_yk2)))

        
    
    
    return



def multi_rev_geo():
    
    # Generic GEO Orbit
    elem = [42164.1, 0.01, 0.1, 90., 1., 1.]
    Xo = np.reshape(astro.kep2cart(elem), (6,1))
    print('Xo true', Xo)
    
    # Time vector
    UTC0 = datetime(2021, 6, 21, 0, 0, 0)
    UTC1 = datetime(2021, 6, 22, 4, 0, 0)
    UTC2 = datetime(2021, 6, 23, 6, 0, 0)
    UTC_list = [UTC0, UTC1, UTC2]
    
    # Sensor data
    sensor_id_list = ['UNSW Falcon']
    sensor_id = sensor_id_list[0]
    sensor_params = sens.define_sensors([sensor_id])
    sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
    sensor = sensor_params[sensor_id]
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Compute measurements
    Yk_list = []
    sensor_id_time_list = []
    rho_list = []
    tk_list = []
    Xk_truth = []
    for UTC in UTC_list:
        
        dt_sec = (UTC - UTC0).total_seconds()
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        Xk = astro.element_conversion(Xo, 1, 1, dt=dt_sec)
        
        for sensor_id in sensor_id_list:
            
            sensor = sensor_params[sensor_id]
            all_meas = mfunc.compute_measurement(Xk, {}, sensor_params, sensor_id, UTC, EOP_data,
                                                 XYs_df, meas_types=['ra', 'dec', 'rg', 'az', 'el'])
            
            el = float(all_meas[4])
            if el > 0.:
                
#                print('')
#                print(UTC)
#                print(sensor_id)
#                print(all_meas)
            
                tk_list.append(UTC)
                Xk_truth.append(Xk)
                Yk = all_meas[0:2].reshape(2,1)
                Yk_list.append(Yk)
                sensor_id_time_list.append(sensor_id)
                rho_list.append(float(all_meas[2]))
                
                
    # Select first, middle, last entries
    mid = int(len(Yk_list)/2)
    tk_list = [tk_list[ii] for ii in [0, mid, -1]]
    Yk_list = [Yk_list[ii] for ii in [0, mid, -1]]
    sensor_id_time_list = [sensor_id_time_list[ii] for ii in [0, mid, -1]]
    rho_list = [rho_list[ii] for ii in [0, mid, -1]]
    Xo_truth = Xk_truth[0]
    
    print(tk_list)
    print(Yk_list)
    print(sensor_id_time_list)
    print(rho_list)
    
    
    # Execute function
    X_list, M_list = iod.gooding_angles_iod(tk_list, Yk_list, sensor_id_time_list,
                                            sensor_params, orbit_regime='GEO',
                                            search_mode='middle_out',
                                            periapsis_check=True,
                                            rootfind='min')
    
    
    print('Final Answers')
    print('Xo_truth', Xo_truth)
    print('X_list', X_list)
    print('M_list', M_list)
    
    for ii in range(len(M_list)):
        
        X_err = X_list[ii] - Xo_truth
        print('')
        print('ii', ii)
        print('M', M_list[ii])
        print('X err', np.linalg.norm(X_err))
        
        
    
    
    ii = 0
    while ii < len(X_list):
        
        Xi = X_list[ii]
        
        print('')
        print('ii', ii)
        
        del_list = []
        for jj in range(len(X_list)):
            
            if jj == ii:
                continue
            
            Xj = X_list[jj]
            if np.linalg.norm(Xi - Xj) < 1e-3:
                del_list.append(jj)
                
        print('del_list', del_list)
        del_list = sorted(del_list, reverse=True)
        for ind in del_list:
            del X_list[ind]
        
        ii += 1
    
    
    # Check final output states and angles
    tof_2 = (tk_list[1] - tk_list[0]).total_seconds()
    tof_f = (tk_list[2] - tk_list[0]).total_seconds()
    sensor0 = sensor_id_time_list[0]
    sensor2 = sensor_id_time_list[1]
    sensorf = sensor_id_time_list[2]
    EOP_data0 = eop.get_eop_data(eop_alldata, tk_list[0])
    EOP_data2 = eop.get_eop_data(eop_alldata, tk_list[1])
    EOP_dataf = eop.get_eop_data(eop_alldata, tk_list[2])
    for ii in range(len(X_list)):
        
        Xi_0 = X_list[ii]
        Xi_2 = astro.element_conversion(Xi_0, 1, 1, dt=tof_2)
        Xi_f = astro.element_conversion(Xi_0, 1, 1, dt=tof_f)
        
        meas0 = mfunc.compute_measurement(Xi_0, {}, sensor_params, sensor0, tk_list[0], EOP_data0,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        meas2 = mfunc.compute_measurement(Xi_2, {}, sensor_params, sensor2, tk_list[1], EOP_data2,
                                          XYs_df, meas_types=['ra', 'dec'])
        measf = mfunc.compute_measurement(Xi_f, {}, sensor_params, sensorf, tk_list[2], EOP_dataf,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        resids0 = (meas0 - Yk_list[0])*(1./arcsec2rad)
        resids2 = (meas2 - Yk_list[1])*(1./arcsec2rad)
        residsf = (measf - Yk_list[2])*(1./arcsec2rad)
        
        # unit vectors
        uhat_meas2 = np.array([[math.cos(meas2[1])*math.cos(meas2[0])],
                               [math.cos(meas2[1])*math.sin(meas2[0])],
                               [math.sin(meas2[1])]])
    
        uhat_yk2 = np.array([[math.cos(Yk_list[1][1])*math.cos(Yk_list[1][0])],
                             [math.cos(Yk_list[1][1])*math.sin(Yk_list[1][0])],
                             [math.sin(Yk_list[1][1])]])
        
        
        print('')
        print('Xi', Xi)
        print('elem_i', astro.cart2kep(X_list[ii]))
        print('resids0', resids0)
        print('resids2', resids2)
        print('residsf', residsf)
        
        print(uhat_meas2)
        print(uhat_yk2)
        
        print(float(np.dot(uhat_meas2.T, uhat_yk2)))

        
    
    
    return


def multi_rev_geo2():
    
    # Time vector
    UTC0 = datetime(2022, 11, 7, 11, 0, 0)
    UTC1 = datetime(2022, 11, 7, 11, 10, 0)
    UTC2 = datetime(2022, 11, 8, 14, 10, 0)
    UTC_list = [UTC0, UTC1, UTC2]
    
    # QZS-1R Orbit
    qzs1r_norad = 49336
    obj_id = qzs1r_norad
    state_dict = tle.propagate_TLE([obj_id], [UTC0])
    
    r0 = state_dict[obj_id]['r_GCRF'][0]
    v0 = state_dict[obj_id]['v_GCRF'][0]
    Xo = np.concatenate((r0, v0), axis=0)
    
    
#    elem = [4.21639888e+04, 7.47880515e-02, 3.48399170e+01, 9.92089475e+01,
#            2.70695246e+02, 3.33331109e+02]
#    Xo = np.reshape(astro.kep2cart(elem), (6,1))
    print('Xo true', Xo)
    
    
    
    # Sensor data
    sensor_id_list = ['RMIT ROO']
    sensor_id = sensor_id_list[0]
    sensor_params = sens.define_sensors([sensor_id])
    sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
    sensor = sensor_params[sensor_id]
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Compute measurements
    Yk_list = []
    sensor_id_time_list = []
    rho_list = []
    tk_list = []
    Xk_truth = []
    for UTC in UTC_list:
        
        dt_sec = (UTC - UTC0).total_seconds()
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        Xk = astro.element_conversion(Xo, 1, 1, dt=dt_sec)
        
        print(UTC)
        print(Xk)
        
        for sensor_id in sensor_id_list:
            
            sensor = sensor_params[sensor_id]
            all_meas = mfunc.compute_measurement(Xk, {}, sensor_params, sensor_id, UTC, EOP_data,
                                                 XYs_df, meas_types=['ra', 'dec', 'rg', 'az', 'el'])
            
            el = float(all_meas[4])
            if el > 0.:
                
#                print('')
#                print(UTC)
#                print(sensor_id)
#                print(all_meas)
            
                tk_list.append(UTC)
                Xk_truth.append(Xk)
                Yk = all_meas[0:2].reshape(2,1)
                Yk_list.append(Yk)
                sensor_id_time_list.append(sensor_id)
                rho_list.append(float(all_meas[2]))
                
                
    # Select first, middle, last entries
    mid = int(len(Yk_list)/2)
    tk_list = [tk_list[ii] for ii in [0, mid, -1]]
    Yk_list = [Yk_list[ii] for ii in [0, mid, -1]]
    sensor_id_time_list = [sensor_id_time_list[ii] for ii in [0, mid, -1]]
    rho_list = [rho_list[ii] for ii in [0, mid, -1]]
    Xo_truth = Xk_truth[0]
    
    print(tk_list)
    print(Yk_list)
    print(sensor_id_time_list)
    print(rho_list)
    
    mistake

    # Execute function
    X_list, M_list = iod.gooding_angles_iod(tk_list, Yk_list, sensor_id_time_list,
                                            sensor_params, orbit_regime='GEO',
                                            search_mode='middle_out',
                                            periapsis_check=True,
                                            rootfind='zeros')
    
    
    print('Final Answers')
    print('Xo_truth', Xo_truth)
    print('X_list', X_list)
    print('M_list', M_list)
    
    for ii in range(len(M_list)):
        
        X_err = X_list[ii] - Xo_truth
        print('')
        print('ii', ii)
        print('M', M_list[ii])
        print('X err', np.linalg.norm(X_err))
        
        
    
    
    ii = 0
    while ii < len(X_list):
        
        Xi = X_list[ii]
        
        print('')
        print('ii', ii)
        
        del_list = []
        for jj in range(len(X_list)):
            
            if jj == ii:
                continue
            
            Xj = X_list[jj]
            if np.linalg.norm(Xi - Xj) < 1e-3:
                del_list.append(jj)
                
        print('del_list', del_list)
        del_list = sorted(del_list, reverse=True)
        for ind in del_list:
            del X_list[ind]
        
        ii += 1
    
    
    # Check final output states and angles
    tof_2 = (tk_list[1] - tk_list[0]).total_seconds()
    tof_f = (tk_list[2] - tk_list[0]).total_seconds()
    sensor0 = sensor_id_time_list[0]
    sensor2 = sensor_id_time_list[1]
    sensorf = sensor_id_time_list[2]
    EOP_data0 = eop.get_eop_data(eop_alldata, tk_list[0])
    EOP_data2 = eop.get_eop_data(eop_alldata, tk_list[1])
    EOP_dataf = eop.get_eop_data(eop_alldata, tk_list[2])
    for ii in range(len(X_list)):
        
        Xi_0 = X_list[ii]
        Xi_2 = astro.element_conversion(Xi_0, 1, 1, dt=tof_2)
        Xi_f = astro.element_conversion(Xi_0, 1, 1, dt=tof_f)
        
        meas0 = mfunc.compute_measurement(Xi_0, {}, sensor_params, sensor0, tk_list[0], EOP_data0,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        meas2 = mfunc.compute_measurement(Xi_2, {}, sensor_params, sensor2, tk_list[1], EOP_data2,
                                          XYs_df, meas_types=['ra', 'dec'])
        measf = mfunc.compute_measurement(Xi_f, {}, sensor_params, sensorf, tk_list[2], EOP_dataf,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        resids0 = (meas0 - Yk_list[0])*(1./arcsec2rad)
        resids2 = (meas2 - Yk_list[1])*(1./arcsec2rad)
        residsf = (measf - Yk_list[2])*(1./arcsec2rad)
        
        # unit vectors
        uhat_meas2 = np.array([[math.cos(meas2[1])*math.cos(meas2[0])],
                               [math.cos(meas2[1])*math.sin(meas2[0])],
                               [math.sin(meas2[1])]])
    
        uhat_yk2 = np.array([[math.cos(Yk_list[1][1])*math.cos(Yk_list[1][0])],
                             [math.cos(Yk_list[1][1])*math.sin(Yk_list[1][0])],
                             [math.sin(Yk_list[1][1])]])
        
        
        print('')
        print('Xi', Xi)
        print('elem_i', astro.cart2kep(X_list[ii]))
        print('resids0', resids0)
        print('resids2', resids2)
        print('residsf', residsf)
        
        print(uhat_meas2)
        print(uhat_yk2)
        
        print(float(np.dot(uhat_meas2.T, uhat_yk2)))

        
    
    
    return


def multi_rev_geo3():
    
    # Time vector
    UTC0 = datetime(2022, 11, 10, 10, 20, 0)
    UTC1 = datetime(2022, 11, 10, 10, 30, 0)
    UTC2 = datetime(2022, 11, 13, 10, 11, 0)
    UTC_list = [UTC0, UTC1, UTC2]

    Xo = np.reshape([ 3.69733077e+04, -2.02670750e+04, -3.04177680e+01, 
                     1.47850927e+00, 2.69592671e+00, -4.50623464e-03], (6,1))
    
    print('Xo true', Xo)
    

    # Sensor data
    sensor_id_list = ['RMIT ROO']
    sensor_id = sensor_id_list[0]
    sensor_params = sens.define_sensors([sensor_id])
    sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
    sensor = sensor_params[sensor_id]
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Compute measurements
    Yk_list = []
    sensor_id_time_list = []
    rho_list = []
    tk_list = []
    Xk_truth = []
    for UTC in UTC_list:
        
        dt_sec = (UTC - UTC0).total_seconds()
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        Xk = astro.element_conversion(Xo, 1, 1, dt=dt_sec)
        
        print(UTC)
        print(Xk)
        
        for sensor_id in sensor_id_list:
            
            sensor = sensor_params[sensor_id]
            all_meas = mfunc.compute_measurement(Xk, {}, sensor_params, sensor_id, UTC, EOP_data,
                                                 XYs_df, meas_types=['ra', 'dec', 'rg', 'az', 'el'])
            
            el = float(all_meas[4])
            if el > 0.:
                
#                print('')
#                print(UTC)
#                print(sensor_id)
#                print(all_meas)
            
                tk_list.append(UTC)
                Xk_truth.append(Xk)
                Yk = all_meas[0:2].reshape(2,1)
                Yk_list.append(Yk)
                sensor_id_time_list.append(sensor_id)
                rho_list.append(float(all_meas[2]))
                
                
    # Select first, middle, last entries
    mid = int(len(Yk_list)/2)
    tk_list = [tk_list[ii] for ii in [0, mid, -1]]
    Yk_list = [Yk_list[ii] for ii in [0, mid, -1]]
    sensor_id_time_list = [sensor_id_time_list[ii] for ii in [0, mid, -1]]
    rho_list = [rho_list[ii] for ii in [0, mid, -1]]
    Xo_truth = Xk_truth[0]
    
    print(tk_list)
    print(Yk_list)
    print(sensor_id_time_list)
    print(rho_list)
    


    # Execute function
    X_list, M_list = iod.gooding_angles_iod(tk_list, Yk_list, sensor_id_time_list,
                                            sensor_params, orbit_regime='GEO',
                                            search_mode='middle_out',
                                            periapsis_check=True, 
                                            rootfind='min', debug=True)
    
    
    print('Final Answers')
    print('Xo_truth', Xo_truth)
    print('X_list', X_list)
    print('M_list', M_list)
    
    for ii in range(len(M_list)):
        
        X_err = X_list[ii] - Xo_truth
        print('')
        print('ii', ii)
        print('M', M_list[ii])
        print('X err', np.linalg.norm(X_err))
        
        
    
    
    ii = 0
    while ii < len(X_list):
        
        Xi = X_list[ii]
        
        print('')
        print('ii', ii)
        
        del_list = []
        for jj in range(len(X_list)):
            
            if jj == ii:
                continue
            
            Xj = X_list[jj]
            if np.linalg.norm(Xi - Xj) < 1e-3:
                del_list.append(jj)
                
        print('del_list', del_list)
        del_list = sorted(del_list, reverse=True)
        for ind in del_list:
            del X_list[ind]
        
        ii += 1
    
    
    # Check final output states and angles
    tof_2 = (tk_list[1] - tk_list[0]).total_seconds()
    tof_f = (tk_list[2] - tk_list[0]).total_seconds()
    sensor0 = sensor_id_time_list[0]
    sensor2 = sensor_id_time_list[1]
    sensorf = sensor_id_time_list[2]
    EOP_data0 = eop.get_eop_data(eop_alldata, tk_list[0])
    EOP_data2 = eop.get_eop_data(eop_alldata, tk_list[1])
    EOP_dataf = eop.get_eop_data(eop_alldata, tk_list[2])
    for ii in range(len(X_list)):
        
        Xi_0 = X_list[ii]
        Xi_2 = astro.element_conversion(Xi_0, 1, 1, dt=tof_2)
        Xi_f = astro.element_conversion(Xi_0, 1, 1, dt=tof_f)
        
        meas0 = mfunc.compute_measurement(Xi_0, {}, sensor_params, sensor0, tk_list[0], EOP_data0,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        meas2 = mfunc.compute_measurement(Xi_2, {}, sensor_params, sensor2, tk_list[1], EOP_data2,
                                          XYs_df, meas_types=['ra', 'dec'])
        measf = mfunc.compute_measurement(Xi_f, {}, sensor_params, sensorf, tk_list[2], EOP_dataf,
                                          XYs_df, meas_types=['ra', 'dec'])
        
        resids0 = (meas0 - Yk_list[0])*(1./arcsec2rad)
        resids2 = (meas2 - Yk_list[1])*(1./arcsec2rad)
        residsf = (measf - Yk_list[2])*(1./arcsec2rad)
        
        # unit vectors
        uhat_meas2 = np.array([[math.cos(meas2[1])*math.cos(meas2[0])],
                               [math.cos(meas2[1])*math.sin(meas2[0])],
                               [math.sin(meas2[1])]])
    
        uhat_yk2 = np.array([[math.cos(Yk_list[1][1])*math.cos(Yk_list[1][0])],
                             [math.cos(Yk_list[1][1])*math.sin(Yk_list[1][0])],
                             [math.sin(Yk_list[1][1])]])
        
        
        print('')
        print('Xi', Xi)
        print('elem_i', astro.cart2kep(X_list[ii]))
        print('resids0', resids0)
        print('resids2', resids2)
        print('residsf', residsf)
        
        print(uhat_meas2)
        print(uhat_yk2)
        
        print(float(np.dot(uhat_meas2.T, uhat_yk2)))

        
    
    
    return


if __name__ == '__main__':
    
    
#    cProfile.run('single_rev_geo()')
    
    single_rev_geo()
    
#    single_rev_leo()
    
#    single_rev_leo_retro()
    
#    single_rev_meo()
    
#    single_rev_heo()
    
#    single_rev_gto()
    
#    single_rev_hyperbola()
    
#    multi_rev_geo()
    
    # multi_rev_geo2()
    
    # multi_rev_geo3()
    
    # linh_test_case()
    
    
    
    





























