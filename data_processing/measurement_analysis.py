import numpy as np
import math
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import sys


from data_processing import data_processing as proc
from sensors import measurement_functions as mfunc
from sensors import sensors as sens
from utilities import coordinate_systems as coord
from utilities import eop_functions as eop
from utilities import numerical_methods as num
from utilities import time_systems as timesys
from utilities import tle_functions as tle

from utilities.constants import arcsec2rad, GME, J2E, wE, Re


###############################################################################
# ROO Data Analysis
###############################################################################

def read_roo_csv_data(fname, meas_time_offset=0., ra_bias=0., dec_bias=0.):
    
    # Load measurement data
    df = pd.read_csv(fname)
    
    # Retrieve UTC times, RA, DEC and convert datetime, radians
    UTC_str_list = df['Date-obs_corrected_midexp'].tolist()
    ra_deg_list = df['RA'].tolist()
    dec_deg_list = df['DEC'].tolist()
    
    UTC_list = [datetime.strptime(UTC_str, '%Y-%m-%dT%H:%M:%S.%f') + 
                timedelta(seconds=meas_time_offset) for UTC_str in UTC_str_list]
    ra_list = [ra*math.pi/180. - ra_bias for ra in ra_deg_list]
    dec_list = [dec*math.pi/180. - dec_bias for dec in dec_deg_list]
    
    
    return UTC_list, ra_list, dec_list


def compute_radec_errors(meas_file, truth_file, norad_id, sp3_id, sensor_id, meas_time_offset=0.):
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Retrieve sensor data
    sensor_params = sens.define_sensors([sensor_id])
    
    # Read measurement and truth data
    UTC_list_meas, ra_list_meas, dec_list_meas = \
        read_roo_csv_data(meas_file, meas_time_offset=meas_time_offset)
    truth_dict = proc.read_sp3_file(truth_file)
    UTC0 = UTC_list_meas[0]
    
    # Convert truth dict times to UTC and states to ECI
    gps_list = truth_dict[sp3_id]['gps_time']
    ecef_list = truth_dict[sp3_id]['r_ecef']
#    clock_list = truth_dict[sp3_id]['clock_offset_sec']
    dt_sec_truth = np.zeros((len(gps_list),))
    ECI_array = np.zeros((len(gps_list), 3))
    for ii in range(len(gps_list)):
        
        gps_time = gps_list[ii]
        sp3_ecef = ecef_list[ii]
        EOP_data = eop.get_eop_data(eop_alldata, gps_time)
        
        # Convert to UTC
        UTC = timesys.gpsdt2utcdt(gps_time, EOP_data['TAI_UTC'])
#        clock_offset = clock_list[ii]
        dt_sec_truth[ii] = (UTC - UTC0).total_seconds() # + clock_offset # timedelta(seconds=clock_offset)
        
        # Convert to GCRF
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        sp3_eci, dum = coord.itrf2gcrf(sp3_ecef, np.zeros((3,1)), UTC, EOP_data, XYs_df)
        ECI_array[ii,:] = sp3_eci.flatten()
        
        
    # Interpolate truth ECI data to measurement times
    ra_resids = np.zeros(len(UTC_list_meas),)
    dec_resids = np.zeros(len(UTC_list_meas),)
    ra_true = np.zeros(len(UTC_list_meas),)
    dec_true = np.zeros(len(UTC_list_meas),)
    az_true = np.zeros(len(UTC_list_meas),)
    el_true = np.zeros(len(UTC_list_meas),)
    
    for ii in range(len(UTC_list_meas)):
        
        # Retrieve values
        UTC = UTC_list_meas[ii]
        dt_sec = (UTC - UTC0).total_seconds()
        ra_meas = ra_list_meas[ii]
        dec_meas = dec_list_meas[ii]
        
        # Interpolate truth data to current measurement time
        r_eci = num.interp_lagrange(dt_sec_truth, ECI_array, dt_sec, 9)
        
        # Compute measurement from interpolated truth data
        meas_types = ['ra', 'dec', 'az', 'el']
        state_params = {}
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        Y = mfunc.compute_measurement(r_eci, state_params, sensor_params,
                                      sensor_id, UTC, EOP_data, XYs_df,
                                      meas_types)
        
        ra_true[ii] = float(Y[0])
        dec_true[ii] = float(Y[1])
        az_true[ii] = float(Y[2])
        el_true[ii] = float(Y[3])
        
        # Compute and store resids
        ra_resids[ii] = ra_meas - ra_true[ii]
        dec_resids[ii] = dec_meas - dec_true[ii]
        
        # Fix quadrant for RA resids
        if ra_resids[ii] > math.pi:
            ra_resids[ii] -= 2.*math.pi
        if ra_resids[ii] < -math.pi:
            ra_resids[ii] += 2.*math.pi
            
    
    return UTC_list_meas, ra_resids, dec_resids, ra_true, dec_true, az_true, el_true


def characterize_measurement_errors(meas_file, truth_file, norad_id, sp3_id,
                                    sensor_id, meas_time_offset=0.):
    
    
    UTC_list, ra_resids, dec_resids, ra_true, dec_true, az_true, el_true = \
        compute_radec_errors(meas_file, truth_file, norad_id, sp3_id, sensor_id,
                             meas_time_offset)
    
    # Fix units
    UTC0 = UTC_list[0]
    thrs = [(UTC - UTC0).total_seconds()/3600. for UTC in UTC_list]
    
    ra_resids *= (1./arcsec2rad)
    dec_resids *= (1./arcsec2rad)
    ra_true *= 180./math.pi
    dec_true *= 180./math.pi 
    az_true *= 180./math.pi
    el_true *= 180./math.pi
    
    print(ra_true+360.)
    print(dec_true)
        
    # Generate plot
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(thrs, ra_resids, 'k.')
    plt.ylabel('RA [arcsec]')
    plt.title('Prefit Residuals (Obs - True (interpolated))')
    plt.subplot(2,1,2)
    plt.plot(thrs, dec_resids, 'k.')
    plt.ylabel('DEC [arcsec]')
    plt.xlabel('Time Since ' + UTC0.strftime('%Y-%m-%d %H:%M:%S') + ' [hours]')
    
    plt.show()
    
    ra_mean = np.mean(ra_resids)
    dec_mean = np.mean(dec_resids)
    ra_std = np.std(ra_resids)
    dec_std = np.std(dec_resids)
    
    outlier_inds = []
    for ii in range(len(ra_resids)):
        if abs(ra_resids[ii] - ra_mean) > 3.*ra_std:
            outlier_inds.append(ii)
            
        if abs(ra_resids[ii]) > 100:
            outlier_inds.append(ii)
            
    for ii in range(len(dec_resids)):
        if abs(dec_resids[ii] - dec_mean) > 3.*dec_std:
            outlier_inds.append(ii)
            
        if abs(dec_resids[ii]) > 100:
            outlier_inds.append(ii)
            
    outlier_inds = sorted(list(set(outlier_inds)))
    
    print('\nError Statistics')
    print('RA mean and std [arcsec]: ' + '{:.3f}'.format(ra_mean) + ', {:.3f}'.format(ra_std))
    print('DEC mean and std [arcsec]: ' + '{:.3f}'.format(dec_mean) + ', {:.3f}'.format(dec_std))
    print('Outlier indices: ', outlier_inds)
    
    return ra_mean, dec_mean


def convert_radec_to_deg(meas_file):
    
    df = pd.read_csv(meas_file, header=None)
    
    print(df)
    
    t0 = datetime(2019, 9, 3, 10, 9, 42)
    
    times = df.iloc[0,:]
    ra = df.iloc[1,:]
    dec = df.iloc[2,:]
    
    print(times)
    print(ra)
    print(dec)
    
    output = np.zeros((3,len(times)))    
    for ii in range(len(times)):
        
        print(times[ii])
        ti = datetime.strptime(times[ii], '%Y-%m-%dT%H:%M:%S.%f')
        ti_sec = (ti - t0).total_seconds()
        
        ra_deg = (float(ra[ii][0:2]) + float(ra[ii][3:5])/60. + float(ra[ii][6:])/3600.)*15.
        dec_deg = float(dec[ii][0:2]) + float(dec[ii][3:5])/60. + float(dec[ii][6:])/3600.
                       
        ra_rad = ra_deg*pi/180.
        if ra_rad > pi:
            ra_rad -= 2*pi
        dec_rad = dec_deg*pi/180.
        
        output[0,ii] = ti_sec
        output[1,ii] = ra_rad
        output[2,ii] = dec_rad
        
    
    
    print(output)
    meas_df = pd.DataFrame(output)    
    csv_name = os.path.join(fdir, 'meas_data_input.csv')
    meas_df.to_csv(csv_name, index=False)
        
        
    
    
    return



###############################################################################
# TLE Data Analysis
###############################################################################


def compute_tle_error(UTC_list, truth_file, norad_id, sp3_id):
    
    # Initial time
    UTC0 = UTC_list[0]
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Convert SP3 dict times to UTC and states to ECI
    sp3_dict = proc.read_sp3_file(truth_file)
    gps_list = sp3_dict[sp3_id]['gps_time']
    ecef_list = sp3_dict[sp3_id]['r_ecef']
#    clock_list = truth_dict[sp3_id]['clock_offset_sec']
    dt_sec_truth = np.zeros((len(gps_list),))
    ECI_array = np.zeros((len(gps_list), 3))
    for ii in range(len(gps_list)):
        
        gps_time = gps_list[ii]
        sp3_ecef = ecef_list[ii]
        EOP_data = eop.get_eop_data(eop_alldata, gps_time)
        
        # Convert to UTC
        UTC = timesys.gpsdt2utcdt(gps_time, EOP_data['TAI_UTC'])
#        clock_offset = clock_list[ii]
        dt_sec_truth[ii] = (UTC - UTC0).total_seconds() # + clock_offset # timedelta(seconds=clock_offset)
        
        # Convert to GCRF
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        sp3_eci, dum = coord.itrf2gcrf(sp3_ecef, np.zeros((3,1)), UTC, EOP_data, XYs_df)
        ECI_array[ii,:] = sp3_eci.flatten()
        
        
    # Store truth data
    truth_dict = {}
    for ii in range(len(UTC_list)):
        
        # Retrieve values
        UTC = UTC_list[ii]
        
        # Interpolate truth ECI data to measurement times
        dt_sec = (UTC - UTC0).total_seconds()
        r_eci = num.interp_lagrange(dt_sec_truth, ECI_array, dt_sec, 9)
        
        # Store data        
        truth_dict[UTC] = r_eci.reshape(3,1)
    
    # TLE Errors
    # Retrieve and propagate TLE data to desired times
    tle_state = tle.propagate_TLE([norad_id], UTC_list)
    
    # Compute errors at each time
    tle_eci_err = np.zeros((3, len(UTC_list)))
    tle_ric_err = np.zeros((3, len(UTC_list)))
    for ii in range(len(UTC_list)):
        
        UTC = UTC_list[ii]        
        r_true = truth_dict[UTC]
        tle_r_eci = tle_state[norad_id]['r_GCRF'][ii].reshape(3,1)
        tle_v_eci = tle_state[norad_id]['v_GCRF'][ii].reshape(3,1)
        
        # Compute RIC errors with TLE data acting as chief satellite
        rho_eci = r_true - tle_r_eci    
        rho_ric = coord.eci2ric(tle_r_eci, tle_v_eci, rho_eci)
        
        # Change sign to set SP3 data as chief (truth)
        rho_ric = -rho_ric      
        
        # Store output
        tle_eci_err[:,ii] = (tle_r_eci - r_true).flatten()
        tle_ric_err[:,ii] = rho_ric.flatten()
        
    # Fix Units
    tle_eci_err *= 1000.
    tle_ric_err *= 1000.
    thrs = [(UTC - UTC0).total_seconds()/3600. for UTC in UTC_list]

    # Compute and print statistics
    print('\n\nState Error Analysis')
    print('\n\t\t\t\t  Mean\t\tSTD')
    print('----------------------------------------')
    print('TLE X ECI [m]\t\t', '{0:0.2E}'.format(np.mean(tle_eci_err[0,:])), '\t{0:0.2E}'.format(np.std(tle_eci_err[0,:])))
    print('TLE Y ECI [m]\t\t', '{0:0.2E}'.format(np.mean(tle_eci_err[1,:])), '\t{0:0.2E}'.format(np.std(tle_eci_err[1,:])))
    print('TLE Z ECI [m]\t\t', '{0:0.2E}'.format(np.mean(tle_eci_err[2,:])), '\t{0:0.2E}'.format(np.std(tle_eci_err[2,:])))
    print('')
    print('TLE Radial [m]\t\t', '{0:0.2E}'.format(np.mean(tle_ric_err[0,:])), '\t{0:0.2E}'.format(np.std(tle_ric_err[0,:])))
    print('TLE In-Track [m]\t', '{0:0.2E}'.format(np.mean(tle_ric_err[1,:])), '\t{0:0.2E}'.format(np.std(tle_ric_err[1,:])))
    print('TLE Cross-Track [m]\t', '{0:0.2E}'.format(np.mean(tle_ric_err[2,:])), '\t{0:0.2E}'.format(np.std(tle_ric_err[2,:])))
    print('')
    

    
    # State Error Plots   
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, tle_ric_err[0,:], 'r.')
    plt.ylabel('Radial [m]')
    plt.title('TLE RIC Error')
    
    plt.subplot(3,1,2)
    plt.plot(thrs, tle_ric_err[1,:], 'r.')
    plt.ylabel('In-Track [m]')
    
    plt.subplot(3,1,3)
    plt.plot(thrs, tle_ric_err[2,:], 'r.')
    plt.ylabel('Cross-Track [m]')

    plt.xlabel('Time since ' + UTC0.strftime('%Y-%m-%d %H:%M:%S') + ' [hours]')
        
    plt.show()
        
    
    return