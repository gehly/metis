import numpy as np
from math import pi
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

sys.path.append('C:\\Users\Steve\Documents\code\metis')

from utilities.data_reader import read_sp3_file
from utilities.tle_functions import propagate_TLE
from utilities.tle_functions import get_spacetrack_tle_data
from sensors.sensors import define_sensors
from utilities.eop_functions import get_celestrak_eop_alldata
from utilities.eop_functions import get_nutation_data
from utilities.eop_functions import get_eop_data
from utilities.coordinate_systems import teme2gcrf
from utilities.coordinate_systems import gcrf2teme
from utilities.coordinate_systems import gcrf2itrf
from utilities.coordinate_systems import itrf2gcrf
from utilities.coordinate_systems import latlonht2ecef
from utilities.coordinate_systems import eci2ric
from utilities.astrodynamics import element_conversion
from utilities.constants import GME
from utilities.time_systems import gpsdt2utcdt




def truth_vs_tle():
    
    plt.close('all')
    
    # Setup
    eop_alldata = get_celestrak_eop_alldata()
    
    sensor_dict = define_sensors(['UNSW Falcon'])
    latlonht = sensor_dict['UNSW Falcon']['geodetic_latlonht']
    lat = latlonht[0]
    lon = latlonht[1]
    ht = latlonht[2]
    stat_ecef = latlonht2ecef(lat, lon, ht)
    
    # Object ID
    norad_id = 42917
    sp3_obj_id = 'PJ07'
    
    # Measurement Times
    t0 = datetime(2019, 9, 3, 10, 9, 42)
    tf = datetime(2019, 9, 3, 18, 4, 42)


    # Read SP3 file
    fdir = 'D:\documents\\teaching\\unsw_ssa_undergrad\lab\\telescope\\truth_data'
    fname = 'gbm20692.sp3'
    sp3_fname = os.path.join(fdir, fname)
    
    sp3_dict = read_sp3_file(sp3_fname)
    
    
    gps_time_list = sp3_dict[sp3_obj_id]['datetime']
    sp3_ecef_list = sp3_dict[sp3_obj_id]['r_ecef']
    
    UTC_list = []
    for ii in range(len(gps_time_list)):
        
        gps_time = gps_time_list[ii]
        sp3_ecef = sp3_ecef_list[ii]
        EOP_data = get_eop_data(eop_alldata, gps_time)
        
        # Convert to UTC
        utc_time = gpsdt2utcdt(gps_time, EOP_data['TAI_UTC'])
        UTC_list.append(utc_time) # + timedelta(seconds=ti) for ti in range(0,101,10)]
        
        
    print(UTC_list[0:10])
    print(UTC_list[0])
    
    obj_id_list = [norad_id]
    
    
    tle_dict, tle_df = get_spacetrack_tle_data(obj_id_list, [UTC_list[0]])
    print(tle_dict)
    
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)
    
    
    ric_err = np.zeros((3, len(UTC_list)))
    ti_hrs = []
    truth_out = []
    tle_ric_err = []
    for ii in range(len(UTC_list)):
        UTC = UTC_list[ii]
        EOP_data = get_eop_data(eop_alldata, UTC)
        tle_r_eci = output_state[obj_id]['r_GCRF'][ii]
        tle_v_eci = output_state[obj_id]['v_GCRF'][ii]
        
#        r_ecef, v_ecef = gcrf2itrf(r_eci, v_eci, UTC, EOP_data)
        
        sp3_ecef = sp3_ecef_list[ii]
        sp3_r_eci, dum = itrf2gcrf(sp3_ecef, np.zeros((3,1)), UTC, EOP_data)
        
        rho_eci = sp3_r_eci - tle_r_eci  # TLE data as chief
        rho_ric = eci2ric(tle_r_eci, tle_v_eci, rho_eci)
        rho_ric = -rho_ric  # treats SP3 data as truth
        
        ric_err[:,ii] = rho_ric.flatten()
        
        ti_hrs.append((UTC - UTC_list[0]).total_seconds()/3600.)
        
        
        if UTC >= t0 and UTC <= tf:
            
            ti_output = (UTC - t0).total_seconds()
            x_output = float(sp3_r_eci[0])
            y_output = float(sp3_r_eci[1])
            z_output = float(sp3_r_eci[2])
            next_out = np.array([[ti_output], [x_output], [y_output], [z_output]])
            
            if len(truth_out) == 0:
                truth_out = next_out.copy()
            else:            
                truth_out = np.concatenate((truth_out, next_out), axis=1)
            
            ric_out = np.insert(ric_err[:,ii], 0, ti_output)
            ric_out = np.reshape(ric_out, (4,1))
            if len(tle_ric_err) == 0:
                tle_ric_err = ric_out
            else:
                tle_ric_err = np.concatenate((tle_ric_err, ric_out), axis=1)
        
    
    print(truth_out)
    truth_df = pd.DataFrame(truth_out)
    print(truth_df)
    
#    csv_name = os.path.join(fdir, 'truth.csv')
#    truth_df.to_csv(csv_name, index=False)
    
#    csv_name2 = os.path.join(fdir, 'tle_ric_err.csv')
#    tle_ric_df = pd.DataFrame(tle_ric_err)
#    tle_ric_df.to_csv(csv_name2, index=False)
    
    print('RMS Values')
    n = len(UTC_list)
    rms_r = np.sqrt((1/n) * np.dot(ric_err[0,:], ric_err[0,:]))
    rms_i = np.sqrt((1/n) * np.dot(ric_err[1,:], ric_err[1,:]))
    rms_c = np.sqrt((1/n) * np.dot(ric_err[2,:], ric_err[2,:]))
    
    print('Radial RMS [km]:', rms_r)
    print('In-Track RMS [km]:', rms_i)
    print('Cross-Track RMS [km]:', rms_c)
    
    
        
    plt.figure()  
    plt.subplot(3,1,1)
    plt.plot(ti_hrs, ric_err[0,:], 'k.')
    plt.ylabel('Radial [km]')
    plt.ylim([-2, 2])
    plt.title('RIC Error TLE vs SP3')
    plt.subplot(3,1,2)
    plt.plot(ti_hrs, ric_err[1,:], 'k.')
    plt.ylabel('In-Track [km]')
    plt.ylim([-5, 5])
    plt.subplot(3,1,3)
    plt.plot(ti_hrs, ric_err[2,:], 'k.')
    plt.ylabel('Cross-Track [km]')
    plt.ylim([-2, 2])
    plt.xlabel('Time Since ' + UTC_list[0].strftime('%Y-%m-%d %H:%M:%S') + ' [hours]')
    
    plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
#        print(UTC)
#        print('ECI \n', r_eci)
#        print('ECEF \n', r_ecef)
#        
#        sp3_ecef = np.array([[-25379.842058],[33676.622067],[51.528803]])
#        
#        print(sp3_ecef - r_ecef)
#        print(np.linalg.norm(sp3_ecef - r_ecef))
#        
#        stat_eci, dum = itrf2gcrf(stat_ecef, np.zeros((3,1)), UTC, EOP_data)
#        
#        print(stat_eci)
#        print(r_eci)
#        
#        
#        rho_eci = np.reshape(r_eci, (3,1)) - np.reshape(stat_eci, (3,1))
#        
#        print(rho_eci)
#        print(r_eci)
#        print(stat_eci)
#        
#        ra = atan2(rho_eci[1], rho_eci[0]) * 180./pi
#        
#        dec = asin(rho_eci[2]/np.linalg.norm(rho_eci)) * 180./pi
#        
#        print('tle data')
#        print(ra)
#        print(dec)
#        
#        
#        sp3_eci, dum = itrf2gcrf(sp3_ecef, np.zeros((3,1)), UTC, EOP_data)
#        
#        rho_eci2 = sp3_eci - stat_eci
#        
#        ra2 = atan2(rho_eci2[1], rho_eci2[0]) * 180./pi
#        
#        dec2 = asin(rho_eci2[2]/np.linalg.norm(rho_eci2)) * 180./pi
#        
#        print('sp3 data')
#        print(ra2)
#        print(dec2)
    
    
    
    return


def compute_ERA_init_state():
    
    
    return


def generate_truth_file():
    
    
    return


def generate_meas_file():
    
    fdir = 'D:\documents\\teaching\\unsw_ssa_undergrad\lab\\telescope\meas_data'
    fname = 'meas_data_all3.csv'
    meas_file_in = os.path.join(fdir, fname)
    
    df = pd.read_csv(meas_file_in, header=None)
    
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


def generate_sensor_location_file():
    
    fdir = r'D:\documents\teaching\unsw_ssa_undergrad\2021\lab\telescope\orbit_determination\student_data'
    fname = 'meas_data_input_Jordan.csv'
    meas_file_in = os.path.join(fdir, fname)
    
    df = pd.read_csv(meas_file_in, header=None)
    
    print(df)
    

    
    t0 = datetime(2019, 9, 3, 10, 9, 42)
    
    eop_alldata = get_celestrak_eop_alldata()    
    
    
    times = df.iloc[0,:]

    
    lat_gs = -35.29
    lon_gs = 149.17
    ht_gs = 0.606 # km	
    
    stat_ecef = latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    output = np.zeros((4,len(times)))    
    for ii in range(len(times)):
        
        print(times[ii])
#        ti = datetime.strptime(times[ii], '%Y-%m-%dT%H:%M:%S.%f')
        ti = t0 + timedelta(seconds=times[ii])
        
        ti_sec = (ti - t0).total_seconds()
        EOP_data = get_eop_data(eop_alldata, ti)
        
        stat_eci, dum = itrf2gcrf(stat_ecef, np.zeros((3,1)), ti, EOP_data)
        
        print(stat_eci)
        
        output[0,ii] = ti_sec
        output[1,ii] = float(stat_eci[0])
        output[2,ii] = float(stat_eci[1])
        output[3,ii] = float(stat_eci[2])
        
    print(output)
    sensor_df = pd.DataFrame(output)    
    csv_name = os.path.join(fdir, 'sensor_eci_Jordan.csv')
    sensor_df.to_csv(csv_name, index=False)  
        
    
    return



if __name__ == '__main__':
    
#    generate_meas_file()
    
    generate_sensor_location_file()
    
    
#    truth_vs_tle()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    