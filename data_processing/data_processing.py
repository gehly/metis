import numpy as np
import math
from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)


import utilities.coordinate_systems as coord
import utilities.eop_functions as eop
import utilities.time_systems as timesys
import utilities.tle_functions as tle



def read_sp3_file(sp3_fname):
    '''
    This function reads an SP3 data file to retrieve precise orbit data for
    GNSS (or other) satellites. Data is reformatted into a dictionary for
    output.
    
    Parameters
    ------
    sp3_fname : string
        path and filename of SP3 data file to read
        
    Returns
    ------
    output_dict : dict
        dictionary indexed by object id (from SP3 file, not NORAD ID) with
        GPS times, ECEF positions, and Clock Offsets

    '''
    
    # Open data file for read
    sp3_file = open(sp3_fname, 'r')
    
    # Read lines
    all_lines = sp3_file.readlines()
    sp3_file.close()

    output_dict = {}
    for ii in range(len(all_lines)):
        
        line = all_lines[ii]
        
        # Save times
        if line[0] == '*':
            year = int(line[3:7])
            month = int(line[8:10])
            day = int(line[11:13])
            hour = int(line[14:16])
            minute = int(line[17:19])
            second = float(line[20:31])
            
            micro, sec = math.modf(second)
            micro = int(micro * 1000000)
            sec = int(sec)
        
        
#            print(year)
#            print(month)
#            print(day)
#            print(hour)
#            print(minute)
#            print(sec)
#            print(micro)
            
            dt = datetime(year, month, day, hour, minute, sec, micro)
#            dt_list.append(dt)
            
            
        # Save object position data
        if line[0] == 'P':
            obj_id = line[0:4]
            x = float(line[5:18])
            y = float(line[19:32])
            z = float(line[33:46])
            clock = float(line[47:60])/1e6   # convert to seconds
            
            r_ecef = np.reshape([x, y, z], (3,1))
            
            if obj_id not in output_dict:
                output_dict[obj_id] = {}
                output_dict[obj_id]['gps_time'] = []
                output_dict[obj_id]['r_ecef'] = []
                output_dict[obj_id]['clock_offset_sec'] = []
                
            output_dict[obj_id]['gps_time'].append(dt)
            output_dict[obj_id]['r_ecef'].append(r_ecef)
            output_dict[obj_id]['clock_offset_sec'].append(clock)

    
    return output_dict


def unit_test_sp3_reader():
    
    fdir = 'unit_test'
    fname = 'qzf22081.sp3'
    sp3_fname = os.path.join(fdir, fname)
    
    sp3_dict = read_sp3_file(sp3_fname)
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    qzs1_norad = 37158
    qzs2_norad = 42738
    qzs3_norad = 42917
    qzs4_norad = 42965
    qzs1r_norad = 49336
    
    # These mappings produce errors on the km level
    # QZS-1 appears to be retired as of 2022
    qzs1r_sp3_id = 'PJ04'
    qzs2_sp3_id = 'PJ02'    
    qzs3_sp3_id = 'PJ07'    
    qzs4_sp3_id = 'PJ03'    
    
    norad_id = qzs4_norad
    sp3_id = qzs4_sp3_id
    
    # Convert SP3 data to UTC and ECI
    gps_list = sp3_dict[sp3_id]['gps_time']
    ecef_list = sp3_dict[sp3_id]['r_ecef']
    clock_list = sp3_dict[sp3_id]['clock_offset_sec']
    UTC_list = []
    ECI_list = []
    for ii in range(len(gps_list)):
        
        gps_time = gps_list[ii]
        sp3_ecef = ecef_list[ii]
        EOP_data = eop.get_eop_data(eop_alldata, gps_time)
        
        # Convert to UTC
        UTC = timesys.gpsdt2utcdt(gps_time, EOP_data['TAI_UTC'])
        UTC_list.append(UTC) # + timedelta(seconds=clock)
        
        # Convert to GCRF
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        sp3_eci, dum = coord.itrf2gcrf(sp3_ecef, np.zeros((3,1)), UTC, EOP_data, XYs_df)
        ECI_list.append(sp3_eci)
        
    # Retrieve and propagate TLE data to desired times
    tle_state = tle.propagate_TLE([norad_id], UTC_list)
    
    # Compute errors at each time
    thrs = []
    ric_err = np.zeros((3, len(UTC_list)))
    for ii in range(len(UTC_list)):
        
        UTC = UTC_list[ii]
        thrs.append((UTC - UTC_list[0]).total_seconds()/3600.)
        
        sp3_r_eci = ECI_list[ii].reshape(3,1)
        tle_r_eci = tle_state[norad_id]['r_GCRF'][ii].reshape(3,1)
        tle_v_eci = tle_state[norad_id]['v_GCRF'][ii].reshape(3,1)
        
        # Compute RIC errors with TLE data acting as chief satellite
        rho_eci = sp3_r_eci - tle_r_eci    
        rho_ric = coord.eci2ric(tle_r_eci, tle_v_eci, rho_eci)
        
        # Change sign to set SP3 data as chief (truth)
        rho_ric = -rho_ric      
        
        # Store output
        ric_err[:,ii] = rho_ric.flatten()
        
    # Generate plots
    plt.figure()  
    plt.subplot(3,1,1)
    plt.plot(thrs, ric_err[0,:], 'k.')
    plt.ylabel('Radial Error [km]')
    plt.title('RIC Error TLE vs SP3')
    plt.subplot(3,1,2)
    plt.plot(thrs, ric_err[1,:], 'k.')
    plt.ylabel('In-Track Error [km]')
    plt.subplot(3,1,3)
    plt.plot(thrs, ric_err[2,:], 'k.')
    plt.ylabel('Cross-Track Error [km]')
    plt.xlabel('Time Since ' + UTC_list[0].strftime('%Y-%m-%d %H:%M:%S') + ' [hours]')
    
    plt.show()
    
    
        
    
    
    
    
    
    
    return



if __name__ == '__main__':
    
    plt.close('all')
    
    unit_test_sp3_reader()
    
    
    
    