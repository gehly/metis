import numpy as np
from math import *
import pandas as pd
import os
from datetime import datetime


def read_sp3_file(sp3_fname):
    
    
    
    # Open data file for read
    sp3_file = open(sp3_fname, 'r')
    
    # Read lines
    all_lines = sp3_file.readlines()
    sp3_file.close()
    
#    dt_list = []
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
            
            micro, sec = modf(second)
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
            clock = float(line[47:60])
            
            r_ecef = np.reshape([x, y, z], (3,1))
            
            if obj_id not in output_dict:
                output_dict[obj_id] = {}
                output_dict[obj_id]['datetime'] = []
                output_dict[obj_id]['r_ecef'] = []
                output_dict[obj_id]['clock_offset'] = []
                
            output_dict[obj_id]['datetime'].append(dt)
            output_dict[obj_id]['r_ecef'].append(r_ecef)
            output_dict[obj_id]['clock_offset'].append(clock)

    
    return output_dict



if __name__ == '__main__':
    
    
    fdir = 'D:\documents\\teaching\\unsw_ssa_undergrad\lab\\telescope\\truth_data'
    fname = 'gbm20692.sp3'
    sp3_fname = os.path.join(fdir, fname)
    
    sp3_dict = read_sp3_file(sp3_fname)
    
    
    print(sp3_dict['PJ07'])
    
    
    