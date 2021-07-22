import numpy as np
from datetime import datetime
import sys
import os
import csv
import pickle

sys.path.append('../')

from utilities.constants import Re
from sensors.visibility_functions import compute_transit_dict
from sensors.sensors import define_sites_from_file


if __name__ == '__main__':
    
    cwd = os.getcwd()
    metis_ind = cwd.find('metis')
    metis_dir = cwd[0:metis_ind+6]
    
            
    # Object list
    obj_id_list = list(np.arange(43010, 43020))
    obj_id_list.append(40146)
    obj_id_list.append(40148)
    
    # Time window
    UTC0 = datetime(2019, 1, 1, 0, 0, 0)
#    UTCf = datetime(2019, 1, 1, 4, 0, 0)
    UTCf = datetime(2019, 1, 10, 0, 0, 0)
    UTC_window = [UTC0, UTCf]
    
    increment = 10.  # seconds
    
    # Site data
    site_data_file = '../input_data/5grid-sites.json' 
    site_dict = define_sites_from_file(site_data_file)
    

    
    # Generate transit dictionary
    transit_dict = compute_transit_dict(UTC_window, obj_id_list, site_dict,
                                        increment, offline_flag=True)
    
    
#    print(transit_dict)
    
    
    # Generate output file
    output_file = 'transits2.csv'
    with open(output_file, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerow(['Site_ID', 'Transit_ID', 'Object_ID', 'Start_UTC',
                             'Stop_UTC', 'Duration_sec', 'TCA_UTC', 'TME_UTC',
                             'Minumum_Range_km', 'Maximum_Elevation_deg'])
                             #'UTC_Times', 'Azimuth [deg]', 'Elevation [deg]',
                             #'Range [km]'])
    
        for site in transit_dict:
            for transit_id in transit_dict[site]:
                
                obj_id = transit_dict[site][transit_id]['NORAD_ID']
                start = transit_dict[site][transit_id]['start']
                stop = transit_dict[site][transit_id]['stop']
                duration = transit_dict[site][transit_id]['duration']
                TCA = transit_dict[site][transit_id]['TCA']
                TME = transit_dict[site][transit_id]['TME']
                rg_min = transit_dict[site][transit_id]['rg_min']
                el_max = transit_dict[site][transit_id]['el_max']
#                UTC_transit = transit_dict[site][transit_id]['UTC_transit']
#                az_transit = transit_dict[site][transit_id]['az_transit']
#                el_transit = transit_dict[site][transit_id]['el_transit']
#                rg_transit = transit_dict[site][transit_id]['rg_transit']
                
                output_row = [site, transit_id, obj_id, start, stop, duration,
                              TCA, TME, rg_min, el_max,] # UTC_transit,
                              # az_transit, el_transit, rg_transit]
                
                filewriter.writerow(output_row)
    
    
    
    
    
    
    