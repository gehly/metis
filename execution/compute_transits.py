import numpy as np
from datetime import datetime
import sys
import os
import csv

sys.path.append('../')

from utilities.constants import Re
from sensors.visibility_functions import compute_transit_dict
from sensors.sensors import define_sites_from_file


if __name__ == '__main__':
    
    cwd = os.getcwd()
    metis_ind = cwd.find('metis')
    metis_dir = cwd[0:metis_ind+6]
    
            
    # Object list
    obj_id_list = [25544, 40146]
    
    # Time window
    UTC0 = datetime(2020, 4, 11, 21, 24, 0)
    UTCf = datetime(2020, 4, 11, 21, 40, 0)
    UTC_window = [UTC0, UTCf]
    
    increment = 10.  # seconds
    
    # Site data
    site_data_file = '../input_data/test_sites.json' 
    site_dict = define_sites_from_file(site_data_file)
    

    
    # Generate transit dictionary
    transit_dict = compute_transit_dict(UTC_window, obj_id_list, site_dict,
                                        increment)
    
    
#    print(transit_dict)
    
    
    # Generate output file
    output_file = 'transits.csv'
    with open(output_file, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerow(['Site ID', 'Transit ID', 'Object ID', 'Start [UTC]',
                             'Stop [UTC]', 'Duration [sec]', 'TCA [UTC]', 'TME [UTC]',
                             'Minumum Range [km]', 'Maximum Elevation [deg]',
                             'UTC Times', 'Azimuth [deg]', 'Elevation [deg]',
                             'Range [km]'])
    
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
                UTC_transit = transit_dict[site][transit_id]['UTC_transit']
                az_transit = transit_dict[site][transit_id]['az_transit']
                el_transit = transit_dict[site][transit_id]['el_transit']
                rg_transit = transit_dict[site][transit_id]['rg_transit']
                
                output_row = [site, transit_id, obj_id, start, stop, duration,
                              TCA, TME, rg_min, el_max, UTC_transit,
                              az_transit, el_transit, rg_transit]
                
                filewriter.writerow(output_row)
    
    
    
    
    
    
    