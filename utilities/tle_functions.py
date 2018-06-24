import numpy as np
import requests
import getpass
from datetime import datetime

from eop_functions import get_celestrak_eop_alldata
from eop_functions import get_nutation_data
from eop_functions import get_eop_data
from coordinate_systems import teme2gcrf
from coordinate_systems import gcrf2itrf

from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs84


###############################################################################
#
# This file contains functions to retrieve and propagate TLEs and convert
# position coordinates to the intertial GCRF frame.
#
###############################################################################


def get_spacetrack_tle_data(obj_id_list, username='', password=''):
    '''
    This function retrieves the latest two-line element (TLE) data for objects
    in the input list from space-track.org.
    
    Parameters
    ------
    obj_id_list : list
        object NORAD IDs (int)
    username : string, optional
        space-track.org username (code will prompt for input if not supplied)
    password : string, optional
        space-track.org password (code will prompt for input if not supplied)
    
    Returns
    ------
    tle_dict : dictionary
        indexed by object ID, each item has two strings, one for each line
    '''
    
    if len(username) == 0:    
        username = input('space-track username: ')
    if len(password) == 0:
        password = getpass.getpass('space-track password: ')    
    
    myString = ",".join(map(str, obj_id_list))
    
    pageData = ('//www.space-track.org/basicspacedata/query/class/tle_latest/'
                'ORDINAL/1/NORAD_CAT_ID/' + myString + '/orderby/'
                'TLE_LINE1 ASC/format/tle')
    payload = {'identity':username, 'password':password, 'submit':'Login'}
    
    with requests.Session() as s:
        s.post("https://www.space-track.org/auth/login", data=payload)
        r = s.get('https:' + pageData)
        if r.status_code != requests.codes.ok:
            print("Error: Page data request failed.")
            
    # Parse response and form output
    tle_dict = {}
    nchar = 69
    nskip = 2
    ii = 0  
    for obj_id in obj_id_list:
        line1_start = ii*2*(nchar+nskip)
        line1_stop = ii*2*(nchar+nskip) + nchar
        line2_start = ii*2*(nchar+nskip) + nchar + nskip
        line2_stop = ii*2*(nchar+nskip) + 2*nchar + nskip
        line1 = r.text[line1_start:line1_stop]
        line2 = r.text[line2_start:line2_stop]

        tle_dict[obj_id] = {}
        tle_dict[obj_id]['line1'] = line1
        tle_dict[obj_id]['line2'] = line2
        
        ii += 1
    
    return tle_dict


def propagate_TLE(obj_id_list, UTC_list, username='', password=''):
    '''
    This function retrieves TLE data for the objects in the input list from
    space-track.org and propagates them to the times given in UTC_list.  The
    output positions and velocities are provided in both the TLE True Equator
    Mean Equinox (TEME) frame and the inertial GCRF frame.
    
    Parameters
    ------
    obj_id_list : list
        object NORAD IDs
    UTC_list : list
        datetime objects in UTC
    username : string, optional
        space-track.org username (code will prompt for input if not supplied)
    password : string, optional
        space-track.org password (code will prompt for input if not supplied)
        
    Returns
    ------
    output_state : dictionary
        indexed by object ID, contains lists for UTC times, and object 
        position/velocity in TEME and GCRF
        
    '''
    
    # Retrieve latest TLE data from space-track.org
    tle_dict = get_spacetrack_tle_data(obj_id_list, username, password)
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = get_celestrak_eop_alldata()
    
    # Retrieve IAU Nutation data from file
    IAU1980_nutation = get_nutation_data()
    
    # Loop over objects
    output_state = {}
    for obj_id in obj_id_list:
        
        line1 = tle_dict[obj_id]['line1']
        line2 = tle_dict[obj_id]['line2']
        
        output_state[obj_id] = {}
        output_state[obj_id]['UTC'] = []
        output_state[obj_id]['r_GCRF'] = []
        output_state[obj_id]['v_GCRF'] = []
        output_state[obj_id]['r_TEME'] = []
        output_state[obj_id]['v_TEME'] = []
        
        # Loop over times
        for UTC in UTC_list:
            
            # Propagate TLE using SGP4
            satellite = twoline2rv(line1, line2, wgs84)
            r_TEME, v_TEME = satellite.propagate(UTC.year, UTC.month, UTC.day,
                                                 UTC.hour, UTC.minute,
                                                 UTC.second + 
                                                 (UTC.microsecond/1e6))
            
            r_TEME = np.reshape(r_TEME, (3,1))
            v_TEME = np.reshape(v_TEME, (3,1))
            
            # Get EOP data for this time
            EOP_data = get_eop_data(eop_alldata, UTC)
            
            # Convert from TEME to GCRF (ECI)
            r_GCRF, v_GCRF = teme2gcrf(r_TEME, v_TEME, UTC, IAU1980_nutation,
                                       EOP_data)

            
            # Store output
            output_state[obj_id]['UTC'].append(UTC)
            output_state[obj_id]['r_GCRF'].append(r_GCRF)
            output_state[obj_id]['v_GCRF'].append(v_GCRF)
            output_state[obj_id]['r_TEME'].append(r_TEME)
            output_state[obj_id]['v_TEME'].append(v_TEME)

    
    return output_state





###############################################################################
# Stand-alone execution
###############################################################################

if __name__ == '__main__' :

    
    obj_id_list = [43014]
    UTC_list = [datetime(2018, 6, 23, 0, 0, 0)]
    
    
    output_state = propagate_TLE(obj_id_list, UTC_list)
    
    print(output_state)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
