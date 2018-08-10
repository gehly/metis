import numpy as np
from math import pi, asin, atan2
import requests
import getpass
from datetime import datetime, timedelta
import sys
import matplotlib.pyplot as plt

sys.path.append('../')

from utilities.eop_functions import get_celestrak_eop_alldata
from utilities.eop_functions import get_nutation_data
from utilities.eop_functions import get_eop_data
from utilities.coordinate_systems import teme2gcrf
from utilities.coordinate_systems import gcrf2teme
from utilities.coordinate_systems import gcrf2itrf
from utilities.coordinate_systems import itrf2gcrf
from utilities.astrodynamics import element_conversion
from utilities.constants import GME

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


def get_database_tle_data(obj_id_list):
    '''
    This function retrieves the latest two-line element (TLE) data for objects
    in the input list from the database.
    
    '''
    
    tle_dict = {}
    
    
    return tle_dict


def put_database_tle_data(tle_dict):
    '''
    This function puts the latest two-line element (TLE) data for objects
    into the database.
    
    '''
    
    
    return


def launch2tle(obj_id_list, launch_elem_dict):
    '''
    This function converts from commonly used launch elements to TLE format.
    
    Parameters
    ------
    obj_id_list : list
        object NORAD IDs (int)
    launch_elem_dict : dictionary
        dictionary of dictionaries containing launch elements, indexed by
        object ID
        each entry contains rp, ra, i, RAAN, w, M, UTC datetime
    
    Returns
    ------
    tle_dict : dictionary
        Two Line Element information, indexed by object ID
    
    '''

    # Initialize output
    tle_dict = {}
    
    # Loop over objects
    for obj_id in obj_id_list:
        
        # Retrieve launch elements for this object
        launch_elem = launch_elem_dict[obj_id]
        ra = launch_elem['ra']
        rp = launch_elem['rp']
        i = launch_elem['i']
        RAAN = launch_elem['RAAN']
        w = launch_elem['w']
        M = launch_elem['M']
        UTC = launch_elem['UTC']
        
        # Compute mean motion in rev/day
        a = (ra + rp)/2.
        n = np.sqrt(GME/a**3.)
        n *= 86400./(2.*pi)
        
        # Compute eccentricity
        e = 1. - rp/a
        
        # Compute launch year and day of year
        year2 = str(UTC.year)[2:4]
        doy = UTC.timetuple().tm_yday
        dfrac = UTC.hour/24. + UTC.minute/1440. + \
            (UTC.second + UTC.microsecond/1e6)/86400.
        
        # Format for output
        line1 = '1 ' + str(obj_id) + 'U ' + year2 + '001A   ' + year2 + \
            str(doy).zfill(3) + '.' + str(dfrac)[2:10] + \
            '  .00000000  00000-0  00000-0 0    10'
            
        line2 = '2 ' + str(obj_id) + ' ' + '{:8.4f}'.format(i) + ' ' + \
            '{:8.4f}'.format(RAAN) + ' ' + str(e)[2:9] + ' ' + \
            '{:8.4f}'.format(w) + ' ' + '{:8.4f}'.format(M) + ' ' + \
            '{:11.8f}'.format(n) + '    10'
            
        # Add to dictionary
        tle_dict[obj_id] = {}
        tle_dict[obj_id]['line1'] = line1
        tle_dict[obj_id]['line2'] = line2
        
        print(line1)
        print(line2)

    return tle_dict


def kep2tle(obj_id_list, kep_dict):    
    '''
    This function converts from Keplerian orbital elements to TLE format.
    
    Parameters
    ------
    obj_id_list : list
        object NORAD IDs (int)
    kep_dict : dictionary
        dictionary of dictionaries containing launch elements, indexed by
        object ID
        each entry contains a, e, i, RAAN, w, M, UTC datetime
    
    Returns
    ------
    tle_dict : dictionary
        Two Line Element information, indexed by object ID
    
    '''
    
    # Initialize output
    tle_dict = {}
    
    # Loop over objects
    for obj_id in obj_id_list:
        
        # Retrieve launch elements for this object
        kep_elem = kep_dict[obj_id]
        a = kep_elem['a']
        e = kep_elem['e']
        i = kep_elem['i']
        RAAN = kep_elem['RAAN']
        w = kep_elem['w']
        M = kep_elem['M']
        UTC = kep_elem['UTC']
        
        # Compute mean motion in rev/day
        n = np.sqrt(GME/a**3.)
        n *= 86400./(2.*pi)

        # Compute launch year and day of year
        year2 = str(UTC.year)[2:4]
        doy = UTC.timetuple().tm_yday
        dfrac = UTC.hour/24. + UTC.minute/1440. + \
            (UTC.second + UTC.microsecond/1e6)/86400.
        
        # Format for output
        line1 = '1 ' + str(obj_id) + 'U ' + year2 + '001A   ' + year2 + \
            str(doy).zfill(3) + '.' + str(dfrac)[2:10] + \
            '  .00000000  00000-0  00000-0 0    10'
            
        line2 = '2 ' + str(obj_id) + ' ' + '{:8.4f}'.format(i) + ' ' + \
            '{:8.4f}'.format(RAAN) + ' ' + str(e)[2:9] + ' ' + \
            '{:8.4f}'.format(w) + ' ' + '{:8.4f}'.format(M) + ' ' + \
            '{:11.8f}'.format(n) + '    10'
            
        # Add to dictionary
        tle_dict[obj_id] = {}
        tle_dict[obj_id]['line1'] = line1
        tle_dict[obj_id]['line2'] = line2
        
        print(line1)
        print(line2)
    
    
    return tle_dict


def launchecef2tle(obj_id_list, ecef_dict, offline_flag=False):
    '''
    This function converts from ECEF position and velocity to TLE format.
    
    Parameters
    ------
    obj_id_list : list
        object NORAD IDs (int)
    ecef_dict : dictionary
        dictionary of dictionaries containing launch coordinates, indexed by
        object ID
        each entry contains r_ITRF, v_ITRF, UTC datetime
    offline_flag : boolean, optional
        flag to determine whether to retrieve EOP data from internet or from
        a locally saved file (default = False)
    
    Returns
    ------
    tle_dict : dictionary
        Two Line Element information, indexed by object ID
    
    '''
    
     # Initialize output
    tle_dict = {}
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = get_celestrak_eop_alldata(offline_flag)
    
    # Retrieve IAU Nutation data from file
    IAU1980_nutation = get_nutation_data()
    
    # Loop over objects
    for obj_id in obj_id_list:
        
        # Retrieve launch coordinates for this object
        r_ITRF = ecef_dict[obj_id]['r_ITRF']
        v_ITRF = ecef_dict[obj_id]['v_ITRF']
        UTC = ecef_dict[obj_id]['UTC']
        
        # Get EOP data for this time
        EOP_data = get_eop_data(eop_alldata, UTC)
        
        # Convert ITRF to GCRF
        r_GCRF, v_GCRF = itrf2gcrf(r_ITRF, v_ITRF, UTC, EOP_data)
        
        # Convert GCRF to TEME
        r_TEME, v_TEME = gcrf2teme(r_GCRF, v_GCRF, UTC, IAU1980_nutation,
                                   EOP_data)
        
        # Convert TEME to Keplerian elements
        x_in = np.concatenate((r_TEME, v_TEME), axis=0)
        x_out = element_conversion(x_in, 1, 0)

        a = float(x_out[0])
        e = float(x_out[1])
        i = float(x_out[2])
        RAAN = float(x_out[3])
        w = float(x_out[4])
        M = float(x_out[5])
        
        # Compute mean motion in rev/day
        n = np.sqrt(GME/a**3.)
        n *= 86400./(2.*pi)

        # Compute launch year and day of year
        year2 = str(UTC.year)[2:4]
        doy = UTC.timetuple().tm_yday
        dfrac = UTC.hour/24. + UTC.minute/1440. + \
            (UTC.second + UTC.microsecond/1e6)/86400.
        
        # Format for output
        line1 = '1 ' + str(obj_id) + 'U ' + year2 + '001A   ' + year2 + \
            str(doy).zfill(3) + '.' + str(dfrac)[2:10] + \
            '  .00000000  00000-0  00000-0 0    10'
            
        line2 = '2 ' + str(obj_id) + ' ' + '{:8.4f}'.format(i) + ' ' + \
            '{:8.4f}'.format(RAAN) + ' ' + str(e)[2:9] + ' ' + \
            '{:8.4f}'.format(w) + ' ' + '{:8.4f}'.format(M) + ' ' + \
            '{:11.8f}'.format(n) + '    10'
            
        # Add to dictionary
        tle_dict[obj_id] = {}
        tle_dict[obj_id]['line1'] = line1
        tle_dict[obj_id]['line2'] = line2
        
        print(line1)
        print(line2)
    
    
    return tle_dict


def tletime2datetime(line1):
    '''
    This function computes a UTC datetime object from the TLE line1 input year,
    day of year, and fractional day.
    
    Parameters
    ------
    line1 : string
        first line of TLE, contains day of year and fractional day
    
    Returns
    ------
    UTC : datetime object
        UTC datetime object
        
    Reference
    ------
    https://celestrak.com/columns/v04n03/#FAQ03
    
    While talking about the epoch, this is perhaps a good place to answer the
    other time-related questions. First, how is the epoch time format
    interpreted? This question is best answered by using an example. An epoch
    of 98001.00000000 corresponds to 0000 UT on 1998 January 01—in other words,
    midnight between 1997 December 31 and 1998 January 01. An epoch of 
    98000.00000000 would actually correspond to the beginning of 
    1997 December 31—strange as that might seem. Note that the epoch day starts
    at UT midnight (not noon) and that all times are measured mean solar rather 
    than sidereal time units (the answer to our third question).
    
    '''
    
    year2 = line1[18:20]
    doy = float(line1[20:32])  
    
    if int(year2) < 50.:
        year = int('20' + year2)
    else:
        year = int('19' + year2)
    
    # Need to subtract 1 from day of year to add to this base datetime
    # In TLE definition doy = 001.000 for Jan 1 Midnight UTC
    base = datetime(year, 1, 1, 0, 0, 0)
    UTC = base + timedelta(days=(doy-1.))
        
    print(doy)
    print(year)
    print(UTC)
    
    doy = UTC.timetuple().tm_yday
    
    print(doy)
    
    return UTC


def plot_tle_spread(tle_dict, UTC_list=[]):
    '''
    This function propagates a set of TLEs to a common time and plots object
    locations in measurement space.
    
    Parameters
    ------
    tle_dict : dictionary
        Two Line Element information, indexed by object ID

    '''
    
    obj_id_list = sorted(tle_dict.keys())
    
    if len(UTC_list) == 0:
        obj_id = obj_id_list[0]
        line1 = tle_dict[obj_id]['line1']
        UTC = tletime2datetime(line1)
        UTC_list = [UTC]
        
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)    
    
    ra_array = np.zeros((len(obj_id_list), len(UTC_list)))
    dec_array = np.zeros((len(obj_id_list), len(UTC_list)))
    ii = 0
    for obj_id in obj_id_list:
        
        r_GCRF_list = output_state[obj_id]['r_GCRF']
        jj = 0
        for r_GCRF in r_GCRF_list:
            x = float(r_GCRF[0])
            y = float(r_GCRF[1])
            z = float(r_GCRF[2])
            r = np.linalg.norm(r_GCRF)
            
            ra_array[ii,jj] = atan2(y,x)*180/pi
            dec_array[ii,jj] = asin(z/r)*180/pi
            
            jj += 1
        
        ii += 1  
            
    
    jj = 0        
    for UTC in UTC_list:
        
        plt.figure()
        
        plt.scatter(ra_array[:,jj], dec_array[:,jj], marker='o', s=50,
                    c=np.linspace(0,1,len(obj_id_list)),
                    cmap=plt.get_cmap('nipy_spectral'))
        
        for label, x, y in zip(obj_id_list, ra_array[:,jj], dec_array[:,jj]):
            plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
            
        plt.xlabel('Right Ascension [deg]')
        plt.ylabel('Declination [deg]')
        plt.title('TLE Measurement Space ' + UTC.strftime("%Y-%m-%d %H:%M:%S"))

        plt.xlim([-180, 180])
        plt.ylim([-90, 90])
        
        jj += 1
        
    
    
    plt.show()
    
    return


def propagate_TLE(obj_id_list, UTC_list, tle_dict={}, offline_flag=False,
                  username='', password=''):
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
    offline_flag : boolean, optional
        flag to determine whether to retrieve EOP data from internet or from
        a locally saved file (default = False)
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
    
    # If no TLE information is provided, retrieve from sources as needed
    if len(tle_dict) == 0:
    
        # Retrieve latest TLE data from space-track.org
        tle_dict = get_spacetrack_tle_data(obj_id_list, username, password)
        
        # Retreive TLE data from database
        
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = get_celestrak_eop_alldata(offline_flag)
    
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

    
    obj_id_list = [43013, 43014, 43015, 43016]
#    UTC_list = [datetime(2018, 6, 23, 0, 0, 0)]
#    
#    
#    output_state = propagate_TLE(obj_id_list, UTC_list)
#    
#    print(output_state)
    
    plt.close('all')
    
    tle_dict = get_spacetrack_tle_data(obj_id_list)
    plot_tle_spread(tle_dict)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
