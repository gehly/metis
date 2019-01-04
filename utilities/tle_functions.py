import numpy as np
from math import pi, asin, atan2
import requests
import getpass
from datetime import datetime, timedelta
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import json

sys.path.append('../')

from sensors.sensors import define_sensors
from utilities.eop_functions import get_celestrak_eop_alldata
from utilities.eop_functions import get_nutation_data
from utilities.eop_functions import get_eop_data
from utilities.coordinate_systems import teme2gcrf
from utilities.coordinate_systems import gcrf2teme
from utilities.coordinate_systems import gcrf2itrf
from utilities.coordinate_systems import itrf2gcrf
from utilities.coordinate_systems import latlonht2ecef
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


def get_spacetrack_tle_data(obj_id_list = [], UTC_list = [], username='',
                            password=''):
    '''
    This function retrieves the two-line element (TLE) data for objects
    in the input list from space-track.org.
    
    Parameters
    ------
    obj_id_list : list, optional
        object NORAD IDs (int)
        - if empty code will retrieve latest available for entire catalog
    UTC_list : list, optional
        UTC datetime objects to specify desired times for TLEs to retrieve
        - if empty code will retrieve latest available
        - if 1 entry, code will retrieve all TLEs in the following 1 day window
        - if 2 entries, code will retrieve all TLEs between first and second time
          (default = empty)
    username : string, optional
        space-track.org username (code will prompt for input if not supplied)
    password : string, optional
        space-track.org password (code will prompt for input if not supplied)
    
    Returns
    ------
    tle_dict : dictionary
        indexed by object ID, each item has two lists of strings for each line
    tle_df : pandas dataframe
        norad, tle line1, tle line2
    '''
    
    if len(username) == 0:    
        username = input('space-track username: ')
    if len(password) == 0:
        password = getpass.getpass('space-track password: ')    
    
    if len(obj_id_list) >= 1:
        myString = ",".join(map(str, obj_id_list))
        
        # If only one time is given, add 1 day increment to produce window
        if len(UTC_list) ==  1:
            UTC_list.append(UTC_list[0] + timedelta(days=1.))
        
        # If times are specified, retrieve from window
        if len(UTC_list) == 2:        
            UTC0 = UTC_list[0].strftime('%Y-%m-%d')
            UTC1 = UTC_list[1].strftime('%Y-%m-%d')
            pageData = ('//www.space-track.org/basicspacedata/query/class/tle/'
                        'EPOCH/' + UTC0 + '--' + UTC1 + '/NORAD_CAT_ID/' + 
                        myString + '/orderby/TLE_LINE1 ASC/format/tle')
            
        # Otherwise, get latest available
        else:    
            pageData = ('//www.space-track.org/basicspacedata/query/class/'
                        'tle_latest/ORDINAL/1/NORAD_CAT_ID/' + myString + 
                        '/orderby/TLE_LINE1 ASC/format/tle')
    else:
        pageData = '//www.space-track.org/basicspacedata/query/class/tle_latest/ORDINAL/1/EPOCH/%3Enow-30/orderby/NORAD_CAT_ID/format/tle'
  
    ST_URL='https://www.space-track.org'
    
    with requests.Session() as s:
        s.post(ST_URL+"/ajaxauth/login", json={'identity':username, 'password':password})
        r = s.get('https:' + pageData)
        if r.status_code != requests.codes.ok:
            print("Error: Page data request failed.")
            
    # Parse response and form output
    tle_dict = {}
    tle_list = []
    nchar = 69
    nskip = 2
    ntle = int(round(len(r.text)/142.))
    for ii in range(ntle):
        
        line1_start = ii*2*(nchar+nskip)
        line1_stop = ii*2*(nchar+nskip) + nchar
        line2_start = ii*2*(nchar+nskip) + nchar + nskip
        line2_stop = ii*2*(nchar+nskip) + 2*nchar + nskip
        line1 = r.text[line1_start:line1_stop]
        line2 = r.text[line2_start:line2_stop]
        UTC = tletime2datetime(line1)

        obj_id = int(line1[2:7])
        
        if obj_id not in tle_dict:
            tle_dict[obj_id] = {}
            tle_dict[obj_id]['UTC_list'] = []
            tle_dict[obj_id]['line1_list'] = []
            tle_dict[obj_id]['line2_list'] = []
        
        tle_dict[obj_id]['UTC_list'].append(UTC)
        tle_dict[obj_id]['line1_list'].append(line1)
        tle_dict[obj_id]['line2_list'].append(line2)
        
        linelist = [obj_id,line1,line2,UTC]
        tle_list.append(linelist)  
        
    tle_df = pd.DataFrame(tle_list, columns=['norad','line1','line2','utc'])
    
    return tle_dict, tle_df


def get_tle_range(username='', password='', start_norad='', stop_norad=''):
    '''
    This function retrieves the "tle_latest" Class from space-track.org for
    objects defined by a norad range.
    
    Parameters
    ------
    username : string, optional
        space-track.org username (code will prompt for input if not supplied)
    password : string, optional
        space-track.org password (code will prompt for input if not supplied)
    start_norad, stop_norad: used to define the range of NORAD IDs to query
   
    Returns
    ------
    tle_df : pandas dataframe
        APOGEE,ARG_OF_PERICENTER,BSTAR, CLASSIFICATION_TYPE, COMMENT,
        ECCENTRICITY, ELEMENT_SET_NO, EPHEMERIS_TYPE, EPOCH, EPOCH_MICROSECONDS,
        E, INCLINATION,	INTLDES, MEAN_ANOMALY, MEAN_MOTION, MEAN_MOTION_DDOT,
        MEAN_MOTION_DOT, NORAD_CAT_ID, OBJECT_ID, OBJECT_NAME, OBJECT_NUMBER,
        ,OBJECT_TYPE, ORDINAL, ORIGINATOR, PERIGEE, PERIOD, RA_OF_ASC_NODE,
        REV_AT_EPOCH, SEMIMAJOR_AXIS,	TLE_LINE0, TLE_LINE1, TLE_LINE2
    '''
    if len(username) == 0:
            username = input('space-track username: ')
    if len(password) == 0:
            password = getpass.getpass('space-track password: ')
    url = 'https://www.space-track.org/basicspacedata/query/class/tle_latest/ORDINAL/1/NORAD_CAT_ID/' + start_norad +'--' + stop_norad +'/orderby/NORAD_CAT_ID%20asc/emptyresult/show'
    login_url ='https://www.space-track.org'
    with requests.Session() as s:
            s.post(login_url+"/ajaxauth/login", json={'identity':username, 'password':password})
            response = s.get(url)
            response_dict = json.loads(response.text)
            if response.status_code != requests.codes.ok:
                print("Error: Page data request failed.")
            tle_df = pd.DataFrame(response_dict)
            return tle_df


def tledict2dataframe(tle_dict):
    '''
    This function computes a pandas dataframe with TLE data given an input
    dictionary with TLE data.
    
    Parameters
    ------
    tle_dict : dictionary
        indexed by object ID, each item has two lists of strings for each line
        
    Returns
    ------
    tle_df : pandas dataframe
        norad, tle line1, tle line2
        
    '''
    
    tle_list = []
    for obj_id in tle_dict:
        for ii in range(len(tle_dict[obj_id]['line1_list'])):
            line1 = tle_dict[obj_id]['line1_list'][ii]
            line2 = tle_dict[obj_id]['line2_list'][ii]
            linelist = [obj_id, line1, line2]
            tle_list.append(linelist)
    
    tle_df = pd.DataFrame(tle_list, columns=['norad','line1','line2'])
    
    return tle_df


def csvstack2tledict(fdir, obj_id_list):
    '''
    This function computes a dictionary of unique TLE information given a 
    stack of CSV files in a directory. The function assumes the directory
    contains only CSV files with TLE data.
    
    Parameters
    ------
    fdir : string
        file directory containing CSV files
    obj_id_list : list
        NORAD IDs (int) of objects to collect TLE data
    
    Returns
    ------
    tle_dict : dictionary
        indexed by object ID, each item has two lists of strings for each line
        as well as object name and UTC times
    
    '''
    
    # Initialize output
    tle_dict = {}
    
    # Loop over each file in directory
    for fname in os.listdir(fdir):
        
        fname = os.path.join(fdir, fname)
        df = pd.read_csv(fname)
        norad_list = df['NORAD_CAT_ID'].tolist()

        # Loop over objects and retrieve TLE data
        for obj_id in obj_id_list:
            if obj_id in norad_list:

                # Retrieve object name and TLE data
                ind = norad_list.index(obj_id)
                name = df.at[ind, 'OBJECT_NAME']
                line1 = df.at[ind, 'TLE_LINE1']
                line2 = df.at[ind, 'TLE_LINE2']
                
                # Compute UTC time
                UTC = tletime2datetime(line1)
                
                # If first occurence of this object, initialize output
                if obj_id not in tle_dict:
                    tle_dict[obj_id] = {}
                    tle_dict[obj_id]['name_list'] = [name]
                    tle_dict[obj_id]['UTC_list'] = [UTC]
                    tle_dict[obj_id]['line1_list'] = [line1]
                    tle_dict[obj_id]['line2_list'] = [line2]
                    
                # Otherwise, check if different from previous entry
                # Only add if different
                else:
                    if ((name != tle_dict[obj_id]['name_list'][-1]) or 
                        (line1 != tle_dict[obj_id]['line1_list'][-1]) or 
                        (line2 != tle_dict[obj_id]['line2_list'][-1])):

                        # Append to output
                        tle_dict[obj_id]['name_list'].append(name)
                        tle_dict[obj_id]['UTC_list'].append(UTC)
                        tle_dict[obj_id]['line1_list'].append(line1)
                        tle_dict[obj_id]['line2_list'].append(line2)                
        
    return tle_dict


def compute_tle_allstate(tle_dict):
    '''
    This function computes a dictionary of state vectors at common times for
    all objects in the input TLE dictionary.
    
    Parameters
    ------
    tle_dict : dictionary
        indexed by object ID, each item has two lists of strings for each line
        as well as object name and UTC times
        
    Returns
    ------
    output : dictionary
        indexed by UTC time, each item has a list of object NORAD IDs, names,
        and pos/vel vectors in GCRF at that time
    
    '''
    
    # Initialize output
    output = {}
    
    # Get unique list of UTC times in order
    UTC_list = []
    for obj_id in tle_dict:
        UTC_list.extend(tle_dict[obj_id]['UTC_list'])
    UTC_list = sorted(list(set(UTC_list)))
    
    # For each time in list, propagate all objects with a TLE at or before that
    # time to current UTC
    for UTC in UTC_list:
        
        print(UTC)
        print('Index: ', UTC_list.index(UTC), ' of ', len(UTC_list), '\n')
        
        # Create reduced obj_id_list with only objects that exist at this time
        obj_id_red = []
        for obj_id in tle_dict:
            if tle_dict[obj_id]['UTC_list'][0] <= UTC:
                obj_id_red.append(obj_id)
                
        # Propagate objects to common time
        state = propagate_TLE(obj_id_red, [UTC], tle_dict)
    
        # Store output
        output[UTC] = {}
        output[UTC]['obj_id_list'] = obj_id_red
        output[UTC]['name_list'] = []
        output[UTC]['r_GCRF'] = []
        output[UTC]['v_GCRF'] = []
        
        for obj_id in obj_id_red:
            ind = [ii for ii in range(len(tle_dict[obj_id]['UTC_list'])) if tle_dict[obj_id]['UTC_list'][ii] <= UTC][-1]
            
            output[UTC]['name_list'].append(tle_dict[obj_id]['name_list'][ind])
            output[UTC]['r_GCRF'].append(state[obj_id]['r_GCRF'][0])
            output[UTC]['v_GCRF'].append(state[obj_id]['v_GCRF'][0])    
        

    return output


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
    
    # Get 2 digit year and day of year
    year2 = line1[18:20]
    doy = float(line1[20:32])  
    
    # Compute century and add to year
    if int(year2) < 50.:
        year = int('20' + year2)
    else:
        year = int('19' + year2)
    
    # Need to subtract 1 from day of year to add to this base datetime
    # In TLE definition doy = 001.000 for Jan 1 Midnight UTC
    base = datetime(year, 1, 1, 0, 0, 0)
    UTC = base + timedelta(days=(doy-1.))
    
    # Compute day of year to check
    doy = UTC.timetuple().tm_yday
    
    return UTC


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
    tle_list = []
    
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
        
        # Compute GCRF position and velocity
        x_in = [a,e,i,RAAN,w,M]
        x_out = element_conversion(x_in, 0, 1)
        r_GCRF = np.reshape(x_out[0:3], (3,1))
        v_GCRF = np.reshape(x_out[3:6], (3,1))
        
        # Compute TLE data for this object
        line1, line2 = gcrf2tle(obj_id, r_GCRF, v_GCRF, UTC)
    
        # Add to dictionary
        tle_dict[obj_id] = {}
        tle_dict[obj_id]['UTC_list'] = [UTC]
        tle_dict[obj_id]['line1_list'] = [line1]
        tle_dict[obj_id]['line2_list'] = [line2]
        
        # Generate dataframe output
        linelist = [obj_id,line1,line2]
        tle_list.append(linelist)
        
    tle_df = pd.DataFrame(tle_list, columns=['norad','line1','line2'])

    return tle_dict, tle_df


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
        indexed by object ID, each item has two lists of strings for each line
    tle_df : pandas dataframe
        norad, tle line1, tle line2

    '''
    
    # Initialize output
    tle_dict = {}
    tle_list = []
    
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
        
        # Compute GCRF position/velocity
        x_in = [a,e,i,RAAN,w,M]
        x_out = element_conversion(x_in, 0, 1)
        r_GCRF = np.reshape(x_out[0:3], (3,1))
        v_GCRF = np.reshape(x_out[3:6], (3,1))
        
        # Compute TLE data for this object
        line1, line2 = gcrf2tle(obj_id, r_GCRF, v_GCRF, UTC)
    
        # Add to dictionary
        tle_dict[obj_id] = {}
        tle_dict[obj_id]['UTC_list'] = [UTC]
        tle_dict[obj_id]['line1_list'] = [line1]
        tle_dict[obj_id]['line2_list'] = [line2]
        
        # Generate dataframe output
        linelist = [obj_id,line1,line2]
        tle_list.append(linelist)
        
    tle_df = pd.DataFrame(tle_list, columns=['norad','line1','line2'])
    
    return tle_dict, tle_df


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
        indexed by object ID, each item has two lists of strings for each line
    tle_df : pandas dataframe
        norad, tle line1, tle line2    
    
    '''
    
     # Initialize output
    tle_dict = {}
    tle_list = []
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = get_celestrak_eop_alldata(offline_flag)
    
    # Retrieve IAU Nutation data from file
    IAU1980nut = get_nutation_data()
    
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
        
        # Compute TLE data for this object
        line1, line2 = gcrf2tle(obj_id, r_GCRF, v_GCRF, UTC, EOP_data,
                                    IAU1980nut)
    
        # Add to dictionary
        tle_dict[obj_id] = {}
        tle_dict[obj_id]['UTC_list'] = [UTC]
        tle_dict[obj_id]['line1_list'] = [line1]
        tle_dict[obj_id]['line2_list'] = [line2]
        
        # Generate dataframe output
        linelist = [obj_id,line1,line2]
        tle_list.append(linelist)
        
    tle_df = pd.DataFrame(tle_list, columns=['norad','line1','line2'])
    
    return tle_dict, tle_df


def gcrf2tle(obj_id, r_GCRF, v_GCRF, UTC, EOP_data=[], IAU1980nut=[],
             offline_flag=False):
    '''
    This function generates Two Line Element (TLE) data in proper format given
    input position and velocity in ECI (GCRF) coordinates.
    
    Parameters
    ------
    obj_id : int
        NORAD ID of object
    r_GCRF : 3x1 numpy array
        position in GCRF [km]
    v_GCRF : 3x1 numpy array
        velocity in GCRF [km/s]
    UTC : datetime object
        epoch time of pos/vel state and TLE in UTC
    EOP_data : list, optional
        Earth orientation parameters, if empty will download from celestrak
        (default=[])
    IAU1980nut : dataframe, optional
        nutation parameters, if empty will load from file (default=[])
    offline_flag : boolean, optional
        flag to indicate internet access, if True will load data from local
        files (default=False)
        
    Returns
    ------
    line1 : string
        first line of TLE
    line2 : string
        second line of TLE
    '''
    
    # Retrieve latest EOP data from celestrak.com, if needed
    if len(EOP_data) == 0:        
        
        eop_alldata = get_celestrak_eop_alldata(offline_flag)
        EOP_data = get_eop_data(eop_alldata, UTC)
    
    # Retrieve IAU Nutation data from file, if needed
    if len(IAU1980nut) == 0:
        IAU1980nut = get_nutation_data()    
    
    # Convert to TEME, recompute osculating elements
    r_TEME, v_TEME = gcrf2teme(r_GCRF, v_GCRF, UTC, IAU1980nut, EOP_data)
    x_in = np.concatenate((r_TEME, v_TEME), axis=0)
    osc_elem = element_conversion(x_in, 1, 0)
    
    print(osc_elem)
    
    # Convert to mean elements
    # TODO currently it appears osculating elements gives more accurate result
    # Need further investigation of proper computation of TLEs.
    # Temporary solution, just use osculating elements instead of mean elements
    mean_elem = list(osc_elem.flatten())
    
    # Retrieve elements
    a = float(mean_elem[0])
    e = float(mean_elem[1])
    i = float(mean_elem[2])
    RAAN = float(mean_elem[3])
    w = float(mean_elem[4])
    M = float(mean_elem[5])
    
    e = '{0:.10f}'.format(e)
    
    # Compute mean motion in rev/day
    n = np.sqrt(GME/a**3.)
    n *= 86400./(2.*pi)
    
    # Compute launch year and day of year
    year2 = str(UTC.year)[2:4]
    doy = UTC.timetuple().tm_yday
    dfrac = UTC.hour/24. + UTC.minute/1440. + \
        (UTC.second + UTC.microsecond/1e6)/86400.
    dfrac = '{0:.15f}'.format(dfrac)
    
    # Format for output
    line1 = '1 ' + str(obj_id) + 'U ' + year2 + '001A   ' + year2 + \
        str(doy).zfill(3) + '.' + str(dfrac)[2:10] + \
        '  .00000000  00000-0  00000-0 0    10'
        
    line2 = '2 ' + str(obj_id) + ' ' + '{:8.4f}'.format(i) + ' ' + \
        '{:8.4f}'.format(RAAN) + ' ' + e[2:9] + ' ' + \
        '{:8.4f}'.format(w) + ' ' + '{:8.4f}'.format(M) + ' ' + \
        '{:11.8f}'.format(n) + '    10'

    return line1, line2


def plot_tle_radec(tle_dict, UTC_list=[], sensor_list=[], display_flag=False,
                   offline_flag=False):
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
        line1 = tle_dict[obj_id]['line1_list'][0]
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
            
        plt.xlabel('Geocentric Right Ascension [deg]')
        plt.ylabel('Geocentric Declination [deg]')
        plt.title('TLE Measurement Space ' + UTC.strftime("%Y-%m-%d %H:%M:%S"))

        plt.xlim([-180, 180])
        plt.ylim([-90, 90])
        
        jj += 1
    
    # If sensor location specified, compute and plot topocentric coordinates
    if len(sensor_list) > 0:
        
        # Retrieve latest EOP data from celestrak.com
        eop_alldata = get_celestrak_eop_alldata(offline_flag)
        
        # Retrive sensor parameters and loop over sensors
        sensor_dict = define_sensors(sensor_list)        
        for sensor_id in sensor_list:
            sensor = sensor_dict[sensor_id]
            latlonht = sensor['geodetic_latlonht']
            lat = latlonht[0]
            lon = latlonht[1]
            ht = latlonht[2]
            sensor_ecef = latlonht2ecef(lat, lon, ht)
            
            center = [85.82990416666667, 5.990788888888889]
            FOV_hlim = [lim*180/pi for lim in sensor['FOV_hlim']]
            FOV_vlim = [lim*180/pi for lim in sensor['FOV_vlim']]
            FOVh = [center[0] + FOV_hlim[0], center[0] + FOV_hlim[1]]
            FOVv = [center[1] + FOV_vlim[0], center[1] + FOV_vlim[1]]
            
            topo_ra_array = np.zeros((len(obj_id_list), len(UTC_list)))
            topo_dec_array = np.zeros((len(obj_id_list), len(UTC_list)))
            ii = 0
            for obj_id in obj_id_list:
                
                r_GCRF_list = output_state[obj_id]['r_GCRF']
                
                jj = 0
                for r_GCRF in r_GCRF_list:
                    x = float(r_GCRF[0])
                    y = float(r_GCRF[1])
                    z = float(r_GCRF[2])
                    r = np.linalg.norm(r_GCRF)
                    
                    UTC = UTC_list[jj]
                    EOP_data = get_eop_data(eop_alldata, UTC)
                    sensor_eci, dum = itrf2gcrf(sensor_ecef, np.zeros((3,1)),
                                                UTC, EOP_data)
                    
                    xs = float(sensor_eci[0])
                    ys = float(sensor_eci[1])
                    zs = float(sensor_eci[2])
                    rg = np.linalg.norm(r_GCRF - sensor_eci)
                    
                    topo_ra_array[ii,jj] = atan2((y-ys),(x-xs))*180/pi
                    topo_dec_array[ii,jj] = asin((z-zs)/rg)*180/pi
                    
                    print(obj_id)
                    print(topo_ra_array)
                    print(topo_dec_array)
                    
                    jj += 1
                
                ii += 1
                
            kk = 0        
            for UTC in UTC_list:
                plt.figure()
                
                plt.plot(FOVh, [FOVv[0], FOVv[0]], 'k--', lw=2)
                plt.plot(FOVh, [FOVv[1], FOVv[1]], 'k--', lw=2)
                plt.plot([FOVh[0], FOVh[0]], FOVv, 'k--', lw=2)
                plt.plot([FOVh[1], FOVh[1]], FOVv, 'k--', lw=2)
            
                plt.scatter(topo_ra_array[:,kk], topo_dec_array[:,kk], marker='o', s=50,
                            c=np.linspace(0,1,len(obj_id_list)),
                            cmap=plt.get_cmap('nipy_spectral'))
                
                for label, x, y in zip(obj_id_list, topo_ra_array[:,kk], topo_dec_array[:,kk]):
                    
                    if label == 39613 or label == 40267:
                        xytext1 = (0, -30)
                    else:
                        xytext1 = (-20, 20)
                    
                    plt.annotate(
                    label,
                    xy=(x, y), xytext=xytext1,
                    textcoords='offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
                    
                plt.xlabel('Topocentric Right Ascension [deg]')
                plt.ylabel('Topocentric Declination [deg]')
                plt.title('TLE Measurement Space (' + sensor_id + ') ' + UTC.strftime("%Y-%m-%d %H:%M:%S"))
        
                plt.xlim([-180, 180])
                plt.ylim([-90, 90])
                
                kk += 1
            
    if display_flag:
        plt.show()
    
    return


def plot_all_tle_common_time(obj_id_list, UTC_list, tle_dict={}):
    '''
    This function retrieves all TLEs for desired objects within the window
    specified by UTC_list. It finds the object with the most TLEs during the 
    window and propagates all other TLEs to each epoch for that object, then
    plots them together in measurement space.
    
    Parameters
    ------
    obj_id_list : list
        object NORAD IDs, int
    UTC_list : list
        2 element list giving start and end times as UTC datetime objects    
    
    '''
    
    # Retrieve all TLEs in window
    if len(tle_dict) == 0:
        tle_dict, tle_df = get_spacetrack_tle_data(obj_id_list, UTC_list)
    
    print(tle_dict)
    
    # Find object with most TLE entries
    nmax = 0
    for obj_id in obj_id_list:
        line1_list = tle_dict[obj_id]['line1_list']
        ntle = len(line1_list)
        if ntle > nmax:
            nmax = ntle
            plot_obj = obj_id
    
    print('plot obj', plot_obj)
    print('nmax', nmax)
    
    # Plot all TLEs at all times
    line1_list = tle_dict[plot_obj]['line1_list']
    for line1 in line1_list:
        UTC = tletime2datetime(line1)
        print(UTC)
        plot_tle_radec(tle_dict, UTC_list=[UTC])
    
    
    plt.show()
    
    return


def find_closest_tle_epoch(line1_list, line2_list, UTC):
    '''
    This function finds the TLE with the epoch closest to the given UTC time.
    
    Parameters
    ------
    line1_list : list
        list of TLE line1 strings
    line2_list : list
        list of TLE line2 strings
    UTC : datetime object
        UTC time for which closest TLE is desired
        
    Returns
    ------
    line1 : string
        TLE line1 with epoch closest to UTC
    line2 : string
        TLE line2 with epoch closest to UTC
    
    '''
    
    minimum = 1e12
    for ii in range(len(line1_list)):
        line1 = line1_list[ii]
        tle_epoch = tletime2datetime(line1)
        dt_sec = abs((UTC - tle_epoch).total_seconds())
        if dt_sec < minimum:
            ind = ii
            minimum = dt_sec
            
    line1 = line1_list[ind]
    line2 = line2_list[ind]
    
    return line1, line2


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
        tle_dict, tle_df = \
            get_spacetrack_tle_data(obj_id_list, username, password)
        
        # Retreive TLE data from database
        
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = get_celestrak_eop_alldata(offline_flag)
    
    # Retrieve IAU Nutation data from file
    IAU1980_nutation = get_nutation_data()
    
    # Loop over objects
    output_state = {}
    for obj_id in obj_id_list:
        
        line1_list = tle_dict[obj_id]['line1_list']
        line2_list = tle_dict[obj_id]['line2_list']
        
        output_state[obj_id] = {}
        output_state[obj_id]['UTC'] = []
        output_state[obj_id]['r_GCRF'] = []
        output_state[obj_id]['v_GCRF'] = []
        output_state[obj_id]['r_TEME'] = []
        output_state[obj_id]['v_TEME'] = []
        
        # Loop over times
        for UTC in UTC_list:
            
            # Find the closest TLE by epoch
            line1, line2 = find_closest_tle_epoch(line1_list, line2_list, UTC)
            
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

    
#    obj_id_list = [2639, 20777, 28544, 29495, 40146, 42816]
#    UTC_list = [datetime(2018, 4, 20, 0, 0, 0),
#                 datetime(2018, 4, 21, 0, 0, 0)]
    
#    obj_id_list = [40940, 39613, 36287, 39487, 40267, 41836]
#    UTC_list = [datetime(2018, 1, 16, 12, 43, 20)]
#    sensor_list = ['RMIT ROO']
#    
#    
    obj_id_list = [37158]
    UTC_list = [datetime(2018, 10, 29, 0, 0, 0)]
    
#    
#    print(output_state)
    
    plt.close('all')
    
    tle_dict, tle_df = get_spacetrack_tle_data(obj_id_list, UTC_list)
    print(tle_dict)
#
    GPS_time = datetime(2018, 10, 29, 9, 50, 0)
    UTC0 = GPS_time - timedelta(seconds=18.)
    UTC_list = [UTC0]
    
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict, offline_flag=True)
    
    for obj_id in obj_id_list:
        r_GCRF = output_state[obj_id]['r_GCRF'][0]
        v_GCRF = output_state[obj_id]['v_GCRF'][0]
        x_in = np.concatenate((r_GCRF, v_GCRF), axis=0)
        print(obj_id)
        print(x_in)
        elem = element_conversion(x_in, 1, 0)
        print(elem)
    
    pos_ecef = np.reshape([-27379.521717,  31685.387589,  10200.667234], (3,1))
    vel_ecef = np.zeros((3,1))
    
    
    # Comparison
    eop_alldata = get_celestrak_eop_alldata(offline_flag=True)
    EOP_data = get_eop_data(eop_alldata, UTC0)
    
    r_GCRF_sp3, vdum = itrf2gcrf(pos_ecef, vel_ecef, UTC0, EOP_data)
    
    print(r_GCRF_sp3)
    print(r_GCRF_sp3 - r_GCRF)
    print(np.linalg.norm(r_GCRF_sp3 - r_GCRF))
    
    
#    plot_tle_radec(tle_dict, UTC_list, sensor_list, display_flag=True)
    
#    plot_all_tle_common_time(obj_id_list, UTC_list)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
