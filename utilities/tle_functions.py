import numpy as np
import math
import requests
import getpass
from datetime import datetime, timedelta
import sys
import os
import inspect
import matplotlib.pyplot as plt
import pandas as pd
import json
import time
import pickle
import copy
from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs84

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from sensors import sensors as sens
from sensors import measurement_functions as mfunc
from utilities import astrodynamics as astro
from utilities import coordinate_systems as coord
from utilities import eop_functions as eop
from utilities.constants import GME, Re, wE




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
          (TLE epochs within last 30 days only)
    UTC_list : list, optional
        UTC datetime objects to specify desired times for TLEs to retrieve
        - if empty code will retrieve latest available
        - if 1 entry, code will retrieve all TLEs in the +/- 2 day window
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
        
    tle_dict = {}
    tle_df = []
    UTC_list = copy.copy(UTC_list)
    

    if len(username) == 0:
        username = input('space-track username: ')
    if len(password) == 0:
        password = getpass.getpass('space-track password: ')

    if len(obj_id_list) >= 1:
        myString = ",".join(map(str, obj_id_list))

        # If only one time is given, add second to produce window
        if len(UTC_list) ==  1:
            UTC_list.append(UTC_list[0] + timedelta(days=3.))
            UTC_list[0] = UTC_list[0] - timedelta(days=3.)

        # If times are specified, retrieve from window
        if len(UTC_list) >= 2:
            
#            if (UTC_list[-1]-UTC_list[0]).total_seconds() < 86400.*2.:
#                UTC_list[-1] = UTC_list[0] + timedelta(days=2.)
            
            # Create expanded window
            UTC_list[0] = UTC_list[0] # - timedelta(days=2.)
            UTC_list[-1] = UTC_list[-1] # + timedelta(days=2.)
            
            UTC0 = UTC_list[0].strftime('%Y-%m-%d')
            UTC1 = UTC_list[-1].strftime('%Y-%m-%d')
            pageData = ('//www.space-track.org/basicspacedata/query/class/' +
                        'gp_history/NORAD_CAT_ID/' + myString + '/orderby/' +
                        'TLE_LINE1 ASC/EPOCH/' + UTC0 + '--' + UTC1 + 
                        '/format/tle')
            
        

        # Otherwise, get latest available
        else:
            pageData = ('//www.space-track.org/basicspacedata/query/class/gp/'
                        'NORAD_CAT_ID/' + myString +
                        '/orderby/NORAD_CAT_ID/format/tle')
    
    # If no objects specified, retrieve data for full catalog    
    else:
        
        # TODO: These requests don't work, need to figure out correct API
        # request to retrieve full catalog for given date range
        
        # If one or more UTC times are given, retrieve data for the window
        if len(UTC_list) == 1:
            UTC_list.append(UTC_list[0] + timedelta(days=2.))
            UTC_list[0] = UTC_list[0] - timedelta(days=2.)
            
        if len(UTC_list) >= 2:
            UTC0 = UTC_list[0].strftime('%Y-%m-%d')
            UTC1 = UTC_list[-1].strftime('%Y-%m-%d')

            pageData = ('//www.space-track.org/basicspacedata/query/class/gp/' +
                        'EPOCH/' + UTC0 + '--' + UTC1 + 
                        '/orderby/NORAD_CAT_ID/format/tle')
            
            
            # pageData = ('//www.space-track.org/basicspacedata/query/class/' +
            #             'gp_history/NORAD_CAT_ID/' + '00001--99999' + '/orderby/' +
            #             'TLE_LINE1 ASC/EPOCH/' + UTC0 + '--' + UTC1 + 
            #             '/format/tle')
            
            
            

        
        # Otherwise return data for all TLEs with epochs in the last 30 days
        else:
            pageData = ('//www.space-track.org/basicspacedata/query/class/gp/'
                        '/EPOCH/>now-30/orderby/NORAD_CAT_ID/format/tle')
    #        print('Error: No Objects Specified!')

    ST_URL='https://www.space-track.org'

    with requests.Session() as s:
        s.post(ST_URL+"/ajaxauth/login", json={'identity':username, 'password':password})
        r = s.get('https:' + pageData)
        if r.status_code == requests.codes.ok:
            
            print(r.text)
            
            # Parse response and form output
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
        
                try:
                    obj_id = int(line1[2:7])
                except:
                    continue
        
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
            
            
        else:
            print("Error: Page data request failed.")

    return tle_dict, tle_df


def read_tle(text):
    
    tle_dict = {}
    
    # Parse response and form output
    tle_list = []
    nchar = 69
    nskip = 2
    ntle = int(round(len(text)/142.))
    for ii in range(ntle):

        line1_start = ii*2*(nchar+nskip)
        line1_stop = ii*2*(nchar+nskip) + nchar
        line2_start = ii*2*(nchar+nskip) + nchar + nskip
        line2_stop = ii*2*(nchar+nskip) + 2*nchar + nskip
        line1 = text[line1_start:line1_stop]
        line2 = text[line2_start:line2_stop]
        UTC = tletime2datetime(line1)

        try:
            obj_id = int(line1[2:7])
        except:
            continue

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


def generate_tle_list(num_obj, UTC_list, max_obj_id, filename, username, password):
    '''
    
    
    '''
    
    # Alice Springs
    lat = -23.7
    lon = 133.9
    ht = 0.
    site_ecef = coord.latlonht2ecef(lat, lon, ht)
        
    
    # Proportions of object catalog
    LEO_polar = 0.5142
    LEO_lilo = 0.0011
    LEO_nota = 0.2490
    Molniya = 0.0184
    GTO = 0.1030
    MEO_low = 0.0083
    MEO_subsynch = 0.0176
    MEO_nota = 0.0022
    GEO_slot = 0.0320
    GEO_graveyard = 0.0158
    GEO_nota = 0.0277
    HEO_nota = 0.0050
    NOTA = 0.0056    
    
    # Dictionary of counts in each category
    total_list = []
    total_list.append(round(LEO_polar*num_obj))
    total_list.append(round(LEO_lilo*num_obj))
    total_list.append(round(LEO_nota*num_obj))
    total_list.append(round(Molniya*num_obj))
    total_list.append(round(GTO*num_obj))
    total_list.append(round(MEO_low*num_obj))
    total_list.append(round(MEO_subsynch*num_obj))
    total_list.append(round(MEO_nota*num_obj))
    total_list.append(round(GEO_slot*num_obj))
    total_list.append(round(GEO_graveyard*num_obj))
    total_list.append(round(GEO_nota*num_obj))
    total_list.append(round(HEO_nota*num_obj))
    total_list.append(round(NOTA*num_obj))
    
    sum_obj = sum(total_list)
        
    diff = num_obj - sum_obj
    total_list[0] += diff
    
    # As of Dec 2018 about 43000 objects in catalog
#    full_list = list(np.arange(1, max_obj_id))
    full_list = read_tle_file(filename)
    
    print(total_list)
    
    # Retrieve tle dictionary for the object list
    tle_dict, dum = \
        get_spacetrack_tle_data([], [], username, password)
        
    full_list = list(tle_dict.keys())
    print(full_list)
    print(len(full_list))

    # Loop until all objects added to list
    obj_id_list = []
    count_list = [0]*13
    while len(obj_id_list) < num_obj:
        
        print(len(full_list))
        print(count_list)
        
        if len(full_list) == 0:
            break
        
        if sorted(full_list)[0] > max_obj_id:
            break
        
        if sum(count_list) > (num_obj*0.99):
            real_count_list = copy.copy(count_list)
            count_list = [0]*13
        
        # Randomly select an object id number and attempt to retrieve it        
        ind = int(np.random.rand()*len(full_list))
        obj_id = full_list[ind]
        
        if obj_id > max_obj_id:
            continue
        
        print('ind', ind)
        print('obj_id', obj_id)
    
        # If found, sort into category
        if obj_id in tle_dict:
            
            line2 = tle_dict[obj_id]['line2_list'][0]            
            elem = parse_tle_line2(line2)
            category = categorize_orbit(elem)
    
            # If category is not full add to list
            if count_list[category] < total_list[category]:
                
#                # Test retrieval from Dec 2018
#                UTC_test = [datetime(2018, 12, 22, 0, 0, 0), datetime(2019, 1, 1, 0, 0, 0)]
#                tle_test, dum = \
#                    get_spacetrack_tle_data([obj_id], UTC_test, username, password)
#                
#                if len(tle_test) == 0:
#                    print('Delete ', full_list[ind])
#                    del full_list[ind]
#                    continue
                
                # Check for GEO over Australia
                if category == 8:
                    line1 = tle_dict[obj_id]['line1_list'][0]
                    UTC = tletime2datetime(line1)
                    
                    output_state = propagate_TLE([obj_id], [UTC], tle_dict)
                    r_ecef = output_state[obj_id]['r_ITRF'][0]
                    az, el, rg = mfunc.ecef2azelrange_deg(r_ecef, site_ecef)
                    
                    if el < 30:
                        
                        print('Delete ', full_list[ind])
                        del full_list[ind]
                        continue
                
                obj_id_list.append(obj_id)
                count_list[category] += 1
                print('Delete ', full_list[ind])
                del full_list[ind]
                
                print(obj_id)
                print(category)
                print(len(obj_id_list))
            
            else:
                print('Delete ', full_list[ind])
                del full_list[ind]
            
        else:
            print('Delete ', full_list[ind])
            del full_list[ind]
        
        if len(obj_id_list) == 1000:      
            
            obj_id_firsthalf = sorted(obj_id_list)[0:500]
            obj_id_secondhalf = sorted(obj_id_list)[500:]
            
            # Test retrieve from Dec 2018
            UTC_list = [datetime(2018, 12, 22, 0, 0, 0), datetime(2019, 1, 1, 0, 0, 0)]
            
            tle_dict_firsthalf, dum = \
                get_spacetrack_tle_data(obj_id_firsthalf, UTC_list, username, password)
                
            tle_dict_secondhalf, dum = \
                get_spacetrack_tle_data(obj_id_secondhalf, UTC_list, username, password)
                
            print(len(tle_dict_firsthalf))
            print(len(tle_dict_secondhalf))
            
            missing_firsthalf = sorted(list(set(obj_id_firsthalf) - set(tle_dict_firsthalf.keys())))
            missing_secondhalf = sorted(list(set(obj_id_secondhalf) - set(tle_dict_secondhalf.keys())))
            
            print(missing_firsthalf)
            print(missing_secondhalf)
            
            obj_id_list = list(set(obj_id_list) - set(missing_firsthalf))
            obj_id_list = list(set(obj_id_list) - set(missing_secondhalf))
            
            print('Restarting with Nobj = ')
            print(len(obj_id_list))
    
    
            
    
    final_count_list = [xi + yi for xi, yi in zip(real_count_list, count_list)]
    print(obj_id_list)
    print(final_count_list)
    print(sum(final_count_list))
    
    
    
    return obj_id_list


def read_tle_file(filename, username='', password=''):
    
    obj_id_list = []
    with open(filename) as f:
        for line in f:
            obj_id = int(line[2:7])
            obj_id_list.append(obj_id)
            
    obj_id_list = sorted(list(set(obj_id_list)))
    
    
    return obj_id_list


def categorize_orbit(elem):
    '''
    This function determines a unique orbit category for a given orbital
    element state vector.
    
    Parameters
    ------
    elem : list
    elem[0] : a
      Semi-Major Axis             [km]
    elem[1] : e
      Eccentricity                [unitless]
    elem[2] : i
      Inclination                 [deg]
    elem[3] : RAAN
      Right Asc Ascending Node    [deg]
    elem[4] : w
      Argument of Periapsis       [deg]
    elem[5] : M
      Mean anomaly                [deg]

    Returns
    ------
    category : int
        0 : LEO_polar (ra < Re + 2000 km, 75 < i < 120 deg)
        1 : LEO_lilo (ra < Re + 2000 km, 0 < i < 20 deg)
        2 : LEO_nota (ra < Re + 2000 km, else)
        3 : Molniya (Re < rp < 15000 km, 37000 < ra < 48000 km, 60 < i < 75 deg)
        4 : GTO_nota (Re < rp < 10000 km, ra < 50000 km, i < 75 deg OR
                      10000 < rp < 40000 km, 35000 < ra < 45000 km, i < 75 deg)
        5 : MEO_low (ra < 10000 km)
        6 : MEO_subsynch (rp > 23000 km, ra < 32000 km, 45 < i < 75 deg)
        7 : MEO_nota (rp > 10000 km, ra < 40000 km, e < 0.1)
        8 : GEO_slot (rp > Rgeo - 50 km, ra < Rgeo + 50 km, 0 < i < 20 deg)
        9 : GEO_graveyard (rp > Rgeo + 50 km, ra < 45000 km, 0 < i < 20 deg)
        10 : GEO_nota (rp > 40000 km, ra < 45000 km, 0 < i < 20 deg)
        11 : HEO_nota (rp > 45000 km)
        12 : NOTA (else)
            
    Reference
    ------
    Holzinger et al., "Uncorrelated-Track Classification, Characterization,
    and Prioritization Using Admissible Regions and Bayesian Inference," 2016   
    
    
    '''
    
    a = elem[0]
    e = elem[1]
    i = elem[2]
    
    Rgeo = 42164.2
    
    rp = a*(1-e)
    ra = a*(1+e)
    
    # Compute checks
    if (ra < (Re + 2000.) and i > 75. and i < 120.):
        category = 0
    elif (ra < (Re + 2000.) and i > 0. and i < 20.):
        category = 1
    elif ra < (Re + 2000.):
        category = 2
    elif (rp > Re and rp < 15000. and ra > 37000. and ra < 40000. and i > 60. and i < 75.):
        category = 3
    elif (rp > Re and rp < 10000. and ra < 50000. and i > 60. and i < 75.):
        category = 4
    elif (rp > 10000. and rp < 40000. and ra > 35000. and ra < 45000. and i < 75.):
        category = 4
    elif (ra < 10000.):
        category = 5
    elif (rp > 23000. and ra < 32000. and i > 45. and i < 75.):
        category = 6
    elif (rp > 10000. and ra < 40000. and e < 0.1):
        category = 7
    elif (rp > (Rgeo - 50.) and ra < (Rgeo + 50.) and i > 0. and i < 20.):
        category = 8
    elif (rp > (Rgeo + 50.) and ra < 45000. and i > 0. and i < 20.):
        category = 9
    elif (rp > 40000. and ra < 45000. and i > 0. and i < 20.):
        category = 10
    elif rp > 45000.:
        category = 11
    else:
        category = 12
    
    return category


def check_category(tle_list_file, username='', password=''):
    
    # Load TLE list
    pklFile = open(tle_list_file, 'rb')
    data = pickle.load(pklFile)
    obj_id_list = data[0]
    pklFile.close()
    
    obj_id_list = sorted(obj_id_list)
    print(len(obj_id_list))
    print(sorted(obj_id_list))
    
    
    tle_dict, dum = \
        get_spacetrack_tle_data([], [], username, password)
        
    count_list = [0]*13
    for obj_id in obj_id_list:
        line2 = tle_dict[obj_id]['line2_list'][0]
        
        elem = parse_tle_line2(line2)
        category = categorize_orbit(elem)
        
        count_list[category] += 1
    
    print(count_list)
    print(sum(count_list))
    
    
    return


def gen_tle_textfiles(tle_list_file, username='', password=''):
    
    # Load TLE list
    pklFile = open(tle_list_file, 'rb')
    data = pickle.load(pklFile)
    obj_id_list = data[0]
    pklFile.close()
    
    obj_id_list = sorted(obj_id_list)
    print(len(obj_id_list))
    
    obj_id_firsthalf = sorted(obj_id_list)[0:500]
    obj_id_secondhalf = sorted(obj_id_list)[500:]
    
    UTC0 = datetime(2019, 2, 20, 0, 0, 0)
#    UTC1 = datetime(2019, 1, 1, 0, 0, 0)
    UTC1 = UTC0 + timedelta(days=10)
    
    while UTC1 < datetime(2019, 12, 31):
    
        UTC_list = [UTC0, UTC1]
        datestr = UTC1.strftime('%Y_%m_%d')
        print('\n\n', datestr)
        
        tle_dict_firsthalf, dum = \
            get_spacetrack_tle_data(obj_id_firsthalf, UTC_list, username, password)
            
        tle_dict_secondhalf, dum = \
            get_spacetrack_tle_data(obj_id_secondhalf, UTC_list, username, password)
            
        print(len(tle_dict_firsthalf))
        print(len(tle_dict_secondhalf))
        
        missing_firsthalf = sorted(list(set(obj_id_firsthalf) - set(tle_dict_firsthalf.keys())))
        missing_secondhalf = sorted(list(set(obj_id_secondhalf) - set(tle_dict_secondhalf.keys())))
        
        print(missing_firsthalf)
        print(missing_secondhalf)
        
        tle_dict_firsthalf.update(tle_dict_secondhalf)
        tle_dict = tle_dict_firsthalf
        print(len(tle_dict))
        
        fname = 'TLE_' + datestr + '.txt'
        output_file = os.path.join( 'D:\documents\\research\sensor_management\site_location', fname )
        
        for obj_id in obj_id_list:
            
            if obj_id in tle_dict:
                line1 = tle_dict[obj_id]['line1_list'][-1]
                line2 = tle_dict[obj_id]['line2_list'][-1]
            else:
#                print(obj_id)
                fname_prior = 'TLE_' + UTC0.strftime('%Y_%m_%d') + '.txt'
#                print(fname_prior)
                input_file = os.path.join( 'D:\documents\\research\sensor_management\site_location', fname_prior )
                with open(input_file) as f:
                    line_list = [line.rstrip() for line in f]
                for ii in range(len(line_list)):
                    line = line_list[ii]
#                    print(int(line[2:7]))
                    if int(line[2:7]) == obj_id:
                        line1 = line
                        line2 = line_list[ii+1]
#                        print(int(line[2:7]))
#                        print(line)
#                        print(all_lines[ii+1])
                        
                        break
                
                
                print(obj_id)
                print(line1)
                print(line2)
                
                    
            
#            print(line1)
#            print(line2)
            
            outfile = open(output_file,'a')
            outfile.write(line1 + '\n')
            outfile.write(line2 + '\n')
            outfile.close()
            
        
        # Increment for next pass
        UTC0 = UTC0 + timedelta(days=10)
        UTC1 = UTC1 + timedelta(days=10)
        
    
    return


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


def tle_archiver():
    tle_df = get_tle_range(username='',password='', start_norad='43689', stop_norad='81000')
    
    os.chdir(r'D:\documents\research\launch_identification\data\tle_archive')
    
    filename = datetime.now().strftime("%Y%m%d-%H%M%S")+'_tle_latest.csv'
    tle_df.to_csv(filename)
    
    return


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
    output_tlechange = {}
    output_periodic = {}
    
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
        state = propagate_TLE(obj_id_red, [UTC], tle_dict, offline_flag=True)
    
        # Store output
        output_tlechange[UTC] = {}
        output_tlechange[UTC]['obj_id_list'] = obj_id_red
        output_tlechange[UTC]['name_list'] = []
        output_tlechange[UTC]['r_GCRF'] = []
        output_tlechange[UTC]['v_GCRF'] = []
        
        for obj_id in obj_id_red:
            ind = [ii for ii in range(len(tle_dict[obj_id]['UTC_list'])) if tle_dict[obj_id]['UTC_list'][ii] <= UTC][-1]
            
            output_tlechange[UTC]['name_list'].append(tle_dict[obj_id]['name_list'][ind])
            output_tlechange[UTC]['r_GCRF'].append(state[obj_id]['r_GCRF'][0])
            output_tlechange[UTC]['v_GCRF'].append(state[obj_id]['v_GCRF'][0])    
    
    
    # Get unique list of UTC times in order
    UTC_list = []
    obj_id = list(tle_dict.keys())[0]
    UTC_list0 = tle_dict[obj_id]['UTC_list']
    line2_list = tle_dict[obj_id]['line2_list']
    
    UTC = UTC_list0[0]
    for ii in range(1, len(UTC_list0)):
        print(ii)
        while UTC < UTC_list0[ii]:
            UTC_list.append(UTC)
            n_revday = float(line2_list[ii-1][52:63])
            period = 86400./n_revday
            UTC += timedelta(seconds=period)

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
        state = propagate_TLE(obj_id_red, [UTC], tle_dict, offline_flag=True)
    
        # Store output
        output_periodic[UTC] = {}
        output_periodic[UTC]['obj_id_list'] = obj_id_red
        output_periodic[UTC]['name_list'] = []
        output_periodic[UTC]['r_GCRF'] = []
        output_periodic[UTC]['v_GCRF'] = []
        
        for obj_id in obj_id_red:
            ind = [ii for ii in range(len(tle_dict[obj_id]['UTC_list'])) if tle_dict[obj_id]['UTC_list'][ii] <= UTC][-1]
            
            output_periodic[UTC]['name_list'].append(tle_dict[obj_id]['name_list'][ind])
            output_periodic[UTC]['r_GCRF'].append(state[obj_id]['r_GCRF'][0])
            output_periodic[UTC]['v_GCRF'].append(state[obj_id]['v_GCRF'][0])  

    return output_tlechange, output_periodic


def compute_tle_elements(tle_dict):
    '''
    This function computes a dictionary of mean orbit elements at for
    all objects in the input TLE dictionary.
    
    Parameters
    ------
    tle_dict : dictionary
        indexed by object ID, each item has two lists of strings for each line
        as well as object name and UTC times
        
    Returns
    ------
    output : dictionary
        indexed by NORAD ID, each item has a list of UTC times and 
        corresponding mean elements extracted from the TLEs, including
        SMA, ECC, INC, RAAN, AoP, M, while retaining the TLE line lists.
    
    '''
    
    for obj_id in tle_dict:
        UTC_list = tle_dict[obj_id]['UTC_list']
        line1_list = tle_dict[obj_id]['line1_list']
        line2_list = tle_dict[obj_id]['line2_list']
        
        tle_dict[obj_id]['elem_list'] = []
        for line2 in line2_list:
            elem = parse_tle_line2(line2)
            tle_dict[obj_id]['elem_list'].append(elem)

    
    return tle_dict


def plot_sma_rp_ra(obj_id_list, UTC_list):
    '''
    
    
    '''
    
    
    # Generate TLE dictionary
    tle_dict, tle_df = get_spacetrack_tle_data(obj_id_list, UTC_list)
    
    # Extract orbit elements
    tle_dict = compute_tle_elements(tle_dict)
    
    # Loop over objects and times to generate and plot arrays of SMA, rp, ra
    plot_dict = {}
    for obj_id in tle_dict:
        plot_dict[obj_id] = {}
        plot_dict[obj_id]['UTC_list'] = tle_dict[obj_id]['UTC_list']
        elem_list = tle_dict[obj_id]['elem_list']
        a_array = np.array([])
        rp_array = np.array([])
        ra_array = np.array([])
        
        print(obj_id)
        inc = elem_list[0][2] * math.pi/180.
        print('ecc', elem_list[0][1])
        print('inc', elem_list[0][2])
        
        for elem in elem_list:
            a = elem[0]
            e = elem[1]
            rp = a*(1. - e)
            ra = a*(1. + e)
            a_array = np.append(a_array, a)
            rp_array = np.append(rp_array, rp)
            ra_array = np.append(ra_array, ra)

            
        plot_dict[obj_id]['a_array'] = a_array
        plot_dict[obj_id]['rp_array'] = rp_array
        plot_dict[obj_id]['ra_array'] = ra_array
        
        
    plt.figure()
    for obj_id in plot_dict:
        UTC_list = plot_dict[obj_id]['UTC_list']
        a_array = plot_dict[obj_id]['a_array']
        rp_array = plot_dict[obj_id]['rp_array']
        ra_array = plot_dict[obj_id]['ra_array']
        
        if obj_id == 43692:
            plt.plot(UTC_list, a_array, 'b.')
        else:
            plt.plot(UTC_list, a_array, 'k.')
            
            
    label_dict = {}
    label_dict[43164] = 'ST'
    label_dict[43692] = 'IBT (NABEO)'
    label_dict[43851] = 'ELaNa'
    label_dict[43863] = 'ELaNa'
    label_dict[44074] = 'DARPA'
    label_dict[44227] = 'STP-27RD'
    label_dict[44372] = 'MIR'
    label_dict[44496] = 'LMNH'
    
    UTC_start_compare = tle_dict[44227]['UTC_list'][4]
    
    # Atmosphere model
    rho0 = 6.967e-13 * 1e9  # kg/km^3
    h0 = 500.  # km
    H = 63.822  # km
            
            
    plt.figure()
    colors = ['b', 'g', 'k', 'r', 'm', 'c']  
    ii = 0
    for obj_id in plot_dict:
        UTC_list = plot_dict[obj_id]['UTC_list']
        a_array = plot_dict[obj_id]['a_array']
        rp_array = plot_dict[obj_id]['rp_array']
        ra_array = plot_dict[obj_id]['ra_array']
        
#        if obj_id == 43692:
#            plt.plot(UTC_list, a_array-Re, 'b.')
#        else:
#            plt.plot(UTC_list, a_array-Re, 'k.')
        
#        label_txt = 
        
        plt.plot(UTC_list[4:], a_array[4:]-Re, '.',  c=colors[ii], label=label_dict[obj_id])
        ii += 1
            
            
#        plt.ylim([495., 530.])
        plt.locator_params(axis='y', nbins=5)
        plt.ylabel('Mean Altitude [km]')
            
            
        plt.xlim([datetime(2018, 11, 1), datetime(2019, 11, 1)])
        plt.xlabel('Date')
        
        plt.legend()
        
        print('\n\nSMA Analysis')
        print(label_dict[obj_id])
        ndays = (UTC_list[-1] - UTC_list[4]).total_seconds()/86400.
        ave_sma = (a_array[-1] - a_array[4])/ndays * 1000.
        print('Ave SMA change [m/day]', ave_sma)
        
        print('Reduced Time SMA Change')
        UTC_compare = [abs((UTCii - UTC_start_compare).total_seconds()) for UTCii in UTC_list]
        ind = UTC_compare.index(min(UTC_compare))
        print(UTC_list[ind])
        ndays = (UTC_list[-1] - UTC_list[ind]).total_seconds()/86400.
        ave_sma = (a_array[-1] - a_array[ind])/ndays * 1000.
        print('Ave SMA change [m/day]', ave_sma)
        
        print('\n\nBallistic Coefficient Estimate')
        print(label_dict[obj_id])
        
        # Compute density and ballistic coefficient info
        ind2 = int(math.floor((ind+len(UTC_list))/2))
        print(len(UTC_list))
        print(len(a_array))
        print(ind)
        print(ind2)
        
        dadt = ave_sma * (1./1000.) * (1./86400.)   # km/s
        
        
        SMA = a_array[ind2]
        n = np.sqrt(GME/SMA**3.)
        h = SMA - Re        
        rho = rho0*math.exp(-(h-h0)/H)
        beta = -dadt/(rho*np.sqrt(GME*a)*(1. - (wE/n)*math.cos(inc))**2.) * 1e6
        
        
        print('h = ', h)
        print('rho = ', rho)
        print('beta = ', beta)
        
        
        
        
        
        
#        plt.locator_params(axis='x', nbins=3)
#        plt.xticks([datetime(2018, 11, 1).strftime('%Y-%m-%d'), 
#                    datetime(2019, 2, 1).strftime('%Y-%m-%d'),
#                    datetime(2019, 5, 1).strftime('%Y-%m-%d'), 
#                    datetime(2019, 8, 1).strftime('%Y-%m-%d'),
#                    datetime(2019, 11, 1).strftime('%Y-%m-%d')])
    
    plt.figure()
    ii = 0    
    for obj_id in plot_dict:
        UTC_list = plot_dict[obj_id]['UTC_list']
        a_array = plot_dict[obj_id]['a_array']
        rp_array = plot_dict[obj_id]['rp_array']
        ra_array = plot_dict[obj_id]['ra_array']
        
#        if obj_id == 43692:
#            plt.plot(UTC_list, a_array-Re, 'b.')
#        else:
#            plt.plot(UTC_list, a_array-Re, 'k.')
        
#        label_txt = 
        
        plt.plot(UTC_list[4:], ra_array[4:]-Re, '+', c=colors[ii], label=label_dict[obj_id])
        plt.plot(UTC_list[4:], rp_array[4:]-Re, 'o', c=colors[ii])
        ii += 1
            
            
#        plt.ylim([495., 530.])
        plt.locator_params(axis='y', nbins=5)
        plt.ylabel('Mean Altitude [km]')
            
            
        plt.xlim([datetime(2018, 11, 1), datetime(2019, 11, 1)])
        plt.xlabel('Date')
        
        plt.legend()
        
        
    plt.figure()
    
    ii = 0
    for obj_id in plot_dict:
        UTC_list = plot_dict[obj_id]['UTC_list']
        a_array = plot_dict[obj_id]['a_array']
        diff_array = np.diff(a_array)
        
#        if obj_id == 43692:
#            plt.plot(UTC_list, a_array-Re, 'b.')
#        else:
#            plt.plot(UTC_list, a_array-Re, 'k.')
        
        plt.plot(UTC_list[1:], diff_array*1000., '.',  c=colors[ii], label=str(obj_id))
        ii += 1
            
            
        plt.ylim([-50., 20.])
        plt.locator_params(axis='y', nbins=5)
        plt.ylabel('Altitude Diff [km]')
            
            
        plt.xlim([datetime(2018, 11, 1), datetime(2019, 11, 1)])
        plt.xlabel('Date')
        
        plt.legend()
        
        
        
    plt.show()
        
    
    
    
    return


def parse_tle_line1(line1):
    
    
    
    
    return


def parse_tle_line2(line2):
    '''
    This function parses Line 2 of the TLE to extract mean orbit elements.
    
    Parameters
    ------
    line2 : string
        second line of TLE
    
    Returns
    ------
    elem : list
        elem[0] : a
          Semi-Major Axis             [km]
        elem[1] : e
          Eccentricity                [unitless]
        elem[2] : i
          Inclination                 [deg]
        elem[3] : RAAN
          Right Asc Ascending Node    [deg]
        elem[4] : w
          Argument of Periapsis       [deg]
        elem[5] : M
          Mean anomaly                [deg]
    '''
    
    i = float(line2[8:16])
    RAAN = float(line2[17:25])
    e = float(line2[26:33])/1e7
    w = float(line2[34:42])
    M = float(line2[43:51])
    n = float(line2[52:63])  # rev/day
    
    n *= 2.*math.pi/86400.  # rad/s
    
    a = astro.meanmot2sma(n)
            
    elem = [a, e, i, RAAN, w, M]    
    
    return elem


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
        n *= 86400./(2.*math.pi)

        # Compute eccentricity
        e = 1. - rp/a

        # Compute GCRF position and velocity
        x_in = [a,e,i,RAAN,w,M]
        x_out = astro.element_conversion(x_in, 0, 1)
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
        x_out = astro.element_conversion(x_in, 0, 1)
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
    eop_alldata = eop.get_celestrak_eop_alldata(offline_flag)

    # Retrieve IAU Nutation data from file
    IAU1980nut = eop.get_nutation_data()

    # Loop over objects
    for obj_id in obj_id_list:

        # Retrieve launch coordinates for this object
        r_ITRF = ecef_dict[obj_id]['r_ITRF']
        v_ITRF = ecef_dict[obj_id]['v_ITRF']
        UTC = ecef_dict[obj_id]['UTC']

        # Get EOP data for this time
        EOP_data = eop.get_eop_data(eop_alldata, UTC)

        # Convert ITRF to GCRF
        r_GCRF, v_GCRF = coord.itrf2gcrf(r_ITRF, v_ITRF, UTC, EOP_data)

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

        eop_alldata = eop.get_celestrak_eop_alldata(offline_flag)
        EOP_data = eop.get_eop_data(eop_alldata, UTC)

    # Retrieve IAU Nutation data from file, if needed
    if len(IAU1980nut) == 0:
        IAU1980nut = eop.get_nutation_data()

    # Convert to TEME, recompute osculating elements
    r_TEME, v_TEME = coord.gcrf2teme(r_GCRF, v_GCRF, UTC, IAU1980nut, EOP_data)
    x_in = np.concatenate((r_TEME, v_TEME), axis=0)
    osc_elem = astro.element_conversion(x_in, 1, 0)

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
    n *= 86400./(2.*math.pi)

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

            ra_array[ii,jj] = math.atan2(y,x)*180/math.pi
            dec_array[ii,jj] = math.asin(z/r)*180/math.pi

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
        eop_alldata = eop.get_celestrak_eop_alldata(offline_flag)

        # Retrive sensor parameters and loop over sensors
        sensor_dict = sens.define_sensors(sensor_list)
        for sensor_id in sensor_list:
            sensor = sensor_dict[sensor_id]
            latlonht = sensor['geodetic_latlonht']
            lat = latlonht[0]
            lon = latlonht[1]
            ht = latlonht[2]
            sensor_ecef = coord.latlonht2ecef(lat, lon, ht)

            center = [85.82990416666667, 5.990788888888889]
            FOV_hlim = [lim*180/math.pi for lim in sensor['FOV_hlim']]
            FOV_vlim = [lim*180/math.pi for lim in sensor['FOV_vlim']]
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
                    EOP_data = eop.get_eop_data(eop_alldata, UTC)
                    sensor_eci, dum = coord.itrf2gcrf(sensor_ecef, np.zeros((3,1)),
                                                      UTC, EOP_data)

                    xs = float(sensor_eci[0])
                    ys = float(sensor_eci[1])
                    zs = float(sensor_eci[2])
                    rg = np.linalg.norm(r_GCRF - sensor_eci)

                    topo_ra_array[ii,jj] = math.atan2((y-ys),(x-xs))*180/math.pi
                    topo_dec_array[ii,jj] = math.asin((z-zs)/rg)*180/math.pi

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


def find_closest_tle_epoch(line1_list, line2_list, UTC, prev_flag=False):
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
    prev_flag : boolean, optional
        only consider TLEs with epochs previous to UTC (true) or consider all
        TLEs (false) (default=false)

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
        dt_sec = (UTC - tle_epoch).total_seconds()

        
        if prev_flag:
            if dt_sec >= 0 and abs(dt_sec) < minimum:
                ind = ii
                minimum = dt_sec
                        
        else:        
            if abs(dt_sec) < minimum:
                ind = ii
                minimum = dt_sec

    line1 = line1_list[ind]
    line2 = line2_list[ind]

    return line1, line2


def propagate_TLE(obj_id_list, UTC_list, tle_dict={}, prev_flag=False,
                  offline_flag=False, frame_flag=True, username='', password=''):
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
    frame_flag : boolean, optional
        flag to determine whether to rotate state vector to GCRF and ITRF to 
        include with output along with state vector in TEME (default = True)
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
    
    total_prop = 0.
    total_teme2gcrf = 0.
    total_gcrf2itrf = 0.
    tle_epoch_time = 0.
    total_eop_time = 0.

    # If no TLE information is provided, retrieve from sources as needed
    if len(tle_dict) == 0:

        # Retrieve latest TLE data from space-track.org
        tle_dict, tle_df = \
            get_spacetrack_tle_data(obj_id_list, UTC_list, username, password)

        # Retreive TLE data from database
    
    
    # If frame rotations to GCRF and ITRF are desired, retrieve EOP data
    if frame_flag:  
        
        # Retrieve latest EOP data from celestrak.com
        eop_alldata = eop.get_celestrak_eop_alldata(offline_flag)
    
        # Retrieve IAU Nutation data from file
        IAU1980_nutation = eop.get_nutation_data()
        
        # Retrieve polar motion data from file
        XYs_df = eop.get_XYs2006_alldata()


    # Loop over objects
    output_state = {}
    for obj_id in obj_id_list:

        line1_list = tle_dict[obj_id]['line1_list']
        line2_list = tle_dict[obj_id]['line2_list']

        output_state[obj_id] = {}
        output_state[obj_id]['UTC'] = []
        output_state[obj_id]['r_GCRF'] = []
        output_state[obj_id]['v_GCRF'] = []
        output_state[obj_id]['r_ITRF'] = []
        output_state[obj_id]['v_ITRF'] = []
        output_state[obj_id]['r_TEME'] = []
        output_state[obj_id]['v_TEME'] = []

        # Loop over times
        for UTC in UTC_list:
            
#            print(obj_id, UTC)
            
            # Find the closest TLE by epoch
            epoch_start = time.time()
            line1, line2 = find_closest_tle_epoch(line1_list, line2_list, UTC,
                                                  prev_flag)
            
            tle_epoch_time += time.time() - epoch_start

            # Propagate TLE using SGP4
            prop_start = time.time()
            satellite = twoline2rv(line1, line2, wgs84)
            r_TEME, v_TEME = satellite.propagate(UTC.year, UTC.month, UTC.day,
                                                 UTC.hour, UTC.minute,
                                                 UTC.second +
                                                 (UTC.microsecond/1e6))

            r_TEME = np.reshape(r_TEME, (3,1))
            v_TEME = np.reshape(v_TEME, (3,1))
            
            total_prop += time.time() - prop_start
            
            # Store output
            output_state[obj_id]['UTC'].append(UTC)
            output_state[obj_id]['r_TEME'].append(r_TEME)
            output_state[obj_id]['v_TEME'].append(v_TEME)

            # If desired, compute state vector in GCRF and ITRF
            # Note this will be slow for large datasets - use batch processing
            if frame_flag:
                
                # Get EOP data for this time
                eop_start = time.time()
                EOP_data = eop.get_eop_data(eop_alldata, UTC)
                
                total_eop_time += time.time() - eop_start
    
                # Convert from TEME to GCRF (ECI)
                teme_start = time.time()
                r_GCRF, v_GCRF = coord.teme2gcrf(r_TEME, v_TEME, UTC, IAU1980_nutation,
                                           EOP_data)
                
                total_teme2gcrf += time.time() - teme_start
                
                # Convert from GCRF to ITRF (ECEF)
                itrf_start = time.time()
                r_ITRF, v_ITRF = coord.gcrf2itrf(r_GCRF, v_GCRF, UTC, EOP_data, XYs_df)
                
                total_gcrf2itrf += time.time() - itrf_start

                # Store output
                output_state[obj_id]['r_GCRF'].append(r_GCRF)
                output_state[obj_id]['v_GCRF'].append(v_GCRF)
                output_state[obj_id]['r_ITRF'].append(r_ITRF)
                output_state[obj_id]['v_ITRF'].append(v_ITRF)
            
            
        print(obj_id)
        print('TLE epoch find: ', tle_epoch_time)
        print('Prop: ', total_prop)
        print('EOP: ', total_eop_time)
        print('TEME: ', total_teme2gcrf)
        print('ITRF: ', total_gcrf2itrf)


    return output_state


def get_planet_ephem():
    '''
    This function retrieves 'planet_mc.tle' and 'planet.states' and
    'jspoc_matches.txt' from http://ephemerides.planet-labs.com/.

    Returns
    ------
    saves 3 output files:
    YYYMMDD-HHMMSS_planet_mc.tle
    YYYMMDD-HHMMSS_planet.states
    YYYMMDD-HHMMSS_jspoc_matches.txt
    '''
    url_list = ['http://ephemerides.planet-labs.com/planet_mc.tle',
                'http://ephemerides.planet-labs.com/planet.states',
                'http://ephemerides.planet-labs.com/jspoc_matches.txt']

    for url in url_list:
        response = requests.get(url, allow_redirects=True)
        filename = datetime.now().strftime("%Y%m%d-%H%M%S")+'_'+url.split(sep='/')[-1]
        open(filename, 'wb').write(response.content)



###############################################################################
# Stand-alone execution
###############################################################################

if __name__ == '__main__' :
    
    plt.close('all')
    

    obj_id = 32789
    # tle_dict, dum = get_spacetrack_tle_data(obj_id_list = [obj_id])
    
    # print(tle_dict)
    
    tle_text = ("1 32789U 07021G   08119.60740078 -.00000054  00000-0  00000+0 0  9999 \n"
                "2 32789 098.0082 179.6267 0015321 307.2977 051.0656 14.81417433    68")
    
    print(tle_text)
    

    
    tle_dict, dum = read_tle(tle_text)
    
    
    
    UTC = tletime2datetime(tle_dict[obj_id]['line1_list'][0])
    
    output_state = propagate_TLE([obj_id], [UTC], tle_dict=tle_dict)
    
    print(UTC)
    # print(output_state)
    
    r_GCRF = output_state[obj_id]['r_GCRF'][0]
    v_GCRF = output_state[obj_id]['v_GCRF'][0]
    X = np.concatenate((r_GCRF, v_GCRF), axis=0)*1000
    print(X.flatten())
    
    # # GEO - Optus 10
    # obj_id = 40146
    # UTC = datetime(2023, 5, 19, 0, 0, 0)
    # output_state = propagate_TLE([obj_id], [UTC])
    
    # r_GCRF = output_state[obj_id]['r_GCRF'][0]
    # v_GCRF = output_state[obj_id]['v_GCRF'][0]
    
    # cart = np.concatenate((r_GCRF, v_GCRF), axis=0)
    
    # print(cart)
    
    # elem = astro.cart2kep(cart)
    
    # print(elem)
    
    
#    obj_id_list = [43164, 43166, 43691, 43692, 43851, 43863, 44074, 44075,
#                   44227, 44228, 44372, 44496]
#    
#    obj_id_list = [43164, 43692, 43851, 44227, 44074, 44496]
#    
#    obj_id_list = [43164, 43692, 43851, 44227]
#    
#    UTC_list = [datetime(2018, 1, 1), datetime(2019, 10, 4)]
#    
#    plot_sma_rp_ra(obj_id_list, UTC_list)
    
    # # Landsat-8 Data
    # landsat8_norad = 39084
    
    # # Sentinel 2 Data
    # sentinel_2a_norad = 40697
    # sentinel_2b_norad = 42063
    
    # # Retrieve TLE and print state vectors
    # obj_id_list = [landsat8_norad, sentinel_2a_norad, sentinel_2b_norad]
    # UTC_start = datetime(2020, 6, 29, 0, 0, 0)
    # UTC_stop = datetime(2020, 7, 1, 0, 0, 0)
    # dt = 10.
    
    # delta_t = (UTC_stop - UTC_start).total_seconds()
    # UTC_list = [UTC_start + timedelta(seconds=ti) for ti in list(np.arange(0, delta_t, dt))]
    
    # print(UTC_list[0], UTC_list[-1])
    
    # # Retrieve TLE 
    # retrieve_list = [UTC_list[0] - timedelta(days=1), UTC_list[-1] + timedelta(days=1)]
    # tle_dict, dum = get_spacetrack_tle_data(obj_id_list, retrieve_list)
    
    # print(tle_dict)
    
    # output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict, prev_flag=True)
    
    # # Save output
    # fname = os.path.join(r'D:\documents\research\cubesats\GeoscienceAustralia\data\check_tle_propagation.pkl')
    # pklFile = open( fname, 'wb' )
    # pickle.dump( [output_state], pklFile, -1 )
    # pklFile.close()
    
#    
#    UTC_list = [datetime(2021, 3, 24, 0, 0, 0)]
#   
#    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)
#    print(output_state)
#
#    r_eci = output_state[25544]['r_GCRF'][0]
    
#    filename = os.path.join('D:\documents\\research\sensor_management\site_location', 'tle_data_2020.txt')

    
#    num_obj = 1000
#    UTC_list = [datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 1, 10, 0, 0, 0)]
#    max_obj_id = 40000
    
#    obj_id_list = [47967]
#    UTC_window = [datetime(2021, 3, 21), datetime(2021, 3, 24)]
#    tle_dict, tle_df = get_spacetrack_tle_data(obj_id_list, UTC_window)
#    
#    print(tle_dict)
    
    
    
#    obj_id_list = generate_tle_list(num_obj, UTC_list, max_obj_id, filename, username, password)
#    print(obj_id_list)
#    
#    # Save data
#    tle_list_file =  os.path.join( 'D:\documents\\research\sensor_management\site_location','tle_obj_file.pkl' )
#    pklFile = open( filename, 'wb' )
#    pickle.dump( [obj_id_list], pklFile, -1 )
#    pklFile.close()
    
#    gen_tle_textfiles(filename, username, password)
    
    
#    check_category(tle_list_file, username, password)
    


#    obj_id_list = [2639, 20777, 28544, 29495, 40146, 42816]
#    UTC_list = [datetime(2018, 4, 20, 0, 0, 0),
#                 datetime(2018, 4, 21, 0, 0, 0)]

#    obj_id_list = [40940, 39613, 36287, 39487, 40267, 41836]
#    UTC_list = [datetime(2018, 1, 16, 12, 43, 20)]
#    sensor_list = ['RMIT ROO']
#    
#   
#    eop_alldata = get_celestrak_eop_alldata()
    
#    gps_time = datetime(2019, 9, 3, 10, 5, 0)
#    
#    EOP_data = get_eop_data(eop_alldata, gps_time)
#    
#    utc_time = gpsdt2utcdt(gps_time, EOP_data['TAI_UTC'])
#    print(utc_time)
    
#    utc_time = datetime(2019, 9, 3, 10, 9, 42)
#
#    print(utc_time)
#    
#    
#    sensor_dict = define_sensors(['UNSW Falcon'])
#    latlonht = sensor_dict['UNSW Falcon']['geodetic_latlonht']
#    lat = latlonht[0]
#    lon = latlonht[1]
#    ht = latlonht[2]
#    stat_ecef = latlonht2ecef(lat, lon, ht)
#    
#    
#    
#    
##    start_time = datetime(2019, 9, 23, 0, 0, 0)
#    UTC_list = [utc_time] # + timedelta(seconds=ti) for ti in range(0,101,10)]
#    obj_id = 42917
#    obj_id_list = [obj_id]
#    
#    output_state = propagate_TLE(obj_id_list, UTC_list)
#    
#    print(output_state)
#    
#    for ii in range(len(UTC_list)):
#        UTC = UTC_list[ii]
#        EOP_data = get_eop_data(eop_alldata, UTC)
#        r_eci = output_state[obj_id]['r_GCRF'][ii]
#        v_eci = output_state[obj_id]['v_GCRF'][ii]
#        
#        r_ecef, v_ecef = gcrf2itrf(r_eci, v_eci, UTC, EOP_data)
#        
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
        
    
    
#    obj_id_list = [37158]
#    UTC_list = [datetime(2018, 10, 29, 0, 0, 0)]
#    
##    
##    print(output_state)
#
#    plt.close('all')
#
#    tle_dict, tle_df = get_spacetrack_tle_data(obj_id_list, UTC_list)
#    print(tle_dict)
##
#    GPS_time = datetime(2018, 10, 29, 9, 50, 0)
#    UTC0 = GPS_time - timedelta(seconds=18.)
#    UTC_list = [UTC0]
#    
#    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict, offline_flag=True)
#    
#    for obj_id in obj_id_list:
#        r_GCRF = output_state[obj_id]['r_GCRF'][0]
#        v_GCRF = output_state[obj_id]['v_GCRF'][0]
#        x_in = np.concatenate((r_GCRF, v_GCRF), axis=0)
#        print(obj_id)
#        print(x_in)
#        elem = element_conversion(x_in, 1, 0)
#        print(elem)
#    
#    pos_ecef = np.reshape([-27379.521717,  31685.387589,  10200.667234], (3,1))
#    vel_ecef = np.zeros((3,1))
#    
#    
#    # Comparison
#    eop_alldata = get_celestrak_eop_alldata(offline_flag=True)
#    EOP_data = get_eop_data(eop_alldata, UTC0)
#    
#    r_GCRF_sp3, vdum = itrf2gcrf(pos_ecef, vel_ecef, UTC0, EOP_data)
#    
#    print(r_GCRF_sp3)
#    print(r_GCRF_sp3 - r_GCRF)
#    print(np.linalg.norm(r_GCRF_sp3 - r_GCRF))
    
    
#    plot_tle_radec(tle_dict, UTC_list, sensor_list, display_flag=True)

#    plot_all_tle_common_time(obj_id_list, UTC_list)
