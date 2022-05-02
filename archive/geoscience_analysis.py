import numpy as np
from math import pi, sin, cos, asin
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pickle
import time
import copy


metis_dir = r'C:\Users\Steve\Documents\code\metis'
sys.path.append(metis_dir)

from utilities.tle_functions import get_spacetrack_tle_data, find_closest_tle_epoch
from utilities.astrodynamics import cart2kep, kep2cart, element_conversion, osc2mean
from utilities.astrodynamics import meanmot2sma, RAAN_to_LTAN, LTAN_to_RAAN, sunsynch_inclination
from utilities.constants import Re, GME, J2E, wE
from utilities.eop_functions import get_celestrak_eop_alldata, get_eop_data
from utilities.eop_functions import get_XYs2006_alldata, batch_eop_rotation_matrices, get_nutation_data
from utilities.coordinate_systems import ecef2latlonht, ric2eci, gcrf2itrf, gcrf2teme
from utilities.coordinate_systems import inc2az, dist2latlon, latlon2dist
from utilities.time_systems import mjd2dt, dt2mjd
from utilities.tle_functions import propagate_TLE as prop_TLE_full


from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs84


###############################################################################
#
# References
# 
# [1] "Handbook of Satellite Orbits: From Kepler to GPS," Capderou, M., 2014.
#
###############################################################################


###############################################################################
#
# TLE Functions
#
###############################################################################


def tle_print():
    
    # Landsat-8 Data
    landsat8_norad = 39084
    
    # Sentinel 2 Data
    sentinel_2a_norad = 40697
    sentinel_2b_norad = 42063
    
    # Retrieve TLE 
    obj_id_list = [landsat8_norad, sentinel_2a_norad, sentinel_2b_norad]
    UTC_list = [datetime(2020, 6, 6, 0, 0, 0)]
    retrieve_list = [UTC_list[0] - timedelta(days=2), UTC_list[-1] + timedelta(days=2)]
    tle_dict, dum = get_spacetrack_tle_data(obj_id_list, retrieve_list, 
                                            username='steve.gehly@gmail.com', 
                                            password='SpaceTrackPword!')
    
    print(tle_dict)
    
    # Retrieve TLE and print state vectors    
    output_state = prop_TLE_full(obj_id_list, UTC_list, tle_dict,
                                 prev_flag=True, offline_flag=False,
                                 username='steve.gehly@gmail.com', 
                                 password='SpaceTrackPword!')
    
 
    print('\n\nOrbit Elements')
    landsat8_pos = output_state[landsat8_norad]['r_GCRF'][0]
    landsat8_vel = output_state[landsat8_norad]['v_GCRF'][0]
    landsat8_cart = np.concatenate((landsat8_pos.flatten(), landsat8_vel.flatten()))
    landsat8_kep = cart2kep(landsat8_cart)
    
    sentinel_2a_pos = output_state[sentinel_2a_norad]['r_GCRF'][0]
    sentinel_2a_vel = output_state[sentinel_2a_norad]['v_GCRF'][0]
    sentinel_2a_cart = np.concatenate((sentinel_2a_pos.flatten(), sentinel_2a_vel.flatten()))
    sentinel_2a_kep = cart2kep(sentinel_2a_cart)
    
    sentinel_2b_pos = output_state[sentinel_2b_norad]['r_GCRF'][0]
    sentinel_2b_vel = output_state[sentinel_2b_norad]['v_GCRF'][0]
    sentinel_2b_cart = np.concatenate((sentinel_2b_pos.flatten(), sentinel_2b_vel.flatten()))
    sentinel_2b_kep = cart2kep(sentinel_2b_cart)
    
    # Compute LTAN data    
    UTC = UTC_list[0]
    eop_alldata = get_celestrak_eop_alldata()
    EOP_data = get_eop_data(eop_alldata, UTC)
    
    landsat8_RAAN = float(landsat8_kep[3])
    sentinel_2a_RAAN = float(sentinel_2a_kep[3])
    sentinel_2b_RAAN = float(sentinel_2b_kep[3])
    
    landsat8_LTAN = RAAN_to_LTAN(landsat8_RAAN, UTC, EOP_data)
    sentinel_2a_LTAN = RAAN_to_LTAN(sentinel_2a_RAAN, UTC, EOP_data)
    sentinel_2b_LTAN = RAAN_to_LTAN(sentinel_2b_RAAN, UTC, EOP_data)
    
    # Check RAAN values
    check1 = LTAN_to_RAAN(landsat8_LTAN, UTC, EOP_data)
    check2 = LTAN_to_RAAN(sentinel_2a_LTAN, UTC, EOP_data)
    check3 = LTAN_to_RAAN(sentinel_2b_LTAN, UTC, EOP_data)

    
    print('\nLandsat-8')
    print('SMA [km]: ', float(landsat8_kep[0]))
    print('Alt [km]: ', float(landsat8_kep[0])-6378.137)
    print('ECC: ', float(landsat8_kep[1]))
    print('INC [deg]: ', float(landsat8_kep[2]))
    print('RAAN [deg]: ', float(landsat8_kep[3]))
    print('RAAN check: ', check1)
    print('AOP [deg]: ', float(landsat8_kep[4]))
    print('TA [deg]: ', float(landsat8_kep[5]))
    print('True Long [deg]: ', float(landsat8_kep[4])+float(landsat8_kep[5]))
    print('LTAN [hours]: ', landsat8_LTAN)
    
    print('\nSentinel-2A')
    print('SMA [km]: ', float(sentinel_2a_kep[0]))
    print('Alt [km]: ', float(sentinel_2a_kep[0])-6378.137)
    print('ECC: ', float(sentinel_2a_kep[1]))
    print('INC [deg]: ', float(sentinel_2a_kep[2]))
    print('RAAN [deg]: ', float(sentinel_2a_kep[3]))
    print('RAAN check: ', check2)
    print('AOP [deg]: ', float(sentinel_2a_kep[4]))
    print('TA [deg]: ', float(sentinel_2a_kep[5]))
    print('True Long [deg]:', float(sentinel_2a_kep[4])+float(sentinel_2a_kep[5]))
    print('LTAN [hours]: ', sentinel_2a_LTAN)
    
    print('\nSentinel-2B')
    print('SMA [km]: ', float(sentinel_2b_kep[0]))
    print('Alt [km]: ', float(sentinel_2a_kep[0])-6378.137)
    print('ECC: ', float(sentinel_2b_kep[1]))
    print('INC [deg]: ', float(sentinel_2b_kep[2]))
    print('RAAN [deg]: ', float(sentinel_2b_kep[3]))
    print('RAAN check: ', check3)
    print('AOP [deg]: ', float(sentinel_2b_kep[4]))
    print('TA [deg]: ', float(sentinel_2b_kep[5]))
    print('True Long [deg]:', float(sentinel_2b_kep[4])+float(sentinel_2b_kep[5]))
    print('LTAN [hours]: ', sentinel_2b_LTAN)
    
    
    
    return


def retrieve_and_prop_TLE(UTC_start, UTC_stop, fname):
    
    print('\nPropagate TLE')
    
    # Landsat-8 Data
    landsat8_norad = 39084
    
    # Sentinel 2 Data
    sentinel_2a_norad = 40697
    sentinel_2b_norad = 42063
    
    # Retrieve TLE and print state vectors
    obj_id_list = [landsat8_norad, sentinel_2a_norad, sentinel_2b_norad]
    dt = 10.
    
    delta_t = (UTC_stop - UTC_start).total_seconds()
    UTC_list = [UTC_start + timedelta(seconds=ti) for ti in list(np.arange(0, delta_t, dt))]
#    print(UTC_list[-1])
    print(UTC_list[0], UTC_list[-1])
    
    # Retrieve TLE 
    retrieve_list = [UTC_list[0] - timedelta(days=1), UTC_list[-1] + timedelta(days=1)]
    tle_dict, dum = get_spacetrack_tle_data(obj_id_list, retrieve_list, 
                                            username='steve.gehly@gmail.com', 
                                            password='SpaceTrackPword!')
    
    # Propagate state vectors
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict, 
                                 offline_flag=False,
                                 username='steve.gehly@gmail.com', 
                                 password='SpaceTrackPword!')
    
    
    # Save output
    pklFile = open( fname, 'wb' )
    pickle.dump( [output_state], pklFile, -1 )
    pklFile.close()
    
    return


def generate_prop_TLE_files():
    
    obj_id_list = list(range(10019, 10033))
    desc = 'small_swath'
    
    # Retrieve TLE 
    tle_filename = os.path.join('../data/tle_data_candidate_orbits_2020_06_01.pkl')
    pklFile = open(tle_filename, 'rb')
    data = pickle.load(pklFile)
    tle_dict = data[0]
    pklFile.close()
    
    ###########################################################################
    # June 2020  
    ###########################################################################   
    UTC_start = datetime(2020, 6, 1)
    UTC_stop = datetime(2020, 7, 1)
    prop_filename = os.path.join('../data/tle_propagation_' + desc + '_2020_06.pkl')
    
    # Time vector
    dt = 10.    
    delta_t = (UTC_stop - UTC_start).total_seconds()
    UTC_list = [UTC_start + timedelta(seconds=ti) for ti in list(np.arange(0, delta_t, dt))]
    print(UTC_list[0], UTC_list[-1])
    
    # Propagate state vectors
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)

    # Save output    
    pklFile = open( prop_filename, 'wb' )
    pickle.dump( [output_state], pklFile, -1 )
    pklFile.close()
    
    ###########################################################################
    # July 2020  
    ###########################################################################   
    UTC_start = datetime(2020, 7, 1)
    UTC_stop = datetime(2020, 8, 1)
    prop_filename = os.path.join('../data/tle_propagation_' + desc + '_2020_07.pkl')
    
    # Time vector
    dt = 10.    
    delta_t = (UTC_stop - UTC_start).total_seconds()
    UTC_list = [UTC_start + timedelta(seconds=ti) for ti in list(np.arange(0, delta_t, dt))]
    print(UTC_list[0], UTC_list[-1])
    
    # Propagate state vectors
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)

    # Save output    
    pklFile = open( prop_filename, 'wb' )
    pickle.dump( [output_state], pklFile, -1 )
    pklFile.close()
    
    ###########################################################################
    # August 2020  
    ###########################################################################
    UTC_start = datetime(2020, 8, 1)
    UTC_stop = datetime(2020, 9, 1)
    prop_filename = os.path.join('../data/tle_propagation_' + desc + '_2020_08.pkl')
    
    # Time vector
    dt = 10.    
    delta_t = (UTC_stop - UTC_start).total_seconds()
    UTC_list = [UTC_start + timedelta(seconds=ti) for ti in list(np.arange(0, delta_t, dt))]
    print(UTC_list[0], UTC_list[-1])
    
    # Propagate state vectors
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)

    # Save output    
    pklFile = open( prop_filename, 'wb' )
    pickle.dump( [output_state], pklFile, -1 )
    pklFile.close()
    
    ###########################################################################
    # September 2020  
    ###########################################################################
    UTC_start = datetime(2020, 9, 1)
    UTC_stop = datetime(2020, 10, 1)
    prop_filename = os.path.join('../data/tle_propagation_' + desc + '_2020_09.pkl')
    
    # Time vector
    dt = 10.    
    delta_t = (UTC_stop - UTC_start).total_seconds()
    UTC_list = [UTC_start + timedelta(seconds=ti) for ti in list(np.arange(0, delta_t, dt))]
    print(UTC_list[0], UTC_list[-1])
    
    # Propagate state vectors
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)

    # Save output    
    pklFile = open( prop_filename, 'wb' )
    pickle.dump( [output_state], pklFile, -1 )
    pklFile.close()
    
    ###########################################################################
    # October 2020  
    ###########################################################################
    UTC_start = datetime(2020, 10, 1)
    UTC_stop = datetime(2020, 11, 1)
    prop_filename = os.path.join('../data/tle_propagation_' + desc + '_2020_10.pkl')
    
    # Time vector
    dt = 10.    
    delta_t = (UTC_stop - UTC_start).total_seconds()
    UTC_list = [UTC_start + timedelta(seconds=ti) for ti in list(np.arange(0, delta_t, dt))]
    print(UTC_list[0], UTC_list[-1])
    
    # Propagate state vectors
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)

    # Save output    
    pklFile = open( prop_filename, 'wb' )
    pickle.dump( [output_state], pklFile, -1 )
    pklFile.close()
    
    ###########################################################################
    # November 2020  
    ###########################################################################
    UTC_start = datetime(2020, 11, 1)
    UTC_stop = datetime(2020, 12, 1)
    prop_filename = os.path.join('../data/tle_propagation_' + desc + '_2020_11.pkl')
    
    # Time vector
    dt = 10.    
    delta_t = (UTC_stop - UTC_start).total_seconds()
    UTC_list = [UTC_start + timedelta(seconds=ti) for ti in list(np.arange(0, delta_t, dt))]
    print(UTC_list[0], UTC_list[-1])
    
    # Propagate state vectors
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)

    # Save output    
    pklFile = open( prop_filename, 'wb' )
    pickle.dump( [output_state], pklFile, -1 )
    pklFile.close()
    
    ###########################################################################
    # December 2020
    ###########################################################################
    UTC_start = datetime(2020, 12, 1)
    UTC_stop = datetime(2021, 1, 1)
    prop_filename = os.path.join('../data/tle_propagation_' + desc + '_2020_12.pkl')
    
    # Time vector
    dt = 10.    
    delta_t = (UTC_stop - UTC_start).total_seconds()
    UTC_list = [UTC_start + timedelta(seconds=ti) for ti in list(np.arange(0, delta_t, dt))]
    print(UTC_list[0], UTC_list[-1])
    
    # Propagate state vectors
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)

    # Save output    
    pklFile = open( prop_filename, 'wb' )
    pickle.dump( [output_state], pklFile, -1 )
    pklFile.close()
    
    ###########################################################################
    # January 2021 
    ###########################################################################
    UTC_start = datetime(2021, 1, 1)
    UTC_stop = datetime(2021, 2, 1)
    prop_filename = os.path.join('../data/tle_propagation_' + desc + '_2021_01.pkl')
    
    # Time vector
    dt = 10.    
    delta_t = (UTC_stop - UTC_start).total_seconds()
    UTC_list = [UTC_start + timedelta(seconds=ti) for ti in list(np.arange(0, delta_t, dt))]
    print(UTC_list[0], UTC_list[-1])
    
    # Propagate state vectors
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)

    # Save output    
    pklFile = open( prop_filename, 'wb' )
    pickle.dump( [output_state], pklFile, -1 )
    pklFile.close()
    
    ###########################################################################
    # February 2021 
    ###########################################################################
    UTC_start = datetime(2021, 2, 1)
    UTC_stop = datetime(2021, 3, 1)
    prop_filename = os.path.join('../data/tle_propagation_' + desc + '_2021_02.pkl')
    
    # Time vector
    dt = 10.    
    delta_t = (UTC_stop - UTC_start).total_seconds()
    UTC_list = [UTC_start + timedelta(seconds=ti) for ti in list(np.arange(0, delta_t, dt))]
    print(UTC_list[0], UTC_list[-1])
    
    # Propagate state vectors
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)

    # Save output    
    pklFile = open( prop_filename, 'wb' )
    pickle.dump( [output_state], pklFile, -1 )
    pklFile.close()
    
    ###########################################################################
    # March 2021 
    ###########################################################################
    UTC_start = datetime(2021, 3, 1)
    UTC_stop = datetime(2021, 4, 1)
    prop_filename = os.path.join('../data/tle_propagation_' + desc + '_2021_03.pkl')
    
    # Time vector
    dt = 10.    
    delta_t = (UTC_stop - UTC_start).total_seconds()
    UTC_list = [UTC_start + timedelta(seconds=ti) for ti in list(np.arange(0, delta_t, dt))]
    print(UTC_list[0], UTC_list[-1])
    
    # Propagate state vectors
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)

    # Save output    
    pklFile = open( prop_filename, 'wb' )
    pickle.dump( [output_state], pklFile, -1 )
    pklFile.close()
    
    ###########################################################################
    # April 2021 
    ###########################################################################
    UTC_start = datetime(2021, 4, 1)
    UTC_stop = datetime(2021, 5, 1)
    prop_filename = os.path.join('../data/tle_propagation_' + desc + '_2021_04.pkl')
    
    # Time vector
    dt = 10.    
    delta_t = (UTC_stop - UTC_start).total_seconds()
    UTC_list = [UTC_start + timedelta(seconds=ti) for ti in list(np.arange(0, delta_t, dt))]
    print(UTC_list[0], UTC_list[-1])
    
    # Propagate state vectors
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)

    # Save output    
    pklFile = open( prop_filename, 'wb' )
    pickle.dump( [output_state], pklFile, -1 )
    pklFile.close()
    
    ###########################################################################
    # May 2021 
    ###########################################################################
    UTC_start = datetime(2021, 5, 1)
    UTC_stop = datetime(2021, 6, 1)
    prop_filename = os.path.join('../data/tle_propagation_' + desc + '_2021_05.pkl')
    
    # Time vector
    dt = 10.    
    delta_t = (UTC_stop - UTC_start).total_seconds()
    UTC_list = [UTC_start + timedelta(seconds=ti) for ti in list(np.arange(0, delta_t, dt))]
    print(UTC_list[0], UTC_list[-1])
    
    # Propagate state vectors
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)

    # Save output    
    pklFile = open( prop_filename, 'wb' )
    pickle.dump( [output_state], pklFile, -1 )
    pklFile.close()
    
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
    
    print('propagate_TLE')
    
    total_prop = 0.
    tle_epoch_time = 0.

    # If no TLE information is provided, retrieve from sources as needed
    if len(tle_dict) == 0:

        # Retrieve latest TLE data from space-track.org
        tle_dict, tle_df = \
            get_spacetrack_tle_data(obj_id_list, UTC_list, username, password)

    # Loop over objects
    output_state = {}
    for obj_id in obj_id_list:

        line1_list = tle_dict[obj_id]['line1_list']
        line2_list = tle_dict[obj_id]['line2_list']

        output_state[obj_id] = {}
        output_state[obj_id]['UTC'] = []
        output_state[obj_id]['r_TEME'] = []
        output_state[obj_id]['v_TEME'] = []
        
        print(obj_id)

        # Loop over times
        ii = 0
        for UTC in UTC_list:
            
            ii += 1
            if ii % 1000 == 0:
                print(obj_id, UTC)
                        
            # Find the closest TLE by epoch
            epoch_start = time.time()
            line1, line2 = find_closest_tle_epoch(line1_list, line2_list, UTC, prev_flag=True)
            
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
            
        print(obj_id)
        print('TLE epoch find: ', tle_epoch_time)
        print('Prop: ', total_prop)
        


    return output_state

###############################################################################
#
# Coordinate Frame Functions
#
###############################################################################


def compute_frame_rotations(UTC_start, UTC_stop, fname):
    
    
    dt = 10.
    
    delta_t = (UTC_stop - UTC_start).total_seconds()
    UTC_list = [UTC_start + timedelta(seconds=ti) for ti in list(np.arange(0, delta_t, dt))]
    
    eop_alldata = get_celestrak_eop_alldata()
    
    GCRF_TEME_list, ITRF_GCRF_list = \
        batch_eop_rotation_matrices(UTC_list, eop_alldata, increment=dt)
        
    
    # Save output    
    pklFile = open( fname, 'wb' )
    pickle.dump( [UTC_list, GCRF_TEME_list, ITRF_GCRF_list], pklFile, -1 )
    pklFile.close()
    
    return


def test_frame_rotations():
    
    # Load data
    fname = os.path.join('..//data/tle_propagation.pkl')
    pklFile = open(fname, 'rb')
    data = pickle.load(pklFile)
    output_state = data[0]
    pklFile.close()
    
    fname = os.path.join('..//data/frame_rotations.pkl')
    pklFile = open(fname, 'rb')
    data = pickle.load(pklFile)
    GCRF_TEME_list = data[0]
    ITRF_GCRF_list = data[1]
    pklFile.close()
    
    
    obj_id = 39084    
    N = len(output_state[obj_id]['UTC'])
    check_mat = np.zeros((3, N))
    for ii in range(N):
        r_GCRF = output_state[obj_id]['r_GCRF'][ii]
        v_GCRF = output_state[obj_id]['v_GCRF'][ii]
        r_ITRF = output_state[obj_id]['r_ITRF'][ii]
        v_ITRF = output_state[obj_id]['v_ITRF'][ii]
        r_TEME = output_state[obj_id]['r_TEME'][ii]
        v_TEME = output_state[obj_id]['v_TEME'][ii]
        
        # Recompute rotations
        GCRF_TEME = GCRF_TEME_list[ii]
        ITRF_GCRF = ITRF_GCRF_list[ii]
        
        check_r_GCRF = np.dot(GCRF_TEME, r_TEME)
        check_v_GCRF = np.dot(GCRF_TEME, v_TEME)
        
        check_r_ITRF = np.dot(ITRF_GCRF, r_GCRF)
        
        check = check_r_GCRF - r_GCRF + check_v_GCRF - v_GCRF + check_r_ITRF - r_ITRF
        
        check_mat[:,ii] = check.flatten()
        
        
    print(sum(sum(check_mat)))
        
        
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(list(range(N)), check_mat[0,:])
    plt.subplot(3,1,2)
    plt.plot(list(range(N)), check_mat[1,:])
    plt.subplot(3,1,3)
    plt.plot(list(range(N)), check_mat[2,:])
    
    plt.show()
        
        
    
    
    return



def validate_frames_and_prop():
    
    # Load data
    fname = os.path.join('..//data/tle_propagation_primary_targets_2020_06.pkl')
    pklFile = open(fname, 'rb')
    data = pickle.load(pklFile)
    output_state = data[0]
    pklFile.close()
    
    fname = os.path.join('..//data/frame_rotations_2020_06.pkl')
    pklFile = open(fname, 'rb')
    data = pickle.load(pklFile)
    UTC_list_frame = data[0]
    GCRF_TEME_list = data[1]
    ITRF_GCRF_list = data[2]
    pklFile.close()
    
    fname = os.path.join('..//data/check_tle_propagation.pkl')
    pklFile = open(fname, 'rb')
    data = pickle.load(pklFile)
    output_state_check = data[0]
    pklFile.close()
    
    
    obj_id = 39084
    UTC_list_obj = output_state[obj_id]['UTC']
    UTC_list_check = output_state_check[obj_id]['UTC']
    
    print(UTC_list_obj[0], UTC_list_obj[-1])
    print(UTC_list_check[0], UTC_list_check[-1])
    print(UTC_list_frame[0], UTC_list_frame[-1])
    
    ind = UTC_list_obj.index(UTC_list_check[0])
    print(ind)
    print(UTC_list_obj[ind])
    
    max_TEME = 0.
    max_GCRF = 0.
    max_ITRF = 0.
    for ii in range(ind, len(UTC_list_obj)):
        
        r_TEME_obj = output_state[obj_id]['r_TEME'][ii]
        v_TEME_obj = output_state[obj_id]['v_TEME'][ii]
        
        GCRF_TEME = GCRF_TEME_list[ii]
        ITRF_GCRF = ITRF_GCRF_list[ii]
        
        r_GCRF_obj = np.dot(GCRF_TEME, r_TEME_obj)
        r_ITRF_obj = np.dot(ITRF_GCRF, r_GCRF_obj)
        
        r_TEME_check = output_state_check[obj_id]['r_TEME'][ii-ind]
        r_GCRF_check = output_state_check[obj_id]['r_GCRF'][ii-ind]
        r_ITRF_check = output_state_check[obj_id]['r_ITRF'][ii-ind]
        
        TEME_diff = r_TEME_obj - r_TEME_check
        GCRF_diff = r_GCRF_obj - r_GCRF_check
        ITRF_diff = r_ITRF_obj - r_ITRF_check
        
        if np.max(abs(TEME_diff)) > max_TEME:
            max_TEME = np.max(abs(TEME_diff))
            
        if np.max(abs(GCRF_diff)) > max_GCRF:
            max_GCRF = np.max(abs(GCRF_diff))
            
        if np.max(abs(ITRF_diff)) > max_ITRF:
            max_ITRF = np.max(abs(ITRF_diff))
            
            
    print(max_TEME, max_GCRF, max_ITRF)
    

    
    return


###############################################################################
#
# Coverage Analysis Functions
#
###############################################################################


def generate_fov_dict():
    
    # Landsat-8 Data
    landsat8_norad = 39084
    
    # Sentinel 2 Data
    sentinel_2a_norad = 40697
    sentinel_2b_norad = 42063
    
    # Store FOV data in radians
    fov_dict = {}
    fov_dict[landsat8_norad] = 15.*pi/180.
    fov_dict[sentinel_2a_norad] = 21.*pi/180.
    fov_dict[sentinel_2b_norad] = 21.*pi/180.
    
    # Candidate orbits
    alpha = 40./Re
    a = Re + 600.
    rho = np.sqrt(Re**2. + a**2. - 2.*Re*a*cos(alpha))
    f = asin((sin(alpha)/rho)*Re)

    fov_dict[0] = 2*f
    fov_dict[1] = fov_dict[0]
    
    fov_dict[10001] = fov_dict[0]
    fov_dict[10002] = fov_dict[0]
    fov_dict[10003] = fov_dict[0]
    fov_dict[10004] = fov_dict[0]
    fov_dict[10005] = fov_dict[0]
    fov_dict[10006] = fov_dict[0]
    fov_dict[10007] = fov_dict[0]
    fov_dict[10008] = fov_dict[0]
    fov_dict[10009] = fov_dict[0]
    fov_dict[10010] = fov_dict[0]
    fov_dict[10011] = fov_dict[0]
    fov_dict[10012] = fov_dict[0]
    fov_dict[10013] = fov_dict[0]
    fov_dict[10014] = fov_dict[0]
    fov_dict[10015] = fov_dict[0]
    fov_dict[10016] = fov_dict[0]
    fov_dict[10017] = fov_dict[0]
    fov_dict[10018] = fov_dict[0]
    fov_dict[10019] = fov_dict[0]
    fov_dict[10020] = fov_dict[0]
    fov_dict[10021] = fov_dict[0]
    fov_dict[10022] = fov_dict[0]
    fov_dict[10023] = fov_dict[0]
    fov_dict[10024] = fov_dict[0]
    fov_dict[10025] = fov_dict[0]
    fov_dict[10026] = fov_dict[0]
    fov_dict[10027] = fov_dict[0]
    fov_dict[10028] = fov_dict[0]
    fov_dict[10029] = fov_dict[0]
    fov_dict[10030] = fov_dict[0]
    fov_dict[10031] = fov_dict[0]
    fov_dict[10032] = fov_dict[0]
    
    
    
    
    
    
    
    fov_dict[20001] = fov_dict[0]
    fov_dict[20002] = fov_dict[0]
    fov_dict[20003] = fov_dict[0]
    fov_dict[20004] = fov_dict[0]
    fov_dict[20005] = fov_dict[0]
    fov_dict[20006] = fov_dict[0]
    fov_dict[20007] = fov_dict[0]
    fov_dict[20008] = fov_dict[0]
    fov_dict[20009] = fov_dict[0]
    fov_dict[20010] = fov_dict[0]
    fov_dict[20011] = fov_dict[0]
    fov_dict[20012] = fov_dict[0]
    fov_dict[20013] = fov_dict[0]
    fov_dict[20014] = fov_dict[0]
    fov_dict[20015] = fov_dict[0]
    fov_dict[20016] = fov_dict[0]
    
    return fov_dict


def compute_anglediff_deg(angle1, angle2):
    '''
    This function computes the difference between two angles accounting for
    rollover.
    
    Angles in degrees.
    
    '''
    
    diff = angle1 - angle2
    if diff < -180.:
        diff += 360.
    elif diff > 180.:
        diff -= 360.
    
    return diff


def compute_intersect(p1, p2, p3, p4):
    '''
    This function computes the intersection of two lines defined by points
    
    line1 = p1 to p2
    line2 = p3 to p4
    
    Reference
    ------
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    
    '''
    
    x1 = float(p1[0])
    y1 = float(p1[1])
    x2 = float(p2[0])
    y2 = float(p2[1])
    x3 = float(p3[0])
    y3 = float(p3[1])
    x4 = float(p4[0])
    y4 = float(p4[1])
    
    # Wrap angles to [-180, 180]
    if x1 < -180.:
        x1 += 360.
    if x1 > 180.:
        x1 -= 360.
        
    if x2 < -180.:
        x2 += 360.
    if x2 > 180.:
        x2 -= 360.
        
    if x3 < -180.:
        x3 += 360.
    if x3 > 180.:
        x3 -= 360.
        
    if x4 < -180.:
        x4 += 360.
    if x4 > 180.:
        x4 -= 360.
    
    # Keep longitudes in same quadrant
    q1 = compute_quadrant(x1)
    q2 = compute_quadrant(x2)
    q3 = compute_quadrant(x3)
    q4 = compute_quadrant(x4)
    

    
    if q1 == 2:
        if q2 == 3:
            x2 += 360.
        if q3 == 3:
            x3 += 360.
        if q4 == 3:
            x4 += 360.
    
    if q1 == 3:
        if q2 == 2:
            x2 -= 360.
        if q3 == 2:
            x3 -= 360.
        if q4 == 2:
            x4 -= 360.
            
#    print(q1, q2, q3, q4)        
#    print(x1, x2, x3, x4)

    
    D = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    
    x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4))/D
    y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4))/D    
    
#    print(x, y)
    
    # Wrap x to [-180, 180]
    if x < -180:
        x += 360.
    if x > 180:
        x -= 360.
    
    return x, y


def compute_quadrant(theta):
    
    theta = theta % 360.
    
    if theta >= 0. and theta < 90.:
        quad = 1
    elif theta < 180.:
        quad = 2
    elif theta < 270.:
        quad = 3
    elif theta < 360.:
        quad = 4
        
        
    return quad


def define_wrs2_grid():
    '''
    
    Reference
    ------
    https://www.usgs.gov/media/files/landsat-wrs-2-corner-points
    
    '''
    
    wrs2_file = os.path.join('..//data/WRScornerPoints_0.csv')
    wrs2_df = pd.read_csv(wrs2_file)
    
    path_list = wrs2_df['PATH'].tolist()
    row_list = wrs2_df['ROW'].tolist()
    ctr_lat_list = wrs2_df['CTR LAT'].tolist()
    ctr_lon_list = wrs2_df['CTR LON'].tolist()
    
    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,
                urcrnrlon=180,resolution='c')
    
    
    wrs2_dict = {}
    wrs2_dict['path_list'] = path_list
    wrs2_dict['row_list'] = row_list
    wrs2_dict['ctr_lat_list'] = ctr_lat_list
    wrs2_dict['ctr_lon_list'] = ctr_lon_list
    
    
    path_min = min(path_list)
    path_max = max(path_list)
    
    row_min = min(row_list)
    row_max = 122  # descending (dayside) only
    
    wrs2_lat = np.zeros((row_max, path_max))
    wrs2_lon = np.zeros((row_max, path_max))
    wrs2_poslon = np.zeros((row_max, path_max))
    wrs2_neglon = np.zeros((row_max, path_max))
        
    for ii in range(len(path_list)):
        path_ind = int(path_list[ii] - 1)
        row_ind = int(row_list[ii] - 1)
        
        if row_ind > (row_max-1):
            continue
        
        lat = ctr_lat_list[ii]
        lon = ctr_lon_list[ii]
        
        wrs2_lat[row_ind, path_ind] = lat
        wrs2_lon[row_ind, path_ind] = lon
        wrs2_poslon[row_ind, path_ind] = lon
        wrs2_neglon[row_ind, path_ind] = lon
        
        if lon > 0.:
            wrs2_neglon[row_ind, path_ind] -= 360.
        if lon < 0.:
            wrs2_poslon[row_ind, path_ind] += 360.
        
    # Do is_land calculations here and store results
    land_flag = np.zeros((row_max, path_max))
    for ii in range(row_max):
        for jj in range(path_max):
            lat = wrs2_lat[ii,jj]
            lon = wrs2_lon[ii,jj]
            
            flag = m.is_land(lon, lat)
            if flag:
                land_flag[ii,jj] = 1.
        
#    plot_lon = []
#    plot_lat = []
#    for ii in range(row_max):
#        for jj in range(path_max):
#            if land_flag[ii,jj]:
#                plot_lon.append(wrs2_lon[ii,jj])
#                plot_lat.append(wrs2_lat[ii,jj])
#        
#
#    
#    # Generate plot
#    plt.figure()
#    plt.plot(plot_lon, plot_lat, 'go', ms=1)
#            
#    m.drawcoastlines()
#    m.drawmeridians(np.arange(-180, 180, 45))
#    m.drawparallels(np.arange(-90, 90, 45))
#    m.drawmapboundary()
#    m.drawcountries()
#    plt.yticks(np.arange(-90, 91, 45))
#    plt.xticks(np.arange(-180, 181, 45))
#
#
#    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,
#                urcrnrlon=180,resolution='c')
#            
#    fig = plt.figure()
#    plt.contourf(wrs2_lon, wrs2_lat, land_flag, cmap=plt.cm.plasma)
#    
#    fig = plt.figure()
#    plt.contourf(land_flag, cmap=plt.cm.plasma)
    
#    m.drawcoastlines()
#    m.drawmeridians(np.arange(-180, 180, 45))
#    m.drawparallels(np.arange(-90, 90, 45))
#    m.drawmapboundary()
#    m.drawcountries()
#    plt.yticks(np.arange(-90, 91, 45))
#    plt.xticks(np.arange(-180, 181, 45))
#
#    plt.colorbar()
#    
#    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,
#                urcrnrlon=180,resolution='c')
    
#    fig = plt.figure()
#    for ii in range(row_max):
#        for jj in range(path_max):
#            lon = wrs2_lon[ii,jj]
#            lat = wrs2_lat[ii,jj]
#            val = land_flag[ii,jj]
#            
#            if val < 1:
#                plt.plot(lon, lat, 'bo')
#            if val >= 1:
#                plt.plot(lon, lat, 'go')
#                
#    m.drawcoastlines()
#    m.drawmeridians(np.arange(-180, 180, 45))
#    m.drawparallels(np.arange(-90, 90, 45))
#    m.drawmapboundary()
#    m.drawcountries()
#    plt.yticks(np.arange(-90, 91, 45))
#    plt.xticks(np.arange(-180, 181, 45))
    
    
    
#    plt.show()
        
    
    return wrs2_dict, wrs2_lon, wrs2_lat, wrs2_neglon, wrs2_poslon, land_flag


def compute_wrs2_path_row(lon, lat, wrs2_lon, wrs2_lat, wrs2_neglon, wrs2_poslon):
    
    
#    path_list = wrs2_dict['path_list']
#    row_list = wrs2_dict['row_list']
#    ctr_lat_list = wrs2_dict['ctr_lat_list']
#    ctr_lon_list = wrs2_dict['ctr_lon_list']
#    
#    row_lat_values = sorted(list(set(ctr_lat_list)))
#    row_inds = [ctr_lat_list.index(lat_ii) for lat_ii in row_lat_values]
#    reduced_row_list1 = [row_list[ii] for ii in row_inds]
#    
#    print(row_lat_values)
#    print(row_inds)
#    print(reduced_row_list1)
#    mistake
#    
#    start = time.time()
#    
#    # Determine candidate row values, keep only those on descending/dayside
#    lat_compare = [abs(lat - lat_ii) for lat_ii in ctr_lat_list]
#    lat_inds = [int(ii) for ii in list(np.argwhere(np.array(lat_compare) < 2.))]
#    candidate_rows = list(set([row_list[ii] for ii in lat_inds]))
#    candidate_rows = [row_ii for row_ii in candidate_rows if row_ii < 123]
#
#    lat_inds2 = []
#    for row in candidate_rows:
#        lat_inds2.extend([i for i,x in enumerate(row_list) if x == row])
#            
#    # Reduce lists to candidates from latitude/row selection
#    reduced_row_list = [row_list[ii] for ii in lat_inds2]
#    reduced_path_list = [path_list[ii] for ii in lat_inds2]
#    reduced_lat_list = [ctr_lat_list[ii] for ii in lat_inds2]
#    reduced_lon_list = [ctr_lon_list[ii] for ii in lat_inds2]
#    lon_compare = [abs(compute_anglediff_deg(lon, lon_ii)) for lon_ii in reduced_lon_list]
#    lon_inds = [int(ii) for ii in list(np.argwhere(np.array(lon_compare) < 2.))]
#    
##    min_ind = lon_compare.index(min(lon_compare))
#    
#    print('row path reduction', time.time() - start)
#
#
#    start2 = time.time()
#
#    min_ind = 0
#    min_dist = 1e6
#    for ii in lon_inds:
#        wrs2_lat = reduced_lat_list[ii]
#        wrs2_lon = reduced_lon_list[ii]
#        
#        lon_dist = compute_anglediff_deg(lon, wrs2_lon)
#        lat_dist = lat - wrs2_lat
#        
#        s = np.sqrt(lon_dist**2. + lat_dist**2.)
#        
##        s, dum, dum = latlon2dist(lat, lon, wrs2_lat, wrs2_lon)
#        
#        if s < min_dist:
#            min_dist = float(s)
#            min_ind = ii
#            
#    print('distance calc', time.time() - start2)
#            
#    row = reduced_row_list[min_ind]
#    path = reduced_path_list[min_ind]
#    wrs2_lat = reduced_lat_list[min_ind]
#    wrs2_lon = reduced_lon_list[min_ind]
    
    start = time.time()
    
#    min_dist = 1e6
#    for ii in range(wrs2_lon.shape[0]):
#        for jj in range(wrs2_lon.shape[1]):
#            
#            lon_dist = compute_anglediff_deg(lon, wrs2_lon[ii,jj])
#            lat_dist = lat - wrs2_lat[ii,jj]
#            
#            dist = np.sqrt(lon_dist**2. + lat_dist**2.)
#            
#            if dist < min_dist:
#                row_ind = ii
#                path_ind = jj
#                min_dist = float(dist)
    
    
    # Longitude from [-180, 180]
    if lon < -180.:
        lon += 360.
    if lon > 180.:
        lon -= 360.
        
    if lon < -170.:
        lon_diff = lon - wrs2_neglon
    elif lon > 170.:
        lon_diff = lon - wrs2_poslon
    else:
        lon_diff = lon - wrs2_lon
        
        
    lat_diff = lat - wrs2_lat
    
    diff_grid = np.multiply(lon_diff, lon_diff) + np.multiply(lat_diff, lat_diff)
    
    inds = np.argwhere(diff_grid == np.min(diff_grid))
    row_ind = int(inds[0][0])
    path_ind = int(inds[0][1])
    
#    print(lon_diff[row_ind, path_ind])
#    print(wrs2_neglon[row_ind, path_ind])
#    print(wrs2_lon[row_ind, path_ind])
        
    
    path = path_ind + 1
    row = row_ind + 1
    pathrow_lon = wrs2_lon[row_ind, path_ind]
    pathrow_lat = wrs2_lat[row_ind, path_ind]
    
#    print('find grid point time', time.time() - start)
    
    return path, row, pathrow_lon, pathrow_lat


def compute_paths(tle_file, frame_file, path_file, dayside_flag='desc', R=Re, GM=GME):
    
    start = time.time()
    
    # Load TLE and Frame Rotation Data
    pklFile = open(tle_file, 'rb')
    data = pickle.load(pklFile)
    output_state = data[0]
    pklFile.close()
    
    pklFile = open(frame_file, 'rb')
    data = pickle.load(pklFile)
    UTC_list_frame = data[0]
    GCRF_TEME_list = data[1]
    ITRF_GCRF_list = data[2]
    pklFile.close()
    
    # Compare UTC times as error check
    obj_id_list = list(output_state.keys())
    UTC_list_obj = output_state[obj_id_list[0]]['UTC']
    
    if (UTC_list_obj[0] != UTC_list_frame[0]) or (len(UTC_list_obj) != len(UTC_list_frame)):
        print('Error on UTC Times!!')
        print('object', UTC_list_obj[0])
        print('frame', UTC_list_frame[0])
        mistake
        
    # Generate fov dictionary
    fov_dict = generate_fov_dict()
        
    # Loop over objects    
    path_dict = {}
    for obj_id in obj_id_list:
        
        print(obj_id)
        print('Compute Paths Time: ', time.time() - start)
        
        # Initialize 
        path_dict[obj_id] = {}
        path_ind = 0
        path_dict[obj_id][path_ind] = {}
        path_dict[obj_id][path_ind]['UTC'] = []
        path_dict[obj_id][path_ind]['UTC_zerolat'] = 0.
        path_dict[obj_id][path_ind]['lon_list'] = []
        path_dict[obj_id][path_ind]['lat_list'] = []
        path_dict[obj_id][path_ind]['lonE_list'] = []
        path_dict[obj_id][path_ind]['latE_list'] = []
        path_dict[obj_id][path_ind]['lonW_list'] = []
        path_dict[obj_id][path_ind]['latW_list'] = []
        min_lat = 100.       
        
        UTC_list_obj = output_state[obj_id]['UTC']
        UTC_prev = UTC_list_obj[0]
        dt = (output_state[obj_id]['UTC'][1] - output_state[obj_id]['UTC'][0]).total_seconds()
        
        # Retrieve field of view in radians
        fov = fov_dict[obj_id]
        
        # Loop over times and rotate coordinates        
        for ii in range(len(UTC_list_obj)):
#        for ii in range(0, 6*180):
            
            # Retrieve data and rotate frames
            UTC = output_state[obj_id]['UTC'][ii]
            r_TEME = output_state[obj_id]['r_TEME'][ii]
            v_TEME = output_state[obj_id]['v_TEME'][ii]
            
#            print(UTC)
            
            GCRF_TEME = GCRF_TEME_list[ii]
            ITRF_GCRF = ITRF_GCRF_list[ii]
            
            r_GCRF = np.dot(GCRF_TEME, r_TEME)
            v_GCRF = np.dot(GCRF_TEME, v_TEME)
            r_ITRF = np.dot(ITRF_GCRF, r_GCRF)
            
            # Compute orbit parameters
            cart = np.concatenate((r_GCRF, v_GCRF), axis=0)
            kep = cart2kep(cart, GM)            
            
            a = float(kep[0])
            inc = float(kep[2])
            w = float(kep[4])
            theta = float(kep[5])
            true_long = (w + theta) % 360.
            
            # Check dayside conditions
            if dayside_flag == 'desc':
                
                # Only compute path if we are approaching descending node
                if true_long < 90. or true_long > 270.:
                    continue
                
            
            # Check if this is a new path and initialize output
            if (UTC - UTC_prev).total_seconds() > dt:
                
                # Check if any data has been saved in Path 0 yet
                # If so, increment and initialize the next path
                if len(path_dict[obj_id][0]['UTC']) > 0:
                    path_ind += 1
                    
                    path_dict[obj_id][path_ind] = {}
                    path_dict[obj_id][path_ind]['UTC'] = []
                    path_dict[obj_id][path_ind]['UTC_zerolat'] = 0.
                    path_dict[obj_id][path_ind]['lon_list'] = []
                    path_dict[obj_id][path_ind]['lat_list'] = []
                    path_dict[obj_id][path_ind]['lonE_list'] = []
                    path_dict[obj_id][path_ind]['latE_list'] = []
                    path_dict[obj_id][path_ind]['lonW_list'] = []
                    path_dict[obj_id][path_ind]['latW_list'] = []
                    min_lat = 100.
                    
            # Compute half-swath angle and lat/long with offsets
            f = (fov/2.)
            zeta = asin(a*sin(f)/R)
            alpha = zeta - f
            half_swath_km = alpha*R
                        
            # Compute latitude, longitude of main groundtrack
            lat, lon, ht = ecef2latlonht(r_ITRF)
            
            # Compute azimuth direction for swath
            az_asc, az_desc = inc2az(lat, inc)
            if dayside_flag == 'desc':
                az = az_desc
            
            if az > 180.:
                alpha_east = az - 90.
                alpha_west = az + 90.
            else:
                alpha_east = az + 90.
                alpha_west = az - 90.
            
            # Compute latitude, longitude of swath in each direction
            latE, lonE, dum = dist2latlon(lat, lon, half_swath_km, alpha_east)
            latW, lonW, dum = dist2latlon(lat, lon, half_swath_km, alpha_west)            
            
            # Check for zero latitude
            if abs(lat) < min_lat:
                path_dict[obj_id][path_ind]['UTC_zerolat'] = UTC
                min_lat = abs(lat)
            
            # Store data
            path_dict[obj_id][path_ind]['UTC'].append(UTC)
            path_dict[obj_id][path_ind]['lon_list'].append(lon)
            path_dict[obj_id][path_ind]['lat_list'].append(lat)
            path_dict[obj_id][path_ind]['lonE_list'].append(lonE)
            path_dict[obj_id][path_ind]['latE_list'].append(latE)
            path_dict[obj_id][path_ind]['lonW_list'].append(lonW)
            path_dict[obj_id][path_ind]['latW_list'].append(latW)
            
            # Update UTC_prev
            UTC_prev = copy.copy(UTC)
            

    
#    lon_list = path_dict[obj_id][0]['lon_list']
#    lat_list = path_dict[obj_id][0]['lat_list']
#    lonE_list = path_dict[obj_id][0]['lonE_list']
#    latE_list = path_dict[obj_id][0]['latE_list']
#    lonW_list = path_dict[obj_id][0]['lonW_list']
#    latW_list = path_dict[obj_id][0]['latW_list']
#    
#    lon_list.extend(path_dict[obj_id][1]['lon_list'])
#    lat_list.extend(path_dict[obj_id][1]['lat_list'])
#    lonE_list.extend(path_dict[obj_id][1]['lonE_list'])
#    latE_list.extend(path_dict[obj_id][1]['latE_list'])
#    lonW_list.extend(path_dict[obj_id][1]['lonW_list'])
#    latW_list.extend(path_dict[obj_id][1]['latW_list'])
#      
#            
#    # Generate plot
#    plt.figure()
#    plt.plot(lon_list, lat_list, 'bo', ms=1)
#    plt.plot(lonE_list, latE_list, 'ro', ms=1)
#    plt.plot(lonW_list, latW_list, 'go', ms=1)
#            
#    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,
#                urcrnrlon=180,resolution='c')
#    m.drawcoastlines()
#    m.drawmeridians(np.arange(-180, 180, 45))
#    m.drawparallels(np.arange(-90, 90, 45))
#    m.drawmapboundary()
#    m.drawcountries()
#    plt.yticks(np.arange(-90, 91, 45))
#    plt.xticks(np.arange(-180, 181, 45))
#
#    
#    plt.show()     
            
    
    print('Compute Paths Time: ', time.time() - start)
    
    # Save output    
    pklFile = open( path_file, 'wb' )
    pickle.dump( [path_dict], pklFile, -1 )
    pklFile.close()
    
    
    return


def compute_coincidence(path1, path2, alpha1, alpha2, latlim1, latlim2, 
                        coincident_dict = {}, cotime_limit=1800.):
    
        
    # Initialize output
    if len(coincident_dict) == 0:
        coincident_dict = {}
        coincident_ind = -1
    else:
        coincident_dict = copy.copy(coincident_dict)
        coincident_ind = sorted(list(coincident_dict.keys()))[-1]
    
    # Generate list of times from path2 dictionary
    UTC2_list = [path2[ii]['UTC_zerolat'] for ii in list(path2.keys())]
    
    # Loop over paths in path1
    for path1_ind in path1:
        
        #######################################################################
        #
        # Note that these check for being near the equator will truncate paths
        # that occur over the dividing line of months. Should be mimimal impact
        # as this is rare (max 12 per year) but right way to fix it would be
        # in compute_paths()
        #
        #######################################################################
        
        # Check if UTC_zerolat is actually near equator
        UTC1 = path1[path1_ind]['UTC_zerolat']
        loc_ind = path1[path1_ind]['UTC'].index(UTC1)
        lat_check = path1[path1_ind]['lat_list'][loc_ind]
        if abs(lat_check) > 2.:
            continue
        
        
        # Find index in path2 for nearest path time to current path1 entry
        diff = [abs((UTC2 - UTC1).total_seconds()) for UTC2 in UTC2_list]
        path2_ind = diff.index(min(diff))
        UTC2 = path2[path2_ind]['UTC_zerolat']
        
        # Check if this entry is actually near equator
        loc_ind = path2[path2_ind]['UTC'].index(UTC2)
        lat_check = path2[path2_ind]['lat_list'][loc_ind]
        if abs(lat_check) > 2.:
            continue
        
#        print(UTC1)
#        print(UTC2)
        
        if UTC1 < datetime(2020, 6, 1, 2, 0, 0):
            continue
#        
#        if UTC1 > datetime(2020, 6, 6, 17, 0, 0):
#            mistake
        
        # Skip if time difference is greater than cotime_limit
        if diff[path2_ind] > cotime_limit:
            continue
        
        # Check longitude at equator
        path1_zerolat_ind = path1[path1_ind]['UTC'].index(UTC1)
        path2_zerolat_ind = path2[path2_ind]['UTC'].index(UTC2)
        eq_lon1 = path1[path1_ind]['lon_list'][path1_zerolat_ind]
        eq_lon2 = path2[path2_ind]['lon_list'][path2_zerolat_ind]
        
#        lon_diff_deg = compute_anglediff_deg(eq_lon1, eq_lon2)      
        
        
        
#        # If paths are farther apart than combined swath, skip
#        if abs(lon_diff_deg * pi/180.) > (alpha1 + alpha2):
#            continue
        
        print('\nCoincident Analysis')
#        print(path1_ind)
#        print(path2_ind)
#        print(latlim1, latlim2)
        
        
        # Remaining paths are close in time 
        # Initialize output
        coincident_ind += 1
        print('coincident_ind', coincident_ind)
        print(UTC1, UTC2)
        print(eq_lon1, eq_lon2)
        
        coincident_dict[coincident_ind] = {}
        coincident_dict[coincident_ind]['UTC1'] = []
        coincident_dict[coincident_ind]['path1_ind'] = path1_ind
        coincident_dict[coincident_ind]['path2_ind'] = path2_ind
        coincident_dict[coincident_ind]['UTC1_zerolat'] = UTC1
        coincident_dict[coincident_ind]['UTC2_zerolat'] = UTC2
        coincident_dict[coincident_ind]['eq_lon1'] = eq_lon1
        coincident_dict[coincident_ind]['eq_lon2'] = eq_lon2
        coincident_dict[coincident_ind]['path1_lat'] = []
        coincident_dict[coincident_ind]['path1_lon'] = []
        coincident_dict[coincident_ind]['path2_lat'] = []
        coincident_dict[coincident_ind]['path2_lon'] = []
        coincident_dict[coincident_ind]['lonE'] = []
        coincident_dict[coincident_ind]['latE'] = []
        coincident_dict[coincident_ind]['lonW'] = []
        coincident_dict[coincident_ind]['latW'] = []
        coincident_dict[coincident_ind]['overlap_km'] = []
        coincident_dict[coincident_ind]['overlap_frac'] = []


        # Retrieve latitude/longitude data
        path1_lat = path1[path1_ind]['lat_list']
        path1_lon = path1[path1_ind]['lon_list']
        path1_lonE = path1[path1_ind]['lonE_list']
        path1_latE = path1[path1_ind]['latE_list']
        path1_lonW = path1[path1_ind]['lonW_list']
        path1_latW = path1[path1_ind]['latW_list']
        
        path2_lat = path2[path2_ind]['lat_list']
        path2_lon = path2[path2_ind]['lon_list']
        path2_lonE = path2[path2_ind]['lonE_list']
        path2_latE = path2[path2_ind]['latE_list']
        path2_lonW = path2[path2_ind]['lonW_list']
        path2_latW = path2[path2_ind]['latW_list']
        
        
        for ii in range(len(path1_lat)):
            
            
            # Check path1 latitude value against constraint
            lat1 = path1_lat[ii]            
            if lat1 > latlim1[0] or lat1 < latlim1[1]:
#                coincident_dict[coincident_ind]['overlap_km'].append(0.)
#                coincident_dict[coincident_ind]['overlap_frac'].append(0.)
#                coincident_dict[coincident_ind]['path1_lat'].append(0.)
#                coincident_dict[coincident_ind]['path1_lon'].append(0.)
#                coincident_dict[coincident_ind]['path2_lat'].append(0.)
#                coincident_dict[coincident_ind]['path2_lon'].append(0.)
#                coincident_dict[coincident_ind]['lonE'].append(0.)
#                coincident_dict[coincident_ind]['latE'].append(0.)
#                coincident_dict[coincident_ind]['lonW'].append(0.)
#                coincident_dict[coincident_ind]['latW'].append(0.)
                continue
            
            
            
            # Find index of closest latitude in path2
            lat_diff = [abs(lat1 - lat2_jj) for lat2_jj in path2_lat]
            
            # Skip if too far away
            if min(lat_diff) > 3.:
                continue
            
            jj = lat_diff.index(min(lat_diff))
            lat2 = path2_lat[jj]
            if lat2 > latlim2[0] or lat2 < latlim2[1]:
#                coincident_dict[coincident_ind]['overlap_km'].append(0.)
#                coincident_dict[coincident_ind]['overlap_frac'].append(0.)
#                coincident_dict[coincident_ind]['path1_lat'].append(0.)
#                coincident_dict[coincident_ind]['path1_lon'].append(0.)
#                coincident_dict[coincident_ind]['path2_lat'].append(0.)
#                coincident_dict[coincident_ind]['path2_lon'].append(0.)
#                coincident_dict[coincident_ind]['lonE'].append(0.)
#                coincident_dict[coincident_ind]['latE'].append(0.)
#                coincident_dict[coincident_ind]['lonW'].append(0.)
#                coincident_dict[coincident_ind]['latW'].append(0.)
                continue
            
            # Retrieve longitude values
            lon1 = path1_lon[ii]
            lon2 = path2_lon[jj]
            
#            print('lat1, lon1', lat1, lon1)
#            print('lat2, lon2', lat2, lon2)
            
            # Check which path is east/west
            dum, az1, az2 = latlon2dist(lat1, lon1, lat2, lon2)
#            print('az1', az1)
#            print('az2', az2)
            
            # Passed latitude checks, store data            
            coincident_dict[coincident_ind]['UTC1'].append(path1[path1_ind]['UTC'][ii])
            coincident_dict[coincident_ind]['path1_lat'].append(lat1)
            coincident_dict[coincident_ind]['path1_lon'].append(lon1)
            coincident_dict[coincident_ind]['path2_lat'].append(lat2)
            coincident_dict[coincident_ind]['path2_lon'].append(lon2)
            
#            print('ii, jj', ii, jj)
#            print('path1')
#            print(path1_lat[ii], path1_lonE[ii], path1_lonW[ii])
#            print('path2')
#            print(path2_lat[jj], path2_lonE[jj], path2_lonW[jj])
#            
            # Compute current swath coverage
            swath1, dum1, dum2 = \
                latlon2dist(path1_latW[ii], path1_lonW[ii], path1_latE[ii], path1_lonE[ii])
            swath2, dum1, dum2 = \
                latlon2dist(path2_latW[jj], path2_lonW[jj], path2_latE[jj], path2_lonE[jj])
                
#            print(swath1, swath2)
            
            # Compute longitude diffs
            # Case 1: Path1 is to the east of Path2
            if az1 > 180.:
                
#                print('path1 east')
                
                # Compute intersection (linear interpolation)
                p1 = [path1_lonW[ii], path1_latW[ii]]
                p2 = [path1_lonE[ii], path1_latE[ii]]
                p3 = [path2_lonE[jj], path2_latE[jj]]
                
                try:
                    p4 = [path2_lonE[jj+1], path2_latE[jj+1]]
                except:
                    p4 = [path2_lonE[jj-1], path2_latE[jj-1]]
                
                lon_intersect, lat_intersect = compute_intersect(p1, p2, p3, p4)
                
#                print(p1, p2, p3, p4)
#                print(lon_intersect, lat_intersect)
                
                
                
                # Check for no overlap - Case 1A
                diff1 = compute_anglediff_deg(lon_intersect, path1_lonW[ii])
                if diff1 <= 0.:
                    coincident_dict[coincident_ind]['overlap_km'].append(0.)
                    coincident_dict[coincident_ind]['overlap_frac'].append(0.)
                    coincident_dict[coincident_ind]['lonE'].append(0.)
                    coincident_dict[coincident_ind]['latE'].append(0.)
                    coincident_dict[coincident_ind]['lonW'].append(0.)
                    coincident_dict[coincident_ind]['latW'].append(0.)
                    
                else:
                    
                    swath_intersect, dum1, dum2 = \
                        latlon2dist(path1_latW[ii], path1_lonW[ii], lat_intersect, lon_intersect)
                        
                    swath_overlap = min([swath_intersect, swath1, swath2])
                    frac = swath_overlap/min([swath1, swath2])
                    
                    
                    coincident_dict[coincident_ind]['overlap_km'].append(swath_overlap)
                    coincident_dict[coincident_ind]['overlap_frac'].append(frac)
                    
                    if swath_overlap > swath1:
                        lonE = path1_lonE[ii]
                        latE = path1_latE[ii]
                        lonW = path1_lonW[ii]
                        latW = path1_latW[ii]
                    elif swath_overlap > swath2:
                        lonE = path2_lonE[jj]
                        latE = path2_latE[jj]
                        lonW = path2_lonW[jj]
                        latW = path2_latW[jj]
                    else:
                        lonW = path1_lonW[ii]
                        latW = path1_latW[ii]
                        lonE = lon_intersect
                        latE = lat_intersect
                    
                    coincident_dict[coincident_ind]['lonE'].append(lonE)
                    coincident_dict[coincident_ind]['latE'].append(latE)
                    coincident_dict[coincident_ind]['lonW'].append(lonW)
                    coincident_dict[coincident_ind]['latW'].append(latW)
                                        
            # Case 2: Path1 is to the west of Path2
            elif az1 < 180.:
                
#                print('path1 west')
                                
                # Compute intersection (linear interpolation)
                p1 = [path1_lonW[ii], path1_latW[ii]]
                p2 = [path1_lonE[ii], path1_latE[ii]]
                p3 = [path2_lonW[jj], path2_latW[jj]]
                
                try:
                    p4 = [path2_lonW[jj+1], path2_latW[jj+1]]
                except:
                    p4 = [path2_lonW[jj-1], path2_latW[jj-1]]
                
                lon_intersect, lat_intersect = compute_intersect(p1, p2, p3, p4)
                
#                print(ii, jj)
#                print(p1, p2, p3, p4)
#                print(lon_intersect, lat_intersect)
                
                
                
                # Check for no overlap - Case 2A
                diff1 = compute_anglediff_deg(path1_lonE[ii], lon_intersect)
#                print(path1_lonE[ii], lon_intersect)
#                print(diff1)
                
                if diff1 <= 0.:
                    coincident_dict[coincident_ind]['overlap_km'].append(0.)
                    coincident_dict[coincident_ind]['overlap_frac'].append(0.)
                    coincident_dict[coincident_ind]['lonE'].append(0.)
                    coincident_dict[coincident_ind]['latE'].append(0.)
                    coincident_dict[coincident_ind]['lonW'].append(0.)
                    coincident_dict[coincident_ind]['latW'].append(0.)
                    
                else:
                                        
                    swath_intersect, dum1, dum2 = \
                        latlon2dist(path1_latE[ii], path1_lonE[ii], lat_intersect, lon_intersect)
                        
                    swath_overlap = min([swath_intersect, swath1, swath2])
                    frac = swath_overlap/min([swath1, swath2])
                    
#                    print(swath_overlap)
#                    print(frac)
                    
                    coincident_dict[coincident_ind]['overlap_km'].append(swath_overlap)
                    coincident_dict[coincident_ind]['overlap_frac'].append(frac)
                    
                    if swath_overlap > swath1:
                        lonE = path1_lonE[ii]
                        latE = path1_latE[ii]
                        lonW = path1_lonW[ii]
                        latW = path1_latW[ii]
                    elif swath_overlap > swath2:
                        lonE = path2_lonE[jj]
                        latE = path2_latE[jj]
                        lonW = path2_lonW[jj]
                        latW = path2_latW[jj]
                    else:
                        lonW = lon_intersect
                        latW = lat_intersect
                        lonE = path1_lonE[ii]
                        latE = path1_latE[ii]
                    
                    coincident_dict[coincident_ind]['lonE'].append(lonE)
                    coincident_dict[coincident_ind]['latE'].append(latE)
                    coincident_dict[coincident_ind]['lonW'].append(lonW)
                    coincident_dict[coincident_ind]['latW'].append(latW)        

            
            # Case 3: Path1 and Path2 cross Equator at same point
            else:
                print('Error - Lon1 == Lon2')
                print(eq_lon1)
                print(eq_lon2)
                mistake
                
       
        # Check if any overlaps occurred and writeover this entry if none
        overlap_frac = coincident_dict[coincident_ind]['overlap_frac']
        overlap_km = coincident_dict[coincident_ind]['overlap_km']
        
        
#        print(overlap_frac)
#        print(overlap_km)
#        print(coincident_dict[coincident_ind]['lonE'])
#        print(coincident_dict[coincident_ind]['latE'])
#        print(coincident_dict[coincident_ind]['lonW'])
#        print(coincident_dict[coincident_ind]['latW'])
#        
#        mistake
        
        
        if sum(overlap_frac) == 0.:
            coincident_ind -= 1
            print('reset coincident ind', coincident_ind)
            
        
        
                
    
    return coincident_dict


def compute_triple_coincidence(landsat_sentinel, path2, latlim2,
                               coincident_dict = {}, cotime_limit=1800.):
    
        
    # Initialize output
    if len(coincident_dict) == 0:
        coincident_dict = {}
        coincident_ind = -1
    else:
        coincident_dict = copy.copy(coincident_dict)
        coincident_ind = sorted(list(coincident_dict.keys()))[-1]
    
    # Generate list of times from path2 dictionary
    UTC2_list = [path2[ii]['UTC_zerolat'] for ii in list(path2.keys())]
    
    # Loop over entries in landsat/sentinel coincident dictionary
    for path1_ind in landsat_sentinel:
        
        #######################################################################
        #
        # Note that these check for being near the equator will truncate paths
        # that occur over the dividing line of months. Should be mimimal impact
        # as this is rare (max 12 per year) but right way to fix it would be
        # in compute_paths()
        #
        #######################################################################
        
        # Check if UTC_zerolat is actually near equator
        UTC1 = landsat_sentinel[path1_ind]['UTC1_zerolat']
        loc_ind = landsat_sentinel[path1_ind]['UTC1'].index(UTC1)
        lat_check = landsat_sentinel[path1_ind]['path1_lat'][loc_ind]
        if abs(lat_check) > 2.:
            continue
        
        
        # Find index in path2 for nearest path time to current path1 entry
        diff = [abs((UTC2 - UTC1).total_seconds()) for UTC2 in UTC2_list]
        path2_ind = diff.index(min(diff))
        UTC2 = path2[path2_ind]['UTC_zerolat']
        
        # Check if this entry is actually near equator
        loc_ind = path2[path2_ind]['UTC'].index(UTC2)
        lat_check = path2[path2_ind]['lat_list'][loc_ind]
        if abs(lat_check) > 2.:
            continue
        
#        print(UTC1)
#        print(UTC2)
        
#        if UTC1 < datetime(2020, 6, 1, 2, 0, 0):
#            continue
#        
#        if UTC1 > datetime(2020, 6, 6, 17, 0, 0):
#            mistake
        
        # Skip if time difference is greater than cotime_limit
        if diff[path2_ind] > cotime_limit:
            continue
        
        # Check longitude at equator
        path1_zerolat_ind = landsat_sentinel[path1_ind]['UTC1'].index(UTC1)
        path2_zerolat_ind = path2[path2_ind]['UTC'].index(UTC2)
        eq_lon1 = landsat_sentinel[path1_ind]['path1_lon'][path1_zerolat_ind]
        eq_lon2 = path2[path2_ind]['lon_list'][path2_zerolat_ind]
        
#        lon_diff_deg = compute_anglediff_deg(eq_lon1, eq_lon2)      
        
        
        
#        # If paths are farther apart than combined swath, skip
#        if abs(lon_diff_deg * pi/180.) > (alpha1 + alpha2):
#            continue
        
        print('\nCoincident Analysis')
#        print(path1_ind)
#        print(path2_ind)
#        print(latlim1, latlim2)
        
        
        # Remaining paths are close in time 
        # Initialize output
        coincident_ind += 1
        print('coincident_ind', coincident_ind)
        print(UTC1, UTC2)
        print(eq_lon1, eq_lon2)
        
        coincident_dict[coincident_ind] = {}
        coincident_dict[coincident_ind]['UTC1'] = []
        coincident_dict[coincident_ind]['path1_ind'] = path1_ind
        coincident_dict[coincident_ind]['path2_ind'] = path2_ind
        coincident_dict[coincident_ind]['UTC1_zerolat'] = UTC1
        coincident_dict[coincident_ind]['UTC2_zerolat'] = UTC2
        coincident_dict[coincident_ind]['eq_lon1'] = eq_lon1
        coincident_dict[coincident_ind]['eq_lon2'] = eq_lon2
        coincident_dict[coincident_ind]['path1_lat'] = []
        coincident_dict[coincident_ind]['path1_lon'] = []
        coincident_dict[coincident_ind]['path2_lat'] = []
        coincident_dict[coincident_ind]['path2_lon'] = []
        coincident_dict[coincident_ind]['lonE'] = []
        coincident_dict[coincident_ind]['latE'] = []
        coincident_dict[coincident_ind]['lonW'] = []
        coincident_dict[coincident_ind]['latW'] = []
        coincident_dict[coincident_ind]['overlap_km'] = []
        coincident_dict[coincident_ind]['overlap_frac'] = []


        # Retrieve latitude/longitude data
#        path1_lat = landsat_sentinel[path1_ind]['path1_lat']
#        path1_lon = landsat_sentinel[path1_ind]['path1_lon']
        path1_lonE = landsat_sentinel[path1_ind]['lonE']
        path1_latE = landsat_sentinel[path1_ind]['latE']
        path1_lonW = landsat_sentinel[path1_ind]['lonW']
        path1_latW = landsat_sentinel[path1_ind]['latW']
        l8s2_overlap_frac = landsat_sentinel[path1_ind]['overlap_frac']
        
        path2_lat = path2[path2_ind]['lat_list']
        path2_lon = path2[path2_ind]['lon_list']
        path2_lonE = path2[path2_ind]['lonE_list']
        path2_latE = path2[path2_ind]['latE_list']
        path2_lonW = path2[path2_ind]['lonW_list']
        path2_latW = path2[path2_ind]['latW_list']
        
        
        for ii in range(len(path1_latE)):
            
            # Skip if no overlap in Landsat8-Sentinel2 data
            if l8s2_overlap_frac[ii] < 1e-6:
                continue
            
            # Compute midpoint of coincident (path1) lat/lon overlap
            lat1 = (path1_latE[ii] - path1_latW[ii])/2. + path1_latW[ii]
            lon1 = compute_anglediff_deg(path1_lonE[ii], path1_lonW[ii])/2. + path1_lonW[ii]
            if lon1 > 180.:
                lon1 -= 360.
            if lon1 < -180.:
                lon1 += 360.
                
                
#            print(path1_latE[ii], path1_latW[ii], lat1)
#            print(path1_lonE[ii], path1_lonW[ii], lon1)
#            mistake
#                
            
            
            
#            # Check path1 latitude value against constraint          
#            if lat1 > latlim1[0] or lat1 < latlim1[1]:
##                coincident_dict[coincident_ind]['overlap_km'].append(0.)
##                coincident_dict[coincident_ind]['overlap_frac'].append(0.)
##                coincident_dict[coincident_ind]['path1_lat'].append(0.)
##                coincident_dict[coincident_ind]['path1_lon'].append(0.)
##                coincident_dict[coincident_ind]['path2_lat'].append(0.)
##                coincident_dict[coincident_ind]['path2_lon'].append(0.)
##                coincident_dict[coincident_ind]['lonE'].append(0.)
##                coincident_dict[coincident_ind]['latE'].append(0.)
##                coincident_dict[coincident_ind]['lonW'].append(0.)
##                coincident_dict[coincident_ind]['latW'].append(0.)
#                continue
            
            
            
            # Find index of closest latitude in path2
            lat_diff = [abs(lat1 - lat2_jj) for lat2_jj in path2_lat]
            
            # Skip if too far away
            if min(lat_diff) > 3.:
                continue
            
            jj = lat_diff.index(min(lat_diff))
            lat2 = path2_lat[jj]
            if lat2 > latlim2[0] or lat2 < latlim2[1]:
#                coincident_dict[coincident_ind]['overlap_km'].append(0.)
#                coincident_dict[coincident_ind]['overlap_frac'].append(0.)
#                coincident_dict[coincident_ind]['path1_lat'].append(0.)
#                coincident_dict[coincident_ind]['path1_lon'].append(0.)
#                coincident_dict[coincident_ind]['path2_lat'].append(0.)
#                coincident_dict[coincident_ind]['path2_lon'].append(0.)
#                coincident_dict[coincident_ind]['lonE'].append(0.)
#                coincident_dict[coincident_ind]['latE'].append(0.)
#                coincident_dict[coincident_ind]['lonW'].append(0.)
#                coincident_dict[coincident_ind]['latW'].append(0.)
                continue
            
            # Retrieve longitude values
            lon2 = path2_lon[jj]
            
#            print('lat1, lon1', lat1, lon1)
#            print('lat2, lon2', lat2, lon2)
            
            # Check which path is east/west
            dum, az1, az2 = latlon2dist(lat1, lon1, lat2, lon2)
#            print('az1', az1)
#            print('az2', az2)
            
            # Passed latitude checks, store data            
            coincident_dict[coincident_ind]['UTC1'].append(landsat_sentinel[path1_ind]['UTC1'][ii])
            coincident_dict[coincident_ind]['path1_lat'].append(lat1)
            coincident_dict[coincident_ind]['path1_lon'].append(lon1)
            coincident_dict[coincident_ind]['path2_lat'].append(lat2)
            coincident_dict[coincident_ind]['path2_lon'].append(lon2)
            
#            print('ii, jj', ii, jj)
#            print('path1')
#            print(path1_lat[ii], path1_lonE[ii], path1_lonW[ii])
#            print('path2')
#            print(path2_lat[jj], path2_lonE[jj], path2_lonW[jj])
#            
            # Compute current swath coverage
            swath1, dum1, dum2 = \
                latlon2dist(path1_latW[ii], path1_lonW[ii], path1_latE[ii], path1_lonE[ii])
            swath2, dum1, dum2 = \
                latlon2dist(path2_latW[jj], path2_lonW[jj], path2_latE[jj], path2_lonE[jj])
                
#            print(swath1, swath2)
            
            # Compute longitude diffs
            # Case 1: Path1 is to the east of Path2
            if az1 > 180.:
                
#                print('path1 east')
                
                # Compute intersection (linear interpolation)
                p1 = [path1_lonW[ii], path1_latW[ii]]
                p2 = [path1_lonE[ii], path1_latE[ii]]
                p3 = [path2_lonE[jj], path2_latE[jj]]
                
                try:
                    p4 = [path2_lonE[jj+1], path2_latE[jj+1]]
                except:
                    p4 = [path2_lonE[jj-1], path2_latE[jj-1]]
                
                lon_intersect, lat_intersect = compute_intersect(p1, p2, p3, p4)
                
#                print(p1, p2, p3, p4)
#                print(lon_intersect, lat_intersect)
                
                
                
                # Check for no overlap - Case 1A
                diff1 = compute_anglediff_deg(lon_intersect, path1_lonW[ii])
                if diff1 <= 0.:
                    coincident_dict[coincident_ind]['overlap_km'].append(0.)
                    coincident_dict[coincident_ind]['overlap_frac'].append(0.)
                    coincident_dict[coincident_ind]['lonE'].append(0.)
                    coincident_dict[coincident_ind]['latE'].append(0.)
                    coincident_dict[coincident_ind]['lonW'].append(0.)
                    coincident_dict[coincident_ind]['latW'].append(0.)
                    
                else:
                    
                    swath_intersect, dum1, dum2 = \
                        latlon2dist(path1_latW[ii], path1_lonW[ii], lat_intersect, lon_intersect)
                        
                    swath_overlap = min([swath_intersect, swath1, swath2])
                    frac = swath_overlap/min([swath1, swath2])
                    
                    
                    coincident_dict[coincident_ind]['overlap_km'].append(swath_overlap)
                    coincident_dict[coincident_ind]['overlap_frac'].append(frac)
                    
                    if swath_overlap > swath1:
                        lonE = path1_lonE[ii]
                        latE = path1_latE[ii]
                        lonW = path1_lonW[ii]
                        latW = path1_latW[ii]
                    elif swath_overlap > swath2:
                        lonE = path2_lonE[jj]
                        latE = path2_latE[jj]
                        lonW = path2_lonW[jj]
                        latW = path2_latW[jj]
                    else:
                        lonW = path1_lonW[ii]
                        latW = path1_latW[ii]
                        lonE = lon_intersect
                        latE = lat_intersect
                    
                    coincident_dict[coincident_ind]['lonE'].append(lonE)
                    coincident_dict[coincident_ind]['latE'].append(latE)
                    coincident_dict[coincident_ind]['lonW'].append(lonW)
                    coincident_dict[coincident_ind]['latW'].append(latW)
                                        
            # Case 2: Path1 is to the west of Path2
            elif az1 < 180.:
                
#                print('path1 west')
                                
                # Compute intersection (linear interpolation)
                p1 = [path1_lonW[ii], path1_latW[ii]]
                p2 = [path1_lonE[ii], path1_latE[ii]]
                p3 = [path2_lonW[jj], path2_latW[jj]]
                
                try:
                    p4 = [path2_lonW[jj+1], path2_latW[jj+1]]
                except:
                    p4 = [path2_lonW[jj-1], path2_latW[jj-1]]
                
                lon_intersect, lat_intersect = compute_intersect(p1, p2, p3, p4)
                
#                print(ii, jj)
#                print(p1, p2, p3, p4)
#                print(lon_intersect, lat_intersect)
                
                
                
                # Check for no overlap - Case 2A
                diff1 = compute_anglediff_deg(path1_lonE[ii], lon_intersect)
#                print(path1_lonE[ii], lon_intersect)
#                print(diff1)
                
                if diff1 <= 0.:
                    coincident_dict[coincident_ind]['overlap_km'].append(0.)
                    coincident_dict[coincident_ind]['overlap_frac'].append(0.)
                    coincident_dict[coincident_ind]['lonE'].append(0.)
                    coincident_dict[coincident_ind]['latE'].append(0.)
                    coincident_dict[coincident_ind]['lonW'].append(0.)
                    coincident_dict[coincident_ind]['latW'].append(0.)
                    
                else:
                                        
                    swath_intersect, dum1, dum2 = \
                        latlon2dist(path1_latE[ii], path1_lonE[ii], lat_intersect, lon_intersect)
                        
                    swath_overlap = min([swath_intersect, swath1, swath2])
                    frac = swath_overlap/min([swath1, swath2])
                    
#                    print(swath_overlap)
#                    print(frac)
                    
                    coincident_dict[coincident_ind]['overlap_km'].append(swath_overlap)
                    coincident_dict[coincident_ind]['overlap_frac'].append(frac)
                    
                    if swath_overlap > swath1:
                        lonE = path1_lonE[ii]
                        latE = path1_latE[ii]
                        lonW = path1_lonW[ii]
                        latW = path1_latW[ii]
                    elif swath_overlap > swath2:
                        lonE = path2_lonE[jj]
                        latE = path2_latE[jj]
                        lonW = path2_lonW[jj]
                        latW = path2_latW[jj]
                    else:
                        lonW = lon_intersect
                        latW = lat_intersect
                        lonE = path1_lonE[ii]
                        latE = path1_latE[ii]
                    
                    coincident_dict[coincident_ind]['lonE'].append(lonE)
                    coincident_dict[coincident_ind]['latE'].append(latE)
                    coincident_dict[coincident_ind]['lonW'].append(lonW)
                    coincident_dict[coincident_ind]['latW'].append(latW)        

            
            # Case 3: Path1 and Path2 cross Equator at same point
            else:
                print('Error - Lon1 == Lon2')
                print(eq_lon1)
                print(eq_lon2)
                mistake
                
       
        # Check if any overlaps occurred and writeover this entry if none
        overlap_frac = coincident_dict[coincident_ind]['overlap_frac']
        overlap_km = coincident_dict[coincident_ind]['overlap_km']
        
        
#        print(overlap_frac)
#        print(overlap_km)
#        print(coincident_dict[coincident_ind]['lonE'])
#        print(coincident_dict[coincident_ind]['latE'])
#        print(coincident_dict[coincident_ind]['lonW'])
#        print(coincident_dict[coincident_ind]['latW'])
#        
#        mistake
        
        
        if sum(overlap_frac) == 0.:
            coincident_ind -= 1
            print('reset coincident ind', coincident_ind)
            
        
        
                
    
    return coincident_dict



def landsat_sentinel_coincidence():
    
    
    # Landsat-8 Data
    landsat8_norad = 39084
    
    # Sentinel 2 Data
    sentinel_2a_norad = 40697
    sentinel_2b_norad = 42063
    
 
    # Setup and execute first case for 2020 06 - Sentinel 2A
    
    # Load path data
    path_file = os.path.join('..//data/path_data_primary_targets_2020_06.pkl')
    pklFile = open(path_file, 'rb')
    data = pickle.load(pklFile)
    path_dict = data[0]
    pklFile.close()
    
    path1 = path_dict[landsat8_norad]
    path2 = path_dict[sentinel_2a_norad]
    
    alpha1 = (185./Re)/2.
    alpha2 = (290./Re)/2.
    
    latlim1 =  [(80. + 47./60.), (-81 - 51./60.)]
    latlim2 = [83., -56.]
    
    cotime_limit = 30.*60.
    
    landsat8_sentinel_coincident = \
        compute_coincidence(path1, path2, alpha1, alpha2, latlim1, latlim2, 
                            {}, cotime_limit)
        
        
    
    
    # Loop over remaining data files
    ending_list = ['2020_07', '2020_08', '2020_09', '2020_10', '2020_11', '2020_12',
                   '2021_01', '2021_02', '2021_03', '2021_04', '2021_05']   
  
    
    for ending in ending_list:
        path_file = os.path.join('..//data/path_data_primary_targets_' + ending + '.pkl')    
        pklFile = open(path_file, 'rb')
        data = pickle.load(pklFile)
        path_dict = data[0]
        pklFile.close()
        
        path1 = path_dict[landsat8_norad]
        path2 = path_dict[sentinel_2a_norad]
        
        landsat8_sentinel_coincident = \
        compute_coincidence(path1, path2, alpha1, alpha2, latlim1, latlim2, 
                            landsat8_sentinel_coincident,
                            cotime_limit)
        
    
    # Run Sentinel 2B data
    ending_list = ['2020_06', '2020_07', '2020_08', '2020_09', '2020_10', '2020_11', '2020_12',
                   '2021_01', '2021_02', '2021_03', '2021_04', '2021_05']   
    
    
    for ending in ending_list:
        path_file = os.path.join('..//data/path_data_primary_targets_' + ending + '.pkl')    
        pklFile = open(path_file, 'rb')
        data = pickle.load(pklFile)
        path_dict = data[0]
        pklFile.close()
        
        path1 = path_dict[landsat8_norad]
        path2 = path_dict[sentinel_2b_norad]
        
        landsat8_sentinel_coincident = \
        compute_coincidence(path1, path2, alpha1, alpha2, latlim1, latlim2, 
                            landsat8_sentinel_coincident,
                            cotime_limit)
    
    
    # Save output    
    coincident_file = os.path.join('..//data/coincident_data_landsat8_sentinel.pkl')
    pklFile = open( coincident_file, 'wb' )
    pickle.dump( [landsat8_sentinel_coincident], pklFile, -1 )
    pklFile.close()
    
    return


def triple_coincidence():
    
  
 
    # Setup and execute first case for 2020 06
    
    
    # Load coincident data Landsat-Sentinel
    coincident_file = os.path.join('..//data/coincident_data_landsat8_sentinel.pkl')
    pklFile = open(coincident_file, 'rb')
    data = pickle.load(pklFile)
    landsat_sentinel = data[0]
    pklFile.close()
    
    
    
    # Setup for execution
    obj_id = 10032
    obj_str = str(obj_id).zfill(5)
    latlim2 =  [(80. + 47./60.), (-81 - 51./60.)]    
    cotime_limit = 30.*60.    
    coincident_dict = {}
    
    
    # Loop over data files
    ending_list = ['2020_06', '2020_07', '2020_08', '2020_09', '2020_10', '2020_11', '2020_12',
                   '2021_01', '2021_02', '2021_03', '2021_04', '2021_05']   
  
    
    for ending in ending_list:
        path_file = os.path.join('..//data/path_data_small_swath_' + ending + '.pkl')    
        pklFile = open(path_file, 'rb')
        data = pickle.load(pklFile)
        path_dict = data[0]
        pklFile.close()
        
        print(path_dict.keys())
        
        path2 = path_dict[obj_id]  
        
        coincident_dict = \
        compute_triple_coincidence(landsat_sentinel, path2, latlim2, 
                                   coincident_dict, cotime_limit)
        
    
    
    
    # Save output    
    coincident_file = os.path.join('..//data/coincident_data_triple_' + obj_str + '.pkl')
    pklFile = open( coincident_file, 'wb' )
    pickle.dump( [coincident_dict], pklFile, -1 )
    pklFile.close()
    
    return


def plot_coincident_data():
    
    
    # Load data
    coincident_file = os.path.join('..//data/coincident_data_triple_10030.pkl')
    pklFile = open(coincident_file, 'rb')
    data = pickle.load(pklFile)
    coincident_dict = data[0]
    pklFile.close()
    
    # Define WRS2 and Basemap
    wrs2_dict, wrs2_lon, wrs2_lat, wrs2_neglon, wrs2_poslon, land_flag = define_wrs2_grid()
    
    npath_doy = [0.]*366
    nbin_doy = [0.]*366
    nbin_land_doy = [0.]*366
    nbin_map = np.zeros(land_flag.shape)
    path_doy_list = []
    path_hrs_list = []
    path_number_list = []
    path_eq_frac_list = []
    path_eq_km_list = []
    
    bin_frac_grid = np.zeros((5000, 366))
    bin_km_grid = np.zeros((5000, 366))
    
    landbin_frac_grid = np.zeros((5000, 366))
    landbin_km_grid = np.zeros((5000, 366))
    
    wrs2_time = 0.
    overall_time = 0.
    
    bin_doy_list = []    
    bin_number_list = []
    bin_frac_list = []
    bin_km_list = []
    
    start = time.time()
    
    bin_dict = {}    
    for ind in coincident_dict:
        print('\nind', ind)
        print('overall time', time.time()-start)
        print('wrs time', wrs2_time)
        
        entry = coincident_dict[ind]
        print('path1', entry['UTC1_zerolat'], entry['eq_lon1'])
        print('path2', entry['UTC2_zerolat'], entry['eq_lon2'])
        
        # Compute Day of Year
        UTC1_zerolat = entry['UTC1_zerolat']
        doy = UTC1_zerolat.timetuple().tm_yday
        doy_ind = doy - 1
        UTC1_baseday = datetime(UTC1_zerolat.year, UTC1_zerolat.month, UTC1_zerolat.day)
        UTC1_hrs = (UTC1_zerolat - UTC1_baseday).total_seconds()/3600.
        
        # Retrieve overlap data
        UTC1_list = entry['UTC1']
        overlap_km = entry['overlap_km']
        overlap_frac = entry['overlap_frac']
        latE = entry['latE']
        lonE = entry['lonE']
        latW = entry['latW']
        lonW = entry['lonW']
        path1_lat = entry['path1_lat']
        path1_lon = entry['path1_lon']
        
        # Equator 
#        eq_ind = path1_lat.index(min([abs(lat_ii) for lat_ii in path1_lat]))
        
#        if ind == 1335 or ind == 2384:
##            continue
#            
#            print(UTC1_list)
#            print(path1_lat)
#            print(UTC1_zerolat)
#            
#            
        
        
#        print(overlap_km)
#        print(overlap_frac)
#        
#        print(path1_lat[eq_ind])
#        print(UTC1_list[eq_ind], UTC1_zerolat)
#        print(overlap_km[eq_ind])
#        print(overlap_frac[eq_ind])
        

        if sum(overlap_frac) > 0:
            npath_doy[doy_ind] += 1
            path_doy_list.append(doy)
            path_hrs_list.append(UTC1_hrs)
            path_number_list.append(npath_doy[doy_ind])
            
            
            try:
                eq_ind = UTC1_list.index(UTC1_zerolat)
                overlap_frac_ii = overlap_frac[eq_ind]
                overlap_km_ii = overlap_km[eq_ind]
            except:
                overlap_frac_ii = 0.
                overlap_km_ii = 0.
            
            path_eq_frac_list.append(overlap_frac_ii)
            path_eq_km_list.append(overlap_km_ii)
        
        # Process bin overlaps
        entry_pathrow_list = []
        entry_wrs2lonlat_list = []
        entry_UTC1_list = []
        entry_doy_list = []
        entry_overlap_frac_list = []
        entry_overlap_km_list = []
        entry_ctrlonlat_list = []
        entry_landflag_list = []
        for jj in range(len(UTC1_list)):
            
            # Check that this bin has nonzero overlap
            overlap_frac_jj = overlap_frac[jj]
            if overlap_frac_jj > 0.:
                
                UTC1_jj = UTC1_list[jj]
                doy_jj =  UTC1_jj.timetuple().tm_yday
                latE_jj = latE[jj]
                lonE_jj = lonE[jj]
                latW_jj = latW[jj]
                lonW_jj = lonW[jj]
                overlap_km_jj = overlap_km[jj]
                
                # Compute center of overlap
                lat_diff = latE_jj - latW_jj
                lat_ctr = latW_jj + 0.5*lat_diff                
                
                lon_diff = compute_anglediff_deg(lonE_jj, lonW_jj)
                lon_ctr = lonW_jj + 0.5*lon_diff
                if lon_ctr < -180.:
                    lon_ctr += 360.
                elif lon_ctr > 180.:
                    lon_ctr -= 360.
                    
                # Compute WRS2 bin
#                print('\n')
#                print(lonE_jj, latE_jj)
#                print(lonW_jj, latW_jj)
#                print(lon_ctr, lat_ctr)
                wrs_start = time.time()
                
                path, row, wrs2_lon_jj, wrs2_lat_jj = \
                    compute_wrs2_path_row(lon_ctr, lat_ctr, wrs2_lon, wrs2_lat, wrs2_neglon, wrs2_poslon)
                
                wrs2_time += time.time() - wrs_start
                
                # Retrieve land_flag for this bin
                is_land = land_flag[row-1, path-1]
                
                # Store data
                # Check if this bin has been counted and overwrite if needed
                if [path, row] in entry_pathrow_list:
                    pathrow_ind = entry_pathrow_list.index([path, row])
                    
                    # Check if this entry has greater overlap
                    if overlap_frac_jj < entry_overlap_frac_list[pathrow_ind]:
                        continue
                    
                    # Overwrite
                    else:
                        entry_pathrow_list[pathrow_ind] = [path, row]
                        entry_wrs2lonlat_list[pathrow_ind] = [wrs2_lon_jj, wrs2_lat_jj]
                        entry_UTC1_list[pathrow_ind] = UTC1_jj
                        entry_doy_list[pathrow_ind] = doy_jj
                        entry_overlap_frac_list[pathrow_ind] = overlap_frac_jj
                        entry_overlap_km_list[pathrow_ind] = overlap_km_jj
                        entry_ctrlonlat_list[pathrow_ind] = [lon_ctr, lat_ctr]
                        
                        bin_ind = int(nbin_doy[doy_jj-1]) - 1
                        doy_ind = doy_jj-1
                        
                        bin_frac_grid[bin_ind, doy_ind] = overlap_frac_jj
                        bin_km_grid[bin_ind, doy_ind] = overlap_km_jj
                        
                        landbin_frac_grid[bin_ind, doy_ind] = overlap_frac_jj * is_land
                        landbin_km_grid[bin_ind, doy_ind] = overlap_km_jj * is_land
                    
                else:
                    entry_pathrow_list.append([path, row])
                    entry_wrs2lonlat_list.append([wrs2_lon_jj, wrs2_lat_jj])
                    entry_UTC1_list.append(UTC1_jj)
                    entry_doy_list.append(doy_jj)
                    entry_overlap_frac_list.append(overlap_frac_jj)
                    entry_overlap_km_list.append(overlap_km_jj)
                    entry_ctrlonlat_list.append([lon_ctr, lat_ctr])
                    entry_landflag_list.append(is_land)
                    
                    nbin_doy[doy_jj-1] += 1.
                    
                    nbin_land_doy[doy_jj-1] += is_land
                    
                    bin_ind = int(nbin_doy[doy_jj-1]) - 1
                    doy_ind = doy_jj-1
                    
                    bin_frac_grid[bin_ind, doy_ind] = overlap_frac_jj
                    bin_km_grid[bin_ind, doy_ind] = overlap_km_jj
                    
                    landbin_frac_grid[bin_ind, doy_ind] = overlap_frac_jj * is_land
                    landbin_km_grid[bin_ind, doy_ind] = overlap_km_jj * is_land
                    
                    nbin_map[row-1, path-1] += 1
                    
                    
                
                
        #        # Store data for output/plots
#        bin_dict[ind] = {}
#        bin_dict[ind]['pathrow_list'] = entry_pathlow_list
#        bin_dict[ind]['wrs2lonlat_list'] = entry_wrs2lonlat_list
#        bin_dict[ind]['UTC1_list'] = entry_UTC1_list
#        bin_dict[ind]['doy_liist'] = entry_doy_list
#        bin_dict[ind]['overlap_frac_list'] = entry_overlap_frac_list
#        bin_dict[ind]['                    
                
        
        
            
            
        
#    print(path_doy_list)
#    print(path_number_list)
#    print(path_eq_frac_list)
#    print(path_eq_km_list)
            
    # Build 3D data set
    xmax = max(path_doy_list) + 1
    ymax = max(path_number_list) + 1

    zfrac = np.zeros((int(ymax), int(xmax)))
    zkm = np.zeros((int(ymax), int(xmax)))
    for ii in range(len(path_doy_list)):
        doy = int(path_doy_list[ii])
        path_number = int(path_number_list[ii])
        zfrac[path_number, doy] = path_eq_frac_list[ii]*100.
        zkm[path_number, doy] = path_eq_km_list[ii]
        
    doy_list = list(np.arange(1, 366.5, 1))
#    print(doy_list)   
    
    
    
    # Bin plots
    max_nbin_doy = int(max(nbin_doy))
    max_nbin_land_doy = int(max(nbin_land_doy))
    
    

    # Generate plots
#    fig = plt.figure()
#    ax = fig.add_axes([0,0,1,1])
#    ax.bar(path_doy_list, path_number_list)
#    ax.set_xlabel('Day of Year')
#    ax.set_ylabel('Number of Paths with Overlap')
    
#    fig = plt.figure()
#    plt.bar(path_doy_list, path_number_list)
#    plt.xlabel('Day of Year')
#    plt.ylabel('Number of Paths with Overlap')
    
    fig = plt.figure()
    plt.bar(doy_list, npath_doy)
    plt.xlabel('Day of Year')
    plt.ylabel('Number of Paths with Overlap')
    
    fig = plt.figure()
    plt.bar(doy_list, nbin_doy)
    plt.xlabel('Day of Year')
    plt.ylabel('Number of WRS2 Bins with Overlap')
    
    fig = plt.figure()
    plt.bar(doy_list, nbin_land_doy)
    plt.xlabel('Day of Year')
    plt.ylabel('Number of WRS2 Bins with Overlap Over Land')
    
    fig = plt.figure()
    plt.contourf(zfrac, levels=list(np.arange(0, 101, 10)), cmap=plt.cm.plasma)
    plt.xlabel('Day of Year')
    plt.ylabel('Path Number of Day')
    plt.title('Percent of Swath Overlap at Equator')
    plt.colorbar()
    
    fig = plt.figure()
    plt.contourf(zkm, levels=list(np.arange(0, 101, 10)), cmap=plt.cm.plasma)
    plt.xlabel('Day of Year')
    plt.ylabel('Path Number of Day')
    plt.title('Swath Overlap at Equator [km]')
    plt.colorbar()
    
    fig = plt.figure()
    plt.contourf(bin_frac_grid[0:max_nbin_doy,:]*100, levels=list(np.arange(0, 101, 10)), cmap=plt.cm.plasma)
    plt.xlabel('Day of Year')
    plt.ylabel('Bin Number of Day')
    plt.title('Percent of Swath Overlap at WRS2 Bins')
    plt.colorbar()
    
    fig = plt.figure()
    plt.contourf(bin_km_grid[0:max_nbin_doy,:], levels=list(np.arange(0, 101, 10)), cmap=plt.cm.plasma)
    plt.xlabel('Day of Year')
    plt.ylabel('Bin Number of Day')
    plt.title('Swath Overlap at WRS2 Bins [km]')
    plt.colorbar()
    
    fig = plt.figure()
    plt.contourf(landbin_frac_grid[0:max_nbin_land_doy,:]*100, levels=list(np.arange(0, 101, 10)), cmap=plt.cm.plasma)
    plt.xlabel('Day of Year')
    plt.ylabel('Bin Number of Day')
    plt.title('Percent of Swath Overlap at WRS2 Bins Over Land')
    plt.colorbar()
    
    fig = plt.figure()
    plt.contourf(landbin_km_grid[0:max_nbin_land_doy,:], levels=list(np.arange(0, 101, 20)), cmap=plt.cm.plasma)
    plt.xlabel('Day of Year')
    plt.ylabel('Bin Number of Day')
    plt.title('Swath Overlap at WRS2 Bins Over Land [km]')
    plt.colorbar()
    
    
    
    # Plot on map
    nbin_plot_data = {}
    nbin_land_plot_data = {}
#    plot_lon = []
#    plot_lat = []
#    plot_nbin = []
#    plot_nbin_land = []
    nbin_land_map = np.multiply(nbin_map, land_flag)
    
    max_nbin = np.max(nbin_map)
    max_nbin_land = np.max(nbin_land_map)

    
    cmap = plt.get_cmap('gnuplot')
    nbin_colors = [cmap(i) for i in np.linspace(0, 1, int(max_nbin)+1)]
    nbin_land_colors = [cmap(i) for i in np.linspace(0, 1, int(max_nbin_land)+1)]
    
    
    
    test_plot_lon = []
    test_plot_lat = []
    for ii in range(wrs2_lon.shape[0]):
        for jj in range(wrs2_lon.shape[1]):
                        
            lon = wrs2_lon[ii,jj]
            lat = wrs2_lat[ii,jj]
            nbin = nbin_map[ii,jj]
            nbin_land = nbin_land_map[ii,jj]            
            
            if nbin not in nbin_plot_data:
                nbin_plot_data[nbin] = {}
                nbin_plot_data[nbin]['plot_lon'] = []
                nbin_plot_data[nbin]['plot_lat'] = []
                
            if nbin_land not in nbin_land_plot_data:
                nbin_land_plot_data[nbin_land] = {}
                nbin_land_plot_data[nbin_land]['plot_lon'] = []
                nbin_land_plot_data[nbin_land]['plot_lat'] = []
                
            if nbin_land > 0:
                test_plot_lon.append(lon)
                test_plot_lat.append(lat)
            
            
            nbin_plot_data[nbin]['plot_lon'].append(lon)
            nbin_plot_data[nbin]['plot_lat'].append(lat)
            nbin_land_plot_data[nbin_land]['plot_lon'].append(lon)
            nbin_land_plot_data[nbin_land]['plot_lat'].append(lat)
    
    
    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,
                urcrnrlon=180,resolution='c')
    
    fig = plt.figure()
    for nbin in nbin_plot_data:
        if nbin == 0:
            continue
        
        plot_lon = nbin_plot_data[nbin]['plot_lon']
        plot_lat = nbin_plot_data[nbin]['plot_lat']
        plt.plot(plot_lon, plot_lat, 'o', color=nbin_colors[int(nbin)], ms=1)
      
        
    m.drawcoastlines()
    m.drawmeridians(np.arange(-180, 180, 45))
    m.drawparallels(np.arange(-90, 90, 45))
    m.drawmapboundary()
    m.drawcountries()
    plt.yticks(np.arange(-90, 91, 45))
    plt.xticks(np.arange(-180, 181, 45))
    
    
    
    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,
                urcrnrlon=180,resolution='c')
    
    fig = plt.figure()
    for nbin_land in nbin_land_plot_data:
        if nbin_land == 0:
            continue
        
        plot_lon = nbin_land_plot_data[nbin_land]['plot_lon']
        plot_lat = nbin_land_plot_data[nbin_land]['plot_lat']
        
        plt.plot(plot_lon, plot_lat, 'o', color=nbin_land_colors[int(nbin_land)], ms=1) 
        
#        if nbin_land == 1.:
#            
##            plt.plot(plot_lon, plot_lat, color=nbin_land_colors[int(nbin_land)], marker='.') 
#            color = nbin_land_colors[1]
#            plt.plot(plot_lon, plot_lat,'o', color=color, ms=1)
#            
#        if nbin_land == 2.:
#            color = nbin_land_colors[2]
#            plt.plot(plot_lon, plot_lat,'o', color=color, ms=1)
        
        
    m.drawcoastlines()
    m.drawmeridians(np.arange(-180, 180, 45))
    m.drawparallels(np.arange(-90, 90, 45))
    m.drawmapboundary()
    m.drawcountries()
    plt.yticks(np.arange(-90, 91, 45))
    plt.xticks(np.arange(-180, 181, 45))
    

#    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,
#                urcrnrlon=180,resolution='c')
#    
#    fig = plt.figure()
#    plt.plot(test_plot_lon, test_plot_lat, 'go', ms=1)
#    
#    m.drawcoastlines()
#    m.drawmeridians(np.arange(-180, 180, 45))
#    m.drawparallels(np.arange(-90, 90, 45))
#    m.drawmapboundary()
#    m.drawcountries()
#    plt.yticks(np.arange(-90, 91, 45))
#    plt.xticks(np.arange(-180, 181, 45))
            
#    fig = plt.figure()
#    plt.contourf(wrs2_lon, wrs2_lat, nbin_map, cmap=plt.cm.plasma)
#    
#    m.drawcoastlines()
#    m.drawmeridians(np.arange(-180, 180, 45))
#    m.drawparallels(np.arange(-90, 90, 45))
#    m.drawmapboundary()
#    m.drawcountries()
#    plt.yticks(np.arange(-90, 91, 45))
#    plt.xticks(np.arange(-180, 181, 45))
#
#    plt.colorbar()
#    
#    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,
#                urcrnrlon=180,resolution='c')
#    
#    fig2 = plt.figure()
#    plt.contourf(wrs2_lon, wrs2_lat, nbin_land_map, cmap=plt.cm.plasma)
#    
#    m.drawcoastlines()
#    m.drawmeridians(np.arange(-180, 180, 45))
#    m.drawparallels(np.arange(-90, 90, 45))
#    m.drawmapboundary()
#    m.drawcountries()
#    plt.yticks(np.arange(-90, 91, 45))
#    plt.xticks(np.arange(-180, 181, 45))
#
#    plt.colorbar()
#            

    
    
#    fig = plt.figure()
#    plt.contourf(path_hrs_list, path_doy_list, zfrac, levels=list(np.arange(0, 101, 10)), cmap=plt.cm.plasma)
#    plt.xlabel('Day of Year')
#    plt.ylabel('UTC hour')
#    plt.title('Percent of Swath Overlap at Equator')
#    plt.colorbar()
    
    
#    cbar.ax.set_ylabel('Percent Overlap')
    
    
#    plt.figure()
#    plt.contourf(xlist, ylist, zgrid)
#    
#    plt.figure()
#    plt.axes.Axes.pcolor(path_doy_list, path_number_list, path_eq_frac_list)
    
#    plt.figure()
#    plt.pcolor(x, y, zgrid)
    
    
    
    
    plt.show()
        
    
    # Compute additional metrics
    total_wrs2_coincident_bins = sum(nbin_doy)
    total_wrs2_land_coincident_bins = sum(nbin_land_doy)
    
    wrs2_grid_count = nbin_map.size
    wrs2_land_grid_count = np.count_nonzero(land_flag)
    
    unique_wrs2_coincident_bins = np.count_nonzero(nbin_map)
    unique_wrs2_land_coincident_bins = np.count_nonzero(nbin_land_map)
    
    print('\n\nFinal Statistics')
    print('Total Number of WRS2 Bin Overlaps: ', total_wrs2_coincident_bins)
    print('Total Number of WRS2 Bin Overlaps over Land: ', total_wrs2_land_coincident_bins)
    print('Mean Overlap Fraction: ', np.sum(bin_frac_grid)/np.count_nonzero(bin_frac_grid))
#    print('Median Overlap Fraction: ', np.median(bin_frac_grid))
    print('Mean Overlap km: ', np.sum(bin_km_grid)/np.count_nonzero(bin_km_grid))
#    print('Median Overlap km: ', np.median(bin_km_grid))
    print('Mean Overlap Fraction over Land: ', np.sum(landbin_frac_grid)/np.count_nonzero(landbin_frac_grid))
#    print('Median Overlap Fraction over Land: ', np.median(landbin_frac_grid))
    print('Mean Overlap km over Land: ', np.sum(landbin_km_grid)/np.count_nonzero(landbin_km_grid))
#    print('Median Overlap km over Land: ', np.median(landbin_km_grid))
    
    
#    print('\nNonzero bin_frac_grid', np.count_nonzero(bin_frac_grid))
#    print('Nonzero bin_km_grid', np.count_nonzero(bin_km_grid))
    
   
    print('\nMaximum Number of Overlaps for a single bin: ', max_nbin)
    print('Maximum Number of Overlaps for a single bin over land: ', max_nbin_land)
    
    print('\nTotal Number of WRS2 Map Bins: ', wrs2_grid_count)
    print('Unique WRS2 Bins with Overlap: ', unique_wrs2_coincident_bins)
    print('Total Number of WRS2 Map Bins over Land: ', wrs2_land_grid_count)
    print('Unique WRS2 Bins with Overlap over Land: ', unique_wrs2_land_coincident_bins)
    
    
    
    return


#def compute_coincidence(path_file):
#    
#    
#    # Conincidence time
#    cotime_limit = 30. * 60.
#    
#    # Landsat-8 Data
#    landsat8_norad = 39084
#    
#    # Sentinel 2 Data
#    sentinel_2a_norad = 40697
#    sentinel_2b_norad = 42063
#    
#    # Operating Latitude
#    obj1_latlim = [(80. + 47./60.), (-81 - 51./60.)]
#    obj2_latlim = [83., -56.]
#    
#    
#    
#    # Load path data
#    pklFile = open(path_file, 'rb')
#    data = pickle.load(pklFile)
#    path_dict = data[0]
#    pklFile.close()
#    
#    path1 = path_dict[landsat8_norad]
#    path2 = path_dict[sentinel_2a_norad]
#    path3 = path_dict[sentinel_2b_norad]
#    
#    for path_ind1 in path1:
#        UTC_zerolat1 = path1[path_ind1]['UTC_zerolat']        
#        
#        for path_ind2 in path2:
#            UTC_zerolat2 = path2[path_ind2]['UTC_zerolat']
#            
#            # Check paths are close in time
#            if abs((UTC_zerolat2 - UTC_zerolat1).total_seconds()) > cotime_limit:
#                continue
#            
#            N1= len(path1[path_ind1]['UTC'])
#            N2 = len(path2[path_ind2]['UTC'])
#            
#            # Check overlap at zerolat
#            #TODO
#                        
#            for ii in range(N1):
#                
#                obj1_lat = path1[path_ind1]['lat_list'][ii]
#                obj1_lat1 = path1[path_ind1]['lat1_list'][ii]
#                obj1_lon1 = path1[path_ind1]['lon1_list'][ii]
#                obj1_lat2 = path1[path_ind1]['lat2_list'][ii]
#                obj1_lon2 = path1[path_ind1]['lon2_list'][ii]
#                
#                # Check operational latitudes
#                if obj1_lat > obj1_latlim[0] or obj1_lat < obj1_latlim[1]:
#                    continue                
#                
#                # Find index of latitude closest to object 1
#                diff = [abs(lat_jj - obj1_lat) for lat_jj in path2[path_ind2]['lat_list']]
#                jj = diff.index(min(diff))
#                
#                obj2_lat1 = path2[path_ind2]['lat1_list'][jj]
#                obj2_lon1 = path2[path_ind2]['lon1_list'][jj]
#                obj2_lat2 = path2[path_ind2]['lat2_list'][jj]
#                obj2_lon2 = path2[path_ind2]['lon2_list'][jj]
#                
#                # Compute overlap
#                lon_diff1 = obj1_lon2 - obj1_lon1
#                lon_diff2 = obj2_lon2 - obj2_lon1
#                
#                
#                # Check for overlap at all
#                #TODO
#                    
#    
#    
#    
#    return


def plot_groundtrack():
    
    # Load data
    fname = os.path.join('..//data/tle_propagation.pkl')
    pklFile = open(fname, 'rb')
    data = pickle.load(pklFile)
    output_state = data[0]
    pklFile.close()
    
    # Compute latitude, longitude
    obj_id = 39084
    lat_list = []
    lon_list = []
    for r_ITRF in output_state[obj_id]['r_ITRF']:
        lat, lon, ht = ecef2latlonht(r_ITRF)
        lat_list.append(lat)
        lon_list.append(lon)
        
    # Print for comparison
    print('UTC', output_state[obj_id]['UTC'][0])
    print('r_GRCF', output_state[obj_id]['r_GCRF'][0])
    print('v_GRCF', output_state[obj_id]['v_GCRF'][0])
    
    print(lat_list)
    print(lon_list)
    
    # Generate plot
    plt.figure()
    plt.plot(lon_list, lat_list, 'bo', ms=1)
            
    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,
                urcrnrlon=180,resolution='c')
    m.drawcoastlines()
    m.drawmeridians(np.arange(-180, 180, 45))
    m.drawparallels(np.arange(-90, 90, 45))
    m.drawmapboundary()
    m.drawcountries()
    plt.yticks(np.arange(-90, 91, 45))
    plt.xticks(np.arange(-180, 181, 45))

    
    plt.show()
        
    
    return


def compute_coverage():
    
    fov = 15. * pi/180.
    GM = GME
    R = Re
    
    # Load data
    fname = os.path.join('..//data/tle_propagation2.pkl')
    pklFile = open(fname, 'rb')
    data = pickle.load(pklFile)
    output_state = data[0]
    pklFile.close()
    
    # EOP and Frame Rotation parameters
    XYs_df = get_XYs2006_alldata()    
    eop_alldata = get_celestrak_eop_alldata()
    
    
    # Compute SMA
    obj_id = 39084
    r_GCRF = output_state[obj_id]['r_GCRF'][0]
    v_GCRF = output_state[obj_id]['v_GCRF'][0]
    r = np.linalg.norm(r_GCRF)
    v2 = np.linalg.norm(v_GCRF)**2
    a = 1./(2./r - v2/GM)     # km
    
    # Compute angles using Ref 1 Eq 12.2 - 12.5
    f = (fov/2.)
    zeta = asin(a*sin(f)/R)     
    alpha = zeta - f
    
    # RIC frame vectors to add to chief orbit
    z = alpha * a
    rho1 = np.array([[0.], [0.], [z]])
    rho2 = np.array([[0.], [0.], [-z]])
    
    # Loop over times    
    N = len(output_state[obj_id]['UTC'])
    lon_list = []
    lat_list = []
    lon1_list = []
    lat1_list = []
    lon2_list = []
    lat2_list = []
    for ii in range(N):
        
        # Retrieve current state
        UTC = output_state[obj_id]['UTC'][ii]
        r_GCRF = output_state[obj_id]['r_GCRF'][ii]
        v_GCRF = output_state[obj_id]['v_GCRF'][ii]
        r_ITRF = output_state[obj_id]['r_ITRF'][ii]
        
        # Compute ECI states for z component in RIC frame
        r_GCRF_1 = ric2eci(r_GCRF, v_GCRF, rho1) + r_GCRF
        r_GCRF_2 = ric2eci(r_GCRF, v_GCRF, rho2) + r_GCRF
        
#        print(r_GCRF)
#        print(r_GCRF_1)
#        print(r_GCRF_2)
#        
#        print(rho1)
#        print(rho2)
        
        # Compute ECEF components
        EOP_data = get_eop_data(eop_alldata, UTC)
        r_ITRF_1, v_ITRF = gcrf2itrf(r_GCRF_1, v_GCRF, UTC, EOP_data, XYs_df)
        r_ITRF_2, v_ITRF = gcrf2itrf(r_GCRF_2, v_GCRF, UTC, EOP_data, XYs_df)
        
        # Compute latitude, longitude
        lat, lon, ht = ecef2latlonht(r_ITRF)
        lat1, lon1, ht = ecef2latlonht(r_ITRF_1)
        lat2, lon2, ht = ecef2latlonht(r_ITRF_2)
        
        lon_list.append(lon)
        lat_list.append(lat)
        lon1_list.append(lon1)
        lat1_list.append(lat1)
        lon2_list.append(lon2)
        lat2_list.append(lat2)
        
#        print(lat, lon)
#        print(lat1, lon1)
#        print(lat2, lon2)
#        
#        mistake
        
        
    # Generate plot
    plt.figure()
    plt.plot(lon_list, lat_list, 'bo', ms=1)
    plt.plot(lon1_list, lat1_list, 'ro', ms=1)
    plt.plot(lon2_list, lat2_list, 'go', ms=1)
            
    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,
                urcrnrlon=180,resolution='c')
    m.drawcoastlines()
    m.drawmeridians(np.arange(-180, 180, 45))
    m.drawparallels(np.arange(-90, 90, 45))
    m.drawmapboundary()
    m.drawcountries()
    plt.yticks(np.arange(-90, 91, 45))
    plt.xticks(np.arange(-180, 181, 45))

    
    plt.show()   
    
    
    
    return


def compute_groundswath(a, fov, R=Re):
    '''
    This function computes the ground swath width in rad and km for a given
    orbit and satellite field of view. FOV is taken to be the full angle 
    visible from the satellite looking toward nadir.
    
    Parameters
    ------
    a : float
        semi-major axis [km]
    fov : float
        field of view [rad]
    R : float, optional
        radius of central body (default=Re)
        
    Returns
    ------
    swath_rad : float
        swath angle on surface of planet at equator [rad]
    swath_km : float
        swath distance on surface of planet at equator [km]
    
    
    '''
    
    # Compute angles using Ref 1 Eq 12.2 - 12.5
    f = (fov/2.)
    zeta = asin(a*sin(f)/R)     
    alpha = zeta - f
    
    # Compute full swath using Ref 1 Eq 
    swath_rad = alpha*2.
    swath_km = swath_rad*R

    return swath_rad, swath_km


def swath2fov(a, swath_rad, R=Re):
    '''
    
    '''
    
    
    alpha = swath_rad/2.
    rho = np.sqrt(R**2. + a**2. - 2.*Re*a*cos(alpha))
    f = asin((sin(alpha)/rho)*Re)
    fov = 2.*f
    
    return fov


def plot_swath_vs_altitude():
    '''
    This function creates a plot of ground swath in degrees vs altitude
    from 400-600 km.
    
    '''
    
    fov_list = [10., 15., 20.]
    alt_list = list(np.arange(400., 600., 1.))
    
    swath_data = np.zeros((len(fov_list), len(alt_list)))
    Nto_data = np.zeros((len(fov_list), len(alt_list)))
    Cto_data = np.zeros((len(fov_list), len(alt_list)))
    for fov in fov_list:
        for alt in alt_list:
            a = Re + alt
            n = np.sqrt(GME/a**3.) * 86400./(2.*pi)     # rev/day
            
            
            swath_rad, swath_km = compute_groundswath(a, fov*pi/180.)
            swath_data[fov_list.index(fov), alt_list.index(alt)] = swath_rad*180./pi
            
            Nto_min = np.ceil(2.*pi/swath_rad)
            Cto_min = np.floor(Nto_min/n)
            Nto_data[fov_list.index(fov), alt_list.index(alt)] = Nto_min
            Cto_data[fov_list.index(fov), alt_list.index(alt)] = Cto_min
            
    
    
    # Generate plot
    plt.figure()
    plt.plot(alt_list, swath_data[0,:], 'r.', label='10deg')
    plt.plot(alt_list, swath_data[1,:], 'g.', label='15deg')
    plt.plot(alt_list, swath_data[2,:], 'b.', label='20deg')
    plt.xlabel('Altitude [km]')
    plt.ylabel('Ground Swath Width [deg]')    
    plt.legend()
    
    plt.figure()
    plt.plot(alt_list, Nto_data[0,:], 'r.', label='10deg')
    plt.plot(alt_list, Nto_data[1,:], 'g.', label='15deg')
    plt.plot(alt_list, Nto_data[2,:], 'b.', label='20deg')
    plt.xlabel('Altitude [km]')
    plt.ylabel('Minimum Number of Revs for Repeat Nto')    
    plt.legend()
    
    plt.figure()
    plt.plot(alt_list, Cto_data[0,:], 'r.', label='10deg')
    plt.plot(alt_list, Cto_data[1,:], 'g.', label='15deg')
    plt.plot(alt_list, Cto_data[2,:], 'b.', label='20deg')
    plt.xlabel('Altitude [km]')
    plt.ylabel('Minimum Number of Days for Repeat Cto')    
    plt.legend()
    
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(alt_list, swath_data[0,:], 'r.', label='10deg')
    plt.plot(alt_list, swath_data[1,:], 'g.', label='15deg')
    plt.plot(alt_list, swath_data[2,:], 'b.', label='20deg')
    plt.ylabel('Swath [deg]')    
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.plot(alt_list, Nto_data[0,:], 'r.', label='10deg')
    plt.plot(alt_list, Nto_data[1,:], 'g.', label='15deg')
    plt.plot(alt_list, Nto_data[2,:], 'b.', label='20deg')
    plt.ylabel('Min Nto')
    
    plt.subplot(3,1,3)
    plt.plot(alt_list, Cto_data[0,:], 'r.', label='10deg')
    plt.plot(alt_list, Cto_data[1,:], 'g.', label='15deg')
    plt.plot(alt_list, Cto_data[2,:], 'b.', label='20deg')
    plt.ylabel('Min Cto')
    plt.xlabel('Altitude [km]')
    
    
    
    
    plt.show()
    
    
    return


###############################################################################
#
# Candidate Orbits
#
###############################################################################
    

def generate_tle_dict():
    
    
    tle_dict = {}
    
    
    # Global Parameters
    LTAN = 22.5
    BSTAR = '28772-4'
    UTC = datetime(2020, 6, 1, 0, 0, 0)
    
    
###############################################################################
# Landsat-8 Trailing Orbits
###############################################################################
    
    # Baseline Case - Same orbit as Landsat-8 but chasing
    # First attempt, use recurrent triple
    obj_id = 0
    vo = 15.
    Dto = -7.
    Cto = 16.
    
    elem = compute_recurrent_orbit_parameters(vo, Dto, Cto, UTC, LTAN)
    
    # Compute TLE
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
#    print(line1)
#    print(line2)
    
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    
    
    # Second attempt, use Landsat-8 TLE data
    obj_id = 1
    a = 7070.403227728693
#    e = 0.0007156594641872634
#    i = 98.1545361611416
    
    elem = compute_orbit_parameters(a, UTC, LTAN)
    
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
#    print(line1)
#    print(line2)
    
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    
#    # Check values
#    obj_id_list = [0, 1]
#    UTC_list = [UTC]
#    output_state = prop_TLE_full(obj_id_list, UTC_list, tle_dict)
    
#    print(output_state)
    
    
    
###############################################################################
# Recurrent orbits - Primary List
###############################################################################
    
    obj_id = 10001
    elem = compute_recurrent_orbit_parameters(15, 3, 16, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10002
    elem = compute_recurrent_orbit_parameters(15, 5, 16, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10003
    elem = compute_recurrent_orbit_parameters(15, 3, 20, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10004
    elem = compute_recurrent_orbit_parameters(15, 7, 20, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10005
    elem = compute_recurrent_orbit_parameters(15, 9, 20, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10006
    elem = compute_recurrent_orbit_parameters(15, 9, 40, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10007
    elem = compute_recurrent_orbit_parameters(15, 11, 40, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10008
    elem = compute_recurrent_orbit_parameters(15, 13, 40, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10009
    elem = compute_recurrent_orbit_parameters(15, 17, 40, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10010
    elem = compute_recurrent_orbit_parameters(15, 19, 40, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10011
    elem = compute_recurrent_orbit_parameters(16, -19, 40, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10012
    elem = compute_recurrent_orbit_parameters(15, -3, 40, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10013
    elem = compute_recurrent_orbit_parameters(15, 3, 40, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10014
    elem = compute_recurrent_orbit_parameters(15, 7, 40, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10015
    elem = compute_recurrent_orbit_parameters(15, -3, 16, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10016
    elem = compute_recurrent_orbit_parameters(15, -3, 20, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10017
    elem = compute_recurrent_orbit_parameters(15, -9, 40, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10018
    elem = compute_recurrent_orbit_parameters(15, -7, 40, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 10019
    elem = compute_recurrent_orbit_parameters(15, -13, 48, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
	
    obj_id = 10020
    elem = compute_recurrent_orbit_parameters(15, -11, 48, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
	
    obj_id = 10021
    elem = compute_recurrent_orbit_parameters(15, -7, 48, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
	
    obj_id = 10022
    elem = compute_recurrent_orbit_parameters(15, -5, 48, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
	
    obj_id = 10023
    elem = compute_recurrent_orbit_parameters(15, -5, 56, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
	
    obj_id = 10024
    elem = compute_recurrent_orbit_parameters(15, -3, 56, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
	
    obj_id = 10025
    elem = compute_recurrent_orbit_parameters(15, -15, 56, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
	
    obj_id = 10026
    elem = compute_recurrent_orbit_parameters(15, -13, 56, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
	
    obj_id = 10027
    elem = compute_recurrent_orbit_parameters(15, -11, 56, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
	
    obj_id = 10028
    elem = compute_recurrent_orbit_parameters(15, -9, 56, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
	
    obj_id = 10029
    elem = compute_recurrent_orbit_parameters(15, -13, 60, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
	
    obj_id = 10030
    elem = compute_recurrent_orbit_parameters(15, -17, 60, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
	
    obj_id = 10031
    elem = compute_recurrent_orbit_parameters(15, -11, 60, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
	
    obj_id = 10032
    elem = compute_recurrent_orbit_parameters(15, -7, 60, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    

###############################################################################
# Non-recurrent SSO
###############################################################################  
    
    obj_id = 20001
    a = 6378.137 + 500.
    elem = compute_orbit_parameters(a, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 20002
    a = 6378.137 + 510.
    elem = compute_orbit_parameters(a, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 20003
    a = 6378.137 + 520.
    elem = compute_orbit_parameters(a, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 20004
    a = 6378.137 + 530.
    elem = compute_orbit_parameters(a, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 20005
    a = 6378.137 + 540.
    elem = compute_orbit_parameters(a, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 20006
    a = 6378.137 + 550.
    elem = compute_orbit_parameters(a, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 20007
    a = 6378.137 + 560.
    elem = compute_orbit_parameters(a, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 20008
    a = 6378.137 + 570.
    elem = compute_orbit_parameters(a, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 20009
    a = 6378.137 + 580.
    elem = compute_orbit_parameters(a, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 20010
    a = 6378.137 + 590.
    elem = compute_orbit_parameters(a, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 20011
    a = 6378.137 + 600.
    elem = compute_orbit_parameters(a, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 20012
    a = 6378.137 + 610.
    elem = compute_orbit_parameters(a, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 20013
    a = 6378.137 + 620.
    elem = compute_orbit_parameters(a, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 20014
    a = 6378.137 + 630.
    elem = compute_orbit_parameters(a, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 20015
    a = 6378.137 + 640.
    elem = compute_orbit_parameters(a, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    obj_id = 20016
    a = 6378.137 + 650.
    elem = compute_orbit_parameters(a, UTC, LTAN)
    line1, line2 = kep2tle(obj_id, elem, UTC, BSTAR)
    tle_dict[obj_id] = {}
    tle_dict[obj_id]['UTC_list'] = [UTC]
    tle_dict[obj_id]['line1_list'] = [line1]
    tle_dict[obj_id]['line2_list'] = [line2]
    
    
    
    
    # Save output
    fname = os.path.join('../data/tle_data_candidate_orbits_2020_06_01.pkl')
    pklFile = open( fname, 'wb' )
    pickle.dump( [tle_dict], pklFile, -1 )
    pklFile.close()
    
    
    return


def kep2tle(obj_id, elem, UTC, BSTAR='00000-0', EOP_data=[], IAU1980nut=[], 
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
    cart = kep2cart(elem)
    r_GCRF = np.reshape(cart[0:3], (3,1))
    v_GCRF = np.reshape(cart[3:6], (3,1))
    r_TEME, v_TEME = gcrf2teme(r_GCRF, v_GCRF, UTC, IAU1980nut, EOP_data)
    cart_TEME = np.concatenate((r_TEME, v_TEME), axis=0)
    osc_elem = element_conversion(cart_TEME, 1, 0)



    print(osc_elem)

    # Convert to mean elements
    # TODO currently it appears osculating elements gives more accurate result
    # Need further investigation of proper computation of TLEs.
    # Temporary solution, just use osculating elements instead of mean elements
    
    mean_elem = osc2mean(osc_elem)
#    mean_elem = list(osc_elem.flatten())

    # Retrieve elements
    a = float(mean_elem[0])
    e = float(mean_elem[1])
    i = float(mean_elem[2])
    RAAN = float(mean_elem[3]) % 360.
    w = float(mean_elem[4]) % 360.
    M = float(mean_elem[5]) % 360.

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
    line1 = '1 ' + str(obj_id).zfill(5) + 'U ' + year2 + '001A   ' + year2 + \
        str(doy).zfill(3) + '.' + str(dfrac)[2:10] + \
        '  .00000000  00000-0  ' + BSTAR + ' 0    10'

    line2 = '2 ' + str(obj_id).zfill(5) + ' ' + '{:8.4f}'.format(i) + ' ' + \
        '{:8.4f}'.format(RAAN) + ' ' + e[2:9] + ' ' + \
        '{:8.4f}'.format(w) + ' ' + '{:8.4f}'.format(M) + ' ' + \
        '{:11.8f}'.format(n) + '    10'

    return line1, line2


###############################################################################
#
# Recurrent Orbit Functions
#
###############################################################################


def compute_recurrence_grid_parameters(vo, Dto, Cto):
    '''
    This function computes the recurrence grid parameters, the angular 
    difference in nodal longitude between consecutive revolutions, 
    consecutive days, and for the full repeating groundtrack.
    
    Parameters
    ------
    vo = int
        whole number of revolutions per day (rounded to nearest int)
    Dto = int
        whole number remainder such that kappa = vo + Dto/Cto
        Dto = mod(Nto, Cto) such that Dto/Cto <= 0.5
        Dto and Cto should be coprime
    Cto = int
        whole number of days before repeat
    
    Returns
    ------
    delta : float
        grid interval at the equator [rad]
    delta_rev : float
        difference in nodal longitude for consecutive revolutions [rad]
    delta_day : float
        difference in nodal longitude for consecutive days [rad]
    '''
    
    # Compute recurrence parameters
    kappa = vo + float(Dto)/float(Cto)       # rev/day
    Nto = vo*Cto + Dto
    
    delta = 2*pi/Nto
    delta_rev = delta*Cto
    delta_day = delta*Dto   
    
    return delta, delta_rev, delta_day


def generate_candidate_recurrent_triples(hmin, hmax, fov, R=Re, GM=GME):
    '''
    This function generates recurrent triples within a user defined altitude
    range.
    
    Parameters
    ------
    hmin : float
        minimum altitude [km]
    hmax : float
        maximum altitude [km]
    fov : float
        field of view [rad]
    R : float, optional
        planet radius [km] (default=Re)
    GM : float, optional
        planet gravitational parameter [km^3/s^2] (default=GME)
    
    Returns
    ------
    triple_primary_list : list
        list of lists, each entry contains [vo, Dto, Cto, Nto, Eto]
    triple_secondary_list : list
        list of lists, each entry contains [vo, Dto, Cto, Nto, Eto]  
    
    '''
    
    Cto_primary_list = [16., 20., 40.]
#    Cto_secondary_list = [10., 12., 15., 24., 25., 30., 32., 35.]
    Cto_secondary_list = [48., 56., 60.]
    
    triple_primary_list = compute_triple_list(hmin, hmax, fov, Cto_primary_list, R, GM)
    triple_secondary_list = compute_triple_list(hmin, hmax, fov, Cto_secondary_list, R, GM)
    
    # Generate data to plot and save in csv
    print(triple_primary_list)
    print(triple_secondary_list)
    
    pandas_data_list = []
    plot_v_primary = []
    plot_Cto_primary = []
    plot_h_primary = []
    for primary_list in triple_primary_list:
        
        vo = primary_list[0]
        Dto = primary_list[1]
        Cto = primary_list[2]
        Nto = primary_list[3]
        Eto = primary_list[4]
        
        # Assume near-circular sunsynch orbit
        a, i = nodal_period_to_sunsynch_orbit(Nto, Cto, 1e-4)
        h = a - R
        
        plot_v_primary.append(Nto/Cto)
        plot_Cto_primary.append(Cto)
        plot_h_primary.append(h)
        
        # Compute grid parameters
        delta, delta_rev, delta_day = compute_recurrence_grid_parameters(vo, Dto, Cto)
        
        # Compute swath and FOV requirements
        swath_km = delta*Re
        fov = swath2fov(a, delta)
        fov_deg = fov * 180./pi
        
        
        
        data_list = [vo, Dto, Cto, Nto, Eto, h, swath_km, fov_deg]
        pandas_data_list.append(data_list)
        
    plot_v_secondary = []
    plot_Cto_secondary = []
    plot_h_secondary = []
    for secondary_list in triple_secondary_list:
        
        vo = secondary_list[0] 
        Dto = secondary_list[1]
        Cto = secondary_list[2]
        Nto = secondary_list[3]
        Eto = secondary_list[4]
        
        # Assume near-circular sunsych orbit
        a, i = nodal_period_to_sunsynch_orbit(Nto, Cto, 1e-4)
        h = a - R
        
        plot_v_secondary.append(Nto/Cto)
        plot_Cto_secondary.append(Cto)
        plot_h_secondary.append(h)
        
        # Compute grid parameters
        delta, delta_rev, delta_day = compute_recurrence_grid_parameters(vo, Dto, Cto)
        
        # Compute swath and FOV requirements
        swath_km = delta*Re
        fov = swath2fov(a, delta)
        fov_deg = fov * 180./pi
        
        data_list = [vo, Dto, Cto, Nto, Eto, h, swath_km, fov_deg]
        pandas_data_list.append(data_list)
        
        
      
    # Generate plots
    n_15 = 15.*2.*pi/86400.
    a_15 = (GM/n_15**2.)**(1./3.)
    h_15 = a_15 - R
        
        
#    plt.figure()
#    plt.plot(plot_Cto_primary, plot_v_primary, 'b*', markersize=8)
#    plt.gca().invert_yaxis()
    
#    fig, ax1 = plt.subplots()
#    ax1.plot(plot_Cto_primary, plot_v_primary, 'b*', markersize=8)
#    ax1.set_xlabel('Repeat Cycle [days]')
#    ax1.set_ylabel('Revolutions Per Day')
    
    
    plt.figure()
    plt.plot(plot_Cto_primary, plot_h_primary, 'k.' ) #, markersize=8, label='Primary')
    plt.plot(plot_Cto_secondary, plot_h_secondary, 'k.' ) #, label='Secondary')
#    plt.plot([10., 45.], [h_15, h_15], 'k--', label='15 rev/day')
    plt.ylim([400., 650.])
    plt.xlim([10., 65.])
    plt.xlabel('Repeat Cycle [days]')
    plt.ylabel('Altitude [km]')
#    plt.legend()
    
    
    
    plt.show()
        
    
    
    # Generate pandas dataframe 
    column_headers = ['vo [rev/day]', 'Dto [revs]', 'Cto [days]',
                      'Nto [revs]', 'Eto [days]', 'Altitude [km]',
                      'Min Swath [km]', 'Min FOV [deg]']
    
    recurrent_df = pd.DataFrame(pandas_data_list, columns = column_headers)

    return recurrent_df


def compute_triple_list(hmin, hmax, fov, Cto_list, R=Re, GM=GME):
    '''
    This function computes a list of recurrent triples for the specified
    range of mean motion in rev/day and desired number of recurrent days 
    Cto.
    
    Parameters
    ------
    n_min : float
        minimum number of revs/day
    n_max : float
        maximum number of revs/day
    Cto_list : list
        desired whole numbers of days in repeat cycly
    
    Returns
    ------
    triple_list : list
        list of lists, each entry contains [vo, Dto, Cto, Nto, Eto]
    
    '''
    
    # Compute minumum and maximum mean motion in rev/day
    # Note that Keplerian period and mean motion are slightly different from 
    # nodal period and mean motion, but should be ok to set up these bounds
    a_min = R + hmin
    a_max = R + hmax
    n_max = np.sqrt(GM/a_min**3.) * 86400./(2.*pi)     # rev/day
    n_min = np.sqrt(GM/a_max**3.) * 86400./(2.*pi)     # rev/day
    
    
    
    # Find values of Nto that create rational numbers for valid ranges of 
    # Cto
    triple_list = []
    for Cto in Cto_list:
        
        # Generate candidate values of Nto
        Nto_min = np.ceil(Cto*n_min)
        Nto_max = np.floor(Cto*n_max)
        Nto_range = list(np.arange(Nto_min, Nto_max))
        
        # Remove entries that are not coprime
        del_list = []
        for Nto in Nto_range:
            if not is_coprime(Nto, Cto):
                del_list.append(Nto)
                
        Nto_list = list(set(Nto_range) - set(del_list))
                
        # Compute triples
        for Nto in Nto_list:
            vo = np.round(Nto/Cto)
            Dto = Nto - vo*Cto
            
            # Skip entries that have Eto = 1
            Eto = compute_Eto(vo, Dto, Cto)
            if Eto == 1:
                continue
            
            # Check delta < swath to ensure full global coverage
            # Assume near-circular sunsynch orbit
            a, i = nodal_period_to_sunsynch_orbit(Nto, Cto, 1e-4)
            swath_rad, swath_km = compute_groundswath(a, fov)
            delta = 2*pi/Nto
            
#            print(Nto, Cto)
#            print(a)
#            print(swath_rad, delta)
            
            if delta > swath_rad : 
                continue            
            
            triple = [vo, Dto, Cto, Nto, Eto]
            triple_list.append(triple)
            
#        print(Nto_list)
#        print(del_list)
#        print(triple_list)
   
    
    return triple_list


def is_coprime(x, y):
    '''
    This function checks if two numbers are coprime
    
    Parameters
    ------
    x : int
        larger number
    y : int
        smaller number
        
    Returns
    ------
    coprime_flag : boolean
        True indicates the numbers are coprime, False indicates they are not
    
    '''
    
    while y != 0:
        x, y = y, x % y
    
    coprime_flag = x == 1    
    
    return coprime_flag


def compute_Eto(vo, Dto, Cto):
    '''
    This function computes the subcycle recurrence Eto, the number of days
    it takes for the first groundtrack to pass within delta of the original
    groundtrack. It is good practice to avoid Eto = 1 to coverage the base 
    interval faster than using the whole repeat cycle.
    
    Parameters
    ------
    vo = int
        whole number of revolutions per day (rounded to nearest int)
    Dto = int
        whole number remainder such that kappa = vo + Dto/Cto
        Dto = mod(Nto, Cto) such that Dto/Cto <= 0.5
        Dto and Cto should be coprime
    Cto = int
        whole number of days before repeat
        
    Returns
    ------
    Eto = int
        whole number of days in subcycle
    
    '''
    
    Eto_list = []
    for ii in range(1, int(Cto)):
        
        if (ii*Dto) % Cto == 1 or (ii*Dto) % Cto == Cto - 1:
            Eto_list.append(ii)
            
    Eto = min(Eto_list)
    
    
    return Eto


def compute_orbit_periods(a, e, i, R=Re, GM=GME, J2=J2E):
    '''
    This function computes the Keplerian, Anomalistic, and Nodal Periods of
    an orbit subject to J2 perturbation.
    
    Parameters
    ------
    a : float
        semi-major axis [km]
    e : float
        eccentricity
    i : float
        inclination [deg]
    R : float, optional
        radius of planet [km] (default=Re)
    GM : float, optional
        gravitiational parameter [km^3/s^2] (default=GME)
    J2 : float, optional
        J2 coefficient (default=J2E)
    
    Returns
    ------
    To : float
        Keplerian orbit period [sec]
    Ta : float
        anomalistic orbit period [sec]
    Td : float
        nodal orbit period [sec]
    
    '''
    
    # Convert inclination to radians
    i = i * pi/180.
    
    # Compute Keplerian orbit period
    no = np.sqrt(GM/a**3.)
    To = 2.*pi/no
    
    # Compute perturbation effects from J2
    dn = (3./(4.*(1-e**2.)**(3./2.))) * no * J2 * (R/a)**2. * (3.*cos(i)**2. - 1)
    dw = (3./(4.*(1-e**2.)**(2.))) * no * J2 * (R/a)**2. * (5.*cos(i)**2. - 1)
    
    # Compute anomalistic orbit period
    na = no + dn
    Ta = 2.*pi/na
    
    # Compute nodal period
    Td = ((1. - dn/no)/(1. + dw/no)) * To
       
    return To, Ta, Td


def nodal_period_to_sunsynch_orbit(Nto, Cto, e, R=Re, GM=GME, J2=J2E):
    
    # Compute constants
    sidereal_day = 2.*pi/wE
    k2 = 0.75 * (360./(2.*pi)) * J2 * np.sqrt(GM) * R**2. * sidereal_day
    
    # Initial guess for SMA
    Td = Cto/Nto * sidereal_day
    n = 2.*pi/Td
    a = meanmot2sma(n, GM)
    
    # Iteratively solve for SMA
    a_prev = float(a)
    diff = 1.
    tol = 1e-4
    while diff > tol:
        
        # Compute inclination
        i = sunsynch_inclination(a, e)         # deg
        i = i * pi/180.                        # rad        
    
        # Compute J2 effects 
        dL = 360.       # deg/sidereal day
        dRAAN = -2.* k2 * a**(-7./2.) * cos(i) * (1. - e**2.)**(-2.)
        dw = k2 * a**(-7./2.) * (5.*cos(i)**2. - 1) * (1. - e**2.)**(-2.)
        dM = k2 * a**(-7./2.) * (3.*cos(i)**2. - 1) * (1. - e**2.)**(-3./2.)
        
        n = (Nto/Cto) * (dL - dRAAN) - (dw + dM)
        a = (GM**(1./3.)) * ((n*pi)/(180.*sidereal_day))**(-2./3.)
        diff = abs(a - a_prev)
        a_prev = float(a)
    
    
    # Convert to deg
    i = i * 180./pi
    
    return a, i


def compute_orbit_parameters(a, UTC, LTAN, GM=GME):
    
    # Choose eccentricity small and set AOP = 0 (periapsis at ascending node)
    e = 1e-4
    w = 0.
    
    # Retrieve EOP data for current time
    eop_alldata = get_celestrak_eop_alldata()
    EOP_data = get_eop_data(eop_alldata, UTC)
        
    # Compute i
    i = sunsynch_inclination(a, e)
    
    # Use LTAN to compute RAAN
    RAAN = LTAN_to_RAAN(LTAN, UTC, EOP_data)
    
    # Retrieve TLE data for Landsat-8 and Sentinel-2
    landsat8_norad = 39084
    sentinel_2a_norad = 40697
    sentinel_2b_norad = 42063
    obj_id_list = [landsat8_norad, sentinel_2a_norad, sentinel_2b_norad]
    UTC_list = [UTC]
    output_state = prop_TLE_full(obj_id_list, UTC_list, tle_dict={}, 
                                 offline_flag=False,
                                 username='steve.gehly@gmail.com', 
                                 password='SpaceTrackPword!')

    landsat8_pos = output_state[landsat8_norad]['r_GCRF'][0]
    landsat8_vel = output_state[landsat8_norad]['v_GCRF'][0]
    landsat8_cart = np.concatenate((landsat8_pos.flatten(), landsat8_vel.flatten()))
    landsat8_kep = cart2kep(landsat8_cart)
    
#    sentinel_2a_pos = output_state[sentinel_2a_norad]['r_GCRF'][0]
#    sentinel_2a_vel = output_state[sentinel_2a_norad]['v_GCRF'][0]
#    sentinel_2a_cart = np.concatenate((sentinel_2a_pos.flatten(), sentinel_2a_vel.flatten()))
#    sentinel_2a_kep = cart2kep(sentinel_2a_cart)
#    
#    sentinel_2b_pos = output_state[sentinel_2b_norad]['r_GCRF'][0]
#    sentinel_2b_vel = output_state[sentinel_2b_norad]['v_GCRF'][0]
#    sentinel_2b_cart = np.concatenate((sentinel_2b_pos.flatten(), sentinel_2b_vel.flatten()))
#    sentinel_2b_kep = cart2kep(sentinel_2b_cart)
    
    landsat8_RAAN = float(landsat8_kep[3])
#    sentinel_2a_RAAN = float(sentinel_2a_kep[3])
#    sentinel_2b_RAAN = float(sentinel_2b_kep[3])
    
    landsat8_LTAN = RAAN_to_LTAN(landsat8_RAAN, UTC, EOP_data)
#    sentinel_2a_LTAN = RAAN_to_LTAN(sentinel_2a_RAAN, UTC, EOP_data)
#    sentinel_2b_LTAN = RAAN_to_LTAN(sentinel_2b_RAAN, UTC, EOP_data)
    
    # Compute true anomaly for this time
    landsat_true_longitude = float(sum(landsat8_kep[4:6]))
    To = 2.*pi*np.sqrt(a**3./GM)        # sec
    theta = landsat_true_longitude + ((landsat8_LTAN - LTAN)*3600./To)*360. - w    
    theta = theta % 360.
    
    print(landsat_true_longitude)
    print(To)
    print(((landsat8_LTAN - LTAN)*3600./To))
    
    # Form vector of orbit elements
    elem = np.reshape([a, e, i, RAAN, w, theta], (6,1))    
    
    
    print('\nLandsat-8')
    print('SMA [km]: ', float(landsat8_kep[0]))
    print('Alt [km]: ', float(landsat8_kep[0])-6378.137)
    print('ECC: ', float(landsat8_kep[1]))
    print('INC [deg]: ', float(landsat8_kep[2]))
    print('RAAN [deg]: ', float(landsat8_kep[3]))
    print('AOP [deg]: ', float(landsat8_kep[4]))
    print('TA [deg]: ', float(landsat8_kep[5]))
    print('True Long [deg]: ', float(landsat8_kep[4])+float(landsat8_kep[5]))
    print('LTAN [hours]: ', landsat8_LTAN)
    
    return elem


def compute_recurrent_orbit_parameters(vo, Dto, Cto, UTC, LTAN, GM=GME):
    '''
    
    '''
    
    # Choose eccentricity small and set AOP = 0 (periapsis at ascending node)
    e = 1e-4
    w = 0.
    
    # Retrieve EOP data for current time
    eop_alldata = get_celestrak_eop_alldata()
    EOP_data = get_eop_data(eop_alldata, UTC)
    
    # Number of revolutions
    Nto = vo*Cto + Dto

    # Compute a, i
    a, i = nodal_period_to_sunsynch_orbit(Nto, Cto, e)
    
    # Use LTAN to compute RAAN
    RAAN = LTAN_to_RAAN(LTAN, UTC, EOP_data)
    
    # Retrieve TLE data for Landsat-8 and Sentinel-2
    landsat8_norad = 39084
    sentinel_2a_norad = 40697
    sentinel_2b_norad = 42063
    obj_id_list = [landsat8_norad, sentinel_2a_norad, sentinel_2b_norad]
    UTC_list = [UTC]
    output_state = prop_TLE_full(obj_id_list, UTC_list, tle_dict={}, 
                                 offline_flag=False,
                                 username='steve.gehly@gmail.com', 
                                 password='SpaceTrackPword!')

    landsat8_pos = output_state[landsat8_norad]['r_GCRF'][0]
    landsat8_vel = output_state[landsat8_norad]['v_GCRF'][0]
    landsat8_cart = np.concatenate((landsat8_pos.flatten(), landsat8_vel.flatten()))
    landsat8_kep = cart2kep(landsat8_cart)
    
#    sentinel_2a_pos = output_state[sentinel_2a_norad]['r_GCRF'][0]
#    sentinel_2a_vel = output_state[sentinel_2a_norad]['v_GCRF'][0]
#    sentinel_2a_cart = np.concatenate((sentinel_2a_pos.flatten(), sentinel_2a_vel.flatten()))
#    sentinel_2a_kep = cart2kep(sentinel_2a_cart)
#    
#    sentinel_2b_pos = output_state[sentinel_2b_norad]['r_GCRF'][0]
#    sentinel_2b_vel = output_state[sentinel_2b_norad]['v_GCRF'][0]
#    sentinel_2b_cart = np.concatenate((sentinel_2b_pos.flatten(), sentinel_2b_vel.flatten()))
#    sentinel_2b_kep = cart2kep(sentinel_2b_cart)
    
    landsat8_RAAN = float(landsat8_kep[3])
#    sentinel_2a_RAAN = float(sentinel_2a_kep[3])
#    sentinel_2b_RAAN = float(sentinel_2b_kep[3])
    
    landsat8_LTAN = RAAN_to_LTAN(landsat8_RAAN, UTC, EOP_data)
#    sentinel_2a_LTAN = RAAN_to_LTAN(sentinel_2a_RAAN, UTC, EOP_data)
#    sentinel_2b_LTAN = RAAN_to_LTAN(sentinel_2b_RAAN, UTC, EOP_data)
    
    # Compute true anomaly for this time
    landsat_true_longitude = float(sum(landsat8_kep[4:6]))
    To = 2.*pi*np.sqrt(a**3./GM)        # sec
    theta = landsat_true_longitude + ((landsat8_LTAN - LTAN)*3600./To)*360. - w    
    theta = theta % 360.
    
    print(landsat_true_longitude)
    print(To)
    print(((landsat8_LTAN - LTAN)*3600./To))
    
    # Form vector of orbit elements
    elem = np.reshape([a, e, i, RAAN, w, theta], (6,1))    
    
    
    print('\nLandsat-8')
    print('SMA [km]: ', float(landsat8_kep[0]))
    print('Alt [km]: ', float(landsat8_kep[0])-6378.137)
    print('ECC: ', float(landsat8_kep[1]))
    print('INC [deg]: ', float(landsat8_kep[2]))
    print('RAAN [deg]: ', float(landsat8_kep[3]))
    print('AOP [deg]: ', float(landsat8_kep[4]))
    print('TA [deg]: ', float(landsat8_kep[5]))
    print('True Long [deg]: ', float(landsat8_kep[4])+float(landsat8_kep[5]))
    print('LTAN [hours]: ', landsat8_LTAN)
    
    return elem


def recurrence_analysis():
    
    # Choose values of recurrence triples in acceptable range
    nu_list = [14.5, 15.5]        # rev/day
    h_list = []
    for nu in nu_list:
        n = nu * (2.*pi)/86400  # rad/s
        a = (meanmot2sma(n))
        h = a - Re
        h_list.append(h)
        
    print(h_list)
    
    
    return


def unit_test_recurrence():
    
    
    UTC = datetime(2020, 6, 1)
    LTAN = 22.5
    
    # Landsat-8
    vo = 15.
    Dto = -7.
    Cto = 16.
    
    elem = compute_recurrent_orbit_parameters(vo, Dto, Cto, UTC, LTAN)
    
    print('Landsat-8 Alt: ', elem[0] - Re)
    print('Landsat-8 Inc: ', elem[2])
    
    
    # Sentinel-2
    vo = 14.
    Dto = 3.
    Cto = 10.
    
    elem = compute_recurrent_orbit_parameters(vo, Dto, Cto, UTC, LTAN)
    
    print('Sentinel-2 Alt: ', elem[0] - Re)
    print('Sentinel-2 Inc: ', elem[2])
    
    
    
    return


def unit_test_intersect():
    
    p1 = [-178, 80]
    p2 = [-175, 80]
    p3 = [175, 75]
    p4 = [175, 85]
    
    x, y = compute_intersect(p1, p2, p3, p4)
    print(x, y)
    
    p1 = [178, 80]
    p2 = [-175, 80]
    p3 = [175, 75]
    p4 = [175, 85]
    
    x, y = compute_intersect(p1, p2, p3, p4)
    print(x, y)
    
    p1 = [175, 80]
    p2 = [178, 80]
    p3 = [-175, 75]
    p4 = [-175, 85]
    
    x, y = compute_intersect(p1, p2, p3, p4)
    print(x, y)
    
    p1 = [-6, 80]
    p2 = [-2, 80]
    p3 = [1, 75]
    p4 = [1, 85]
    
    x, y = compute_intersect(p1, p2, p3, p4)
    print(x, y)
    
    p1 = [-2, 80]
    p2 = [2, 80]
    p3 = [1, 75]
    p4 = [3, 85]
    
    x, y = compute_intersect(p1, p2, p3, p4)
    print(x, y)
    
    p1 = [-2, 80]
    p2 = [2, 80]
    p3 = [-1, 75]
    p4 = [1, 85]
    
    x, y = compute_intersect(p1, p2, p3, p4)
    print(x, y)
    
    p1 = [2, 80]
    p2 = [6, 80]
    p3 = [-1, 75]
    p4 = [1, 85]
    
    x, y = compute_intersect(p1, p2, p3, p4)
    print(x, y)
    
    
    
    # Issues
    p1 = [-176.30337283664238, 81.24118988658537]
    p2 = [-171.27774692527836, 79.83460741960796]
    p3 = [-162.3089040725263, 81.75128021717111]
    p4 = [-165.85010072594073, 81.45199842358777]
    
    x, y = compute_intersect(p1, p2, p3, p4)
    print(x, y)
    
    plt.figure()
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r')
    plt.plot([p3[0], p4[0]], [p3[1], p4[1]], 'b')
    plt.plot(x,y, 'k*')
    
    
    p1 = [-182.58819427599371, 80.5361370296789]
    p2 = [-177.09963619225147, 79.22019178131197]
    p3 = [-169.147447907603, 81.12340485637964]
    p4 = [-172.20706080359136, 80.7686254082039]
    
    x, y = compute_intersect(p1, p2, p3, p4)
    print(x, y)
    
    plt.figure()
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r')
    plt.plot([p3[0], p4[0]], [p3[1], p4[1]], 'b')
    plt.plot(x,y, 'k*')
    
    
    p1 = [174.6137696119451, 80.14724349481844]
    p2 = [180.23798763843462, 78.8766586631735]
    p3 = [-175.0395698647564, 80.39055557300085]
    p4 = [-177.65836358098363, 79.99183430825221]
    
    x, y = compute_intersect(p1, p2, p3, p4)
    print(x, y)
    
    plt.figure()
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r')
    plt.plot([p3[0]+360, p4[0]+360], [p3[1], p4[1]], 'b')
    plt.plot(x,y, 'k*')
    
    
    p1 = [172.02763915703207, 79.73772051969485]
    p2 = [177.73580623778582, 78.51175219217555]
    p3 = [-177.65836358098363, 79.99183430825221]
    p4 = [-180.07833681366827, 79.57482465506928]
    
    x, y = compute_intersect(p1, p2, p3, p4)
    print(x, y)
    
    plt.figure()
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r')
    plt.plot([p3[0]+360, p4[0]+360], [p3[1], p4[1]], 'b')
    plt.plot(x,y, 'k*')
    
    
    p1 = [169.63827291736044, 79.30993390709958]
    p2 = [175.3873484708237, 78.12743857419116]
    p3 = [-180.07833681366827, 79.57482465506928]
    p4 = [-182.31483631727, 79.1416333205416]
    
    x, y = compute_intersect(p1, p2, p3, p4)
    print(x, y)
    
    plt.figure()
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r')
    plt.plot([p3[0]+360, p4[0]+360], [p3[1], p4[1]], 'b')
    plt.plot(x,y, 'k*')
    
    
    plt.show()
    
    
    
    
    return


def unit_test_wrs2():
    
    
    start = time.time()
    wrs2_dict, wrs2_lon, wrs2_lat, wrs2_neglon, wrs2_poslon, land_flag = define_wrs2_grid()   
    print('WRS2 grid time', time.time() - start)
    
    lat = -20
    lon = 179.9    
    
    path, row, pathrow_lon, pathrow_lat = compute_wrs2_path_row(lon, lat, wrs2_lon, wrs2_lat, wrs2_neglon, wrs2_poslon)
    
    
    print(path, row)
    print(pathrow_lon, pathrow_lat)
    print('\n')
    
    
    lat = 35.
    lon = -179.2   
    
    path, row, pathrow_lon, pathrow_lat = compute_wrs2_path_row(lon, lat, wrs2_lon, wrs2_lat, wrs2_neglon, wrs2_poslon)
    
    
    print(path, row)
    print(pathrow_lon, pathrow_lat)
    print('\n')
    
    lat = 81.
    lon = 10.    
    
    path, row, pathrow_lon, pathrow_lat = compute_wrs2_path_row(lon, lat, wrs2_lon, wrs2_lat, wrs2_neglon, wrs2_poslon)
    
    
    print(path, row)
    print(pathrow_lon, pathrow_lat)
    print('\n')
    
    lat = -81.
    lon = -21.  
    
    path, row, pathrow_lon, pathrow_lat = compute_wrs2_path_row(lon, lat, wrs2_lon, wrs2_lat, wrs2_neglon, wrs2_poslon)
    
    
    print(path, row)
    print(pathrow_lon, pathrow_lat)
    print('\n')
    
    return


def test_basemap():
    
    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,
                  urcrnrlon=180,resolution='c')
    
    
    lon_list = list(np.arange(-180., 180., 1.))
    lat_list = list(np.arange(-89., 89., 1.))
    
    land_flag1 = []
    land_flag2 = []
    plot_lon = []
    plot_lat = []
    for lon in lon_list:
        for lat in lat_list:
            
            
#            x, y = m(lon, lat)
            land_flag = m.is_land(lon, lat)
            
            if land_flag:
                plot_lon.append(lon)
                plot_lat.append(lat)
                
    
    # Generate plot
    plt.figure()
    plt.plot(plot_lon, plot_lat, 'go', ms=1)
            
    m.drawcoastlines()
    m.drawmeridians(np.arange(-180, 180, 45))
    m.drawparallels(np.arange(-90, 90, 45))
    m.drawmapboundary()
    m.drawcountries()
    plt.yticks(np.arange(-90, 91, 45))
    plt.xticks(np.arange(-180, 181, 45))

    
    plt.show()   
    
    
    
    
    return


def plot_fov_vs_swath(h):
    
    
    a = Re + 500.
    
    swath_array = np.arange(20, 100., 0.1)
    fov_array = np.zeros(swath_array.shape)
    
    for ii in range(len(swath_array)):
        
        swath = swath_array[ii]
        alpha = swath/(2.*Re)
        rho = np.sqrt(Re**2. + a**2. - 2.*Re*a*cos(alpha))
        f = asin((sin(alpha)/rho)*Re)
        fov = 2.*f*180./pi
        
        fov_array[ii] = fov
        
    plt.figure()
    
    
    plt.plot(swath_array, fov_array, 'r.', label='500km')
    
    
    a = Re + 550.
    fov_array = np.zeros(swath_array.shape)
    
    for ii in range(len(swath_array)):
        
        swath = swath_array[ii]
        alpha = swath/(2.*Re)
        rho = np.sqrt(Re**2. + a**2. - 2.*Re*a*cos(alpha))
        f = asin((sin(alpha)/rho)*Re)
        fov = 2.*f*180./pi
        
        fov_array[ii] = fov
        
    plt.plot(swath_array, fov_array, 'b.', label='550km')
    
    a = Re + 600.
    fov_array = np.zeros(swath_array.shape)
    
    for ii in range(len(swath_array)):
        
        swath = swath_array[ii]
        alpha = swath/(2.*Re)
        rho = np.sqrt(Re**2. + a**2. - 2.*Re*a*cos(alpha))
        f = asin((sin(alpha)/rho)*Re)
        fov = 2.*f*180./pi
        
        fov_array[ii] = fov
        
    plt.plot(swath_array, fov_array, 'g.', label='600km')
    
    a = Re + 650.
    fov_array = np.zeros(swath_array.shape)
    
    for ii in range(len(swath_array)):
        
        swath = swath_array[ii]
        alpha = swath/(2.*Re)
        rho = np.sqrt(Re**2. + a**2. - 2.*Re*a*cos(alpha))
        f = asin((sin(alpha)/rho)*Re)
        fov = 2.*f*180./pi
        
        fov_array[ii] = fov
        
    plt.plot(swath_array, fov_array, 'k.', label='650km')
    
    
    
    
    plt.xlabel('Swath [km]')
    plt.ylabel('FOV [deg]')
    plt.legend()
    
    plt.show()
    
    
    return



if __name__ == '__main__':
    
    plt.close('all')
    
#    tle_print()
    
#    tle_analysis()
    
#    retrieve_and_prop_TLE()
    
#    validate_frames_and_prop()
    
    plot_fov_vs_swath(500.)
    
#    ending_list = ['2020_06', '2020_07', '2020_08', '2020_09', '2020_10', '2020_11',
#                   '2020_12', '2021_01', '2021_02', '2021_03', '2021_04', '2021_05']
#    
#    
#    print(len(ending_list))
#    
#    
#    for ending in ending_list:        
#   
#        tle_file = os.path.join('..//data/tle_propagation_small_swath_' + ending + '.pkl')
#        frame_file = os.path.join('..//data/frame_rotations_' + ending + '.pkl')
#        path_file = os.path.join('..//data/path_data_small_swath_' + ending + '.pkl')
#        compute_paths(tle_file, frame_file, path_file)
        
#        tle_file = os.path.join('..//data/tle_propagation_sunsynch_' + ending + '.pkl')
#        frame_file = os.path.join('..//data/frame_rotations_' + ending + '.pkl')
#        path_file = os.path.join('..//data/path_data_sunsynch_' + ending + '.pkl')
#        compute_paths(tle_file, frame_file, path_file)
    
    
#    landsat_sentinel_coincidence()
    
#    triple_coincidence()
    
#    plot_coincident_data()
    
#    define_wrs2_grid()
    
#    unit_test_wrs2()
    
#    test_basemap()
    
#    
#    plot_groundtrack()
    
#    compute_coverage()
    
#    generate_tle_dict()
    
    
        
#    generate_prop_TLE_files()
    

#    unit_test_recurrence()

#    unit_test_intersect()
    
#    UTC_start = datetime(2020, 7, 1, 0, 0, 0)
#    UTC_stop = datetime(2020, 8, 1, 0, 0, 0)    
#    
#    fname = os.path.join('..//data/frame_rotations_2020_07.pkl')
#    compute_frame_rotations(UTC_start, UTC_stop, fname)
#
#    fname = os.path.join('..//data/tle_propagation_primary_targets_2020_07.pkl')
#    retrieve_and_prop_TLE(UTC_start, UTC_stop, fname)
#    
#    UTC_start = datetime(2020, 8, 1, 0, 0, 0)
#    UTC_stop = datetime(2020, 9, 1, 0, 0, 0)    
#    
#    fname = os.path.join('..//data/frame_rotations_2020_08.pkl')
#    compute_frame_rotations(UTC_start, UTC_stop, fname)
#
#    fname = os.path.join('..//data/tle_propagation_primary_targets_2020_08.pkl')
#    retrieve_and_prop_TLE(UTC_start, UTC_stop, fname)
#    
#    UTC_start = datetime(2020, 9, 1, 0, 0, 0)
#    UTC_stop = datetime(2020, 10, 1, 0, 0, 0)    
#    
#    fname = os.path.join('..//data/frame_rotations_2020_09.pkl')
#    compute_frame_rotations(UTC_start, UTC_stop, fname)
#
#    fname = os.path.join('..//data/tle_propagation_primary_targets_2020_09.pkl')
#    retrieve_and_prop_TLE(UTC_start, UTC_stop, fname)
#    
#    UTC_start = datetime(2020, 10, 1, 0, 0, 0)
#    UTC_stop = datetime(2020, 11, 1, 0, 0, 0)    
#    
#    fname = os.path.join('..//data/frame_rotations_2020_10.pkl')
#    compute_frame_rotations(UTC_start, UTC_stop, fname)
#
#    fname = os.path.join('..//data/tle_propagation_primary_targets_2020_10.pkl')
#    retrieve_and_prop_TLE(UTC_start, UTC_stop, fname)
#    
#    UTC_start = datetime(2020, 11, 1, 0, 0, 0)
#    UTC_stop = datetime(2020, 12, 1, 0, 0, 0)    
#    
#    fname = os.path.join('..//data/frame_rotations_2020_11.pkl')
#    compute_frame_rotations(UTC_start, UTC_stop, fname)
#
#    fname = os.path.join('..//data/tle_propagation_primary_targets_2020_11.pkl')
#    retrieve_and_prop_TLE(UTC_start, UTC_stop, fname)
#    
#    UTC_start = datetime(2020, 12, 1, 0, 0, 0)
#    UTC_stop = datetime(2021, 1, 1, 0, 0, 0)    
#    
#    fname = os.path.join('..//data/frame_rotations_2020_12.pkl')
#    compute_frame_rotations(UTC_start, UTC_stop, fname)
#
#    fname = os.path.join('..//data/tle_propagation_primary_targets_2020_12.pkl')
#    retrieve_and_prop_TLE(UTC_start, UTC_stop, fname)
#    
#    UTC_start = datetime(2021, 1, 1, 0, 0, 0)
#    UTC_stop = datetime(2021, 2, 1, 0, 0, 0)    
#    
#    fname = os.path.join('..//data/frame_rotations_2021_01.pkl')
#    compute_frame_rotations(UTC_start, UTC_stop, fname)
#
#    fname = os.path.join('..//data/tle_propagation_primary_targets_2021_01.pkl')
#    retrieve_and_prop_TLE(UTC_start, UTC_stop, fname)
#    
#    UTC_start = datetime(2021, 2, 1, 0, 0, 0)
#    UTC_stop = datetime(2021, 3, 1, 0, 0, 0)    
#    
#    fname = os.path.join('..//data/frame_rotations_2021_02.pkl')
#    compute_frame_rotations(UTC_start, UTC_stop, fname)
#
#    fname = os.path.join('..//data/tle_propagation_primary_targets_2021_02.pkl')
#    retrieve_and_prop_TLE(UTC_start, UTC_stop, fname)
#    
#    UTC_start = datetime(2021, 3, 1, 0, 0, 0)
#    UTC_stop = datetime(2021, 4, 1, 0, 0, 0)    
#    
#    fname = os.path.join('..//data/frame_rotations_2021_03.pkl')
#    compute_frame_rotations(UTC_start, UTC_stop, fname)
#
#    fname = os.path.join('..//data/tle_propagation_primary_targets_2021_03.pkl')
#    retrieve_and_prop_TLE(UTC_start, UTC_stop, fname)
#    
#    UTC_start = datetime(2021, 4, 1, 0, 0, 0)
#    UTC_stop = datetime(2021, 5, 1, 0, 0, 0)    
#    
#    fname = os.path.join('..//data/frame_rotations_2021_04.pkl')
#    compute_frame_rotations(UTC_start, UTC_stop, fname)
#
#    fname = os.path.join('..//data/tle_propagation_primary_targets_2021_04.pkl')
#    retrieve_and_prop_TLE(UTC_start, UTC_stop, fname)
#    
#    UTC_start = datetime(2021, 5, 1, 0, 0, 0)
#    UTC_stop = datetime(2021, 6, 1, 0, 0, 0)    
#    
#    fname = os.path.join('..//data/frame_rotations_2021_05.pkl')
#    compute_frame_rotations(UTC_start, UTC_stop, fname)
#
#    fname = os.path.join('..//data/tle_propagation_primary_targets_2021_05.pkl')
#    retrieve_and_prop_TLE(UTC_start, UTC_stop, fname)
    
    
    
    
    
    
   
#    test_frame_rotations()
    
#    recurrence_analysis()
    
#    plot_swath_vs_altitude()
    
    
#    hmin = 400.
#    hmax = 660.
#    fov = 25.*pi/180.
#    recurrent_df = generate_candidate_recurrent_triples(hmin, hmax, fov)
#    
#    print(recurrent_df)
#    recurrent_df.to_csv('SSO_20deg_fov_recurrence_params_v6.csv')
#    
#    
#    
#    vo = 15
#    Dto = -7
#    Cto = 16.
#    UTC = datetime(2021, 4, 15, 0, 0, 0)
#    elem = compute_orbit_parameters(vo, Dto, Cto, UTC)
    
#    print(elem)
    
    
#    a = Re + 880.55
#    e = 1e-4
#    i = 28.
#    To, Ta, Td = compute_orbit_periods(a, e, i)
#    print(To, Ta, Td)
    
    
    
    
#    nodal_period_to_sunsynch_orbit(0, 0)

    
#    compute_Eto(15, -7, 16)
    
#    print(is_coprime(13,17))
    
    
#    # Swath parameters
#    a_landsat = Re + 705
#    a_sentinel = Re + 786.
#    fov_landsat = 15.*pi/180.
#    fov_sentinel = 21.*pi/180.
#    
#    swath_rad_landsat, swath_km_landsat = compute_groundswath(a_landsat, fov_landsat)
#    swath_rad_sentinel, swath_km_sentinel = compute_groundswath(a_sentinel, fov_sentinel)
#    
#    print(fov_landsat*705)
#    print(fov_sentinel*785)
#    
#    print(swath_rad_landsat*180/pi, swath_km_landsat)
#    print(swath_rad_sentinel*180/pi, swath_km_sentinel)
#    
#    
#    delta, delta_rev, delta_day = compute_recurrence_grid_parameters(14, 3, 10) 
#    print(delta*180/pi, delta_rev*180/pi, delta_day*180/pi)
    
    
    





#def nodal_period_to_sunsynch_orbit(Td, e, i, R=Re, GM=GME, J2=J2E):
#    '''
#    
#    
#    '''
#    
##    i = 82.56 * pi/180.
##    Td = 109.421425*60
##    e = 0.
#    
#    # First pass, guess To = Td
#    To = Td
#    
#    # Compute SMA 
#    no = 2.*pi/To
#    ao = meanmot2sma(no, GM) 
#    
#    print('Ho', ao - Re)
#    
#    a = float(ao)
#    diff = 1.
#    count = 0.
#    while diff > 1e-2:
#        
#        count += 1
#        if count > 10:
#            break
#    
#    #    # Compute sunsynchronous inclination 
#    #    i = sunsynch_inclination(ao, e)         # deg
#    #    i = i * pi/180.                         # rad
#        
#        # Compute perturbation effects from J2
#        dn = (3./(4.*(1-e**2.)**(3./2.))) * no * J2 * (R/a)**2. * (3.*cos(i)**2. - 1)
#        dw = (3./(4.*(1-e**2.)**(2.))) * no * J2 * (R/a)**2. * (5.*cos(i)**2. - 1)
#        
#        # Update SMA
#        da = (2./3.) * ((dw + dn)/no) * a
#        a = ao + da
#        
#        # Update To and no
#        no = np.sqrt(GM/a**3.)
#        To = 2.*pi/no
#        
#        # Compute Td and compare against true
#        Td_check = ((1. - dn/no)/(1. + dw/no)) * To
#        diff = abs(Td_check - Td)
#        
#        print('\n\n')
#        print(Td, Td_check)
#        print(To)
#        
#        print(da)
#        print(a)
#        print('alt', a - Re)
#
#    
#    return  
#    

#def compute_paths(tle_file, frame_file, path_file, dayside_flag='desc', R=Re, GM=GME):
#    
#    start = time.time()
#    
#    # Load TLE and Frame Rotation Data
#    pklFile = open(tle_file, 'rb')
#    data = pickle.load(pklFile)
#    output_state = data[0]
#    pklFile.close()
#    
#    pklFile = open(frame_file, 'rb')
#    data = pickle.load(pklFile)
#    UTC_list_frame = data[0]
#    GCRF_TEME_list = data[1]
#    ITRF_GCRF_list = data[2]
#    pklFile.close()
#    
#    # Compare UTC times as error check
#    obj_id_list = list(output_state.keys())
#    UTC_list_obj = output_state[obj_id_list[0]]['UTC']
#    
#    if (UTC_list_obj[0] != UTC_list_frame[0]) or (len(UTC_list_obj) != len(UTC_list_frame)):
#        print('Error on UTC Times!!')
#        print('object', UTC_list_obj[0])
#        print('frame', UTC_list_frame[0])
#        mistake
#        
#    # Generate fov dictionary
#    fov_dict = generate_fov_dict()
#        
#    # Loop over objects    
#    path_dict = {}
#    for obj_id in obj_id_list:
#        
#        print(obj_id)
#        print('Compute Paths Time: ', time.time() - start)
#        
#        # Initialize 
#        path_dict[obj_id] = {}
#        path_ind = 0
#        path_dict[obj_id][path_ind] = {}
#        path_dict[obj_id][path_ind]['UTC'] = []
#        path_dict[obj_id][path_ind]['UTC_zerolat'] = 0.
#        path_dict[obj_id][path_ind]['lon_list'] = []
#        path_dict[obj_id][path_ind]['lat_list'] = []
#        path_dict[obj_id][path_ind]['lon1_list'] = []
#        path_dict[obj_id][path_ind]['lat1_list'] = []
#        path_dict[obj_id][path_ind]['lon2_list'] = []
#        path_dict[obj_id][path_ind]['lat2_list'] = []
#        min_lat = 100.       
#        
#        UTC_list_obj = output_state[obj_id]['UTC']
#        UTC_prev = UTC_list_obj[0]
#        dt = (output_state[obj_id]['UTC'][1] - output_state[obj_id]['UTC'][0]).total_seconds()
#        
#        # Retrive field of view in radians
#        fov = fov_dict[obj_id]
#        
#        # Loop over times and rotate coordinates        
#        for ii in range(len(UTC_list_obj)):
##        for ii in range(0, 6*360):
#            
#            # Retrieve data and rotate frames
#            UTC = output_state[obj_id]['UTC'][ii]
#            r_TEME = output_state[obj_id]['r_TEME'][ii]
#            v_TEME = output_state[obj_id]['v_TEME'][ii]
#            
#            GCRF_TEME = GCRF_TEME_list[ii]
#            ITRF_GCRF = ITRF_GCRF_list[ii]
#            
#            r_GCRF = np.dot(GCRF_TEME, r_TEME)
#            v_GCRF = np.dot(GCRF_TEME, v_TEME)
#            r_ITRF = np.dot(ITRF_GCRF, r_GCRF)
#            
#            # Compute orbit parameters
#            cart = np.concatenate((r_GCRF, v_GCRF), axis=0)
#            kep = cart2kep(cart, GM)            
#            
#            a = float(kep[0])
#            w = float(kep[4])
#            theta = float(kep[5])
#            true_long = (w + theta) % 360.
#            
#            # Check dayside conditions
#            if dayside_flag == 'desc':
#                
#                # Only compute path if we are approaching descending node
#                if true_long < 90. or true_long > 270.:
#                    continue
#                
#            
#            # Check if this is a new path and initialize output
#            if (UTC - UTC_prev).total_seconds() > dt:
#                
#                # Check if any data has been saved in Path 0 yet
#                # If so, increment and initialize the next path
#                if len(path_dict[obj_id][0]['UTC']) > 0:
#                    path_ind += 1
#                    
#                    path_dict[obj_id][path_ind] = {}
#                    path_dict[obj_id][path_ind]['UTC'] = []
#                    path_dict[obj_id][path_ind]['UTC_zerolat'] = 0.
#                    path_dict[obj_id][path_ind]['lon_list'] = []
#                    path_dict[obj_id][path_ind]['lat_list'] = []
#                    path_dict[obj_id][path_ind]['lon1_list'] = []
#                    path_dict[obj_id][path_ind]['lat1_list'] = []
#                    path_dict[obj_id][path_ind]['lon2_list'] = []
#                    path_dict[obj_id][path_ind]['lat2_list'] = []
#                    min_lat = 100.
#                    
#            
#            # Compute half-swath angle and lat/long with offsets
#            f = (fov/2.)
#            zeta = asin(a*sin(f)/R)
#            alpha = zeta - f
#            
#            # RIC frame vectors to add to chief orbit
#            z = alpha * a
#            rho1 = np.array([[0.], [0.], [z]])
#            rho2 = np.array([[0.], [0.], [-z]])
#            
#            # Compute ECI states for z component in RIC frame
#            r_GCRF_1 = ric2eci(r_GCRF, v_GCRF, rho1) + r_GCRF
#            r_GCRF_2 = ric2eci(r_GCRF, v_GCRF, rho2) + r_GCRF
#            
#            # Compute ECEF coordinates
#            r_ITRF_1 = np.dot(ITRF_GCRF, r_GCRF_1)
#            r_ITRF_2 = np.dot(ITRF_GCRF, r_GCRF_2)
#            
#            # Compute latitude, longitude
#            lat, lon, ht = ecef2latlonht(r_ITRF)
#            lat1, lon1, ht = ecef2latlonht(r_ITRF_1)
#            lat2, lon2, ht = ecef2latlonht(r_ITRF_2)
#            
#            # Check for zero latitude
#            if abs(lat) < min_lat:
#                path_dict[obj_id][path_ind]['UTC_zerolat'] = UTC
#                min_lat = abs(lat)
#            
#            # Store data
#            path_dict[obj_id][path_ind]['UTC'].append(UTC)
#            path_dict[obj_id][path_ind]['lon_list'].append(lon)
#            path_dict[obj_id][path_ind]['lat_list'].append(lat)
#            path_dict[obj_id][path_ind]['lon1_list'].append(lon1)
#            path_dict[obj_id][path_ind]['lat1_list'].append(lat1)
#            path_dict[obj_id][path_ind]['lon2_list'].append(lon2)
#            path_dict[obj_id][path_ind]['lat2_list'].append(lat2)
#            
#            # Update UTC_prev
#            UTC_prev = copy.copy(UTC)
#            
#
#    
##    lon_list = path_dict[obj_id][1]['lon_list']
##    lat_list = path_dict[obj_id][1]['lat_list']
##    lon1_list = path_dict[obj_id][1]['lon1_list']
##    lat1_list = path_dict[obj_id][1]['lat1_list']
##    lon2_list = path_dict[obj_id][1]['lon2_list']
##    lat2_list = path_dict[obj_id][1]['lat2_list']
##    
##      
##            
##    # Generate plot
##    plt.figure()
##    plt.plot(lon_list, lat_list, 'bo', ms=1)
##    plt.plot(lon1_list, lat1_list, 'ro', ms=1)
##    plt.plot(lon2_list, lat2_list, 'go', ms=1)
##            
##    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,
##                urcrnrlon=180,resolution='c')
##    m.drawcoastlines()
##    m.drawmeridians(np.arange(-180, 180, 45))
##    m.drawparallels(np.arange(-90, 90, 45))
##    m.drawmapboundary()
##    m.drawcountries()
##    plt.yticks(np.arange(-90, 91, 45))
##    plt.xticks(np.arange(-180, 181, 45))
##
##    
##    plt.show()     
#            
#    
#    print('Compute Paths Time: ', time.time() - start)
#    
#    # Save output    
#    pklFile = open( path_file, 'wb' )
#    pickle.dump( [path_dict], pklFile, -1 )
#    pklFile.close()
#    
#    
#    return
    

# Compute longitude diffs
#            # Case 1: Path1 is to the east of Path2
#            if lon_diff_deg > 0.:
#                
#                # Compute intersection (linear interpolation)
#                p1 = [path1_lonW[ii], path1_latW[ii]]
#                p2 = [path1_lonE[ii], path1_latE[ii]]
#                p3 = [path2_lonE[jj], path2_latE[jj]]
#                
#                try:
#                    p4 = [path2_lonE[jj+1], path2_latE[jj+1]]
#                except:
#                    p4 = [path2_lonE[jj+1], path2_latE[jj+1]]
#                
#                lon_intersect, lat_intersect = compute_intersect(p1, p2, p3, p4)
#                
#                
#                
#                
#                
#                # Check for no overlap - Case 1A
##                diff1 = compute_anglediff_deg(path2_lonE[jj], path1_lonW[ii])
#                diff1 = compute_anglediff_deg(lon_intersect, path1_lonW[ii])
#                if diff1 <= 0.:
#                    coincident_dict[coincident_ind]['overlap_km'].append(0.)
#                    coincident_dict[coincident_ind]['overlap_frac'].append(0.)
#                    coincident_dict[coincident_ind]['lonE'].append(0.)
#                    coincident_dict[coincident_ind]['latE'].append(0.)
#                    coincident_dict[coincident_ind]['lonW'].append(0.)
#                    coincident_dict[coincident_ind]['latW'].append(0.)
#                    
#                else:
#                    
#                    # Compute current swath coverage
#                    latlon2dist(lat1, lon1, lat2, lon2)
#                    
#                    
#                    
#                    # Check for partial or full overlap
#                    diff2 = compute_anglediff_deg(path2_lonW[jj], path1_lonW[ii])
#                    
#                    # Full overlap (swath1 covers swath2) - Case 1C
#                    if diff2 > 0:
#                        coincident_dict[coincident_ind]['overlap_km'].append(swath2_deg*pi/180.*Re)
#                        coincident_dict[coincident_ind]['overlap_frac'].append(1.)
#                        coincident_dict[coincident_ind]['lonE'].append(path2_lonE[jj])
#                        coincident_dict[coincident_ind]['lonW'].append(path2_lonW[jj])
#
#                    
#                    else:
#                        
#                        # Check for partial or full overlap
#                        diff3 = compute_anglediff_deg(path2_lonE[jj], path1_lonE[ii])
#                        
#                        # Partial overlap - Case 1B
#                        if diff3 < 1.:
#                            overlap_deg = compute_anglediff_deg(path2_lonE[jj], path1_lonW[ii])
#                            frac = overlap_deg/min_swath
#                            coincident_dict[coincident_ind]['overlap_km'].append(overlap_deg*pi/180.*Re)
#                            coincident_dict[coincident_ind]['overlap_frac'].append(frac)
#                            coincident_dict[coincident_ind]['lonE'].append(path2_lonW[jj])
#                            coincident_dict[coincident_ind]['lonW'].append(path1_lonE[ii])
#                        
#                        # Full overlap (swath2 covers swath1) - Case 1D
#                        else:
#                            
#                            coincident_dict[coincident_ind]['overlap_km'].append(swath1_deg*pi/180.*Re)
#                            coincident_dict[coincident_ind]['overlap_frac'].append(1.)
#                            coincident_dict[coincident_ind]['lonE'].append(path1_lonE[ii])
#                            coincident_dict[coincident_ind]['lonW'].append(path1_lonW[ii])
#                            
#                    
#                
#                
#            # Case 2: Path1 is to the left of Path2
#            elif lon_diff_deg < 0.:
#                
#                # Check for no overlap - Case 2A
#                diff1 = compute_anglediff_deg(path1_lonE[ii], path2_lonW[jj])
#                if diff1 <= 0.:
#                    coincident_dict[coincident_ind]['overlap_km'].append(0.)
#                    coincident_dict[coincident_ind]['overlap_frac'].append(0.)
#                    coincident_dict[coincident_ind]['lonE'].append(0.)
#                    coincident_dict[coincident_ind]['lonW'].append(0.)
#                    
#                else:
#                    
#                    # Compute current swath coverage
#                    swath1_deg = compute_anglediff_deg(path1_lonE[ii], path1_lonW[ii])
#                    swath2_deg = compute_anglediff_deg(path2_lonE[jj], path2_lonW[jj])
#                    min_swath = min(swath1_deg, swath2_deg)
#                    
#                    # Check for partial or full overlap
#                    diff2 = compute_anglediff_deg(path1_lonW[ii], path2_lonW[jj])
#                    
#                    # Full overlap (swath2 covers swath1) - Case 2C
#                    if diff2 > 0:
#                        coincident_dict[coincident_ind]['overlap_km'].append(swath1_deg*pi/180.*Re)
#                        coincident_dict[coincident_ind]['overlap_frac'].append(1.)
#                        coincident_dict[coincident_ind]['lonE'].append(path1_lonE[ii])
#                        coincident_dict[coincident_ind]['lonW'].append(path1_lonW[ii])
#
#                    
#                    else:
#                        
#                        # Check for partial or full overlap
#                        diff3 = compute_anglediff_deg(path1_lonE[ii], path2_lonE[jj])
#                        
#                        # Partial overlap - Case 2B
#                        if diff3 < 1.:
#                            overlap_deg = compute_anglediff_deg(path1_lonE[ii], path2_lonW[jj])
#                            frac = overlap_deg/min_swath
#                            coincident_dict[coincident_ind]['overlap_km'].append(overlap_deg*pi/180.*Re)
#                            coincident_dict[coincident_ind]['overlap_frac'].append(frac)
#                            coincident_dict[coincident_ind]['lonE'].append(path1_lonW[ii])
#                            coincident_dict[coincident_ind]['lonW'].append(path2_lonE[jj])
#                        
#                        # Full overlap (swath1 covers swath2) - Case 2D
#                        else:
#                            
#                            coincident_dict[coincident_ind]['overlap_km'].append(swath2_deg*pi/180.*Re)
#                            coincident_dict[coincident_ind]['overlap_frac'].append(1.)
#                            coincident_dict[coincident_ind]['lonE'].append(path2_lonE[jj])
#                            coincident_dict[coincident_ind]['lonW'].append(path2_lonW[jj])
#                
#                
#            
#            # Case 3: Path1 and Path2 cross Equator at same point
#            else:
#                print('Error - Lon1 == Lon2')
#                print(eq_lon1)
#                print(eq_lon2)
#                mistake



