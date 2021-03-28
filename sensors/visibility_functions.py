import numpy as np
from math import pi, cos, sin, acos, asin, log10
import os
import sys
import csv
import time
from datetime import datetime, timedelta
import getpass

sys.path.append('../')

from skyfield.constants import ERAD
from skyfield.api import Topos, EarthSatellite, Loader

from sensors.sensors import define_sensors
from utilities.tle_functions import get_spacetrack_tle_data
from utilities.tle_functions import find_closest_tle_epoch
from utilities.tle_functions import propagate_TLE

from utilities.eop_functions import get_eop_data
from utilities.eop_functions import get_celestrak_eop_alldata
from utilities.coordinate_systems import latlonht2ecef
from utilities.coordinate_systems import gcrf2itrf
from utilities.coordinate_systems import itrf2gcrf
from utilities.time_systems import utcdt2ttjd
from utilities.time_systems import jd2cent
from utilities.constants import Re, AU_km
from sensors.measurements import compute_measurement
from sensors.measurements import ecef2azelrange
from sensors.measurements import ecef2azelrange_rad


def define_RSOs(obj_id_list, UTC_list, tle_dict={}, offline_flag=False,
                source='spacetrack', username='', password=''):
    '''
    This function generates the resident space object (RSO) dictionary by 
    retrieving data about RSOs including recent position/velocity states
    and phyical parameters relating to size and shape from websites
    or databases as available.
    
    Parameters
    ------
    obj_id_list : list
        object NORAD IDs (int)
    tle_dict : dictionary, optional
        Two Line Element information, indexed by object ID (default = empty)
        If none provided, script will retrieve from source
    source : string, optional
        designates source of tle_dict information if empty
        (default = spacetrack)
    
    Returns
    ------
    rso_dict : dictionary
        RSO state and parameters indexed by object NORAD ID
    '''
    
    # Load TLE Data and propagate to times of interest
    # Include options here to import from space-track, celestrak, text file,
    # other URL, graph database, ...
    
    if len(tle_dict) == 0:
        
        # Download from space-track.org
        if source == 'spacetrack':            
            
            rso_dict = propagate_TLE(obj_id_list, UTC_list, tle_dict, offline_flag=False,
                  username='', password='')
            
        # Retrieve from graph database
        if source == 'database':
            tle_dict = {}
#            tle_dict = get_database_tle_data(obj_id_list)

    # Initialize object size
    # Include options here for RCS from SATCAT, graph database, ...  
    
    # Retrieve from database
    if source == 'database':
        rso_dict = {}
        
#        rso_dict = get_database_object_params(rso_dict)
    
    # Use default values
    else:
        for obj_id in obj_id_list:
            
            # Dummy value for all satellites    
            rso_dict[obj_id]['radius_m'] = 1.
            rso_dict[obj_id]['albedo'] = 0.1
            rso_dict[obj_id]['listen_flag'] = True
            rso_dict[obj_id]['frequency_hz'] = 900e6
            rso_dict[obj_id]['laser_lim'] = 10.
    
    return rso_dict


def get_database_object_params(rso_dict):
    '''
    This function retrieve object parameters such as radius and albedo from
    the database.
    
    '''
    
    # List of objects to retrieve data for
    obj_id_list = sorted(rso_dict.keys())
    
    # Retrieve data for each object in list
    
    # Add data to rso_dict
    
    
    return rso_dict


def compute_visible_passes(UTC_list, obj_id_list, sensor_id_list, tle_dict={},
                           offline_flag=False, source='spacetrack',
                           username='', password=''):
    '''
    This function computes the visible passes for a given list of 
    resident space objects (RSOs) from one or more sensors. Output includes
    the start and stop times of each pass, as well as the time of closest
    approach (TCA) and time of maximum elevation (TME).  
    
    Parameters
    ------
    UTC_list : list
        datetime object times to compute visibility conditions 
    obj_id_list : list
        object NORAD IDs (int)
    sensor_id_list : list
        sensor IDs (str)
    tle_dict : dictionary, optional
        Two Line Element information, indexed by object ID (default = empty)
        If none provided, script will retrieve from source
    source : string, optional
        designates source of tle_dict information if empty
        (default = spacetrack)
        
    
    Returns
    ------
    vis_dict : dictionary
        contains sorted lists of pass start and stop times for each object
        and sensor, as well as TCA and TME times, and maximum elevation angle
        and minimum range to the RSO during the pass

    '''
    
    # Generate resident space object dictionary
    rso_dict = define_RSOs(obj_id_list, UTC_list, tle_dict, offline_flag,
                           source, username, password)
    
    # Load sensor data
    # Include options here to load from file, URL, graph database, ...
    
    # Load from database
    if source == 'database':        
        sensor_dict = get_database_sensor_data(sensor_id_list)
        
    else:        
        sensor_dict = define_sensors(sensor_id_list)
        
    # Compute sensor location in ITRF
    for sensor_id in sensor_dict:
        lat, lon, ht = sensor_dict[sensor_id]['geodetic_latlonht']
        sensor_dict[sensor_id]['r_ITRF'] = latlonht2ecef(lat, lon, ht)
        
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = get_celestrak_eop_alldata(offline_flag)

    # Retrieve sun and moon positions for full timespan
    sun_gcrf_list = []
    moon_gcrf_list = []
    for UTC in UTC_list:
        EOP_data = get_eop_data(eop_alldata, UTC)
        TT_JD = utcdt2ttjd(UTC, EOP_data['TAI_UTC'])
        TT_cent = jd2cent(TT_JD)
        sun_eci_geom, sun_eci_app = compute_sun_coords(TT_cent)
        moon_eci_geom, moon_eci_app = compute_moon_coords(TT_cent)
        sun_gcrf_list.append(sun_eci_app)
        moon_gcrf_list.append(moon_eci_app)

    # Initialize output
    start_list_all = []
    stop_list_all = []
    TCA_list_all = []
    TME_list_all = []
    rg_min_list_all = []
    el_max_list_all = []
    obj_id_list_all = []
    sensor_id_list_all = []
    
    # Loop over RSOs
    for obj_id in rso_dict:
        rso = rso_dict[obj_id]
        rso_gcrf_list = rso['r_GCRF']
        rso_itrf_list = rso['r_ITRF']
        
        # Retrieve object size, albedo
        radius_km = rso['radius_m']/1000.
        albedo = rso['albedo']

        # Loop over sensors        
        for sensor_id in sensor_dict:
            sensor = sensor_dict[sensor_id]
            sensor_itrf = sensor['r_ITRF']
            
            # Compute topocentric RSO position
            # For earth satellites, calling observe and apparent is costly
            # and unnecessary except for meter level accuracy
            difference = [r_itrf - sensor_itrf for r_itrf in rso_itrf_list]
            diff_enu = [ecef2enu(diff_ecef) for diff_ecef in difference]
            rg_list = [np.linalg.norm(enu) for enu in diff_enu]
            
            el_array, az_array, rg_array = rso_topo.altaz()
            
            # Compute topocentric sun position
            # Need both sun and sensor positions referenced from solar
            # system barycenter
            sensor_ssb = earth + sensor['statTopos']
            sun_topo = sensor_ssb.at(UTC_array).observe(sun).apparent()
            sun_el_array, sun_az_array, sun_rg_array = sun_topo.altaz()            
                        
            # Constraint parameters
            el_lim = sensor['el_lim']
            az_lim = sensor['az_lim']
            rg_lim = sensor['rg_lim']
            
            # Find indices where az/el/range constraints are met
            el_inds0 = np.where(el_array.radians > el_lim[0])[0]
            el_inds1 = np.where(el_array.radians < el_lim[1])[0]
            az_inds0 = np.where(az_array.radians > az_lim[0])[0]
            az_inds1 = np.where(az_array.radians < az_lim[1])[0]
            rg_inds0 = np.where(rg_array.km > rg_lim[0])[0]
            rg_inds1 = np.where(rg_array.km < rg_lim[1])[0]            
            
            # Find all common elements to create candidate visible index list
            # based on position constraints
            common_el = set(el_inds0).intersection(set(el_inds1))
            common_az = set(az_inds0).intersection(set(az_inds1))
            common_rg = set(rg_inds0).intersection(set(rg_inds1))
            common1 = common_el.intersection(common_az)
            common_pos = common1.intersection(common_rg)
            
#            print(sensor_id)
#            print(common_pos)
            
            # Sunlit/station dark constraint
            if 'sun_elmask' in sensor:
                sun_elmask = sensor['sun_elmask']
                
                # Check sun constraint (ensures station is dark if needed)
                sun_el_inds = np.where(sun_el_array.radians < sun_elmask)[0]                
                common_inds = list(common_pos.intersection(set(sun_el_inds)))
                
#                print(sensor_id)
#                print('sun_elmask', sun_elmask)
#                print(common_inds)
                
            # Laser constraints
            if 'laser_output' in sensor and rso['laser_lim'] > 0.:
                laser_lim = rso['laser_lim']
                laser_output = sensor['laser_output']
                if laser_output < laser_lim:
                    common_inds = list(common_pos)
                    
#                print(sensor_id)
#                print('laser lim', laser_lim)
#                print('laser output', laser_output)
                
            if 'laser_output' in sensor and rso['laser_lim'] <= 0.:
                common_inds = []
                
#                print(sensor_id)
#                print('laser lim', rso['laser_lim'])
                
            # Radio constraints
            if 'freq_lim' in sensor and rso['listen_flag']:
                frequency = rso['frequency_hz']
                freq_lim = sensor['freq_lim']
                if frequency > freq_lim[0] and frequency < freq_lim[1]:
                    common_inds = list(common_pos)
                    
#                print(sensor_id)
#                print('freq lim', freq_lim)
#                print('freq', frequency)
            
            if 'freq_lim' in sensor and not rso['listen_flag']:
                common_inds = []
                
#                print(sensor_id)
#                print('freq lim', sensor['freq_lim'])
#                print(rso['listen_flag'])
            
            # Initialze visibility array for this sensor and object
#            print(common_inds)
            vis_array = np.zeros(rso_gcrf_array.shape[1],)
            vis_array[common_inds] = True
            
            # For remaining indices compute angles and visibility conditions
            ecclipse_inds = []
            mapp_inds = []
            for ii in common_inds:
                rso_gcrf = rso_gcrf_array[:,ii]
                sensor_gcrf = sensor_gcrf_array[:,ii]
                sun_gcrf = sun_gcrf_array[:,ii]
                moon_gcrf = moon_gcrf_array[:,ii]
                rg_km = rg_array.km[ii]
                
                # Compute angles
                phase_angle, sun_angle, moon_angle = \
                    compute_angles(rso_gcrf, sun_gcrf, moon_gcrf, sensor_gcrf)
            
                # Check for eclipse - if sun angle is less than half cone angle
                # the sun is behind the earth
                # First check valid orbit - radius greater than Earth radius
                r = np.linalg.norm(rso_gcrf)
                if r < Re:
                    vis_array[ii] = False
                else:
                    half_cone = asin(Re/r)
                    if sun_angle < half_cone:
                        vis_array[ii] = False
                        ecclipse_inds.append(ii)

                # Check too close to moon
                if 'moon_angle_lim' in sensor:
                    moon_angle_lim = sensor['moon_angle_lim']
                    if moon_angle < moon_angle_lim:
                        vis_array[ii] = False
                                
                # Check apparent magnitude
                # Optional input for albedo could be retrieved for each object
                # from catalog
                if 'mapp_lim' in sensor:
                    mapp_lim = sensor['mapp_lim']
                    mapp = compute_mapp(phase_angle, rg_km, radius_km, albedo)
                    if mapp > mapp_lim:
                        vis_array[ii] = False
                        mapp_inds.append(ii)
            
            vis_inds = np.where(vis_array)[0]
            UTC_vis = UTC_array[vis_inds]
            rg_vis = rg_array.km[vis_inds]
            el_vis = el_array.radians[vis_inds]
            
#            print('ecclipse', ecclipse_inds)
#            print('mapp', mapp_inds)
            
            # Compute pass start and stop times
            start_list, stop_list, TCA_list, TME_list, rg_min_list, el_max_list = \
                compute_pass(UTC_vis, rg_vis, el_vis, sensor)

            # Store output
            npass = len(start_list)
            start_list_all.extend(start_list)
            stop_list_all.extend(stop_list)
            TCA_list_all.extend(TCA_list)
            TME_list_all.extend(TME_list)
            rg_min_list_all.extend(rg_min_list)
            el_max_list_all.extend(el_max_list)
            obj_id_list_all.extend([obj_id]*npass)
            sensor_id_list_all.extend([sensor_id]*npass)

    # Sort results
    start_list_JD = [ti.tdb for ti in start_list_all]
    sort_ind = np.argsort(start_list_JD)
    sorted_start = [start_list_all[ii] for ii in sort_ind]
    sorted_stop = [stop_list_all[ii] for ii in sort_ind]
    sorted_TCA = [TCA_list_all[ii] for ii in sort_ind]
    sorted_TME = [TME_list_all[ii] for ii in sort_ind]
    sorted_rg_min = [rg_min_list_all[ii] for ii in sort_ind]
    sorted_el_max = [el_max_list_all[ii] for ii in sort_ind]
    sorted_obj_id = [obj_id_list_all[ii] for ii in sort_ind]
    sorted_sensor_id = [sensor_id_list_all[ii] for ii in sort_ind]

    # Final output
    vis_dict = {}
    vis_dict['start_list'] = sorted_start
    vis_dict['stop_list'] = sorted_stop
    vis_dict['TCA_list'] = sorted_TCA
    vis_dict['TME_list'] = sorted_TME
    vis_dict['rg_min_list'] = sorted_rg_min
    vis_dict['el_max_list'] = sorted_el_max
    vis_dict['obj_id_list'] = sorted_obj_id
    vis_dict['sensor_id_list'] = sorted_sensor_id

    return vis_dict


def compute_transit_dict(UTC_window, obj_id_list, site_dict, increment=10.,
                         offline_flag=False, source='spacetrack',
                         username='', password=''):
    '''

    '''
    
    start = time.time()
    
    if len(username) == 0:
        username = input('space-track username: ')
    if len(password) == 0:
        password = getpass.getpass('space-track password: ')
    
    # Generate TLE dictionary
    # Retrieve all TLEs from the given time window, including 3 days before
    # and after the given start and end times
    UTC0 = UTC_window[0]
    UTCf = UTC_window[-1]
    
    # Download from space-track.org
    if source == 'spacetrack':            
        
        UTC_window_more = [UTC0 - timedelta(days=3.), UTCf + timedelta(days=3.)]
        tle_dict, tle_df = get_spacetrack_tle_data(obj_id_list, UTC_window_more,
                                                    username, password)
    
    # Propagate TLEs for all times in actual window, using increment
    window_seconds = (UTCf - UTC0).total_seconds()
    delta_seconds = np.arange(0., window_seconds + 1., increment)
    UTC_list_full = [UTC0 + timedelta(seconds=ti) for ti in delta_seconds]
    
#    print(UTC_list_full)
    
#    print(tle_dict)
    
    
     
    print('\nGet TLE Time: ', time.time()-start)
    
    state_dict = propagate_TLE(obj_id_list, UTC_list_full, tle_dict,
                               offline_flag, username, password)
    
    
    print('nPropagate Time: ', time.time() - start)
    
#    print(state_dict)
    
#    mistake
     
    site_time = time.time()
   
    # Loop over objects
    transit_dict = {}
    for obj_id in obj_id_list:
        
        # Retrieve spacecraft ITRF (ECEF) positions
        ITRF_list = state_dict[obj_id]['r_ITRF']
        
        # Loop over sites
        for site in site_dict:
            
            # Compute ECEF site location
            latlonht = site_dict[site]['geodetic_latlonht']
            site_ecef = latlonht2ecef(latlonht[0], latlonht[1], latlonht[2])
            
            # Loop over times
            az_list = []
            el_list = []
            rg_list = []
            UTC_list = []
            for ii in range(len(UTC_list_full)):
                
                # Current time and position vector in ECEF
                r_ecef = ITRF_list[ii]
                
                # Compute az, el, range
                az, el, rg = ecef2azelrange(r_ecef, site_ecef)
                
                if el > 0.:
                    az_list.append(az)
                    el_list.append(el)
                    rg_list.append(rg)
                    UTC_list.append(UTC_list_full[ii])
                    
                    
            
            # Compile data into transits
            
#            print(obj_id)
#            print(site)
#            print(UTC_list)
#            print(el_list)
#            print(rg_list)
            
            transit_dict = compile_transit_data(transit_dict, site, obj_id,
                                                UTC_list, az_list, el_list,
                                                rg_list, increment)
                
    print('\nSite Time: ', time.time() - site_time)
    
    print('Total Time: ', time.time() - start)

    return transit_dict


def compile_transit_data(transit_dict, site, obj_id, UTC_list, az_list,
                         el_list, rg_list, increment):
    
    
#    print('\n\n compile transit')
#    print(obj_id)
#    print(site)
#    print(transit_dict)
    
    # Number of unique digits in transit IDs
    zfill_count = 10
    
    # Only process if needed
    if len(UTC_list) > 0 :
        
        # Initialze output
        if site not in transit_dict:
            transit_dict[site] = {}
            transit_id = site + '_' + str(1).zfill(zfill_count)
        else:
            transit_list = sorted(transit_dict[site].keys())
            last_id = int(transit_list[-1][-zfill_count:])
            transit_id = site + '_' + str(last_id+1).zfill(zfill_count)
#            print('new id')
#            print(transit_list)
#            print(last_id)
            
            
#        print(transit_id)
        
        
        # Loop over times
        rg_min = 1e12
        el_max = -1.
        ti_prior = UTC_list[0]
        start = UTC_list[0]
        stop = UTC_list[0]
        TCA = UTC_list[0]
        TME = UTC_list[0]
        UTC_transit = []
        az_transit = []
        el_transit = []
        rg_transit = []
        for ii in range(len(UTC_list)):
            
            ti = UTC_list[ii]
            rg_km = rg_list[ii]
            el_deg = el_list[ii]
            
            # If current time is close to previous, pass continues
            if (ti - ti_prior).total_seconds() < (increment + 1.):
    
                # Update pass stop time and ti_prior for next iteration
                stop = ti
                ti_prior = ti
                
                # Check if this is pass time of closest approach (TCA)
                if rg_km < rg_min:
                    TCA = ti
                    rg_min = float(rg_km)
                
                # Check if this is pass time of maximum elevation (TME)
                if el_deg > el_max:
                    TME = ti
                    el_max = float(el_deg)
                    
                # Store current time and measurement values for output
                UTC_transit.append(ti.strftime('%Y-%m-%dT%H:%M:%S'))
                az_transit.append(az_list[ii])
                el_transit.append(el_list[ii])
                rg_transit.append(rg_list[ii])
                
    
            # If current time is far from previous or if we reached
            # the end of UTC list, transit has ended
            if ((ti - ti_prior).total_seconds() >= (increment + 1.) or ii == (len(UTC_list)-1)):
    
                if ii == (len(UTC_list)-1):
                    stop = ti
                    UTC_transit.append(ti.strftime('%Y-%m-%dT%H:%M:%S'))
                    az_transit.append(az_list[ii])
                    el_transit.append(el_list[ii])
                    rg_transit.append(rg_list[ii])
                    
                duration = (stop - start).total_seconds()
                
                # Store output
                transit_dict[site][transit_id] = {}
                transit_dict[site][transit_id]['NORAD_ID'] = obj_id
                transit_dict[site][transit_id]['start'] = start.strftime('%Y-%m-%dT%H:%M:%S')
                transit_dict[site][transit_id]['stop'] = stop.strftime('%Y-%m-%dT%H:%M:%S')
                transit_dict[site][transit_id]['duration'] = duration
                transit_dict[site][transit_id]['TCA'] = TCA.strftime('%Y-%m-%dT%H:%M:%S')
                transit_dict[site][transit_id]['TME'] = TME.strftime('%Y-%m-%dT%H:%M:%S')
                transit_dict[site][transit_id]['rg_min'] = rg_min
                transit_dict[site][transit_id]['el_max'] = el_max
#                transit_dict[site][transit_id]['UTC_transit'] = UTC_transit
#                transit_dict[site][transit_id]['az_transit'] = az_transit
#                transit_dict[site][transit_id]['el_transit'] = el_transit
#                transit_dict[site][transit_id]['rg_transit'] = rg_transit
                

                # Reset for new transit next round
                start = ti
                TCA = ti
                TME = ti
                stop = ti
                ti_prior = ti
                rg_min = 1e12
                el_max = -1
                UTC_transit = []
                az_transit = []
                el_transit = []
                rg_transit = []
                transit_int = int(transit_id[-zfill_count:])
                transit_id = site + '_' + str(transit_int+1).zfill(zfill_count)
    
    return transit_dict


def check_visibility(state, UTC_times, sun_gcrf_array, moon_gcrf_array, sensor,
                     spacecraftConfig, surfaces, eop_alldata, XYs_df=[]):
    
    
    start = time.time()
    
    # Sensor parameters
#    mapp_lim = sensor['mapp_lim']
    az_lim = sensor['az_lim']
    el_lim = sensor['el_lim']
    rg_lim = sensor['rg_lim']
#    sun_elmask = sensor['sun_elmask']
#    moon_angle_lim = sensor['moon_angle_lim']
    geodetic_latlonht = sensor['geodetic_latlonht']
    meas_types = ['rg', 'az', 'el']
    
    # Sensor coordinates
    lat = geodetic_latlonht[0]
    lon = geodetic_latlonht[1]
    ht = geodetic_latlonht[2]
    sensor_itrf = latlonht2ecef(lat, lon, ht)

    # Loop over times and check visiblity conditions
    vis_inds = []
    for ii in range(len(UTC_times)):
        
        # Retrieve time and current sun and object locations in ECI
        UTC = UTC_times[ii]
        Xi = state[ii,:]
        sun_gcrf = sun_gcrf_array[:,ii].reshape(3,1)
        
#        print(UTC)
        
        if ii % 100 == 0:
            print(ii)
            print(UTC)
            print('time elapsed:', time.time() - start)
        
        # Compute measurements
        EOP_data = get_eop_data(eop_alldata, UTC)
        
        compmeas = time.time()
        Yi = compute_measurement(Xi, sun_gcrf, sensor, spacecraftConfig,
                                 surfaces, UTC, EOP_data, meas_types,
                                 XYs_df)
        
#        print('Compute meas:', time.time() - compmeas)
        
#        print(Yi)
    
        rg = float(Yi[0])
        az = float(Yi[1])
        el = float(Yi[2])
        
        
        
        # Check against constraints
        vis_flag = True
        if el < el_lim[0]:
            vis_flag = False
        if el > el_lim[1]:
            vis_flag = False
        if az < az_lim[0]:
            vis_flag = False
        if az > az_lim[1]:
            vis_flag = False
        if rg < rg_lim[0]:
            vis_flag = False
        if rg > rg_lim[1]:
            vis_flag = False        
        
        # Optical constraints
        # Sunlit/station dark constraint
        if 'sun_elmask' in sensor:
            
            sun_elmask = sensor['sun_elmask']
            
            # Compute sun elevation angle
            gcrftime = time.time()
            sun_itrf, dum = gcrf2itrf(sun_gcrf, np.zeros((3,1)), UTC, EOP_data,
                                      XYs_df)
            
    #        print('GCRF time', time.time() - gcrftime)
            
            sun_az, sun_el, sun_rg = ecef2azelrange_rad(sun_itrf, sensor_itrf)
            
            if sun_el > sun_elmask:
                vis_flag = False
         
            # If passed constraints, check for eclipse and moon angle
            if vis_flag:
                
                print('visible')
                print(UTC)
                
                # Compute angles
                rso_gcrf = Xi[0:3].reshape(3,1)
                moon_gcrf = moon_gcrf_array[:,ii].reshape(3,1)
                sensor_gcrf, dum = \
                    itrf2gcrf(sensor_itrf, np.zeros((3,1)), UTC, EOP_data,
                              XYs_df)
                phase_angle, sun_angle, moon_angle = \
                    compute_angles(rso_gcrf, sun_gcrf, moon_gcrf, sensor_gcrf)
            
                # Check for eclipse - if sun angle is less than half cone angle
                # the sun is behind the earth
                # First check valid orbit - radius greater than Earth radius
                r = np.linalg.norm(rso_gcrf)
                if r < Re:
                    vis_flag = False
                else:
                    half_cone = asin(Re/r)
                    if sun_angle < half_cone:
                        vis_flag = False                
                
    #            # Check too close to moon
    #            if moon_angle < moon_angle_lim:
    #                vis_flag = False
    
                #TODO Moon Limits based on phase of moon (Meeus algorithm?)
        
        # If still good, compute apparent mag
        if vis_flag:
            
            print('visible')
            
#            print('az', az*180/pi)
#            print('el', el*180/pi)
#            print('sun az', sun_az*180/pi)
#            print('sun el', sun_el*180/pi)
            
            if 'mapp_lim' in sensor:
                mapp_lim = sensor['mapp_lim']
                meas_types_mapp = ['mapp']
                Yi = compute_measurement(Xi, sun_gcrf, sensor, spacecraftConfig,
                                         surfaces, UTC, EOP_data, meas_types_mapp,
                                         XYs_df)
            
                print(Yi)
                
                if float(Yi[0]) > mapp_lim:
                    vis_flag = False
        
        # If passed all checks, append to list
        if vis_flag:
            vis_inds.append(ii)
    
    return vis_inds


def compute_pass(UTC_vis, rg_vis, el_vis, sensor):
    '''
    This function computes times of importance during the pass, such as start,
    stop, TCA, and TME.  Also computes the minimum range and maximum elevation
    angle achieved during the pass. Computes pass times for one object and one
    sensor at a time.
    
    Parameters
    ------
    UTC_vis : 1D numpy array
        times when object is visible to this sensor
        stored as skyfield time objects that can be extracted in multiple
        time systems or representations
    rg_vis : 1D numpy array
        range values when object is visible to this sensor [km]
    el_vis : 1D numpy array
        elevation angle values when object is visible to this sensor [rad]
    sensor : dict
        sensor parameters including maximum gap between visible times for
        continous pass
    
    Returns
    ------
    start_list : list
        pass start times
    stop_list : list
        pass stop times
    TCA_list : list
        pass times of closest apporach
    TME_list : list
        pass times of maximum elevation
    rg_min_list : list
        pass minimum range values [km]
    el_max_list : list
        pass maximum elevation angle values [rad]

    '''
    
    # Retrieve pass length and gap parameters
    max_gap = sensor['max_gap']
    
    # Initialze output
    start_list = []
    stop_list = []
    TCA_list = []
    TME_list = []
    rg_min_list = []
    el_max_list = []
    
    # Only process if needed
    if len(UTC_vis) > 0 :
    
        # Loop over times
        rg_min = 1e12
        el_max = -1.
        ti_prior = UTC_vis[0]
        start = UTC_vis[0]
        stop = UTC_vis[0]
        TCA = UTC_vis[0]
        TME = UTC_vis[0]
        for ii in range(len(UTC_vis)):
            
            ti = UTC_vis[ii]
            rg_km = rg_vis[ii]
            el_rad = el_vis[ii]
            
            # If current time is close to previous, pass continues
            if (ti.tdb - ti_prior.tdb)*86400. < (max_gap+1.):
    
                # Update pass stop time and JD_prior for next iteration
                stop = ti
                ti_prior = ti
                
                # Check if this is pass time of closest approach (TCA)
                if rg_km < rg_min:
                    TCA = ti
                    rg_min = float(rg_km)
                
                # Check if this is pass time of maximum elevation (TME)
                if el_rad > el_max:
                    TME = ti
                    el_max = float(el_rad)
    
            # If current time is far from previous or if we reached
            # the end of UTC list, pass has ended
            if ((ti.tdb - ti_prior.tdb)*86400. >= (max_gap+1.) or ii == (len(UTC_vis)-1)):
                
                # TODO - LOGIC ERROR
                # Test this code for stop time if reached the end of UTC_vis
                if ii == (len(UTC_vis)-1):
                    stop = ti
                
                # Store output
                start_list.append(start)
                stop_list.append(stop)
                TCA_list.append(TCA)
                TME_list.append(TME)
                rg_min_list.append(rg_min)
                el_max_list.append(el_max)
                
                # Reset for new pass next round
                start = ti
                TCA = ti
                TME = ti
                stop = ti
                ti_prior = ti
                
                # TODO - LOGIC ERROR
                # Test this code to reset these params
                rg_min = 1e12
                el_max = -1
                        
    return start_list, stop_list, TCA_list, TME_list, rg_min_list, el_max_list


def compute_mapp(phase_angle, sat_rg, sat_radius, albedo=0.1):
    '''
    This function computes the apparent magnitude of a space object
    due to reflected sunlight. Assumes object is diffuse sphere.

    Reference
    ------
    Cognion, "Observations and Modeling of GEO Satellites at Large
    Phase Angles," 2013.

    Parameters
    ------
    phase_angle : float
        angle at satellite between sun vector and observer vector [rad]
    sat_rg : float
        range from observer to satellite [km]
    sat_radius : float
        radius of spherical satellite [km]
    albedo : float, optional
        unitless reflectivity parameter of object (default = 0.1)

    Returns
    ------
    mapp : float
        apparent magnitude
    '''

    # Fraction of incident solar flux reflected by diffuse sphere satellite
    F_diff = (2./3.) * (albedo/pi) * (sat_radius/sat_rg)**2. * \
        (sin(phase_angle) + (pi - phase_angle)*cos(phase_angle))

    if sat_radius == 0.:
        mapp = 100.
    else:
        mapp = -26.74 - 2.5*log10(F_diff)

    return mapp


def rcs2radius_meters(rcs_m2):
    '''
    This function computes a satellite radius in meters from a given
    RCS obtained from SATCAT. Assumes object is diffuse sphere.

    Reference
    ------
    Cognion, "Observations and Modeling of GEO Satellites at Large
    Phase Angles," 2013.

    Parameters
    ------
    rcs_m2 : float
        radar cross section [m^2]

    Returns
    ------
    r_m : float
        satellite radius [m]
    '''

    # Cognion formula works for RCS > 0.0268 m^2
    # Otherwise, the RCS drops off rapidly as function of true size
    if rcs_m2 < 0.:
        r_m = 0.

    elif rcs_m2 > 0.0268:
        r_m = 2. * np.sqrt(rcs_m2/pi)   # m

    else:
        lam = 1.23  # m
        r_m = (4.*rcs_m2*lam**4. / (9.*pi**5.))**(1./6.)   # m

    return r_m


def compute_angles(rso_gcrf, sun_gcrf, moon_gcrf, sensor_gcrf):
    '''
    This function computes a set of 3 angles for visibility checks:
    1. phase_angle between the sun-satellite-station (satellite at vertex)
    2. sun_angle between sun-satellite-Earth CM (satellite at vertex)
    3. moon_angle between moon-station-satellite (station at vertex)

    Parameters
    ------
    rso_gcrf : 3x1 numpy array
        satellite position in GCRF [km]
    sun_gcrf : 3x1 numpy array
        sun position in GCRF [km]
    moon_gcrf : 3x1 numpy array
        moon position in GCRF [km]
    sensor_gcrf : 3x1 numpy array
        sensor position in GCRF [km]

    Returns
    ------
    phase_angle : float
        sun-satellite-station angle [rad]
    sun_angle : float
        sun-satellite-earth angle [rad]
    moon_angle : float
        moon-station-satellite angle [rad]
    '''
    
    # Compute relative position vectors
    sat2sun = sun_gcrf - rso_gcrf
    sat2sensor = sensor_gcrf - rso_gcrf
    moon2sensor = sensor_gcrf - moon_gcrf

    # Unit vectors and angles
    u_sun = sat2sun.flatten()/np.linalg.norm(sat2sun)
    u_sensor = sat2sensor.flatten()/np.linalg.norm(sat2sensor)
    u_sat = rso_gcrf.flatten()/np.linalg.norm(rso_gcrf)
    u_moon = moon2sensor.flatten()/np.linalg.norm(moon2sensor)

    phase_angle = acos(np.dot(u_sun, u_sensor))
    sun_angle = acos(np.dot(u_sun, -u_sat))
    moon_angle = acos(np.dot(u_moon, u_sensor))

    return phase_angle, sun_angle, moon_angle


def compute_sun_coords(TT_cent):
    '''
    This function computes sun coordinates using the simplified model in
    Meeus Ch 25.  The results here follow the "low accuracy" model and are
    expected to have an accuracy within 0.01 deg.
    
    Parameters
    ------
    TT_cent : float
        Julian centuries since J2000 TT
        
    Returns
    ------
    sun_eci_geom : 3x1 numpy array
        geometric position vector of sun in ECI [km]
    sun_eci_app : 3x1 numpy array
        apparent position vector of sun in ECI [km]
        
    Reference
    ------
    [1] Meeus, J., "Astronomical Algorithms," 2nd ed., 1998, Ch 25.
    
    Note that Meeus Ch 7 and Ch 10 describe time systems TDT and TDB as 
    essentially the same for the purpose of these calculations (they will
    be within 0.0017 seconds of one another).  The time system TT = TDT is 
    chosen for consistency with the IAU Nutation calculations which are
    explicitly given in terms of TT.
    
    '''
    
    # Conversion
    deg2rad = pi/180.
    
    # Geometric Mean Longitude of the Sun (Mean Equinox of Date)
    Lo = 280.46646 + (36000.76983 + 0.0003032*TT_cent)*TT_cent   # deg
    Lo = Lo % 360.
    
    # Mean Anomaly of the Sun
    M = 357.52911 + (35999.05028 - 0.0001537*TT_cent)*TT_cent    # deg
    M = M % 360.
    Mrad = M*deg2rad                                             # rad
    
    # Eccentricity of Earth's orbit
    ecc = 0.016708634 + (-0.000042037 - 0.0000001267*TT_cent)*TT_cent
    
    # Sun's Equation of Center
    C = (1.914602 - 0.004817*TT_cent - 0.000014*TT_cent*TT_cent)*sin(Mrad) + \
        (0.019993 - 0.000101*TT_cent)*sin(2.*Mrad) + 0.000289*sin(3.*Mrad)  # deg
        
    # Sun True Longitude and True Anomaly
    true_long = Lo + C  # deg
    true_anom = M + C   # deg
    true_long_rad = true_long*deg2rad
    true_anom_rad = true_anom*deg2rad
    
    # Sun radius (distance from Earth)
    R_AU = 1.000001018*(1. - ecc**2)/(1 + ecc*cos(true_anom_rad))       # AU
    R_km = R_AU*AU_km                                                   # km
    
    # Compute Sun Apparent Longitude
    Omega = 125.04 - 1934.136*TT_cent                                   # deg
    Omega_rad = Omega*deg2rad                                           # rad
    apparent_long = true_long - 0.00569 - 0.00478*sin(Omega_rad)        # deg
    apparent_long_rad = apparent_long*deg2rad                           # rad
    
    # Obliquity of the Ecliptic (Eq 22.2)
    Eps0 = (((0.001813*TT_cent - 0.00059)*TT_cent - 46.8150)*TT_cent 
              + 84381.448)/3600.                                        # deg
    Eps0_rad = Eps0*deg2rad                                             # rad
    cEps0 = cos(Eps0_rad)
    sEps0 = sin(Eps0_rad)
    
    # Geometric Coordinates
    sun_ecliptic_geom = R_km*np.array([[cos(true_long_rad)],
                                       [sin(true_long_rad)],
                                       [                0.]])

    # r_Equator = R1(-Eps) * r_Ecliptic
    R1 = np.array([[1.,       0.,       0.],
                   [0.,    cEps0,   -sEps0],
                   [0.,    sEps0,    cEps0]])
    
    sun_eci_geom = np.dot(R1, sun_ecliptic_geom)
    
    # Apparent Coordinates
    Eps_true = Eps0 + 0.00256*cos(Omega_rad)    # deg
    Eps_true_rad = Eps_true*deg2rad 
    cEpsA = cos(Eps_true_rad)
    sEpsA = sin(Eps_true_rad) 
    
    sun_ecliptic_app = R_km*np.array([[cos(apparent_long_rad)],
                                      [sin(apparent_long_rad)],
                                      [                    0.]])
    
    # r_Equator = R1(-Eps) * r_Ecliptic 
    R1 = np.array([[1.,       0.,       0.],
                   [0.,    cEpsA,   -sEpsA],
                   [0.,    sEpsA,    cEpsA]])
    
    sun_eci_app = np.dot(R1, sun_ecliptic_app)

    
    return sun_eci_geom, sun_eci_app


def compute_moon_coords(TT_cent):
    '''
    This function computes moon coordinates using the simplified model in
    Meeus Ch 47.
    
    Parameters
    ------
    TT_cent : float
        Julian centuries since J2000 TT
        
    Returns
    ------
    moon_eci_geom : 3x1 numpy array
        geometric position vector of sun in ECI [km]
    moon_eci_app : 3x1 numpy array
        apparent position vector of sun in ECI [km]
    
        
    Reference
    ------
    [1] Meeus, J., "Astronomical Algorithms," 2nd ed., 1998, Ch 47.
    
    Note that Meeus Ch 7 and Ch 10 describe time systems TDT and TDB as 
    essentially the same for the purpose of these calculations (they will
    be within 0.0017 seconds of one another).  The time system TT = TDT is 
    chosen for consistency with the IAU Nutation calculations which are
    explicitly given in terms of TT.
    
    '''
    
    # Conversion
    deg2rad = pi/180.
    arcsec2rad  = (1./3600.) * (pi/180.)
    
    # Compute fundamental arguments of nutation   
    moon_mean_longitude = (218.3164477 + 481267.88123421*TT_cent -
                           0.0015786*TT_cent**2. + (TT_cent**3.)/538841. -
                           (TT_cent**4.)/65194000.) * deg2rad

    moon_mean_elongation = (297.8501921 + 445267.1114034*TT_cent -
                            0.0018819*TT_cent**2. + (TT_cent**3.)/545868. -
                            (TT_cent**4.)/113065000.) * deg2rad

    sun_mean_anomaly = (357.5291092 + 35999.0502909*TT_cent - 0.0001536*TT_cent**2. +
                        (TT_cent**3.)/24490000.) * deg2rad

    moon_mean_anomaly = (134.9633964 + 477198.8675055*TT_cent + 0.0087414*TT_cent**2. +
                         (TT_cent**3.)/69699. - (TT_cent**4.)/14712000.) * deg2rad

    moon_arg_lat = (93.2720950 + 483202.0175233*TT_cent - 0.0036539*TT_cent**2. -
                    (TT_cent**3.)/3526000. + (TT_cent**4.)/863310000.) * deg2rad

    moon_loan = (125.04452 - 1934.136261*TT_cent + 0.0020708*TT_cent**2. +
                 (TT_cent**3.)/450000) * deg2rad

    # Additioanl Arguments
    A1 = (119.75 + 131.849*TT_cent) * deg2rad
    A2 = (53.09 + 479264.290*TT_cent) * deg2rad
    A3 = (313.45 + 481266.484*TT_cent) * deg2rad
    
    # Correction term for changing Earth eccentricity
    E = 1. - 0.002516*TT_cent - 0.0000074*TT_cent**2.
    
    # Coefficient lists for longitude (L) and distance (R) (Table 47.A) 
    mat1 = np.zeros((60,4))
    mat1[:,0] = [0,2,2,0,0,0,2,2,2,2,0,1,0,2,0,0,4,0,4,2,2,1,1,2,2,4,2,0,2,2,1,2,
                 0,0,2,2,2,4,0,3,2,4,0,2,2,2,4,0,4,1,2,0,1,3,4,2,0,1,2,2]

    mat1[:,1] = [0,0,0,0,1,0,0,-1,0,-1,1,0,1,0,0,0,0,0,0,1,1,0,1,-1,0,0,0,1,0,-1,
                 0,-2,1,2,-2,0,0,-1,0,0,1,-1,2,2,1,-1,0,0,-1,0,1,0,1,0,0,-1,2,1,
                 0,0]

    mat1[:,2] = [1,-1,0,2,0,0,-2,-1,1,0,-1,0,1,0,1,1,-1,3,-2,-1,0,-1,0,1,2,0,-3,
                -2,-1,-2,1,0,2,0,-1,1,0,-1,2,-1,1,-2,-1,-1,-2,0,1,4,0,-2,0,2,1,
                -2,-3,2,1,-1,3,-1]

    mat1[:,3] = [0,0,0,0,0,2,0,0,0,0,0,0,0,-2,2,-2,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,
               0,0,0,0,-2,2,0,2,0,0,0,0,0,0,-2,0,0,0,0,-2,-2,0,0,0,0,0,0,0,-2]
    
    L_coeff = [6288774,1274027,658314,213618,-185116,-114332,58793,57066,53322,
               45758,-40923,-34720,-30383,15327,-12528,10980,10675,10034,8548,
               -7888,-6766,-5163,4987,4036,3994,3861,3665,-2689,-2602,2390,
               -2348,2236,-2120,-2069,2048,-1773,-1595,1215,-1110,-892,-810,
               759,-713,-700,691,596,549,537,520,-487,-399,-381,351,-340,330,
               327,-323,299,294,0]
    
    R_coeff = [-20905355,-3699111,-2955968,-569925,48888,-3149,246158,-152138,
               -170733,-204586,-129620,108743,104755,10321,0,79661,-34782,
               -23210,-21636,24208,30824,-8379,-16675,-12831,-10445,-11650,
               14403,-7003,0,10056,6322,-9884,5751,0,-4950,4130,0,-3958,0,3258,
               2616,-1897,-2117,2354,0,0,-1423,-1117,-1571,-1739,0,-4421,0,0,0,
               0,1165,0,0,8752]
    
    # Coefficient lists for latitude (B) (Table 47.B) 
    mat2 = np.zeros((60, 4))
    mat2[:,0] = [0,0,0,2,2,2,2,0,2,0,2,2,2,2,2,2,2,0,4,0,0,0,1,0,0,0,1,0,4,4,0,4,
               2,2,2,2,0,2,2,2,2,4,2,2,0,2,1,1,0,2,1,2,0,4,4,1,4,1,4,2]
    
    mat2[:,1] = [0,0,0,0,0,0,0,0,0,0,-1,0,0,1,-1,-1,-1,1,0,1,0,1,0,1,1,1,0,0,0,0,
               0,0,0,0,-1,0,0,0,0,1,1,0,-1,-2,0,1,1,1,1,1,0,-1,1,0,-1,0,0,0,-1,
               -2]
    
    mat2[:,2] = [0,1,1,0,-1,-1,0,2,1,2,0,-2,1,0,-1,0,-1,-1,-1,0,0,-1,0,1,1,0,0,
                3,0,-1,1,-2,0,2,1,-2,3,2,-3,-1,0,0,1,0,1,1,0,0,-2,-1,1,-2,2,-2,
                -1,1,1,-1,0,0]
    
    mat2[:,3] = [1,1,-1,-1,1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,-1,-1,-1,1,3,1,1,1,-1,
               -1,-1,1,-1,1,-3,1,-3,-1,-1,1,-1,1,-1,1,1,1,1,-1,3,-1,-1,1,-1,-1,
               1,-1,1,-1,-1,-1,-1,-1,-1,1]
    
    B_coeff = [5128122,280602,277693,173237,55413,46271,32573,17198,9266,8822,
               8216,4324,4200,-3359,2463,2211,2065,-1870,1828,-1794,-1749,
               -1565,-1491,-1475,-1410,-1344,-1335,1107,1021,833,777,671,607,
               596,491,-451,439,422,421,-366,-351,331,315,302,-283,-229,223,
               223,-220,-220,-185,181,-177,176,166,-164,132,-119,115,107]


    # Update amplitude of sin/cos terms to correct for changing eccentricity 
    # of Earth orbit
    E_list1 = [E**abs(Mcoeff) for Mcoeff in mat1[:,1]]
    E_list2 = [E**abs(Mcoeff) for Mcoeff in mat1[:,1]]
    
    L_coeff = list(np.multiply(E_list1, L_coeff))    
    R_coeff = list(np.multiply(E_list1, R_coeff))
    B_coeff = list(np.multiply(E_list2, B_coeff))
    
    # Vectorize accumulation of sums of longitude, latitude, distance
    args_vec = np.reshape([moon_mean_elongation, sun_mean_anomaly,
                           moon_mean_anomaly, moon_arg_lat], (4,1))
    arg1 = np.dot(mat1, args_vec)
    arg2 = np.dot(mat2, args_vec)
    L_sum = np.dot(L_coeff, np.sin(arg1))
    R_sum = np.dot(R_coeff, np.cos(arg1))
    B_sum = np.dot(B_coeff, np.sin(arg2)) 


    # Additional corrections due to Venus (A1), Jupiter (A2), and flattening
    # of Earth (moon_mean_longitude)
    # Units of L_sum and B_sum are 1e-6 deg
    L_sum += 3958.*sin(A1) + 1962.*sin(moon_mean_longitude - moon_arg_lat) \
        + 318.*sin(A2)

    B_sum += -2235.*sin(moon_mean_longitude) + 382.*sin(A3) \
        + 175.*sin(A1 - moon_arg_lat) + 175.*sin(A1 + moon_arg_lat) \
        + 127.*sin(moon_mean_longitude - moon_mean_anomaly) \
        - 115.*sin(moon_mean_longitude + moon_mean_anomaly)

    # Calculation moon coordinates    
    lon_rad = moon_mean_longitude + (L_sum/1e6) * deg2rad
    lat_rad = (B_sum/1e6) * deg2rad
    r_km = 385000.56 + R_sum/1000.

    
    # Obliquity of the Ecliptic (Eq 22.2)
    Eps0 = (((0.001813*TT_cent - 0.00059)*TT_cent - 46.8150)*TT_cent + 84381.448)/3600.   # deg
    Eps0_rad = Eps0*deg2rad   
    cEps0 = cos(Eps0_rad)
    sEps0 = sin(Eps0_rad)
    
    # Geometric coordinates
    moon_ecliptic_geom = r_km * np.array([[cos(lon_rad)*cos(lat_rad)],
                                          [sin(lon_rad)*cos(lat_rad)],
                                          [sin(lat_rad)]])
    
    # r_Equator = R1(-Eps0) * r_Ecliptic
    R1 = np.array([[1.,       0.,       0.],
                   [0.,    cEps0,   -sEps0],
                   [0.,    sEps0,    cEps0]])
    
    moon_eci_geom = np.dot(R1, moon_ecliptic_geom)
    
    
    # Apparent coordinates
    sun_mean_longitude = (280.4665 + 36000.7689*TT_cent)*deg2rad
    dPsi = (-17.2*sin(moon_loan) - 1.32*sin(2*sun_mean_longitude) 
            - 0.23*sin(2*moon_mean_longitude) + 0.21*sin(2*moon_loan))*arcsec2rad
    dEps = (9.2*cos(moon_loan) + 0.57*cos(2*sun_mean_longitude) 
            + 0.1*cos(2*moon_mean_longitude) - 0.09*cos(2*moon_loan))*arcsec2rad
    
    Eps_true_rad = Eps0_rad + dEps   # rad
    cEpsA = cos(Eps_true_rad)
    sEpsA = sin(Eps_true_rad)
    
    
    
    lon_app_rad = lon_rad + dPsi
    
    moon_ecliptic_app = r_km * np.array([[cos(lon_app_rad)*cos(lat_rad)],
                                         [sin(lon_app_rad)*cos(lat_rad)],
                                         [sin(lat_rad)]])
    
    # r_Equator = R1(-EpsA) * r_Ecliptic
    R1 = np.array([[1.,       0.,       0.],
                   [0.,    cEpsA,   -sEpsA],
                   [0.,    sEpsA,    cEpsA]])
    
    moon_eci_app = np.dot(R1, moon_ecliptic_app)

    
    return moon_eci_geom, moon_eci_app


def generate_visibility_file(vis_dict, vis_file, vis_file_min_el):
    '''
    This function produces the visibility .CSV file with output passes
    sorted by start times.
    
    Parameters
    ------
    vis_dict : dictionary
        sorted lists of pass start/stop/TCA/TME times and minimum range
        and maximum elevation angle
    vis_file : string
        path and filename for output .CSV file
    vis_file_min_el : float
        minimum elevation angle desired for passes in output file
        can be different from sensor minimum elevation constraints used to 
        determine the pass start and stop times
        
    '''
    
    
    # Retrieve sorted lists
    start_list = vis_dict['start_list']
    stop_list = vis_dict['stop_list']
    TCA_list = vis_dict['TCA_list']
    TME_list = vis_dict['TME_list']
    rg_min_list = vis_dict['rg_min_list']
    el_max_list = vis_dict['el_max_list']
    obj_id_list = vis_dict['obj_id_list']
    sensor_id_list = vis_dict['sensor_id_list']
    
    with open(vis_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Sensor ID','Object NORAD ID', 'UTC Start', 
                         'UTC Stop', 'UTC Time Closest Approach',
                         'UTC Time Max Elevation', 'Min Range [km]',
                         'Max Elevation [deg]'])
    
        for ii in range(len(start_list)):
            UTC_start = start_list[ii].utc_iso()
            UTC_stop = stop_list[ii].utc_iso()
            UTC_TCA = TCA_list[ii].utc_iso()
            UTC_TME = TME_list[ii].utc_iso()
            rg_min = str(int(round(rg_min_list[ii])))
            el_max = str(int(round(el_max_list[ii]*180./pi)))
            obj_id = str(obj_id_list[ii])
            sensor_id = sensor_id_list[ii]
            
            if float(el_max) < vis_file_min_el:
                continue
    
            writer.writerow([sensor_id, obj_id, UTC_start, UTC_stop,
                             UTC_TCA, UTC_TME, rg_min, el_max])
    
    csvfile.close()
    
    return





