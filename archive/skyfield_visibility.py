import numpy as np
from math import pi, cos, sin, acos, asin, log10
import os
import sys
import csv

sys.path.append('../')

from skyfield.constants import ERAD
from skyfield.api import Topos, EarthSatellite, Loader

from sensors import define_sensors
from utilities.tle_functions import get_spacetrack_tle_data


def define_RSOs(obj_id_list, tle_dict={}):
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
        If none provided, script will retrieve from space-track.org
    
    Returns
    ------
    rso_dict : dictionary
        RSO state and parameters indexed by object NORAD ID
    '''
    
    
    # Initialize output
    rso_dict = {}    
    
    # Load TLE Data
    # Include options here to import from space-track, celestrak, text file,
    # other URL, graph database, ...
    
    if len(tle_dict) == 0:
        
        # Download from space-track.org
        tle_dict = get_spacetrack_tle_data(obj_id_list)    

    # Retrieve TLE data and form RSO dictionary using skyfield
    for obj_id in obj_id_list:
        line1 = tle_dict[obj_id]['line1']
        line2 = tle_dict[obj_id]['line2']
        
        # Instantiate skyfield object
        satellite = EarthSatellite(line1, line2, name=str(obj_id))
        rso_dict[obj_id] = {}
        rso_dict[obj_id]['satellite'] = satellite
        
        
    # Initialize object size
    # Include options here for RCS from SATCAT, graph database, ...    
    for obj_id in obj_id_list:
        
        # Dummy value for all satellites
        rso_dict[obj_id]['rcs_m2'] = 1.
    
    return rso_dict



def compute_visible_passes(UTC_array, obj_id_list, sensor_id_list, ephemeris,
                           tle_dict={}):
    '''
    This function computes the visible passes for a given list of 
    resident space objects (RSOs) from one or more sensors. Output includes
    the start and stop times of each pass, as well as the time of closest
    approach (TCA) and time of maximum elevation (TME).  
    
    Parameters
    ------
    UTC_array : 1D numpy array
        times to compute visibility conditions
        stored as skyfield time objects that can be extracted in multiple
        time systems or representations
    obj_id_list : list
        object NORAD IDs (int)
    sensor_id_list : list
        sensor IDs (str)
    ephemeris : skyfield object
        contains data about sun, moon, and planets loaded from skyfield
    
    Returns
    ------
    vis_dict : dictionary
        contains sorted lists of pass start and stop times for each object
        and sensor, as well as TCA and TME times, and maximum elevation angle
        and minimum range to the RSO during the pass

    '''
    
    # Constants
    Re = ERAD/1000.   # km
    
    # Generate resident space object dictionary
    rso_dict = define_RSOs(obj_id_list, tle_dict)
    
    # Load sensor data
    # Include options here to load from file, URL, graph database, ...
    
    # Load from script
    sensor_dict = define_sensors()
    
    # Remove sensors not in list
    sensor_remove_list = list(set(sensor_dict.keys()) - set(sensor_id_list))
    for sensor_id in sensor_remove_list:
        sensor_dict.pop(sensor_id, None)
    
    # Instantiate a skyfield object for each sensor in list
    for sensor_id in sensor_dict.keys():
        geodetic_latlonht = sensor_dict[sensor_id]['geodetic_latlonht']
        lat = geodetic_latlonht[0]
        lon = geodetic_latlonht[1]
        elevation_m = geodetic_latlonht[2]*1000.
        statTopos = Topos(latitude_degrees=lat, longitude_degrees=lon,
                          elevation_m=elevation_m)
        sensor_dict[sensor_id]['statTopos'] = statTopos

    # Retrieve sun and moon positions for full timespan
    earth = ephemeris['earth']
    sun = ephemeris['sun']
    moon = ephemeris['moon']
    moon_gcrf_array = earth.at(UTC_array).observe(moon).position.km
    sun_gcrf_array = earth.at(UTC_array).observe(sun).position.km

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
        rso_gcrf_array = rso['satellite'].at(UTC_array).position.km
        
        # Retrieve object size and albedo if available/needed
        radius_km = rcs2radius_meters(rso['rcs_m2'])/1000.

        # Loop over sensors        
        for sensor_id in sensor_dict:
            sensor = sensor_dict[sensor_id]
            sensor_gcrf_array = sensor['statTopos'].at(UTC_array).position.km
            
            # Constraint parameters
            el_lim = sensor['el_lim']
            az_lim = sensor['az_lim']
            rg_lim = sensor['rg_lim']
            sun_elmask = sensor['sun_elmask']
            mapp_lim = sensor['mapp_lim']
            moon_angle_lim = sensor['moon_angle_lim']
            
            # Compute topocentric RSO position
            # For earth satellites, calling observe and apparent is costly
            # and unnecessary except for meter level accuracy
            difference = rso['satellite'] - sensor['statTopos']
            rso_topo = difference.at(UTC_array)
            el_array, az_array, rg_array = rso_topo.altaz()
            
            # Compute topocentric sun position
            # Need both sun and sensor positions referenced from solar
            # system barycenter
            sensor_ssb = earth + sensor['statTopos']
            sun_topo = sensor_ssb.at(UTC_array).observe(sun).apparent()
            sun_el_array, sun_az_array, sun_rg_array = sun_topo.altaz()
            
            # Find indices where az/el/range constraints are met
            el_inds0 = np.where(el_array.radians > el_lim[0])[0]
            el_inds1 = np.where(el_array.radians < el_lim[1])[0]
            az_inds0 = np.where(az_array.radians > az_lim[0])[0]
            az_inds1 = np.where(az_array.radians < az_lim[1])[0]
            rg_inds0 = np.where(rg_array.km > rg_lim[0])[0]
            rg_inds1 = np.where(rg_array.km < rg_lim[1])[0]
            
            # Check sun constraint (ensures station is dark if needed)
            sun_el_inds = np.where(sun_el_array.radians < sun_elmask)[0]
            
            # Find all common elements to create candidate visible index list
            common_el = set(el_inds0).intersection(set(el_inds1))
            common_az = set(az_inds0).intersection(set(az_inds1))
            common_rg = set(rg_inds0).intersection(set(rg_inds1))
            common1 = common_el.intersection(common_az)
            common2 = common1.intersection(common_rg)
            common_inds = list(common2.intersection(set(sun_el_inds)))
            
            # Initialze visibility array for this sensor and object
            vis_array = np.zeros(rso_gcrf_array.shape[1],)
            vis_array[common_inds] = True
            
            # For remaining indices compute angles and visibility conditions
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
                
                # Check too close to moon
                if moon_angle < moon_angle_lim:
                    vis_array[ii] = False
                                
                # Check apparent magnitude
                # Optional input for albedo could be retrieved for each object
                # from catalog                
                mapp = compute_mapp(phase_angle, rg_km, radius_km)
                if mapp > mapp_lim:
                    vis_array[ii] = False
            
            vis_inds = np.where(vis_array)[0]
            UTC_vis = UTC_array[vis_inds]
            rg_vis = rg_array.km[vis_inds]
            el_vis = el_array.radians[vis_inds]
            
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
                rg_min = float(rg_km)
                el_max = float(el_rad)
                        
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


###############################################################################
# Stand-alone execution
###############################################################################


if __name__ == '__main__':    
    
    load = Loader(os.path.join(metis_dir, 'skyfield_data'))
    ts = load.timescale()
    ephemeris = load('de430t.bsp')

    obj_id_list = [43014]
    sensor_id_list = ['PSU Falcon', 'NJC Falcon', 'FLC Falcon']
    
    # Minimum elevation pass for output file
    vis_file_min_el = 20.  # deg
    
    # Create an array of times
    ndays = 6
    dt = 10  # sec    
    UTC_now = ts.now().utc
    print(UTC_now)
    mistake
    sec_array = list(range(0,86400*ndays,dt))
    UTC_array = ts.utc(UTC_now[0], UTC_now[1], UTC_now[2]+3, UTC_now[3],
                       UTC_now[4], sec_array)
    
    # Generate visible pass dictionary
    vis_dict = compute_visible_passes(UTC_array, obj_id_list, sensor_id_list,
                                      ephemeris)

    # Generate output file
    outdir = os.path.join(metis_dir, 'skyfield_data')
    vis_file = os.path.join(outdir, 'visible_passes.csv')
    generate_visibility_file(vis_dict, vis_file, vis_file_min_el)


