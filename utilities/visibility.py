import numpy as np
from math import *

import ephem



###############################################################################
# This script contains functions to perform visibility checks for the
# SERC Scheduler.
#
# Functions:
#  generate_visibility_dict
#  check_visibility
#  compute_angles
#  compute_mapp
#  rcs2radius_meters
#  station_dark
#  compute_sun_eci
#  compute_moon_eci
#
###############################################################################


def generate_visibity_dict(GMM_full_time, sensor_id_list, sensor_dict, inputs):
    '''
    This function checks the visibility conditions of all objects in the
    catalog across all sensors for the duration of the schedule. This includes
    checks against contraints in az/el/range and apparent magnitude, as well
    as ensuring the station is dark and the object is not close to the moon.

    Parameters
    ------
    GMM_full_time : dictionary
        GMM through whole schedule duration, indexed by time
    sensor_id_list : list
        list of sensor IDs
    sensor_dict : dictionary
        sensor parameters including location and measurement type
    inputs : dictionary
        parameters including frame rotation details

    Returns
    ------
    vis_dict : dictionary
        times and measurements of visible objects, indexed by sensor id

    '''
    

    # Loop over objects
    vis_dict = {}
    for obj_id in GMM_full_time.keys():
        
        # Retrieve times and object state vectors        
        JD_list = GMM_full_time[obj_id]['JD_list']
        state_list = GMM_full_time[obj_id]['state_list']

        # Loop over sensors
        for sensor_id in sensor_id_list:
            
            # Initialize output
            if sensor_id not in vis_dict.keys():
                vis_dict[sensor_id] = {}
               
            # Retrive sensor location in ECEF
            sensor = sensor_dict[sensor_id]
            stat_ecef = sensor['stat_ecef']

            # Loop over times
            for ii in xrange(len(JD_list)):
                
                # Retrive current time and state
                JED_JD = float(JD_list[ii])
                mobj = np.array(state_list[ii][0:6]).reshape(6,1)
                r_eci = np.reshape(mobj[0:3], (3, 1))
                
                # Convert sensor location to ECI                
                stat_eci = conv.ecef2eci(stat_ecef, inputs, JED_JD)

                # Check visibility against constraints
                vis_flag, az, el, rg, mapp = \
                    check_visibility(r_eci, obj_id, sensor, stat_eci, inputs,
                                     JED_JD)

                # If visible, add time to dictionary
                if vis_flag:

                    # Initialize output
                    if obj_id not in vis_dict[sensor_id].keys():
                        vis_dict[sensor_id][obj_id] = {}
                        vis_dict[sensor_id][obj_id]['JD_list'] = []
                        vis_dict[sensor_id][obj_id]['meas_list'] = []
                        vis_dict[sensor_id][obj_id]['mapp_list'] = []
                        vis_dict[sensor_id][obj_id]['state_list'] = []

                    # Store output
                    meas = np.reshape([az, el, rg], (3, 1))
                    vis_dict[sensor_id][obj_id]['JD_list'].append(JED_JD)
                    vis_dict[sensor_id][obj_id]['meas_list'].append(meas)
                    vis_dict[sensor_id][obj_id]['mapp_list'].append(mapp)
                    vis_dict[sensor_id][obj_id]['state_list'].append(mobj)

    return vis_dict


def check_visibility(r_eci, obj_id, sensor, stat_eci, inputs, JED_JD):
    '''
    This function checks the visibilty conditions for one object at one time
    from one sensor.

    Parameters
    ------
    r_eci : 3x1 numpy array
        object position in ECI [km]
    obj_id : int
        object NORAD ID
    sensor : dictionary
        dictionary of sensor parameters including az/el/range/mapp limits
    stat_eci : 3x1 numpy array
        station position in ECI [km]
    inputs : dictionary
        parameters including RCS values and frame rotations
    JED_JD : float
        current time [JED] in julian date format

    Returns
    ------
    vis_flag : int
        boolean-like descriptor of visibility status
        (0 = not visible, 1 = visible)
    '''

    # Constraint parameters
    el_lim = sensor['el_lim']
    az_lim = sensor['az_lim']
    rg_lim = sensor['rg_lim']
    mapp_lim = sensor['mapp_lim']
    #phase_angle_stat_lim = sensor['phase_angle_stat_lim']
    moon_angle_lim = sensor['moon_angle_lim']

    # Object size parameters
    try:
        rcs_m2 = inputs['rcs_dict'][obj_id]
    except:
        rcs_m2 = 0.
    radius_km = rcs2radius_meters(rcs_m2)/1000.

    # Output flag
    vis_flag = 1

    # Compute range, azimuth and elevation
    r_ecef = conv.eci2ecef(r_eci, inputs, JED_JD)
    stat_ecef = sensor['stat_ecef']
    az, el, rg = conv.ecef2azelrange_rad(r_ecef, stat_ecef)

    # Check az/el/range constraints
    if el < el_lim[0]:
        vis_flag = 0
    if el > el_lim[1]:
        vis_flag = 0
    if az < az_lim[0]:
        vis_flag = 0
    if az > az_lim[1]:
        vis_flag = 0
    if rg < rg_lim[0]:
        vis_flag = 0
    if rg > rg_lim[1]:
        vis_flag = 0

    # Compute the sun/moon position in ECI
    sun_eci = compute_sun_eci2(JED_JD)
    moon_eci = compute_moon_eci2(JED_JD)

    # Check if station is dark
    if not station_dark(sun_eci, sensor, inputs, JED_JD):
        vis_flag = 0

    # Compute phase angle (sun-satellite-station), station phase angle
    # (sun-station-satellite), sun angle (sun-satellite-earth),
    # moon_angle (moon-station-satellite)
    phase_angle, phase_angle_stat, sun_angle, moon_angle = \
        compute_angles(r_eci, sun_eci, moon_eci, stat_eci)

    # Check for eclipse - if sun angle is less than half cone angle
    # the sun is behind the earth
    # First check valid orbit - radius greater than Earth
    r = np.linalg.norm(r_eci)
    if r <= inputs['Re']:
        vis_flag = 0
    else:
        half_cone = asin(inputs['Re']/r)
        if sun_angle <= half_cone:
            vis_flag = 0

#    # Check mimPhaseAngle - temporary check, can remove if mapp check is
#    # accurate
#    if phase_angle_stat <= phase_angle_stat_lim:
#        vis_flag = 0

    # Check apparent magnitude
    # Optional input for albedo could be retrieved for each object
    # from catalog - as should RCS or some other size parameter to
    # get radius
    mapp = compute_mapp(phase_angle, rg, radius_km)
    if mapp >= mapp_lim:
        vis_flag = 0

    # Check too close to moon
    if moon_angle <= moon_angle_lim:
        vis_flag = 0

    return vis_flag, az, el, rg, mapp


def compute_angles(r_eci, sun_eci, moon_eci, stat_eci):
    '''
    This function computes a set of 4 angles for visibility checks:
    1. phase_angle between the sun-satellite-station (satellite at vertex)
    2. phase_angle_stat between the sun-station-satellite (station at vertex)
    3. sun_angle between sun-satellite-Earth CM (satellite at vertex)
    4. moon_angle between moon-station-satellite (station at vertex)

    Parameters
    ------
    r_eci : 3x1 numpy array
        satellite position in ECI [km]
    sun_eci : 3x1 numpy array
        sun position in ECI [km]
    moon_eci : 3x1 numpy array
        moon position in ECI [km]
    stat_eci : 3x1 numpy array
        ground station position in ECI [km]

    Returns
    ------
    phase_angle : float
        sun-satellite-station angle [rad]
    phase_angle_stat : float
        sun-station-satellite angle [rad]
    sun_angle : float
        sun-satellite-earth angle [rad]
    moon_angle : float
        moon-station-satellite angle [rad]
    '''
    # Compute relative position vectors
    sat2sun = sun_eci - r_eci
    sat2stat = stat_eci - r_eci
    stat2sun = sun_eci - stat_eci
    moon2stat = stat_eci - moon_eci

    # Unit vectors and angles
    u_sun = sat2sun.flatten()/np.linalg.norm(sat2sun)
    u_stat = sat2stat.flatten()/np.linalg.norm(sat2stat)
    u_stat2sun = stat2sun.flatten()/np.linalg.norm(stat2sun)
    u_sat = r_eci.flatten()/np.linalg.norm(r_eci)
    u_moon = moon2stat.flatten()/np.linalg.norm(moon2stat)

    phase_angle = acos(np.dot(u_sun, u_stat))
    phase_angle_stat = acos(np.dot(u_stat2sun, -u_stat))
    sun_angle = acos(np.dot(u_sun, -u_sat))
    moon_angle = acos(np.dot(u_moon, u_stat))

    return phase_angle, phase_angle_stat, sun_angle, moon_angle


def compute_mapp(phase_angle, sat_rg, sat_radius, albedo=0.1):
    '''
    This function computes the apparent magnitude of a space object
    due to reflected sunlight.

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
    RCS obtained from SATCAT. Assumes diffuse sphere satellite.

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


def station_dark(sun_eci, sensor, inputs, JED_JD):
    '''
    This function determines if the station is dark (not sunlit).

    Parameters
    ------
    sun_eci : 3x1 numpy array
        sun position in ECI [km]
    sensor : dictionary
        sensor parameters including ground station position in ECEF
    inputs : dictionary
        parameters for frame rotations
    JED_JD : float
        current time [JED] in julian date format

    Returns
    ------
    isDark : int
        boolean-like descriptor of station status (0 = sunlit, 1 = dark)
    '''

    # Compute sun and station position vectors in ECEF
    sun_ecef = conv.eci2ecef(sun_eci, inputs, JED_JD)
    stat_ecef = sensor['stat_ecef']

    # Compute sun elevation angle
    az, el, rg = conv.ecef2azelrange(sun_ecef, stat_ecef)

    isDark = 1
    if el > sensor['sun_elmask']:
        isDark = 0

    return isDark


def compute_sun_eci(UTC):
    '''
    This function computes the current sun position in ECI using the PyEphem
    module.

    Parameters
    ------
    UTC : tuple
        Current time in UTC (yyyy, mm, dd, hh, mm, ss)

    Returns
    ------
    sun_eci : 3x1 numpy array
        sun position in ECI [km]
    '''

    # Set up sun object
    sun = ephem.Sun()

    # Get angles and position
    sun.compute(UTC)
    ra = sun.a_ra*1.
    dec = sun.a_dec*1.
    dist = sun.earth_distance*149597870.700
    
    print( 'ra', ra*180/pi )
    print( 'dec', dec*180/pi )
    print( 'dist', dist )

    sun_eci = dist * np.array([[cos(ra)*cos(dec)], [sin(ra)*cos(dec)],
                               [sin(dec)]])

    return sun_eci


def compute_moon_eci(UTC):
    '''
    This function computes the current moon position in ECI using the PyEphem
    module.

    Parameters
    ------
    UTC : tuple
        Current time in UTC (yyyy, mm, dd, hh, mm, ss)

    Returns
    ------
    moon_eci : 3x1 numpy array
        moon position in ECI [km]
    '''
    # Set up moon object
    moon = ephem.Moon()

    # Get angles and position
    moon.compute(UTC)
    ra = moon.a_ra*1.
    dec = moon.a_dec*1.
    dist = moon.earth_distance*149597870.700

    moon_eci = dist * np.array([[cos(ra)*cos(dec)], [sin(ra)*cos(dec)],
                                [sin(dec)]])

    return moon_eci


