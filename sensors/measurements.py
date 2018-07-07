import numpy as np
from math import asin, atan2


from utilities.coordinate_systems import latlonht2ecef
from utilities.coordinate_systems import itrf2gcrf
from utilities.coordinate_systems import gcrf2itrf
from utilities.coordinate_systems import ecef2enu


def compute_measurement(X, sun_gcrf, sensor, spacecraftConfig, surface, UTC,
                        EOP_data, meas_types=[]):
    
    # Retrieve sensor parameters
    if len(meas_types) == 0:
        meas_types = sensor['meas_types']
    geodetic_latlonht = sensor['geodetic_latlonht']
    
    # Compute station location in GCRF
    lat = geodetic_latlonht[0]
    lon = geodetic_latlonht[1]
    ht = geodetic_latlonht[2]
    stat_itrf = latlonht2ecef(lat, lon, ht)
    stat_gcrf, dum = itrf2gcrf(stat_itrf, np.zeros(3,1), UTC, EOP_data)
    
    # Object location in GCRF
    r_gcrf = X[0:3].reshape(3,1)
    
    # Compute range and line of sight vector
    rg = np.linalg.norm(r_gcrf - stat_gcrf)
    rho_hat_gcrf = r_gcrf/rg
    
    # Rotate to ENU frame
    rho_hat_itrf = gcrf2itrf(rho_hat_gcrf, np.zeros(3,1), UTC, EOP_data)
    rho_hat_enu = ecef2enu(rho_hat_itrf, stat_itrf)
    
    # Loop over measurement types
    Y = np.zeros(len(meas_types),1)
    ii = 0
    for mtype in meas_types:
        
        if mtype == 'rg':
            Y[ii] = rg  # km
            
        elif mtype == 'ra':
            Y[ii] = atan2(rho_hat_gcrf[1], rho_hat_gcrf[0]) #rad
            
        elif mtype == 'dec':
            Y[ii] = asin(rho_hat_gcrf[2])  #rad
    
        elif mtype == 'az':
            Y[ii] = atan2(rho_hat_enu[0], rho_hat_enu[1])  # rad  
            
        elif mtype == 'el':
            Y[ii] = asin(rho_hat_enu[2])  # rad
            
        elif mtype == 'mapp':
            
            sat2sun = sun_gcrf - r_gcrf
            sat2obs = stat_gcrf - r_gcrf
            if spacecraftConfig['type'] == '3DoF':
                Y[ii] = compute_mapp(sat2sun, sat2obs, spacecraftConfig, surfaces)
            
        else:
            print('Invalid Measurement Type! Entered: ', mtype)
            
        ii += 1
    
    return Y


def compute_mapp():
    
    
    return


def ecef2azelrange(r_sat, r_site):
    '''
    This function computes the azimuth, elevation, and range of a satellite
    from a given ground station, all position in ECEF.

    Parameters
    ------
    r_sat : 3x1 numpy array
      satellite position vector in ECEF [km]
    r_site : 3x1 numpy array
      ground station position vector in ECEF [km]

    Returns
    ------
    az : float
      azimuth, degrees clockwise from north [deg]
    el : float
      elevation, degrees up from horizon [deg]
    rg : float
      scalar distance from site to sat [km]
    '''

    # Compute vector from site to satellite and range
    rho_ecef = r_sat - r_site
    rg = np.linalg.norm(rho_ecef)  # km

    # Compute unit vector in LOS direction from site to sat
    rho_hat_ecef = rho_ecef/rg

    # Rotate to ENU
    rho_hat_enu = ecef2enu(rho_hat_ecef, r_site)

    # Get components
    rho_x = float(rho_hat_enu[0])
    rho_y = float(rho_hat_enu[1])
    rho_z = float(rho_hat_enu[2])

    # Compute Azimuth and Elevation
    el = asin(rho_z) * 180/pi  # deg
    az = atan2(rho_x, rho_y) * 180/pi  # deg

    # Convert az to range 0-360
    if az < 0:
        az = az + 360

    return az, el, rg


def ecef2azelrange_rad(r_sat, r_site):
    '''
    This function computes the azimuth, elevation, and range of a satellite
    from a given ground station, all position in ECEF.

    Parameters
    ------
    r_sat : 3x1 numpy array
      satellite position vector in ECEF [km]
    r_site : 3x1 numpy array
      ground station position vector in ECEF [km]

    Returns
    ------
    az : float
      azimuth, clockwise from north [0-2pi] [rad]
    el : float
      elevation, up from horizon [0-pi/2] [rad]
    rg : float
      scalar distance from site to sat [km]
    '''

    # Compute vector from site to satellite and range
    rho_ecef = r_sat - r_site
    rg = np.linalg.norm(rho_ecef)  # km

    # Compute unit vector in LOS direction from site to sat
    rho_hat_ecef = rho_ecef/rg

    # Rotate to ENU
    rho_hat_enu = ecef2enu(rho_hat_ecef, r_site)

    # Get components
    rho_x = float(rho_hat_enu[0])
    rho_y = float(rho_hat_enu[1])
    rho_z = float(rho_hat_enu[2])

    # Compute Azimuth and Elevation
    el = asin(rho_z)  # rad
    az = atan2(rho_x, rho_y)  # rad

    # Convert az to range 0-2*pi
    if az < 0:
        az += 2*pi

    return az, el, rg