import numpy as np
from math import pi, asin, atan2
import sys

sys.path.append('../')

from sensors.brdf_models import compute_mapp
from utilities.coordinate_systems import latlonht2ecef
from utilities.coordinate_systems import itrf2gcrf
from utilities.coordinate_systems import gcrf2itrf
from utilities.coordinate_systems import ecef2enu


def compute_measurement(X, state_params, sensor, UTC, EOP_data, XYs_df=[], meas_types=[],
                        sun_gcrf=[]):
    
    # Retrieve measurement types
    if len(meas_types) == 0:
        meas_types = sensor['meas_types']
    
    # Compute station location in GCRF
    sensor_itrf = sensor['site_ecef']
    sensor_gcrf, dum = itrf2gcrf(sensor_itrf, np.zeros((3,1)), UTC, EOP_data,
                                 XYs_df)
    
    # Object location in GCRF
    r_gcrf = X[0:3].reshape(3,1)
    
    # Compute range and line of sight vector
    rg = np.linalg.norm(r_gcrf - sensor_gcrf)
    rho_hat_gcrf = (r_gcrf - sensor_gcrf)/rg
    
    # Rotate to ENU frame
    rho_hat_itrf, dum = gcrf2itrf(rho_hat_gcrf, np.zeros((3,1)), UTC, EOP_data,
                                  XYs_df)
    rho_hat_enu = ecef2enu(rho_hat_itrf, sensor_itrf)
    
    # Loop over measurement types
    Y = np.zeros((len(meas_types),1))
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
            if Y[ii] < 0.:
                Y[ii] += 2.*pi
            
        elif mtype == 'el':
            Y[ii] = asin(rho_hat_enu[2])  # rad
            
        elif mtype == 'mapp':
            
            Y[ii] = 1.
            
#            sat2sun = sun_gcrf - r_gcrf
#            sat2obs = stat_gcrf - r_gcrf
#            if spacecraftConfig['type'] == '3DoF':
#                mapp = compute_mapp(sat2sun, sat2obs, spacecraftConfig, surfaces)                
#                Y[ii] = mapp
#               
#                    
#            elif spacecraftConfig['type'] == '6DoF' or spacecraftConfig['type'] == '3att':
#                q_BI = X[6:10].reshape(4,1)                
#                mapp = compute_mapp(sat2sun, sat2obs, spacecraftConfig, surfaces, q_BI)                
#                Y[ii] = mapp
                
        else:
            print('Invalid Measurement Type! Entered: ', mtype)
            
        ii += 1
    
    return Y



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
      azimuth, degrees clockwise from north [0 - 360 deg]
    el : float
      elevation, degrees up from horizon [-90 - 90 deg]
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
      azimuth, clockwise from north [0 - 2pi rad]
    el : float
      elevation, up from horizon [-pi/2 - pi/2 rad]
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