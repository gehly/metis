import numpy as np
from math import pi
import os
import pickle
import pandas as pd
import sys
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

import utilities.coordinate_systems as coord


def define_sensors(sensor_id_list=[]):
    
    # Initialize
    sensor_dict = {}
    
    arcsec2rad = pi/(3600.*180.)
    
    # Set up sensors
    print('RMIT ROO')
    
    # FOV dimensions
    LAM_dim = 0.73   # deg
    PHI_dim = 0.59   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_dim*pi/180
    PHI_half = 0.5*PHI_dim*pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
    
    # Constraints/Limits
    az_lim = [0., 2.*pi]  # rad
    el_lim = [10.*pi/180., pi/2.]  # rad
    rg_lim = [0., 1e6]   # km
    mapp_lim = 16.
    moon_angle_lim = 0.32  # rad
    sun_el_mask = -10.*pi/180.  # rad
    
    # Measurement types and noise
    meas_types = ['ra', 'dec', 'mapp']
    sigma_dict = {}
    sigma_dict['ra'] = 5.*arcsec2rad   # rad
    sigma_dict['dec'] = 5.*arcsec2rad  # rad
    sigma_dict['mapp'] = 0.1
    
    # Pass parameters
    max_pass = 6000.
    min_pass = 60.
    max_gap = 60.
    obs_gap = 1.
    
    # Station Coordinates
    lat_gs = -37.68
    lon_gs = 145.06
    ht_gs = 0.1724 # km	
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Location and constraints
    sensor_dict['RMIT ROO'] = {}
    sensor_dict['RMIT ROO']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['RMIT ROO']['site_ecef'] = site_ecef
    sensor_dict['RMIT ROO']['mapp_lim'] = mapp_lim
    sensor_dict['RMIT ROO']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['RMIT ROO']['el_lim'] = el_lim
    sensor_dict['RMIT ROO']['az_lim'] = az_lim
    sensor_dict['RMIT ROO']['rg_lim'] = rg_lim
    sensor_dict['RMIT ROO']['FOV_hlim'] = FOV_hlim
    sensor_dict['RMIT ROO']['FOV_vlim'] = FOV_vlim
    sensor_dict['RMIT ROO']['sun_elmask'] = sun_el_mask
    sensor_dict['RMIT ROO']['passive_optical'] = True
    
    # Measurements and noise
    sensor_dict['RMIT ROO']['meas_types'] = meas_types
    sensor_dict['RMIT ROO']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['RMIT ROO']['max_gap'] = max_gap
    sensor_dict['RMIT ROO']['obs_gap'] = obs_gap
    sensor_dict['RMIT ROO']['min_pass'] = min_pass
    sensor_dict['RMIT ROO']['max_pass'] = max_pass


    print('UNSW Viper')
    
    # FOV dimensions
    LAM_dim = 4.0   # deg
    PHI_dim = 4.0   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_dim*pi/180
    PHI_half = 0.5*PHI_dim*pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
    
    # Constraints/Limits
    az_lim = [0., 2.*pi]  # rad
    el_lim = [10.*pi/180., pi/2.]  # rad
    rg_lim = [0., 1e6]   # km
    mapp_lim = 16.
    moon_angle_lim = 0.32  # rad
    sun_el_mask = -10.*pi/180.  # rad
    
    # Measurement types and noise
    meas_types = ['ra', 'dec', 'mapp']
    sigma_dict = {}
    sigma_dict['ra'] = 5.*arcsec2rad   # rad
    sigma_dict['dec'] = 5.*arcsec2rad  # rad
    sigma_dict['mapp'] = 0.1
    
    # Pass parameters
    max_pass = 6000.
    min_pass = 60.
    max_gap = 60.
    obs_gap = 1.
    
    # Station Coordinates
    lat_gs = -34.74
    lon_gs = 148.84
    ht_gs = 0.570 # km	
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Location and constraints
    sensor_dict['UNSW Viper'] = {}
    sensor_dict['UNSW Viper']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['UNSW Viper']['site_ecef'] = site_ecef
    sensor_dict['UNSW Viper']['mapp_lim'] = mapp_lim
    sensor_dict['UNSW Viper']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['UNSW Viper']['el_lim'] = el_lim
    sensor_dict['UNSW Viper']['az_lim'] = az_lim
    sensor_dict['UNSW Viper']['rg_lim'] = rg_lim
    sensor_dict['UNSW Viper']['FOV_hlim'] = FOV_hlim
    sensor_dict['UNSW Viper']['FOV_vlim'] = FOV_vlim
    sensor_dict['UNSW Viper']['sun_elmask'] = sun_el_mask
    sensor_dict['UNSW Viper']['passive_optical'] = True
    
    # Measurements and noise
    sensor_dict['UNSW Viper']['meas_types'] = meas_types
    sensor_dict['UNSW Viper']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['UNSW Viper']['max_gap'] = max_gap
    sensor_dict['UNSW Viper']['obs_gap'] = obs_gap
    sensor_dict['UNSW Viper']['min_pass'] = min_pass
    sensor_dict['UNSW Viper']['max_pass'] = max_pass


    # Falcon Telescopes
    # Common Parameters
    
    # FOV dimensions
    LAM_dim = 30./60.   # deg
    PHI_dim = 30./60.   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_dim*pi/180
    PHI_half = 0.5*PHI_dim*pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
    
    # Constraints/Limits
    az_lim = [0., 2.*pi]  # rad
    el_lim = [10.*pi/180., pi/2.]  # rad
    rg_lim = [0., 1e6]   # km
    mapp_lim = 16.
    moon_angle_lim = 0.32  # rad
    sun_el_mask = -10.*pi/180.  # rad
    
    # Measurement types and noise
    meas_types = ['ra', 'dec', 'mapp']
    sigma_dict = {}
    sigma_dict['ra'] = 5.*arcsec2rad   # rad
    sigma_dict['dec'] = 5.*arcsec2rad  # rad
    sigma_dict['mapp'] = 0.1
    
    # Pass parameters
    max_pass = 6000.
    min_pass = 60.
    max_gap = 60.
    obs_gap = 1.
    
    
    print('UNSW Falcon')
    lat_gs = -35.29
    lon_gs = 149.17
    ht_gs = 0.606 # km	
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Location and constraints
    sensor_dict['UNSW Falcon'] = {}
    sensor_dict['UNSW Falcon']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['UNSW Falcon']['site_ecef'] = site_ecef
    sensor_dict['UNSW Falcon']['mapp_lim'] = mapp_lim
    sensor_dict['UNSW Falcon']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['UNSW Falcon']['el_lim'] = el_lim
    sensor_dict['UNSW Falcon']['az_lim'] = az_lim
    sensor_dict['UNSW Falcon']['rg_lim'] = rg_lim
    sensor_dict['UNSW Falcon']['FOV_hlim'] = FOV_hlim
    sensor_dict['UNSW Falcon']['FOV_vlim'] = FOV_vlim
    sensor_dict['UNSW Falcon']['sun_elmask'] = sun_el_mask
    sensor_dict['UNSW Falcon']['passive_optical'] = True
    
    # Measurements and noise
    sensor_dict['UNSW Falcon']['meas_types'] = meas_types
    sensor_dict['UNSW Falcon']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['UNSW Falcon']['max_gap'] = max_gap
    sensor_dict['UNSW Falcon']['obs_gap'] = obs_gap
    sensor_dict['UNSW Falcon']['min_pass'] = min_pass
    sensor_dict['UNSW Falcon']['max_pass'] = max_pass


    print('USAFA Falcon')
    lat_gs = 39.01
    lon_gs = 255.01
    ht_gs = 2.805 # km	
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Location and constraints
    sensor_dict['USAFA Falcon'] = {}
    sensor_dict['USAFA Falcon']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['USAFA Falcon']['site_ecef'] = site_ecef
    sensor_dict['USAFA Falcon']['mapp_lim'] = mapp_lim
    sensor_dict['USAFA Falcon']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['USAFA Falcon']['el_lim'] = el_lim
    sensor_dict['USAFA Falcon']['az_lim'] = az_lim
    sensor_dict['USAFA Falcon']['rg_lim'] = rg_lim
    sensor_dict['USAFA Falcon']['FOV_hlim'] = FOV_hlim
    sensor_dict['USAFA Falcon']['FOV_vlim'] = FOV_vlim
    sensor_dict['USAFA Falcon']['sun_elmask'] = sun_el_mask
    sensor_dict['USAFA Falcon']['passive_optical'] = True
    
    # Measurements and noise
    sensor_dict['USAFA Falcon']['meas_types'] = meas_types
    sensor_dict['USAFA Falcon']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['USAFA Falcon']['max_gap'] = max_gap
    sensor_dict['USAFA Falcon']['obs_gap'] = obs_gap
    sensor_dict['USAFA Falcon']['min_pass'] = min_pass
    sensor_dict['USAFA Falcon']['max_pass'] = max_pass
    
    	
    print('FLC Falcon')
    lat_gs = 37.23
    lon_gs = 251.93
    ht_gs = 1.969 # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Location and constraints
    sensor_dict['FLC Falcon'] = {}
    sensor_dict['FLC Falcon']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['FLC Falcon']['site_ecef'] = site_ecef
    sensor_dict['FLC Falcon']['mapp_lim'] = mapp_lim
    sensor_dict['FLC Falcon']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['FLC Falcon']['el_lim'] = el_lim
    sensor_dict['FLC Falcon']['az_lim'] = az_lim
    sensor_dict['FLC Falcon']['rg_lim'] = rg_lim
    sensor_dict['FLC Falcon']['FOV_hlim'] = FOV_hlim
    sensor_dict['FLC Falcon']['FOV_vlim'] = FOV_vlim
    sensor_dict['FLC Falcon']['sun_elmask'] = sun_el_mask
    sensor_dict['FLC Falcon']['passive_optical'] = True
    
    # Measurements and noise
    sensor_dict['FLC Falcon']['meas_types'] = meas_types
    sensor_dict['FLC Falcon']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['FLC Falcon']['max_gap'] = max_gap
    sensor_dict['FLC Falcon']['obs_gap'] = obs_gap
    sensor_dict['FLC Falcon']['min_pass'] = min_pass
    sensor_dict['FLC Falcon']['max_pass'] = max_pass

    print('CMU Falcon')
    
    # From Chun paper
#    lat_gs = 39.96
#    lon_gs = 251.76
#    ht_gs = 1.380 # km
    
    # From TheSkyX at site
    lat_gs = 38. + 57./60. + 48.12/3600.       # deg
    lon_gs = -(108. + 14./60. + 15.84/3600.)   # deg
    ht_gs = 1.86116    # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Location and constraints
    sensor_dict['CMU Falcon'] = {}
    sensor_dict['CMU Falcon']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['CMU Falcon']['site_ecef'] = site_ecef
    sensor_dict['CMU Falcon']['mapp_lim'] = mapp_lim
    sensor_dict['CMU Falcon']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['CMU Falcon']['el_lim'] = el_lim
    sensor_dict['CMU Falcon']['az_lim'] = az_lim
    sensor_dict['CMU Falcon']['rg_lim'] = rg_lim
    sensor_dict['CMU Falcon']['FOV_hlim'] = FOV_hlim
    sensor_dict['CMU Falcon']['FOV_vlim'] = FOV_vlim
    sensor_dict['CMU Falcon']['sun_elmask'] = sun_el_mask
    sensor_dict['CMU Falcon']['passive_optical'] = True
    
    # Measurements and noise
    sensor_dict['CMU Falcon']['meas_types'] = meas_types
    sensor_dict['CMU Falcon']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['CMU Falcon']['max_gap'] = max_gap
    sensor_dict['CMU Falcon']['obs_gap'] = obs_gap
    sensor_dict['CMU Falcon']['min_pass'] = min_pass
    sensor_dict['CMU Falcon']['max_pass'] = max_pass
	
    print('NJC Falcon')
    lat_gs = 40.65
    lon_gs = 256.80
    ht_gs = 1.177 # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Location and constraints
    sensor_dict['NJC Falcon'] = {}
    sensor_dict['NJC Falcon']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['NJC Falcon']['site_ecef'] = site_ecef
    sensor_dict['NJC Falcon']['mapp_lim'] = mapp_lim
    sensor_dict['NJC Falcon']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['NJC Falcon']['el_lim'] = el_lim
    sensor_dict['NJC Falcon']['az_lim'] = az_lim
    sensor_dict['NJC Falcon']['rg_lim'] = rg_lim
    sensor_dict['NJC Falcon']['FOV_hlim'] = FOV_hlim
    sensor_dict['NJC Falcon']['FOV_vlim'] = FOV_vlim
    sensor_dict['NJC Falcon']['sun_elmask'] = sun_el_mask
    sensor_dict['NJC Falcon']['passive_optical'] = True
    
    # Measurements and noise
    sensor_dict['NJC Falcon']['meas_types'] = meas_types
    sensor_dict['NJC Falcon']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['NJC Falcon']['max_gap'] = max_gap
    sensor_dict['NJC Falcon']['obs_gap'] = obs_gap
    sensor_dict['NJC Falcon']['min_pass'] = min_pass
    sensor_dict['NJC Falcon']['max_pass'] = max_pass
	
    print('OJC Falcon')
    lat_gs = 37.97
    lon_gs = 256.46
    ht_gs = 1.221 # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Location and constraints
    sensor_dict['OJC Falcon'] = {}
    sensor_dict['OJC Falcon']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['OJC Falcon']['site_ecef'] = site_ecef
    sensor_dict['OJC Falcon']['mapp_lim'] = mapp_lim
    sensor_dict['OJC Falcon']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['OJC Falcon']['el_lim'] = el_lim
    sensor_dict['OJC Falcon']['az_lim'] = az_lim
    sensor_dict['OJC Falcon']['rg_lim'] = rg_lim
    sensor_dict['OJC Falcon']['FOV_hlim'] = FOV_hlim
    sensor_dict['OJC Falcon']['FOV_vlim'] = FOV_vlim
    sensor_dict['OJC Falcon']['sun_elmask'] = sun_el_mask
    sensor_dict['OJC Falcon']['passive_optical'] = True
    
    # Measurements and noise
    sensor_dict['OJC Falcon']['meas_types'] = meas_types
    sensor_dict['OJC Falcon']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['OJC Falcon']['max_gap'] = max_gap
    sensor_dict['OJC Falcon']['obs_gap'] = obs_gap
    sensor_dict['OJC Falcon']['min_pass'] = min_pass
    sensor_dict['OJC Falcon']['max_pass'] = max_pass
	
    print('PSU Falcon')
    lat_gs = 40.86
    lon_gs = 282.17
    ht_gs = 0.347 # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]	
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Location and constraints
    sensor_dict['PSU Falcon'] = {}
    sensor_dict['PSU Falcon']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['PSU Falcon']['site_ecef'] = site_ecef
    sensor_dict['PSU Falcon']['mapp_lim'] = mapp_lim
    sensor_dict['PSU Falcon']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['PSU Falcon']['el_lim'] = el_lim
    sensor_dict['PSU Falcon']['az_lim'] = az_lim
    sensor_dict['PSU Falcon']['rg_lim'] = rg_lim
    sensor_dict['PSU Falcon']['FOV_hlim'] = FOV_hlim
    sensor_dict['PSU Falcon']['FOV_vlim'] = FOV_vlim
    sensor_dict['PSU Falcon']['sun_elmask'] = sun_el_mask
    sensor_dict['PSU Falcon']['passive_optical'] = True
    
    # Measurements and noise
    sensor_dict['PSU Falcon']['meas_types'] = meas_types
    sensor_dict['PSU Falcon']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['PSU Falcon']['max_gap'] = max_gap
    sensor_dict['PSU Falcon']['obs_gap'] = obs_gap
    sensor_dict['PSU Falcon']['min_pass'] = min_pass
    sensor_dict['PSU Falcon']['max_pass'] = max_pass
	
    print('Mamalluca Falcon')
    lat_gs = -29.99
    lon_gs = 289.32
    ht_gs = 1.139 # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)

    # Location and constraints
    sensor_dict['Mamalluca Falcon'] = {}
    sensor_dict['Mamalluca Falcon']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['Mamalluca Falcon']['site_ecef'] = site_ecef
    sensor_dict['Mamalluca Falcon']['mapp_lim'] = mapp_lim
    sensor_dict['Mamalluca Falcon']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['Mamalluca Falcon']['el_lim'] = el_lim
    sensor_dict['Mamalluca Falcon']['az_lim'] = az_lim
    sensor_dict['Mamalluca Falcon']['rg_lim'] = rg_lim
    sensor_dict['Mamalluca Falcon']['FOV_hlim'] = FOV_hlim
    sensor_dict['Mamalluca Falcon']['FOV_vlim'] = FOV_vlim
    sensor_dict['Mamalluca Falcon']['sun_elmask'] = sun_el_mask
    sensor_dict['Mamalluca Falcon']['passive_optical'] = True
    
    # Measurements and noise
    sensor_dict['Mamalluca Falcon']['meas_types'] = meas_types
    sensor_dict['Mamalluca Falcon']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['Mamalluca Falcon']['max_gap'] = max_gap
    sensor_dict['Mamalluca Falcon']['obs_gap'] = obs_gap
    sensor_dict['Mamalluca Falcon']['min_pass'] = min_pass
    sensor_dict['Mamalluca Falcon']['max_pass'] = max_pass
    
    
    print('Perth Falcon')
    lat_gs = -31.95
    lon_gs = 115.86
    ht_gs = 0. # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Location and constraints
    sensor_dict['Perth Falcon'] = {}
    sensor_dict['Perth Falcon']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['Perth Falcon']['site_ecef'] = site_ecef
    sensor_dict['Perth Falcon']['mapp_lim'] = mapp_lim
    sensor_dict['Perth Falcon']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['Perth Falcon']['el_lim'] = el_lim
    sensor_dict['Perth Falcon']['az_lim'] = az_lim
    sensor_dict['Perth Falcon']['rg_lim'] = rg_lim
    sensor_dict['Perth Falcon']['FOV_hlim'] = FOV_hlim
    sensor_dict['Perth Falcon']['FOV_vlim'] = FOV_vlim
    sensor_dict['Perth Falcon']['sun_elmask'] = sun_el_mask
    sensor_dict['Perth Falcon']['passive_optical'] = True
    
    # Measurements and noise
    sensor_dict['Perth Falcon']['meas_types'] = meas_types
    sensor_dict['Perth Falcon']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['Perth Falcon']['max_gap'] = max_gap
    sensor_dict['Perth Falcon']['obs_gap'] = obs_gap
    sensor_dict['Perth Falcon']['min_pass'] = min_pass
    sensor_dict['Perth Falcon']['max_pass'] = max_pass



    print('Stromlo Laser')
    lat_gs = -35.3161
    lon_gs = 149.0099
    ht_gs = 0.805  # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Constraints/Limits
    az_lim = [0., 2.*pi]  # rad
    el_lim = [20.*pi/180., 80.*pi/180.]  # rad
    rg_lim = [0., 10000.]   # km
    
    # FOV dimensions
    LAM_dim = 0.5   # deg
    PHI_dim = 0.5   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_dim*pi/180
    PHI_half = 0.5*PHI_dim*pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
   
    # Measurement types and noise
    meas_types = ['rg', 'az', 'el']
    sigma_dict = {}
    sigma_dict['rg'] = 0.001  # km
    sigma_dict['az'] = 5.*arcsec2rad   # rad
    sigma_dict['el'] = 5.*arcsec2rad  # rad
    
    # Pass parameters
    max_pass = 6000.
    min_pass = 60.
    max_gap = 60.
    obs_gap = 1.
    
    # Location and constraints
    sensor_dict['Stromlo Laser'] = {}
    sensor_dict['Stromlo Laser']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['Stromlo Laser']['site_ecef'] = site_ecef
    sensor_dict['Stromlo Laser']['el_lim'] = el_lim
    sensor_dict['Stromlo Laser']['az_lim'] = az_lim
    sensor_dict['Stromlo Laser']['rg_lim'] = rg_lim
    sensor_dict['Stromlo Laser']['FOV_hlim'] = FOV_hlim
    sensor_dict['Stromlo Laser']['FOV_vlim'] = FOV_vlim
    sensor_dict['Stromlo Laser']['laser_output'] = 1.  # Watts
    sensor_dict['Stromlo Laser']['passive_optical'] = False
    
    # Measurements and noise
    sensor_dict['Stromlo Laser']['meas_types'] = meas_types
    sensor_dict['Stromlo Laser']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['Stromlo Laser']['max_gap'] = max_gap
    sensor_dict['Stromlo Laser']['obs_gap'] = obs_gap
    sensor_dict['Stromlo Laser']['min_pass'] = min_pass
    sensor_dict['Stromlo Laser']['max_pass'] = max_pass


    print('Stromlo Optical')
    lat_gs = -35.3161
    lon_gs = 149.0099
    ht_gs = 0.805  # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Constraints/Limits
    az_lim = [0., 2.*pi]  # rad
    el_lim = [10.*pi/180., 80.*pi/180.]  # rad
    rg_lim = [0., 1e6]   # km
    mapp_lim = 16.
    sun_el_mask = -10.*pi/180.
    
    # FOV dimensions
    LAM_dim = 0.5   # deg
    PHI_dim = 0.5   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_dim*pi/180
    PHI_half = 0.5*PHI_dim*pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
   
    # Measurement types and noise
    meas_types = ['az', 'el']
    sigma_dict = {}
    sigma_dict['az'] = 5.*arcsec2rad   # rad
    sigma_dict['el'] = 5.*arcsec2rad  # rad
    
    # Pass parameters
    max_pass = 6000.
    min_pass = 60.
    max_gap = 60.
    obs_gap = 1.
    
    # Location and constraints
    sensor_dict['Stromlo Optical'] = {}
    sensor_dict['Stromlo Optical']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['Stromlo Optical']['site_ecef'] = site_ecef
    sensor_dict['Stromlo Optical']['mapp_lim'] = mapp_lim
    sensor_dict['Stromlo Optical']['sun_elmask'] = sun_el_mask
    sensor_dict['Stromlo Optical']['el_lim'] = el_lim
    sensor_dict['Stromlo Optical']['az_lim'] = az_lim
    sensor_dict['Stromlo Optical']['rg_lim'] = rg_lim
    sensor_dict['Stromlo Optical']['FOV_hlim'] = FOV_hlim
    sensor_dict['Stromlo Optical']['FOV_vlim'] = FOV_vlim
    sensor_dict['Stromlo Optical']['passive_optical'] = True
    
    # Measurements and noise
    sensor_dict['Stromlo Optical']['meas_types'] = meas_types
    sensor_dict['Stromlo Optical']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['Stromlo Optical']['max_gap'] = max_gap
    sensor_dict['Stromlo Optical']['obs_gap'] = obs_gap
    sensor_dict['Stromlo Optical']['min_pass'] = min_pass
    sensor_dict['Stromlo Optical']['max_pass'] = max_pass


    print('Zimmerwald Laser')
    lat_gs = 46.8772
    lon_gs = 7.4652
    ht_gs = 0.9512  # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Constraints/Limits
    az_lim = [0., 2.*pi]  # rad
    el_lim = [20.*pi/180., 80.*pi/180.]  # rad
    rg_lim = [0., 10000.]   # km
    
    # FOV dimensions
    LAM_dim = 0.5   # deg
    PHI_dim = 0.5   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_dim*pi/180
    PHI_half = 0.5*PHI_dim*pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
   
    # Measurement types and noise
    meas_types = ['rg', 'az', 'el']
    sigma_dict = {}
    sigma_dict['rg'] = 0.001  # km
    sigma_dict['az'] = 5.*arcsec2rad   # rad
    sigma_dict['el'] = 5.*arcsec2rad  # rad
    
    # Pass parameters
    max_pass = 6000.
    min_pass = 60.
    max_gap = 60.
    obs_gap = 1.
    
    # Location and constraints
    sensor_dict['Zimmerwald Laser'] = {}
    sensor_dict['Zimmerwald Laser']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['Zimmerwald Laser']['site_ecef'] = site_ecef
    sensor_dict['Zimmerwald Laser']['el_lim'] = el_lim
    sensor_dict['Zimmerwald Laser']['az_lim'] = az_lim
    sensor_dict['Zimmerwald Laser']['rg_lim'] = rg_lim
    sensor_dict['Zimmerwald Laser']['FOV_hlim'] = FOV_hlim
    sensor_dict['Zimmerwald Laser']['FOV_vlim'] = FOV_vlim
    sensor_dict['Zimmerwald Laser']['laser_output'] = 1.  # Watts
    sensor_dict['Zimmerwald Laser']['passive_optical'] = False
    
    # Measurements and noise
    sensor_dict['Zimmerwald Laser']['meas_types'] = meas_types
    sensor_dict['Zimmerwald Laser']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['Zimmerwald Laser']['max_gap'] = max_gap
    sensor_dict['Zimmerwald Laser']['obs_gap'] = obs_gap
    sensor_dict['Zimmerwald Laser']['min_pass'] = min_pass
    sensor_dict['Zimmerwald Laser']['max_pass'] = max_pass


    print('Zimmerwald Optical')
    lat_gs = 46.8772
    lon_gs = 7.4652
    ht_gs = 0.9512  # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Constraints/Limits
    az_lim = [0., 2.*pi]  # rad
    el_lim = [20.*pi/180., 80.*pi/180.]  # rad
    rg_lim = [0., 1e6]   # km
    mapp_lim = 16.
    sun_el_mask = -10.*pi/180.
    
    # FOV dimensions
    LAM_dim = 0.5   # deg
    PHI_dim = 0.5   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_dim*pi/180
    PHI_half = 0.5*PHI_dim*pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
   
    # Measurement types and noise
    meas_types = ['az', 'el']
    sigma_dict = {}
    sigma_dict['az'] = 5.*arcsec2rad   # rad
    sigma_dict['el'] = 5.*arcsec2rad  # rad
    
    # Pass parameters
    max_pass = 6000.
    min_pass = 60.
    max_gap = 60.
    obs_gap = 1.
    
    # Location and constraints
    sensor_dict['Zimmerwald Optical'] = {}
    sensor_dict['Zimmerwald Optical']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['Zimmerwald Optical']['site_ecef'] = site_ecef
    sensor_dict['Zimmerwald Optical']['mapp_lim'] = mapp_lim
    sensor_dict['Zimmerwald Optical']['sun_elmask'] = sun_el_mask
    sensor_dict['Zimmerwald Optical']['el_lim'] = el_lim
    sensor_dict['Zimmerwald Optical']['az_lim'] = az_lim
    sensor_dict['Zimmerwald Optical']['rg_lim'] = rg_lim
    sensor_dict['Zimmerwald Optical']['FOV_hlim'] = FOV_hlim
    sensor_dict['Zimmerwald Optical']['FOV_vlim'] = FOV_vlim
    sensor_dict['Zimmerwald Optical']['passive_optical'] = True
    
    # Measurements and noise
    sensor_dict['Zimmerwald Optical']['meas_types'] = meas_types
    sensor_dict['Zimmerwald Optical']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['Zimmerwald Optical']['max_gap'] = max_gap
    sensor_dict['Zimmerwald Optical']['obs_gap'] = obs_gap
    sensor_dict['Zimmerwald Optical']['min_pass'] = min_pass
    sensor_dict['Zimmerwald Optical']['max_pass'] = max_pass


    print('Arequipa Laser')
    lat_gs = -16.4657
    lon_gs =  -71.4930
    ht_gs = 2.48905  # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Constraints/Limits
    az_lim = [0., 2.*pi]  # rad
    el_lim = [20.*pi/180., 80.*pi/180.]  # rad
    rg_lim = [0., 10000.]   # km
    
    # FOV dimensions
    LAM_dim = 0.5   # deg
    PHI_dim = 0.5   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_dim*pi/180
    PHI_half = 0.5*PHI_dim*pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
   
    # Measurement types and noise
    meas_types = ['rg', 'az', 'el']
    sigma_dict = {}
    sigma_dict['rg'] = 0.001  # km
    sigma_dict['az'] = 5.*arcsec2rad   # rad
    sigma_dict['el'] = 5.*arcsec2rad  # rad
    
    # Pass parameters
    max_pass = 6000.
    min_pass = 60.
    max_gap = 60.
    obs_gap = 1.
    
    # Location and constraints
    sensor_dict['Arequipa Laser'] = {}
    sensor_dict['Arequipa Laser']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['Arequipa Laser']['site_ecef'] = site_ecef
    sensor_dict['Arequipa Laser']['el_lim'] = el_lim
    sensor_dict['Arequipa Laser']['az_lim'] = az_lim
    sensor_dict['Arequipa Laser']['rg_lim'] = rg_lim
    sensor_dict['Arequipa Laser']['FOV_hlim'] = FOV_hlim
    sensor_dict['Arequipa Laser']['FOV_vlim'] = FOV_vlim
    sensor_dict['Arequipa Laser']['laser_output'] = 1.  # Watts
    sensor_dict['Arequipa Laser']['passive_optical'] = False
    
    # Measurements and noise
    sensor_dict['Arequipa Laser']['meas_types'] = meas_types
    sensor_dict['Arequipa Laser']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['Arequipa Laser']['max_gap'] = max_gap
    sensor_dict['Arequipa Laser']['obs_gap'] = obs_gap
    sensor_dict['Arequipa Laser']['min_pass'] = min_pass
    sensor_dict['Arequipa Laser']['max_pass'] = max_pass


    print('Arequipa Optical')
    lat_gs = -16.4657
    lon_gs =  -71.4930
    ht_gs = 2.48905  # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Constraints/Limits
    az_lim = [0., 2.*pi]  # rad
    el_lim = [20.*pi/180., 80.*pi/180.]  # rad
    rg_lim = [0., 1e6]   # km
    mapp_lim = 16.
    sun_el_mask = -10.*pi/180.
    
    # FOV dimensions
    LAM_dim = 0.5   # deg
    PHI_dim = 0.5   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_dim*pi/180
    PHI_half = 0.5*PHI_dim*pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
   
    # Measurement types and noise
    meas_types = ['az', 'el']
    sigma_dict = {}
    sigma_dict['az'] = 5.*arcsec2rad   # rad
    sigma_dict['el'] = 5.*arcsec2rad  # rad
    
    # Pass parameters
    max_pass = 6000.
    min_pass = 60.
    max_gap = 60.
    obs_gap = 1.
    
    # Location and constraints
    sensor_dict['Arequipa Optical'] = {}
    sensor_dict['Arequipa Optical']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['Arequipa Optical']['site_ecef'] = site_ecef
    sensor_dict['Arequipa Optical']['mapp_lim'] = mapp_lim
    sensor_dict['Arequipa Optical']['sun_elmask'] = sun_el_mask
    sensor_dict['Arequipa Optical']['el_lim'] = el_lim
    sensor_dict['Arequipa Optical']['az_lim'] = az_lim
    sensor_dict['Arequipa Optical']['rg_lim'] = rg_lim
    sensor_dict['Arequipa Optical']['FOV_hlim'] = FOV_hlim
    sensor_dict['Arequipa Optical']['FOV_vlim'] = FOV_vlim
    sensor_dict['Arequipa Optical']['passive_optical'] = True
    
    # Measurements and noise
    sensor_dict['Arequipa Optical']['meas_types'] = meas_types
    sensor_dict['Arequipa Optical']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['Arequipa Optical']['max_gap'] = max_gap
    sensor_dict['Arequipa Optical']['obs_gap'] = obs_gap
    sensor_dict['Arequipa Optical']['min_pass'] = min_pass
    sensor_dict['Arequipa Optical']['max_pass'] = max_pass


    print('Haleakala Laser')
    lat_gs =  20.706489
    lon_gs =  203.743079
    ht_gs = 3.056272  # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Constraints/Limits
    az_lim = [0., 2.*pi]  # rad
    el_lim = [20.*pi/180., 80.*pi/180.]  # rad
    rg_lim = [0., 10000.]   # km
    
    # FOV dimensions
    LAM_dim = 0.5   # deg
    PHI_dim = 0.5   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_dim*pi/180
    PHI_half = 0.5*PHI_dim*pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
   
    # Measurement types and noise
    meas_types = ['rg', 'az', 'el']
    sigma_dict = {}
    sigma_dict['rg'] = 0.001  # km
    sigma_dict['az'] = 5.*arcsec2rad   # rad
    sigma_dict['el'] = 5.*arcsec2rad  # rad
    
    # Pass parameters
    max_pass = 6000.
    min_pass = 60.
    max_gap = 60.
    obs_gap = 1.
    
    # Location and constraints
    sensor_dict['Haleakala Laser'] = {}
    sensor_dict['Haleakala Laser']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['Haleakala Laser']['site_ecef'] = site_ecef
    sensor_dict['Haleakala Laser']['el_lim'] = el_lim
    sensor_dict['Haleakala Laser']['az_lim'] = az_lim
    sensor_dict['Haleakala Laser']['rg_lim'] = rg_lim
    sensor_dict['Haleakala Laser']['FOV_hlim'] = FOV_hlim
    sensor_dict['Haleakala Laser']['FOV_vlim'] = FOV_vlim
    sensor_dict['Haleakala Laser']['laser_output'] = 1.  # Watts
    sensor_dict['Haleakala Laser']['passive_optical'] = False
    
    # Measurements and noise
    sensor_dict['Haleakala Laser']['meas_types'] = meas_types
    sensor_dict['Haleakala Laser']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['Haleakala Laser']['max_gap'] = max_gap
    sensor_dict['Haleakala Laser']['obs_gap'] = obs_gap
    sensor_dict['Haleakala Laser']['min_pass'] = min_pass
    sensor_dict['Haleakala Laser']['max_pass'] = max_pass


    print('Haleakala Optical')
    lat_gs =  20.706489
    lon_gs =  203.743079
    ht_gs = 3.056272  # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Constraints/Limits
    az_lim = [0., 2.*pi]  # rad
    el_lim = [20.*pi/180., 80.*pi/180.]  # rad
    rg_lim = [0., 1e6]   # km
    mapp_lim = 16.
    sun_el_mask = -10.*pi/180.
    
    # FOV dimensions
    LAM_dim = 0.5   # deg
    PHI_dim = 0.5   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_dim*pi/180
    PHI_half = 0.5*PHI_dim*pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
   
    # Measurement types and noise
    meas_types = ['az', 'el']
    sigma_dict = {}
    sigma_dict['az'] = 5.*arcsec2rad   # rad
    sigma_dict['el'] = 5.*arcsec2rad  # rad
    
    # Pass parameters
    max_pass = 6000.
    min_pass = 60.
    max_gap = 60.
    obs_gap = 1.
    
    # Location and constraints
    sensor_dict['Haleakala Optical'] = {}
    sensor_dict['Haleakala Optical']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['Haleakala Optical']['site_ecef'] = site_ecef
    sensor_dict['Haleakala Optical']['mapp_lim'] = mapp_lim
    sensor_dict['Haleakala Optical']['sun_elmask'] = sun_el_mask
    sensor_dict['Haleakala Optical']['el_lim'] = el_lim
    sensor_dict['Haleakala Optical']['az_lim'] = az_lim
    sensor_dict['Haleakala Optical']['rg_lim'] = rg_lim
    sensor_dict['Haleakala Optical']['FOV_hlim'] = FOV_hlim
    sensor_dict['Haleakala Optical']['FOV_vlim'] = FOV_vlim
    sensor_dict['Haleakala Optical']['passive_optical'] = True
    
    # Measurements and noise
    sensor_dict['Haleakala Optical']['meas_types'] = meas_types
    sensor_dict['Haleakala Optical']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['Haleakala Optical']['max_gap'] = max_gap
    sensor_dict['Haleakala Optical']['obs_gap'] = obs_gap
    sensor_dict['Haleakala Optical']['min_pass'] = min_pass
    sensor_dict['Haleakala Optical']['max_pass'] = max_pass

    
    print('Yarragadee Laser')
    lat_gs =  -29.0464
    lon_gs =  115.3467
    ht_gs = 0.244  # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Constraints/Limits
    az_lim = [0., 2.*pi]  # rad
    el_lim = [20.*pi/180., 80.*pi/180.]  # rad
    rg_lim = [0., 10000.]   # km
    
    # FOV dimensions
    LAM_dim = 0.5   # deg
    PHI_dim = 0.5   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_dim*pi/180
    PHI_half = 0.5*PHI_dim*pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
   
    # Measurement types and noise
    meas_types = ['rg', 'az', 'el']
    sigma_dict = {}
    sigma_dict['rg'] = 0.001  # km
    sigma_dict['az'] = 5.*arcsec2rad   # rad
    sigma_dict['el'] = 5.*arcsec2rad  # rad
    
    # Pass parameters
    max_pass = 6000.
    min_pass = 60.
    max_gap = 60.
    obs_gap = 1.
    
    # Location and constraints
    sensor_dict['Yarragadee Laser'] = {}
    sensor_dict['Yarragadee Laser']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['Yarragadee Laser']['site_ecef'] = site_ecef
    sensor_dict['Yarragadee Laser']['el_lim'] = el_lim
    sensor_dict['Yarragadee Laser']['az_lim'] = az_lim
    sensor_dict['Yarragadee Laser']['rg_lim'] = rg_lim
    sensor_dict['Yarragadee Laser']['FOV_hlim'] = FOV_hlim
    sensor_dict['Yarragadee Laser']['FOV_vlim'] = FOV_vlim
    sensor_dict['Yarragadee Laser']['laser_output'] = 1.  # Watts
    sensor_dict['Yarragadee Laser']['passive_optical'] = False
    
    # Measurements and noise
    sensor_dict['Yarragadee Laser']['meas_types'] = meas_types
    sensor_dict['Yarragadee Laser']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['Yarragadee Laser']['max_gap'] = max_gap
    sensor_dict['Yarragadee Laser']['obs_gap'] = obs_gap
    sensor_dict['Yarragadee Laser']['min_pass'] = min_pass
    sensor_dict['Yarragadee Laser']['max_pass'] = max_pass


    print('Yarragadee Optical')
    lat_gs =  -29.0464
    lon_gs =  115.3467
    ht_gs = 0.244  # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Constraints/Limits
    az_lim = [0., 2.*pi]  # rad
    el_lim = [20.*pi/180., 80.*pi/180.]  # rad
    rg_lim = [0., 1e6]   # km
    mapp_lim = 16.
    sun_el_mask = -10.*pi/180.
    
    # FOV dimensions
    LAM_dim = 0.5   # deg
    PHI_dim = 0.5   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_dim*pi/180
    PHI_half = 0.5*PHI_dim*pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
   
    # Measurement types and noise
    meas_types = ['az', 'el']
    sigma_dict = {}
    sigma_dict['az'] = 5.*arcsec2rad   # rad
    sigma_dict['el'] = 5.*arcsec2rad  # rad
    
    # Pass parameters
    max_pass = 6000.
    min_pass = 60.
    max_gap = 60.
    obs_gap = 1.
    
    # Location and constraints
    sensor_dict['Yarragadee Optical'] = {}
    sensor_dict['Yarragadee Optical']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['Yarragadee Optical']['site_ecef'] = site_ecef
    sensor_dict['Yarragadee Optical']['mapp_lim'] = mapp_lim
    sensor_dict['Yarragadee Optical']['sun_elmask'] = sun_el_mask
    sensor_dict['Yarragadee Optical']['el_lim'] = el_lim
    sensor_dict['Yarragadee Optical']['az_lim'] = az_lim
    sensor_dict['Yarragadee Optical']['rg_lim'] = rg_lim
    sensor_dict['Yarragadee Optical']['FOV_hlim'] = FOV_hlim
    sensor_dict['Yarragadee Optical']['FOV_vlim'] = FOV_vlim
    sensor_dict['Yarragadee Optical']['passive_optical'] = True
    
    # Measurements and noise
    sensor_dict['Yarragadee Optical']['meas_types'] = meas_types
    sensor_dict['Yarragadee Optical']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['Yarragadee Optical']['max_gap'] = max_gap
    sensor_dict['Yarragadee Optical']['obs_gap'] = obs_gap
    sensor_dict['Yarragadee Optical']['min_pass'] = min_pass
    sensor_dict['Yarragadee Optical']['max_pass'] = max_pass


    
    print('ADFA UHF Radio')
    lat_gs = -35.29
    lon_gs = 149.17
    ht_gs = 0.606 # km	
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    site_ecef = coord.latlonht2ecef(lat_gs, lon_gs, ht_gs)
    
    # Constraints/Limits
    az_lim = [0., 2.*pi]  # rad
    el_lim = [0., pi/2.]  # rad
    rg_lim = [0., 5000.]   # km
    
    # FOV dimensions
    LAM_dim = 20.   # deg
    PHI_dim = 20.   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_dim*pi/180
    PHI_half = 0.5*PHI_dim*pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
    
    # Frequency limiits
    freq_lim = [300e6, 3e9]  # Hz
   
    # Measurement types and noise
    meas_types = ['rg', 'az', 'el']
    sigma_dict = {}
    sigma_dict['rg'] = 0.010  # km
    sigma_dict['az'] = 5.*pi/180.   # rad
    sigma_dict['el'] = 5.*pi/180.  # rad
    
    # Pass parameters
    max_pass = 6000.
    min_pass = 60.
    max_gap = 60.
    obs_gap = 1.
    
    # Location and constraints
    sensor_dict['ADFA UHF Radio'] = {}
    sensor_dict['ADFA UHF Radio']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['ADFA UHF Radio']['site_ecef'] = site_ecef
    sensor_dict['ADFA UHF Radio']['el_lim'] = el_lim
    sensor_dict['ADFA UHF Radio']['az_lim'] = az_lim
    sensor_dict['ADFA UHF Radio']['rg_lim'] = rg_lim
    sensor_dict['ADFA UHF Radio']['FOV_hlim'] = FOV_hlim
    sensor_dict['ADFA UHF Radio']['FOV_vlim'] = FOV_vlim
    sensor_dict['ADFA UHF Radio']['freq_lim'] = freq_lim
    sensor_dict['ADFA UHF Radio']['passive_optical'] = False
    
    # Measurements and noise
    sensor_dict['ADFA UHF Radio']['meas_types'] = meas_types
    sensor_dict['ADFA UHF Radio']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['ADFA UHF Radio']['max_gap'] = max_gap
    sensor_dict['ADFA UHF Radio']['obs_gap'] = obs_gap
    sensor_dict['ADFA UHF Radio']['min_pass'] = min_pass
    sensor_dict['ADFA UHF Radio']['max_pass'] = max_pass



    # Statistical Orbit Determination Baseline Case Sensors
    # Based on Tapley, Schutz, Born
    print('Born s101')
    
    site_ecef = np.reshape([-5127.510, -3794.160, 0.], (3,1))
    lat, lon, ht = coord.ecef2latlonht(site_ecef)
    geodetic_latlonht = [lat, lon, ht]
    
    # Constraints/Limits
    az_lim = [0., 2.*pi]  # rad
    el_lim = [0.*pi/180., pi/2.]  # rad
    rg_lim = [0., 40000.]   # km
    
    # FOV dimensions
    LAM_dim = 20.   # deg
    PHI_dim = 20.   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_dim*pi/180
    PHI_half = 0.5*PHI_dim*pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
   
    # Measurement types and noise
    meas_types = ['rg', 'ra', 'dec']
    sigma_dict = {}
    sigma_dict['rg'] = 1e-5  # km
    sigma_dict['ra'] = 5.*arcsec2rad   # rad
    sigma_dict['dec'] = 5.*arcsec2rad  # rad
    
    # Pass parameters
    max_pass = 6000.
    min_pass = 60.
    max_gap = 60.
    obs_gap = 1.
    
    # Location and constraints
    sensor_dict['Born s101'] = {}
    sensor_dict['Born s101']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['Born s101']['site_ecef'] = site_ecef
    sensor_dict['Born s101']['el_lim'] = el_lim
    sensor_dict['Born s101']['az_lim'] = az_lim
    sensor_dict['Born s101']['rg_lim'] = rg_lim
    sensor_dict['Born s101']['FOV_hlim'] = FOV_hlim
    sensor_dict['Born s101']['FOV_vlim'] = FOV_vlim
    sensor_dict['Born s101']['passive_optical'] = False
    
    # Measurements and noise
    sensor_dict['Born s101']['meas_types'] = meas_types
    sensor_dict['Born s101']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['Born s101']['max_gap'] = max_gap
    sensor_dict['Born s101']['obs_gap'] = obs_gap
    sensor_dict['Born s101']['min_pass'] = min_pass
    sensor_dict['Born s101']['max_pass'] = max_pass

    print('Born s337')
    
    site_ecef = np.reshape([3860.900, 3238.500, 3898.100], (3,1))
    lat, lon, ht = coord.ecef2latlonht(site_ecef)
    geodetic_latlonht = [lat, lon, ht]
    
    # Constraints/Limits
    az_lim = [0., 2.*pi]  # rad
    el_lim = [0.*pi/180., pi/2.]  # rad
    rg_lim = [0., 40000.]   # km
    
    # FOV dimensions
    LAM_dim = 20.   # deg
    PHI_dim = 20.   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_dim*pi/180
    PHI_half = 0.5*PHI_dim*pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
   
    # Measurement types and noise
    meas_types = ['rg', 'ra', 'dec']
    sigma_dict = {}
    sigma_dict['rg'] = 1e-5  # km
    sigma_dict['ra'] = 5.*arcsec2rad   # rad
    sigma_dict['dec'] = 5.*arcsec2rad  # rad
    
    # Pass parameters
    max_pass = 6000.
    min_pass = 60.
    max_gap = 60.
    obs_gap = 1.
    
    # Location and constraints
    sensor_dict['Born s337'] = {}
    sensor_dict['Born s337']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['Born s337']['site_ecef'] = site_ecef
    sensor_dict['Born s337']['el_lim'] = el_lim
    sensor_dict['Born s337']['az_lim'] = az_lim
    sensor_dict['Born s337']['rg_lim'] = rg_lim
    sensor_dict['Born s337']['FOV_hlim'] = FOV_hlim
    sensor_dict['Born s337']['FOV_vlim'] = FOV_vlim
    sensor_dict['Born s337']['passive_optical'] = False
    
    # Measurements and noise
    sensor_dict['Born s337']['meas_types'] = meas_types
    sensor_dict['Born s337']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['Born s337']['max_gap'] = max_gap
    sensor_dict['Born s337']['obs_gap'] = obs_gap
    sensor_dict['Born s337']['min_pass'] = min_pass
    sensor_dict['Born s337']['max_pass'] = max_pass
    
    
    print('Born s394')
    
    site_ecef = np.reshape([549.500, -1380.870, 6182.200], (3,1))
    lat, lon, ht = coord.ecef2latlonht(site_ecef)
    geodetic_latlonht = [lat, lon, ht]
    
    # Constraints/Limits
    az_lim = [0., 2.*pi]  # rad
    el_lim = [0.*pi/180., pi/2.]  # rad
    rg_lim = [0., 40000.]   # km
    
    # FOV dimensions
    LAM_dim = 20.   # deg
    PHI_dim = 20.   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_dim*pi/180
    PHI_half = 0.5*PHI_dim*pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
   
    # Measurement types and noise
    meas_types = ['rg', 'ra', 'dec']
    sigma_dict = {}
    sigma_dict['rg'] = 1e-5  # km
    sigma_dict['ra'] = 5.*arcsec2rad   # rad
    sigma_dict['dec'] = 5.*arcsec2rad  # rad
    
    # Pass parameters
    max_pass = 6000.
    min_pass = 60.
    max_gap = 60.
    obs_gap = 1.
    
    # Location and constraints
    sensor_dict['Born s394'] = {}
    sensor_dict['Born s394']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['Born s394']['site_ecef'] = site_ecef
    sensor_dict['Born s394']['el_lim'] = el_lim
    sensor_dict['Born s394']['az_lim'] = az_lim
    sensor_dict['Born s394']['rg_lim'] = rg_lim
    sensor_dict['Born s394']['FOV_hlim'] = FOV_hlim
    sensor_dict['Born s394']['FOV_vlim'] = FOV_vlim
    sensor_dict['Born s394']['passive_optical'] = False
    
    # Measurements and noise
    sensor_dict['Born s394']['meas_types'] = meas_types
    sensor_dict['Born s394']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['Born s394']['max_gap'] = max_gap
    sensor_dict['Born s394']['obs_gap'] = obs_gap
    sensor_dict['Born s394']['min_pass'] = min_pass
    sensor_dict['Born s394']['max_pass'] = max_pass

    # Remove sensors not in list
    if len(sensor_id_list) > 0:
        sensor_remove_list = list(set(sensor_dict.keys()) - set(sensor_id_list))
        for sensor_id in sensor_remove_list:
            sensor_dict.pop(sensor_id, None)

    
    return sensor_dict


def define_sites_from_file(site_data_file):
    
    # Determine file type
    fname, ext = os.path.splitext(site_data_file)
    
    if ext == '.json':
        site_df = pd.read_json(site_data_file)

        site_list = site_df['site'].tolist()
        lat_list = site_df['lat'].tolist()
        lon_list = site_df['lon'].tolist()

        # Form dictionary output
        site_dict = {}
        for ii in range(len(site_list)):
            site = site_list[ii]
            lat = float(lat_list[ii])
            lon = float(lon_list[ii])
            ht = 0.
            
            site_dict[site] = {}
            site_dict[site]['geodetic_latlonht'] = [lat, lon, ht]

    
    return site_dict


def generate_sensor_file(sensor_file, sensor_id_list=[]):
    
    sensor_dict = define_sensors(sensor_id_list)
    
    # Save data
    fname, ext = os.path.splitext(sensor_file) 

    if ext == '.pkl':
        pklFile = open( sensor_file, 'wb' )
        pickle.dump( [sensor_dict], pklFile, -1 )
        pklFile.close()
    
    if ext == '.csv':
        sensors = pd.DataFrame.from_dict(sensor_dict).T
        sensors.to_csv('sensors.csv')
    
    return
