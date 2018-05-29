from math import *


def define_sensors():
    
    # Initialize
    sensor_dict = {}
    
    
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
    mapp_lim = 16.5
    moon_angle_lim = 0.32  # rad
    sun_el_mask = -10.*pi/180.  # rad
    
    # Measurement types and noise
    meas_types = ['ra', 'dec']
    sigma_dict = {}
    sigma_dict['ra'] = 5./206265.   # rad
    sigma_dict['dec'] = 5./206265.  # rad

    # Set up sensors
    print('UNSW Falcon')
    lat_gs = -35.29
    lon_gs = 149.17
    ht_gs = 0.606 # km	
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    
    # Location and constraints
    sensor_dict['UNSW Falcon'] = {}
    sensor_dict['UNSW Falcon']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['UNSW Falcon']['mapp_lim'] = mapp_lim
    sensor_dict['UNSW Falcon']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['UNSW Falcon']['el_lim'] = el_lim
    sensor_dict['UNSW Falcon']['az_lim'] = az_lim
    sensor_dict['UNSW Falcon']['rg_lim'] = rg_lim
    sensor_dict['UNSW Falcon']['FOV_hlim'] = FOV_hlim
    sensor_dict['UNSW Falcon']['FOV_vlim'] = FOV_vlim
    sensor_dict['UNSW Falcon']['sun_elmask'] = sun_el_mask
    
    # Measurements and noise
    sensor_dict['UNSW Falcon']['meas_types'] = meas_types
    sensor_dict['UNSW Falcon']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['UNSW Falcon']['max_gap'] = 60.
    
    	
    print('FLC Falcon')
    lat_gs = 37.23
    lon_gs = 251.93
    ht_gs = 1.969 # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    
    # Location and constraints
    sensor_dict['FLC Falcon'] = {}
    sensor_dict['FLC Falcon']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['FLC Falcon']['mapp_lim'] = mapp_lim
    sensor_dict['FLC Falcon']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['FLC Falcon']['el_lim'] = el_lim
    sensor_dict['FLC Falcon']['az_lim'] = az_lim
    sensor_dict['FLC Falcon']['rg_lim'] = rg_lim
    sensor_dict['FLC Falcon']['FOV_hlim'] = FOV_hlim
    sensor_dict['FLC Falcon']['FOV_vlim'] = FOV_vlim
    sensor_dict['FLC Falcon']['sun_elmask'] = sun_el_mask
    
    # Measurements and noise
    sensor_dict['FLC Falcon']['meas_types'] = meas_types
    sensor_dict['FLC Falcon']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['FLC Falcon']['max_gap'] = 60.
	
    print('NJC Falcon')
    lat_gs = 40.65
    lon_gs = 256.80
    ht_gs = 1.177 # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    
    # Location and constraints
    sensor_dict['NJC Falcon'] = {}
    sensor_dict['NJC Falcon']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['NJC Falcon']['mapp_lim'] = mapp_lim
    sensor_dict['NJC Falcon']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['NJC Falcon']['el_lim'] = el_lim
    sensor_dict['NJC Falcon']['az_lim'] = az_lim
    sensor_dict['NJC Falcon']['rg_lim'] = rg_lim
    sensor_dict['NJC Falcon']['FOV_hlim'] = FOV_hlim
    sensor_dict['NJC Falcon']['FOV_vlim'] = FOV_vlim
    sensor_dict['NJC Falcon']['sun_elmask'] = sun_el_mask
    
    # Measurements and noise
    sensor_dict['NJC Falcon']['meas_types'] = meas_types
    sensor_dict['NJC Falcon']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['NJC Falcon']['max_gap'] = 60.
	
    print('OJC Falcon')
    lat_gs = 37.97
    lon_gs = 256.46
    ht_gs = 1.221 # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    
    # Location and constraints
    sensor_dict['OJC Falcon'] = {}
    sensor_dict['OJC Falcon']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['OJC Falcon']['mapp_lim'] = mapp_lim
    sensor_dict['OJC Falcon']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['OJC Falcon']['el_lim'] = el_lim
    sensor_dict['OJC Falcon']['az_lim'] = az_lim
    sensor_dict['OJC Falcon']['rg_lim'] = rg_lim
    sensor_dict['OJC Falcon']['FOV_hlim'] = FOV_hlim
    sensor_dict['OJC Falcon']['FOV_vlim'] = FOV_vlim
    sensor_dict['OJC Falcon']['sun_elmask'] = sun_el_mask
    
    # Measurements and noise
    sensor_dict['OJC Falcon']['meas_types'] = meas_types
    sensor_dict['OJC Falcon']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['OJC Falcon']['max_gap'] = 60.
	
    print('PSU Falcon')
    lat_gs = 40.86
    lon_gs = 282.17
    ht_gs = 0.347 # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]	
    
    # Location and constraints
    sensor_dict['PSU Falcon'] = {}
    sensor_dict['PSU Falcon']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['PSU Falcon']['mapp_lim'] = mapp_lim
    sensor_dict['PSU Falcon']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['PSU Falcon']['el_lim'] = el_lim
    sensor_dict['PSU Falcon']['az_lim'] = az_lim
    sensor_dict['PSU Falcon']['rg_lim'] = rg_lim
    sensor_dict['PSU Falcon']['FOV_hlim'] = FOV_hlim
    sensor_dict['PSU Falcon']['FOV_vlim'] = FOV_vlim
    sensor_dict['PSU Falcon']['sun_elmask'] = sun_el_mask
    
    # Measurements and noise
    sensor_dict['PSU Falcon']['meas_types'] = meas_types
    sensor_dict['PSU Falcon']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['PSU Falcon']['max_gap'] = 60.
	
    print('Mamalluca Falcon')
    lat_gs = -29.99
    lon_gs = 289.32
    ht_gs = 1.139 # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]

    # Location and constraints
    sensor_dict['Mamalluca Falcon'] = {}
    sensor_dict['Mamalluca Falcon']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['Mamalluca Falcon']['mapp_lim'] = mapp_lim
    sensor_dict['Mamalluca Falcon']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['Mamalluca Falcon']['el_lim'] = el_lim
    sensor_dict['Mamalluca Falcon']['az_lim'] = az_lim
    sensor_dict['Mamalluca Falcon']['rg_lim'] = rg_lim
    sensor_dict['Mamalluca Falcon']['FOV_hlim'] = FOV_hlim
    sensor_dict['Mamalluca Falcon']['FOV_vlim'] = FOV_vlim
    sensor_dict['Mamalluca Falcon']['sun_elmask'] = sun_el_mask
    
    # Measurements and noise
    sensor_dict['Mamalluca Falcon']['meas_types'] = meas_types
    sensor_dict['Mamalluca Falcon']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['Mamalluca Falcon']['max_gap'] = 60.
    
    
    print('Perth Falcon')
    lat_gs = -31.95
    lon_gs = 115.86
    ht_gs = 0. # km
    geodetic_latlonht = [lat_gs, lon_gs, ht_gs]
    
    # Location and constraints
    sensor_dict['Perth Falcon'] = {}
    sensor_dict['Perth Falcon']['geodetic_latlonht'] = geodetic_latlonht
    sensor_dict['Perth Falcon']['mapp_lim'] = mapp_lim
    sensor_dict['Perth Falcon']['moon_angle_lim'] = moon_angle_lim
    sensor_dict['Perth Falcon']['el_lim'] = el_lim
    sensor_dict['Perth Falcon']['az_lim'] = az_lim
    sensor_dict['Perth Falcon']['rg_lim'] = rg_lim
    sensor_dict['Perth Falcon']['FOV_hlim'] = FOV_hlim
    sensor_dict['Perth Falcon']['FOV_vlim'] = FOV_vlim
    sensor_dict['Perth Falcon']['sun_elmask'] = sun_el_mask
    
    # Measurements and noise
    sensor_dict['Perth Falcon']['meas_types'] = meas_types
    sensor_dict['Perth Falcon']['sigma_dict'] = sigma_dict
    
    # Pass parameters
    sensor_dict['Perth Falcon']['max_gap'] = 60.
	
	
    
    
    return sensor_dict




