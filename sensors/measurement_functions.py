import numpy as np
from math import pi, asin, atan2
import sys
import os
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

#from sensors.brdf_models import compute_mapp
import utilities.coordinate_systems as coord
import utilities.eop_functions as eop



def compute_measurement(X, state_params, sensor_params, sensor_id, UTC, 
                        EOP_data=[], XYs_df=[], meas_types=[], sun_gcrf=[]):
    
    # Retrieve data from sensor params
    sensor = sensor_params[sensor_id]
    
    if len(EOP_data) == 0:
        eop_alldata = sensor_params['eop_alldata']
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
    
    if len(XYs_df) == 0:    
        XYs_df = sensor_params['XYs_df']    
    
    # Retrieve measurement types
    if len(meas_types) == 0:
        meas_types = sensor['meas_types']
    
    # Compute station location in GCRF
    sensor_itrf = sensor['site_ecef']
    sensor_gcrf, dum = coord.itrf2gcrf(sensor_itrf, np.zeros((3,1)), UTC, EOP_data,
                                 XYs_df)
    
#    print('sensor_gcrf', sensor_gcrf)
    
    # Object location in GCRF
    r_gcrf = X[0:3].reshape(3,1)
    
    # Compute range and line of sight vector
    rg = np.linalg.norm(r_gcrf - sensor_gcrf)
    rho_hat_gcrf = (r_gcrf - sensor_gcrf)/rg
    
    # Rotate to ENU frame
    rho_hat_itrf, dum = coord.gcrf2itrf(rho_hat_gcrf, np.zeros((3,1)), UTC, EOP_data,
                                  XYs_df)
    rho_hat_enu = coord.ecef2enu(rho_hat_itrf, sensor_itrf)
    
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



def ecef2azelrange_deg(r_sat, r_site):
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
      azimuth, degrees clockwise from north [0 to 360 deg]
    el : float
      elevation, degrees up from horizon [-90 to 90 deg]
    rg : float
      scalar distance from site to sat [km]
    '''

    # Compute vector from site to satellite and range
    rho_ecef = r_sat - r_site
    rg = np.linalg.norm(rho_ecef)  # km

    # Compute unit vector in LOS direction from site to sat
    rho_hat_ecef = rho_ecef/rg

    # Rotate to ENU
    rho_hat_enu = coord.ecef2enu(rho_hat_ecef, r_site)

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
      azimuth, clockwise from north [0 to 2pi rad]
    el : float
      elevation, up from horizon [-pi/2 to pi/2 rad]
    rg : float
      scalar distance from site to sat [km]
    '''

    # Compute vector from site to satellite and range
    rho_ecef = r_sat - r_site
    rg = np.linalg.norm(rho_ecef)  # km

    # Compute unit vector in LOS direction from site to sat
    rho_hat_ecef = rho_ecef/rg

    # Rotate to ENU
    rho_hat_enu = coord.ecef2enu(rho_hat_ecef, r_site)

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


###############################################################################
# Generic Estimation Measurement Functions
###############################################################################
    

def H_linear1d_rg(tk, Xref, state_params, sensor_params, sensor_id):
    
    # Break out state
    x = float(Xref[0])
    
    # Measurement information
    sensor_kk = sensor_params[sensor_id]
    meas_types = sensor_kk['meas_types']
    sigma_dict = sensor_kk['sigma_dict']
    p = len(meas_types)
    Rk = np.zeros((p, p))
    for ii in range(p):
        mtype = meas_types[ii]
        sig = sigma_dict[mtype]
        Rk[ii,ii] = sig**2.   
    
    # Hk_til and Gi
    Hk_til = np.array([[1., 0.]])
    Gk = np.array([[x]])
    
    return Hk_til, Gk, Rk


def unscented_linear1d_rg(tk, chi, state_params, sensor_params, sensor_id):
    
    # Number of states
    n = int(chi.shape[0])
    
    # Measurement information
    sensor_kk = sensor_params[sensor_id]
    meas_types = sensor_kk['meas_types']
    sigma_dict = sensor_kk['sigma_dict']
    p = len(meas_types)
    Rk = np.zeros((p, p))
    for ii in range(p):
        mtype = meas_types[ii]
        sig = sigma_dict[mtype]
        Rk[ii,ii] = sig**2.
    
    # Compute transformed sigma points   
    gamma_til = np.zeros((p, (2*n+1)))
    for jj in range(2*n+1):
        
        rg = chi[0,jj]
        gamma_til[0,jj] = rg

    return gamma_til, Rk


def H_balldrop(tk, Xref, state_params, sensor_params, sensor_id):
    
    # Break out state
    y = float(Xref[0])
    dy = float(Xref[1])
    
    # Measurement information
    sensor_kk = sensor_params[sensor_id]
    meas_types = sensor_kk['meas_types']
    sigma_dict = sensor_kk['sigma_dict']
    p = len(meas_types)
    Rk = np.zeros((p, p))
    for ii in range(p):
        mtype = meas_types[ii]
        sig = sigma_dict[mtype]
        Rk[ii,ii] = sig**2.   
    
    # Hk_til and Gi
    Hk_til = np.diag([1.,1.])
    Gk = np.array([[y],[dy]])
    
    return Hk_til, Gk, Rk


def unscented_balldrop(tk, chi, state_params, sensor_params, sensor_id):
    
    # Number of states
    n = int(chi.shape[0])
    
    # Measurement information
    sensor_kk = sensor_params[sensor_id]
    meas_types = sensor_kk['meas_types']
    sigma_dict = sensor_kk['sigma_dict']
    p = len(meas_types)
    Rk = np.zeros((p, p))
    for ii in range(p):
        mtype = meas_types[ii]
        sig = sigma_dict[mtype]
        Rk[ii,ii] = sig**2.
    
    # Compute transformed sigma points   
    gamma_til = np.zeros((p, (2*n+1)))
    for jj in range(2*n+1):
        
        y = chi[0,jj]
        dy = chi[1,jj]
        
        gamma_til[0,jj] = y
        gamma_til[1,jj] = dy

    return gamma_til, Rk




###############################################################################
# Orbit Estimation Measurement Functions
###############################################################################
 

def H_rgradec(tk, Xref, state_params, sensor_params, sensor_id):
    
    # Number of states
    n = len(Xref)
    
    # Compute sensor position in GCRF
    eop_alldata = sensor_params['eop_alldata']
    XYs_df = sensor_params['XYs_df']
    EOP_data = eop.get_eop_data(eop_alldata, tk)
    
    sensor_kk = sensor_params[sensor_id]
    sensor_itrf = sensor_kk['site_ecef']
    sensor_gcrf, dum = coord.itrf2gcrf(sensor_itrf, np.zeros((3,1)), tk, EOP_data, XYs_df)
    
    # Measurement noise
    meas_types = sensor_kk['meas_types']
    sigma_dict = sensor_kk['sigma_dict']
    p = len(meas_types)
    Rk = np.zeros((p, p))
    for ii in range(p):
        mtype = meas_types[ii]
        sig = sigma_dict[mtype]
        Rk[ii,ii] = sig**2.   
    
    
    # Object location in GCRF
    r_gcrf = Xref[0:3].reshape(3,1)
    
    # Compute range and line of sight vector
    rho_gcrf = r_gcrf - sensor_gcrf
    rg = np.linalg.norm(rho_gcrf)
    rho_hat_gcrf = rho_gcrf/rg
    
    ra = atan2(rho_hat_gcrf[1], rho_hat_gcrf[0]) #rad
    dec = asin(rho_hat_gcrf[2])  #rad

    # Calculate partials of rho
    drho_dx = rho_hat_gcrf[0]
    drho_dy = rho_hat_gcrf[1]
    drho_dz = rho_hat_gcrf[2]
    
    # Calculate partials of right ascension
    d_atan = 1./(1. + (rho_gcrf[1]/rho_gcrf[0])**2.)
    dra_dx = d_atan*(-(rho_gcrf[1])/((rho_gcrf[0])**2.))
    dra_dy = d_atan*(1./(rho_gcrf[0]))
    
    # Calculate partials of declination
    d_asin = 1./np.sqrt(1. - ((rho_gcrf[2])/rg)**2.)
    ddec_dx = d_asin*(-(rho_gcrf[2])/rg**2.)*drho_dx
    ddec_dy = d_asin*(-(rho_gcrf[2])/rg**2.)*drho_dy
    ddec_dz = d_asin*(1./rg - ((rho_gcrf[2])/rg**2.)*drho_dz)

    # Hk_til and Gi
    Gk = np.reshape([rg, ra, dec], (3,1))
    
    Hk_til = np.zeros((3,n))
    Hk_til[0,0] = drho_dx
    Hk_til[0,1] = drho_dy
    Hk_til[0,2] = drho_dz
    Hk_til[1,0] = dra_dx
    Hk_til[1,1] = dra_dy
    Hk_til[2,0] = ddec_dx
    Hk_til[2,1] = ddec_dy
    Hk_til[2,2] = ddec_dz    
    
    
    return Hk_til, Gk, Rk


def unscented_rgradec(tk, chi, state_params, sensor_params, sensor_id):
    
    # Number of states
    n = int(chi.shape[0])
    
    # Compute sensor position in GCRF
    eop_alldata = sensor_params['eop_alldata']
    XYs_df = sensor_params['XYs_df']
    EOP_data = eop.get_eop_data(eop_alldata, tk)
    
    sensor_kk = sensor_params[sensor_id]
    sensor_itrf = sensor_kk['site_ecef']
    sensor_gcrf, dum = coord.itrf2gcrf(sensor_itrf, np.zeros((3,1)), tk, EOP_data, XYs_df)
    
    # Measurement information    
    meas_types = sensor_kk['meas_types']
    sigma_dict = sensor_kk['sigma_dict']
    p = len(meas_types)
    Rk = np.zeros((p, p))
    for ii in range(p):
        mtype = meas_types[ii]
        sig = sigma_dict[mtype]
        Rk[ii,ii] = sig**2.
    
    # Compute transformed sigma points   
    gamma_til = np.zeros((p, (2*n+1)))
    for jj in range(2*n+1):
        
        x = chi[0,jj]
        y = chi[1,jj]
        z = chi[2,jj]
        
        # Object location in GCRF
        r_gcrf = np.reshape([x,y,z], (3,1))
        
        # Compute range and line of sight vector
        rho_gcrf = r_gcrf - sensor_gcrf
        rg = np.linalg.norm(rho_gcrf)
        rho_hat_gcrf = rho_gcrf/rg
        
        ra = atan2(rho_hat_gcrf[1], rho_hat_gcrf[0]) #rad
        dec = asin(rho_hat_gcrf[2])  #rad
        
        # Store quadrant info of mean sigma point        
        if jj == 0:
            quad = 0
            if ra > pi/2. and ra < pi:
                quad = 2
            if ra < -pi/2. and ra > -pi:
                quad = 3
                
        # Check and update quadrant of subsequent sigma points
        else:
            if quad == 2 and ra < 0.:
                ra += 2.*pi
            if quad == 3 and ra > 0.:
                ra -= 2.*pi
                
        # Form Output
        gamma_til[0,jj] = rg
        gamma_til[1,jj] = ra
        gamma_til[2,jj] = dec


    return gamma_til, Rk


def H_radec(tk, Xref, state_params, sensor_params, sensor_id):
    
    # Number of states
    n = len(Xref)    
    
    # Compute sensor position in GCRF
    eop_alldata = sensor_params['eop_alldata']
    XYs_df = sensor_params['XYs_df']
    EOP_data = eop.get_eop_data(eop_alldata, tk)
    
    sensor_kk = sensor_params[sensor_id]
    sensor_itrf = sensor_kk['site_ecef']
    sensor_gcrf, dum = coord.itrf2gcrf(sensor_itrf, np.zeros((3,1)), tk, EOP_data, XYs_df)
    
    # Measurement noise
    meas_types = sensor_kk['meas_types']
    sigma_dict = sensor_kk['sigma_dict']
    p = len(meas_types)
    Rk = np.zeros((p, p))
    for ii in range(p):
        mtype = meas_types[ii]
        sig = sigma_dict[mtype]
        Rk[ii,ii] = sig**2.   
    
    
    # Object location in GCRF
    r_gcrf = Xref[0:3].reshape(3,1)
    
    # Compute range and line of sight vector
    rho_gcrf = r_gcrf - sensor_gcrf
    rg = np.linalg.norm(rho_gcrf)
    rho_hat_gcrf = rho_gcrf/rg
    
    ra = atan2(rho_hat_gcrf[1], rho_hat_gcrf[0]) #rad
    dec = asin(rho_hat_gcrf[2])  #rad

    # Calculate partials of rho
    drho_dx = rho_hat_gcrf[0]
    drho_dy = rho_hat_gcrf[1]
    drho_dz = rho_hat_gcrf[2]
    
    # Calculate partials of right ascension
    d_atan = 1./(1. + (rho_gcrf[1]/rho_gcrf[0])**2.)
    dra_dx = d_atan*(-(rho_gcrf[1])/((rho_gcrf[0])**2.))
    dra_dy = d_atan*(1./(rho_gcrf[0]))
    
    # Calculate partials of declination
    d_asin = 1./np.sqrt(1. - ((rho_gcrf[2])/rg)**2.)
    ddec_dx = d_asin*(-(rho_gcrf[2])/rg**2.)*drho_dx
    ddec_dy = d_asin*(-(rho_gcrf[2])/rg**2.)*drho_dy
    ddec_dz = d_asin*(1./rg - ((rho_gcrf[2])/rg**2.)*drho_dz)

    # Hk_til and Gi
    Gk = np.reshape([ra, dec], (2,1))
    
    Hk_til = np.zeros((2,n))
    Hk_til[0,0] = dra_dx
    Hk_til[0,1] = dra_dy
    Hk_til[1,0] = ddec_dx
    Hk_til[1,1] = ddec_dy
    Hk_til[1,2] = ddec_dz    
    
    
    return Hk_til, Gk, Rk


def unscented_radec(tk, chi, state_params, sensor_params, sensor_id):
    
    # Number of states
    n = int(chi.shape[0])
    
    # Compute sensor position in GCRF
    eop_alldata = sensor_params['eop_alldata']
    XYs_df = sensor_params['XYs_df']
    EOP_data = eop.get_eop_data(eop_alldata, tk)
    
    sensor_kk = sensor_params[sensor_id]
    sensor_itrf = sensor_kk['site_ecef']
    sensor_gcrf, dum = coord.itrf2gcrf(sensor_itrf, np.zeros((3,1)), tk, EOP_data, XYs_df)
    
    # Measurement information    
    meas_types = sensor_kk['meas_types']
    sigma_dict = sensor_kk['sigma_dict']
    p = len(meas_types)
    Rk = np.zeros((p, p))
    for ii in range(p):
        mtype = meas_types[ii]
        sig = sigma_dict[mtype]
        Rk[ii,ii] = sig**2.
    
    # Compute transformed sigma points   
    gamma_til = np.zeros((p, (2*n+1)))
    for jj in range(2*n+1):
        
        x = chi[0,jj]
        y = chi[1,jj]
        z = chi[2,jj]
        
        # Object location in GCRF
        r_gcrf = np.reshape([x,y,z], (3,1))
        
        # Compute range and line of sight vector
        rho_gcrf = r_gcrf - sensor_gcrf
        rg = np.linalg.norm(rho_gcrf)
        rho_hat_gcrf = rho_gcrf/rg
        
        ra = atan2(rho_hat_gcrf[1], rho_hat_gcrf[0]) #rad
        dec = asin(rho_hat_gcrf[2])  #rad
        
        # Store quadrant info of mean sigma point        
        if jj == 0:
            quad = 0
            if ra > pi/2. and ra < pi:
                quad = 2
            if ra < -pi/2. and ra > -pi:
                quad = 3
                
        # Check and update quadrant of subsequent sigma points
        else:
            if quad == 2 and ra < 0.:
                ra += 2.*pi
            if quad == 3 and ra > 0.:
                ra -= 2.*pi
                
        # Form Output
        
        gamma_til[0,jj] = ra
        gamma_til[1,jj] = dec

    return gamma_til, Rk


def unscented_rg(tk, chi, state_params, sensor_params, sensor_id):
    
    # Number of states
    n = int(chi.shape[0])
    
    # Compute sensor position in GCRF
    eop_alldata = sensor_params['eop_alldata']
    XYs_df = sensor_params['XYs_df']
    EOP_data = eop.get_eop_data(eop_alldata, tk)
    
    sensor_kk = sensor_params[sensor_id]
    sensor_itrf = sensor_kk['site_ecef']
    sensor_gcrf, dum = coord.itrf2gcrf(sensor_itrf, np.zeros((3,1)), tk, EOP_data, XYs_df)
    
    # Measurement information    
    meas_types = sensor_kk['meas_types']
    sigma_dict = sensor_kk['sigma_dict']
    p = len(meas_types)
    Rk = np.zeros((p, p))
    for ii in range(p):
        mtype = meas_types[ii]
        sig = sigma_dict[mtype]
        Rk[ii,ii] = sig**2.
    
    # Compute transformed sigma points   
    gamma_til = np.zeros((p, (2*n+1)))
    for jj in range(2*n+1):
        
        x = chi[0,jj]
        y = chi[1,jj]
        z = chi[2,jj]
        
        # Object location in GCRF
        r_gcrf = np.reshape([x,y,z], (3,1))
        
        # Compute range and line of sight vector
        rho_gcrf = r_gcrf - sensor_gcrf
        rg = np.linalg.norm(rho_gcrf)
                
        # Form Output
        gamma_til[0,jj] = rg


    return gamma_til, Rk


def H_cwrho(tk, Xref, state_params, sensor_params, sensor_id):

    
    # Measurement noise
    sensor_kk = sensor_params[sensor_id]
    sigma_dict = sensor_kk['sigma_dict']
    Rk = np.zeros((1, 1))
    sig = sigma_dict['rho']
    Rk[0,0] = sig**2.   
    
    # Object location in RIC
    x = float(Xref[0])
    y = float(Xref[1])
    z = float(Xref[2])
    
    # Compute range and line of sight vector
    rho = np.linalg.norm([x, y, z])
    
#    print('\n H fcn')
#    print(Xref)
#    print(x)
#    print(y)
#    print(z)
#    print(rho)

    # Hk_til and Gi
    Gk = np.zeros((1,1))
    Gk[0] = rho
    
    Hk_til = np.zeros((1,6))
    Hk_til[0,0] = x/rho
    Hk_til[0,1] = y/rho
    Hk_til[0,2] = z/rho  
    
#    print('Gk', Gk)
#    print('Hk_til', Hk_til)
    
    
    return Hk_til, Gk, Rk


def H_cwxyz(tk, Xref, state_params, sensor_params, sensor_id):

    
    # Measurement noise
    sensor_kk = sensor_params[sensor_id]
    sigma_dict = sensor_kk['sigma_dict']
    Rk = np.zeros((6,6))
    Rk[0,0] = sigma_dict['x']**2.   
    Rk[1,1] = sigma_dict['y']**2.
    Rk[2,2] = sigma_dict['z']**2.
    Rk[3,3] = sigma_dict['dx']**2.
    Rk[4,4] = sigma_dict['dy']**2.
    Rk[5,5] = sigma_dict['dz']**2.
    
    # Object location in RIC
    x = float(Xref[0])
    y = float(Xref[1])
    z = float(Xref[2])
    dx = float(Xref[3])
    dy = float(Xref[4])
    dz = float(Xref[5])

    # Hk_til and Gi
#    Gk = np.zeros((6,1))
#    Gk[0] = x
#    Gk[1] = y
#    Gk[2] = z
#    Gk[3]
    
    Gk = Xref.reshape(6,1)
    
#    Hk_til = np.zeros((6,6))
#    Hk_til[0,0] = 1.
#    Hk_til[1,1] = 1.
#    Hk_til[2,2] = 1.
    Hk_til = np.eye(6)
    
#    print('Gk', Gk)
#    print('Hk_til', Hk_til)
    
    
    return Hk_til, Gk, Rk


def H_nonlincw_full(tk, Xref, state_params, sensor_params, sensor_id):

    
    # Measurement noise
    sensor_kk = sensor_params[sensor_id]
    sigma_dict = sensor_kk['sigma_dict']
    Rk = np.zeros((6,6))
    Rk[0,0] = sigma_dict['x']**2.   
    Rk[1,1] = sigma_dict['y']**2.
    Rk[2,2] = sigma_dict['z']**2.
    Rk[3,3] = sigma_dict['dx']**2.
    Rk[4,4] = sigma_dict['dy']**2.
    Rk[5,5] = sigma_dict['dz']**2.
    
    # Object location in RIC
    x = float(Xref[0])
    y = float(Xref[1])
    z = float(Xref[2])
    dx = float(Xref[3])
    dy = float(Xref[4])
    dz = float(Xref[5])

    # Hk_til and Gi
#    Gk = np.zeros((6,1))
#    Gk[0] = x
#    Gk[1] = y
#    Gk[2] = z
#    Gk[3]
    
    Gk = Xref[0:6].reshape(6,1)
    
    Hk_til = np.zeros((6,9))
    Hk_til[0,0] = 1.
    Hk_til[1,1] = 1.
    Hk_til[2,2] = 1.
    Hk_til[3,3] = 1.
    Hk_til[4,4] = 1.
    Hk_til[5,5] = 1.
    
    
#    print('Gk', Gk)
#    print('Hk_til', Hk_til)
    
    
    return Hk_til, Gk, Rk