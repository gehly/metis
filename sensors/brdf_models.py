import numpy as np
from math import pi, sin, cos, tan, asin, acos, atan, atan2, log10
import sys

sys.path.append('../')

import utilities.attitude as att



def compute_mapp(sat2sun, sat2obs, spacecraftConfig, surfaces, q_IB=[]):
    '''
    This function computes the apparent magnitude of a space object
    
    Parameters
    ------
    sat2sun : 3x1 numpy array
        position vector from space object to sun
    sat2obs : 3x1 numpy array
        position vector from space object to observer
    brdf_function : function handle
        function handle for BRDF calculation
    surfaces : dict
        contains parameters for the different space object surfaces, including
        body fixed unit vectors and reflectance parameters
    q_IB : 4x1 numpy array, optional (required for 6DOF)
        attitude quaternion to transform coordinates from body fixed to 
        inertial frame
    
    Returns
    ------
    mapp : float
        apparent magnitude of space object    
    
    Reference
    ------
    [1] Linares, et al., "Space Object Shape Characterization and Tracking
    Using Light Curve and Angles Data," 2014.
    
    [2] Linares, et al., "Astrometric and Photometric Data Fusion for 
    Inactive Space Object Feature Estimation," IAC 2011.
    
    [3] Cognion, "Observations and Modeling of GEO Satellites at Large Phase
    Angles," AMOS 2013.
    
    '''

    # Compute unit vectors
    u_sun = sat2sun/np.linalg.norm(sat2sun)
    u_obs = sat2obs/np.linalg.norm(sat2obs)
    
    brdf_function = spacecraftConfig['brdf_function']
    if brdf_function == lambertian_sphere:
        
        # Compute BRDF for the sphere
        brdf_params = surfaces[0]['brdf_params']
        C_sunvis = brdf_params['cSunVis']
        brdf = brdf_function(u_sun, u_obs, brdf_params)
        
        # Compute Fobs per Reference 3
        sat_radius = spacecraftConfig['radius']
        sat_rg = np.linalg.norm(sat2obs)
        phase_angle = acos(float(np.dot(u_sun.T, u_obs)))
        
        sum_Fobs = C_sunvis * (4./9.) * brdf * (sat_radius/sat_rg)**2. * \
        (sin(phase_angle) + (pi - phase_angle)*cos(phase_angle))
    
    else:
    
        # Loop over all surfaces to compute total light reflected to observer
        # per Reference 1
        sum_Fobs = 0.
        for ii in surfaces:
            
            # Retrieve BRDF parameters
            brdf_params = surfaces[ii]['brdf_params']
            norm_body_hat = brdf_params['norm_body_hat']
            u_body_hat = brdf_params['u_body_hat']
            v_body_hat = brdf_params['v_body_hat']
    
            # Convert to ECI as needed
            norm_eci_hat = att.quat_rotate(q_IB, norm_body_hat)
            brdf_params['norm_eci_hat'] = norm_eci_hat
            brdf_params['u_eci_hat'] = att.quat_rotate(q_IB, u_body_hat)
            brdf_params['v_eci_hat'] = att.quat_rotate(q_IB, v_body_hat)
            
            # Check angles, if greater than 90 degrees, no light is reflected
            # from this surface to the observer, no need to compute BRDF
            dot_n_sun = float(np.dot(norm_eci_hat.T, u_sun))
            dot_n_obs = float(np.dot(norm_eci_hat.T, u_obs))
            if dot_n_sun < 0. or dot_n_obs < 0.:
                continue
            
            # Compute BRDF for this surface
            brdf = brdf_function(u_sun, u_obs, brdf_params)
            
            # Compute Fsun and Fobs
            Fsun = C_sunvis*brdf*dot_n_sun
            Fobs = Fsun*A*dot_n_obs/float(np.dot(sat2obs.T, sat2obs))
            
            sum_Fobs += Fobs
    
    # Compute mapp
    mapp = -26.74 - 2.5*log10(sum_Fobs/C_sunvis)
    
    return mapp


def lambertian_sphere(u_sun, u_obs, brdf_params):
    '''
    This function computes the Lambertian (diffuse) sphere BRDF value.
    
    Parameters
    ------
    u_sun : 3x1 numpy array
        unit vector from surface to sun in ECI frame
    u_obs : 3x1 numpy array
        unit vector from surface to observer in ECI frame
    brdf_params : dictionary
        contains additional parameters about reflectance property of surface
        and unit vectors defining body fixed directions in ECI frame
    
    Returns
    ------
    brdf : float
        bidirectional reflectance distribution function (>= 0.)    
    
    Reference
    ------        
    [1] Wetterer, et al., "Refining space object radiation pressure modeling 
    with bidirectional reflectance distribution functions," 2014.
    
    '''
    
    # Break out inputs
    d = brdf_params['d']
    rho = brdf_params['rho']
    
    # Compute diffuse BRDF
    brdf = d*rho/pi    
    
    return brdf


#def lambert_plus_specular(sun_eci_hat, obs_eci_hat, brdf_params):
#    '''
#    This function computes a simple BRDF combining diffuse Lambertian
#    and specular reflection.
#    
#    Parameters
#    ------
#    sun_eci_hat : 3x1 numpy array
#        unit vector from surface to sun in ECI frame
#    obs_eci_hat : 3x1 numpy array
#        unit vector from surface to observer in ECI frame
#    brdf_params : dictionary
#        contains additional parameters about reflectance property of surface
#        and unit vectors defining body fixed directions in ECI frame
#    
#    Returns
#    ------
#    brdf : float
#        bidirection reflectance distribution function (>= 0.)    
#    
#    Reference
#    ------        
#    [1] Wetterer, et al., "Refining space object radiation pressure modeling 
#    with bidirectional reflectance distribution functions," 2014, Eq. 4.
#    
#    '''
#    
#    # Break out inputs
#    d = brdf_params['d']
#    s = brdf_params['s']
#    rho = brdf_params['rho']
#    Fo = brdf_params['Fo']
#    
#    # Compute diffuse BRDF
#    brdf_diff = d*rho/pi    
#    
#    # Compute specular BRDF
#    
#    
#    
#    return brdf


#def cook_torrance(sun_eci_hat, obs_eci_hat, brdf_params):
#    '''
#    This function computes the Cook-Torrance BRDF value.
#    
#    Parameters
#    ------
#    sun_eci_hat : 3x1 numpy array
#        unit vector from surface to sun in ECI frame
#    obs_eci_hat : 3x1 numpy array
#        unit vector from surface to observer in ECI frame
#    brdf_params : dictionary
#        contains additional parameters about reflectance property of surface
#        and unit vectors defining body fixed directions in ECI frame
#    
#    Returns
#    ------
#    brdf : float
#        bidirectional reflectance distribution function (>= 0.)    
#    
#    Reference
#    ------    
#    [1] Cook, et al., "A reflectance model for computer graphics," 1982.
#    
#    [2] Wetterer, et al., "Refining space object radiation pressure modeling 
#    with bidirectional reflectance distribution functions," 2014.
#    
#    '''
#    
#    # Break out inputs
#    d = brdf_params['d']
#    s = brdf_params['s']
#    rho = brdf_params['rho']
#    Fo = brdf_params['Fo']
#    m = brdf_params['m']
#    norm_eci_hat = brdf_params['norm_eci_hat']
#    
#    # Compute half angle unit vector
#    half_eci = (sun_eci_hat + obs_eci_hat)
#    half_eci_hat = half_eci/np.linalg.norm(half_eci)
#    
#    # Compute dot products
#    dot_n_sun = float(np.dot(norm_eci_hat.T, sun_eci_hat))
#    dot_n_obs = float(np.dot(norm_eci_hat.T, obs_eci_hat))
#    dot_n_h = float(np.dot(norm_eci_hat.T, half_eci_hat))
#    dot_h_obs = float(np.dot(half_eci_hat.T, obs_eci_hat))
#    
#    # Compute diffuse BRDF
#    brdf_diff = d*rho/pi
#    
#    # Compute specular BRDF
#    nest = (1. + np.sqrt(Fo))/(1 - np.sqrt(Fo))
#    g = nest**2. + dot_h_obs**2. - 1.
#    alpha = acos(dot_n_h)
#    
#    D = 1./(pi*m**2.*dot_n_h**4.)*exp(-(tan(alpha/m)**2.))
#    G = min([1., (2.*dot_n_h*dot_n_obs/dot_h_obs), (2.*dot_n_h*dot_n_sun/dot_h_obs)])
#    F = ((g - dot_h_obs)**2./(2.*(g + dot_h_obs)**2.)) * \
#        ((dot_h_obs*(g + dot_h_obs) - 1.)**2.)/((dot_h_obs*(g - dot_h_obs) + 1.)**2.)
#    
#    brdf_spec = s*D*G*F/(4.*dot_n_sun*dot_n_obs)
#    
#    # Compute total BRDF
#    brdf = brdf_diff + brdf_spec
#    
#    return brdf
    
    
def ashikhmin_shirley(u_sun, u_obs, brdf_params):
    '''
    This function computes the Ashikhmin-Shirley BRDF value.
    
    Parameters
    ------
    u_sun : 3x1 numpy array
        unit vector from surface to sun in ECI frame
    obs_eci_hat : 3x1 numpy array
        unit vector from surface to observer in ECI frame
    brdf_params : dictionary
        contains additional parameters about reflectance property of surface
        and unit vectors defining body fixed directions in ECI frame
    
    Returns
    ------
    brdf : float
        bidirectional reflectance distribution function (>= 0.)    
    
    Reference
    ------    
    [1] Ashikhmin, et al., "An Anistropic Phong BRDF Model," 2000.
    
    [2] Linares, et al., "Astrometric and photometric data fusion for 
    resident space object orbit, attitude, and shape determination via 
    multiple-model adaptive estimation," 2010.
    
    [3] Wetterer, et al., "Refining space object radiation pressure modeling 
    with bidirectional reflectance distribution functions," 2014.
    
    '''
    
    # Break out inputs
    d = brdf_params['d']
    s = brdf_params['s']
    rho = brdf_params['rho']
    Fo = brdf_params['Fo']
    nu = brdf_params['nu']
    nv = brdf_params['nv']
    norm_eci_hat = brdf_params['norm_eci_hat']
    u_eci_hat = brdf_params['u_eci_hat']
    v_eci_hat = brdf_params['v_eci_hat']
    
    Rdiff = d*rho
    Rspec = s*Fo
    
    # Compute half angle unit vector
    half_eci = (u_sun + u_obs)
    half_eci_hat = half_eci/np.linalg.norm(half_eci)
    
    # Compute dot products
    dot_n_sun = float(np.dot(norm_eci_hat.T, u_sun))
    dot_n_obs = float(np.dot(norm_eci_hat.T, u_obs))
    dot_n_h = float(np.dot(norm_eci_hat.T, half_eci_hat))
    dot_h_obs = float(np.dot(half_eci_hat.T, u_obs))
    dot_h_u = float(np.dot(half_eci_hat.T, u_eci_hat))
    dot_h_v = float(np.dot(half_eci_hat.T, v_eci_hat))
    maxdot = max([dot_n_sun, dot_n_obs])
    
    # Compute diffuse BRDF
    brdf_diff = (28.*Rdiff/(23.*pi))*(1. - Rspec)*(1. - (1.-dot_n_sun/2.)**5.) \
                * (1. - (1.-dot_n_obs/2.)**5.)
    
    # Compute specular BRDF
    F = Rspec + (1. - Rspec)*(1. - dot_h_obs)**5.
    z = (nu*dot_h_u**2. + nv*dot_h_v**2.)/(1. - dot_n_sun**2.)
    brdf_spec = np.sqrt((nu + 1.)*(nv + 1.))/(8.*pi) \
                * dot_n_h**z/(dot_h_obs*maxdot) * F    
    
    # Compute total BRDF
    brdf = brdf_diff + brdf_spec
    
    return brdf


def ashikhmin_premoze(u_sun, u_obs, brdf_params):
    '''
    This function computes the Ashikhmin-Premoze BRDF value.
    
    Parameters
    ------
    u_sun : 3x1 numpy array
        unit vector from surface to sun in ECI frame
    u_obs : 3x1 numpy array
        unit vector from surface to observer in ECI frame
    brdf_params : dictionary
        contains additional parameters about reflectance property of surface
        and unit vectors defining body fixed directions in ECI frame
    
    Returns
    ------
    brdf : float
        bidirectional reflectance distribution function (>= 0.)    
    
    Reference
    ------    
    [1] Ashikhmin, et al., "Distribution-based BRDFs," 2007.
    
    [2] Linares, et al., "Space Object Shape Characterization and Tracking
    Using Light Curve and Angles Data," 2014.
        
    '''
    
    # Break out inputs
    d = brdf_params['d']
    s = brdf_params['s']
    rho = brdf_params['rho']
    Fo = brdf_params['Fo']
    nu = brdf_params['nu']
    nv = brdf_params['nv']
    norm_eci_hat = brdf_params['norm_eci_hat']
    u_eci_hat = brdf_params['u_eci_hat']
    v_eci_hat = brdf_params['v_eci_hat']
    
    Rdiff = d*rho
    Rspec = s*Fo
    
    # Compute half angle unit vector
    half_eci = (u_sun + u_obs)
    half_eci_hat = half_eci/np.linalg.norm(half_eci)
    
    # Compute dot products
    dot_n_sun = float(np.dot(norm_eci_hat.T, u_sun))
    dot_n_obs = float(np.dot(norm_eci_hat.T, u_obs))
    dot_n_h = float(np.dot(norm_eci_hat.T, half_eci_hat))
    dot_h_obs = float(np.dot(half_eci_hat.T, u_obs))
    dot_h_u = float(np.dot(half_eci_hat.T, u_eci_hat))
    dot_h_v = float(np.dot(half_eci_hat.T, v_eci_hat))
    
    # Compute diffuse BRDF
    brdf_diff = (28.*Rdiff/(23.*pi))*(1. - Rspec)*(1. - (1.-dot_n_sun/2.)**5.) \
                * (1. - (1.-dot_n_obs/2.)**5.)
    
    # Compute specular BRDF
    F = Rspec + (1. - Rspec)*(1. - dot_h_obs)**5.
    z = (nu*dot_h_u**2. + nv*dot_h_v**2.)/(1. - dot_n_sun**2.)
    brdf_spec = np.sqrt((nu + 1.)*(nv + 1.))/(8.*pi) \
                * dot_n_h**z*F/(dot_n_sun + dot_n_obs - (dot_n_sun*dot_n_obs))  
    
    # Compute total BRDF
    brdf = brdf_diff + brdf_spec
    
    return brdf
    
