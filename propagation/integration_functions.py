import numpy as np
from math import exp, pi, sin, cos, asin, atan2, acos
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

from skyfield.api import utc

sys.path.append('../')

from utilities.attitude import euler_dynamics
from utilities.attitude import quat_derivative
from utilities.attitude import quat_inverse
from utilities.attitude import quat_rotate
from utilities.constants import GME, Re, wE, J2E, SF, c_light, AU_km
from utilities.constants import stdatm_rho0, stdatm_ro, stdatm_H
from utilities.eop_functions import get_eop_data
from utilities.time_systems import jd2cent
from utilities.time_systems import utcdt2ttjd

###############################################################################
# This file contains functions to perform numerical integration using odeint
# Functions:
#
###############################################################################

###############################################################################
# Orbit Propagation Routines
###############################################################################

def int_twobody(X, t, spacecraftConfig, forcesCoeff, surfaces):
    '''
    This function works with odeint to propagate object assuming
    simple two-body dynamics.  No perturbations included.

    Parameters
    ------
    X : 6 element list
      cartesian state vector (Inertial Frame)
    t : m element list
      vector of times when output is desired
    args : tuple
        additional arguments

    Returns
    ------
    dX : 6 element list
      state derivative vector
    '''
    

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])

    # Compute radius
    r = np.linalg.norm([x, y, z])

    # Derivative vector
    dX = [0.]*6

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3
    dX[4] = -GM*y/r**3
    dX[5] = -GM*z/r**3

    return dX


def int_twobody_ukf(X, t, spacecraftConfig, forcesCoeff, surfaces):
    '''
    This function works with ode to propagate object assuming
    simple two-body dynamics.  No perturbations included.

    Parameters
    ------
    X : (n*(2n+1)) element list
      initial condition vector of cartesian state and sigma points
    t : m element list
      vector of times when output is desired
    inputs : dictionary
     input parameters

    Returns
    ------
    dX : (n*(2n+1)) element list
      derivative vector

    '''

    # Initialize
    dX = [0]*len(X)
    n = int((-1 + np.sqrt(1. + 8.*len(X)))/4.)

    for ind in range(0, 2*n+1):

        # Pull out relevant values from X
        x = float(X[ind*n])
        y = float(X[ind*n + 1])
        z = float(X[ind*n + 2])
        dx = float(X[ind*n + 3])
        dy = float(X[ind*n + 4])
        dz = float(X[ind*n + 5])

        # Compute radius
        r = np.linalg.norm([x, y, z])

        # Solve for components of dX
        dX[ind*n] = dx
        dX[ind*n + 1] = dy
        dX[ind*n + 2] = dz

        dX[ind*n + 3] = -GM*x/r**3
        dX[ind*n + 4] = -GM*y/r**3
        dX[ind*n + 5] = -GM*z/r**3

    return dX


def ode_twobody(t, X, params):
    '''
    This function works with ode to propagate object assuming
    simple two-body dynamics.  No perturbations included.

    Parameters
    ------
    X : 6 element list
      cartesian state vector (Inertial Frame)
    t : m element list
      vector of times when output is desired
    args : tuple
        additional arguments

    Returns
    ------
    dX : 6 element list
      state derivative vector
    '''
    

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])

    # Compute radius
    r = np.linalg.norm([x, y, z])

    # Derivative vector
    dX = [0.]*6

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3
    dX[4] = -GM*y/r**3
    dX[5] = -GM*z/r**3
    
    
    return dX


def ode_twobody_ukf(t, X, params):
    '''
    This function works with odeint to propagate object assuming
    simple two-body dynamics.  No perturbations included.

    Parameters
    ------
    X : (n*(2n+1)) element list
      initial condition vector of cartesian state and sigma points
    t : m element list
      vector of times when output is desired
    inputs : dictionary
     input parameters

    Returns
    ------
    dX : (n*(2n+1)) element list
      derivative vector

    '''
    
    # Initialize
    dX = [0]*len(X)
    n = int((-1 + np.sqrt(1. + 8.*len(X)))/4.)

    for ind in range(0, 2*n+1):

        # Pull out relevant values from X
        x = float(X[ind*n])
        y = float(X[ind*n + 1])
        z = float(X[ind*n + 2])
        dx = float(X[ind*n + 3])
        dy = float(X[ind*n + 4])
        dz = float(X[ind*n + 5])

        # Compute radius
        r = np.linalg.norm([x, y, z])

        # Solve for components of dX
        dX[ind*n] = dx
        dX[ind*n + 1] = dy
        dX[ind*n + 2] = dz

        dX[ind*n + 3] = -GM*x/r**3
        dX[ind*n + 4] = -GM*y/r**3
        dX[ind*n + 5] = -GM*z/r**3

    return dX


def ode_twobody_j2_drag(t, X, params):
    '''
    This function works with ode to propagate object assuming
    J2 and drag perturbations.

    Parameters
    ------
    X : 6 element list
      cartesian state vector (Inertial Frame)
    t : m element list
      vector of times when output is desired
    args : tuple
        additional arguments

    Returns
    ------
    dX : 6 element list
      state derivative vector
    '''

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])
    
    # Retrieve object parameters
    spacecraftConfig = params[0]
    forcesCoeff = params[1]
#    surfaces = params[2]
#    ephemeris = params[3]
#    ts = params[4]
#    eop_alldata = params[3]
    radius = spacecraftConfig['radius']
    mass = spacecraftConfig['mass']
    A_m = pi*radius**2./mass
    Cd = forcesCoeff['dragCoeff']

    # Compute radius
    r_vect = np.array([[x], [y], [z]])
    r = np.linalg.norm(r_vect)
    
    # Compute drag component
    # Find vector va of spacecraft relative to atmosphere
    v_vect = np.array([[dx], [dy], [dz]])
    w_vect = np.array([[0.], [0.], [wE]])
    va_vect = v_vect - np.cross(w_vect, r_vect, axis=0)
    va = np.linalg.norm(va_vect)
    va_x = float(va_vect[0])
    va_y = float(va_vect[1])
    va_z = float(va_vect[2])
    
    drag = -0.5*A_m*Cd*stdatm_rho0*exp(-(r-stdatm_ro)/stdatm_H)

    x_drag = drag*va*va_x
    y_drag = drag*va*va_y
    z_drag = drag*va*va_z
    
    # Compute J2 component
    x_j2 = - 1.5*J2E*Re**2.*GME*((x/r**5.) - (5.*x*z**2./r**7.))
    y_j2 = - 1.5*J2E*Re**2.*GME*((y/r**5.) - (5.*y*z**2./r**7.))
    z_j2 = - 1.5*J2E*Re**2.*GME*((3.*z/r**5.) - (5.*z**3./r**7.)) 
    

    # Derivative vector
    dX = [0.]*6

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz
    
    dX[3] = -GME*x/r**3. + x_j2 + x_drag
    dX[4] = -GME*y/r**3. + y_j2 + y_drag
    dX[5] = -GME*z/r**3. + z_j2 + z_drag
    
    
    return dX


def ode_twobody_j2_drag_srp(t, X, params):
    '''
    This function works with ode to propagate object assuming
    J2, drag, and SRP perturbations.

    Parameters
    ------
    X : 6 element list
      cartesian state vector (Inertial Frame)
    t : m element list
      vector of times when output is desired
    args : tuple
        additional arguments

    Returns
    ------
    dX : 6 element list
      state derivative vector
    '''

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])
    
    # Retrieve object parameters
    spacecraftConfig = params[0]
    forcesCoeff = params[1]
    surfaces = params[2]
#    ephemeris = params[3]
#    ts = params[4]
#    eop_alldata = params[3]
    radius = spacecraftConfig['radius']
    mass = spacecraftConfig['mass']
    A_m = pi*radius**2./mass
    Cd = forcesCoeff['dragCoeff']
    rho_diff = surfaces[0]['brdf_params']['rho']
    UTC0 = spacecraftConfig['time']

    # Sun and earth data
#    earth = ephemeris['earth']
#    sun = ephemeris['sun']
    
#    # Skyfield time and sun position
    UTC = UTC0 + timedelta(seconds=t)
#    UTC_skyfield = ts.utc(UTC.replace(tzinfo=utc))
#    sun_gcrf = earth.at(UTC_skyfield).observe(sun).position.km
#    sun_gcrf = np.reshape(sun_gcrf, (3,1))

    # Compute radius
    r_vect = np.array([[x], [y], [z]])
    r = np.linalg.norm(r_vect)
    
    # Compute drag component
    # Find vector va of spacecraft relative to atmosphere
    v_vect = np.array([[dx], [dy], [dz]])
    w_vect = np.array([[0.], [0.], [wE]])
    va_vect = v_vect - np.cross(w_vect, r_vect, axis=0)
    va = np.linalg.norm(va_vect)
    va_x = float(va_vect[0])
    va_y = float(va_vect[1])
    va_z = float(va_vect[2])
    
    drag = -0.5*A_m*Cd*stdatm_rho0*exp(-(r-stdatm_ro)/stdatm_H)

    x_drag = drag*va*va_x
    y_drag = drag*va*va_y
    z_drag = drag*va*va_z
    
    # Compute SRP component
    
    # Compute current solar position in ECI
    # Time in days since J2000
    # Compute TT in JD format
#    EOP_data = get_eop_data(eop_alldata, UTC)
    TT_JD = utcdt2ttjd(UTC, 37.)
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = jd2cent(TT_JD)

    # Convert to centuries and compute mean longitude of the sun
    lam = 280.460 + 36000.771*TT_cent
    lam = (lam % 360.) * pi/180.  # rad

    # Compute mean anomaly of sun
    M = 357.5277233 + 35999.05034*TT_cent
    M = (M % 360.) * pi/180.  # rad

    # Compute ecliptic long/lat
    lam_ec = lam + (1.914666471*sin(M) + 0.019994643*sin(2.*M))*pi/180.  # rad

    # Compute distance to sun in AU and obliquity of ecliptic plane
    r_AU = 1.000140612 - 0.016708617*cos(M) - 0.000139589*cos(2.*M)  # AU
    ep = (23.439291 - 0.0130042*TT_cent) * pi/180.

    # Compute sun position vector in Mean Equator of Date (MOD) Frame
    r_sun = r_AU*AU_km*np.array([[cos(lam_ec)], [cos(ep)*sin(lam_ec)],
                                 [sin(ep)*sin(lam_ec)]])
    rg_sun = np.linalg.norm(r_sun)

    # Compute ra/dec of sun
    dec = asin(sin(ep)*sin(lam_ec))
    ra = atan2((cos(ep)*sin(lam_ec)/cos(dec)), (cos(lam_ec)/cos(dec)))

    # Convert to ECI
    sun_x = rg_sun*cos(dec)*cos(ra)
    sun_y = rg_sun*cos(dec)*sin(ra)
    sun_z = rg_sun*sin(dec)
    sun_gcrf = np.array([[sun_x], [sun_y], [sun_z]])
    
    # Compute current position vector from sun
    r_fromsun = r_vect - sun_gcrf
    d = np.linalg.norm(r_fromsun)
    u_sun = -r_fromsun/d

    # SRP acceleration
    Cr = 1 + (2./3.)*pi*rho_diff
    beta = Cr*A_m
    a_srp = -(SF/(c_light*(d/AU_km)**2.))*beta*u_sun  # km/s^2
    
    # check for eclipse
    u_sat = r_vect/r
    sun_angle = acos(np.dot(u_sun.flatten(), -u_sat.flatten()))
    half_cone = asin(Re/r)
    if sun_angle < half_cone:
        a_srp = np.zeros((3,1))
        
    x_srp = float(a_srp[0])
    y_srp = float(a_srp[1])
    z_srp = float(a_srp[2])
    
    # Compute J2 component
    x_j2 = - 1.5*J2E*Re**2.*GME*((x/r**5.) - (5.*x*z**2./r**7.))
    y_j2 = - 1.5*J2E*Re**2.*GME*((y/r**5.) - (5.*y*z**2./r**7.))
    z_j2 = - 1.5*J2E*Re**2.*GME*((3.*z/r**5.) - (5.*z**3./r**7.)) 
    

    # Derivative vector
    dX = [0.]*6

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz
    
    dX[3] = -GME*x/r**3. + x_j2 + x_drag + x_srp
    dX[4] = -GME*y/r**3. + y_j2 + y_drag + y_srp
    dX[5] = -GME*z/r**3. + z_j2 + z_drag + z_srp
    
    
    return dX


def ode_twobody_j2_drag_ukf(t, X, params):
    '''
    This function works with ode to propagate object assuming
    simple including J2 and drag perturbations.

    Parameters
    ------
    X : 6 element list
      cartesian state vector (Inertial Frame)
    t : m element list
      vector of times when output is desired
    args : tuple
        additional arguments

    Returns
    ------
    dX : 6 element list
      state derivative vector
    '''

    
    
    # Retrieve object parameters
    spacecraftConfig = params[0]
    forcesCoeff = params[1]
#    surfaces = params[2]
#    ephemeris = params[3]
#    ts = params[4]
#    eop_alldata = params[3]
    radius = spacecraftConfig['radius']
    mass = spacecraftConfig['mass']
    A_m = pi*radius**2./mass
    Cd = forcesCoeff['dragCoeff']    
    
    # Initialize
    dX = [0]*len(X)
    n = int((-1 + np.sqrt(1. + 8.*len(X)))/4.)

    for ind in range(0, 2*n+1):

        # Pull out relevant values from X
        x = float(X[ind*n])
        y = float(X[ind*n + 1])
        z = float(X[ind*n + 2])
        dx = float(X[ind*n + 3])
        dy = float(X[ind*n + 4])
        dz = float(X[ind*n + 5])

        # Compute radius
        r_vect = np.array([[x], [y], [z]])
        r = np.linalg.norm(r_vect)
        
        # Compute drag component
        # Find vector va of spacecraft relative to atmosphere
        v_vect = np.array([[dx], [dy], [dz]])
        w_vect = np.array([[0.], [0.], [wE]])
        va_vect = v_vect - np.cross(w_vect, r_vect, axis=0)
        va = np.linalg.norm(va_vect)
        va_x = float(va_vect[0])
        va_y = float(va_vect[1])
        va_z = float(va_vect[2])
        
        drag = -0.5*A_m*Cd*stdatm_rho0*exp(-(r-stdatm_ro)/stdatm_H)
    
        x_drag = drag*va*va_x
        y_drag = drag*va*va_y
        z_drag = drag*va*va_z

        # Compute J2 component
        x_j2 = - 1.5*J2E*Re**2.*GME*((x/r**5.) - (5.*x*z**2./r**7.))
        y_j2 = - 1.5*J2E*Re**2.*GME*((y/r**5.) - (5.*y*z**2./r**7.))
        z_j2 = - 1.5*J2E*Re**2.*GME*((3.*z/r**5.) - (5.*z**3./r**7.)) 

        # Derivative vector
        dX[ind*n] = dx
        dX[ind*n + 1] = dy
        dX[ind*n + 2] = dz
        
        dX[ind*n + 3] = -GME*x/r**3. + x_j2 + x_drag
        dX[ind*n + 4] = -GME*y/r**3. + y_j2 + y_drag
        dX[ind*n + 5] = -GME*z/r**3. + z_j2 + z_drag
    

    return dX


def ode_twobody_j2_drag_srp_ukf(t, X, params):
    '''
    This function works with ode to propagate object assuming
    simple two-body dynamics.  No perturbations included.

    Parameters
    ------
    X : 6 element list
      cartesian state vector (Inertial Frame)
    t : m element list
      vector of times when output is desired
    args : tuple
        additional arguments

    Returns
    ------
    dX : 6 element list
      state derivative vector
    '''

    
    
    # Retrieve object parameters
    spacecraftConfig = params[0]
    forcesCoeff = params[1]
    surfaces = params[2]
#    ephemeris = params[3]
#    ts = params[4]
#    eop_alldata = params[3]
    radius = spacecraftConfig['radius']
    mass = spacecraftConfig['mass']
    A_m = pi*radius**2./mass
    Cd = forcesCoeff['dragCoeff']
    rho_diff = surfaces[0]['brdf_params']['rho']
    UTC0 = spacecraftConfig['time']
    UTC = UTC0 + timedelta(seconds=t)
    
    # Compute current solar position in ECI
    # Time in days since J2000
    TT_JD = utcdt2ttjd(UTC, 37.)
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = jd2cent(TT_JD)

    # Convert to centuries and compute mean longitude of the sun
    lam = 280.460 + 36000.771*TT_cent
    lam = (lam % 360.) * pi/180.  # rad

    # Compute mean anomaly of sun
    M = 357.5277233 + 35999.05034*TT_cent
    M = (M % 360.) * pi/180.  # rad

    # Compute ecliptic long/lat
    lam_ec = lam + (1.914666471*sin(M) + 0.019994643*sin(2.*M))*pi/180.  # rad

    # Compute distance to sun in AU and obliquity of ecliptic plane
    r_AU = 1.000140612 - 0.016708617*cos(M) - 0.000139589*cos(2.*M)  # AU
    ep = (23.439291 - 0.0130042*TT_cent) * pi/180.

    # Compute sun position vector in Mean Equator of Date (MOD) Frame
    r_sun = r_AU*AU_km*np.array([[cos(lam_ec)], [cos(ep)*sin(lam_ec)],
                                 [sin(ep)*sin(lam_ec)]])
    rg_sun = np.linalg.norm(r_sun)

    # Compute ra/dec of sun
    dec = asin(sin(ep)*sin(lam_ec))
    ra = atan2((cos(ep)*sin(lam_ec)/cos(dec)), (cos(lam_ec)/cos(dec)))

    # Convert to ECI
    sun_x = rg_sun*cos(dec)*cos(ra)
    sun_y = rg_sun*cos(dec)*sin(ra)
    sun_z = rg_sun*sin(dec)
    sun_gcrf = np.array([[sun_x], [sun_y], [sun_z]])
    
    
    # Initialize
    dX = [0]*len(X)
    n = int((-1 + np.sqrt(1. + 8.*len(X)))/4.)

    for ind in range(0, 2*n+1):

        # Pull out relevant values from X
        x = float(X[ind*n])
        y = float(X[ind*n + 1])
        z = float(X[ind*n + 2])
        dx = float(X[ind*n + 3])
        dy = float(X[ind*n + 4])
        dz = float(X[ind*n + 5])

        # Compute radius
        r_vect = np.array([[x], [y], [z]])
        r = np.linalg.norm(r_vect)
        
        # Compute drag component
        # Find vector va of spacecraft relative to atmosphere
        v_vect = np.array([[dx], [dy], [dz]])
        w_vect = np.array([[0.], [0.], [wE]])
        va_vect = v_vect - np.cross(w_vect, r_vect, axis=0)
        va = np.linalg.norm(va_vect)
        va_x = float(va_vect[0])
        va_y = float(va_vect[1])
        va_z = float(va_vect[2])
        
        drag = -0.5*A_m*Cd*stdatm_rho0*exp(-(r-stdatm_ro)/stdatm_H)
    
        x_drag = drag*va*va_x
        y_drag = drag*va*va_y
        z_drag = drag*va*va_z
        
        # Compute current position vector from sun
        r_fromsun = r_vect - sun_gcrf
        d = np.linalg.norm(r_fromsun)
        u_sun = -r_fromsun/d
    
        # SRP acceleration
        Cr = 1 + (2./3.)*pi*rho_diff
        beta = Cr*A_m
        a_srp = -(SF/(c_light*(d/AU_km)**2.))*beta*u_sun  # km/s^2
        
        # check for eclipse
        u_sat = r_vect/r
        sun_angle = acos(np.dot(u_sun.flatten(), -u_sat.flatten()))
        half_cone = asin(Re/r)
        if sun_angle < half_cone:
            a_srp = np.zeros((3,1))
            
        x_srp = float(a_srp[0])
        y_srp = float(a_srp[1])
        z_srp = float(a_srp[2])
        
        # Compute J2 component
        x_j2 = - 1.5*J2*Re**2.*GM*((x/r**5.) - (5.*x*z**2./r**7.))
        y_j2 = - 1.5*J2*Re**2.*GM*((y/r**5.) - (5.*y*z**2./r**7.))
        z_j2 = - 1.5*J2*Re**2.*GM*((3.*z/r**5.) - (5.*z**3./r**7.)) 

        # Derivative vector
        dX[ind*n] = dx
        dX[ind*n + 1] = dy
        dX[ind*n + 2] = dz
        
        dX[ind*n + 3] = -GM*x/r**3. + x_j2 + x_drag + x_srp
        dX[ind*n + 4] = -GM*y/r**3. + y_j2 + y_drag + y_srp
        dX[ind*n + 5] = -GM*z/r**3. + z_j2 + z_drag + z_srp
    

    return dX


###############################################################################
# Attitude Dynamics
###############################################################################


def int_euler_dynamics_notorque(X, t, spacecraftConfig, forcesCoeff, surfaces):
    
    
    # Initialize
    dX = [0.]*len(X)
    
    # Current attitude state
    q_BN = np.reshape(X[0:4], (4,1))
    w_BN = np.reshape(X[4:7], (3,1))
    
    # Moment of inertia
    I = spacecraftConfig['moi']
    
    # Torque vector
    L = np.zeros((3,1))
    
    # Compute derivative vector
    q_BN_dot = quat_derivative(q_BN, w_BN)
    w_BN_dot = euler_dynamics(w_BN, I, L)
    
    dX[0:4] = q_BN_dot.flatten()
    dX[4:7] = w_BN_dot.flatten() 
    
    return dX


def int_twobody_6dof_notorque(X, t, spacecraftConfig, forcesCoeff, surfaces):
    '''
    This function works with odeint to propagate object assuming
    simple two-body dynamics and including attitude states assuming
    no torques.  No perturbations included.

    Parameters
    ------
    X : 6 element list
      cartesian state vector (Inertial Frame)
    t : m element list
      vector of times when output is desired
    args : tuple
        additional arguments

    Returns
    ------
    dX : 6 element list
      state derivative vector
    '''
    
    # Initialize
    dX = [0.]*len(X)

    # Position states
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])

    # Compute radius
    r = np.linalg.norm([x, y, z])
    
    # Attitude states
    q_BN = np.reshape(X[6:10], (4,1))
    w_BN = np.reshape(X[10:13], (3,1))
    
    # Moment of inertia
    I = spacecraftConfig['moi']
    
    # Torque vector
    L = np.zeros((3,1))
    
    # Compute derivative vector
    q_BN_dot = quat_derivative(q_BN, w_BN)
    w_BN_dot = euler_dynamics(w_BN, I, L)

    # Derivative vector
    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3
    dX[4] = -GM*y/r**3
    dX[5] = -GM*z/r**3
    
    dX[6:10] = q_BN_dot.flatten()
    dX[10:13] = w_BN_dot.flatten() 
    


    return dX


def ode_twobody_6dof_notorque(t, X, params):
    '''
    This function works with ode to propagate object assuming
    simple two-body dynamics and including attitude states assuming
    no torques.  No perturbations included.

    Parameters
    ------
    X : 6 element list
      cartesian state vector (Inertial Frame)
    t : m element list
      vector of times when output is desired
    args : tuple
        additional arguments

    Returns
    ------
    dX : 6 element list
      state derivative vector
    '''
    
    # Input parameters
    spacecraftConfig = params[0]
    
    # Initialize
    dX = [0.]*len(X)

    # Position states
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])

    # Compute radius
    r = np.linalg.norm([x, y, z])
    
    # Attitude states
    q_BN = np.reshape(X[6:10], (4,1))
    w_BN = np.reshape(X[10:13], (3,1))
    
    # Moment of inertia
    I = spacecraftConfig['moi']
    
    # Torque vector
    L = np.zeros((3,1))
    
    # Compute derivative vector
    q_BN_dot = quat_derivative(q_BN, w_BN)
    w_BN_dot = euler_dynamics(w_BN, I, L)

    # Derivative vector
    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3
    dX[4] = -GM*y/r**3
    dX[5] = -GM*z/r**3
    
    dX[6:10] = q_BN_dot.flatten()
    dX[10:13] = w_BN_dot.flatten() 

    return dX


def ode_twobody_6dof_notorque_ukf(t, X, params):
    '''
    This function works with ode to propagate object assuming
    simple two-body dynamics and including attitude states assuming
    no torques.  No perturbations included.

    Parameters
    ------
    X : 6 element list
      cartesian state vector (Inertial Frame)
    t : m element list
      vector of times when output is desired
    args : tuple
        additional arguments

    Returns
    ------
    dX : 6 element list
      state derivative vector
    '''
    
    # Input parameters
    spacecraftConfig = params[0]
    
    # Initialize
    dX = [0.]*len(X)
    n = 13

    # Loop over sigma points
    for ind in range(25):

        # Position states
        x = float(X[ind*n])
        y = float(X[ind*n + 1])
        z = float(X[ind*n + 2])
        dx = float(X[ind*n + 3])
        dy = float(X[ind*n + 4])
        dz = float(X[ind*n + 5])
    
        # Compute radius
        r = np.linalg.norm([x, y, z])
        
        # Attitude states
        qind = ind*n+6
        wind = ind*n+10
        q_BN = np.reshape(X[qind:qind+4], (4,1))
        w_BN = np.reshape(X[wind:wind+3], (3,1))
        
        # Moment of inertia
        I = spacecraftConfig['moi']
        
        # Torque vector
        L = np.zeros((3,1))
        
        # Compute derivative vector
        q_BN_dot = quat_derivative(q_BN, w_BN)
        w_BN_dot = euler_dynamics(w_BN, I, L)
    
        # Derivative vector
        dX[ind*n] = dx
        dX[ind*n + 1] = dy
        dX[ind*n + 2] = dz

        dX[ind*n + 3] = -GM*x/r**3
        dX[ind*n + 4] = -GM*y/r**3
        dX[ind*n + 5] = -GM*z/r**3
        
        dX[qind:qind+4] = q_BN_dot.flatten()
        dX[wind:wind+3] = w_BN_dot.flatten() 

    return dX


def ode_twobody_j2_drag_srp_notorque(t, X, params):
    '''
    This function works with ode to propagate object assuming
    simple two-body dynamics.  No perturbations included.

    Parameters
    ------
    X : 6 element list
      cartesian state vector (Inertial Frame)
    t : m element list
      vector of times when output is desired
    args : tuple
        additional arguments

    Returns
    ------
    dX : 6 element list
      state derivative vector
    '''
    
    # Retrieve object parameters
    spacecraftConfig = params[0]
    forcesCoeff = params[1]
    surfaces = params[2]

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])
    
    # Attitude states
    q_BN = np.reshape(X[6:10], (4,1))
    w_BN = np.reshape(X[10:13], (3,1))
    q_NB = quat_inverse(q_BN)
    
    # Moment of inertia
    I = spacecraftConfig['moi']
    
    # Torque vector
    L = np.zeros((3,1))
    
    # Compute derivative vector
    q_BN_dot = quat_derivative(q_BN, w_BN)
    w_BN_dot = euler_dynamics(w_BN, I, L)
    
    # Full body parameters
    mass = spacecraftConfig['mass']
    Cd = forcesCoeff['dragCoeff']
    emissivity = forcesCoeff['emissivity']
    UTC0 = spacecraftConfig['time']
    
    # Current time
    UTC = UTC0 + timedelta(seconds=t)

    # Compute radius
    r_vect = np.array([[x], [y], [z]])
    r = np.linalg.norm(r_vect)

    # Compute current solar position in ECI
    TT_JD = utcdt2ttjd(UTC, 37.)
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = jd2cent(TT_JD)

    # Convert to centuries and compute mean longitude of the sun
    lam = 280.460 + 36000.771*TT_cent
    lam = (lam % 360.) * pi/180.  # rad

    # Compute mean anomaly of sun
    M = 357.5277233 + 35999.05034*TT_cent
    M = (M % 360.) * pi/180.  # rad

    # Compute ecliptic long/lat
    lam_ec = lam + (1.914666471*sin(M) + 0.019994643*sin(2.*M))*pi/180.  # rad

    # Compute distance to sun in AU and obliquity of ecliptic plane
    r_AU = 1.000140612 - 0.016708617*cos(M) - 0.000139589*cos(2.*M)  # AU
    ep = (23.439291 - 0.0130042*TT_cent) * pi/180.

    # Compute sun position vector in Mean Equator of Date (MOD) Frame
    r_sun = r_AU*AU_km*np.array([[cos(lam_ec)], [cos(ep)*sin(lam_ec)],
                                 [sin(ep)*sin(lam_ec)]])
    rg_sun = np.linalg.norm(r_sun)

    # Compute ra/dec of sun
    dec = asin(sin(ep)*sin(lam_ec))
    ra = atan2((cos(ep)*sin(lam_ec)/cos(dec)), (cos(lam_ec)/cos(dec)))

    # Convert to ECI
    sun_x = rg_sun*cos(dec)*cos(ra)
    sun_y = rg_sun*cos(dec)*sin(ra)
    sun_z = rg_sun*sin(dec)
    sun_gcrf = np.array([[sun_x], [sun_y], [sun_z]])
    
    # Compute current position vector from sun
    r_fromsun = r_vect - sun_gcrf
    dsun = np.linalg.norm(r_fromsun)
    u_sun = -r_fromsun/dsun
    
    # Compute drag component
    # Find vector va of spacecraft relative to atmosphere
    v_vect = np.array([[dx], [dy], [dz]])
    w_vect = np.array([[0.], [0.], [wE]])
    va_vect = v_vect - np.cross(w_vect, r_vect, axis=0)
    va = np.linalg.norm(va_vect)
    va_hat = va_vect/va
    va_x = float(va_vect[0])
    va_y = float(va_vect[1])
    va_z = float(va_vect[2])
    
    # Compute total surface area perpendicular to atmosphere velocity
    drag_area = 0.
    a_srp = np.zeros((3,1))
    for ii in surfaces:
        area = surfaces[ii]['area']
        norm_body_hat = surfaces[ii]['norm_body_hat']
        norm_eci_hat = quat_rotate(q_NB, norm_body_hat)
        va_dot = float(np.dot(va_hat.flatten(), norm_eci_hat.flatten()))        
        if va_dot > 0:
            drag_area += area*va_dot
            
        # check for eclipse
        u_sat = r_vect/r
        if r < Re:
            continue
        
        sun_angle = acos(np.dot(u_sun.flatten(), -u_sat.flatten()))
        half_cone = asin(Re/r)
        if sun_angle < half_cone:
            continue
            
        d = surfaces[ii]['brdf_params']['d']
        s = surfaces[ii]['brdf_params']['s']
        rho = surfaces[ii]['brdf_params']['rho']
        Fo = surfaces[ii]['brdf_params']['Fo']
        
        Rdiff = d*rho
        Rspec = s*Fo
        Rabs = 1. - Rdiff - Rspec
    
        sun_dot = float(np.dot(u_sun.flatten(), norm_eci_hat.flatten())) 
        if sun_dot > 0.:
            u_srp = 2.*((Rdiff/3.) + (Rabs*emissivity/3.) + Rspec*sun_dot) \
                * norm_eci_hat + (1. - Rspec)*u_sun
                
            a_srp += -(SF*area*sun_dot**2./(mass*c_light*(dsun/AU_km)**2.)) * u_srp


    x_srp = float(a_srp[0])
    y_srp = float(a_srp[1])
    z_srp = float(a_srp[2])
    
    A_m = drag_area/mass
    drag = -0.5*A_m*Cd*stdatm_rho0*exp(-(r-stdatm_ro)/stdatm_H)

    x_drag = drag*va*va_x
    y_drag = drag*va*va_y
    z_drag = drag*va*va_z

    # Compute J2 component
    x_j2 = - 1.5*J2*Re**2.*GM*((x/r**5.) - (5.*x*z**2./r**7.))
    y_j2 = - 1.5*J2*Re**2.*GM*((y/r**5.) - (5.*y*z**2./r**7.))
    z_j2 = - 1.5*J2*Re**2.*GM*((3.*z/r**5.) - (5.*z**3./r**7.)) 
    
    # Derivative vector
    dX = [0.]*len(X)

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz
    
    dX[3] = -GM*x/r**3. + x_j2 + x_drag + x_srp
    dX[4] = -GM*y/r**3. + y_j2 + y_drag + y_srp
    dX[5] = -GM*z/r**3. + z_j2 + z_drag + z_srp
    
    dX[6:10] = q_BN_dot.flatten()
    dX[10:13] = w_BN_dot.flatten() 
    
    
    return dX


def ode_twobody_j2_drag_srp_notorque_ukf(t, X, params):
    '''
    This function works with ode to propagate object assuming
    simple two-body dynamics.  No perturbations included.

    Parameters
    ------
    X : 6 element list
      cartesian state vector (Inertial Frame)
    t : m element list
      vector of times when output is desired
    args : tuple
        additional arguments

    Returns
    ------
    dX : 6 element list
      state derivative vector
    '''
    
    # Derivative vector
    dX = [0.]*len(X)
    
    # Retrieve object parameters
    spacecraftConfig = params[0]
    forcesCoeff = params[1]
    surfaces = params[2]
    
    # Attitude states
    q_BN = np.reshape(X[6:10], (4,1))
    w_BN = np.reshape(X[10:13], (3,1))
    q_NB = quat_inverse(q_BN)
    
    # Moment of inertia
    I = spacecraftConfig['moi']
    
    # Torque vector
    L = np.zeros((3,1))
    
    # Compute derivative vector
    q_BN_dot = quat_derivative(q_BN, w_BN)
    w_BN_dot = euler_dynamics(w_BN, I, L)
    
    # Full body parameters
    mass = spacecraftConfig['mass']
    Cd = forcesCoeff['dragCoeff']
    emissivity = forcesCoeff['emissivity']
    UTC0 = spacecraftConfig['time']
    
    # Current time
    UTC = UTC0 + timedelta(seconds=t)

    # Compute current solar position in ECI
    TT_JD = utcdt2ttjd(UTC, 37.)
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = jd2cent(TT_JD)

    # Convert to centuries and compute mean longitude of the sun
    lam = 280.460 + 36000.771*TT_cent
    lam = (lam % 360.) * pi/180.  # rad

    # Compute mean anomaly of sun
    M = 357.5277233 + 35999.05034*TT_cent
    M = (M % 360.) * pi/180.  # rad

    # Compute ecliptic long/lat
    lam_ec = lam + (1.914666471*sin(M) + 0.019994643*sin(2.*M))*pi/180.  # rad

    # Compute distance to sun in AU and obliquity of ecliptic plane
    r_AU = 1.000140612 - 0.016708617*cos(M) - 0.000139589*cos(2.*M)  # AU
    ep = (23.439291 - 0.0130042*TT_cent) * pi/180.

    # Compute sun position vector in Mean Equator of Date (MOD) Frame
    r_sun = r_AU*AU_km*np.array([[cos(lam_ec)], [cos(ep)*sin(lam_ec)],
                                 [sin(ep)*sin(lam_ec)]])
    rg_sun = np.linalg.norm(r_sun)

    # Compute ra/dec of sun
    dec = asin(sin(ep)*sin(lam_ec))
    ra = atan2((cos(ep)*sin(lam_ec)/cos(dec)), (cos(lam_ec)/cos(dec)))

    # Convert to ECI
    sun_x = rg_sun*cos(dec)*cos(ra)
    sun_y = rg_sun*cos(dec)*sin(ra)
    sun_z = rg_sun*sin(dec)
    sun_gcrf = np.array([[sun_x], [sun_y], [sun_z]])
    
    # State Vector
    for kk in range(13):
        
        if kk == 0:
            x = float(X[0])
            y = float(X[1])
            z = float(X[2])
            dx = float(X[3])
            dy = float(X[4])
            dz = float(X[5])
            
        else:
            x = float(X[(kk-1)*6+13])
            y = float(X[(kk-1)*6+14])
            z = float(X[(kk-1)*6+15])
            dx = float(X[(kk-1)*6+16])
            dy = float(X[(kk-1)*6+17])
            dz = float(X[(kk-1)*6+18])
    
        # Compute radius
        r_vect = np.array([[x], [y], [z]])
        r = np.linalg.norm(r_vect)
        
        # Compute current position vector from sun
        r_fromsun = r_vect - sun_gcrf
        dsun = np.linalg.norm(r_fromsun)
        u_sun = -r_fromsun/dsun
        
        # Compute drag component
        # Find vector va of spacecraft relative to atmosphere
        v_vect = np.array([[dx], [dy], [dz]])
        w_vect = np.array([[0.], [0.], [wE]])
        va_vect = v_vect - np.cross(w_vect, r_vect, axis=0)
        va = np.linalg.norm(va_vect)
        va_hat = va_vect/va
        va_x = float(va_vect[0])
        va_y = float(va_vect[1])
        va_z = float(va_vect[2])
        
        # Compute total surface area perpendicular to atmosphere velocity
        drag_area = 0.
        a_srp = np.zeros((3,1))
        for ii in surfaces:
            area = surfaces[ii]['area']
            norm_body_hat = surfaces[ii]['norm_body_hat']
            norm_eci_hat = quat_rotate(q_NB, norm_body_hat)
            va_dot = float(np.dot(va_hat.flatten(), norm_eci_hat.flatten()))        
            if va_dot > 0:
                drag_area += area*va_dot
                
            # Check for eclipse
            u_sat = r_vect/r
            if r < Re:
                continue
            
            sun_angle = acos(np.dot(u_sun.flatten(), -u_sat.flatten()))
            half_cone = asin(Re/r)
            if sun_angle < half_cone:
                continue
                
            d = surfaces[ii]['brdf_params']['d']
            s = surfaces[ii]['brdf_params']['s']
            rho = surfaces[ii]['brdf_params']['rho']
            Fo = surfaces[ii]['brdf_params']['Fo']
            
            Rdiff = d*rho
            Rspec = s*Fo
            Rabs = 1. - Rdiff - Rspec
        
            sun_dot = float(np.dot(u_sun.flatten(), norm_eci_hat.flatten())) 
            if sun_dot > 0.:
                u_srp = 2.*((Rdiff/3.) + (Rabs*emissivity/3.) + Rspec*sun_dot) \
                    * norm_eci_hat + (1. - Rspec)*u_sun
                    
                a_srp += -(SF*area*sun_dot**2./(mass*c_light*(dsun/AU_km)**2.)) * u_srp
    
    
        x_srp = float(a_srp[0])
        y_srp = float(a_srp[1])
        z_srp = float(a_srp[2])
        
        A_m = drag_area/mass
        drag = -0.5*A_m*Cd*stdatm_rho0*exp(-(r-stdatm_ro)/stdatm_H)
    
        x_drag = drag*va*va_x
        y_drag = drag*va*va_y
        z_drag = drag*va*va_z
    
        # Compute J2 component
        x_j2 = - 1.5*J2*Re**2.*GM*((x/r**5.) - (5.*x*z**2./r**7.))
        y_j2 = - 1.5*J2*Re**2.*GM*((y/r**5.) - (5.*y*z**2./r**7.))
        z_j2 = - 1.5*J2*Re**2.*GM*((3.*z/r**5.) - (5.*z**3./r**7.)) 
    
        
        if kk == 0:
            dX[0] = dx
            dX[1] = dy
            dX[2] = dz
            
            dX[3] = -GM*x/r**3. + x_j2 + x_drag + x_srp
            dX[4] = -GM*y/r**3. + y_j2 + y_drag + y_srp
            dX[5] = -GM*z/r**3. + z_j2 + z_drag + z_srp
            
            dX[6:10] = q_BN_dot.flatten()
            dX[10:13] = w_BN_dot.flatten()
            
        else:            
            dX[(kk-1)*6+13] = dx
            dX[(kk-1)*6+14] = dy
            dX[(kk-1)*6+15] = dz
            
            dX[(kk-1)*6+16] = -GM*x/r**3. + x_j2 + x_drag + x_srp
            dX[(kk-1)*6+17] = -GM*y/r**3. + y_j2 + y_drag + y_srp
            dX[(kk-1)*6+18] = -GM*z/r**3. + z_j2 + z_drag + z_srp
            
#    print(X[10:13])
#    print(dX[10:13])
    
    return dX


#def int_twobody_diff_entropy(X, t, inputs):
#    '''
#    This function works with odeint to propagate object assuming
#    simple two-body dynamics.  No perturbations included.
#    Partials for the STM dynamics are included. Differential Entropy is
#    included in the state vector and STM.
#
#    Parameters
#    ------
#    X : (n+n^2) element list
#      initial condition vector of entropy and cartesian state and
#      STM (Inertial Frame)
#    t : m element list
#      vector of times when output is desired
#    inputs : dictionary
#      input parameters
#
#    Returns
#    ------
#    dX : (n+n^2) element list
#      derivative vector
#    '''
#
#    # Input data
#    GM = inputs['GM']
#
#    # State Vector
#    n = 7
#    ej = X[0]
#    x = X[1]
#    y = X[2]
#    z = X[3]
#    dx = X[4]
#    dy = X[5]
#    dz = X[6]
#
#    # Compute radius
#    r = np.linalg.norm([x, y, z])
#
#    # Find elements of A matrix
#    xx_cf = -GM/r**3 + 3.*GM*x**2/r**5
#    xy_cf = 3.*GM*x*y/r**5
#    xz_cf = 3.*GM*x*z/r**5
#    yy_cf = -GM/r**3 + 3.*GM*y**2/r**5
#    yx_cf = xy_cf
#    yz_cf = 3.*GM*y*z/r**5
#    zz_cf = -GM/r**3 + 3.*GM*z**2/r**5
#    zx_cf = xz_cf
#    zy_cf = yz_cf
#
#    # Generate A matrix
#    A = np.zeros((n, n))
#
#    A[0,3] = 1.
#    A[1,4] = 1.
#    A[2,5] = 1.
#
#    A[3,0] = xx_cf
#    A[3,1] = xy_cf
#    A[3,2] = xz_cf
#
#    A[4,0] = yx_cf
#    A[4,1] = yy_cf
#    A[4,2] = yz_cf
#
#    A[5,0] = zx_cf
#    A[5,1] = zy_cf
#    A[5,2] = zz_cf
#
#    # Derivative vector
#    dX = np.zeros((7, 1))
#
#    dX[0] = np.trace(A)
#
#    dX[1] = dx
#    dX[2] = dy
#    dX[3] = dz
#
#    dX[4] = -GM*x/r**3
#    dX[5] = -GM*y/r**3
#    dX[6] = -GM*z/r**3
#
#    dX = dX.flatten()
#
#    return dX
#



#def int_twobody_stm(X, t, inputs):
#    '''
#    This function works with odeint to propagate object assuming
#    simple two-body dynamics.  No perturbations included.
#    Partials for the STM dynamics are included.
#
#    Parameters
#    ------
#    X : (n+n^2) element list
#      initial condition vector of cartesian state and STM (Inertial Frame)
#    t : m element list
#      vector of times when output is desired
#    inputs : dictionary
#      input parameters
#
#    Returns
#    ------
#    dX : (n+n^2) element list
#      derivative vector
#    '''
#
#    # Input data
#    GM = inputs['GM']
#
#    # Compute number of states
#    n = (-1 + np.sqrt(1 + 4*len(X)))/2
#
#    # State Vector
#    x = X[0]
#    y = X[1]
#    z = X[2]
#    dx = X[3]
#    dy = X[4]
#    dz = X[5]
#
#    # Compute radius
#    r = np.linalg.norm([x, y, z])
#
#    # Find elements of A matrix
#    xx_cf = -GM/r**3 + 3.*GM*x**2/r**5
#    xy_cf = 3.*GM*x*y/r**5
#    xz_cf = 3.*GM*x*z/r**5
#    yy_cf = -GM/r**3 + 3.*GM*y**2/r**5
#    yx_cf = xy_cf
#    yz_cf = 3.*GM*y*z/r**5
#    zz_cf = -GM/r**3 + 3.*GM*z**2/r**5
#    zx_cf = xz_cf
#    zy_cf = yz_cf
#
#    # Generate A matrix
#    A = np.zeros((n, n))
#
#    A[0,3] = 1.
#    A[1,4] = 1.
#    A[2,5] = 1.
#
#    A[3,0] = xx_cf
#    A[3,1] = xy_cf
#    A[3,2] = xz_cf
#
#    A[4,0] = yx_cf
#    A[4,1] = yy_cf
#    A[4,2] = yz_cf
#
#    A[5,0] = zx_cf
#    A[5,1] = zy_cf
#    A[5,2] = zz_cf
#
#    # Compute STM components dphi = A*phi
#    phi_mat = np.reshape(X[n:], (n, n))
#    dphi_mat = np.dot(A, phi_mat)
#    dphi_v = np.reshape(dphi_mat, (n**2, 1))
#
#    # Derivative vector
#    dX = np.zeros((n+n**2, 1))
#
#    dX[0] = dx
#    dX[1] = dy
#    dX[2] = dz
#
#    dX[3] = -GM*x/r**3
#    dX[4] = -GM*y/r**3
#    dX[5] = -GM*z/r**3
#
#    dX[n:] = dphi_v
#
#    dX = dX.flatten()
#
#    return dX



#def int_twobody_j2(X, t, inputs):
#    '''
#    This function works with odeint to propagate object assuming
#    simple two-body dynamics and perturbations due to J2.
#
#    Parameters
#    ------
#    X : 6 element list
#      cartesian state vector (Inertial Frame)
#    t : m element list
#      vector of times when output is desired
#    inputs : dictionary
#      input parameters
#
#    Returns
#    ------
#    dX : 6 element list
#      state derivative vector
#    '''
#
#    # Input data
#    GM = inputs['GM']
#    J2 = inputs['J2']
#    Re = inputs['Re']
#
#    # State Vector
#    n = 6
#    x = float(X[0])
#    y = float(X[1])
#    z = float(X[2])
#    dx = float(X[3])
#    dy = float(X[4])
#    dz = float(X[5])
#
#    # Compute radius
#    r_vect = np.array([[x], [y], [z]])
#    r = np.linalg.norm(r_vect)
#
#    # Derivative vector
#    dX = np.zeros((n, 1))
#
#    dX[0] = dx
#    dX[1] = dy
#    dX[2] = dz
#
#    dX[3] = -GM*x/r**3. - 1.5*J2*Re**2.*GM*((x/r**5.) - (5.*x*z**2./r**7.))
#    dX[4] = -GM*y/r**3. - 1.5*J2*Re**2.*GM*((y/r**5.) - (5.*y*z**2./r**7.))
#    dX[5] = -GM*z/r**3. - 1.5*J2*Re**2.*GM*((3.*z/r**5.) - (5.*z**3./r**7.))
#
#    dX = dX.flatten()
#
#    return dX
#
#
#def int_twobody_j2_ukf(X, t, inputs):
#    '''
#    This function works with odeint to propagate object assuming
#    simple two-body dynamics and perturbations due to J2. Outputs sigma
#    points for use in UKF covariance propagation.
#
#    Parameters
#    ------
#    X : (n*(2n+1)) element list
#      initial condition vector of cartesian state and sigma points
#    t : m element list
#      vector of times when output is desired
#    inputs : dictionary
#     input parameters
#
#    Returns
#    ------
#    dX : (n*(2n+1)) element list
#      derivative vector
#
#    '''
#
#    # Break out inputs
#    GM = inputs['GM']
#    J2 = inputs['J2']
#    Re = inputs['Re']
#
#    # Initialize
#    n = int((-1 + np.sqrt(1. + 8.*len(X)))/4.)    
#    dX = np.zeros(((2*n+1)*n, 1))
#
#    for ind in range(0, 2*n+1):
#
#        # Pull out relevant values from X
#        x = float(X[ind*n])
#        y = float(X[ind*n + 1])
#        z = float(X[ind*n + 2])
#        dx = float(X[ind*n + 3])
#        dy = float(X[ind*n + 4])
#        dz = float(X[ind*n + 5])
#
#        # Compute radius
#        r = np.linalg.norm([x, y, z])
#
#        # Solve for components of dX
#        dX[ind*n] = dx
#        dX[ind*n + 1] = dy
#        dX[ind*n + 2] = dz
#
#        dX[ind*n + 3] = -GM*x/r**3. - 1.5*J2*Re**2.*GM*((x/r**5.) -
#                                                        (5.*x*z**2./r**7.))
#        dX[ind*n + 4] = -GM*y/r**3. - 1.5*J2*Re**2.*GM*((y/r**5.) -
#                                                        (5.*y*z**2./r**7.))
#        dX[ind*n + 5] = -GM*z/r**3. - 1.5*J2*Re**2.*GM*((3.*z/r**5.) -
#                                                        (5.*z**3./r**7.))
#
#    dX = dX.flatten()
#
#    return dX
#
#
#def int_twobody_j2_drag(X, t, inputs):
#    '''
#    This function works with odeint to propagate object assuming
#    simple two-body dynamics and perturbations due to J2 and drag.
#
#    Parameters
#    ------
#    X : 6 element list
#      cartesian state vector (Inertial Frame)
#    t : m element list
#      vector of times when output is desired
#    inputs : dictionary
#      input parameters
#
#    Returns
#    ------
#    dX : 6 element list
#      state derivative vector
#    '''
#
#    # Input data
#    GM = inputs['GM']
#    J2 = inputs['J2']
#    Cd = inputs['Cd']
#    Re = inputs['Re']
#    dtheta = inputs['dtheta']
#    rho0 = inputs['rho0']
#    ro = inputs['ro']
#    H = inputs['H']
#    A_m = inputs['A_m']
#    
#    
#
#    # State Vector
#    n = 6
#    x = float(X[0])
#    y = float(X[1])
#    z = float(X[2])
#    dx = float(X[3])
#    dy = float(X[4])
#    dz = float(X[5])
#
#    # Compute radius
#    r_vect = np.array([[x], [y], [z]])
#    r = np.linalg.norm(r_vect)
#    
#    # Compute drag component
#    # Find vector va of spacecraft relative to atmosphere
#    v_vect = np.array([[dx], [dy], [dz]])
#    w_vect = np.array([[0.], [0.], [dtheta]])
#    va_vect = v_vect - np.cross(w_vect, r_vect, axis=0)
#    va = np.linalg.norm(va_vect)
#    va_x = float(va_vect[0])
#    va_y = float(va_vect[1])
#    va_z = float(va_vect[2])
#    
#    drag = -0.5*A_m*Cd*rho0*exp(-(r-ro)/H)
#    
##    print r_vect
##    print v_vect
##    print va_vect
##    print va
##    print drag
##    mistake
#    
#    x_drag = drag*va*va_x
#    y_drag = drag*va*va_y
#    z_drag = drag*va*va_z
#    
##    print x_drag
##    print y_drag
##    print z_drag
##    mistake
#
#    # Derivative vector
#    dX = np.zeros((n, 1))
#
#    dX[0] = dx
#    dX[1] = dy
#    dX[2] = dz
#
#    dX[3] = -GM*x/r**3. + x_drag - 1.5*J2*Re**2.*GM*((x/r**5.) - (5.*x*z**2./r**7.))
#    dX[4] = -GM*y/r**3. + y_drag - 1.5*J2*Re**2.*GM*((y/r**5.) - (5.*y*z**2./r**7.))
#    dX[5] = -GM*z/r**3. + z_drag - 1.5*J2*Re**2.*GM*((3.*z/r**5.) - (5.*z**3./r**7.))
#    
#    dX = dX.flatten()
#    
##    print dX
##    mistake
#
#    return dX
#
#
#def int_twobody_j2_drag_stm(X, t, inputs):
#    '''
#    This function works with odeint to propagate object assuming
#    simple two-body dynamics and perturbations due to J2 and drag.
#    Includes components of the state transition matrix.
#
#    Parameters
#    ------
#    X : 42 element list
#      cartesian state vector (Inertial Frame)
#    t : m element list
#      vector of times when output is desired
#    inputs : dictionary
#      input parameters
#
#    Returns
#    ------
#    dX : 42 element list
#      state derivative vector
#    '''
#
#    # Input data
#    GM = inputs['GM']
#    J2 = inputs['J2']
#    Cd = inputs['Cd']
#    Re = inputs['Re']
#    dtheta = inputs['dtheta']
#    rho0 = inputs['rho0']
#    ro = inputs['ro']
#    H = inputs['H']
#    A_m = inputs['A_m']
#
#    # State Vector
#    n = 6
#    x = float(X[0])
#    y = float(X[1])
#    z = float(X[2])
#    dx = float(X[3])
#    dy = float(X[4])
#    dz = float(X[5])
#
#    # Compute radius
#    r_vect = np.array([[x], [y], [z]])
#    r = np.linalg.norm(r_vect)
#    
#    # Compute drag component
#    # Find vector va of spacecraft relative to atmosphere
#    v_vect = np.array([[dx], [dy], [dz]])
#    w_vect = np.array([[0.], [0.], [dtheta]])
#    va_vect = v_vect - np.cross(w_vect, r_vect, axis=0)
#    va = np.linalg.norm(va_vect)
#    va_x = float(va_vect[0])
#    va_y = float(va_vect[1])
#    va_z = float(va_vect[2])
#    
#    drag = -0.5*A_m*Cd*rho0*exp(-(r-ro)/H)
#    
#    x_drag = drag*va*va_x
#    y_drag = drag*va*va_y
#    z_drag = drag*va*va_z
#    
#    # Find elements of A matrix
#    xx_cf = -GM/r**3. + 3.*GM*x**2./r**5.
#    xx_drag = drag*((-x*va*va_x/(H*r)) - dtheta*va_y*va_x/va)
#    xx_J2 = -1.5*J2*GM*Re**2./r**5. - 7.5*J2*GM*Re**2./r**7.*(-x**2. - z**2. + 7.*x**2.*z**2./r**2.)
#
#    xy_cf = 3.*GM*x*y/r**5.
#    xy_drag = drag*((-y*va*va_x/(H*r)) + dtheta*va_x**2./va + va*dtheta)
#    xy_J2 = -7.5*x*y/r**7. * J2*Re**2.*GM*(-1. + 7.*z**2./r**2.)
#
#    xz_cf = 3.*GM*x*z/r**5.
#    xz_drag = drag*(-z*va*va_x/(H*r))
#    xz_J2 = -7.5*x*z/r**7. * J2*Re**2.*GM*(-3. + 7.*z**2./r**2.)
#
#    yy_cf = -GM/r**3. + 3.*GM*y**2./r**5.
#    yy_drag = drag*((-y*va*va_y/(H*r)) + dtheta*va_x*va_y/va)
#    yy_J2 = -1.5*J2*GM*Re**2./r**5. - 7.5*J2*GM*Re**2./r**7.*(-y**2. - z**2. + 7.*y**2.*z**2./r**2.)
#
#    yx_cf = xy_cf
#    yx_drag = drag*((-x*va*va_y/(H*r)) - dtheta*va_y**2./va - va*dtheta)
#    yx_J2 = xy_J2
#
#    yz_cf = 3.*GM*y*z/r**5.
#    yz_drag = drag*(-z*va*va_y/(H*r))
#    yz_J2 = -7.5*y*z/r**7. * J2*Re**2.*GM*(-3. + 7.*z**2./r**2.)
#
#    zz_cf = -GM/r**3. + 3.*GM*z**2./r**5.
#    zz_drag = drag*(-z*va*va_z/(H*r))
#    zz_J2 = -4.5*J2*Re**2.*GM/r**5. - 7.5*J2*Re**2.*GM/r**7.*(-6.*z**2. + 7.*z**4./r**2.)
#
#    zx_cf = xz_cf
#    zx_drag = drag*((-x*va*va_z/(H*r)) - dtheta*va_y*va_z/va)
#    zx_J2 = xz_J2
#
#    zy_cf = yz_cf
#    zy_drag = drag*((-y*va*va_z/(H*r)) + dtheta*va_x*va_z/va)
#    zy_J2 = yz_J2
#    
#    # Generate A matrix using partials from above
#    A = np.zeros((n,n))
#
#    A[0,3] = 1. 
#    A[1,4] = 1. 
#    A[2,5] = 1.
#
#    A[3,0] = xx_cf + xx_drag + xx_J2
#    A[3,1] = xy_cf + xy_drag + xy_J2
#    A[3,2] = xz_cf + xz_drag + xz_J2
#    A[3,3] = drag*(va_x**2./va + va)
#    A[3,4] = drag*(va_y*va_x/va)
#    A[3,5] = drag*(va_z*va_x/va)      # Note, va_z = dz
#
#    A[4,0] = yx_cf + yx_drag + yx_J2
#    A[4,1] = yy_cf + yy_drag + yy_J2
#    A[4,2] = yz_cf + yz_drag + yz_J2
#    A[4,3] = drag*(va_y*va_x/va)
#    A[4,4] = drag*(va_y**2./va + va)
#    A[4,5] = drag*(va_y*va_z/va)       # Note, va_z = dz
#
#    A[5,0] = zx_cf + zx_drag + zx_J2
#    A[5,1] = zy_cf + zy_drag + zy_J2
#    A[5,2] = zz_cf + zz_drag + zz_J2
#    A[5,3] = drag*(va_x*va_z/va)
#    A[5,4] = drag*(va_y*va_z/va)
#    A[5,5] = drag*(va_z**2./va + va)
#    
#    # Compute STM components dphi = A*phi
#    phi_mat = np.reshape(X[n:], (n, n))
#    dphi_mat = np.dot(A, phi_mat)
#    dphi_v = np.reshape(dphi_mat, (n**2, 1))
#
#    # Derivative vector
#    dX = np.zeros((n+n**2, 1))
#
#    dX[0] = dx
#    dX[1] = dy
#    dX[2] = dz
#
#    dX[3] = -GM*x/r**3 + x_drag - 1.5*J2*Re**2.*GM*((x/r**5.) - (5.*x*z**2./r**7.))
#    dX[4] = -GM*y/r**3 + y_drag - 1.5*J2*Re**2.*GM*((y/r**5.) - (5.*y*z**2./r**7.))
#    dX[5] = -GM*z/r**3 + z_drag - 1.5*J2*Re**2.*GM*((3.*z/r**5.) - (5.*z**3./r**7.))
#    
#    dX[n:] = dphi_v
#
#    dX = dX.flatten()
#    
#
#    return dX
#
#
#def int_twobody_srp(X, t, inputs):
#    '''
#    This function works with odeint to propagate object assuming
#    simple two-body dynamics and SRP perturbation.  C_r and A_m are
#    given parameters, not included in the state vector.
#
#    Parameters
#    ------
#    X : n element list
#      initial condition vector of cartesian state (Inertial Frame)
#    t : m element list
#      vector of times when output is desired
#    inputs : dictionary
#      input parameters
#
#    Returns
#    ------
#    dX : n element list
#      derivative vector
#    '''
#
#    # Input data
#    GM = inputs['GM']
#    Cr = inputs['Cr']
#    A_m = inputs['A_m']
#
#    # State Vector
#    n = 6
#    x = X[0]
#    y = X[1]
#    z = X[2]
#    dx = X[3]
#    dy = X[4]
#    dz = X[5]
#
#    # Compute radius
#    r_vect = np.reshape([x, y, z], (3, 1))
#    r = np.linalg.norm(r_vect)
#
#    # Compute perturbation due to srp
#    beta = Cr * A_m
#    a_srp = compute_srp(r_vect, beta, inputs, t)
#
#    # Derivative vector
#    dX = np.zeros((n, 1))
#
#    dX[0] = dx
#    dX[1] = dy
#    dX[2] = dz
#
#    dX[3] = -GM*x/r**3 + a_srp[0]
#    dX[4] = -GM*y/r**3 + a_srp[1]
#    dX[5] = -GM*z/r**3 + a_srp[2]
#
#    dX = dX.flatten()
#
#    return dX
#
#
#def compute_srp(r_vect, beta, inputs, t):
#    '''
#    This function computes the acceleration due to solar radiation pressure.
#
#    Parameters
#    ------
#    r_vect : 3x1 numpy array
#      position vector in ECI
#    beta : float
#      SRP parameter (Cr*A/m) [km^2/kg]
#    inputs : dictionary
#      input parameters
#    t : float
#      current time of day [sec]
#    Returns
#    ------
#    a_srp : 3x1 numpy array
#      acceleration vector from SRP perturbing force
#    '''
#
#    # Break out inputs
#    d0 = inputs['d0']
#    phi_sun = inputs['phi_sun']
#    c = inputs['c']
#    start_tdays = inputs['start_tdays']
#
#    # Compute current solar position in ECI
#    # Time in days since J2000
#    tdays = start_tdays + t/86400.
#
#    # Convert to centuries and compute mean longitude of the sun
#    tcent = tdays/(365.25*100.)
#    lam = 280.460 + 36000.771*tcent
#    lam = (lam % 360.) * pi/180.  # rad
#
#    # Compute mean anomaly of sun
#    M = 357.5277233 + 35999.05034*tcent
#    M = (M % 360.) * pi/180.  # rad
#
#    # Compute ecliptic long/lat
#    lam_ec = lam + (1.914666471*sin(M) + 0.019994643*sin(2.*M))*pi/180.  # rad
#
#    # Compute distance to sun in AU and obliquity of ecliptic plane
#    r = 1.000140612 - 0.016708617*cos(M) - 0.000139589*cos(2.*M)  # AU
#    ep = (23.439291 - 0.0130042*tcent) * pi/180.
#
#    # Compute sun position vector in Mean Equator of Date (MOD) Frame
#    r_sun = r*d0*np.array([[cos(lam_ec)], [cos(ep)*sin(lam_ec)],
#                           [sin(ep)*sin(lam_ec)]])
#    rg = np.norm(r_sun)
#
#    # Compute ra/dec of sun
#    dec = asin(sin(ep)*sin(lam_ec))
#    ra = atan2((cos(ep)*sin(lam_ec)/cos(dec)), (cos(lam_ec)/cos(dec)))
#
#    # Convert to ECI
#    sun_eci = conv.radec2eci(ra, dec, rg)
#
#    # Compute current position vector from sun
#    r_fromsun = r_vect - sun_eci
#    d = np.norm(r_fromsun)
#    u_sun = -r_fromsum/d
#
#    # SRP acceleration
#    a_srp = -(phi_sun/(c*(d/d0)**2.))*beta*(1./1000.)*u_sun  # km/s^2
#
#    return a_srp
