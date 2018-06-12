import numpy as np
from math import *
import copy
from datetime import datetime, timedelta
import os

#
#from TurboProp import DataDir
#import TurboProp.PyUtils.Time as TPT
#import Coords.coordUtils as CU

###############################################################################
# This file contains functions to perform various conversions
# Functions:
#  GMST_eci2ecef
#  GMST_ecef2eci
#  eci2ric
#  ric2eci
#  GMST_eci2enu
#  eci2enu
#  ecef2enu
#  enu2ecef
#  ecef2latlonht
#  latlonht2ecef
#  ecef2azelrange
#  ecef2alelrange_rad
#  radec2eci
#  rhohat2reci
#  enuangles2latlonht
#  latlonht2enuangles
#  enuangles2azel
#  azel2enuangles
#  enuangles2radec
#  radec2enuangles
#  LAMlat2PHI
#  mean2ecc
#  mean2hyp
#  element conversion
#  tle2kep
#  cart2tle
###############################################################################



###############################################################################
# Time Systems
###############################################################################





def UTC_G_2_JED_JD(UTC):

    JD = TPT.TimeFrame([UTC], 'UTC_G', 'JED_JD')[0]

    return JD


def UTC_G_2_jde(UTC):

    dum = modf(UTC[5])
    sec = int(dum[1])
    msec = int(dum[0]*1e6)
    utc_datetime = datetime.datetime(UTC[0], UTC[1], UTC[2], UTC[3], UTC[4],
                                     sec, msec)
    jde = julian.GetJulianDay(utc_datetime)

    return jde


def JED_JD_2_UTC_G(JD, round_flag=1):

    UTC = TPT.TimeFrame([JD], 'JED_JD', 'UTC_G')[0]
    
    
    # Round to nearest second
    if round_flag:
        
        # Number of days in each month
        daynum = [ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ]
        year = UTC[0]
        if ( year%4 == 0 and year%100 != 0 ) or ( year%400 == 0 ):
            daynum[1] = 29    

        UTC[5] = round(UTC[5])
        
        if UTC[5] == 60.:
            UTC[5] -= 60.
            UTC[4] += 1
            
        if UTC[4] == 60:
            UTC[4] -= 60
            UTC[3] += 1
        
        if UTC[3] == 24:
            UTC[3] -= 24
            UTC[2] += 1
        
        month = UTC[1]
        if UTC[2] == daynum[month-1]+1:
            UTC[2] = 1
            UTC[1] += 1
        
        if UTC[1] == 13:
            UTC[1] = 1
            UTC[0] += 1
        

    return UTC


def UTC_JD_2_UTC_G(JD, round_flag=1):

    UTC = TPT.TimeFrame([JD], 'UTC_JD', 'UTC_G')[0]
    
    
    # Round to nearest second
    if round_flag:
        
        # Number of days in each month
        daynum = [ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ]
        year = UTC[0]
        if ( year%4 == 0 and year%100 != 0 ) or ( year%400 == 0 ):
            daynum[1] = 29    

        UTC[5] = round(UTC[5])
        
        if UTC[5] == 60.:
            UTC[5] -= 60.
            UTC[4] += 1
            
        if UTC[4] == 60:
            UTC[4] -= 60
            UTC[3] += 1
        
        if UTC[3] == 24:
            UTC[3] -= 24
            UTC[2] += 1
        
        month = UTC[1]
        if UTC[2] == daynum[month-1]+1:
            UTC[2] = 1
            UTC[1] += 1
        
        if UTC[1] == 13:
            UTC[1] = 1
            UTC[0] += 1
        

    return UTC



def JED_JD_2_jde(JD):

    UTC = jdtt2utc(JD)
    jde = utc2jde(UTC)

    return jde

###############################################################################
# Coordinate Frame Rotations
###############################################################################


def setup_coord_frame(inputs):
    '''
    This function sets up coordinate frame transformations using
    Ben Bradley/Brandon Jones EOP code

    Parameters
    ------
    inputs : dictionary
        dictonary of input parameters

    Returns
    ------
    inputs : dictonary
        dictionary of input parameters
    '''

    eopFile = os.path.join(DataDir, 'EOP_1962_DATA.txt')
    xysFile = os.path.join(DataDir, 'IAU2006_XYs.txt')
    lsFile = os.path.join(DataDir, 'leapsec.dat')
    myIAU = CU.IAU2006CIO(EOPFile=eopFile, XYsFile=xysFile, LeapSecFile=lsFile)
    inputs['myIAU'] = myIAU

    return inputs


def GMST_eci2ecef(r_eci, theta):
    '''
    This function converts the coordinates of a position vector from
    the ECI to ECEF frame using simple z-axis rotation only.

    Parameters
    ------
    r_eci : 3x1 numpy array
      position vector in ECI
    theta : float
      earth rotation angle [rad]

    Returns
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF
    '''

    R3 = np.array([[cos(theta),  sin(theta), 0.],
                   [-sin(theta), cos(theta), 0.],
                   [0.,          0.,         1.]])

    r_ecef = np.dot(R3, r_eci)

    return r_ecef


def GMST_ecef2eci(r_ecef, theta):
    '''
    This function converts the coordinates of a position vector from
    the ECEF to ECI frame using simple z-axis rotation only.

    Parameters
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF
    theta : float
      earth rotation angle [rad]

    Returns
    ------
    r_eci : 3x1 numpy array
      position vector in ECI
    '''

    R3 = np.array([[cos(theta), -sin(theta), 0.],
                   [sin(theta),  cos(theta), 0.],
                   [0.,          0.,         1.]])

    r_eci = np.dot(R3, r_ecef)

    return r_eci


def eci2ecef(r_eci, inputs, JED_JD):
    '''
    This function rotates a vector coordinates from ECI to ECEF using
    full Ben Bradley/Brandon Jones EOP code.

    Parameters
    ------
    r_eci : 3x1 numpy array
        object position vector in ECI
    inputs : dict
        dictionary including frame rotation parameters
    JED_JD : float
        current time

    Returns
    ------
    r_ecef : 3x1 numpy array
        object position vector in ECEF
    '''

    myIAU = inputs['myIAU']
    dum = modf(JED_JD)
    jdTime = [int(dum[1]), dum[0]]

    # Rotate to ENU
    res = myIAU.ECI2ECEF(jdTime, position=r_eci)
    r_ecef = np.reshape(res[0], (3, 1))

    return r_ecef


def ecef2eci(r_ecef, inputs, JED_JD):
    '''
    This function rotates a vector coordinates from ECEF to ECI using
    full Ben Bradley/Brandon Jones EOP code.

    Parameters
    ------
    r_ecef : 3x1 numpy array
        object position vector in ECEF
    inputs : dict
        dictionary including frame rotation parameters
    JED_JD : float
        current time

    Returns
    ------
    r_eci : 3x1 numpy array
        object position vector in ECI
    '''

    myIAU = inputs['myIAU']
    dum = modf(JED_JD)
    jdTime = [int(dum[1]), dum[0]]

    # Rotate to ENU
    res = myIAU.ECEF2ECI(jdTime, position=r_ecef)
    r_eci = np.reshape(res[0], (3, 1))

    return r_eci


def eci2ric(rc_vect, vc_vect, Qin):
    '''
    This function computes the rotation from ECI to RIC and rotates input
    Qin (vector or matrix) to RIC.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    Qin : 3x1 or 3x3 numpy array
      vector or matrix in ECI

    Returns
    ------
    Qout : 3x1 or 3x3 numpy array
      vector or matrix in RIC
    '''

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)

    ON = np.concatenate((OR.T, OT.T, OH.T))

    # Rotate Qin as appropriate for vector or matrix
    if Qin.shape[1] == 1:
        Qout = np.dot(ON, Qin)
    else:
        Qout = np.dot(np.dot(ON, Qin), ON.T)

    return Qout


def ric2eci(rc_vect, vc_vect, Qin):
    '''
    This function computes the rotation from RIC to ECI and rotates input
    Qin (vector or matrix) to ECI.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    Qin : 3x1 or 3x3 numpy array
      vector or matrix in RIC

    Returns
    ------
    Qout : 3x1 or 3x3 numpy array
      vector or matrix in ECI
    '''

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)

    ON = np.concatenate((OR.T, OT.T, OH.T))
    NO = ON.T

    # Rotate Qin as appropriate for vector or matrix
    if Qin.shape[1] == 1:
        Qout = np.dot(NO, Qin)
    else:
        Qout = np.dot(np.dot(NO, Qin), NO.T)

    return Qout


def GMST_eci2enu(r_eci, stat_ecef, theta):
    '''
    This function converts the coordinates of a position vector from
    the ECI to ENU frame with a simple z-axis rotation used for the
    intermediate ECEF frame.

    Parameters
    ------
    r_eci : 3x1 numpy array
      vector of satellite position in ECI [km]
    stat_ecef : 3x1 numpy array
      vector of station location in ECEF [km]
    theta : float
      current earth rotation angle [rad]

    Returns
    ------
    r_enu : 3x1 numpy array
      vector of satellite position in ENU [km]
    R : 3x3 numpy array
      rotation matrix from ECI to ENU
    '''

    # Compute the station lat/lon
    lat, lon, ht = ecef2latlonht(stat_ecef)
    lat = lat*pi/180  # rad
    lon = lon*pi/180  # rad

    # Compute rotation matrix
    lat1 = pi/2 - lat
    lon1 = pi/2 + lon

    R1 = np.array([[1.,          0.,        0.],
                   [0.,   cos(lat1), sin(lat1)],
                   [0.,  -sin(lat1), cos(lat1)]])

    R2 = np.array([[cos(lon1),  sin(lon1), 0.],
                   [-sin(lon1), cos(lon1), 0.],
                   [0.,           0.,      1.]])

    R3 = np.array([[cos(theta),  sin(theta), 0.],
                   [-sin(theta), cos(theta), 0.],
                   [0.,          0.,         1.]])

    R = np.dot(np.dot(R1, R2), R3)
    r_enu = np.dot(R, r_eci)

    return r_enu, R


def eci2enu(r_eci, stat_ecef, inputs, JED_JD):
    '''
    This function rotates a vector coordinates from ECI to ENU through
    an intermediate transform to ECEF. The ECI to ECEF conversion uses
    full Ben Bradley/Brandon Jones EOP code.

    Parameters
    ------
    r_eci : 3x1 numpy array
        object position vector in ECI
    stat_ecef : 3x1 numpy array
        ground station position vector in ECEF
    inputs : dict
        dictionary including frame rotation parameters
    JED_JD : float
        current time

    Returns
    ------
    r_enu : 3x1 numpy array
        object position vector in ENU
    '''

    myIAU = inputs['myIAU']
    dum = modf(JED_JD)
    jdTime = [int(dum[1]), dum[0]]

    # Rotate to ENU
    res = myIAU.ECI2ECEF(jdTime, position=r_eci)
    r_ecef = np.reshape(res[0], (3, 1))
    r_enu = ecef2enu(r_ecef, stat_ecef)

    return r_enu


def ecef2enu(r_ecef, r_site):
    '''
    This function converts the coordinates of a position vector from
    the ECEF to ENU frame.

    Parameters
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF [km]
    r_site : 3x1 numpy array
      station position vector in ECEF [km]

    Returns
    ------
    r_enu : 3x1 numpy array
      position vector in ENU [km]
    '''

    # Compute lat,lon,ht of ground station
    lat, lon, ht = ecef2latlonht(r_site)
    lat = lat*pi/180  # rad
    lon = lon*pi/180  # rad

    # Compute rotation matrix
    lat1 = pi/2 - lat
    lon1 = pi/2 + lon

    R1 = np.array([[1.,   0.,               0.],
                   [0.,   cos(lat1), sin(lat1)],
                   [0.,  -sin(lat1), cos(lat1)]])

    R3 = np.array([[cos(lon1),  sin(lon1), 0.],
                   [-sin(lon1), cos(lon1), 0.],
                   [0.,          0.,       1.]])

    R = np.dot(R1, R3)

    r_enu = np.dot(R, r_ecef)

    return r_enu


def enu2ecef(r_enu, r_site):
    '''
    This function converts the coordinates of a position vector from
    the ENU to ECEF frame.

    Parameters
    ------
    r_enu : 3x1 numpy array
      position vector in ENU [km]
    r_site : 3x1 numpy array
      station position vector in ECEF [km]

    Returns
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF
    '''

    # Compute lat,lon,ht of ground station
    lat, lon, ht = ecef2latlonht(r_site)
    lat = lat*pi/180  # rad
    lon = lon*pi/180  # rad

    # Compute rotation matrix
    lat1 = pi/2 - lat
    lon1 = pi/2 + lon

    R1 = np.array([[1.,   0.,               0.],
                   [0.,   cos(lat1), sin(lat1)],
                   [0.,  -sin(lat1), cos(lat1)]])

    R3 = np.array([[cos(lon1),  sin(lon1), 0.],
                   [-sin(lon1), cos(lon1), 0.],
                   [0.,          0.,       1.]])

    R = np.dot(R1, R3)

    R2 = R.T

    r_ecef = np.dot(R2, r_enu)

    return r_ecef


###############################################################################
# Angles and Measurements
###############################################################################

def eci2radec(r_eci):
    '''
    This function computes the astrometric (J2000 ECI) right ascension and 
    declination from an ECI position vector.

    Parameters
    ------
    r_eci : 3x1 numpy array
      position vector in ECI [km]

    Returns
    ------
    ra : float
      right ascension [rad] [-pi,pi]
    dec : float
      declination [rad] [-pi/2,pi/2]
    '''
    
    r = np.linalg.norm(r_eci)
    ra = atan2(r_eci[1], r_eci[0])
    dec = asin(r_eci[2]/r)    
    
    return ra, dec


def ecef2latlonht(r_ecef):
    '''
    This function converts the coordinates of a position vector from
    the ECEF frame to geodetic latitude, longitude, and height.

    Parameters
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF [km]

    Returns
    ------
    lat : float
      latitude [deg] [-90,90]
    lon : float
      longitude [deg] [-180,180]
    ht : float
      height [km]
    '''

    # WGS84 Data (Pratap and Misra P. 103)
    a = 6378.1370   # km
    rec_f = 298.257223563

    # Get components from position vector
    x = float(r_ecef[0])
    y = float(r_ecef[1])
    z = float(r_ecef[2])

    # Compute longitude
    f = 1/rec_f
    e = np.sqrt(2*f - f**2)
    lon = atan2(y, x) * 180/pi  # deg

    # Iterate to find height and latitude
    p = np.sqrt(x**2 + y**2)  # km
    lat = 0.*pi/180.
    lat_diff = 1.
    tol = 1e-12

    while abs(lat_diff) > tol:
        lat0 = copy.copy(lat)  # rad
        N = a/np.sqrt(1 - e**2*(sin(lat0)**2))  # km
        ht = p/cos(lat0) - N
        lat = atan((z/p)/(1 - e**2*(N/(N + ht))))
        lat_diff = lat - lat0

    lat = lat*180/pi  # deg

    return lat, lon, ht


def latlonht2ecef(lat, lon, ht):
    '''
    This function converts geodetic latitude, longitude and height
    to a position vector in ECEF.

    Parameters
    ------
    lat : float
      geodetic latitude [deg]
    lon : float
      geodetic longitude [deg]
    ht : float
      geodetic height [km]

    Returns
    ------
    r_ecef = 3x1 numpy array
      position vector in ECEF [km]
    '''

    # WGS84 Data (Pratap and Misra P. 103)
    Re = 6378.1370   # km
    rec_f = 298.257223563

    # Convert to radians
    lat = lat*pi/180    # rad
    lon = lon*pi/180    # rad

    # Compute flattening and eccentricity
    f = 1/rec_f
    e = np.sqrt(2*f - f**2)

    # Compute ecliptic plane and out of plane components
    C = Re/np.sqrt(1 - e**2*sin(lat)**2)
    S = Re*(1 - e**2)/np.sqrt(1 - e**2*sin(lat)**2)

    rd = (C + ht)*cos(lat)
    rk = (S + ht)*sin(lat)

    # Compute ECEF position vector
    r_ecef = np.array([[rd*cos(lon)], [rd*sin(lon)], [rk]])

    return r_ecef


def geocentric2geodetic(lat_gc, lon_gc, r):
	
    '''
    This function conmputes the geodetic latitude of an object given its 
    geocentric latitude and range from center of the earth.
    
    Reference
    ------
    https://au.mathworks.com/help/aeroblks/geocentrictogeodeticlatitude.html

    Parameters
    ------
    lat_gc : float
        geocentric latitude [deg]
    lon_gc : floag
        geocentric longitude [deg]
    r : float
        geocentric range [km]

    Returns
    ------
    lat_gd : float
        geodetic latitude [deg]
    lon_gd : float
        geodetic longitude [deg]
    ht : float
        geodetic height [km]
    '''

    # WGS84 Data (Pratap and Misra P. 103)
    Re = 6378.1370   # km
    rec_f = 298.257223563

    # Convert to radians
    lat_gc = lat_gc*pi/180    # rad

    # Compute flattening and eccentricity
    f = 1/rec_f

    # Geocentric latitude of point at surface
    xa = (1. - f)*Re/np.sqrt(tan(lat_gc)**2. + (1-f)**2.)
    ra = xa/cos(lat_gc)
    mu_a = atan2(tan(lat_gc), (1.-f)**2.)

    # Intermediate distances and angles
    L = r - ra
    dlam = mu_a - lat_gc
    ht = L*cos(dlam)
    rho_a = Re*(1. - f)**2./((1. - (2.*f - f**2.)*sin(mu_a)**2.)**1.5)

    # Geodetic latitude
    dmu = atan2(L*sin(dlam), (rho_a + ht))
    lat_gd = (mu_a - dmu)*180./pi

    return lat_gd, lon_gc, ht


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


def radec2eci(ra, dec, rg):
    '''
    This function computes position vector in ECI given
    geocentric right ascension, declination, and range.

    Parameters
    ------
    ra : float
      right ascension [rad]
    dec : float
      declination [rad]
    rg : float
      range

    Returns
    ------
    r_eci : 3x1 numpy array
      position vector in ECI
    '''

    x = rg*cos(dec)*cos(ra)
    y = rg*cos(dec)*sin(ra)
    z = rg*sin(dec)

    r_eci = np.array([[x], [y], [z]])

    return r_eci


def rhohat2reci(rho_hat_eci, r, stat_eci):
    '''
    This function computes position vector in ECI given
    a unit vector in the ECI frame and a radius and
    ground station location in ECI.

    Parameters
    ------
    rho_hat_eci : 3x1 numpy array
      unit vector in ECI frame from ground station to target
    r : float
      desired orbit radius (SMA) of target [km]
    stat_eci : 3x1 numpy array
      postion vector of ground station location in ECI [km]

    Returns
    ------
    r_eci : 3x1 numpy array
        position vector of target in ECI [km]

    '''

    # Compute range to target
    q = np.linalg.norm(stat_eci)
    costhe = np.dot(-stat_eci.T, rho_hat_eci)/q
    rg = 0.5*(2*q*costhe + np.sqrt((2*q*costhe)**2 - 4*(q**2 - r**2)))

    # Compute output vector
    r_eci = rg*rho_hat_eci + stat_eci

    return r_eci


def enuangles2latlonht(LAM, PHI, r, r_site):
    '''
    This function computes geodetic lat, lon, ht given
    a set of angles in the ENU frame and a radius and
    ground station location.

    Parameters
    ------
    LAM : float
      ENU longitude measured from N-U plane [deg]
    PHI : float
      ENU latitude measured from E-U plane [deg]
    r : float
      desired orbit radius (SMA) of target [km]
    r_site : 3x1 numpy array
      postion vector of ground station location in ECEF [km]

    Returns
    ------
    lat : float
      geodetic latitude [deg]
    lon : float
      geodetic longitude [deg]
    ht : float
      geodetic height [km]

    '''

    # Convert angles to rad
    LAM = LAM*pi/180.
    PHI = PHI*pi/180.

    # Compute unit vector in ENU
    rho_hat_enu = np.array([[cos(PHI)*sin(LAM)], [sin(PHI)],
                            [cos(PHI)*cos(LAM)]])

    # Rotate to ECEF
    rho_hat_ecef = enu2ecef(rho_hat_enu, r_site)

    # Iterate to find r_ecef
    rg_max = r + 20000.
    rg_min = max([0., r - 20000.])
    rg_mid = (rg_max + rg_min)/2.
    diff = 10.
    tol = 0.1

    while abs(diff) > tol:

        # Save for comparison
        rg_mid0 = copy.copy(rg_mid)

        # Compute vector to point in ECEF
        rho_ecef = rg_mid * rho_hat_ecef
        r_ecef = rho_ecef + r_site

        # Check magnitude of r_ecef
        if np.linalg.norm(r_ecef) < r:
            rg_min = copy.copy(rg_mid)
        else:
            rg_max = copy.copy(rg_mid)

        # Compute diff for loop exit
        rg_mid = (rg_max + rg_min)/2.
        diff = rg_mid - rg_mid0

    # Compute lat,lon,ht
    lat, lon, ht = ecef2latlonht(r_ecef)

    return lat, lon, ht


def latlonht2enuangles(lat, lon, ht, r_site):
    '''
    This function computes a set of angles
    in the ENU frame and a range given geodetic
    lat,lon,ht and ground station location.

    Parameters
    ------
    lat : float
      geodetic latitude [deg]
    lon : float
      geodetic longitude [deg]
    ht : float
      geodetic height [km]
    r_site : 3x1 numpy array
      postion vector of ground station location in ECEF [km]

    Returns
    ------
    LAM : float
      ENU longitude measured from Up [deg]
    PHI : float
      ENU latitude measured from E-U plane [deg]
    rg : float
      range to target [km]

    '''

    # Compute position vector in ECEF
    r_ecef = latlonht2ecef(lat, lon, ht)

    # Compute unit vector in ECEF
    rho_ecef = r_ecef - r_site
    rg = np.linalg.norm(rho_ecef)
    rho_hat_ecef = rho_ecef/rg

    # Rotate to ENU
    rho_hat_enu = ecef2enu(rho_hat_ecef, r_site)

    # Compute angles
    PHI = asin(rho_hat_enu[1])*180./pi
    LAM = atan2(rho_hat_enu[0], rho_hat_enu[2])*180./pi

    return LAM, PHI, rg


def enuvec2azel(rho_hat_enu):
    '''
    This function computes az/el given LOS unit vector in ENU frame.
    Parameters
    ------
    rho_hat_enu : 3x1 numpy array
        LOS unit vector in ENU frame

    Returns
    ------
    az : float
        azimuth measured clockwise from North [deg]
    el : float
        elevation measured up from horizon [deg]
    '''
    # Compute Azimuth and Elevation
    el = asin(rho_hat_enu[2]) * 180./pi  # deg
    az = atan2(rho_hat_enu[0], rho_hat_enu[1]) * 180./pi  # deg

    # Convert az to range 0-360
    if az < 0:
        az = az + 360.

    return az, el


def azel2enuvec(az, el):
    '''
    This function computes az/el given LOS unit vector in ENU frame.
    Parameters
    ------
    az : float
        azimuth measured clockwise from North [deg]
    el : float
        elevation measured up from horizon [deg]

    Returns
    ------
    rho_hat_enu : 3x1 numpy array
        LOS unit vector in ENU frame
    '''
    
    # Convert to rad
    az *= pi/180.
    el *= pi/180.
    
    # Compute unit vector
    rho_hat_enu = np.array([[sin(az)*cos(el)], [cos(az)*cos(el)], [sin(el)]])
    
    return rho_hat_enu


def enuangles2azel(LAM, PHI):
    '''
    This function computes az/el given
    a set of angles in the ENU frame.

    Parameters
    ------
    LAM : float
      ENU longitude measured from Up [deg]
    PHI : float
      ENU latitude measured from E-U plane [deg]

    Returns
    ------
    az : float
      azimuth [deg]
    el : float
      elevation [deg]

    '''

    # Convert angles to rad
    LAM = LAM*pi/180.
    PHI = PHI*pi/180.

    # Compute unit vector in ENU
    rho_hat_enu = np.array([[cos(PHI)*sin(LAM)], [sin(PHI)],
                            [cos(PHI)*cos(LAM)]])

    # Compute Azimuth and Elevation
    el = asin(rho_hat_enu[2]) * 180/pi  # deg
    az = atan2(rho_hat_enu[0], rho_hat_enu[1]) * 180/pi  # deg

    # Convert az to range 0-360
    if az < 0:
        az += 360.

    return az, el


def azel2enuangles(az, el):
    '''
    This function computes a set of angles in the ENU frame
    given az/el.

    Parameters
    ------
    az : float
      azimuth [deg]
    el : float
      elevation [deg]

    Returns
    ------
    LAM : float
      ENU longitude measured from Up [deg]
    PHI : float
      ENU latitude measured from E-U plane [deg]
    '''

    # Convert angles to rad
    az = az*pi/180.
    el = el*pi/180.

    # Compute unit vector in ENU
    rho_hat_enu = np.array([[cos(el)*sin(az)], [cos(el)*cos(az)], [sin(el)]])

    # Compute angles
    PHI = asin(rho_hat_enu[1])*180./pi
    LAM = atan2(rho_hat_enu[0], rho_hat_enu[2])*180./pi

    return LAM, PHI


def enuangles2radec(LAM, PHI, r_site, JD, inputs):
    '''
    This function computes topocentric ra/dec given
    a set of angles in the ENU frame, julian date and
    ground station location.

    Parameters
    ------
    LAM : float
      ENU longitude measured from N-U plane [rad]
    PHI : float
      ENU latitude measured from E-U plane [rad]
    r_site : 3x1 numpy array
      postion vector of ground station location in ECEF [km]
    JD : float
      julian date
    inputs: dictionary
      input parameters

    Returns
    ------
    ra : float
      topocentric right ascension [rad]
    dec : float
      topocentric declination [rad]
    '''

    # Compute unit vector in ENU
    rho_hat_enu = np.array([[cos(PHI)*sin(LAM)], [sin(PHI)],
                            [cos(PHI)*cos(LAM)]])

    # Rotate to ECEF
    rho_hat_ecef = enu2ecef(rho_hat_enu, r_site)

    # Rotate to ECI
    myIAU = inputs['myIAU']
    dum = modf(float(JD))
    jdTime = [int(dum[1]), dum[0]]

    res = myIAU.ECEF2ECI(jdTime, position=rho_hat_ecef)
    rho_hat_eci = np.reshape(res[0], (3, 1))

    # Compute angles
    ra = atan2(rho_hat_eci[1], rho_hat_eci[0])
    dec = asin(rho_hat_eci[2])

    return ra, dec


def radec2enuangles(ra, dec, r_site, JD, inputs):
    '''
    This function computes a set of angles in the ENU frame
    given a unit vector in ECEF and ground station location.

    Parameters
    ------
    ra : float
      topocentric right ascension [rad]
    dec : float
      topocentric declination [rad]
    r_site : 3x1 numpy array
      postion vector of ground station location in ECEF [km]
    JD : float
      julian date
    inputs : dictionary
      input parameters

    Returns
    ------
    LAM : float
      ENU longitude measured from Up [rad]
    PHI : float
      ENU latitude measured from E-U plane [rad]

    '''

    # Compute unit vector in ECI
    rho_hat_eci = np.array([[cos(ra)*cos(dec)], [sin(ra)*cos(dec)],
                            [sin(dec)]])

    # Rotate to ECEF
    myIAU = inputs['myIAU']
    dum = modf(float(JD))
    jdTime = [int(dum[1]), dum[0]]

    res = myIAU.ECI2ECEF(jdTime, position=rho_hat_eci)
    rho_hat_ecef = np.reshape(res[0], (3, 1))

    # Rotate to ENU
    rho_hat_enu = ecef2enu(rho_hat_ecef, r_site)

    # Compute angles
    PHI = asin(rho_hat_enu[1])
    LAM = atan2(rho_hat_enu[0], rho_hat_enu[2])

    return LAM, PHI


def LAMlat2PHI(LAM, lat, r, r_site):
    '''
    This function computes the required sensor frame
    ENU angle PHI given a desired sensor frame
    ENU angle LAM, geodetic latitude, orbit radius
    and ground station.

    Parameters
    ------
    LAM : float
      ENU longitude measured from Up [deg]
    lat : float
      geodetic latitude [deg]
    r : float
      scalar orbit radius [km]
    r_site : 3x1 numpy array
      postion vector of ground station location in ECEF [km]

    Returns
    ------
    PHI : float
      ENU latitude measured from E-U plane [deg]
    '''

    # Set up bounds for PHI
    PHI_hi = lat + 40.
    PHI_lo = lat - 40.
    PHI = (PHI_hi + PHI_lo)/2.

    # Iterate to find PHI
    tol = 0.01
    diff = 1.
    while abs(diff) > tol:

        # Save for comparison
        PHI0 = copy.copy(PHI)

        # Compute lat for this PHI
        lat1, lon1, ht1 = enuangles2latlonht(LAM, PHI, r, r_site)

        # Update PHI bounds as appropriate
        if lat1 > lat:
            PHI_hi = copy.copy(PHI)
        else:
            PHI_lo = copy.copy(PHI)

        # Compute diff
        PHI = (PHI_hi + PHI_lo)/2.
        diff = PHI - PHI0

    return PHI


############################################################################
# Orbit Stuff
############################################################################

def mean2ecc(M, e):
    '''
    This function converts from Mean Anomaly to Eccentric Anomaly

    Parameters
    ------
    M : float
      mean anomaly [rad]
    e : float
      eccentricity

    Returns
    ------
    E : float
      eccentric anomaly [rad]
    '''

    # Ensure M is between 0 and pi
    while M < 0:
        M = M + 2*pi

    if M > 2*pi:
        M = fmod(M, 2*pi)

    flag = 0
    if M > pi:
        flag = 1
        M = 2*pi - M

    # Starting guess for E
    E = M + e*sin(M)/(1 - sin(M + e) + sin(M))

    # Initialize loop variable
    f = 1
    tol = 1e-8

    # Iterate using Newton-Raphson Method
    while fabs(f) > tol:
        f = E - e*sin(E) - M
        df = 1 - e*cos(E)
        E = E - f/df

    # Correction for M > pi
    if flag == 1:
        E = 2*pi - E

    return E


def ecc2mean(E, e):
    '''
    This function converts from Eccentric Anomaly to Mean Anomaly

    Parameters
    ------
    E : float
      eccentric anomaly [rad]
    e : float
      eccentricity

    Returns
    ------
    M : float
      mean anomaly [rad]
    '''
    
    M = E - e*sin(E)
    
    return M


def ecc2true(E, e):
    '''
    This function converts from Eccentric Anomaly to True Anomaly

    Parameters
    ------
    E : float
      eccentric anomaly [rad]
    e : float
      eccentricity

    Returns
    ------
    f : float
      true anomaly [rad]
    '''

    f = 2*atan(np.sqrt((1+e)/(1-e))*tan(E/2))

    return f


def true2ecc(f, e):
    '''
    This function converts from True Anomaly to Eccentric Anomaly

    Parameters
    ------
    f : float
      true anomaly [rad]
    e : float
      eccentricity

    Returns
    ------
    E : float
      eccentric anomaly [rad]
    '''

    E = 2*atan(np.sqrt((1-e)/(1+e))*tan(f/2))

    return E


def mean2hyp(M, e):
    '''
    This function converts from Mean Anomaly to Hyperbolic Anomaly

    Parameters
    ------
    M : float
      mean anomaly [rad]
    e : float
      eccentricity

    Returns
    ------
    H : float
      hyperbolic anomaly [rad]
    '''

    # Ensure M is between -pi and pi
    if M > pi or M < -pi:
        print('Error: Expected -pi < M < pi!')

    # Form starting guess for H
    H = M

    # Initialize loop variable
    f = 1
    tol = 1e-8

    # Iterate using Newton-Raphson Method
    while fabs(f) > tol:
        f = e*sinh(H) - H - M
        df = e*cosh(H) - 1
        H = H - f/df

    return H


def element_conversion(x_in, iflag, oflag, GM=3.986004e5, dt=0.):
    '''
    This funciton converts between Keplerian orbital elements
    and inertial frame cartesian coordinates.  The script has an
    optional dt input which will propagate the current orbit state
    using Kepler's equation and assuming two-body dynamics.

    Parameters
    ------
    x_in : 6x1 numpy array
      vector of elements or cartesian coordinates at t0
    iflag : int
      input flag (0 = orbital elements, 1 = cartesian coordiantes,
                  2 = launch elements)
    oflag : int
      output flag (0 = orbital elements, 1 = cartesian coordinates)
    GM : float, optional
      graviational parameter [km^3/s^2] (default=3.986004e5)
    dt : float, optional
      time to propagate [sec] (default=0)

    Returns
    ------
    x_out : 6x1 numpy array
      vector of elements or cartesian coordinates at t0 + dt


    Assumed form and units of x_in and x_out
    ------
    Keplerian Orbital Elements
    ------
    x[0] : a
      Semi-Major Axis             [km]
    x[1] : e
      Eccentricity                [unitless]
    x[2] : i
      Inclination                 [deg]
    x[3] : RAAN
      Right Asc Ascending Node    [deg]
    x[4] : w
      Argument of Periapsis       [deg]
    x[5] : Mo
      Mean anomaly at t0          [deg]

    Cartesian Coordinates (Inertial Frame)
    ------
    x[0] : x
      Position in x               [km]
    x[1] : y
      Position in y               [km]
    x[2] : z
      Position in z               [km]
    x[3] : dx
      Velocity in x               [km/s]
    x[4] : dy
      Velocity in y               [km/s]
    x[5] : dz
      Velocity in z               [km/s]
      
    Launch Elements
    ------
    x[0] : ra
          Radius of apoapsis        [km]
    x[1] : rp
        Radius of periapsis         [km]
    x[2] : i
        Inclination                 [deg]
    x[3] : LTAN
        Local Time of Asc Node      [h,m,s]
    x[4] : w
        Argument of Periapsis       [deg]
    x[5] : lat
        Geodetic lat of launch site [deg]
    x[6] : lon
        Geodetic lon of launch site [deg]
    x[7] : t0
        Time of launch              [UTC_G]
    '''

    # Get initial orbit elements
    if iflag == 0:

        # Retrieve input elements, convert to radians
        a = float(x_in[0])
        e = float(x_in[1])
        i = float(x_in[2]) * pi/180
        RAAN = float(x_in[3]) * pi/180
        w = float(x_in[4]) * pi/180
        Mo = float(x_in[5]) * pi/180

        # Calculate h
        p = a*(1 - e**2)
        h = np.sqrt(GM*p)

        # Calculate n
        if a > 0:
            n = np.sqrt(GM/a**3)
        elif a < 0:
            n = np.sqrt(GM/-a**3)
        else:
            print('Error, input orbit is parabolic, a = ', a)

    elif iflag == 1:

        # Retrieve input cartesian coordinates
        r_vect = x_in[0:3]
        v_vect = x_in[3:6]

        # Calculate orbit parameters
        r = np.linalg.norm(r_vect)
        ir_vect = r_vect/r
        v2 = np.linalg.norm(v_vect)**2
        h_vect = np.cross(r_vect, v_vect, axis=0)
        h = np.linalg.norm(h_vect)

        # Calculate semi-major axis
        a = 1/(2/r - v2/GM)     # km

        # Calculate RAAN and inclination
        ih_vect = h_vect/h
        RAAN = atan2(ih_vect[0], -ih_vect[1])   # rad
        i = acos(ih_vect[2])   # rad

        # Calculate eccentricity
        e_vect = np.cross(v_vect, h_vect, axis=0)/GM - ir_vect
        e = np.linalg.norm(e_vect)

        # Apply correction for circular orbit, choose e_vect to point
        # to ascending node
        if e != 0:
            ie_vect = e_vect/e
        else:
            ie_vect = np.array([[cos(RAAN)], [sin(RAAN)], [0.]])

        # Find orthogonal unit vector to complete perifocal frame
        ip_vect = np.cross(ih_vect, ie_vect, axis=0)

        # Form rotation matrix PN
        PN = np.concatenate(([ie_vect], [ip_vect], [ih_vect]))

        # Calculate argument of periapsis
        w = atan2(PN[0][2], PN[1][2])  # rad

        # Calculate true anomaly, eccentric/hyperbolic anomaly, mean anomaly
        cross1 = np.cross(ie_vect, ir_vect, axis=0)
        tan1 = np.dot(cross1.T, ih_vect)
        tan2 = np.dot(ie_vect.T, ir_vect)
        f = atan2(tan1, tan2)    # rad

        # Calculate M
        if a > 0:
            n = np.sqrt(GM/a**3)
            Erad = 2*atan(np.sqrt((1-e)/(1+e))*tan(f/2))    # rad
            Mo = Erad - e*sin(Erad)   # rad
            while Mo < 0:
                Mo = Mo + 2*pi
        elif a < 0:
            n = np.sqrt(GM/-a**3)
            Hrad = 2*atanh(sqrt((e-1)/(e+1))*tan(f/2))  # rad
            Mo = e*sinh(Hrad) - Hrad  # rad
        else:
            print('Error, input orbit is parabolic, a = ', a)
    
    elif iflag == 2:
        
        # Setup inputs structure
        inputs = {}
        eopFile = os.path.join(DataDir, 'EOP_1962_DATA.txt')
        xysFile = os.path.join(DataDir, 'IAU2006_XYs.txt')
        lsFile = os.path.join(DataDir, 'leapsec.dat')
        myIAU = CU.IAU2006CIO(EOPFile=eopFile, XYsFile=xysFile, LeapSecFile=lsFile)
        inputs['myIAU'] = myIAU
        
        # Retrieve input elements, convert to radians
        ra = float(x_in[0])
        rp = float(x_in[1])
        i = float(x_in[2]) * pi/180
        LTAN_h = float(x_in[3][0])
        LTAN_m = float(x_in[3][1])
        LTAN_s = float(x_in[3][2])
        w = float(x_in[4]) * pi/180
        site_lat = float(x_in[5]) * pi/180
        site_lon = float(x_in[6]) * pi/180
        t0 = x_in[7]
        
        # Compute semi-major axis and eccentricity
        a = (ra + rp)/2.
        e = 1 - (rp/a)
        
        # Compute mean motion and angular momentum
        n = np.sqrt(GM/a**3)
        p = a*(1 - e**2.)
        h = np.sqrt(GM*p)
        
#        print 'ra', ra
#        print 'rp', rp
#        print 'a', a
#        print 'e', e
#        print 'p', p
#        print 'n', n
#        print 'h', h
        
        # Compute curren JED_JD
        JED_JD = UTC_G_2_JED_JD(t0)
#        print 'JED_JD', JED_JD
        
        # Compute LTAN in rad
        LTAN = LTAN_h*(pi/12.) + LTAN_m*(pi/720.) + LTAN_s*(pi/43200)
#        print 'LTAN', LTAN
        
        # Compute current sun right ascension
        sun = ephem.Sun()
        sun.compute((t0[0], t0[1], t0[2], t0[3], t0[4], t0[5]))
        sun_ra = sun.g_ra
        
#        print 'sun_ra', float(sun_ra)
        
        # Compute RAAN (sun LTAN = pi)
        RAAN = sun_ra + (LTAN - pi)
#        print 'RAAN', RAAN
        
        # Compute geodetic longitude of ascending node
        f = -w  # theta = 0 at ascending node
        Erad = 2*atan(np.sqrt((1-e)/(1+e))*tan(f/2))
        r = a*(1 - e*cos(Erad))
        node_eci = np.array([[r*cos(RAAN)], [r*sin(RAAN)], [0.]])
        node_ecef = eci2ecef(node_eci, inputs, JED_JD)
        node_lat, node_lon, node_ht = ecef2latlonht(node_ecef)
        node_lat *= pi/180.
        node_lon *= pi/180.
        
#        print 'node_lat', node_lat
#        print 'node_lon', node_lon
        
        # Find Mo corresponding to lat/lon
        delta_lon = site_lon - node_lon
        delta_lat = site_lat - node_lat
        theta_site = acos(cos(delta_lon)*cos(delta_lat))
        f_site = theta_site - w
        E_site = 2*atan(np.sqrt((1-e)/(1+e))*tan(f_site/2))
        Mo = E_site - e*sin(E_site)
        
#        print 'theta_site', theta_site
#        print 'f_site', f_site
#        print 'Mo', Mo

        
    else:
        print('Error: Invalid Input Flag!')

    # Solve for M(t) = Mo + n*dt
    M = Mo + n*dt   # rad

    # Generate output vector x_out
    if oflag == 0:

        # Convert angles to degrees
        i = i * 180/pi
        RAAN = RAAN * 180/pi
        w = w * 180/pi
        M = M * 180/pi

        x_out = np.array([[a], [e], [i], [RAAN], [w], [M]])

    elif oflag == 1:

        # Find eccentric/hyperbolic anomaly and true anomaly
        if a > 0:
            Erad = mean2ecc(M, e)    # rad
            f = 2*atan(np.sqrt((1+e)/(1-e))*tan(Erad/2))    # rad
            r = a*(1 - e*cos(Erad))     # km
        elif a < 0:
            Hrad = mean2hyp(M, e)    # rad
            f = 2*atan(np.sqrt((e+1)/(e-1))*tanh(Hrad/2))    # rad
            r = a*(1 - e*cos(Hrad))     # km

        # Calculate theta
        theta = f + w   # rad

        # Calculate r_vect and v_vect
        r_vect2 = r * \
            np.array([[cos(RAAN)*cos(theta) - sin(RAAN)*sin(theta)*cos(i)],
                      [sin(RAAN)*cos(theta) + cos(RAAN)*sin(theta)*cos(i)],
                      [sin(theta)*sin(i)]])

        vv1 = cos(RAAN)*(sin(theta) + e*sin(w)) + \
            sin(RAAN)*(cos(theta) + e*cos(w))*cos(i)

        vv2 = sin(RAAN)*(sin(theta) + e*sin(w)) - \
            cos(RAAN)*(cos(theta) + e*cos(w))*cos(i)

        vv3 = -(cos(theta) + e*cos(w))*sin(i)
        v_vect2 = -GM/h * np.array([[vv1], [vv2], [vv3]])

        x_out = np.concatenate([r_vect2, v_vect2])

    else:
        print('Error: Invalid Output Flag!')

    return x_out


def tle2kep(i, RAAN, e, w, M, n, GM=3.986004e5):
    '''
    This function converts a two line element set to Keplerian orbital elements

    Parameters
    ------
    i : float
      inclination [deg]
    RAAN : float
      Right Ascension of Ascending Node [deg]
    e : float
      eccentricity
    w : float
      argument of periapsis [deg]
    M : float
      mean anomaly [deg]
    n : float
      mean motion [rev/mean solar day]
    GM : float, optional
      gravitational parameter (default=3.986004e5)

    Returns
    ------
    x_out : 6x1 numpy array
      Keplerian Orbital Elements
    x[0] : a
      Semi-Major Axis             [km]
    x[1] : e
      Eccentricity                [unitless]
    x[2] : i
      Inclination                 [deg]
    x[3] : RAAN
      Right Asc Ascending Node    [deg]
    x[4] : w
      Argument of Periapsis       [deg]
    x[5] : Mo
      Mean anomaly at t0          [deg]
    '''

    # Compute semi-major axis
    n = n*2*pi/86400.    # rad/s
    a = (GM/n**2)**(1./3.)   # km

    # Form output vector
    x_out = [a, e, i, RAAN, w, M]

    return x_out


def cart2tle(x_in, GM=3.986004e5):
    '''
    This function converts an input state vector in inertial cartesian
    coordinates to a two-line element set.

    Parameters
    ------
    x_in : 6x1 numpy array
      Cartesian Coordinates (Inertial Frame)
    x[0] : x
      Position in x               [km]
    x[1] : y
      Position in y               [km]
    x[2] : z
      Position in z               [km]
    x[3] : dx
      Velocity in x               [km/s]
    x[4] : dy
      Velocity in y               [km/s]
    x[5] : dz
      Velocity in z               [km/s]
    GM : float, optional
      Gravitiational Parameter [km^3/s^2] (default=3.986004e5)

    Returns
    ------
    x_out : list
      Two-line Element set values
    x[0] : i
      Inclination [deg]
    x[1] : RAAN
      Right Asc Ascending Node [deg]
    x[2] : e
      Eccentricity
    x[3] : w
      Argument of Periapsis [deg]
    x[4] : M
      Mean Anomaly [deg]
    x[5] : n
      Mean Motion [rev/mean solar day]
    '''

    # Compute orbit elements
    elem = element_conversion(x_in, 1, 0)

    # Break out elements
    a = float(elem[0])
    e = float(elem[1])
    i = float(elem[2])
    RAAN = float(elem[3])
    w = float(elem[4])
    M = float(elem[5])

    if w < 0:
        w += 360.

    # Compute mean motion [rev/day]
    n = np.sqrt(GM/(a**3))  # rad/sec
    n = n*86400./(2*pi)  # rev/day

    # Form output vector
    x_out = [i, RAAN, e, w, M, n]

    return x_out
