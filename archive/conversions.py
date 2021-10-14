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
