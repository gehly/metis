import numpy as np
# from math import pi, sin, cos, tan, asin, acos, atan, atan2
import math
# from datetime import datetime
# import sys
# import os
# import inspect
import copy
# import time

# filename = inspect.getframeinfo(inspect.currentframe()).filename
# current_dir = os.path.dirname(os.path.abspath(filename))
#
# ind = current_dir.find('metis')
# metis_dir = current_dir[0:ind+5]
# sys.path.append(metis_dir)

from utilities import eop_functions as eop
from utilities import time_systems as timesys
from utilities.constants import Re, rec_f



###############################################################################
#
# This file contains function to perform transformations between different
# coordinate systems.
#
# References:
#
#  [1] Petit, Gerard and Brian Luzum, "IERS Conventions (2010)," 
#        International Earth Rotation and Reference Systems Service (IERS),
#        IERS Conventions Center, IERS Technical Note No. 36, December 2010.
# 
#  [2] International Astronomical Union, "Standards of Fundamental 
#         Astronomy: SOFA Tools for Earth Attitude," December 21, 2009.
#
#  [3] Wallace, P.T. and N. Capitaine, "Precession-Nutation Precedures
#         Consistent with IAU 2006 Resolutions," Astronomy & Astrophysics,
#         Volume 459, pages 981-985, December 2006.
#
#  [4] Coppola, V., J.H. Seago, and D.A. Vallado, "The IAU 2000A and IAU
#         2006 Precession-Nutation Theories and their Implementation," 
#         AAS 09-159, 2009.
#
#  [5] Bradley, B.K., Sibois, A., and Axelrad, P., "Influence of ITRS/GCRS
#        implementation for astrodynamics: Coordinate transformations,"
#        Advances in Space Research, Vol 57, 2016, pp 850-866.
#
#  [6] Vallado, D.A. "Fundamentals of Astrodynamics and Applications".
#         Third Edition. Microcosm Press. 2007.
#
###############################################################################


def teme2gcrf(r_TEME, v_TEME, UTC, IAU1980nut, EOP_data):
    '''
    This function converts position and velocity vectors from the True
    Equator Mean Equinox (TEME) frame used for TLEs to the GCRF inertial
    frame.
    
    Parameters
    ------
    r_TEME : 3x1 numpy array
        position vector in TEME frame
    v_TEME : 3x1 numpy array
        velocity vector in TEME frame
    UTC : datetime object
        time in UTC
    IAU1980nut : 2D numpy array
        nutation coefficients
    EOP_data : dictionary
        EOP data for the given time including pole coordinates and offsets,
        time offsets, and length of day  
    
    Returns
    ------
    r_GCRF : 3x1 numpy array
        position vector in GCRF frame
    v_GCRF : 3x1 numpy array
        velocity vector in GCRF frame    
    '''
    
    # Compute TT in JD format
    TT_JD = timesys.utcdt2ttjd(UTC, EOP_data['TAI_UTC'])
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = timesys.jd2cent(TT_JD)
    
    # IAU 1976 Precession
    P = eop.compute_precession_IAU1976(TT_cent)
    
    # IAU 1980 Nutation
    N, FA, Eps0, Eps_true, dPsi, dEps = \
        eop.compute_nutation_IAU1980(IAU1980nut, TT_cent, EOP_data['ddPsi'],
                                 EOP_data['ddEps'])

    # Equation of the Equinonx 1982
    R = eop.eqnequinox_IAU1982_simple(dPsi, Eps0)
    
    # Compute transformation matrix and output
    GCRF_TEME = np.dot(P, np.dot(N, R))
    
    r_GCRF = np.dot(GCRF_TEME, r_TEME)
    v_GCRF = np.dot(GCRF_TEME, v_TEME)
    
    return r_GCRF, v_GCRF


def gcrf2teme(r_GCRF, v_GCRF, UTC, IAU1980nut, EOP_data):
    '''
    This function converts position and velocity vectors from the GCRF inertial
    frame to the True Equator Mean Equinox (TEME) frame used for TLEs. 
    
    Parameters
    ------
    r_GCRF : 3x1 numpy array
        position vector in GCRF frame
    v_GCRF : 3x1 numpy array
        velocity vector in GCRF frame 
    UTC : datetime object
        time in UTC
    IAU1980nut : 2D numpy array
        nutation coefficients
    EOP_data : dictionary
        EOP data for the given time including pole coordinates and offsets,
        time offsets, and length of day  
    
    Returns
    ------ 
    r_TEME : 3x1 numpy array
        position vector in TEME frame
    v_TEME : 3x1 numpy array
        velocity vector in TEME frame
    '''
    
    # Compute TT in JD format
    TT_JD = timesys.utcdt2ttjd(UTC, EOP_data['TAI_UTC'])
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = timesys.jd2cent(TT_JD)
    
    # IAU 1976 Precession
    P = eop.compute_precession_IAU1976(TT_cent)
    
    # IAU 1980 Nutation
    N, FA, Eps0, Eps_true, dPsi, dEps = \
        eop.compute_nutation_IAU1980(IAU1980nut, TT_cent, EOP_data['ddPsi'],
                                 EOP_data['ddEps'])

    # Equation of the Equinonx 1982
    R = eop.eqnequinox_IAU1982_simple(dPsi, Eps0)
    
    # Compute transformation matrix and output
    GCRF_TEME = np.dot(P, np.dot(N, R))
    
    r_TEME = np.dot(GCRF_TEME.T, r_GCRF)
    v_TEME = np.dot(GCRF_TEME.T, v_GCRF)
    
    return r_TEME, v_TEME


def gcrf2itrf(r_GCRF, v_GCRF, UTC, EOP_data, XYs_df=[]):
    '''
    This function converts a position and velocity vector in the GCRF(ECI)
    frame to the ITRF(ECEF) frame using the IAU 2006 precession and 
    IAU 2000A_R06 nutation theories. This routine employs a hybrid of the 
    "Full Theory" using Fukushima-Williams angles and the CIO-based method.  

    Specifically, this routine interpolates a table of X,Y,s values and then
    uses them to construct the BPN matrix directly.  The X,Y,s values in the 
    data table were generated using Fukushima-Williams angles and the 
    IAU 2000A_R06 nutation theory.  This general scheme is outlined in [3]
    and [4].
    
    Parameters
    ------
    r_GCRF : 3x1 numpy array
        position vector in GCRF
    v_GCRF : 3x1 numpy array
        velocity vector in GCRF
    UTC : datetime object
        time in UTC
    EOP_data : dictionary
        EOP data for the given time including pole coordinates and offsets,
        time offsets, and length of day  
    
    Returns
    ------
    r_ITRF : 3x1 numpy array
        position vector in ITRF
    v_ITRF : 3x1 numpy array
        velocity vector in ITRF
    
    '''
    
    # Form column vectors
    r_GCRF = np.reshape(r_GCRF, (3,1))
    v_GCRF = np.reshape(v_GCRF, (3,1))
        
    # Compute UT1 in JD format
    UT1_JD = timesys.utcdt2ut1jd(UTC, EOP_data['UT1_UTC'])
    
    # Compute TT in JD format
    TT_JD = timesys.utcdt2ttjd(UTC, EOP_data['TAI_UTC'])
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = timesys.jd2cent(TT_JD)
    
    # Construct polar motion matrix (ITRS to TIRS)
    W = eop.compute_polarmotion(EOP_data['xp'], EOP_data['yp'], TT_cent)
    
    # Contruct Earth rotaion angle matrix (TIRS to CIRS)
    R = eop.compute_ERA(UT1_JD)
    
    # Construct Bias-Precessino-Nutation matrix (CIRS to GCRS/ICRS)
    XYs_data = eop.init_XYs2006(UTC, UTC, XYs_df)
    
    X, Y, s = eop.get_XYs(XYs_data, TT_JD)
    
    # Add in Free Core Nutation (FCN) correction
    X = EOP_data['dX'] + X  # rad
    Y = EOP_data['dY'] + Y  # rad
    
    # Compute Bias-Precssion-Nutation (BPN) matrix
    BPN = eop.compute_BPN(X, Y, s)
    
    # Transform position vector
    eci2ecef = np.dot(W.T, np.dot(R.T, BPN.T))
    r_ITRF = np.dot(eci2ecef, r_GCRF)

    # Transform velocity vector
    # Calculate Earth rotation rate, rad/s (Vallado p227)
    wE = 7.29211514670639e-5*(1 - EOP_data['LOD']/86400)                    
    r_TIRS = np.dot(W, r_ITRF)
        
    v_ITRF = np.dot(W.T, (np.dot(R.T, np.dot(BPN.T, v_GCRF)) - 
                          np.cross(np.array([[0.],[0.],[wE]]), r_TIRS, axis=0)))
    
    
    
    return r_ITRF, v_ITRF


def itrf2gcrf(r_ITRF, v_ITRF, UTC, EOP_data, XYs_df=[]):
    '''
    This function converts a position and velocity vector in the ITRF(ECEF)
    frame to the GCRF(ECI) frame using the IAU 2006 precession and 
    IAU 2000A_R06 nutation theories. This routine employs a hybrid of the 
    "Full Theory" using Fukushima-Williams angles and the CIO-based method.  

    Specifically, this routine interpolates a table of X,Y,s values and then
    uses them to construct the BPN matrix directly.  The X,Y,s values in the 
    data table were generated using Fukushima-Williams angles and the 
    IAU 2000A_R06 nutation theory.  This general scheme is outlined in [3]
    and [4].
    
    Parameters
    ------
    r_ITRF : 3x1 numpy array
        position vector in ITRF
    v_ITRF : 3x1 numpy array
        velocity vector in ITRF
    UTC : datetime object
        time in UTC
    EOP_data : dictionary
        EOP data for the given time including pole coordinates and offsets,
        time offsets, and length of day  
    
    Returns
    ------
    r_GCRF : 3x1 numpy array
        position vector in GCRF
    v_GCRF : 3x1 numpy array
        velocity vector in GCRF
    
    '''
    
    # Form column vectors
    r_ITRF = np.reshape(r_ITRF, (3,1))
    v_ITRF = np.reshape(v_ITRF, (3,1))
    
    # Compute UT1 in JD format
    UT1_JD = timesys.utcdt2ut1jd(UTC, EOP_data['UT1_UTC'])
    
    # Compute TT in JD format
    TT_JD = timesys.utcdt2ttjd(UTC, EOP_data['TAI_UTC'])
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = timesys.jd2cent(TT_JD)
    
    # Construct polar motion matrix (ITRS to TIRS)
    W = eop.compute_polarmotion(EOP_data['xp'], EOP_data['yp'], TT_cent)
    
    # Contruct Earth rotaion angle matrix (TIRS to CIRS)
    R = eop.compute_ERA(UT1_JD)
    
    # Construct Bias-Precessino-Nutation matrix (CIRS to GCRS/ICRS)
    XYs_data = eop.init_XYs2006(UTC, UTC, XYs_df)
    X, Y, s = eop.get_XYs(XYs_data, TT_JD)
    
    # Add in Free Core Nutation (FCN) correction
    X = EOP_data['dX'] + X  # rad
    Y = EOP_data['dY'] + Y  # rad
    
    # Compute Bias-Precssion-Nutation (BPN) matrix
    BPN = eop.compute_BPN(X, Y, s)
    
    # Transform position vector
    ecef2eci = np.dot(BPN, np.dot(R, W))
    r_GCRF = np.dot(ecef2eci, r_ITRF)
    
    # Transform velocity vector
    # Calculate Earth rotation rate, rad/s (Vallado p227)
    wE = 7.29211514670639e-5*(1 - EOP_data['LOD']/86400)                    
    r_TIRS = np.dot(W, r_ITRF)
    
    v_GCRF = np.dot(BPN, np.dot(R, (np.dot(W, v_ITRF) + 
                    np.cross(np.array([[0.],[0.],[wE]]), r_TIRS, axis=0))))  
    
    return r_GCRF, v_GCRF


def eci2ric(rc_vect, vc_vect, Q_eci=[]):
    '''
    This function computes the rotation from ECI to RIC and rotates input
    Q_eci (vector or matrix) to RIC.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    Q_eci : 3x1 or 3x3 numpy array
      vector or matrix in ECI

    Returns
    ------
    Q_ric : 3x1 or 3x3 numpy array
      vector or matrix in RIC
    '''
    
    # Reshape inputs
    rc_vect = rc_vect.reshape(3,1)
    vc_vect = vc_vect.reshape(3,1)

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)

    ON = np.concatenate((OR.T, OT.T, OH.T))

    # Rotate Q_eci as appropriate for vector or matrix
    if len(Q_eci) == 0:
        Q_ric = ON
    elif np.size(Q_eci) == 3:
        Q_eci = Q_eci.reshape(3,1)
        Q_ric = np.dot(ON, Q_eci)
    else:
        Q_ric = np.dot(np.dot(ON, Q_eci), ON.T)

    return Q_ric


def ric2eci(rc_vect, vc_vect, Q_ric=[]):
    '''
    This function computes the rotation from RIC to ECI and rotates input
    Q_ric (vector or matrix) to ECI.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    Q_ric : 3x1 or 3x3 numpy array
      vector or matrix in RIC

    Returns
    ------
    Q_ric : 3x1 or 3x3 numpy array
      vector or matrix in ECI
    '''
    
    # Reshape inputs
    rc_vect = rc_vect.reshape(3,1)
    vc_vect = vc_vect.reshape(3,1)

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
    if len(Q_ric) == 0:
        Q_eci = NO
    elif np.size(Q_ric) == 3:
        Q_eci = np.dot(NO, Q_ric)
    else:
        Q_eci = np.dot(np.dot(NO, Q_ric), NO.T)

    return Q_eci


def eci2ric_vel(rc_vect, vc_vect, rho_ric, drho_eci):
    '''
    This function computes the rotation from ECI to RIC and rotates input
    relative velocity drho_eci to RIC.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    rho_ric : 3x1 numpy array
      relative position vector in RIC
    drho_eci : 3x1 numpy array
      relative velocity vector in ECI

    Returns
    ------
    drho_ric : 3x1 numpy array
      relative velocity in RIC

    '''
    
    # Reshape inputs
    rc_vect = rc_vect.reshape(3,1)
    vc_vect = vc_vect.reshape(3,1)
    rho_ric = rho_ric.reshape(3,1)
    drho_eci = drho_eci.reshape(3,1)

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)

    ON = np.concatenate((OR.T, OT.T, OH.T))

    # Compute angular velocity vector
    dtheta = h/rc**2.
    w = np.reshape([0., 0., dtheta], (3,1))
    
    # Compute relative velocity vector using kinematic identity
    drho_ric = np.dot(ON, drho_eci) - np.cross(w, rho_ric, axis=0)

    return drho_ric


def ric2eci_vel(rc_vect, vc_vect, rho_ric, drho_ric):
    '''
    This function computes the rotation from RIC to ECI and rotates input
    relative velocity drho_ric to ECI.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    rho_ric : 3x1 numpy array
      relative position vector in RIC
    drho_ric : 3x1 numpy array
      relative velocity vector in RIC

    Returns
    ------
    drho_eci : 3x1 numpy array
      relative velocity in ECI

    '''
    
    # Reshape inputs
    rc_vect = rc_vect.reshape(3,1)
    vc_vect = vc_vect.reshape(3,1)
    rho_ric = rho_ric.reshape(3,1)
    drho_ric = drho_ric.reshape(3,1)

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)

    ON = np.concatenate((OR.T, OT.T, OH.T))
    NO = ON.T
    
    # Compute angular velocity vector
    dtheta = h/rc**2.
    w = np.reshape([0., 0., dtheta], (3,1))
    
    # Compute relative velocity vector using kinematic identity
    drho_eci = np.dot(NO, (drho_ric + np.cross(w, rho_ric, axis=0))) 
    
    return drho_eci


def lvlh2ric():
    '''
    This function computes the rotation matrix to convert coordinates from the
    LVLH frame to the orbit (RIC) frame. LVLH is used to define roll-pitch-yaw
    angles (x-axis along track, z-axis toward nadir)
    
    r_RIC = OL * r_LVLH
    
    Parameters
    ------
    None
    
    Returns
    ------
    OL : 3x3 numpy array
        DCM rotation matrix
        
    '''

    OL = np.array([[0.,  0., -1.],
                   [1.,  0.,  0.],
                   [0., -1.,  0.]])
    
    return OL


def ric2lvlh():
    '''
    This function computes the rotation matrix to convert coordinates from the
    orbit (RIC) frame to the LVLH frame. LVLH is used to define roll-pitch-yaw
    angles (x-axis along track, z-axis toward nadir)
    
    r_LVLH = LO * r_RIC
    
    Parameters
    ------
    None
    
    Returns
    ------
    LO : 3x3 numpy array
        DCM rotation matrix
        
    '''

    OL = np.array([[0.,  0., -1.],
                   [1.,  0.,  0.],
                   [0., -1.,  0.]])
    
    LO = OL.T
    
    return LO


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
    lat = lat*math.pi/180  # rad
    lon = lon*math.pi/180  # rad

    # Compute rotation matrix
    lat1 = math.pi/2 - lat
    lon1 = math.pi/2 + lon

    R1 = np.array([[1.,               0.,             0.],
                   [0.,   math.cos(lat1), math.sin(lat1)],
                   [0.,  -math.sin(lat1), math.cos(lat1)]])

    R3 = np.array([[math.cos(lon1),  math.sin(lon1), 0.],
                   [-math.sin(lon1), math.cos(lon1), 0.],
                   [0.,              0.,             1.]])

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
    lat = lat*math.pi/180  # rad
    lon = lon*math.pi/180  # rad

    # Compute rotation matrix
    lat1 = math.pi/2 - lat
    lon1 = math.pi/2 + lon

    R1 = np.array([[1.,               0.,             0.],
                   [0.,   math.cos(lat1), math.sin(lat1)],
                   [0.,  -math.sin(lat1), math.cos(lat1)]])

    R3 = np.array([[math.cos(lon1),   math.sin(lon1), 0.],
                   [-math.sin(lon1),  math.cos(lon1), 0.],
                   [0.,                           0., 1.]])

    R = np.dot(R1, R3)

    R2 = R.T

    r_ecef = np.dot(R2, r_enu)

    return r_ecef


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

    a = Re   # km

    # Get components from position vector
    x = float(r_ecef[0])
    y = float(r_ecef[1])
    z = float(r_ecef[2])

    # Compute longitude
    f = 1./rec_f
    e = np.sqrt(2.*f - f**2.)
    lon = math.atan2(y, x) * 180./math.pi  # deg

    # Iterate to find height and latitude
    p = np.sqrt(x**2. + y**2.)  # km
    lat = 0.*math.pi/180.
    lat_diff = 1.
    tol = 1e-12

    while abs(lat_diff) > tol:
        lat0 = copy.copy(lat)  # rad
        N = a/np.sqrt(1 - e**2*(math.sin(lat0)**2))  # km
        ht = p/math.cos(lat0) - N
        lat = math.atan((z/p)/(1 - e**2*(N/(N + ht))))
        lat_diff = lat - lat0

    lat = lat*180./math.pi  # deg

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

    # Convert to radians
    lat = lat*math.pi/180    # rad
    lon = lon*math.pi/180    # rad

    # Compute flattening and eccentricity
    f = 1/rec_f
    e = np.sqrt(2*f - f**2)

    # Compute ecliptic plane and out of plane components
    C = Re/np.sqrt(1 - e**2*math.sin(lat)**2)
    S = Re*(1 - e**2)/np.sqrt(1 - e**2*math.sin(lat)**2)

    rd = (C + ht)*math.cos(lat)
    rk = (S + ht)*math.sin(lat)

    # Compute ECEF position vector
    r_ecef = np.array([[rd*math.cos(lon)], [rd*math.sin(lon)], [rk]])

    return r_ecef


def latlon2dist(lat1, lon1, lat2, lon2):
    '''
    This function computes the distance in km between two points on the WGS-84
    ellipsoid, given by geodetic latitude and longitude.
    
    Parameters
    ------
    lat1 : float
        geodetic latitude of first location [deg]
    lon1 : float
        longitude of first location [deg]
    lat2 : float
        geodetic latitude of second location [deg]
    lon2 : float
        longitude of second location [deg]
        
    Returns
    ------
    s : float
        ellipsoidal distance between the two points [km]
    alpha1 : float
        forward azimuth at location 1 [deg]
    alpha2 : float
        forward azimuth at location 2 [deg]
    
    
    Reference
    ------
    Vincenty, T., "Direct and Inverse Solutions of Geodesics on the Ellipsoid
    with Application of Nested Equations," Survey Review No 176, 1975.
    
    '''
    
    # Convert angles to rad
    lat1 = lat1*math.pi/180.
    lon1 = lon1*math.pi/180.
    lat2 = lat2*math.pi/180.
    lon2 = lon2*math.pi/180.
    
    # Earth Parameters - WGS84 
    a = Re
    f = 1/rec_f
    b = (1-f)*a
    
    # Compute U1, U2, L
    U1 = math.atan((1-f) * math.tan(lat1))
    U2 = math.atan((1-f) * math.tan(lat2))
    L = lon2 - lon1
    
    # Iterate to convergence
    lam = L
    diff = 1.
    tol = 1e-10
    count = 0
    max_iters = 100
    while diff > tol:
        
        # Loop parameters
        count += 1
        lam_prev = float(lam)
                
        sin_sigma = np.sqrt( (math.cos(U2)*math.sin(lam))**2. +
                             (math.cos(U1)*math.sin(U2) - math.sin(U1)*math.cos(U2)*math.cos(lam))**2. )
        cos_sigma = math.sin(U1)*math.sin(U2) + math.cos(U1)*math.cos(U2)*math.cos(lam)
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = math.cos(U1)*math.cos(U2)*math.sin(lam)/math.sin(sigma)
        cos_2sigm = math.cos(sigma) - (2.*math.sin(U1)*math.sin(U2))/(1. - sin_alpha**2.)
        C = (f/16.) * (1. - sin_alpha**2.) * (4. + f*(4. - 3.*(1 - sin_alpha**2.)))
        
        lam = L + (1.-C)*f*sin_alpha*(sigma + C*sin_sigma*(cos_2sigm + C*cos_sigma*(-1. + 2.*cos_2sigm**2.)))
        diff = abs(lam - lam_prev)
        
        if count > max_iters:
            print(count)
            print(diff)
            break
    
        
    u2 = (1. - sin_alpha**2.) * ((a**2. - b**2.)/b**2.)
    A = 1. + (u2/16384.)*(4096. + u2*(-768. + u2*(320. - 175.*u2)))
    B = (u2/1024.)*(256. + u2*(-128. + u2*(74. - 47.*u2)))
    delta_sigma = B*sin_sigma*(cos_2sigm + 0.25*B*(cos_sigma*(-1. + 2.*cos_2sigm**2.) - (B/6.)*cos_2sigm*(-3. + 4.*sin_sigma**2.)*(-3. + 4.*cos_2sigm**2.)))
    s = b*A*(sigma - delta_sigma)
    alpha1 = math.atan2((math.cos(U2)*math.sin(lam)), (math.cos(U1)*math.sin(U2) - math.sin(U1)*math.cos(U2)*math.cos(lam))) * 180./math.pi
    alpha2 = math.atan2((math.cos(U1)*math.sin(lam)), (-math.sin(U1)*math.cos(U2) + math.cos(U1)*math.sin(U2)*math.cos(lam))) * 180./math.pi
    
    alpha1 = alpha1 % 360.
    alpha2 = alpha2 % 360.
        
    return s, alpha1, alpha2


def dist2latlon(lat1, lon1, s, alpha1):
    '''
    This function computes the latitude and longitude of a point a given
    distance and azimuth angle from an initial latitude/longitude.
    
    Parameters
    ------
    lat1 : float
        geodetic latitude of first location [deg]
    lon1 : float
        longitude of first location [deg]
    s : float
        ellipsoidal distance between the two points [km]
    alpha1 : float
        forward azimuth at location 1 [deg]    
        
    Returns
    ------
    lat2 : float
        geodetic latitude of second location [deg]
    lon2 : float
        longitude of second location [deg]    
    alpha2 : float
        forward azimuth at location 2 [deg]
    
    
    Reference
    ------
    Vincenty, T., "Direct and Inverse Solutions of Geodesics on the Ellipsoid
    with Application of Nested Equations," Survey Review No 176, 1975.
    
    '''
    
    # Convert to radians
    lat1 = lat1*math.pi/180.
    lon1 = lon1*math.pi/180.
    alpha1 = alpha1*math.pi/180.
    
    # Earth Parameters - WGS84 
    a = Re
    f = 1/rec_f
    b = (1-f)*a
    
    # Compute values
    U1 = math.atan((1-f) * math.tan(lat1))
    sigma1 = math.atan2(math.tan(U1), math.cos(alpha1))
    sin_alpha = math.cos(U1)*math.sin(alpha1)
    u2 = (1. - sin_alpha**2.) * ((a**2. - b**2.)/b**2.)
    A = 1. + (u2/16384.)*(4096. + u2*(-768. + u2*(320. - 175.*u2)))
    B = (u2/1024.)*(256. + u2*(-128. + u2*(74. - 47.*u2)))
    
    # Iterate to convergence
    sigma = s/(b*A)
    diff = 1.
    tol = 1e-10
    count = 0
    max_iters = 100
    while diff > tol:
        
        sigma_prev = float(sigma)
        
        sin_sigma = math.sin(sigma)
        cos_sigma = math.cos(sigma)
        cos_2sigm = math.cos(2.*sigma1 + sigma)
        delta_sigma = B*sin_sigma*(cos_2sigm + 0.25*B*(cos_sigma*(-1. + 2.*cos_2sigm**2.) -
                                                       (B/6.)*cos_2sigm*(-3. + 4.*sin_sigma**2.) *
                                                       (-3. + 4.*cos_2sigm**2.)))
        
        sigma = s/(b*A) + delta_sigma
        diff = abs(sigma - sigma_prev)

        count += 1
        if count > max_iters:
            print(count)
            print(diff)
            break
        
    sin_sigma = math.sin(sigma)
    cos_sigma = math.cos(sigma)
    cos_2sigm = math.cos(2.*sigma1 + sigma)
        
    lat2 = math.atan2( (math.sin(U1)*cos_sigma + math.cos(U1)*sin_sigma*math.cos(alpha1)),
                       (1.-f)*np.sqrt(sin_alpha**2. +
                                      (math.sin(U1)*sin_sigma - math.cos(U1)*cos_sigma*math.cos(alpha1))**2.) )

    lam = math.atan2( sin_sigma*math.sin(alpha1), (math.cos(U1)*cos_sigma - math.sin(U1)*sin_sigma*math.cos(alpha1)) )
    C = (f/16.) * (1. - sin_alpha**2.) * (4. + f*(4. - 3.*(1 - sin_alpha**2.)))
    L = lam - (1.-C)*f*sin_alpha*(sigma + C*sin_sigma*(cos_2sigm + C*cos_sigma*(-1. + 2.*cos_2sigm**2.)))
    lon2 = L + lon1
    alpha2 = math.atan2(sin_alpha, (-math.sin(U1)*sin_sigma + math.cos(U1)*cos_sigma*math.cos(alpha1)))
    
    # Convert to deg
    lat2 = lat2 * 180./math.pi
    lon2 = lon2 * 180./math.pi
    alpha2 = alpha2 * 180./math.pi
    
    alpha2 = alpha2 % 360.
    
    return lat2, lon2, alpha2


def inc2az(lat, inc):
    '''
    This function computes the aziumth of an orbit given for a given location
    given inclination and latitude of a point on the ground
    
    Parameters
    ------
    lat : float
        geodetic latitude [deg]
    inc : float
        inclination [deg]
        
    Returns
    ------
    az : float
        azimuth [deg]
        
    
    '''
    
    # Conver to rad
    lat = lat * math.pi/180.
    inc = inc * math.pi/180.
        
    # Compute azimuth in deg for both ascending and descending passage
    if math.cos(inc)/math.cos(lat) > 1.:
        az_asc = 90.
    elif math.cos(inc)/math.cos(lat) < -1.:
        az_asc = 270.
    else:    
        az_asc = math.asin(math.cos(inc)/math.cos(lat)) * 180./math.pi
        
    az_desc = 180. - az_asc
    
    # Wrap to 360
    az_asc = az_asc % 360.
    az_desc = az_desc % 360.
    
    return az_asc, az_desc


###############################################################################
# Unit Test
###############################################################################
    

def unit_test_latlondist():
    
    lat1 = -20
    lon1 = 15.
    lat2 = -30.
    lon2 = 18.
    
    s, alpha1, alpha2 = latlon2dist(lat1, lon1, lat2, lon2)
    
    print(s, alpha1, alpha2)
    
    lat2, lon2, alpha2 = dist2latlon(lat1, lon1, s, alpha1)
    
    print(lat2, lon2, alpha2)
    
    
    return


def unit_test_inc2az():
    
    lat = -82.
    inc = 98.
    
    az_asc, az_desc = inc2az(lat, inc)
    
    print(az_asc, az_desc)
    
    return


if __name__ == '__main__':
    
    
    # unit_test_latlondist()
    
    # unit_test_inc2az()
    
    

    obj_id_list = [43014]
#    UTC = datetime(2018, 6, 23, 0, 0, 0)   
#    
#    r_GCRF = np.array([[ 1970.55034496],
#                       [-5808.61407296],
#                       [ 3477.84103265]])
#    v_GCRF = np.array([[ 0.01631311],
#                       [-3.68975479],
#                       [-6.53041229]])
#    
#    eop_alldata = get_celestrak_eop_alldata()    
#    EOP_data = get_eop_data(eop_alldata, UTC)
#    
#    print(EOP_data)
#    
#    r_ITRF, v_ITRF = gcrf2itrf(r_GCRF, v_GCRF, UTC, EOP_data)
#    
#    print(r_ITRF)
#    print(v_ITRF)
#    
#    UTC = datetime(2018, 6, 23, 2, 13, 21)
#    r_GCRF = np.array([[-7.47620899e+02],
#                       [ 3.91726014e+03],
#                       [-5.60901124e+03]])
#    
#    v_GCRF = np.array([[ 2.71877455e-03],
#                       [-6.23954021e+00],
#                       [-4.34951830e+00]])
#    
#    r_ITRF = np.array([[-3651.380321],
#                       [ 1598.487431],
#                       [-5610.448359]])
#    
#    v_ITRF = np.array([[ 5.276523548],
#                       [-3.242081015],
#                       [-4.349310553]])
#    
#    EOP_data = get_eop_data(eop_alldata, UTC)
#    r_check, v_check = gcrf2itrf(r_GCRF, v_GCRF, UTC, EOP_data)
#    
#    print(r_check)
#    print(v_check)
#    
#    print(r_check - r_ITRF)
#    print(v_check - v_ITRF)
#    
#    r_check2, v_check2 = itrf2gcrf(r_check, v_check, UTC, EOP_data)
#    
#    print(r_check2)
#    print(v_check2)
#    
#    print(r_check2 - r_GCRF)
#    print(v_check2 - v_GCRF)
#    
#    
#    r_GCRF = np.array([[10000.], [5000.], [3000.]])
#    v_GCRF = np.array([[-2.], [-3.], [4.]])
#    
#    UTC = datetime(2018, 6, 25, 6, 30, 10)
#    EOP_data = get_eop_data(eop_alldata, UTC)
#    print(EOP_data)
#    
#    r_ITRF, v_ITRF = gcrf2itrf(r_GCRF, v_GCRF, UTC, EOP_data)
#    r_check, v_check = itrf2gcrf(r_ITRF, v_ITRF, UTC, EOP_data)
#    
#    print(r_ITRF)
#    print(v_ITRF)
#    print(r_GCRF - r_check)
#    print(v_GCRF - v_check)
#    
#    
#    IAU1980nut = get_nutation_data()
#    r_TEME, v_TEME = gcrf2teme(r_GCRF, v_GCRF, UTC, IAU1980nut, EOP_data)
#    r_check, v_check = teme2gcrf(r_TEME, v_TEME, UTC, IAU1980nut, EOP_data)
#    
#    print(r_TEME)
#    print(v_TEME)
#    print(r_check - r_GCRF)
#    print(v_check - v_GCRF)
#    
#    rc_vect = np.array([7000., 2000., 1000.])
#    vc_vect = np.array([0., -7., -2.])
#    
#    Q_eci = np.random.rand(3,)
#    print(Q_eci)
#    Q_ric = eci2ric(rc_vect, vc_vect, Q_eci)
#    Q_eci2 = ric2eci(rc_vect, vc_vect, Q_ric)
#    print(Q_eci2.flatten())
#    print(Q_eci - Q_eci2.flatten())
    
    
#    lat_gs = -35.29
#    lon_gs = 149.17
#    ht_gs = 0.606 # km	
#    
#    r_site = latlonht2ecef(lat_gs, lon_gs, ht_gs)
#    
#    print(r_site)
#    
#    r_enu = ecef2enu(r_site, r_site)    
#    
#    print(r_enu)
    
    
