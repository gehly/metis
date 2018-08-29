import numpy as np
from math import pi, sin, cos, tan, fmod, fabs, atan, atan2, acos, asin
from math import sinh, cosh, tanh, atanh
from datetime import datetime
import os
import sys

sys.path.append('../')

from utilities.constants import GME

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


def mean2osc(elem):
    '''
    This function converts mean Keplerian elements to osculating Keplerian
    elements using Brouwer-Lyddane Theory.
    
    References
    ------
    [1] Schaub, H. and Junkins, J.L., Analytical Mechanics of Space Systems."
        2nd ed., 2009.
    '''
    
    
    return


def osc2mean(elem):
    '''
    This function converts osculating Keplerian elements to mean Keplerian
    elements using Brouwer-Lyddane Theory.
    
    References
    ------
    [1] Schaub, H. and Junkins, J.L., Analytical Mechanics of Space Systems."
        2nd ed., 2009.
    '''
    
    
    return


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
            Hrad = 2*atanh(np.sqrt((e-1)/(e+1))*tan(f/2))  # rad
            Mo = e*sinh(Hrad) - Hrad  # rad
        else:
            print('Error, input orbit is parabolic, a = ', a)
    
#    elif iflag == 2:
#        
#        # Setup inputs structure
#        inputs = {}
#        eopFile = os.path.join(DataDir, 'EOP_1962_DATA.txt')
#        xysFile = os.path.join(DataDir, 'IAU2006_XYs.txt')
#        lsFile = os.path.join(DataDir, 'leapsec.dat')
#        myIAU = CU.IAU2006CIO(EOPFile=eopFile, XYsFile=xysFile, LeapSecFile=lsFile)
#        inputs['myIAU'] = myIAU
#        
#        # Retrieve input elements, convert to radians
#        ra = float(x_in[0])
#        rp = float(x_in[1])
#        i = float(x_in[2]) * pi/180
#        LTAN_h = float(x_in[3][0])
#        LTAN_m = float(x_in[3][1])
#        LTAN_s = float(x_in[3][2])
#        w = float(x_in[4]) * pi/180
#        site_lat = float(x_in[5]) * pi/180
#        site_lon = float(x_in[6]) * pi/180
#        t0 = x_in[7]
#        
#        # Compute semi-major axis and eccentricity
#        a = (ra + rp)/2.
#        e = 1 - (rp/a)
#        
#        # Compute mean motion and angular momentum
#        n = np.sqrt(GM/a**3)
#        p = a*(1 - e**2.)
#        h = np.sqrt(GM*p)
#        
##        print 'ra', ra
##        print 'rp', rp
##        print 'a', a
##        print 'e', e
##        print 'p', p
##        print 'n', n
##        print 'h', h
#        
#        # Compute curren JED_JD
#        JED_JD = UTC_G_2_JED_JD(t0)
##        print 'JED_JD', JED_JD
#        
#        # Compute LTAN in rad
#        LTAN = LTAN_h*(pi/12.) + LTAN_m*(pi/720.) + LTAN_s*(pi/43200)
##        print 'LTAN', LTAN
#        
#        # Compute current sun right ascension
#        sun = ephem.Sun()
#        sun.compute((t0[0], t0[1], t0[2], t0[3], t0[4], t0[5]))
#        sun_ra = sun.g_ra
#        
##        print 'sun_ra', float(sun_ra)
#        
#        # Compute RAAN (sun LTAN = pi)
#        RAAN = sun_ra + (LTAN - pi)
##        print 'RAAN', RAAN
#        
#        # Compute geodetic longitude of ascending node
#        f = -w  # theta = 0 at ascending node
#        Erad = 2*atan(np.sqrt((1-e)/(1+e))*tan(f/2))
#        r = a*(1 - e*cos(Erad))
#        node_eci = np.array([[r*cos(RAAN)], [r*sin(RAAN)], [0.]])
#        node_ecef = eci2ecef(node_eci, inputs, JED_JD)
#        node_lat, node_lon, node_ht = ecef2latlonht(node_ecef)
#        node_lat *= pi/180.
#        node_lon *= pi/180.
#        
##        print 'node_lat', node_lat
##        print 'node_lon', node_lon
#        
#        # Find Mo corresponding to lat/lon
#        delta_lon = site_lon - node_lon
#        delta_lat = site_lat - node_lat
#        theta_site = acos(cos(delta_lon)*cos(delta_lat))
#        f_site = theta_site - w
#        E_site = 2*atan(np.sqrt((1-e)/(1+e))*tan(f_site/2))
#        Mo = E_site - e*sin(E_site)
#        
##        print 'theta_site', theta_site
##        print 'f_site', f_site
##        print 'Mo', Mo

        
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