import numpy as np
from math import pi, sin, cos, tan, fmod, fabs, atan, atan2, acos, asin
from math import sinh, cosh, tanh, atanh
from datetime import datetime
import os
import sys
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from utilities.constants import GME, J2E, Re, wE
from utilities.time_systems import utcdt2ttjd, jd2cent
from utilities.ephemeris import compute_sun_coords
from utilities.eop_functions import get_celestrak_eop_alldata, get_eop_data


###############################################################################
# Compute Orbit Parameters
###############################################################################


def meanmot2sma(n, GM=GME):
    '''
    This function computs the semi-major axis given mean motion.
    
    Parameters
    ------
    n : float
        mean motion [rad/s]
    GM : float, optional
        gravitational parameter, default is earth GME [km^3/s^2]
    
    Returns
    ------
    a : float
        semi-major axis [km]
    '''
    
    a = (GM/n**2.)**(1./3.)    
    
    return a


def sma2meanmot(a, GM=GME):
    '''
    This function computs the mean motion given semi-major axis.
    
    Parameters
    ------
    a : float
        semi-major axis [km]    
    GM : float, optional
        gravitational parameter, default is earth GME [km^3/s^2]
    
    Returns
    ------
    n : float
        mean motion [rad/s]
    '''
    
    n = np.sqrt(GM/a**3.)
    
    return n


def smaecc2semilatusrectum(a, e):
    '''
    This function computs the mean motion given semi-major axis.
    
    Parameters
    ------
    a : float
        semi-major axis [km]    
    e : float
        eccenticity
    
    Returns
    ------
    p : float
        semi-latus rectum
    '''
    
    p = a*(1. - e**2.)
    
    return p


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

    # Ensure M is between 0 and 2*pi
    M = fmod(M, 2*pi)
    if M < 0:
        M += 2*pi

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


def sunsynch_inclination(a, e):
    '''
    This function computes the inclination required for a sunsynchronous Earth 
    orbit at the given altitude.
    
    Parameters
    ------
    a : float
        semi-major axis [km]
    e : float
        eccentricity
        
    Returns
    ------
    i : float
        inclination [deg]
    
    Reference
    ------
    Vallado, D. "Fundamentals of Astrodynamics and Applications (4th Ed.)"
        Section 11.4.1
    
    
    '''
    
    # Sun-synch condition
    dRAAN = 360./365.2421897 * pi/180. * 1./86400.  # rad/sec
    
    # Compute required inclination
    n = sma2meanmot(a)
    p = smaecc2semilatusrectum(a, e)
    cosi = -(dRAAN*2.*p**2.)/(3.*n*Re**2.*J2E)  # Eq. 9-37
    i = acos(cosi) * 180./pi  # deg    
    
    return i


def RAAN_to_LTAN(RAAN, UTC, EOP_data):
    '''
    This function computes the Local Time of Ascending Node for a 
    sunsynchronous orbit.
    
    Parameters
    ------
    RAAN : float
        Right Ascension of Ascending Node [deg]
    UTC : datetime object
        time in UTC
    EOP_data : dictionary
        EOP data for the given time including pole coordinates and offsets,
        time offsets, and length of day 
        
    Returns
    ------
    LTAN : float
        local time of ascending node, decimal hour in range [0, 24) 
    '''
    
    # Compute TT in JD format
    TT_JD = utcdt2ttjd(UTC, EOP_data['TAI_UTC'])
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = jd2cent(TT_JD)
    
    # Compute apparent right ascension of the sun
    sun_eci_geom, sun_eci_app = compute_sun_coords(TT_cent)
    sun_ra = atan2(sun_eci_app[1], sun_eci_app[0]) * 180./pi     # deg
    
    # Compute LTAN in decimal hours
    LTAN = ((RAAN - sun_ra)/15. + 12.) % 24.    # decimal hours    
    
    return LTAN


def LTAN_to_RAAN(LTAN, UTC, EOP_data):
    '''
    This function computes the Right Ascension of the Ascending Node for a
    sunsynchronous orbit given LTAN.
    
    Parameters
    ------
    LTAN : float
        local time of ascending node, decimal hour in range [0, 24)     
    UTC : datetime object
        time in UTC
    EOP_data : dictionary
        EOP data for the given time including pole coordinates and offsets,
        time offsets, and length of day 
        
    Returns
    ------
    RAAN : float
        Right Ascension of Ascending Node [deg]
        
    '''
    
    # Compute TT in JD format
    TT_JD = utcdt2ttjd(UTC, EOP_data['TAI_UTC'])
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = jd2cent(TT_JD)
    
    # Compute apparent right ascension of the sun
    sun_eci_geom, sun_eci_app = compute_sun_coords(TT_cent)
    sun_ra = atan2(sun_eci_app[1], sun_eci_app[0]) * 180./pi     # deg
    
    # Compute RAAN in degrees
    RAAN = ((LTAN - 12.)*15. + sun_ra) % 360.      
    
    return RAAN



def compute_vc(r, GM=GME):
    '''
    This function computes the circular orbital velocity for a given orbit
    radius and gravitational parameter.
    
    Parameters
    ------
    r : float
        orbit radius [km]
    GM : float, optional
        gravitational parameter (default=GME) [km^3/s^2]
        
    Returns
    ------
    vc : float
        circular orbit velocity [km/s]
    '''
    
    vc = np.sqrt(GM/r)    
    
    return vc


def compute_visviva(r, a, GM=GME):
    '''
    This function computes the orbital velocity for a given orbit radius,
    semi-major axis and gravitational parameter, using Vis-Viva.
    
    Parameters
    ------
    r : float
        orbit radius [km]
    a : float
        orbit semi-major axis [km]
    GM : float, optional
        gravitational parameter (default=GME) [km^3/s^2]
        
    Returns
    ------
    v : float
        orbit velocity [km/s]
    '''
    
    v = np.sqrt(2*GM/r - GM/a)    
    
    return v










############################################################################
# Orbit Element Conversions
############################################################################




def mean2osc(mean_elem):
    '''
    This function converts mean Keplerian elements to osculating Keplerian
    elements using Brouwer-Lyddane Theory.
    
    Parameters
    ------
    mean_elem : list
        Mean Keplerian orbital elements [km, deg]
        [a,e,i,RAAN,w,M]
    
    Returns
    ------
    osc_elem : list
        Osculating Keplerian orbital elements [km, deg]
        [a,e,i,RAAN,w,M]
    
    References
    ------
    [1] Schaub, H. and Junkins, J.L., Analytical Mechanics of Space Systems."
        2nd ed., 2009.
    '''
    
    # Retrieve input elements, convert to radians
    a0 = float(mean_elem[0])
    e0 = float(mean_elem[1])
    i0 = float(mean_elem[2]) * pi/180
    RAAN0 = float(mean_elem[3]) * pi/180
    w0 = float(mean_elem[4]) * pi/180
    M0 = float(mean_elem[5]) * pi/180
    
    # Compute gamma parameter
    gamma0 = (J2E/2.) * (Re/a0)**2.
    
    # Compute first order Brouwer-Lyddane transformation
    a1,e1,i1,RAAN1,w1,M1 = brouwer_lyddane(a0,e0,i0,RAAN0,w0,M0,gamma0)
    
    # Convert angles to degree for output
    i1 *= 180./pi
    RAAN1 *= 180./pi
    w1 *= 180./pi
    M1 *= 180./pi
    
    osc_elem = [a1,e1,i1,RAAN1,w1,M1]
    
    return osc_elem


def osc2mean(osc_elem):
    '''
    This function converts osculating Keplerian elements to mean Keplerian
    elements using Brouwer-Lyddane Theory.
    
    Parameters
    ------
    elem0 : list
        Osculating Keplerian orbital elements [km, deg]
        [a,e,i,RAAN,w,M]
    
    Returns
    ------
    elem1 : list
        Mean Keplerian orbital elements [km, deg]
        [a,e,i,RAAN,w,M]
    
    References
    ------
    [1] Schaub, H. and Junkins, J.L., Analytical Mechanics of Space Systems."
        2nd ed., 2009.
    '''
    
    # Retrieve input elements, convert to radians
    a0 = float(osc_elem[0])
    e0 = float(osc_elem[1])
    i0 = float(osc_elem[2]) * pi/180
    RAAN0 = float(osc_elem[3]) * pi/180
    w0 = float(osc_elem[4]) * pi/180
    M0 = float(osc_elem[5]) * pi/180
    
    # Compute gamma parameter
    gamma0 = -(J2E/2.) * (Re/a0)**2.
    
    # Compute first order Brouwer-Lyddane transformation
    a1,e1,i1,RAAN1,w1,M1 = brouwer_lyddane(a0,e0,i0,RAAN0,w0,M0,gamma0)
    
    # Convert angles to degree for output
    i1 *= 180./pi
    RAAN1 *= 180./pi
    w1 *= 180./pi
    M1 *= 180./pi
    
    mean_elem = [a1,e1,i1,RAAN1,w1,M1]    
    
    return mean_elem


def osc2perifocal(elem):
    '''
    This function computes position coordinates in the perifocal frame given
    osculating Keplerian elements.
    
    Parameters
    ------
    elem : 6x1 numpy array
        osculating Keplerian orbital elements
    
    Returns
    ------
    x : float
        perifocal frame x-coordinate
    y : float
        perifocal frame y-coordinate
    '''
    
    # Retrieve orbit parameters
    a = float(elem[0])
    e = float(elem[1])
    M = float(elem[5])*pi/180.
    
    # Compute true anomaly
    E = mean2ecc(M, e)
    f = ecc2true(E, e)
    
    # Compute semi-latus rectum
    p = smaecc2semilatusrectum(a, e)
    
    # Compute orbit radius
    r = p/(1 + e*cos(f))
    
    # Compute cartesian coordinates
    x = r*cos(f)
    y = r*sin(f)
    
    
    return x, y


def brouwer_lyddane(a0,e0,i0,RAAN0,w0,M0,gamma0):
    '''
    This function converts between osculating and mean Keplerian elements
    using Brouwer-Lyddane Theory. The input gamma value determines whether 
    the transformation is from osculating to mean elements or vice versa.
    The same calculations are performed in either case.
    
    Parameters
    ------
    a0 : float
        semi-major axis [km]
    e0 : float
        eccentricity
    i0 : float
        inclination [rad]
    RAAN0 : float
        right ascension of ascending node [rad]
    w0 : float 
        argument of periapsis [rad]
    M0 : float
        mean anomaly [rad]
    gamma0 : float
        intermediate calculation parameter        
    
    Returns 
    ------
    a1 : float
        semi-major axis [km]
    e1 : float
        eccentricity
    i1 : float
        inclination [rad]
    RAAN1 : float
        right ascension of ascending node [rad]
    w1 : float 
        argument of periapsis [rad]
    M1 : float
        mean anomaly [rad]
    
    References
    ------
    [1] Schaub, H. and Junkins, J.L., Analytical Mechanics of Space Systems."
        2nd ed., 2009.
    
    '''
    
    # Compute transformation parameters
    eta = np.sqrt(1. - e0**2.)
    gamma1 = gamma0/eta**4.
    
    # Compute true anomaly
    E0 = mean2ecc(M0, e0)
    f0 = ecc2true(E0, e0)
    
    # Compute intermediate terms
    a_r = (1. + e0*cos(f0))/eta**2.
    
    de1 = (gamma1/8.)*e0*eta**2.*(1. - 11.*cos(i0)**2. - 40.*((cos(i0)**4.) / 
                                  (1.-5.*cos(i0)**2.)))*cos(2.*w0)
    
    de = de1 + (eta**2./2.) * \
        (gamma0*((3.*cos(i0)**2. - 1.)/(eta**6.) * 
                 (e0*eta + e0/(1.+eta) + 3.*cos(f0) + 3.*e0*cos(f0)**2. + e0**2.*cos(f0)**3.) + 
              3.*(1.-cos(i0)**2.)/eta**6.*(e0 + 3.*cos(f0) + 3.*e0*cos(f0)**2. + e0**2.*cos(f0)**3.) * cos(2.*w0 + 2.*f0))
                - gamma1*(1.-cos(i0)**2.)*(3.*cos(2*w0 + f0) + cos(2.*w0 + 3.*f0)))

    di = -(e0*de1/(eta**2.*tan(i0))) + (gamma1/2.)*cos(i0)*np.sqrt(1.-cos(i0)**2.) * \
          (3.*cos(2*w0 + 2.*f0) + 3.*e0*cos(2.*w0 + f0) + e0*cos(2.*w0 + 3.*f0))
          
    MwRAAN1 = M0 + w0 + RAAN0 + (gamma1/8.)*eta**3. * \
              (1. - 11.*cos(i0)**2. - 40.*((cos(i0)**4.)/(1.-5.*cos(i0)**2.))) - (gamma1/16.) * \
              (2. + e0**2. - 11.*(2.+3.*e0**2.)*cos(i0)**2. 
               - 40.*(2.+5.*e0**2.)*((cos(i0)**4.)/(1.-5.*cos(i0)**2.))  
               - 400.*e0**2.*(cos(i0)**6.)/((1.-5.*cos(i0)**2.)**2.)) + (gamma1/4.) * \
              (-6.*(1.-5.*cos(i0)**2.)*(f0 - M0 + e0*sin(f0)) 
               + (3.-5.*cos(i0)**2.)*(3.*sin(2.*w0 + 2.*f0) + 3.*e0*sin(2.*w0 + f0) + e0*sin(2.*w0 + 3.*f0))) \
               - (gamma1/8.)*e0**2.*cos(i0) * \
              (11. + 80.*(cos(i0)**2.)/(1.-5.*cos(i0)**2.) + 200.*(cos(i0)**4.)/((1.-5.*cos(i0)**2.)**2.)) \
               - (gamma1/2.)*cos(i0) * \
              (6.*(f0 - M0 + e0*sin(f0)) - 3.*sin(2.*w0 + 2.*f0) - 3.*e0*sin(2.*w0 + f0) - e0*sin(2.*w0 + 3.*f0))
               
    edM = (gamma1/8.)*e0*eta**3. * \
          (1. - 11.*cos(i0)**2. - 40.*((cos(i0)**4.)/(1.-5.*cos(i0)**2.))) - (gamma1/4.)*eta**3. * \
          (2.*(3.*cos(i0)**2. - 1.)*((a_r*eta)**2. + a_r + 1.)*sin(f0) + 
           3.*(1. - cos(i0)**2.)*((-(a_r*eta)**2. - a_r + 1.)*sin(2.*w0 + f0) +
           ((a_r*eta)**2. + a_r + (1./3.))*sin(2*w0 + 3.*f0)))
          
    dRAAN = -(gamma1/8.)*e0**2.*cos(i0) * \
             (11. + 80.*(cos(i0)**2.)/(1.-5.*cos(i0)**2.) + 
              200.*(cos(i0)**4.)/((1.-5.*cos(i0)**2.)**2.)) - (gamma1/2.)*cos(i0) * \
             (6.*(f0 - M0 + e0*sin(f0)) - 3.*sin(2.*w0 + 2.*f0) - 
              3.*e0*sin(2.*w0 + f0) - e0*sin(2.*w0 + 3.*f0))  

    d1 = (e0 + de)*sin(M0) + edM*cos(M0)
    d2 = (e0 + de)*cos(M0) - edM*sin(M0)
    d3 = (sin(i0/2.) + cos(i0/2.)*(di/2.))*sin(RAAN0) + sin(i0/2.)*dRAAN*cos(RAAN0)
    d4 = (sin(i0/2.) + cos(i0/2.)*(di/2.))*cos(RAAN0) - sin(i0/2.)*dRAAN*sin(RAAN0)
    
    # Compute transformed elements
    a1 = a0 + a0*gamma0*((3.*cos(i0)**2. - 1.)*(a_r**3. - (1./eta)**3.) + 
                         (3.*(1.-cos(i0)**2.)*a_r**3.*cos(2.*w0 + 2.*f0)))
    
    e1 = np.sqrt(d1**2. + d2**2.)
    
    i1 = 2.*asin(np.sqrt(d3**2. + d4**2.))
    
    RAAN1 = atan2(d3, d4)
    
    M1 = atan2(d1, d2)
    
    w1 = MwRAAN1 - RAAN1 - M1
    
    while w1 > 2.*pi:
        w1 -= 2.*pi
        
    while w1 < 0.:
        w1 += 2.*pi
                             
    
    return a1, e1, i1, RAAN1, w1, M1


def cart2kep(cart, GM=GME):
    '''
    This function converts a Cartesian state vector in inertial frame to
    Keplerian orbital elements.
    
    Parameters
    ------
    cart : 6x1 numpy array
    
    Cartesian Coordinates (Inertial Frame)
    ------
    cart[0] : x
      Position in x               [km]
    cart[1] : y
      Position in y               [km]
    cart[2] : z
      Position in z               [km]
    cart[3] : dx
      Velocity in x               [km/s]
    cart[4] : dy
      Velocity in y               [km/s]
    cart[5] : dz
      Velocity in z               [km/s]
      
    Returns
    ------
    elem : 6x1 numpy array
    
    Keplerian Orbital Elements
    ------
    elem[0] : a
      Semi-Major Axis             [km]
    elem[1] : e
      Eccentricity                [unitless]
    elem[2] : i
      Inclination                 [deg]
    elem[3] : RAAN
      Right Asc Ascending Node    [deg]
    elem[4] : w
      Argument of Periapsis       [deg]
    elem[5] : theta
      True Anomaly                [deg]    
      
    '''
    
    # Retrieve input cartesian coordinates
    r_vect = cart[0:3].reshape(3,1)
    v_vect = cart[3:6].reshape(3,1)

    # Calculate orbit parameters
    r = np.linalg.norm(r_vect)
    ir_vect = r_vect/r
    v2 = np.linalg.norm(v_vect)**2
    h_vect = np.cross(r_vect, v_vect, axis=0)
    h = np.linalg.norm(h_vect)

    # Calculate semi-major axis
    a = 1./(2./r - v2/GM)     # km
    
    # Calculate eccentricity
    e_vect = np.cross(v_vect, h_vect, axis=0)/GM - ir_vect
    e = np.linalg.norm(e_vect)

    # Calculate RAAN and inclination
    ih_vect = h_vect/h
    RAAN = atan2(ih_vect[0], -ih_vect[1])   # rad
    i = acos(ih_vect[2])   # rad
    if RAAN < 0.:
        RAAN += 2.*pi

    # Apply correction for circular orbit, choose e_vect to point
    # to ascending node
    if e != 0:
        ie_vect = e_vect/e
    else:
        ie_vect = np.array([[cos(RAAN)], [sin(RAAN)], [0.]])

    # Find orthogonal unit vector to complete perifocal frame
    ip_vect = np.cross(ih_vect, ie_vect, axis=0)

    # Form rotation matrix PN
    PN = np.concatenate((ie_vect, ip_vect, ih_vect), axis=1).T

    # Calculate argument of periapsis
    w = atan2(PN[0,2], PN[1,2])  # rad
    if w < 0.:
        w += 2.*pi

    # Calculate true anomaly
    cross1 = np.cross(ie_vect, ir_vect, axis=0)
    tan1 = np.dot(cross1.T, ih_vect)
    tan2 = np.dot(ie_vect.T, ir_vect)
    theta = atan2(tan1, tan2)    # rad
    if theta < 0.:
        theta += 2.*pi
    
    # Convert angles to deg
    i *= 180./pi
    RAAN *= 180./pi
    w *= 180./pi
    theta *= 180./pi
    
    # Form output
    elem = np.array([[a], [e], [i], [RAAN], [w], [theta]])
      
    return elem


def kep2cart(elem, GM=GME):
    '''
    This function converts a vector of Keplerian orbital elements to a
    Cartesian state vector in inertial frame.
    
    Parameters
    ------
    elem : 6x1 numpy array
    
    Keplerian Orbital Elements
    ------
    elem[0] : a
      Semi-Major Axis             [km]
    elem[1] : e
      Eccentricity                [unitless]
    elem[2] : i
      Inclination                 [deg]
    elem[3] : RAAN
      Right Asc Ascending Node    [deg]
    elem[4] : w
      Argument of Periapsis       [deg]
    elem[5] : theta
      True Anomaly                [deg]
      
      
    Returns
    ------
    cart : 6x1 numpy array
    
    Cartesian Coordinates (Inertial Frame)
    ------
    cart[0] : x
      Position in x               [km]
    cart[1] : y
      Position in y               [km]
    cart[2] : z
      Position in z               [km]
    cart[3] : dx
      Velocity in x               [km/s]
    cart[4] : dy
      Velocity in y               [km/s]
    cart[5] : dz
      Velocity in z               [km/s]  
      
    '''
    
    # Retrieve input elements, convert to radians
    a = float(elem[0])
    e = float(elem[1])
    i = float(elem[2]) * pi/180
    RAAN = float(elem[3]) * pi/180
    w = float(elem[4]) * pi/180
    theta = float(elem[5]) * pi/180

    # Calculate h and r
    p = a*(1 - e**2)
    h = np.sqrt(GM*p)
    r = p/(1. + e*cos(theta))

    # Calculate r_vect and v_vect
    r_vect = r * \
        np.array([[cos(RAAN)*cos(theta+w) - sin(RAAN)*sin(theta+w)*cos(i)],
                  [sin(RAAN)*cos(theta+w) + cos(RAAN)*sin(theta+w)*cos(i)],
                  [sin(theta+w)*sin(i)]])

    vv1 = cos(RAAN)*(sin(theta+w) + e*sin(w)) + \
          sin(RAAN)*(cos(theta+w) + e*cos(w))*cos(i)

    vv2 = sin(RAAN)*(sin(theta+w) + e*sin(w)) - \
          cos(RAAN)*(cos(theta+w) + e*cos(w))*cos(i)

    vv3 = -(cos(theta+w) + e*cos(w))*sin(i)
    
    v_vect = -GM/h * np.array([[vv1], [vv2], [vv3]])

    cart = np.concatenate([r_vect, v_vect])
    
    return cart


def element_conversion(x_in, iflag, oflag, GM=GME, dt=0.):
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
      input flag (0 = orbital elements, 1 = cartesian coordiantes)
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
        while RAAN < 0.:
            RAAN += 2.*pi

        # Calculate eccentricity
        e_vect = np.cross(v_vect, h_vect, axis=0)/GM - ir_vect
        e = np.linalg.norm(e_vect)
        
        k = np.array([[0.], [0.], [1.]])
        N = np.cross(k, h_vect, axis=0)

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
        while w < 0.:
            w += 2.*pi

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
        
    else:
        print('Error: Invalid Input Flag!')

    # Solve for M(t) = Mo + n*dt
    M = Mo + n*dt   # rad
    
    while M < 0:
        M += 2*pi
    while M > 2*pi:
        M -= 2*pi

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


if __name__ == '__main__':
    
    RAAN = 0.
    UTC = datetime(2000, 1, 1, 12, 0, 0)
    eop_alldata = get_celestrak_eop_alldata()
    EOP_data = get_eop_data(eop_alldata, UTC)
    
    
    LTAN = RAAN_to_LTAN(RAAN, UTC, EOP_data)
    print(LTAN)
    
    
###############################################################################
# Orbit Transfers
###############################################################################

def compute_launch_velocity(lat_rad, R=Re, w=wE):
    '''
    This function computes the launch velocity component contributed by the 
    planet's rotation for a given latitude.
    
    Parameters
    ------
    lat_rad : float
        geodetic latitude [radians]
    R : float, optional
        planet radius (default=Re) [km]
    w : float, optional
        planet rotational velocity (default=wE) [rad/s]
        
    Returns
    ------
    v0 : float
        velocity magnitude [km/s]
    
    '''
    
    v0 = R*w*cos(lat_rad)
    
    return v0    