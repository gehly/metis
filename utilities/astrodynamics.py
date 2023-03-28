import numpy as np
import math
import os
import sys
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from utilities import time_systems as timesys
from utilities import ephemeris as eph
from utilities.constants import GME, J2E, Re, wE


###############################################################################
# Compute Orbit Parameters
###############################################################################


def meanmot2sma(n, GM=GME):
    '''
    This function computes the semi-major axis given mean motion.
    
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
    This function computes the mean motion given semi-major axis.
    
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


def period2sma(P, GM=GME):
    '''
    This function computes the semi-major axis given orbit period.
    
    Parameters
    ------
    P : float
        orbit period [sec]
    GM : float, optional
        gravitational parameter, default is earth GME [km^3/s^2]
    
    Returns
    ------
    a : float
        semi-major axis [km]
    '''
    
    n = 2.*np.pi/P
    a = (GM/n**2.)**(1./3.)    
    
    return a


def sma2period(a, GM=GME):
    '''
    This function computes the orbit period given semi-major axis.
    
    Parameters
    ------
    a : float
        semi-major axis [km]    
    GM : float, optional
        gravitational parameter, default is earth GME [km^3/s^2]
    
    Returns
    ------
    P : float
        orbit period [sec]
    '''
    
    P = 2.*np.pi*np.sqrt(a**3./GM)
    
    return P


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
    M = math.fmod(M, 2*math.pi)
    if M < 0:
        M += 2*math.pi

    # Starting guess for E
    E = M + e*math.sin(M)/(1 - math.sin(M + e) + math.sin(M))

    # Initialize loop variable
    f = 1
    tol = 1e-8

    # Iterate using Newton-Raphson Method
    while math.fabs(f) > tol:
        f = E - e*math.sin(E) - M
        df = 1 - e*math.cos(E)
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
    
    M = E - e*math.sin(E)
    
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

    f = 2*math.atan(np.sqrt((1+e)/(1-e))*math.tan(E/2))

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

    E = 2*math.atan(np.sqrt((1-e)/(1+e))*math.tan(f/2))

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

#    # Ensure M is between -pi and pi
#    if M > pi or M < -pi:
#        print('Error: Expected -pi < M < pi!')
#        print('M', M)
#        
#    if M > pi:
#        M -= 2.*pi
#        
#    if M < -pi:
#        M += 2.*pi

    # Form starting guess for H
    H = M

    # Initialize loop variable
    f = 1
    tol = 1e-8

    # Iterate using Newton-Raphson Method
    while math.fabs(f) > tol:
        f = e*math.sinh(H) - H - M
        df = e*math.cosh(H) - 1
        H = H - f/df

    return H


def hyp2mean(H, e):
    '''
    This function converts from Hyperbolic Anomaly to Mean Anomaly

    Parameters
    ------
    H : float
      hyperbolic anomaly [rad]
    e : float
      eccentricity

    Returns
    ------
    M : float
      mean anomaly [rad]
    
    '''
    
    M = e*math.sinh(H) - H
    
    return M


def true2hyp(f, e):
    '''
    This function converts from True Anomaly to Hyperbolic Anomaly

    Parameters
    ------
    f : float
      true anomaly [rad]
    e : float
      eccentricity

    Returns
    ------
    H : float
      hyperbolic anomaly [rad]
    '''
    
    H = 2*math.atanh(np.sqrt((e-1)/(e+1))*math.tan(f/2))
    
    return H


def hyp2true(H, e):
    '''
    This function converts from Hyperbolic Anomaly to True Anomaly

    Parameters
    ------
    H : float
      hyperbolic anomaly [rad]
    e : float
      eccentricity

    Returns
    ------    
    f : float
      true anomaly [rad]
    '''
    
    f = 2*math.atan(np.sqrt((e+1)/(e-1))*math.tanh(H/2))
    
    return f


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
    dRAAN = 360./365.2421897 * math.pi/180. * 1./86400.  # rad/sec
    
    # Compute required inclination
    n = sma2meanmot(a)
    p = smaecc2semilatusrectum(a, e)
    cosi = -(dRAAN*2.*p**2.)/(3.*n*Re**2.*J2E)  # Eq. 9-37
    i = math.acos(cosi) * 180./math.pi  # deg
    
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
    TT_JD = timesys.utcdt2ttjd(UTC, EOP_data['TAI_UTC'])
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = timesys.jd2cent(TT_JD)
    
    # Compute apparent right ascension of the sun
    sun_eci_geom, sun_eci_app = eph.compute_sun_coords(TT_cent)
    sun_ra = math.atan2(sun_eci_app[1], sun_eci_app[0]) * 180./math.pi     # deg
    
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
    TT_JD = timesys.utcdt2ttjd(UTC, EOP_data['TAI_UTC'])
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = timesys.jd2cent(TT_JD)
    
    # Compute apparent right ascension of the sun
    sun_eci_geom, sun_eci_app = eph.compute_sun_coords(TT_cent)
    sun_ra = math.atan2(sun_eci_app[1], sun_eci_app[0]) * 180./math.pi     # deg
    
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


def compute_rp(r_vect, v_vect, GM):
    '''
    This function computes the radius of periapsis for a given position and
    velocity vector.
    
    Parameters
    ------
    r_vect : 3x1 numpy array
        inertial position vector [km]
    v_vect  : 3x1 numpy array
        inertial velocity vector [km/s]
    GM : float
        gravitational parameter of central body [km^3/s^2]
        
    Returns
    ------
    rp : float
        radius of periapsis [km]
    
    '''
    r_vect = np.reshape(r_vect, (3,1))
    v_vect = np.reshape(v_vect, (3,1))
    r = np.linalg.norm(r_vect)
    v = np.linalg.norm(v_vect)
    
    # Semi-major axis
    a = 1./(2./r - v**2./GM)
    
    # Eccentricity vector 
    h_vect = np.cross(r_vect, v_vect, axis=0)
    cross1 = np.cross(v_vect, h_vect, axis=0)

    e_vect = cross1/GM - r_vect/r
    e = np.linalg.norm(e_vect)
    
    rp = a*(1. - e)
    
    return rp


def compute_p(r_vect, v_vect, GM):
    '''
    This function computes the semi-latus rectum for a given position and
    velocity vector.
    
    Parameters
    ------
    r_vect : 3x1 numpy array
        inertial position vector [km]
    v_vect  : 3x1 numpy array
        inertial velocity vector [km/s]
    GM : float
        gravitational parameter of central body [km^3/s^2]
        
    Returns
    ------
    p : float
        semi-latus rectum [km]
    
    '''
    r_vect = np.reshape(r_vect, (3,1))
    v_vect = np.reshape(v_vect, (3,1))

    h_vect = np.cross(r_vect, v_vect, axis=0)
    h = np.linalg.norm(h_vect)
    
    p = h**2./GM
    
    return p



###############################################################################
# Physical Models
###############################################################################


def atmosphere_lookup(h):
    '''
    This function acts as a lookup table for atmospheric density reference
    values, reference heights, and scale heights for a range of different 
    altitudes from 100 - 1000+ km.  Values from Vallado 4th ed. Table 8-4.
    
    Parameters
    ------
    h : float
        altitude [km]
    
    Returns
    ------
    rho0 : float
        reference density [kg/km^3]
    h0 : float
        reference altitude [km]
    H : float
        scale height [km]

    '''
    
    if h <= 100:
        # Assume at this height we have re-entered atmosphere
        rho0 = 0
        h0 = 1
        H = 1
    elif h < 110:
        rho0 = 5.297e-7 * 1e9  # kg/km^3
        h0 = 100.    # km
        H = 5.877    # km    
    elif h < 120:
        rho0 = 9.661e-8 * 1e9  # kg/km^3
        h0 = 110.    # km
        H = 7.263    # km   
    elif h < 130:
        rho0 = 2.438e-8 * 1e9  # kg/km^3
        h0 = 120.    # km
        H = 9.473    # km   
    elif h < 140: 
        rho0 = 8.484e-9 * 1e9  # kg/km^3
        h0 = 130.    # km
        H = 12.636   # km       
    elif h < 150:
        rho0 = 3.845e-9 * 1e9  # kg/km^3
        h0 = 140.    # km
        H = 16.149   # km       
    elif h < 180:
        rho0 = 2.070e-9 * 1e9  # kg/km^3
        h0 = 150.    # km
        H = 22.523   # km       
    elif h < 200:
        rho0 = 5.464e-10 * 1e9  # kg/km^3
        h0 = 180.    # km
        H = 29.740   # km     
    elif h < 250:
        rho0 = 2.789e-10 * 1e9  # kg/km^3
        h0 = 200.    # km
        H = 37.105   # km   
    elif h < 300:
        rho0 = 7.248e-11 * 1e9  # kg/km^3
        h0 = 250.    # km
        H = 45.546   # km       
    elif h < 350:
        rho0 = 2.418e-11 * 1e9  # kg/km^3
        h0 = 300.    # km
        H = 53.628   # km       
    elif h < 400:
        rho0 = 9.518e-12 * 1e9  # kg/km^3
        h0 = 350.    # km
        H = 53.298   # km       
    elif h < 450:
        rho0 = 3.725e-12 * 1e9   # kg/km^3
        h0 = 400.    # km
        H = 58.515   # km     
    elif h < 500:
        rho0 = 1.585e-12 * 1e9   # kg/km^3
        h0 = 450.    # km
        H = 60.828   # km   
    elif h < 600:
        rho0 = 6.967e-13 * 1e9   # kg/km^3
        h0 = 500.    # km
        H = 63.822   # km
    elif h < 700:
        rho0 = 1.454e-13 * 1e9   # kg/km^3
        h0 = 600.    # km
        H = 71.835   # km
    elif h < 800:
        rho0 = 3.614e-14 * 1e9   # kg/km^3
        h0 = 700.    # km
        H = 88.667   # km       
    elif h < 900:
        rho0 = 1.17e-14 * 1e9    # kg/km^3
        h0 = 800.    # km
        H = 124.64   # km       
    elif h < 1000:
        rho0 = 5.245e-15 * 1e9   # kg/km^3
        h0 = 900.    # km
        H = 181.05   # km       
    else:
        rho0 = 3.019e-15 * 1e9   # kg/km^3
        h0 = 1000.   # km
        H = 268.00   # km
    
    
    return rho0, h0, H





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
    i0 = float(mean_elem[2]) * math.pi/180
    RAAN0 = float(mean_elem[3]) * math.pi/180
    w0 = float(mean_elem[4]) * math.pi/180
    M0 = float(mean_elem[5]) * math.pi/180
    
    # Compute gamma parameter
    gamma0 = (J2E/2.) * (Re/a0)**2.
    
    # Compute first order Brouwer-Lyddane transformation
    a1,e1,i1,RAAN1,w1,M1 = brouwer_lyddane(a0,e0,i0,RAAN0,w0,M0,gamma0)
    
    # Convert angles to degree for output
    i1 *= 180./math.pi
    RAAN1 *= 180./math.pi
    w1 *= 180./math.pi
    M1 *= 180./math.pi
    
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
    i0 = float(osc_elem[2]) * math.pi/180
    RAAN0 = float(osc_elem[3]) * math.pi/180
    w0 = float(osc_elem[4]) * math.pi/180
    M0 = float(osc_elem[5]) * math.pi/180
    
    # Compute gamma parameter
    gamma0 = -(J2E/2.) * (Re/a0)**2.
    
    # Compute first order Brouwer-Lyddane transformation
    a1,e1,i1,RAAN1,w1,M1 = brouwer_lyddane(a0,e0,i0,RAAN0,w0,M0,gamma0)
    
    # Convert angles to degree for output
    i1 *= 180./math.pi
    RAAN1 *= 180./math.pi
    w1 *= 180./math.pi
    M1 *= 180./math.pi
    
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
    M = float(elem[5])*math.pi/180.
    
    # Compute true anomaly
    E = mean2ecc(M, e)
    f = ecc2true(E, e)
    
    # Compute semi-latus rectum
    p = smaecc2semilatusrectum(a, e)
    
    # Compute orbit radius
    r = p/(1 + e*math.cos(f))
    
    # Compute cartesian coordinates
    x = r*math.cos(f)
    y = r*math.sin(f)
    
    
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
    a_r = (1. + e0*math.cos(f0))/eta**2.
    
    de1 = (gamma1/8.)*e0*eta**2.*(1. - 11.*math.cos(i0)**2. - 40.*((math.cos(i0)**4.) /
                                  (1.-5.*math.cos(i0)**2.)))*math.cos(2.*w0)
    
    de = de1 + (eta**2./2.) * \
        (gamma0*((3.*math.cos(i0)**2. - 1.)/(eta**6.) *
                 (e0*eta + e0/(1.+eta) + 3.*math.cos(f0) + 3.*e0*math.cos(f0)**2. + e0**2.*math.cos(f0)**3.) +
              3.*(1.-math.cos(i0)**2.)/eta**6.*(e0 + 3.*math.cos(f0) + 3.*e0*math.cos(f0)**2. + e0**2.*math.cos(f0)**3.) * math.cos(2.*w0 + 2.*f0))
                - gamma1*(1.-math.cos(i0)**2.)*(3.*math.cos(2*w0 + f0) + math.cos(2.*w0 + 3.*f0)))

    di = -(e0*de1/(eta**2.*math.tan(i0))) + (gamma1/2.)*math.cos(i0)*np.sqrt(1.-math.cos(i0)**2.) * \
          (3.*math.cos(2*w0 + 2.*f0) + 3.*e0*math.cos(2.*w0 + f0) + e0*math.cos(2.*w0 + 3.*f0))
          
    MwRAAN1 = M0 + w0 + RAAN0 + (gamma1/8.)*eta**3. * \
              (1. - 11.*math.cos(i0)**2. - 40.*((math.cos(i0)**4.)/(1.-5.*math.cos(i0)**2.))) - (gamma1/16.) * \
              (2. + e0**2. - 11.*(2.+3.*e0**2.)*math.cos(i0)**2.
               - 40.*(2.+5.*e0**2.)*((math.cos(i0)**4.)/(1.-5.*math.cos(i0)**2.))
               - 400.*e0**2.*(math.cos(i0)**6.)/((1.-5.*math.cos(i0)**2.)**2.)) + (gamma1/4.) * \
              (-6.*(1.-5.*math.cos(i0)**2.)*(f0 - M0 + e0*math.sin(f0))
               + (3.-5.*math.cos(i0)**2.)*(3.*math.sin(2.*w0 + 2.*f0) + 3.*e0*math.sin(2.*w0 + f0) + e0*math.sin(2.*w0 + 3.*f0))) \
               - (gamma1/8.)*e0**2.*math.cos(i0) * \
              (11. + 80.*(math.cos(i0)**2.)/(1.-5.*math.cos(i0)**2.) + 200.*(math.cos(i0)**4.)/((1.-5.*math.cos(i0)**2.)**2.)) \
               - (gamma1/2.)*math.cos(i0) * \
              (6.*(f0 - M0 + e0*math.sin(f0)) - 3.*math.sin(2.*w0 + 2.*f0) - 3.*e0*math.sin(2.*w0 + f0) - e0*math.sin(2.*w0 + 3.*f0))
               
    edM = (gamma1/8.)*e0*eta**3. * \
          (1. - 11.*math.cos(i0)**2. - 40.*((math.cos(i0)**4.)/(1.-5.*math.cos(i0)**2.))) - (gamma1/4.)*eta**3. * \
          (2.*(3.*math.cos(i0)**2. - 1.)*((a_r*eta)**2. + a_r + 1.)*math.sin(f0) +
           3.*(1. - math.cos(i0)**2.)*((-(a_r*eta)**2. - a_r + 1.)*math.sin(2.*w0 + f0) +
           ((a_r*eta)**2. + a_r + (1./3.))*math.sin(2*w0 + 3.*f0)))
          
    dRAAN = -(gamma1/8.)*e0**2.*math.cos(i0) * \
             (11. + 80.*(math.cos(i0)**2.)/(1.-5.*math.cos(i0)**2.) +
              200.*(math.cos(i0)**4.)/((1.-5.*math.cos(i0)**2.)**2.)) - (gamma1/2.)*math.cos(i0) * \
             (6.*(f0 - M0 + e0*math.sin(f0)) - 3.*math.sin(2.*w0 + 2.*f0) -
              3.*e0*math.sin(2.*w0 + f0) - e0*math.sin(2.*w0 + 3.*f0))

    d1 = (e0 + de)*math.sin(M0) + edM*math.cos(M0)
    d2 = (e0 + de)*math.cos(M0) - edM*math.sin(M0)
    d3 = (math.sin(i0/2.) + math.cos(i0/2.)*(di/2.))*math.sin(RAAN0) + math.sin(i0/2.)*dRAAN*math.cos(RAAN0)
    d4 = (math.sin(i0/2.) + math.cos(i0/2.)*(di/2.))*math.cos(RAAN0) - math.sin(i0/2.)*dRAAN*math.sin(RAAN0)
    
    # Compute transformed elements
    a1 = a0 + a0*gamma0*((3.*math.cos(i0)**2. - 1.)*(a_r**3. - (1./eta)**3.) +
                         (3.*(1.-math.cos(i0)**2.)*a_r**3.*math.cos(2.*w0 + 2.*f0)))
    
    e1 = np.sqrt(d1**2. + d2**2.)
    
    i1 = 2.*math.asin(np.sqrt(d3**2. + d4**2.))
    
    RAAN1 = math.atan2(d3, d4)
    
    M1 = math.atan2(d1, d2)
    
    w1 = MwRAAN1 - RAAN1 - M1
    
    while w1 > 2.*math.pi:
        w1 -= 2.*math.pi
        
    while w1 < 0.:
        w1 += 2.*math.pi
                             
    
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
    RAAN = math.atan2(ih_vect[0], -ih_vect[1])   # rad
    i = math.acos(ih_vect[2])   # rad
    if RAAN < 0.:
        RAAN += 2.*math.pi

    # Apply correction for circular orbit, choose e_vect to point
    # to ascending node
    if e != 0:
        ie_vect = e_vect/e
    else:
        ie_vect = np.array([[math.cos(RAAN)], [math.sin(RAAN)], [0.]])

    # Find orthogonal unit vector to complete perifocal frame
    ip_vect = np.cross(ih_vect, ie_vect, axis=0)

    # Form rotation matrix PN
    PN = np.concatenate((ie_vect, ip_vect, ih_vect), axis=1).T

    # Calculate argument of periapsis
    w = math.atan2(PN[0,2], PN[1,2])  # rad
    if w < 0.:
        w += 2.*math.pi

    # Calculate true anomaly
    cross1 = np.cross(ie_vect, ir_vect, axis=0)
    tan1 = np.dot(cross1.T, ih_vect)
    tan2 = np.dot(ie_vect.T, ir_vect)
    theta = math.atan2(tan1, tan2)    # rad
    
    # Update range of true anomaly for elliptical orbits
    if a > 0. and theta < 0.:
        theta += 2.*math.pi
    
    # Convert angles to deg
    i *= 180./math.pi
    RAAN *= 180./math.pi
    w *= 180./math.pi
    theta *= 180./math.pi
    
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
    i = float(elem[2]) * math.pi/180
    RAAN = float(elem[3]) * math.pi/180
    w = float(elem[4]) * math.pi/180
    theta = float(elem[5]) * math.pi/180

    # Calculate h and r
    p = a*(1 - e**2)
    h = np.sqrt(GM*p)
    r = p/(1. + e*math.cos(theta))

    # Calculate r_vect and v_vect
    r_vect = r * \
        np.array([[math.cos(RAAN)*math.cos(theta+w) - math.sin(RAAN)*math.sin(theta+w)*math.cos(i)],
                  [math.sin(RAAN)*math.cos(theta+w) + math.cos(RAAN)*math.sin(theta+w)*math.cos(i)],
                  [math.sin(theta+w)*math.sin(i)]])

    vv1 = math.cos(RAAN)*(math.sin(theta+w) + e*math.sin(w)) + \
          math.sin(RAAN)*(math.cos(theta+w) + e*math.cos(w))*math.cos(i)

    vv2 = math.sin(RAAN)*(math.sin(theta+w) + e*math.sin(w)) - \
          math.cos(RAAN)*(math.cos(theta+w) + e*math.cos(w))*math.cos(i)

    vv3 = -(math.cos(theta+w) + e*math.cos(w))*math.sin(i)
    
    v_vect = -GM/h * np.array([[vv1], [vv2], [vv3]])

    cart = np.concatenate((r_vect, v_vect), axis=0)
    
    return cart


def kep2eqn(kep):
    '''
    This function converts a vector of Keplerian orbital elements to a vector
    of equinoctial orbital elements
    
    Parameters
    ------
    kep : 6x1 numpy array
    
    Keplerian Orbital Elements
    ------
    kep[0] : a
      Semi-Major Axis             [km]
    kep[1] : e
      Eccentricity                [unitless]
    kep[2] : i
      Inclination                 [deg]
    kep[3] : RAAN
      Right Asc Ascending Node    [deg]
    kep[4] : w
      Argument of Periapsis       [deg]
    kep[5] : theta
      True Anomaly                [deg]
      
      
    Returns
    ------
    eqn : 6x1 numpy array
    
    Equinoctial Orbital Elements
    ------
    eqn[0] : a
      Semi-Major Axis               [km]
    eqn[1] : k
      k = e*cos(w + RAAN)           [unitless]
    cart[2] : h
      h = e*sin(w + RAAN)           [unitless]
    cart[3] : q
      q = tan(i/2)*cos(RAAN)        [unitless]
    cart[4] : p
      p = tan(i/2)*sin(RAAN)        [unitless]
    cart[5] : lambda
      Mean Longitude                [deg]  
      
    '''
    
    # Retrieve input elements, convert to radians
    a = float(kep[0])
    e = float(kep[1])
    i = float(kep[2]) * math.pi/180
    RAAN = float(kep[3]) * math.pi/180
    w = float(kep[4]) * math.pi/180
    theta = float(kep[5]) * math.pi/180
    
    # Compute mean anomaly
    E = true2ecc(theta, e)
    M = ecc2mean(E, e)
    
    # Compute equinoctial elements
    k = e*math.cos(w + RAAN)
    h = e*math.sin(w + RAAN)
    q = math.tan(i/2)*math.cos(RAAN)
    p = math.tan(i/2)*math.sin(RAAN)
    lam = (RAAN + w + M)*180./math.pi
    
    # Reset range to 0-360
    lam = lam % 360.
    
    # Output
    eqn = np.array([[a], [k], [h], [q], [p], [lam]])    
    
    return eqn


def kep2modeqn(kep):
    '''
    This function converts a vector of Keplerian orbital elements to a vector
    of modified equinoctial orbital elements
    
    Parameters
    ------
    kep : 6x1 numpy array
    
    Keplerian Orbital Elements
    ------
    kep[0] : a
      Semi-Major Axis             [km]
    kep[1] : e
      Eccentricity                [unitless]
    kep[2] : i
      Inclination                 [deg]
    kep[3] : RAAN
      Right Asc Ascending Node    [deg]
    kep[4] : w
      Argument of Periapsis       [deg]
    kep[5] : theta
      True Anomaly                [deg]
      
      
    Returns
    ------
    modeqn : 6x1 numpy array
    
    Modified Equinoctial Orbital Elements
    ------
    eqn[0] : p
      Semi-Latus Rectum             [km]
    eqn[1] : k
      f = e*cos(w + RAAN)           [unitless]
    cart[2] : h
      g = e*sin(w + RAAN)           [unitless]
    cart[3] : q
      h = tan(i/2)*cos(RAAN)        [unitless]
    cart[4] : p
      k = tan(i/2)*sin(RAAN)        [unitless]
    cart[5] : L
      True Longitude                [deg]  
      
    '''
    
    # Retrieve input elements, convert to radians
    a = float(kep[0])
    e = float(kep[1])
    i = float(kep[2]) * math.pi/180
    RAAN = float(kep[3]) * math.pi/180
    w = float(kep[4]) * math.pi/180
    theta = float(kep[5]) * math.pi/180
    
    # Compute modified equinoctial elements
    p = a*(1. - e**2.)
    f = e*math.cos(w + RAAN)
    g = e*math.sin(w + RAAN)
    h = math.tan(i/2)*math.cos(RAAN)
    k = math.tan(i/2)*math.sin(RAAN)
    L = (RAAN + w + theta)*180./math.pi
    
    # Reset range to 0-360
    L = L % 360.
    
    # Output
    modeqn = np.array([[p], [f], [g], [h], [k], [L]])    
    
    return modeqn


def cart2eqn(cart, GM=GME):
    '''
    This function converts a Cartesian state vector in inertial frame to
    equinoctial orbital elements.
    
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
    eqn : 6x1 numpy array
    
    Equinoctial Orbital Elements
    ------
    eqn[0] : a
      Semi-Major Axis               [km]
    eqn[1] : k
      k = e*cos(w + RAAN)           [unitless]
    cart[2] : h
      h = e*sin(w + RAAN)           [unitless]
    cart[3] : q
      q = tan(i/2)*cos(RAAN)        [unitless]
    cart[4] : p
      p = tan(i/2)*sin(RAAN)        [unitless]
    cart[5] : lambda
      Mean Longitude                [deg]  
      
    '''
    
    kep = cart2kep(cart, GM)
    eqn = kep2eqn(kep)    
    
    return eqn


def cart2modeqn(cart, GM=GME):
    '''
    This function converts a Cartesian state vector in inertial frame to
    modified equinoctial orbital elements.
    
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
    modeqn : 6x1 numpy array
    
    Modified Equinoctial Orbital Elements
    ------
    eqn[0] : p
      Semi-Latus Rectum             [km]
    eqn[1] : k
      f = e*cos(w + RAAN)           [unitless]
    cart[2] : h
      g = e*sin(w + RAAN)           [unitless]
    cart[3] : q
      h = tan(i/2)*cos(RAAN)        [unitless]
    cart[4] : p
      k = tan(i/2)*sin(RAAN)        [unitless]
    cart[5] : L
      True Longitude                [deg]   
      
    '''
    
    kep = cart2kep(cart, GM)
    modeqn = kep2modeqn(kep)    
    
    return modeqn


def modeqn2cart(modeqn, GM=GME):
    '''
    This function converts a vector of modified equinoctial orbital elements 
    to a Cartesian state vector in inertial frame.
    
    Parameters
    ------
    modeqn : 6x1 numpy array
    
    Modified Equinoctial Orbital Elements
    ------
    eqn[0] : p
      Semi-Latus Rectum             [km]
    eqn[1] : k
      f = e*cos(w + RAAN)           [unitless]
    cart[2] : h
      g = e*sin(w + RAAN)           [unitless]
    cart[3] : q
      h = tan(i/2)*cos(RAAN)        [unitless]
    cart[4] : p
      k = tan(i/2)*sin(RAAN)        [unitless]
    cart[5] : L
      True Longitude                [deg]
      
      
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
      
      
    Reference
    ------
    [1] Betts, J.T., "Practical Method for Optimal Control and Estimation 
        Using Nonlinear Programming," 2nd ed. 2010.
      
    '''
    
    # Retrieve input elements, convert to radians
    p = float(modeqn[0])
    f = float(modeqn[1])
    g = float(modeqn[2])
    h = float(modeqn[3])
    k = float(modeqn[4])
    L = float(modeqn[5]) * math.pi/180
    
    # Compute intermediate quantities (Betts Eq 6.37-6.41)
    q = 1. + f*math.cos(L) + g*math.sin(L)
    r = p/q
    alpha2 = h**2. - k**2.
    chi = np.sqrt(h**2. + k**2.)
    s2 = 1. + chi**2.
    
    # Compute inertial position and velocity vectors
    r_vect = (r/s2)*np.array([[math.cos(L) + alpha2*math.cos(L) + 2.*h*k*math.sin(L)],
                              [math.sin(L) - alpha2*math.sin(L) + 2.*h*k*math.cos(L)],
                              [2.*(h*math.sin(L) - k*math.cos(L))]])
    
    v1 =  math.sin(L) + alpha2*math.sin(L) - 2.*h*k*math.cos(L) + g - 2.*f*h*k + alpha2*g
    v2 = -math.cos(L) + alpha2*math.cos(L) + 2.*h*k*math.sin(L) - f + 2.*g*h*k + alpha2*f
    v3 = -2.*(h*math.cos(L) + k*math.sin(L) + f*h + g*k)
    
    v_vect = (-1./s2)*np.sqrt(GM/p)*np.array([[v1], [v2], [v3]])
    
    cart = np.concatenate((r_vect, v_vect), axis=0)    
    
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
      input flag (0 = orbital elements, 1 = cartesian coordinates)
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
        i = float(x_in[2]) * math.pi/180
        RAAN = float(x_in[3]) * math.pi/180
        w = float(x_in[4]) * math.pi/180
        Mo = float(x_in[5]) * math.pi/180

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
        RAAN = math.atan2(ih_vect[0], -ih_vect[1])   # rad
        i = math.acos(ih_vect[2])   # rad
        while RAAN < 0.:
            RAAN += 2.*math.pi

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
            ie_vect = np.array([[math.cos(RAAN)], [math.sin(RAAN)], [0.]])

        # Find orthogonal unit vector to complete perifocal frame
        ip_vect = np.cross(ih_vect, ie_vect, axis=0)

        # Form rotation matrix PN
        PN = np.concatenate(([ie_vect], [ip_vect], [ih_vect]))

        # Calculate argument of periapsis
        w = math.atan2(PN[0][2], PN[1][2])  # rad
        while w < 0.:
            w += 2.*math.pi

        # Calculate true anomaly, eccentric/hyperbolic anomaly, mean anomaly
        cross1 = np.cross(ie_vect, ir_vect, axis=0)
        tan1 = np.dot(cross1.T, ih_vect)
        tan2 = np.dot(ie_vect.T, ir_vect)
        f = math.atan2(tan1, tan2)    # rad

        # Calculate M
        if a > 0:
            n = np.sqrt(GM/a**3)
#            Erad = 2*atan(np.sqrt((1-e)/(1+e))*tan(f/2))    # rad
            Erad = true2ecc(f, e)
            Mo = Erad - e*math.sin(Erad)   # rad
            while Mo < 0:
                Mo = Mo + 2*math.pi
        elif a < 0:
            n = np.sqrt(GM/-a**3)
#            Hrad = 2*atanh(np.sqrt((e-1)/(e+1))*tan(f/2))  # rad
            Hrad = true2hyp(f, e)
            Mo = e*math.sinh(Hrad) - Hrad  # rad

        else:
            print('Error, input orbit is parabolic, a = ', a)
        
    else:
        print('Error: Invalid Input Flag!')

    # Solve for M(t) = Mo + n*dt
    M = Mo + n*dt   # rad

    # Generate output vector x_out
    if oflag == 0:
        
        # Ensure M is between 0 and 2*pi for elliptical orbits
        if a > 0:
            
            M = math.fmod(M, 2*math.pi)
            if M < 0:
                M += 2*math.pi

        # Convert angles to degrees
        i = i * 180/math.pi
        RAAN = RAAN * 180/math.pi
        w = w * 180/math.pi
        M = M * 180/math.pi

        x_out = np.array([[a], [e], [i], [RAAN], [w], [M]])

    elif oflag == 1:

        # Find eccentric/hyperbolic anomaly and true anomaly
        if a > 0:
            Erad = mean2ecc(M, e)    # rad
            f = ecc2true(Erad, e)
#            f = 2*atan(np.sqrt((1+e)/(1-e))*tan(Erad/2))    # rad
            r = a*(1 - e*math.cos(Erad))     # km
        elif a < 0:
            Hrad = mean2hyp(M, e)    # rad
            f = hyp2true(Hrad, e)
#            f = 2*atan(np.sqrt((e+1)/(e-1))*tanh(Hrad/2))    # rad
            r = a*(1 - e*math.cosh(Hrad))     # km
            
        # Calculate theta
        theta = f + w   # rad

        # Calculate r_vect and v_vect
        r_vect2 = r * \
            np.array([[math.cos(RAAN)*math.cos(theta) - math.sin(RAAN)*math.sin(theta)*math.cos(i)],
                      [math.sin(RAAN)*math.cos(theta) + math.cos(RAAN)*math.sin(theta)*math.cos(i)],
                      [math.sin(theta)*math.sin(i)]])

        vv1 = math.cos(RAAN)*(math.sin(theta) + e*math.sin(w)) + \
            math.sin(RAAN)*(math.cos(theta) + e*math.cos(w))*math.cos(i)

        vv2 = math.sin(RAAN)*(math.sin(theta) + e*math.sin(w)) - \
            math.cos(RAAN)*(math.cos(theta) + e*math.cos(w))*math.cos(i)

        vv3 = -(math.cos(theta) + e*math.cos(w))*math.sin(i)
        v_vect2 = -GM/h * np.array([[vv1], [vv2], [vv3]])

        x_out = np.concatenate([r_vect2, v_vect2])

    else:
        print('Error: Invalid Output Flag!')

    return x_out



    
    
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
    
    v0 = R*w*math.cos(lat_rad)
    
    return v0    



if __name__ == '__main__':
    
#    RAAN = 0.
#    UTC = datetime(2000, 1, 1, 12, 0, 0)
#    eop_alldata = get_celestrak_eop_alldata()
#    EOP_data = get_eop_data(eop_alldata, UTC)
#    
#    
#    LTAN = RAAN_to_LTAN(RAAN, UTC, EOP_data)
#    print(LTAN)
    
    
#    cart = np.array([710.87, 5151.26, 5075.69, 0.9039, -5.222, 5.1735])
#    kep = cart2kep(cart)
#    print(kep)
    
    kep = np.array([42164.1, 0., 0., 20., 30., 0.])
    cart = kep2cart(kep)
    eqn = kep2eqn(kep)
    modeqn = kep2modeqn(kep)
    
    cart2 = modeqn2cart(modeqn)
    err = np.linalg.norm(cart - cart2)
    
    print('kep', kep)
    print('cart', cart)
    print('eqn', eqn)
    print('modeqn', modeqn)
    print('cart2', cart2)
    print('err', err)
    
    
    
    
    