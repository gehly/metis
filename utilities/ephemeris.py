#import ephem
#import pysolar.julian as julian
#import pysolar.solar as solar

#import TurboProp.PyUtils.Time as TPT



# Compute the sun/moon position in ECI
#sun_eci = compute_sun_eci2(JED_JD)
#moon_eci = compute_moon_eci2(JED_JD)

import numpy as np
from math import pi, sin, cos, asin, atan2
from utilities.constants import AU_km


def compute_solar_coords(JED_JD):
    '''
    This function computes solar coordinates using the simplified model in
    Meeus Ch 25.
    
    Parameters
    ------
    JED_JD : float
        Julian Ephemeris Date (TDB) in JD format
        
    Returns
    ------
    ra : float
        apparent geocentric right ascension of sun
    dec : float
        apparent geocentric declinaion of sun
        
    Reference
    ------
    [1] Meeus, J., "Astronomical Algorithms," 2nd ed., 1998, Ch 25.
    
    '''
    
    deg2rad = pi/180.
    
    # Compute T in centuries from J2000
    T = (JED_JD - 2451545.)/36525.
    
    # Geometric Mean Longitude of the Sun (Mean Equinox of Date)
    Lo = 280.46646 + (36000.76983 + 0.0003032*T)*T  # deg
    
    # Mean Anomaly of the Sun
    M = 357.52911 + (35999.05028 - 0.0001537*T)*T   # deg
    Mrad = M*deg2rad                                # rad
    
    # Eccentricity of Earth's orbit
    ecc = 0.016708634 + (-0.000042037 - 0.0000001267*T)*T
    
    # Sun's Equation of Center
    C = (1.914602 - 0.004817*T - 0.000014*T*T)*sin(Mrad) + \
        (0.0019993 - 0.000101*T)*sin(2.*Mrad) + 0.00289*sin(3.*Mrad)  # deg
        
    # Sun True Longitude and True Anomaly
    true_long = Lo + C  # deg
    true_anom = M + C   # deg
    true_long_rad = true_long*deg2rad
    true_anom_rad = true_anom*deg2rad
    
    # Sun radius (distance from Earth)
    R_AU = 1.000001018*(1. - ecc**2)/(1 + ecc*cos(true_anom_rad))   # AU
    R_km = R_AU*AU_km                                                   # km
    
    # Compute Sun Apparent Longitude
    Omega = 125.04 - 1934.136*T                                         # deg
    Omega_rad = Omega*deg2rad                                           # rad
    apparent_long = true_long - 0.00569 - 0.00478*sin(Omega_rad)        # deg
    apparent_long_rad = apparent_long*deg2rad                           # rad
    
    # Obliquity of the Ecliptic (Eq 22.2)
    Eps0 = (((0.001813*T - 0.00059)*T - 46.8150)*T + 84381.448)/3600.   # deg
    Eps0_rad = Eps0*deg2rad                                             # rad
    
    # Geometric Coordinates
    ra_geom = atan2(cos(Eps0_rad)*sin(true_long_rad), cos(true_long_rad)) # rad
    dec_geom = asin(sin(Eps0_rad)*sin(true_long_rad))                     # rad
    sun_eci_geom = R_km*np.array([[cos(ra_geom)*cos(dec_geom)],
                                  [sin(ra_geom)*cos(dec_geom)],
                                  [sin(dec_geom)]])
    
    # Apparent Coordinates
    EpsA = Eps0 + 0.00256*cos(Omega_rad)    # deg
    EpsA_rad = EpsA*deg2rad 
    
    ra_app = atan2(cos(EpsA_rad)*sin(apparent_long_rad), cos(apparent_long_rad)) # rad
    dec_app = asin(sin(EpsA_rad)*sin(apparent_long_rad))                         # rad
    sun_eci_app = R_km*np.array([[cos(ra_app)*cos(dec_app)],
                                 [sin(ra_app)*cos(dec_app)],
                                 [sin(dec_app)]])

    
    return sun_eci_geom, sun_eci_app





























#def compute_sun_eci(JED_JD):
#    '''
#    This function computes the current sun position in ECI using the PyEphem
#    module.
#
#    Parameters
#    ------
#    JED_JD : float
#        current time [JED] in julian date format
#
#    Returns
#    ------
#    sun_eci : 3x1 numpy array
#        sun position in ECI [km]
#    '''
#
#    # Convert times
#    UTC_JD = TPT.TimeFrame([JED_JD], 'JED_JD', 'UTC_JD')[0]
#
#    # Set up sun object
#    sun = ephem.Sun()
#
#    # Get angles and position
#    sun.compute(UTC_JD)
#    ra = sun.a_ra*1.
#    dec = sun.a_dec*1.
#    dist = sun.earth_distance*149597870.700
#    
#    print 'ra', ra*180/pi
#    print 'dec', dec*180/pi
#
#    sun_eci = dist * np.array([[cos(ra)*cos(dec)], [sin(ra)*cos(dec)],
#                               [sin(dec)]])
#
#    return sun_eci
#
#
#def compute_moon_eci(JED_JD):
#    '''
#    This function computes the current moon position in ECI using the PyEphem
#    module.
#
#    Parameters
#    ------
#    JED_JD : float
#        current time [JED] in julian date format
#
#    Returns
#    ------
#    moon_eci : 3x1 numpy array
#        moon position in ECI [km]
#    '''
#
#    # Convert times
#    UTC_JD = TPT.TimeFrame([JED_JD], 'JED_JD', 'UTC_JD')[0]
#
#    # Set up moon object
#    moon = ephem.Moon()
#
#    # Get angles and position
#    moon.compute(UTC_JD)
#    ra = moon.a_ra*1.
#    dec = moon.a_dec*1.
#    dist = moon.earth_distance*149597870.700
#
#    moon_eci = dist * np.array([[cos(ra)*cos(dec)], [sin(ra)*cos(dec)],
#                                [sin(dec)]])
#
#    return moon_eci


#def compute_sun_eci2(jde):
#    '''
#    output is good for apparent ra/dec as opposed to astrometric ra/dec
#    '''
#
#    # time-dependent calculations
#    jce = julian.GetJulianEphemerisCentury(jde)
#    jme = julian.GetJulianEphemerisMillenium(jce)
#    geocentric_latitude = solar.GetGeocentricLatitude(jme)
#    geocentric_longitude = solar.GetGeocentricLongitude(jme)
#    radius_vector = solar.GetRadiusVector(jme)
#    aberration_correction = solar.GetAberrationCorrection(radius_vector)
#    nutation = solar.GetNutation(jde)
#    true_ecliptic_obliquity = solar.GetTrueEclipticObliquity(jme, nutation)
#
#    # calculations dependent on location and time
#    apparent_sun_longitude = \
#        solar.GetApparentSunLongitude(geocentric_longitude, nutation,
#                                      aberration_correction)
#    geocentric_sun_right_ascension = \
#        solar.GetGeocentricSunRightAscension(apparent_sun_longitude,
#                                             true_ecliptic_obliquity,
#                                             geocentric_latitude)
#    geocentric_sun_declination = \
#        solar.GetGeocentricSunDeclination(apparent_sun_longitude,
#                                          true_ecliptic_obliquity,
#                                          geocentric_latitude)
#
#    ra_rad = geocentric_sun_right_ascension * pi/180.
#    dec_rad = geocentric_sun_declination * pi/180.
#    
#    radius_km = np.linalg.norm(radius_vector)*149597870.700
#    
##    print 'ra', ra_rad*180/pi
##    print 'dec', dec_rad*180/pi
##    print 'radius', np.linalg.norm(radius_vector)
#    
#
#    sun_eci = np.array([[cos(ra_rad)*cos(dec_rad)], [sin(ra_rad)*cos(dec_rad)],
#                        [sin(dec_rad)]]) * radius_km
#
#    # sun_hat_eci = sun_eci/np.linalg.norm(sun_eci)
#
#    return sun_eci
#
#
#def compute_moon_eci2(jde):
#    '''
#    Output is good for apparent ra/dec as opposed to astrometric ra/dec
#    '''
#
#    jce = julian.GetJulianEphemerisCentury(jde)
#
#    moon_mean_longitude = (218.3164477 + 481267.88123421*jce -
#                           0.0015786*jce**2. + (jce**3.)/538841. -
#                           (jce**4.)/65194000.) * pi/180.
#
#    moon_mean_elongation = (297.8501921 + 445267.1114034*jce -
#                            0.0018819*jce**2. + (jce**3.)/545868. -
#                            (jce**4.)/113065000.) * pi/180.
#
#    sun_mean_anomaly = (357.5291092 + 35999.0502909*jce - 0.0001536*jce**2. +
#                        (jce**3.)/24490000.) * pi/180.
#
#    moon_mean_anomaly = (134.9633964 + 477198.8675055*jce + 0.0087414*jce**2. +
#                         (jce**3.)/69699. - (jce**4.)/14712000.) * pi/180.
#
#    moon_arg_lat = (93.2720950 + 483202.0175233*jce - 0.0036539*jce**2. -
#                    (jce**3.)/3526000. + (jce**4.)/863310000.) * pi/180.
#
#    moon_loan = (125.04452 - 1934.136261*jce + 0.0020708*jce**2. +
#                 (jce**3.)/450000)*pi/180.
#
#    sun_mean_longitude = (280.4665 + 36000.7689*jce)*pi/180.
#
#    obliquity0 = (23.*3600. + 26.*60. + 21.448 - 46.8150*jce -
#                  0.00059*jce**2. + 0.001813*jce**3)
#    obliquity0 *= (1./206265.)
#
#    nut_longitude = (-17.2*sin(moon_loan) - 1.32*sin(2*sun_mean_longitude) -
#                     0.23*sin(2*moon_mean_longitude) + 0.21*sin(2*moon_loan))
#    nut_longitude *= (1./206265.)
#
#    nut_obliquity = (9.20*cos(moon_loan) + 0.57*cos(2*sun_mean_longitude) +
#                     0.1*cos(2*moon_mean_longitude) - 0.09*cos(2*moon_loan))
#    nut_obliquity *= (1./206265.)
#
#    A1 = (119.75 + 131.849*jce) * pi/180.
#    A2 = (53.09 + 479264.290*jce) * pi/180.
#    A3 = (313.45 + 481266.484*jce) * pi/180.
#
#    E = 1. - 0.002516*jce - 0.0000074*jce**2.
#
##    print 'jce', jce
##    print 'Lp', (moon_mean_longitude*180./pi)%360
##    print 'D', (moon_mean_elongation*180./pi)%360
##    print 'M', (sun_mean_anomaly*180./pi)%360
##    print 'Mp', (moon_mean_anomaly*180./pi)%360
##    print 'F', (moon_arg_lat*180./pi)%360
##    print 'A1', (A1*180./pi)%360
##    print 'A2', (A2*180./pi)%360
##    print 'A3', (A3*180./pi)%360
##    print 'E', E
#    
#
#    D_list1 = [0,2,2,0,0,0,2,2,2,2,0,1,0,2,0,0,4,0,4,2,2,1,1,2,2,4,2,0,2,2,1,2,
#               0,0,2,2,2,4,0,3,2,4,0,2,2,2,4,0,4,1,2,0,1,3,4,2,0,1,2,2]
#
#    M_list1 = [0,0,0,0,1,0,0,-1,0,-1,1,0,1,0,0,0,0,0,0,1,1,0,1,-1,0,0,0,1,0,-1,
#               0,-2,1,2,-2,0,0,-1,0,0,1,-1,2,2,1,-1,0,0,-1,0,1,0,1,0,0,-1,2,1,
#               0,0]
#
#    Mp_list1 = [1,-1,0,2,0,0,-2,-1,1,0,-1,0,1,0,1,1,-1,3,-2,-1,0,-1,0,1,2,0,-3,
#                -2,-1,-2,1,0,2,0,-1,1,0,-1,2,-1,1,-2,-1,-1,-2,0,1,4,0,-2,0,2,1,
#                -2,-3,2,1,-1,3,-1]
#
#    F_list1 = [0,0,0,0,0,2,0,0,0,0,0,0,0,-2,2,-2,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,
#               0,0,0,0,-2,2,0,2,0,0,0,0,0,0,-2,0,0,0,0,-2,-2,0,0,0,0,0,0,0,-2]
#    
#    L_coeff = [6288774,1274027,658314,213618,-185116,-114332,58793,57066,53322,
#               45758,-40923,-34720,-30383,15327,-12528,10980,10675,10034,8548,
#               -7888,-6766,-5163,4987,4036,3994,3861,3665,-2689,-2602,2390,
#               -2348,2236,-2120,-2069,2048,-1773,-1595,1215,-1110,-892,-810,
#               759,-713,-700,691,596,549,537,520,-487,-399,-381,351,-340,330,
#               327,-323,299,294,0]
#    
#    R_coeff = [-20905355,-3699111,-2955968,-569925,48888,-3149,246158,-152138,
#               -170733,-204586,-129620,108743,104755,10321,0,79661,-34782,
#               -23210,-21636,24208,30824,-8379,-16675,-12831,-10445,-11650,
#               14403,-7003,0,10056,6322,-9884,5751,0,-4950,4130,0,-3958,0,3258,
#               2616,-1897,-2117,2354,0,0,-1423,-1117,-1571,-1739,0,-4421,0,0,0,
#               0,1165,0,0,8752]
#    
#    D_list2 = [0,0,0,2,2,2,2,0,2,0,2,2,2,2,2,2,2,0,4,0,0,0,1,0,0,0,1,0,4,4,0,4,
#               2,2,2,2,0,2,2,2,2,4,2,2,0,2,1,1,0,2,1,2,0,4,4,1,4,1,4,2]
#    
#    M_list2 = [0,0,0,0,0,0,0,0,0,0,-1,0,0,1,-1,-1,-1,1,0,1,0,1,0,1,1,1,0,0,0,0,
#               0,0,0,0,-1,0,0,0,0,1,1,0,-1,-2,0,1,1,1,1,1,0,-1,1,0,-1,0,0,0,-1,
#               -2]
#    
#    Mp_list2 = [0,1,1,0,-1,-1,0,2,1,2,0,-2,1,0,-1,0,-1,-1,-1,0,0,-1,0,1,1,0,0,
#                3,0,-1,1,-2,0,2,1,-2,3,2,-3,-1,0,0,1,0,1,1,0,0,-2,-1,1,-2,2,-2,
#                -1,1,1,-1,0,0]
#    
#    F_list2 = [1,1,-1,-1,1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,-1,-1,-1,1,3,1,1,1,-1,
#               -1,-1,1,-1,1,-3,1,-3,-1,-1,1,-1,1,-1,1,1,1,1,-1,3,-1,-1,1,-1,-1,
#               1,-1,1,-1,-1,-1,-1,-1,-1,1]
#    
#    B_coeff = [5128122,280602,277693,173237,55413,46271,32573,17198,9266,8822,
#               8216,4324,4200,-3359,2463,2211,2065,-1870,1828,-1794,-1749,
#               -1565,-1491,-1475,-1410,-1344,-1335,1107,1021,833,777,671,607,
#               596,491,-451,439,422,421,-366,-351,331,315,302,-283,-229,223,
#               223,-220,-220,-185,181,-177,176,166,-164,132,-119,115,107]
#    
#    L_sum = 0.
#    R_sum = 0.
#    B_sum = 0.
#    for ii in xrange(len(D_list1)):
#        D1 = D_list1[ii]
#        M1 = M_list1[ii]
#        Mp1 = Mp_list1[ii]
#        F1 = F_list1[ii]
#        D2 = D_list2[ii]
#        M2 = M_list2[ii]
#        Mp2 = Mp_list2[ii]
#        F2 = F_list2[ii]
#        L = L_coeff[ii]
#        R = R_coeff[ii]
#        B = B_coeff[ii]
#
#        if abs(M1) == 1.:
#            L *= E
#            R *= E
#        if abs(M1) == 2.:
#            L *= E**2.
#            R *= E**2.
#        if abs(M2) == 1.:
#            B *= E
#        if abs(M2) == 2.:
#            B *= E**2.
#
#        L_sum += L*sin(D1*moon_mean_elongation + M1*sun_mean_anomaly +
#                       Mp1*moon_mean_anomaly + F1*moon_arg_lat)
#
#        R_sum += R*cos(D1*moon_mean_elongation + M1*sun_mean_anomaly +
#                       Mp1*moon_mean_anomaly + F1*moon_arg_lat)
#
#        B_sum += B*sin(D2*moon_mean_elongation + M2*sun_mean_anomaly +
#                       Mp2*moon_mean_anomaly + F2*moon_arg_lat)
#
#    L_sum += 3958.*sin(A1) + 1962.*sin(moon_mean_longitude - moon_arg_lat) \
#        + 318.*sin(A2)
#
#    B_sum += -2235.*sin(moon_mean_longitude) + 382.*sin(A3) \
#        + 175.*sin(A1 - moon_arg_lat) + 175.*sin(A1 + moon_arg_lat) \
#        + 127.*sin(moon_mean_longitude - moon_mean_anomaly) \
#        - 115.*sin(moon_mean_longitude + moon_mean_anomaly)
#
#    lon_rad = moon_mean_longitude + (L_sum/1e6)*pi/180.
#    lon_deg = (lon_rad*180./pi) % 360.
#    lat_deg = (B_sum/1e6)
#    r = 385000.56 + R_sum/1000.
#    
##    print 'lon deg', lon_deg
##    print 'nut_long', nut_longitude*180./pi
##    print 'obl0', obliquity0*180./pi
##    print 'nut ob', nut_obliquity*180./pi
#
#    lon_deg += nut_longitude
#    obliquity_rad = obliquity0 + nut_obliquity
#    lon_rad = lon_deg * pi/180.
#    lat_rad = lat_deg * pi/180.
#
##    print 'L_sum', L_sum
##    print 'R_sum', R_sum
##    print 'B_sum', B_sum
##
##    print 'gc lon', lon_deg
##    print 'gc lat', lat_deg
##    print 'r', r
##    print 'obliquity', obliquity_rad*180./pi
#
#    ra = atan2(sin(lon_rad)*cos(obliquity_rad) -
#               tan(lat_rad)*sin(obliquity_rad), cos(lon_rad))
#    dec = asin(sin(lat_rad)*cos(obliquity_rad) +
#               cos(lat_rad)*sin(obliquity_rad)*sin(lon_rad))
#
#    moon_hat_eci = np.array([[cos(ra)*cos(dec)],
#                             [sin(ra)*cos(dec)],
#                             [sin(dec)]])
#                             
#    moon_eci = r * moon_hat_eci
#
##    print 'ra', atan2(moon_hat_eci[1], moon_hat_eci[0])*180./pi
##    print 'dec', asin(moon_hat_eci[2])*180./pi
#
#    return moon_eci