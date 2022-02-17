import numpy as np
from math import pi, sin, cos, tan, fmod, fabs, atan, atan2, acos, asin
from math import sinh, cosh, tanh, atanh
from datetime import datetime
import os
import sys
import inspect
import matplotlib.pyplot as plt

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from utilities.constants import GME, J2E, Re, wE


###############################################################################
#
# References
# 
# [1] "Handbook of Satellite Orbits: From Kepler to GPS," Capderou, M., 2014.
#
###############################################################################


###############################################################################
# Swath and FOV
###############################################################################

def compute_groundswath(a, fov, R=Re):
    '''
    This function computes the ground swath width in rad and km for a given
    orbit and satellite field of view. FOV is taken to be the full angle 
    visible from the satellite looking toward nadir.
    
    Parameters
    ------
    a : float
        semi-major axis [km]
    fov : float
        field of view [rad]
    R : float, optional
        radius of central body (default=Re)
        
    Returns
    ------
    swath_rad : float
        swath angle on surface of planet at equator [rad]
    swath_km : float
        swath distance on surface of planet at equator [km]
    
    
    '''
    
    # Compute angles using Ref 1 Eq 12.2 - 12.5
    f = (fov/2.)
    zeta = asin(a*sin(f)/R)     
    alpha = zeta - f
    
    # Compute full swath using Ref 1 Eq 
    swath_rad = alpha*2.
    swath_km = swath_rad*R

    return swath_rad, swath_km


def swath2fov(a, swath_rad, R=Re):
    '''
    
    '''
    
    
    alpha = swath_rad/2.
    rho = np.sqrt(R**2. + a**2. - 2.*Re*a*cos(alpha))
    f = asin((sin(alpha)/rho)*Re)
    fov = 2.*f
    
    return fov


def plot_swath_vs_altitude():
    '''
    This function creates a plot of ground swath in degrees vs altitude
    from 400-600 km.
    
    '''
    
    fov_list = [10., 15., 20.]
    alt_list = list(np.arange(400., 600., 1.))
    
    swath_data = np.zeros((len(fov_list), len(alt_list)))
    Nto_data = np.zeros((len(fov_list), len(alt_list)))
    Cto_data = np.zeros((len(fov_list), len(alt_list)))
    for fov in fov_list:
        for alt in alt_list:
            a = Re + alt
            n = np.sqrt(GME/a**3.) * 86400./(2.*pi)     # rev/day
            
            
            swath_rad, swath_km = compute_groundswath(a, fov*pi/180.)
            swath_data[fov_list.index(fov), alt_list.index(alt)] = swath_rad*180./pi
            
            Nto_min = np.ceil(2.*pi/swath_rad)
            Cto_min = np.floor(Nto_min/n)
            Nto_data[fov_list.index(fov), alt_list.index(alt)] = Nto_min
            Cto_data[fov_list.index(fov), alt_list.index(alt)] = Cto_min
            
    
    
    # Generate plot
    plt.figure()
    plt.plot(alt_list, swath_data[0,:], 'r.', label='10deg')
    plt.plot(alt_list, swath_data[1,:], 'g.', label='15deg')
    plt.plot(alt_list, swath_data[2,:], 'b.', label='20deg')
    plt.xlabel('Altitude [km]')
    plt.ylabel('Ground Swath Width [deg]')    
    plt.legend()
    
    plt.figure()
    plt.plot(alt_list, Nto_data[0,:], 'r.', label='10deg')
    plt.plot(alt_list, Nto_data[1,:], 'g.', label='15deg')
    plt.plot(alt_list, Nto_data[2,:], 'b.', label='20deg')
    plt.xlabel('Altitude [km]')
    plt.ylabel('Minimum Number of Revs for Repeat Nto')    
    plt.legend()
    
    plt.figure()
    plt.plot(alt_list, Cto_data[0,:], 'r.', label='10deg')
    plt.plot(alt_list, Cto_data[1,:], 'g.', label='15deg')
    plt.plot(alt_list, Cto_data[2,:], 'b.', label='20deg')
    plt.xlabel('Altitude [km]')
    plt.ylabel('Minimum Number of Days for Repeat Cto')    
    plt.legend()
    
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(alt_list, swath_data[0,:], 'r.', label='10deg')
    plt.plot(alt_list, swath_data[1,:], 'g.', label='15deg')
    plt.plot(alt_list, swath_data[2,:], 'b.', label='20deg')
    plt.ylabel('Swath [deg]')    
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.plot(alt_list, Nto_data[0,:], 'r.', label='10deg')
    plt.plot(alt_list, Nto_data[1,:], 'g.', label='15deg')
    plt.plot(alt_list, Nto_data[2,:], 'b.', label='20deg')
    plt.ylabel('Min Nto')
    
    plt.subplot(3,1,3)
    plt.plot(alt_list, Cto_data[0,:], 'r.', label='10deg')
    plt.plot(alt_list, Cto_data[1,:], 'g.', label='15deg')
    plt.plot(alt_list, Cto_data[2,:], 'b.', label='20deg')
    plt.ylabel('Min Cto')
    plt.xlabel('Altitude [km]')
    
    
    
    
    plt.show()
    
    
    return


###############################################################################
# Recurrent Orbit Functions
###############################################################################


def compute_recurrence_grid_parameters(vo, Dto, Cto):
    '''
    This function computes the recurrence grid parameters, the angular 
    difference in nodal longitude between consecutive revolutions, 
    consecutive days, and for the full repeating groundtrack.
    
    Parameters
    ------
    vo = int
        whole number of revolutions per day (rounded to nearest int)
    Dto = int
        whole number remainder such that kappa = vo + Dto/Cto
        Dto = mod(Nto, Cto) such that Dto/Cto <= 0.5
        Dto and Cto should be coprime
    Cto = int
        whole number of days before repeat
    
    Returns
    ------
    delta : float
        grid interval at the equator [rad]
    delta_rev : float
        difference in nodal longitude for consecutive revolutions [rad]
    delta_day : float
        difference in nodal longitude for consecutive days [rad]
    '''
    
    # Compute recurrence parameters
    kappa = vo + float(Dto)/float(Cto)       # rev/day
    Nto = vo*Cto + Dto
    
    delta = 2*pi/Nto
    delta_rev = delta*Cto
    delta_day = delta*Dto   
    
    return delta, delta_rev, delta_day


def generate_candidate_recurrent_triples(hmin, hmax, fov, R=Re, GM=GME):
    '''
    This function generates recurrent triples within a user defined altitude
    range.
    
    Parameters
    ------
    hmin : float
        minimum altitude [km]
    hmax : float
        maximum altitude [km]
    fov : float
        field of view [rad]
    R : float, optional
        planet radius [km] (default=Re)
    GM : float, optional
        planet gravitational parameter [km^3/s^2] (default=GME)
    
    Returns
    ------
    triple_primary_list : list
        list of lists, each entry contains [vo, Dto, Cto, Nto, Eto]
    triple_secondary_list : list
        list of lists, each entry contains [vo, Dto, Cto, Nto, Eto]  
    
    '''
    
    Cto_primary_list = [16., 20., 40.]
#    Cto_secondary_list = [10., 12., 15., 24., 25., 30., 32., 35.]
    Cto_secondary_list = [48., 56., 60.]
    
    triple_primary_list = compute_triple_list(hmin, hmax, fov, Cto_primary_list, R, GM)
    triple_secondary_list = compute_triple_list(hmin, hmax, fov, Cto_secondary_list, R, GM)
    
    # Generate data to plot and save in csv
    print(triple_primary_list)
    print(triple_secondary_list)
    
    pandas_data_list = []
    plot_v_primary = []
    plot_Cto_primary = []
    plot_h_primary = []
    for primary_list in triple_primary_list:
        
        vo = primary_list[0]
        Dto = primary_list[1]
        Cto = primary_list[2]
        Nto = primary_list[3]
        Eto = primary_list[4]
        
        # Assume near-circular sunsynch orbit
        a, i = nodal_period_to_sunsynch_orbit(Nto, Cto, 1e-4)
        h = a - R
        
        plot_v_primary.append(Nto/Cto)
        plot_Cto_primary.append(Cto)
        plot_h_primary.append(h)
        
        # Compute grid parameters
        delta, delta_rev, delta_day = compute_recurrence_grid_parameters(vo, Dto, Cto)
        
        # Compute swath and FOV requirements
        swath_km = delta*Re
        fov = swath2fov(a, delta)
        fov_deg = fov * 180./pi
        
        
        
        data_list = [vo, Dto, Cto, Nto, Eto, h, swath_km, fov_deg]
        pandas_data_list.append(data_list)
        
    plot_v_secondary = []
    plot_Cto_secondary = []
    plot_h_secondary = []
    for secondary_list in triple_secondary_list:
        
        vo = secondary_list[0] 
        Dto = secondary_list[1]
        Cto = secondary_list[2]
        Nto = secondary_list[3]
        Eto = secondary_list[4]
        
        # Assume near-circular sunsych orbit
        a, i = nodal_period_to_sunsynch_orbit(Nto, Cto, 1e-4)
        h = a - R
        
        plot_v_secondary.append(Nto/Cto)
        plot_Cto_secondary.append(Cto)
        plot_h_secondary.append(h)
        
        # Compute grid parameters
        delta, delta_rev, delta_day = compute_recurrence_grid_parameters(vo, Dto, Cto)
        
        # Compute swath and FOV requirements
        swath_km = delta*Re
        fov = swath2fov(a, delta)
        fov_deg = fov * 180./pi
        
        data_list = [vo, Dto, Cto, Nto, Eto, h, swath_km, fov_deg]
        pandas_data_list.append(data_list)
        
        
      
    # Generate plots
    n_15 = 15.*2.*pi/86400.
    a_15 = (GM/n_15**2.)**(1./3.)
    h_15 = a_15 - R
        
        
#    plt.figure()
#    plt.plot(plot_Cto_primary, plot_v_primary, 'b*', markersize=8)
#    plt.gca().invert_yaxis()
    
#    fig, ax1 = plt.subplots()
#    ax1.plot(plot_Cto_primary, plot_v_primary, 'b*', markersize=8)
#    ax1.set_xlabel('Repeat Cycle [days]')
#    ax1.set_ylabel('Revolutions Per Day')
    
    
    plt.figure()
    plt.plot(plot_Cto_primary, plot_h_primary, 'k.' ) #, markersize=8, label='Primary')
    plt.plot(plot_Cto_secondary, plot_h_secondary, 'k.' ) #, label='Secondary')
#    plt.plot([10., 45.], [h_15, h_15], 'k--', label='15 rev/day')
    plt.ylim([400., 650.])
    plt.xlim([10., 65.])
    plt.xlabel('Repeat Cycle [days]')
    plt.ylabel('Altitude [km]')
#    plt.legend()
    
    
    
    plt.show()
        
    
    
    # Generate pandas dataframe 
    column_headers = ['vo [rev/day]', 'Dto [revs]', 'Cto [days]',
                      'Nto [revs]', 'Eto [days]', 'Altitude [km]',
                      'Min Swath [km]', 'Min FOV [deg]']
    
    recurrent_df = pd.DataFrame(pandas_data_list, columns = column_headers)

    return recurrent_df


def compute_triple_list(hmin, hmax, fov, Cto_list, R=Re, GM=GME):
    '''
    This function computes a list of recurrent triples for the specified
    range of mean motion in rev/day and desired number of recurrent days 
    Cto.
    
    Parameters
    ------
    n_min : float
        minimum number of revs/day
    n_max : float
        maximum number of revs/day
    Cto_list : list
        desired whole numbers of days in repeat cycly
    
    Returns
    ------
    triple_list : list
        list of lists, each entry contains [vo, Dto, Cto, Nto, Eto]
    
    '''
    
    # Compute minumum and maximum mean motion in rev/day
    # Note that Keplerian period and mean motion are slightly different from 
    # nodal period and mean motion, but should be ok to set up these bounds
    a_min = R + hmin
    a_max = R + hmax
    n_max = np.sqrt(GM/a_min**3.) * 86400./(2.*pi)     # rev/day
    n_min = np.sqrt(GM/a_max**3.) * 86400./(2.*pi)     # rev/day
    
    
    
    # Find values of Nto that create rational numbers for valid ranges of 
    # Cto
    triple_list = []
    for Cto in Cto_list:
        
        # Generate candidate values of Nto
        Nto_min = np.ceil(Cto*n_min)
        Nto_max = np.floor(Cto*n_max)
        Nto_range = list(np.arange(Nto_min, Nto_max))
        
        # Remove entries that are not coprime
        del_list = []
        for Nto in Nto_range:
            if not is_coprime(Nto, Cto):
                del_list.append(Nto)
                
        Nto_list = list(set(Nto_range) - set(del_list))
                
        # Compute triples
        for Nto in Nto_list:
            vo = np.round(Nto/Cto)
            Dto = Nto - vo*Cto
            
            # Skip entries that have Eto = 1
            Eto = compute_Eto(vo, Dto, Cto)
            if Eto == 1:
                continue
            
            # Check delta < swath to ensure full global coverage
            # Assume near-circular sunsynch orbit
            a, i = nodal_period_to_sunsynch_orbit(Nto, Cto, 1e-4)
            swath_rad, swath_km = compute_groundswath(a, fov)
            delta = 2*pi/Nto
            
#            print(Nto, Cto)
#            print(a)
#            print(swath_rad, delta)
            
            if delta > swath_rad : 
                continue            
            
            triple = [vo, Dto, Cto, Nto, Eto]
            triple_list.append(triple)
            
#        print(Nto_list)
#        print(del_list)
#        print(triple_list)
   
    
    return triple_list


def is_coprime(x, y):
    '''
    This function checks if two numbers are coprime
    
    Parameters
    ------
    x : int
        larger number
    y : int
        smaller number
        
    Returns
    ------
    coprime_flag : boolean
        True indicates the numbers are coprime, False indicates they are not
    
    '''
    
    while y != 0:
        x, y = y, x % y
    
    coprime_flag = x == 1    
    
    return coprime_flag


def compute_Eto(vo, Dto, Cto):
    '''
    This function computes the subcycle recurrence Eto, the number of days
    it takes for the first groundtrack to pass within delta of the original
    groundtrack. It is good practice to avoid Eto = 1 to coverage the base 
    interval faster than using the whole repeat cycle.
    
    Parameters
    ------
    vo = int
        whole number of revolutions per day (rounded to nearest int)
    Dto = int
        whole number remainder such that kappa = vo + Dto/Cto
        Dto = mod(Nto, Cto) such that Dto/Cto <= 0.5
        Dto and Cto should be coprime
    Cto = int
        whole number of days before repeat
        
    Returns
    ------
    Eto = int
        whole number of days in subcycle
    
    '''
    
    Eto_list = []
    for ii in range(1, int(Cto)):
        
        if (ii*Dto) % Cto == 1 or (ii*Dto) % Cto == Cto - 1:
            Eto_list.append(ii)
            
    Eto = min(Eto_list)
    
    
    return Eto


def compute_orbit_periods(a, e, i, R=Re, GM=GME, J2=J2E):
    '''
    This function computes the Keplerian, Anomalistic, and Nodal Periods of
    an orbit subject to J2 perturbation.
    
    Parameters
    ------
    a : float
        semi-major axis [km]
    e : float
        eccentricity
    i : float
        inclination [deg]
    R : float, optional
        radius of planet [km] (default=Re)
    GM : float, optional
        gravitiational parameter [km^3/s^2] (default=GME)
    J2 : float, optional
        J2 coefficient (default=J2E)
    
    Returns
    ------
    To : float
        Keplerian orbit period [sec]
    Ta : float
        anomalistic orbit period [sec]
    Td : float
        nodal orbit period [sec]
    
    '''
    
    # Convert inclination to radians
    i = i * pi/180.
    
    # Compute Keplerian orbit period
    no = np.sqrt(GM/a**3.)
    To = 2.*pi/no
    
    # Compute perturbation effects from J2
    dn = (3./(4.*(1-e**2.)**(3./2.))) * no * J2 * (R/a)**2. * (3.*cos(i)**2. - 1)
    dw = (3./(4.*(1-e**2.)**(2.))) * no * J2 * (R/a)**2. * (5.*cos(i)**2. - 1)
    
    # Compute anomalistic orbit period
    na = no + dn
    Ta = 2.*pi/na
    
    # Compute nodal period
    Td = ((1. - dn/no)/(1. + dw/no)) * To
       
    return To, Ta, Td


def nodal_period_to_sunsynch_orbit(Nto, Cto, e, R=Re, GM=GME, J2=J2E):
    
    # Compute constants
    sidereal_day = 2.*pi/wE
    k2 = 0.75 * (360./(2.*pi)) * J2 * np.sqrt(GM) * R**2. * sidereal_day
    
    # Initial guess for SMA
    Td = Cto/Nto * sidereal_day
    n = 2.*pi/Td
    a = meanmot2sma(n, GM)
    
    # Iteratively solve for SMA
    a_prev = float(a)
    diff = 1.
    tol = 1e-4
    while diff > tol:
        
        # Compute inclination
        i = sunsynch_inclination(a, e)         # deg
        i = i * pi/180.                        # rad        
    
        # Compute J2 effects 
        dL = 360.       # deg/sidereal day
        dRAAN = -2.* k2 * a**(-7./2.) * cos(i) * (1. - e**2.)**(-2.)
        dw = k2 * a**(-7./2.) * (5.*cos(i)**2. - 1) * (1. - e**2.)**(-2.)
        dM = k2 * a**(-7./2.) * (3.*cos(i)**2. - 1) * (1. - e**2.)**(-3./2.)
        
        n = (Nto/Cto) * (dL - dRAAN) - (dw + dM)
        a = (GM**(1./3.)) * ((n*pi)/(180.*sidereal_day))**(-2./3.)
        diff = abs(a - a_prev)
        a_prev = float(a)
    
    
    # Convert to deg
    i = i * 180./pi
    
    return a, i