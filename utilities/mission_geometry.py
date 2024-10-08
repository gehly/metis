import numpy as np
import math
from datetime import datetime
import pandas as pd
import os
import sys
import inspect
import matplotlib.pyplot as plt

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from utilities import astrodynamics as astro
from utilities import eop_functions as eop
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
    zeta = math.asin(a*math.sin(f)/R)
    alpha = zeta - f
    
    # Compute full swath using Ref 1 Eq 
    swath_rad = alpha*2.
    swath_km = swath_rad*R

    return swath_rad, swath_km


def swath2fov(a, swath_rad, R=Re):
    '''
    
    '''
    
    
    alpha = swath_rad/2.
    rho = np.sqrt(R**2. + a**2. - 2.*Re*a*math.cos(alpha))
    f = math.asin((math.sin(alpha)/rho)*Re)
    fov = 2.*f
    
    return fov


def alpha2f(alpha, a, R=Re):
    
    
    rho = np.sqrt(R**2. + a**2. - 2.*R*a*math.cos(alpha))
    f = math.asin((math.sin(alpha)/rho)*R)
    
    return f

def f2el(f, a, R=Re):
    
    sinx = a*math.sin(f)/R
    el = math.pi/2. - math.asin(sinx)
    
    return el


def swath2Nto(swath_km, R=Re):
    '''
    This function computes the minimum number of revs before repeat Nto 
    required for full global coverage with no gaps, given the swath in km.
    
    Parameters
    ------
    swath_km : float
        swath width on surface of planet at equator [km]
    R : float, optional
        radius of central body (default=Re)
        
    Returns
    ------
    Nto : int
        whole number of orbit revolutions before repeat
        
    '''
    
    swath_rad = swath_km/R
    Nto = int(np.ceil(2.*math.pi/swath_rad))
    
    return Nto


def compute_minimum_repeat(h, fov, R=Re):
    '''
    This function computes the minimum number of revs and days required for
    full global coverage with no gaps.
    
    Parameters
    ------
    h : float
        altitude [km]
    fov : float
        field of view [rad]
    R : float, optional
        radius of central body (default=Re)
        
    Returns
    ------
    Nto : int
        whole number of orbit revolutions before repeat
    Cto : int
        whole number of days before repeat
    
    '''
    
    # Compute swath
    a = h + R
    swath_rad, swath_km = compute_groundswath(a, fov, R)
    
    # Compute minumum Nto
    Nto = swath2Nto(swath_km, R)
    
    # Compute minimum Cto
    n = astro.sma2meanmot(a)
    n_revday = n * 86400./(2.*math.pi)
    Cto = int(np.floor(Nto/n_revday))    
    
    return Nto, Cto


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
            n = np.sqrt(GME/a**3.) * 86400./(2.*math.pi)     # rev/day
            
            
            swath_rad, swath_km = compute_groundswath(a, fov*math.pi/180.)
            swath_data[fov_list.index(fov), alt_list.index(alt)] = swath_rad*180./math.pi
            
            Nto_min = np.ceil(2.*math.pi/swath_rad)
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
    
    delta = 2*math.pi/Nto
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
        fov_deg = fov * 180./math.pi
        
        
        
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
        fov_deg = fov * 180./math.pi
        
        data_list = [vo, Dto, Cto, Nto, Eto, h, swath_km, fov_deg]
        pandas_data_list.append(data_list)
        
        
      
    # Generate plots
    n_15 = 15.*2.*math.pi/86400.
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
    n_max = np.sqrt(GM/a_min**3.) * 86400./(2.*math.pi)     # rev/day
    n_min = np.sqrt(GM/a_max**3.) * 86400./(2.*math.pi)     # rev/day
    
    
    
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
            delta = 2*math.pi/Nto
            
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


def Nto_to_triple(Nto_required, hmin, hmax, Cto_list, R=Re, GM=GME):
    
    # Compute minumum and maximum mean motion in rev/day
    # Note that Keplerian period and mean motion are slightly different from 
    # nodal period and mean motion, but should be ok to set up these bounds
    a_min = R + hmin
    a_max = R + hmax
    n_max = np.sqrt(GM/a_min**3.) * 86400./(2.*math.pi)     # rev/day
    n_min = np.sqrt(GM/a_max**3.) * 86400./(2.*math.pi)     # rev/day
    
    
    # Find values of Nto that create rational numbers for valid ranges of 
    # Cto
    data_list = []
    for Cto in Cto_list:
        
        # Generate candidate values of Nto
        Nto_min = np.ceil(Cto*n_min)
        Nto_max = np.floor(Cto*n_max)
        Nto_range = list(np.arange(Nto_min, Nto_max))
        
        if Nto_max < Nto_required:
            continue
        
        # Remove entries that are too small or not coprime
        del_list = []
        for Nto in Nto_range:
            if Nto < Nto_required:
                del_list.append(Nto)
                
            if not is_coprime(Nto, Cto):
                del_list.append(Nto)
                
        Nto_list = list(set(Nto_range) - set(del_list))
                
        # Compute triples
        for Nto in Nto_list:
            vo = np.round(Nto/Cto)
            Dto = Nto - vo*Cto
            
            # Skip entries that have Eto = 1
            Eto = compute_Eto(vo, Dto, Cto)
#            if Eto == 1:
#                continue
            
            # Check delta < swath to ensure full global coverage
            # Assume near-circular sunsynch orbit
            a, i = nodal_period_to_sunsynch_orbit(Nto, Cto, 1e-4)
            swath_rad, swath_km = compute_groundswath(a, fov)
            delta = 2*math.pi/Nto
            
#            print(Nto, Cto)
#            print(a)
#            print(swath_rad, delta)
            
           
            if delta > swath_rad : 
                continue   
            
            # Compute delta at Equator
            delta = 2*math.pi/Nto
            delta_km = R*delta
            
            
            # Orbit Altitude
            # Assume near-circular sunsynch orbit
            a, i = nodal_period_to_sunsynch_orbit(Nto, Cto, 1e-4)
            h = a - R
            
            
            data_list.append([vo, Dto, Cto, Nto, Eto, h, delta_km])
            
#        print(Nto_list)
#        print(del_list)
#        print(triple_list)
   
    
    
    # Generate pandas dataframe 
    column_headers = ['vo [rev/day]', 'Dto [revs]', 'Cto [days]',
                      'Nto [revs]', 'Eto [days]', 'Altitude [km]',
                      'Delta_Equator [km]']
    
    recurrent_df = pd.DataFrame(data_list, columns = column_headers)
    
    return recurrent_df


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
    i = i * math.pi/180.
    
    # Compute Keplerian orbit period
    no = np.sqrt(GM/a**3.)
    To = 2.*math.pi/no
    
    # Compute perturbation effects from J2
    dn = (3./(4.*(1-e**2.)**(3./2.))) * no * J2 * (R/a)**2. * (3.*math.cos(i)**2. - 1)
    dw = (3./(4.*(1-e**2.)**(2.))) * no * J2 * (R/a)**2. * (5.*math.cos(i)**2. - 1)
    
    # Compute anomalistic orbit period
    na = no + dn
    Ta = 2.*math.pi/na
    
    # Compute nodal period
    Td = ((1. - dn/no)/(1. + dw/no)) * To
       
    return To, Ta, Td


def nodal_period_to_sunsynch_orbit(Nto, Cto, e, R=Re, GM=GME, J2=J2E):
    
    # Compute constants
    sidereal_day = 2.*math.pi/wE
    k2 = 0.75 * (360./(2.*math.pi)) * J2 * np.sqrt(GM) * R**2. * sidereal_day
    
    # Initial guess for SMA
    Td = Cto/Nto * sidereal_day
    n = 2.*math.pi/Td
    a = astro.meanmot2sma(n, GM)
    
    # Iteratively solve for SMA
    a_prev = float(a)
    diff = 1.
    tol = 1e-4
    while diff > tol:
        
        # Compute inclination
        i = astro.sunsynch_inclination(a, e)         # deg
        i = i * math.pi/180.                        # rad
    
        # Compute J2 effects 
        dL = 360.       # deg/sidereal day
        dRAAN = -2.* k2 * a**(-7./2.) * math.cos(i) * (1. - e**2.)**(-2.)
        dw = k2 * a**(-7./2.) * (5.*math.cos(i)**2. - 1) * (1. - e**2.)**(-2.)
        dM = k2 * a**(-7./2.) * (3.*math.cos(i)**2. - 1) * (1. - e**2.)**(-3./2.)
        
        n = (Nto/Cto) * (dL - dRAAN) - (dw + dM)
        a = (GM**(1./3.)) * ((n*math.pi)/(180.*sidereal_day))**(-2./3.)
        diff = abs(a - a_prev)
        a_prev = float(a)
    
    
    # Convert to deg
    i = i * 180./math.pi
    
    return a, i


if __name__ == '__main__':
    
    
    print('\n\nLarge Satellite Case')
    
    h = 650
    fov = 20.*math.pi/180.
    Nto, Cto = compute_minimum_repeat(h, fov)
    
    print(Nto)
    print(Cto)
    
    Cto_list = [11, 12, 13, 14, 15]
    
    
    recurrent_df = Nto_to_triple(Nto, 600., 660., Cto_list)
    
    print(recurrent_df)
    
    fdir = r'D:\documents\teaching\unsw_orbital_mechanics\2022\lab'
    fname = os.path.join(fdir, 'large_sat_recurrent.csv')
#    recurrent_df.to_csv(fname)
    
    h = 633.2428
    el = f2el(fov/2., h+Re)
    
    print('el deg', el*180/math.pi)
    
    # Compute orbit parameters for Large Satellite
    e = 1e-4
    Nto = 192
    Cto = 13
    a, i = nodal_period_to_sunsynch_orbit(Nto, Cto, e)
    
    print(a)
    print(i)
    
    LTAN = 22.5
    UTC = datetime(2022, 2, 28, 22, 30, 0)
    eop_alldata = eop.get_celestrak_eop_alldata()
    EOP_data = eop.get_eop_data(eop_alldata, UTC)
    
    RAAN = astro.LTAN_to_RAAN(LTAN, UTC, EOP_data)
    
    print(RAAN)
    
    
    
    print('\n\nSmall Satellite Case')
    
    h = 575
    fov = 10.*math.pi/180.
    Nto, Cto = compute_minimum_repeat(h, fov)
    
    print(Nto)
    print(Cto)
    
    Cto_list = [25, 26, 27, 28, 29, 30, 31, 32]
    
    
    recurrent_df = Nto_to_triple(Nto, 500., 575., Cto_list)
    
    print(recurrent_df)
    
    fdir = r'D:\documents\teaching\unsw_orbital_mechanics\2022\lab'
    fname = os.path.join(fdir, 'small_sat_recurrent.csv')
#    recurrent_df.to_csv(fname)
    
    h = 549.975
    el = f2el(fov/2., h+Re)
    
    print('el deg', el*180/math.pi)
    
    # Compute orbit parameters for Large Satellite
    e = 1e-4
    Nto = 421
    Cto = 28
    a, i = nodal_period_to_sunsynch_orbit(Nto, Cto, e)
    
    print(a)
    print(i)
    
    LTAN = 22.5
    UTC = datetime(2022, 2, 28, 22, 30, 0)
    eop_alldata = eop.get_celestrak_eop_alldata()
    EOP_data = eop.get_eop_data(eop_alldata, UTC)
    
    RAAN = astro.LTAN_to_RAAN(LTAN, UTC, EOP_data)
    
    print(RAAN)
    
    
    
    
    
    
#    swath_km = 16.
#    h = 550.
#    
#    
#    Nto = swath2Nto(swath_km)
#    
#    print('Nto', Nto)
#    
#    n = sma2meanmot(Re+h)
#    n_revday = n * 86400./(2.*pi)
#    print('n [rev/day]', n_revday)
#    
#    print('min days in repeat', Nto/n_revday)
#    
#    
#    
#    
#    hmin = 540.
#    hmax = 650.
#    Cto_list = [160, 170, 180, 190, 200]
#    recurrent_df = Nto_to_triple(Nto, hmin, hmax, Cto_list)
#    
#    print(recurrent_df)
#    
##    fdir = r'D:\documents\research\cubesats\OzFuel'
##    fname = os.path.join(fdir, 'OzFuel_Recurrent_Orbits.csv')
##    recurrent_df.to_csv(fname)
#    
#    
#    # Final Orbit Elements/Setup
#    print('\n\nFinal Orbit Setup')
#    LTAN = 13.5
#    triple = [15, 7, 170]
#    Cto = triple[2]
#    Nto = triple[2]*triple[0] + triple[1]
#    
#    print(Nto, Cto)
#    e = 1e-4
#    a, i = nodal_period_to_sunsynch_orbit(Nto, Cto, e)
#    
#    h = a - Re
#    
#    swath_rad = swath_km/Re
#    alpha = swath_rad/2.
#    f = alpha2f(alpha, (Re+h))
#    fov_deg = f*2*180./pi
#    
#    print('fov_deg', fov_deg)
#    
#    
#    for slew_deg in [0., 5., 10., 15., 20.]:
#        slew_f = f + slew_deg*pi/180
#        min_el = f2el(slew_f, (Re+h))
#        
#        print('slew [deg]', slew_deg, 'min el [deg]', min_el*180./pi)
#    
#    UTC = datetime(2022, 2, 22, 13, 30, 0)
#    eop_alldata = get_celestrak_eop_alldata()
#    EOP_data = get_eop_data(eop_alldata, UTC)
#    RAAN = LTAN_to_RAAN(LTAN, UTC, EOP_data)
#    
#    print(a, e, i, RAAN)
#    
#    
#    
#    
#    a = Re + 550.
#    fov = 1.67*pi/180.
#    
#    for fov_deg in [1.67, 10, 20, 30, 40]:
#        
#        fov = fov_deg * pi/180.    
#        swath_rad, swath_km = compute_groundswath(a, fov, R=Re)
#        Nto = swath2Nto(swath_km)
#        Cto = Nto/15.
#    
#        print(swath_km)
#        print(Nto)
#        print(Cto)
#        
#        
#    
#    Nto = 100
#    hmin = 500.
#    hmax = 600.
#    Cto_list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#    recurrent_df = Nto_to_triple(Nto, hmin, hmax, Cto_list)
#    
#    print(recurrent_df)
#    
#    fdir = r'D:\documents\research\cubesats\OzFuel'
#    fname = os.path.join(fdir, 'OzFuel_Recurrent_Orbits2.csv')
#    recurrent_df.to_csv(fname)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    