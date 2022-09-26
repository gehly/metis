import numpy as np
import math
from scipy.integrate import odeint, ode
import sys
import os
import inspect
from datetime import datetime, timedelta
import time
import itertools

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

import dynamics.numerical_integration as numint
import utilities.astrodynamics as astro
import utilities.coordinate_systems as coord
import utilities.eop_functions as eop
import utilities.time_systems as timesys

from utilities.constants import Re, GME

###############################################################################
# Classical (Deterministic) Methods
###############################################################################


###############################################################################
# Lambert Solvers (2PBVP)
###############################################################################



def lambert_iod(tk_list, Yk_list, sensor_params):
    
    
    return



def izzo_lambert(r1_vect, r2_vect, tof, M_star=np.nan, lr_star='none', 
                 GM=GME, R=Re, results_flag='all', periapsis_check=True,
                 maxiters=35, rtol=1e-8):
    '''
    This function implements a solution to the twobody orbit boundary value
    problem (Lambert's Problem). Given initial and final position vectors and
    the time of flight, the method will compute the initial and final velocity
    vectors, which constitute a complete description of the orbit for all 
    times. The method is suitable for all orbit types and orbit regimes, and 
    works for single revolution and multiple revolution (elliptical orbit) 
    cases. For any given instance of Lambert's Problem, there may be multiple
    solutions that fit the boundary values and time of flight, with different
    number of orbit revolutions and orbiting in prograde or retrograde 
    directions. The function outputs all possible orbits that fit the 
    boundary conditions by default, but can be restricted to consider only
    prograde or retrogade orbits.    
    
    The method does not include any effects of perturbing forces.
    
    Parameters
    ------
    r1_vect : 3x1 numpy array
        initial position vector in inertial coordinates [km]
    r2_vect : 3x1 numpy array
        final position vector in inertial coordinates [km]
    tof : float
        time of flight between r1_vect and r2_vect [sec]
    M_star : int, optional
        exact integer number of complete orbit revolutions traversed
        (default=NAN)
    GM : float, optional
        gravitational parameter of central body (default=GME) [km^3/s^2]        
    R : float, optional
        radius of central body (default=Re) [km]
    results_flag : string, optional
        flag to determine what results to output 
        ('prograde', 'retrograde', or 'all', default='all')
    periapsis_check : boolean, optional
        flag to determine whether to check the orbit does not intersect the
        central body (rp > R) (default=True)
    maxiters : int, optional
        maximum number of iterations for the Halley and Householder 
        root-finding steps (default=35)
    rtol : float, optional
        convergence tolerance for the Halley and Householder root-finding 
        steps (default=1e-8)
        
    Returns
    ------
    v1_list : list
        list of 3x1 numpy arrays corresponding to valid initial velocity
        vectors in inertial coordinates [km/s]    
    v2_list : list
        list of 3x1 numpy arrays corresponding to valid final velocity 
        vectors in inertial coordinates [km/s]    
    M_list : list
        list of integer number of revolutions corresponding to initial/final
        velocity vector solutions
        
    Reference
    ------
    [1] Izzo, D. "Revisiting Lambert's Problem," CMDA 2015
    
    [2] Rodrigues, J.L.C. et al., Poliastro distribution 
    (DOI 10.5281/zenodo.6817189) identifies several bugs and improvements, 
    e.g. https://github.com/poliastro/poliastro/issues/1362
    
    '''
    
    # Input checking
    assert tof > 0
    
    # Compute chord and unit vectors
    r1_vect = np.reshape(r1_vect, (3,1))
    r2_vect = np.reshape(r2_vect, (3,1))
    c_vect = r2_vect - r1_vect
    
    r1 = np.linalg.norm(r1_vect)
    r2 = np.linalg.norm(r2_vect)
    c = np.linalg.norm(c_vect)
    
    # Compute values to reconstruct output
    s = 0.5 * (r1 + r2 + c)
    lam2 = 1. - (c/s)
    lam = np.sqrt(lam2)
    gamma = np.sqrt(GM*s/2.)
    rho = (r1 - r2)/c
    sigma = np.sqrt(1. - rho**2.)

    # Renormalize (cross product of unit vectors not necessarily unit length)
    ihat_r1 = r1_vect/r1
    ihat_r2 = r2_vect/r2
    ihat_h = np.cross(ihat_r1, ihat_r2, axis=0)
    ihat_h = ihat_h/np.linalg.norm(ihat_h)
    
    # Compute unit vectors (note error in Izzo paper)
    if float(ihat_h[2]) < 0.:
        lam = -lam
        ihat_t1 = np.cross(ihat_r1, ihat_h, axis=0)
        ihat_t2 = np.cross(ihat_r2, ihat_h, axis=0)
        
    else:
        ihat_t1 = np.cross(ihat_h, ihat_r1, axis=0)
        ihat_t2 = np.cross(ihat_h, ihat_r2, axis=0)
        
    # Correction for retrograde orbits
    if results_flag == 'retrograde':
        lam = -lam
        ihat_t1 = -ihat_t1
        ihat_t2 = -ihat_t2
    
    # Compute non-dimensional time of flight T
    T = np.sqrt(2*GM/s**3.) * tof
    
#    print('T', T)
    
    # Compute all possible x,y values that fit T
    if np.isnan(M_star):
        x_list1, y_list1, M_list1, lr_list1 = find_xy(lam, T, maxiters, rtol)
    else:
        x_list1, y_list1, M_list1, lr_list1 = \
            find_single_xy(lam, T, M_star, lr_star, maxiters, rtol)
        
#    print('x_list1', x_list1)
#    print('y_list1', y_list1)
#    print('M_list1', M_list1)
    
    # Loop over x,y values and compute output velocities
    v1_list = []
    v2_list = []
    M_list = []
    lr_list = []
    for ii in range(len(x_list1)):
        xi = x_list1[ii]
        yi = y_list1[ii]
        Mi = M_list1[ii]
        lri = lr_list1[ii]
        
        Vr1 =  gamma*((lam*yi - xi) - rho*(lam*yi + xi))/r1
        Vr2 = -gamma*((lam*yi - xi) + rho*(lam*yi + xi))/r2
        Vt1 =  gamma*sigma*(yi + lam*xi)/r1
        Vt2 =  gamma*sigma*(yi + lam*xi)/r2
        
        v1_vect = Vr1*ihat_r1 + Vt1*ihat_t1
        v2_vect = Vr2*ihat_r2 + Vt2*ihat_t2
        
        # Check for valid radius of periapsis
        rp = astro.compute_rp(r1_vect, v1_vect, GM)
        
#        print('rp', rp)
#        print('R', R)
        
        if (periapsis_check or Mi > 0) and rp < R:
            continue
        
        v1_list.append(v1_vect)
        v2_list.append(v2_vect)
        M_list.append(Mi)
        lr_list.append(lri)
    
    # Generate list to identify output orbit type
    if results_flag == 'retrograde':
        type_list = ['retrograde']*len(M_list)
    else:
        type_list = ['prograde']*len(M_list)
        
        
    # If it is desired to produce all possible cases, repeat execution for
    # retrograde cases (previous results cover prograde cases)
    if results_flag == 'all':
        lam = -lam
        ihat_t1 = -ihat_t1
        ihat_t2 = -ihat_t2
        
        # Compute all possible x,y values that fit T
        if np.isnan(M_star):
            x_list2, y_list2, M_list2, lr_list2 = find_xy(lam, T, maxiters, rtol)
        else:
            x_list2, y_list2, M_list2, lr_list2 = \
                find_single_xy(lam, T, M_star, lr_star, maxiters, rtol)
        
        # Loop over x,y values and compute output velocities
        for ii in range(len(x_list2)):
            xi = x_list2[ii]
            yi = y_list2[ii]
            Mi = M_list2[ii]
            lri = lr_list2[ii]
            
            Vr1 =  gamma*((lam*yi - xi) - rho*(lam*yi + xi))/r1
            Vr2 = -gamma*((lam*yi - xi) + rho*(lam*yi + xi))/r2
            Vt1 =  gamma*sigma*(yi + lam*xi)/r1
            Vt2 =  gamma*sigma*(yi + lam*xi)/r2
            
            v1_vect = Vr1*ihat_r1 + Vt1*ihat_t1
            v2_vect = Vr2*ihat_r2 + Vt2*ihat_t2
            
            # Check for valid radius of periapsis
            rp = astro.compute_rp(r1_vect, v1_vect, GM)
            if (periapsis_check or Mi > 0) and rp < R:
                continue
            
            v1_list.append(v1_vect)
            v2_list.append(v2_vect)
            M_list.append(Mi)
            lr_list.append(lri)
            type_list.append('retrograde')
            
            

    return v1_list, v2_list, M_list, type_list, lr_list


def find_xy(lam, T, maxiters=35, rtol=1e-8):
    '''
    This function computes all possible x,y values that fit the input orbit
    non-dimensional time of flight T.
    
    Parameters
    ------
    lam : float
        non-dimensional orbit parameter in range [-1, 1]
        lam > 0 for theta [0,pi] and lam < 0 for theta [pi, 2pi]
    T : float
        non-dimensional time of flight
    maxiters : int, optional
        maximum number of iterations for the Halley and Householder 
        root-finding steps (default=35)
    rtol : float, optional
        convergence tolerance for the Halley and Householder root-finding 
        steps (default=1e-8)
    
    Returns
    ------
    x_list : list
        all x values that fit boundary conditions
    y_list : list
        all y values that fit boundary conditions
    M_list : list
        all integer orbit revolutions that fit boundary conditions
    
    '''
    
    # Check inputs
    assert abs(lam) <= 1.
    assert T > 0.  # note error in Izzo paper
    
    # Initialize output
    x_list = []
    y_list = []
    M_list = []
    lr_list = []
    
    # Compute maximum number of complete revolutions that would fit in 
    # the simplest sense T_0M = T_00 + M*pi
    M_max = int(np.floor(T / math.pi))
    
#    print('lam', lam)
#    print('M_max', M_max)
    
    # Evaluate non-dimensional time of flight T
    # T(x=0) denoted T_0M
    # T_00 is T_0 for single revolution (M=0) case  
    # T_0M = T_00 + M*pi
    T_00 = math.acos(lam) + lam*np.sqrt(1. - lam**2.)
    
#    print('T_00', T_00)
    
    # Check if input T is less than minimum T required for M_max revolutions
    # If so, reduce M_max by 1
    if T < (T_00 + M_max*math.pi) and M_max > 0:
        
        # Start Halley iterations from x=0, T=To and find T_min(M_max)
        dum, T_min, exit_flag = compute_T_min(lam, M_max, maxiters, rtol)
        
        if T < T_min and exit_flag == 1:
            M_max -= 1
            
#    print('M_max', M_max)
            
    # Compute T(x=1) parabolic case (Izzo Eq 21)
    T_1 = (2./3.) * (1. - lam**3.)
    
    # Form initial guess for single revolution case (Izzo Eq 30)
    if T >= T_00:
        x_0 = (T_00/T)**(2./3.) - 1.

    elif T < T_1:
        x_0 = (5./2.) * (T_1*(T_1 - T))/(T*(1.-lam**5.)) + 1.

    else:        
        # Modified initial condition from poliastro
        # https://github.com/poliastro/poliastro/issues/1362
        x_0 = np.exp(np.log(2.) * np.log(T / T_00) / np.log(T_1 / T_00)) - 1.  
        
        
#    print('initial guess x_0', x_0)
        
    # Run Householder iterations for x_0 to get x,y for single rev case
    M = 0
    x, exit_flag = householder(x_0, T, lam, M, maxiters, rtol)
    
#    print('x', x)
#    print('exit_flag', exit_flag)
#    print('y', np.sqrt(1. - lam**2.*(1. - x**2.)))
    if exit_flag == 1:
        y = np.sqrt(1. - lam**2.*(1. - x**2.))
        x_list.append(x)
        y_list.append(y)
        M_list.append(M)
        lr_list.append('none')

    # Loop over M values and compute x,y using Householder iterations
    for M in range(1,M_max+1):        

        # Form initial x0_l and x0_r from Izzo Eq 31
        x0_l = (((M*math.pi + math.pi)/(8.*T))**(2./3.) - 1.) * \
            1./(((M*math.pi + math.pi)/(8.*T))**(2./3.) + 1.)
            
        x0_r = (((8.*T)/(M*math.pi))**(2./3.) - 1.) * \
            1./(((8.*T)/(M*math.pi))**(2./3.) + 1.)
    
        # Run Householder iterations for x0_l and x0_r for multirev cases
        xl, exit_flag = householder(x0_l, T, lam, M, maxiters, rtol)
        
        if exit_flag == 1:
            yl = np.sqrt(1. - lam**2.*(1. - xl**2.))
            x_list.append(xl)
            y_list.append(yl)
            M_list.append(M)
            lr_list.append('left')
        
        xr, exit_flag = householder(x0_r, T, lam, M, maxiters, rtol)
        
        if exit_flag == 1:
            yr = np.sqrt(1. - lam**2.*(1. - xr**2.))
            x_list.append(xr)
            y_list.append(yr)
            M_list.append(M)
            lr_list.append('right')

    return x_list, y_list, M_list, lr_list


def find_single_xy(lam, T, M_star, lr_star, maxiters=35, rtol=1e-8):
    '''
    This function computes all possible x,y values that fit the input orbit
    non-dimensional time of flight T and specific orbit revolution number 
    M_star.
    
    Parameters
    ------
    lam : float
        non-dimensional orbit parameter in range [-1, 1]
        lam > 0 for theta [0,pi] and lam < 0 for theta [pi, 2pi]
    T : float
        non-dimensional time of flight
    M_star : int
        exact integer number of complete orbit revolutions traversed
    maxiters : int, optional
        maximum number of iterations for the Halley and Householder 
        root-finding steps (default=35)
    rtol : float, optional
        convergence tolerance for the Halley and Householder root-finding 
        steps (default=1e-8)
    
    Returns
    ------
    x_list : list
        all x values that fit boundary conditions
    y_list : list
        all y values that fit boundary conditions
    M_list : list
        all integer orbit revolutions that fit boundary conditions
    
    '''
    
    # Check inputs
    assert abs(lam) <= 1.
    assert T > 0.  # note error in Izzo paper
    
    # Initialize output
    x_list = []
    y_list = []
    M_list = []
    lr_list = []
    
    # Compute maximum number of complete revolutions that would fit in 
    # the simplest sense T_0M = T_00 + M*pi
    M_max = int(np.floor(T / math.pi))
    if M_star > M_max:
        return x_list, y_list, M_list, lr_list
    
    # Evaluate non-dimensional time of flight T
    # T(x=0) denoted T_0M
    # T_00 is T_0 for single revolution (M=0) case  
    # T_0M = T_00 + M*pi
    T_00 = math.acos(lam) + lam*np.sqrt(1. - lam**2.)
            
    # Compute T(x=1) parabolic case (Izzo Eq 21)
    T_1 = (2./3.) * (1. - lam**3.)
    
    # Single revolution case
    if M_star == 0:
        
        # Form initial guess for single revolution case (Izzo Eq 30)
        if T >= T_00:
            x_0 = (T_00/T)**(2./3.) - 1.
    
        elif T < T_1:
            x_0 = (5./2.) * (T_1*(T_1 - T))/(T*(1.-lam**5.)) + 1.
    
        else:        
            # Modified initial condition from poliastro
            # https://github.com/poliastro/poliastro/issues/1362
            x_0 = np.exp(np.log(2.) * np.log(T / T_00) / np.log(T_1 / T_00)) - 1.        
            
        # Run Householder iterations for x_0 to get x,y for single rev case
        x, exit_flag = householder(x_0, T, lam, M_star, maxiters, rtol)
        if exit_flag == 1:
            y = np.sqrt(1. - lam**2.*(1. - x**2.))
            x_list.append(x)
            y_list.append(y)
            M_list.append(M_star)
            lr_list.append('none')
            
    # Multiple revolution case
    else:
        
        if lr_star == 'left':

            # Form initial x0_l from Izzo Eq 31
            x0_l = (((M_star*math.pi + math.pi)/(8.*T))**(2./3.) - 1.) * \
                1./(((M_star*math.pi + math.pi)/(8.*T))**(2./3.) + 1.)
        
            # Run Householder iterations for x0_l and x0_r for multirev cases
            xl, exit_flag = householder(x0_l, T, lam, M_star, maxiters, rtol)
            
            if exit_flag == 1:
                yl = np.sqrt(1. - lam**2.*(1. - xl**2.))
                x_list.append(xl)
                y_list.append(yl)
                M_list.append(M_star)
                lr_list.append('left')
            
        elif lr_star == 'right':
            
            # Form initial x0_r from Izzo Eq 31
            x0_r = (((8.*T)/(M_star*math.pi))**(2./3.) - 1.) * \
                1./(((8.*T)/(M_star*math.pi))**(2./3.) + 1.)
            
            xr, exit_flag = householder(x0_r, T, lam, M_star, maxiters, rtol)
            
            if exit_flag == 1:
                yr = np.sqrt(1. - lam**2.*(1. - xr**2.))
                x_list.append(xr)
                y_list.append(yr)
                M_list.append(M_star)
                lr_list.append('right')

    return x_list, y_list, M_list, lr_list


def compute_T_min(lam, M, maxiters=35, rtol=1e-8):
    '''
    This function computes the minimum non-dimensional time of flight for a
    given number of integer orbit revolutions.
    
    Parameters
    ------
    lam : float
        non-dimensional orbit parameter in range [-1, 1]
        lam > 0 for theta [0,pi] and lam < 0 for theta [pi, 2pi]
    M : int
        integer number of complete orbit revolutions traversed
    maxiters : int, optional
        maximum number of iterations for the Halley and Householder 
        root-finding steps (default=35)
    rtol : float, optional
        convergence tolerance for the Halley and Householder root-finding 
        steps (default=1e-8)
        
    Results
    ------
    x_T_min : float
        x value corresponding to T_min
    T_min : float
        minimum possible non-dimensional time of flight 
    exit_flag : int
        pass/fail criteria for Halley iterations (1 = pass, -1 = fail)    
    
    '''
    
    if lam == 1.:
        x_T_min = 0.
        T_min = compute_T(x_T_min, lam, M)
    else:
        if M == 0:
            x_T_min = np.inf
            T_min = 0.
        else:
            # Choose x_i > 0 to avoid problems at lam = -1
            x_i = 0.1
            T_i = compute_T(x_i, lam, M)
            x_T_min, exit_flag = halley(x_i, T_i, lam, M, maxiters, rtol)
            T_min = compute_T(x_T_min, lam, M)
            
#    print('x_T_min', x_T_min)
#    print('T_min', T_min)
#    print('exit_flag', exit_flag)
    
    return x_T_min, T_min, exit_flag


def halley(x0, T0, lam, M, maxiters=35, tol=1e-8):
    '''
    This function uses Halley iterations to solve the root finding problem to
    determine x that will minimize T.    
    
    Note, the poliastro code uses T0 to compute the derivates in each 
    iteration, while this code uses an updated value of T to do so. Izzo paper
    just says to start from T0, but I'm pretty sure updating T is the right
    way to proceed.
    
    Furthermore, poliastro Householder iterations do use the updated T value
    in each iteration, so makes sense to also do so here. The only application
    of the Halley iterations is to determine if M_max needs to be reduced by
    1, so it is pretty low consequence either way.
    
    Parameters
    ------
    x0 : float
        initial guess at x value
    T0 : float
        initial guess at T value    
    lam : float
        non-dimensional orbit parameter in range [-1, 1]
        lam > 0 for theta [0,pi] and lam < 0 for theta [pi, 2pi]
    M : int
        integer number of complete orbit revolutions traversed
    maxiters : int, optional
        maximum number of iterations for the Halley and Householder 
        root-finding steps (default=35)
    rtol : float, optional
        convergence tolerance for the Halley and Householder root-finding 
        steps (default=1e-8)
        
    Returns
    ------
    x : float
        x value corresponding to T_min
    exit_flag : int
        pass/fail criteria for Halley iterations (1 = pass, -1 = fail)   

    '''
    
#    print('Halley iters')
    
    diff = 1.
    iters = 0
    exit_flag = 1
    while diff > tol:
        
#        print('iters', iters)
#        print('x0', x0)
        
        T = compute_T(x0, lam, M)
        dT, ddT, dddT = compute_T_der(x0, T, lam)
        
        # Halley step, cubic
        x = x0 - 2.*dT*ddT/(2.*ddT**2. - dT*dddT)
        
#        print('x', x)
#        print('dT', dT)
#        print('ddT', ddT)
#        print('dddT', dddT)
        
        diff = abs(x - x0)
        x0 = float(x)
        iters += 1
        
        if iters > maxiters:
            exit_flag = -1
            break

    return x, exit_flag


def householder(x0, T_star, lam, M, maxiters=35, tol=1e-8):
    '''
    This function uses Halley iterations to solve the root-finding problem
    to find x corresponding to the input non-dimensional time of flight T_star.
    
    Parameters
    ------
    x0 : float
        initial guess at x value
    T_star : float
        desired value of T   
    lam : float
        non-dimensional orbit parameter in range [-1, 1]
        lam > 0 for theta [0,pi] and lam < 0 for theta [pi, 2pi]
    M : int
        integer number of complete orbit revolutions traversed
    maxiters : int, optional
        maximum number of iterations for the Halley and Householder 
        root-finding steps (default=35)
    rtol : float, optional
        convergence tolerance for the Halley and Householder root-finding 
        steps (default=1e-8)
        
    Returns
    ------
    x : float
        x value corresponding to T_star
    exit_flag : int
        pass/fail criteria for Halley iterations (1 = pass, -1 = fail) 
    '''
    
#    print('householder iterations')
    
    diff = 1.
    iters = 0
    exit_flag = 1
    while diff > tol:
        
#        print('iters', iters)
#        print('x0', x0)
        
        T_x = compute_T(x0, lam, M)
        f_x = T_x - T_star
        dT, ddT, dddT = compute_T_der(x0, T_x, lam)
        
        x = x0 - f_x*((dT**2. - f_x*ddT/2.)/
                      (dT*(dT**2. - f_x*ddT) + dddT*f_x**2./6.))
        
#        print('x', x)
#        print('T_x', T_x)
#        print('dT', dT)
#        print('ddT', ddT)
#        print('dddT', dddT)
    
        diff = abs(x - x0)
        x0 = float(x)
        iters += 1
        
        if iters > maxiters:
            exit_flag = -1
            break

    return x, exit_flag


def compute_T(x, lam, M):
    '''
    This function computes the non-dimensional time of flight T corresponding
    to x, lam, and M. The computations use Izzo Eq 18 and Eq 20 for the case x 
    close to 1 (determination of "close" condition from poliastro).
    
    Parameters
    ------
    x : float
        non-dimensional coordinate
    lam : float
        non-dimensional orbit parameter in range [-1, 1]
        lam > 0 for theta [0,pi] and lam < 0 for theta [pi, 2pi]
    M : int
        integer number of complete orbit revolutions traversed    
    
    Returns
    ------
    T : float
        non-dimensional time of flight    
    
    '''
    
    y = np.sqrt(1. - lam**2.*(1. - x**2.))
    
    # Izzo Eq 20
    if M == 0 and x > np.sqrt(0.6) and x < np.sqrt(1.4):
        eta = y - lam*x
        S_1 = 0.5 * (1. - lam - x*eta)
        Q = (4./3.) * compute_hypergeom_2F1(3., 1., 2.5, S_1)
        T = 0.5 * (eta**3.*Q + 4.*lam*eta)
        
    # Izzo Eq 18
    else:
        psi = compute_psi(x, y, lam)
        T = (1./(1. - x**2.)) * ((psi + M*math.pi)/
             np.sqrt(abs(1. - x**2.)) - x + lam*y)
    
    return T


def compute_T_der(x, T, lam):
    '''
    This function computes the first three derivatives of T(x) wrt x used for
    the root-finding functions. The formulas come from Izzo Eq 22 with the
    note that there are issues for issues for cases where lam^2 = 1, x = 0 
    and for x = 1.
    
    See LancasterBlanchard function (archive) for alternate forumlation and 
    exception handling.
    
    Parameters
    ------
    x : float
        non-dimensional coordinate
    T : float
        non-dimensional time of flight 
    lam : float
        non-dimensional orbit parameter in range [-1, 1]
        lam > 0 for theta [0,pi] and lam < 0 for theta [pi, 2pi]
    
    Returns
    ------
    dT : float
        first derivative dT/dx
    ddT : float
        second derivative d2T/dx2
    dddT : float
        third derivative d3t/dx3
    
    '''
    
    # Izzo Eq 22
    y    = np.sqrt(1. - lam**2.*(1. - x**2.))
    dT   = (1./(1. - x**2.)) * (3.*T*x - 2. + 2.*lam**3.*(x/y))
    ddT  = (1./(1. - x**2.)) * (3.*T + 5.*x*dT + 2.*(1. - lam**2.)*(lam**3./y**3.))
    dddT = (1./(1. - x**2.)) * (7.*x*ddT + 8.*dT - 6.*(1-lam**2.)*lam**5.*(x/y**5.))
    
    return dT, ddT, dddT


def compute_psi(x, y, lam):
    '''
    This function computes psi using Izzo Eq 17.
    
    Parameters
    ------
    x : float
        non-dimensional coordinate
    y : float
        non-dimensional coordinate
    lam : float
        non-dimensional orbit parameter in range [-1, 1]
        lam > 0 for theta [0,pi] and lam < 0 for theta [pi, 2pi]
        
    Returns
    ------
    psi : float
        auxiliary angle [rad]
    
    '''
    
    # Elliptic case
    if -1. <= x < 1.:
        cos_psi = x*y + lam*(1. - x**2.)
        psi = math.acos(cos_psi)
#        sin_psi = (y - x*lam)*np.sqrt(1. - x**2.)
#        psi = math.atan2(sin_psi, cos_psi)
        
    # Hyperbolic case
    elif x > 1.:
        sinh_psi = (y - x*lam)*np.sqrt(x**2. - 1.)
        psi = math.asinh(sinh_psi)
    
    # Parabolic
    else:
        psi = 0.
    
    return psi


def compute_hypergeom_2F1(a, b, c, d):
    '''
    Hypergeometric series function 2F1 adapted from poliastro code.
    
    Parameters
    ------
    a, b, c, d : float
        input numbers for series calculation, require |d| < 1
        
    Returns
    ------
    res : float
        hypergeometric series evaluated at given input values    
    
    Reference
    ------
    https://en.wikipedia.org/wiki/Hypergeometric_function
    
    '''    
    
    if d >= 1.0:
        return np.inf
    else:
        res = 1.
        term = 1.
        ii = 0
        while True:
            term = term * (a + ii) * (b + ii) / (c + ii) * d / (ii + 1)
            res_old = res
            res += term
            if res_old == res:
                return res
            ii += 1






###############################################################################
# Angles-Only IOD Methods
###############################################################################

def gooding_angles_iod(tk_list, Yk_list, sensor_id_list, sensor_params,
                       time_format='datetime', eop_alldata=[], XYs_df=[],
                       periapsis_check=True):
    '''
    
    
    '''
    
    gooding_start = time.time()
    
    GM = GME
    R = Re
    
    # Retrieve/load EOP and polar motion data if needed
    if len(eop_alldata) == 0:        
        eop_alldata = eop.get_celestrak_eop_alldata()
        
    if len(XYs_df) == 0:
        XYs_df = eop.get_XYs2006_alldata()            
    
    # Compute time parameters from given input
    if time_format == 'datetime':
        UTC_list = tk_list

    elif time_format == 'JD':
        JD_list = tk_list
        UTC_list = [timesys.jd2dt(JD) for JD in JD_list]
    
    # For each measurement, compute the associated sensor location and 
    # line of sight vector in ECI
    Lmat = np.zeros((3,3))
    Rmat = np.zeros((3,3))
    for kk in range(len(tk_list)):
        
        # Retrieve current values
        UTC = UTC_list[kk]
        Yk = Yk_list[kk]
        sensor_id = sensor_id_list[kk]
        site_ecef = sensor_params[sensor_id]['site_ecef']
        meas_types = sensor_params[sensor_id]['meas_types']
        
        # Compute sensor location in ECI
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        site_eci, dum = coord.itrf2gcrf(site_ecef, np.zeros((3,1)), UTC,
                                        EOP_data, XYs_df)
        
        # Compute measurement line of sight vector in ECI
        if 'ra' in meas_types and 'dec' in meas_types:
            ra = Yk[meas_types.index('ra')]
            dec = Yk[meas_types.index('dec')]
            
            rho_hat_eci = np.array([[math.cos(dec)*math.cos(ra)],
                                    [math.cos(dec)*math.sin(ra)],
                                    [math.sin(dec)]])
    
        elif 'az' in meas_types and 'el' in meas_types:
            az = Yk[meas_types.index('az')]
            el = Yk[meas_types.index('el')]
            
            rho_hat_enu = np.array([[math.cos(el)*math.sin(az)],
                                    [math.cos(el)*math.cos(az)],
                                    [math.sin(el)]])
    
            rho_hat_ecef = coord.enu2ecef(rho_hat_enu, site_ecef)
            
            rho_hat_eci, dum = coord.itrf2gcrf(rho_hat_ecef, np.zeros((3,1)),
                                               UTC, EOP_data, XYs_df)
       
        # Store values in columns of L and R
        Lmat[:,kk] = rho_hat_eci.flatten()
        Rmat[:,kk] = site_eci.flatten()
    
    # Time of flight
    tof = (UTC_list[-1] - UTC_list[0]).total_seconds()
    
    # Sensor and LOS vectors
    rho0_hat = Lmat[:,0].reshape(3,1)
    q0_vect = Rmat[:,0].reshape(3,1)
    rhof_hat = Lmat[:,-1].reshape(3,1)
    qf_vect = Rmat[:,-1].reshape(3,1)

    # Compute maximum number of revolutions that could occur during tof
    # assuming very low circular orbit
    M_max = compute_M_max(Lmat, Rmat, tof, GM, R)
    M_candidates = list(range(M_max+1))
    
    print('M_max', M_max)
    print('M_candidates', M_candidates)

    # Loop over possible values of orbit revolution number M
    Xo_output = []
    M_output = []
    rho0_output_list = []
    rhof_output_list = []
    M_list = []
    lr_list = []
    type_list = []
    multirev_time_list = []
    for M_star in M_candidates:
        
        print('\nM_star', M_star)
        
        if M_star == 0:
            
            
            # Try for a fast solution for single-rev cases
            
            start = time.time()
        
            # Prograde single revolution case
            lr_star = 'none'
            orbit_type = 'prograde'
            rho0_list, rhof_list = \
                M_star_to_3rho(Lmat, Rmat, UTC_list, tof, M_star, lr_star,
                               orbit_type, periapsis_check=periapsis_check,
                               step=1000., nfail_exit=True)
                
#            mistake
            
            # Build outputs
            nout = len(rho0_list)
            rho0_output_list.extend(rho0_list)
            rhof_output_list.extend(rhof_list)
            M_list.extend([M_star]*nout)
            lr_list.extend([lr_star]*nout)
            type_list.extend([orbit_type]*nout)
            
            single_rev_prograde_time = time.time() - start
            
            start = time.time()
            
            # Retrograde single revolution case
            lr_star = 'none'
            orbit_type = 'retrograde'
            rho0_list, rhof_list = \
                M_star_to_3rho(Lmat, Rmat, UTC_list, tof, M_star, lr_star,
                               orbit_type, periapsis_check=periapsis_check,
                               step=1000., nfail_exit=True)
            
            # Build outputs
            nout = len(rho0_list)
            rho0_output_list.extend(rho0_list)
            rhof_output_list.extend(rhof_list)
            M_list.extend([M_star]*nout)
            lr_list.extend([lr_star]*nout)
            type_list.extend([orbit_type]*nout)
            
            single_rev_retrograde_time = time.time() - start
            
            
            # If no single rev solutions found, try full search through all
            # range values
            if len(rho0_output_list) == 0:
                
                start = time.time()
        
                # Prograde single revolution case
                lr_star = 'none'
                orbit_type = 'prograde'
                rho0_list, rhof_list = \
                    M_star_to_3rho(Lmat, Rmat, UTC_list, tof, M_star, lr_star,
                                   orbit_type, periapsis_check=periapsis_check,
                                   step=1000., nfail_exit=False)
                
                # Build outputs
                nout = len(rho0_list)
                rho0_output_list.extend(rho0_list)
                rhof_output_list.extend(rhof_list)
                M_list.extend([M_star]*nout)
                lr_list.extend([lr_star]*nout)
                type_list.extend([orbit_type]*nout)
                
                single_rev_prograde_time += time.time() - start
                
                start = time.time()
                
                # Retrograde single revolution case
                lr_star = 'none'
                orbit_type = 'retrograde'
                rho0_list, rhof_list = \
                    M_star_to_3rho(Lmat, Rmat, UTC_list, tof, M_star, lr_star,
                                   orbit_type, periapsis_check=periapsis_check,
                                   step=1000., nfail_exit=False)
                
                # Build outputs
                nout = len(rho0_list)
                rho0_output_list.extend(rho0_list)
                rhof_output_list.extend(rhof_list)
                M_list.extend([M_star]*nout)
                lr_list.extend([lr_star]*nout)
                type_list.extend([orbit_type]*nout)
                
                single_rev_retrograde_time += time.time() - start
                
            
        else:
            
            start = time.time()
            
            # Prograde multi-rev case - left branch
            lr_star = 'left'
            orbit_type = 'prograde'
            rho0_list, rhof_list = \
                M_star_to_3rho(Lmat, Rmat, UTC_list, tof, M_star, lr_star,
                               orbit_type, periapsis_check=periapsis_check,
                               step=1000.)
            
            # Build outputs
            nout = len(rho0_list)
            rho0_output_list.extend(rho0_list)
            rhof_output_list.extend(rhof_list)
            M_list.extend([M_star]*nout)
            lr_list.extend([lr_star]*nout)
            type_list.extend([orbit_type]*nout)
            
            multirev_time_list.append(time.time() - start)
            
            start = time.time()
            
            # Prograde multi-rev case - right branch
            lr_star = 'right'
            orbit_type = 'prograde'
            rho0_list, rhof_list = \
                M_star_to_3rho(Lmat, Rmat, UTC_list, tof, M_star, lr_star,
                               orbit_type, periapsis_check=periapsis_check,
                               step=1000.)
            
            # Build outputs
            nout = len(rho0_list)
            rho0_output_list.extend(rho0_list)
            rhof_output_list.extend(rhof_list)
            M_list.extend([M_star]*nout)
            lr_list.extend([lr_star]*nout)
            type_list.extend([orbit_type]*nout)
            
            multirev_time_list.append(time.time() - start)
            
            start = time.time()
            
            # Retrograde multi-rev case - left branch
            lr_star = 'left'
            orbit_type = 'retrograde'
            rho0_list, rhof_list = \
                M_star_to_3rho(Lmat, Rmat, UTC_list, tof, M_star, lr_star,
                               orbit_type, periapsis_check=periapsis_check,
                               step=1000.)
            
            # Build outputs
            nout = len(rho0_list)
            rho0_output_list.extend(rho0_list)
            rhof_output_list.extend(rhof_list)
            M_list.extend([M_star]*nout)
            lr_list.extend([lr_star]*nout)
            type_list.extend([orbit_type]*nout)
            
            multirev_time_list.append(time.time() - start)
            
            start = time.time()
            
            # Retrograde multi-rev case - right branch
            lr_star = 'right'
            orbit_type = 'retrograde'
            rho0_list, rhof_list = \
                M_star_to_3rho(Lmat, Rmat, UTC_list, tof, M_star, lr_star,
                               orbit_type, periapsis_check=periapsis_check,
                               step=1000.)
            
            # Build outputs
            nout = len(rho0_list)
            rho0_output_list.extend(rho0_list)
            rhof_output_list.extend(rhof_list)
            M_list.extend([M_star]*nout)
            lr_list.extend([lr_star]*nout)
            type_list.extend([orbit_type]*nout)
            
            multirev_time_list.append(time.time() - start)
                
 
    # Compute solutions  
    for ii in range(len(rho0_output_list)):
        
        rho0 = rho0_output_list[ii]
        rhof = rhof_output_list[ii]
        M_ii = M_list[ii]
        lr_ii = lr_list[ii]
        type_ii = type_list[ii]
        
        r0_final = q0_vect + rho0*rho0_hat
        rf_final = qf_vect + rhof*rhof_hat
        v0_final_list, vf_final_list, M_final, type_final, lr = \
            izzo_lambert(r0_final, rf_final, tof, M_star=M_ii, 
                         lr_star=lr_ii, results_flag=type_ii,
                         periapsis_check=periapsis_check)

        v0_final = v0_final_list[0]
        Xo = np.concatenate((r0_final, v0_final), axis=0)
        elem0 = astro.cart2kep(Xo)
        
        print(r0_final)
        print(v0_final)
        print(elem0)

        # There should only be one solution with everything specified
        if len(M_final) > 1:
            print(v0_final_list)
            print(vf_final_list)
            print(M_final)
            print(type_final)
            mistake
            
        Xo_output.append(Xo)
        M_output.append(M_ii)
        
    gooding_total = time.time() - gooding_start
        
    print('')
    print('Execution times')
    print('single rev prograde', single_rev_prograde_time)
    print('single rev retrograde', single_rev_retrograde_time)
    print('M_candidates', M_candidates)
    print('multirev times', multirev_time_list)
    print('gooding total time', gooding_total)
    
    return Xo_output, M_output


def M_star_to_3rho(Lmat, Rmat, UTC_list, tof, M_star, lr_star, orbit_type,
                   periapsis_check=True, step=1000., nfail_exit=True):
    
    # Sensor and LOS vectors
    rho0_hat = Lmat[:,0].reshape(3,1)
    q0_vect = Rmat[:,0].reshape(3,1)
    rhof_hat = Lmat[:,-1].reshape(3,1)
    qf_vect = Rmat[:,-1].reshape(3,1)
    
    # Compute range search arrays
    rho0_array, rhof_array = compute_range_search_arrays(Lmat, Rmat, M_star, tof)
    range_pairs = itertools.product(rho0_array, rhof_array)
    range_pairs_list = [list(rp) for rp in range_pairs]
    nrange = len(range_pairs_list)
    range_ind = int(np.floor(nrange/2))
    increment = int(np.floor(nrange/4))
    
    while math.gcd(nrange, increment) != 1:
        increment += 1
    
#    range_ind = 0
#    increment = 1
    
    rho0_bounds = [rho0_array[0], rho0_array[-1]]
    rhof_bounds = [rhof_array[0], rhof_array[-1]]
    
#    # Compute bounds on range values for this value of M
#    rho0_bounds, rhof_bounds = compute_range_bounds(Lmat, Rmat, M_star, tof)
    
    # Form initial guess for rho0 and rhof and loop to find all solutions
#    rho0 = rho0_bounds[0]
#    rhof = rhof_bounds[0]
    rho0_lim = False
    rhof_lim = False
    rho0_output_list = []
    rhof_output_list = []
    ind_list = []
#    while len(rho0_output_list) < 3:
    
#    for range_pair in range_pairs_list:
    
    for ii in range(nrange):
        
        if range_ind in ind_list:
            print(ind_list)
            print(range_ind)
            mistake
        
        range_pair = range_pairs_list[range_ind]
        ind_list.append(range_ind)
        range_ind += increment
        range_ind = range_ind % nrange
        
        rho0 = float(range_pair[0])
        rhof = float(range_pair[1])
        
        print('rho0', rho0)
        print('rhof', rhof)
        
        # Attempt to solve Lambert's problem for this rho0/rhof pair
        r0_vect = q0_vect + rho0*rho0_hat
        rf_vect = qf_vect + rhof*rhof_hat
        
#        if M_star == 0:
#            v0_list, vf_list, M_list, type_list, lr_list = \
#            izzo_lambert(r0_vect, rf_vect, tof, M_star=M_star, lr_star='none',
#                         results_flag=results_flag,
#                         periapsis_check=periapsis_check)
#            
#        else:
#            v0_list, vf_list, M_list, type_list, lr_list = \
#            izzo_lambert(r0_vect, rf_vect, tof, M_star=M_star, lr_star='left',
#                         results_flag=results_flag,
#                         periapsis_check=periapsis_check)
#            
#            v0_list1, vf_list1, M_list1, type_list1, lr_list1 = \
#            izzo_lambert(r0_vect, rf_vect, tof, M_star=M_star, lr_star='right',
#                         results_flag=results_flag,
#                         periapsis_check=periapsis_check)
#            
#            v0_list.extend(v0_list1)
#            vf_list.extend(vf_list1)
#            M_list.extend(M_list1)
#            type_list.extend(type_list1)
#            lr_list.extend(lr_list1)
            
        
        v0_list, vf_list, M_list, type_list, lr_list = \
            izzo_lambert(r0_vect, rf_vect, tof, M_star=M_star, lr_star=lr_star,
                         results_flag=orbit_type,
                         periapsis_check=periapsis_check)

        # If there is a Lambert solution, iterate to find range solutions to
        # fit the middle observation(s)
        if len(M_list) > 0:
        
            # Lambert solver has returned exactly one solution, iterate to find
            # all possible range value pairs that fit
            rho0_output_list, rhof_output_list, exit_flag = \
                iterate_rho(rho0, rhof, tof, M_star, lr_star, orbit_type, Lmat,
                            Rmat, UTC_list, rho0_output_list, rhof_output_list,
                            rho0_bounds, rhof_bounds,
                            periapsis_check=periapsis_check)
            
            print('M_star_to_3rho')
            print(rho0_output_list)
            print(rhof_output_list)    
            
            # If successful or nfails exceeded, exit
            if len(rho0_output_list) > 0 or (nfail_exit and exit_flag == -1):
                break
                
#            # If successful, take the max values from solutions to continue
#            # range search loop
#            if exit_flag == 1:
#                rho0 = max(rho0_output_list)
#                rhof = max(rhof_output_list)
            
#        # Increment the current range values and continue the loop
#        if rho0 + step < rho0_bounds[-1]:
#            rho0 += step
#            continue
#        
#        else:
#            if not rho0_lim:
#                rho0 = rho0_bounds[-1]
#                rho0_lim = True
#                continue
#            
#            if rhof + step < rhof_bounds[-1]:
#                rhof += step
#                rho0 = rho0_bounds[0]
#                rho0_lim = False
#                continue
#            else:
#                if not rhof_lim:
#                    rhof = rhof_bounds[-1]
#                    rhof_lim = True
#                    rho0 = rho0_bounds[0]
#                    rho0_lim = False
#                    continue
#            
#        # If none of the above conditions are met, it means all possible
#        # values of rho0 and rhof have been attempted within the bounds
#        break

    return rho0_output_list, rhof_output_list


def iterate_rho(rho0_init, rhof_init, tof, M_star, lr_star, orbit_type, Lmat,
                Rmat, UTC_list, rho0_output_list, rhof_output_list,
                rho0_bounds, rhof_bounds, periapsis_check=True, HN=1.):
    '''
    
    Gooding 1993:
    We know that, even for a fixed value of k (in particular for k = 0 or 1), it is 0
common for there to be more than one solution to the angles-only problem. Thus, we have
seen (in section 2) that the general number of solutions for short-arc coverage is three.
    
    '''
    
    # Current guess
    rho0 = float(rho0_init)
    rhof = float(rhof_init)
    
    # Compute LOS unit vectors and sensor positions
    rho0_hat = Lmat[:,0].reshape(3,1)
    q0_vect = Rmat[:,0].reshape(3,1)    
    rhof_hat = Lmat[:,-1].reshape(3,1)
    qf_vect = Rmat[:,-1].reshape(3,1)
    
    # Gooding (1996) suggests 1e-5, orekit uses 1e-6
    finite_diff_step = 1e-6
    
    # Gooding (1996) suggests 1e-12, orekit uses 1e-14
    tol = 1e-14
    
    # Criteria for near-singular matrix, use geometric mean of NR and 
    # (Halley or mNR) solutions
    crit_gm = 1e-3
    
    conv_crit = 1.
    iters = 0
    maxiters = 100
    exit_flag = 0
    crit_min = 1.
    f_old = np.inf
    nfail = 0
    
    # Loop
    while len(rho0_output_list) < 3:
        
        print('\nstart loop')
        print('iters', iters)
        print('rho0', rho0)
        print('rhof', rhof)

        # Check exit condition
        if nfail > 4:
            exit_flag = -1
            break
        
        if len(rho0_output_list) == 0 and nfail > 4:
            exit_flag = -1
            break
        
        # Check bad rho values
        if rho0 < 0 or rhof < 0:
            nfail += 1
            iters = 0
            rho0, rhof = modify_start_rho(Lmat, Rmat, nfail, rho0, rhof,
                                          rho0_bounds, rhof_bounds)
            
            print('nfail', nfail)
            print('rho below zero')
            print('rho0', rho0)
            print('rhof', rhof)
            
            continue
        
        # Check for converge on previous solution
        restart_flag = False
        for ii in range(len(rho0_output_list)):
            rho0_diff = rho0 - rho0_output_list[ii]
            rhof_diff = rhof - rhof_output_list[ii]
            
            if np.sqrt(rho0_diff**2. + rhof_diff**2.) > 1.:
                continue
            else:                
                nfail += 1
                iters = 0
                restart_flag = True
                rho0, rhof = modify_start_rho(Lmat, Rmat, nfail, rho0, rhof,
                                          rho0_bounds, rhof_bounds)
            
                print('nfail', nfail)
                print('converge on previous')
                print('rho0', rho0)
                print('rhof', rhof)
                print('rho0_diff', rho0_diff)
                print('rhof_diff', rhof_diff)                
                
                break
        
        if restart_flag:
            continue
            
        
        
    
        # Solve Lambert problem to get LOS vector at intermediate time
        rhok_list, rhok_inds = \
            compute_intermediate_rho(rho0, rhof, tof, M_star, lr_star, 
                                     orbit_type, Lmat, Rmat, UTC_list,
                                     periapsis_check=periapsis_check)
            
        print('len rhok_list', len(rhok_list))
            
        # Error check
        if len(rhok_list) == 0:
            
            nfail += 1
            iters = 0
            
            rho0, rhof = modify_start_rho(Lmat, Rmat, nfail, rho0, rhof,
                                          rho0_bounds, rhof_bounds)
            
            print('nfail', nfail)
            print('no Lambert solution')
            print('rho0', rho0)
            print('rhof', rhof)
            
            continue
            
        # Assume a single intermediate point for now
        rhok_calc_vect = rhok_list[0]
        rk_vect = Rmat[:,1].reshape(3,1) + rhok_calc_vect
        rk = np.linalg.norm(rk_vect)
        
        # Construct basis vectors
        rhok_obs_hat = Lmat[:,1].reshape(3,1)
        u_vect = np.cross(rhok_obs_hat, rhok_calc_vect, axis=0)
        
        # Solution can converge on rhok_calc_hat pointing 180 degrees away
        # from rhok_obs_hat - check and reset as needed
        rhok_dot = float(np.dot(rhok_obs_hat.T, rhok_calc_vect))
        if rhok_dot/np.linalg.norm(rhok_calc_vect) < -0.99:
            nfail += 1
            iters = 0
            rho0, rhof = modify_start_rho(Lmat, Rmat, nfail, rho0, rhof,
                                          rho0_bounds, rhof_bounds)
            
            print('nfail', nfail)
            print('rho0', rho0)
            print('rhof', rhof)
            print(rhok_dot)
            print(np.linalg.norm(rhok_calc_vect))
            
            continue
        
        # Exit condition (rhok_obs_hat = rhok_calc_hat)
        if np.linalg.norm(u_vect) == 0.:
            rho0_output_list.append(rho0)
            rhof_output_list.append(rhof)
            
            # Gooding 1996 Section 3.5 update to initial guess
            rho0_prev = float(rho0)
            rhof_prev = float(rhof)
            rho0 = max(2*rho0 - rho0_init, rho0_bounds[0])
            rhof = max(2*rhof - rhof_init, rhof_bounds[0])
            rho0_init = rho0_prev
            rhof_init = rhof_prev  
            iters = 0
            nfail = 0
            crit_min = 1.
            exit_flag = 1
            
            continue
        
        p_vect = np.cross(u_vect, rhok_obs_hat, axis=0)
        p_hat = p_vect/np.linalg.norm(p_vect)
        en_vect = np.cross(rhok_obs_hat, p_hat, axis=0)
        en_hat = en_vect/np.linalg.norm(en_vect)
        
        # Compute basic f and g penalty functions
        f = float(np.dot(p_hat.T, rhok_calc_vect))
        g = float(np.dot(en_hat.T, rhok_calc_vect))
        fc = float(f)
        
        # Update fc to account for previous solutions
        epsilon_list = []
        eta_list = []
        beta_list = []
        gamma_list = []
        for ii in range(len(rho0_output_list)):
            rho0_ii = rho0_output_list[ii]
            rhof_ii = rhof_output_list[ii]
            r0 = np.linalg.norm(q0_vect + rho0_ii*rho0_hat)
            rf = np.linalg.norm(qf_vect + rhof_ii*rhof_hat)
            
            epsilon = rho0 - rho0_ii
            eta = rhof - rhof_ii
            beta = np.sqrt(epsilon**2. + eta**2.)
            gamma = np.sqrt(beta**2. + r0**2. + rf**2.)
            
            
            fc *= gamma/beta
            
            epsilon_list.append(epsilon)
            eta_list.append(eta)
            beta_list.append(beta)
            gamma_list.append(gamma)

            
            
        # Error Check on step size
        # fc_old
        
        
        
        
        
        
        # Compute penalty
#        f = float(np.dot(p_hat.T, rhok_calc_vect))
#        g = float(np.dot(en_hat.T, rhok_calc_vect))
#        f, g = compute_penalty(rhok_calc_vect, rho0, rhof, p_hat, en_hat,
#                               rho0_output_list, rhof_output_list, Lmat, Rmat)
        
        
        print('f', f)
        print('g', g)
        
        # Use central finite difference to compute numerical derivatives of f
        # and g with respect to small changes in rho0 and rhof
        drho0 = rho0 * finite_diff_step
        drhof = rhof * finite_diff_step
        drho02 = drho0**2.
        drhof2 = drhof**2.
        
        print('drho0', drho0)
        print('drhof', drhof)
        
        # Range rho0 minus delta_rho
        rho0_minus = rho0 - drho0
        rhok_list, rhok_inds = \
            compute_intermediate_rho(rho0_minus, rhof, tof, M_star, lr_star,
                                     orbit_type, Lmat, Rmat, UTC_list, 
                                     periapsis_check=periapsis_check)
            
        cm0 = rhok_list[0]
        
        
        
#        fm_rho0, gm_rho0 = compute_penalty(cm0, rho0, rhof, p_hat, en_hat,
#                                           rho0_output_list, rhof_output_list,
#                                           Lmat, Rmat)
        
        
        fm_rho0 = float(np.dot(p_hat.T, cm0))
        gm_rho0 = float(np.dot(en_hat.T, cm0))
        
        # Range rho0 plus delta_rho
        rho0_plus = rho0 + drho0
        rhok_list, rhok_inds = \
            compute_intermediate_rho(rho0_plus, rhof, tof, M_star, lr_star, 
                                     orbit_type, Lmat, Rmat, UTC_list,
                                     periapsis_check=periapsis_check)
            
        cp0 = rhok_list[0]
#        fp_rho0, gp_rho0 = compute_penalty(cp0, rho0, rhof, p_hat, en_hat,
#                                           rho0_output_list, rhof_output_list,
#                                           Lmat, Rmat)
        fp_rho0 = float(np.dot(p_hat.T, cp0))
        gp_rho0 = float(np.dot(en_hat.T, cp0))
        
        # Range rhof minus delta_rho
        rhof_minus = rhof - drhof
        rhok_list, rhok_inds = \
            compute_intermediate_rho(rho0, rhof_minus, tof, M_star, lr_star,
                                     orbit_type, Lmat, Rmat, UTC_list,
                                     periapsis_check=periapsis_check)
            
        cmf = rhok_list[0]
#        fm_rhof, gm_rhof = compute_penalty(cmf, rho0, rhof, p_hat, en_hat,
#                                           rho0_output_list, rhof_output_list,
#                                           Lmat, Rmat)
        fm_rhof = float(np.dot(p_hat.T, cmf))
        gm_rhof = float(np.dot(en_hat.T, cmf))
        
        # Range rhof plus delta_rho
        rhof_plus = rhof + drhof
        rhok_list, rhok_inds = \
            compute_intermediate_rho(rho0, rhof_plus, tof, M_star, lr_star,
                                     orbit_type, Lmat, Rmat, UTC_list,
                                     periapsis_check=periapsis_check)
            
        cpf = rhok_list[0]
#        fp_rhof, gp_rhof = compute_penalty(cpf, rho0, rhof, p_hat, en_hat,
#                                           rho0_output_list, rhof_output_list,
#                                           Lmat, Rmat)
        fp_rhof = float(np.dot(p_hat.T, cpf))
        gp_rhof = float(np.dot(en_hat.T, cpf))
        
        # Multivariate partial (rho0_plus and rhof_plus)
        rhok_list, rhok_inds = \
            compute_intermediate_rho(rho0_plus, rhof_plus, tof, M_star, lr_star,
                                     orbit_type, Lmat, Rmat, UTC_list,
                                     periapsis_check=periapsis_check)
            
        cp0f = rhok_list[0]
        
        fp_rho0f = float(np.dot(p_hat.T, cp0f))
        gp_rho0f = float(np.dot(en_hat.T, cp0f))
        
        
        # Compute derivatives
        df_drho0 = (fp_rho0 - fm_rho0)/(2.*drho0)
        df_drhof = (fp_rhof - fm_rhof)/(2.*drhof)
        dg_drho0 = (gp_rho0 - gm_rho0)/(2.*drho0)
        dg_drhof = (gp_rhof - gm_rhof)/(2.*drhof)
        
        d2f_drho02 = (fp_rho0 + fm_rho0 - 2.*f)/drho02
        d2g_drho02 = (gp_rho0 + gm_rho0 - 2.*g)/drho02
        d2f_drhof2 = (fp_rhof + fm_rhof - 2.*f)/drhof2
        d2g_drhof2 = (gp_rhof + gm_rhof - 2.*g)/drhof2
        
        d2f_drho0f = (fp_rho0f - f)/(drho0*drhof) - (df_drho0/drhof + df_drhof/drho0) \
            - 0.5*(d2f_drho02*(drho0/drhof) + d2f_drhof2*(drhof/drho0))
             
        d2g_drho0f = (gp_rho0f - g)/(drho0*drhof) - (dg_drho0/drhof + dg_drhof/drho0) \
            - 0.5*(d2g_drho02*(drho0/drhof) + d2g_drhof2*(drhof/drho0))
            
        # Error Check - if any of the finite difference steps did not return
        # a valid lambert solution, decrease step by factor of 10 and repeat
        # up to 3 times
        
        
        # Compute adjusted derivatives accounting for known solutions
        for ii in range(len(rho0_output_list)):
            epsilon = epsilon_list[ii]
            eta = eta_list[ii]
            beta = beta_list[ii]
            gamma = gamma_list[ii]
            
            w = 1./beta**2. - 1./gamma**2.
            uw = w - (2./beta**2. + 2./gamma**2.)
            w0 = epsilon*w
            wf = eta*w
            
            df_drho0 -= f*w0
            df_drhof -= f*wf
            
            d2f_drho02 -= 2.*df_drho0*w0 + w*f*(1. + epsilon**2.*uw)
            d2g_drho02 -= 2.*dg_drho0*w0
            d2f_drhof2 -= 2.*df_drhof*wf + w*f*(1. + eta**2.*uw)
            d2g_drhof2 -= 2.*dg_drhof*wf
            
            d2f_drho0f -= df_drhof*w0 + df_drho0*wf + w*f*epsilon*eta*uw
            d2g_drho0f -= dg_drhof*w0 + dg_drho0*wf
            
            
        # Compute Newton-Raphson increments
        D_NR = df_drho0*dg_drhof - df_drhof*dg_drho0
        
        delta_rho0_NR = -(1./D_NR) * f * dg_drhof
        delta_rhof_NR =  (1./D_NR) * f * dg_drho0
        
        print('D_NR', D_NR)
        print('delta_rho0_NR', delta_rho0_NR)
        print('delta_rhof_NR', delta_rhof_NR)
        
        # Compute Halley/mNR derivatives
        # If HN = 0.5 use Halley formula
        # If HN = 1.0 use modifed Newton-Raphson formula
        # Use of modified Newton-Raphson should be more robust in case of 
        # neighboring solutions
        df_drho0_H = df_drho0 + HN*(d2f_drho02*delta_rho0_NR + d2f_drho0f*delta_rhof_NR)
        df_drhof_H = df_drhof + HN*(d2f_drho0f*delta_rho0_NR + d2f_drhof2*delta_rhof_NR)
        dg_drho0_H = dg_drho0 + HN*(d2g_drho02*delta_rho0_NR + d2g_drho0f*delta_rhof_NR)
        dg_drhof_H = dg_drhof + HN*(d2g_drho0f*delta_rho0_NR + d2g_drhof2*delta_rhof_NR)
        
        D_H = df_drho0_H*dg_drhof_H - df_drhof_H*dg_drho0_H
        
        delta_rho0 = -(1./D_H) * f * dg_drhof_H
        delta_rhof =  (1./D_H) * f * dg_drho0_H
        
        # Check for near singular derivative matrix
        H = df_drho0**2. + df_drhof**2. + dg_drho0**2. + dg_drhof**2.
        dd = 2.*abs(D_NR)/(H + np.sqrt(H**2. - 4.*D_NR**2.))
        
        # If below threshold, use geometric mean of NR and (Halley or mNR)
        if dd < crit_gm:
            delta_rho0 = np.sign(delta_rho0_NR) * np.sqrt(abs(delta_rho0_NR*delta_rho0))
            delta_rhof = np.sign(delta_rhof_NR) * np.sqrt(abs(delta_rhof_NR*delta_rhof))
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#        mat = np.array([[df_drho0, df_drhof],
#                        [dg_drho0, dg_drhof]])
#        
#        print('derivatives')
#        print('df_drho0', df_drho0)
#        print('df_drhof', df_drhof)
#        print('dg_drho0', dg_drho0)
#        print('dg_drhof', dg_drhof)
#        
#        
#        # Newton's Method
#        # Compute determinant D and changes for rho0 and rhof
#        D = df_drho0*dg_drhof - df_drhof*dg_drho0
#        
#        delta_rho0 = -(1./D) * f * dg_drhof
#        delta_rhof =  (1./D) * f * dg_drho0
#        
#        print('D', D)
#        print('delta_rho0', delta_rho0)
#        print('delta_rhof', delta_rhof)
#        
#        print('mat', mat)
#        print('check D', np.linalg.det(mat))
#
#        
#        delta2 = -np.dot(np.linalg.inv(mat), np.reshape([f, 0.], (2,1)))
#        print('delta2', delta2)

        rho0 += delta_rho0
        rhof += delta_rhof
        
#        rho0 += float(delta2[0])
#        rhof += float(delta2[1])
        
#        # Check for overshoot and adjust
#        if iters > 1 and f > 2.*f_old:
#            rho0 = (f*rho0_old + f_old*rho0)/(f + f_old)
#            rhof = (f*rhof_old + f_old*rhof)/(f + f_old)
#            print('overshoot')
        
        print('rho0', rho0)
        print('rhof', rhof)
        
        # If multiple solutions already found, use original f to check 
        # convergence criteria to be consistent for all solutions
        if len(rho0_output_list) == 0:
            fconv = f
        else:
            fconv, dum = compute_penalty(rhok_calc_vect, rho0, rhof, p_hat,
                                         en_hat, [], [], Lmat, Rmat)
        
        conv_crit = abs(fconv)/max(rk, rhok_dot)
        print('conv_crit', conv_crit)
        print('denom', rk, rhok_dot)
        
        
        # For converged solution, store answer and update initial guess
        if conv_crit < tol:
            rho0_output_list.append(rho0)
            rhof_output_list.append(rhof)
            
            # Gooding 1996 Section 3.5 update to initial guess
            rho0_prev = float(rho0)
            rhof_prev = float(rhof)
            rho0 = max(2*rho0 - rho0_init, rho0_bounds[0])
            rhof = max(2*rhof - rhof_init, rhof_bounds[0])
            rho0_init = rho0_prev
            rhof_init = rhof_prev
            iters = 0
            nfail = 0
            crit_min = 1.
            exit_flag = 1
            print('rho0_output_list', rho0_output_list)
            print('rhof_output_list', rhof_output_list)
            
            continue        
        
        if conv_crit < crit_min:
            crit_min = float(conv_crit)
            
        # Store values for future comparison
        f_old = float(fc)
        rho0_old = float(rho0)
        rhof_old = float(rhof)
        
        print('rho0_output_list', rho0_output_list)
        print('rhof_output_list', rhof_output_list)
        
        # Increment counter and exit condition
        iters += 1
        if iters > maxiters:
            break
    
    print('crit_min', crit_min)

    
    return rho0_output_list, rhof_output_list, exit_flag


def compute_M_max(Lmat, Rmat, tof, GM=GME, R=Re):

    # Minimum orbit radius
    rmin = R + 100.
    
    # Vectors
    rho0_hat = Lmat[:,0].reshape(3,1)
    q0_vect = Rmat[:,0].reshape(3,1)
    rho0_min = radius2rho(rmin, rho0_hat, q0_vect)
    r0_vect = q0_vect + rho0_min*rho0_hat
    
    print(rho0_hat)
    print(q0_vect)
    print(rho0_min)
    print(r0_vect)
    
    rhof_hat = Lmat[:,-1].reshape(3,1)
    qf_vect = Rmat[:,-1].reshape(3,1)
    rhof_min = radius2rho(rmin, rhof_hat, qf_vect)
    rf_vect = qf_vect + rhof_min*rhof_hat
    
    print(rhof_hat)
    print(qf_vect)
    print(rhof_min)
    print(rf_vect)
        
    # Compute chord and minimum energy ellipse
    c_vect = rf_vect - r0_vect    
    r0 = np.linalg.norm(r0_vect)
    rf = np.linalg.norm(rf_vect)
    c = np.linalg.norm(c_vect)
    
    # Compute values to reconstruct output
    s = 0.5 * (r0 + rf + c)
    
    # Test for minimum energy ellipse (s = 2*a_min) must be valid orbit
    if s < 2.*(R + 100.):
        s = 2.*(R + 100.)
    
    # Compute non-dimensional time of flight T
    T = np.sqrt(2*GM/s**3.) * tof
    
    print('r0', r0)
    print('rf', rf)
    print('c', c)
    print('s', s)
    print('T', T)
    
    # Compute maximum number of orbit revolutions
    M_max = int(np.floor(T/math.pi))
    
    return M_max


def compute_range_search_arrays(Lmat, Rmat, M_star, tof, step=1000., 
                                rp=Re+100., GM=GME):
    '''
    
    
    '''
    
    # Vectors
    rho0_hat = Lmat[:,0].reshape(3,1)
    q0_vect = Rmat[:,0].reshape(3,1)
    
    rhof_hat = Lmat[:,-1].reshape(3,1)
    qf_vect = Rmat[:,-1].reshape(3,1)
    
    # Compute minimum ranges
    rmin = rp
    rho0_min = radius2rho(rmin, rho0_hat, q0_vect)
    rhof_min = radius2rho(rmin, rhof_hat, qf_vect)
    
    # Single revolution
    if M_star == 0:
        
        # Max range limited by optical detection limit
        rho0_max = 50000.
        rhof_max = 50000.
        
    
    # Orbit parameters for multi-rev cases
    if M_star > 0:
        
        # Compute orbit parameters for extreme case to get upper bound on 
        # orbit radius and range
        n_min = (M_star/tof)*2.*math.pi
        a_max = (GM/n_min**2.)**(1./3.)
        e_max = 1. - (rp/a_max)        
        
        rmin = rp
        rmax = a_max*(1. + e_max)
        
        print('M_star', M_star)
        print('a_max', a_max)
        print('e_max', e_max)
        print('rmax', rmax)
        
        rho0_max = radius2rho(rmax, rho0_hat, q0_vect)
        rhof_max = radius2rho(rmax, rhof_hat, qf_vect)
        
        
    print('rho0_min', rho0_min)
    print('rho0_max', rho0_max)
    print('rhof_min', rhof_min)
    print('rhof_max', rhof_max)
        
    
    rho0_array = np.arange(rho0_min, rho0_max, step)
    rho0_array = np.append(rho0_array, rho0_max)
    rhof_array = np.arange(rhof_min, rhof_max, step)
    rhof_array = np.append(rhof_array, rhof_max)
    
    return rho0_array, rhof_array


def compute_range_bounds(Lmat, Rmat, M_star, tof, rp=Re+100., GM=GME):
    '''
    
    
    '''
    
    # Vectors
    rho0_hat = Lmat[:,0].reshape(3,1)
    q0_vect = Rmat[:,0].reshape(3,1)
    
    rhof_hat = Lmat[:,-1].reshape(3,1)
    qf_vect = Rmat[:,-1].reshape(3,1)
    
    # Compute minimum ranges
    rmin = rp
    rho0_min = radius2rho(rmin, rho0_hat, q0_vect)
    rhof_min = radius2rho(rmin, rhof_hat, qf_vect)
    
    # Single revolution
    if M_star == 0:
        
        # Max range limited by optical detection limit
        rho0_max = 50000.
        rhof_max = 50000.
        
    
    # Orbit parameters for multi-rev cases
    if M_star > 0:
        
        # Compute orbit parameters for extreme case to get upper bound on 
        # orbit radius and range
        n_min = (M_star/tof)*2.*math.pi
        a_max = (GM/n_min**2.)**(1./3.)
        e_max = 1. - (rp/a_max)        
        
        rmin = rp
        rmax = a_max*(1. + e_max)
        
        print('M_star', M_star)
        print('a_max', a_max)
        print('e_max', e_max)
        print('rmax', rmax)
        
        rho0_max = radius2rho(rmax, rho0_hat, q0_vect)
        rhof_max = radius2rho(rmax, rhof_hat, qf_vect)
        
        
    print('rho0_min', rho0_min)
    print('rho0_max', rho0_max)
    print('rhof_min', rhof_min)
    print('rhof_max', rhof_max)
        
    
    rho0_bounds = [rho0_min, rho0_max]
    rhof_bounds = [rhof_min, rhof_max]
    
    return rho0_bounds, rhof_bounds


def radius2rho(r, rho_hat_eci, site_eci):
    '''
    This function computes range value to yield a given orbit radius
    
    Parameters
    ------
    r : float
        orbit radius
    rho_hat_eci : 3x1 numpy array
        LOS unit vector in ECI
    site_eci : 3x1 numpy array
        sensor position vector in ECI [km]    
        
    Returns
    ------
    rho : float
        range value corresponding to specified orbit radius        
    
    '''
    
    a = 1.
    b = float(2.*np.dot(rho_hat_eci.T, site_eci))
    c = float(np.dot(site_eci.T, site_eci)) - r**2.
    
    rho = float((-b + np.sqrt(b**2. - 4.*a*c))/(2.*a))
    
    return rho






def compute_penalty(rhok_vect, rho0, rhof, p_hat, en_hat, rho0_list, rhof_list,
                    Lmat, Rmat):    
    '''
    
    
    '''
    
    # Compute basic f and g penalty functions
    f = float(np.dot(p_hat.T, rhok_vect))
    g = float(np.dot(en_hat.T, rhok_vect))
    
    # If previous solutions exist, modify f and g to avoid converging on same
    if len(rho0_list) > 0:
        
        # Compute LOS unit vectors and sensor positions
        rho0_hat = Lmat[:,0].reshape(3,1)
        q0_vect = Rmat[:,0].reshape(3,1)    
        rhof_hat = Lmat[:,-1].reshape(3,1)
        qf_vect = Rmat[:,-1].reshape(3,1)
    
    
        for ii in range(len(rho0_list)):
            rho0_ii = rho0_list[ii]
            rhof_ii = rhof_list[ii]
            r0 = np.linalg.norm(q0_vect + rho0_ii*rho0_hat)
            rf = np.linalg.norm(qf_vect + rhof_ii*rhof_hat)
            
            epsilon = rho0 - rho0_ii
            eta = rhof - rhof_ii
            beta = np.sqrt(epsilon**2. + eta**2.)
            gamma = np.sqrt(beta**2. + r0**2. + rf**2.)
            
            f *= gamma/beta
            g *= gamma/beta
    
    return f, g


def modify_start_rho(Lmat, Rmat, nfail, rho0, rhof, rho0_bounds, rhof_bounds):
    '''
    
    
    '''
    
#    # Compute LOS unit vectors and sensor positions
#    rho0_hat = Lmat[:,0].reshape(3,1)
#    q0_vect = Rmat[:,0].reshape(3,1)    
#    rhof_hat = Lmat[:,-1].reshape(3,1)
#    qf_vect = Rmat[:,-1].reshape(3,1)
#    
#    # Compute updated guess for rho0, rhof
#    if nfail == 1:
#        rho0 = max(-float(np.dot(q0_vect.T, rho0_hat)), 1000.)
#        rhof = max(-float(np.dot(qf_vect.T, rhof_hat)), 1000.)
#        
#    if nfail == 2:
#        
#        q0f_vect = qf_vect - q0_vect
#        D1 = np.dot(q0f_vect.T, rho0_hat)
#        D3 = np.dot(q0f_vect.T, rhof_hat)
#        D2 = np.dot(rho0_hat.T, rhof_hat)
#        D4 = 1. - D2**2.
#        
#        rho0 = max((D1-(D3*D2))/D4, 100.)
#        rhof = max(((D1*D2)-D3)/D4, 100.)
    
    rho0_mid = np.mean(rho0_bounds)
    rhof_mid = np.mean(rhof_bounds)
    
    
    if nfail == 1:
        rho0 = rho0_mid
        rhof = rhof_mid
        
    if nfail == 2:
        rho0 = rho0_mid
        rhof = float(rhof_bounds[1])
    
    if nfail == 3:
        rho0 = float(rho0_bounds[1])
        rhof = rhof_mid
        
    if nfail == 4:
        rho0 = float(rho0_bounds[1])
        rhof = float(rhof_bounds[1])
    
    return rho0, rhof




#def compute_delta_rho(rho0, rhof, tof, M_star, orbit_type, Lmat, Rmat,
#                      UTC_list):
#    
#    '''
#    
#    
#    '''
#    
#    # Basic Gooding assume 3 angles and use middle value to compute penalty
#    kk_list = [1]
#    
#    
#    # Compute range vectors at all intermediate times in kk_list
#    rhok_list, rhok_inds = \
#        compute_intermediate_rho(rho0, rhof, tof, M_star, orbit_type, Lmat,
#                                 Rmat, UTC_list, kk_list)
#        
#    
#    
#    print('rhok_list', rhok_list)
#    
#    # Exit condition
#    if len(rhok_list) == 0:
#        fail_flag = True
#        return 0., 0., fail_flag
#    else: 
#        fail_flag = False
#    
#    # Compute penalty for these values    
#    f, g = compute_penalty(rhok_list, rhok_inds, Lmat, Rmat, kk_list)
#    
#    print('f', f)
#    print('g', g)
#    
#
#    
#    # Use central finite difference to compute numerical derivatives of f
#    # with respect to small changes in rho0 and rhof
#    drho0 = 1e-8
#    drhof = 1e-8
#    
#    # First range rho0 minus delta_rho
#    rho0_minus = rho0 - drho0
#    rhok_list, rhok_inds = \
#        compute_intermediate_rho(rho0_minus, rhof, tof, M_star, orbit_type,
#                                 Lmat, Rmat, UTC_list, kk_list)
#    
#    fm0, gm0 = compute_penalty(rhok_list, rhok_inds, Lmat, Rmat, kk_list)
#    
#    # First range rho0 plus delta_rho
#    rho0_plus = rho0 + drho0
#    rhok_list, rhok_inds = \
#        compute_intermediate_rho(rho0_plus, rhof, tof, M_star, orbit_type,
#                                 Lmat, Rmat, UTC_list, kk_list)
#       
#    fp0, gp0 = compute_penalty(rhok_list, rhok_inds, Lmat, Rmat, kk_list)
#    
#    # Last range rhof minus delta_rho
#    rhof_minus = rhof - drhof
#    rhok_list, rhok_inds = \
#        compute_intermediate_rho(rho0, rhof_minus, tof, M_star, orbit_type,
#                                 Lmat, Rmat, UTC_list, kk_list)
#    
#    fmf, gmf = compute_penalty(rhok_list, rhok_inds, Lmat, Rmat, kk_list)
#    
#    # Last range rhof plus delta_rho
#    rhof_plus = rhof + drhof
#    rhok_list, rhok_inds = \
#        compute_intermediate_rho(rho0, rhof_plus, tof, M_star, orbit_type,
#                                 Lmat, Rmat, UTC_list, kk_list)
#       
#    fpf, gpf = compute_penalty(rhok_list, rhok_inds, Lmat, Rmat, kk_list)
#    
#    
#    # Compute numerical derivatives using central finite difference
#    df_0 = (fp0 - fm0)/(2.*drho0)
#    dg_0 = (gp0 - gm0)/(2.*drho0)      
#    df_f = (fpf - fmf)/(2.*drhof)
#    dg_f = (gpf - gmf)/(2.*drhof)  
#    
#    print('df_0', df_0)
#    print('dg_0', dg_0)
#    print('df_f', df_f)
#    print('dg_f', dg_f)
#    
#    mat = np.array([[df_0, df_f],
#                    [dg_0, dg_f]])
#    
#    check = -np.dot(np.linalg.inv(mat), np.reshape([f, 0.], (2,1)))
#    print('check', check)
#    
#    # Compute determinant D and changes for rho0 and rhof
#    D = df_0*dg_f - df_f*dg_0    
#    
#    delta_rho0 = -(1./D) * f * dg_f
#    delta_rhof =  (1./D) * f * dg_0
#    
#    print('delta_rho0', delta_rho0)
#    print('delta_rhof', delta_rhof)
#    
#    print('matrix det', np.linalg.det(mat))
#    print('D', D)
#    
#    return delta_rho0, delta_rhof, fail_flag
#
#
def compute_intermediate_rho(rho0, rhof, tof, M_star, lr_star, orbit_type,
                             Lmat, Rmat, UTC_list, periapsis_check=False):
    
    '''
    
    '''
    
    kk_list = [1]
    
    # Compute initial and final position vectors
    rho0_hat = Lmat[:,0].reshape(3,1)
    q0_vect = Rmat[:,0].reshape(3,1)    
    rhof_hat = Lmat[:,-1].reshape(3,1)
    qf_vect = Rmat[:,-1].reshape(3,1)
    
    r0_vect = q0_vect + rho0*rho0_hat
    rf_vect = qf_vect + rhof*rhof_hat
    
    # Compute Lambert solution for these inputs
    v0_list, vf_list, M_list, type_list, lr_list = \
        izzo_lambert(r0_vect, rf_vect, tof, M_star=M_star, lr_star=lr_star,
                     results_flag=orbit_type, periapsis_check=periapsis_check)
    
    # There should only be one solution with everything specified
    if len(M_list) > 1:
        print(v0_list)
        print(vf_list)
        print(M_list)
        print(type_list)
        mistake
        
    # If no valid solutions are found, exit    
    if len(v0_list) == 0:
        return [], []
        
    v0_vect = v0_list[0]
    
    # Full cartesian state vector at t0
    Xo = np.concatenate((r0_vect, v0_vect), axis=0)
    
    # Loop over intermediate times and compute rho_vect
    rhok_list = []   
    rhok_inds = []
    for kk in kk_list:
    
        dt_sec = (UTC_list[kk] - UTC_list[0]).total_seconds()
        qk_vect = Rmat[:,kk].reshape(3,1)
        Xk = astro.element_conversion(Xo, 1, 1, dt=dt_sec)
        rk_vect = Xk[0:3].reshape(3,1)
        rhok_vect = rk_vect - qk_vect
    
        rhok_list.append(rhok_vect)
        rhok_inds.append(kk)
    
    return rhok_list, rhok_inds
#
#
#def compute_penalty(rhok_list, rhok_inds, Lmat, Rmat, kk_list):
#    '''
#    
#    
#    '''
#    
#    for kk in kk_list:
#    
#        # Observed LOS unit vector at time tk
#        rhok_hat_obs = Lmat[:,kk].reshape(3,1)
#        
#        # Calcuated LOS range vector at time tk
#        rhok_vect_calc = rhok_list[rhok_inds.index(kk)]
#        rhok_calc = np.linalg.norm(rhok_vect_calc)
#        
#        # Compute penalty function
#        en_vect = np.cross(rhok_hat_obs, rhok_vect_calc, axis=0)
#        p_vect = np.cross(en_vect, rhok_hat_obs, axis=0)
#        f = float(np.dot(p_vect.T, rhok_vect_calc)/rhok_calc)
#        g = float(np.dot(en_vect.T, rhok_vect_calc)/np.linalg.norm(en_vect))
#    
#    return f, g






def gauss_angles_iod(tk_list, Yk_list, sensor_id_list, sensor_params,
                     time_format='datetime', eop_alldata=[], XYs_df=[]):
    '''
    This function implements the Gauss angles-only IOD method, which uses
    three line of sight vectors (defined by RA/DEC or Az/El measurements) to
    compute an orbit. The method is not particularly accurate, does not 
    include perturbations, is not appropriate for multiple revolutions between
    observations, etc. It works best for angular separations less than 60 deg,
    ideally less than 10 deg. In general, other methods such as Gooding or 
    double-r iteration should be used instead.
    
    Parameters
    ------
    tk_list : list
        observation times in JD or datetime object format [UTC]
    Yk_list : list
        measurements (RA/DEC or Az/El pairs) [rad]
    sensor_id_list : list
        sensor id corresponding to each measurement in Yk_list
    sensor_params : dict
        includes site_ecef and meas_types for each sensor used
    time_format : string, optional
        defines format of input tk_list ('JD' or 'datetime') 
        (default='datetime')
    eop_alldata : string, optional
        text containing EOP data, if blank, function will retrieve from
        celestrak.com
    XYs_df : dataframe, optional
        pandas dataframe containing polar motion data, if blank, function will
        read from file in input_data directory
                
    Returns
    ------
    UTC2 : datetime object
        time of second observation [UTC]
    r2_vect : 3x1 numpy array
        inertial position vector at t2 [km]
    v2_vect : 3x1 numpy array
        inertial velocity vector at t2 [km/s]
    exit_flag : int
        pass/fail criteria for Halley iterations (1 = pass, -1 = fail)  
        
    
    References
    ------
    [1] Vallado, D., "Fundamentals of Astrodynamics and Applications," 4th ed,
        2013. (Algorithm 52)
    
    '''
    
    # Constants
    GM = GME
    
    # Retrieve/load EOP and polar motion data if needed
    if len(eop_alldata) == 0:        
        eop_alldata = eop.get_celestrak_eop_alldata()
        
    if len(XYs_df) == 0:
        XYs_df = eop.get_XYs2006_alldata()            
    
    # Compute time parameters from given input
    if time_format == 'datetime':
        UTC_list = tk_list

    elif time_format == 'JD':
        JD_list = tk_list
        UTC_list = [timesys.jd2dt(JD) for JD in JD_list]
        
    # Output time
    UTC2 = UTC_list[1]
    
    # For each measurement, compute the associated sensor location and 
    # line of sight vector in ECI
    Lmat = np.zeros((3,3))
    Rmat = np.zeros((3,3))
    for kk in range(len(tk_list)):
        
        # Retrieve current values
        UTC = UTC_list[kk]
        Yk = Yk_list[kk]
        sensor_id = sensor_id_list[kk]
        site_ecef = sensor_params[sensor_id]['site_ecef']
        meas_types = sensor_params[sensor_id]['meas_types']
        
        # Compute sensor location in ECI
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        site_eci, dum = coord.itrf2gcrf(site_ecef, np.zeros((3,1)), UTC,
                                        EOP_data, XYs_df)
        
        # Compute measurement line of sight vector in ECI
        if 'ra' in meas_types and 'dec' in meas_types:
            ra = Yk[meas_types.index('ra')]
            dec = Yk[meas_types.index('dec')]
            
            rho_hat_eci = np.array([[math.cos(dec)*math.cos(ra)],
                                    [math.cos(dec)*math.sin(ra)],
                                    [math.sin(dec)]])
    
        elif 'az' in meas_types and 'el' in meas_types:
            az = Yk[meas_types.index('az')]
            el = Yk[meas_types.index('el')]
            
            rho_hat_enu = np.array([[math.cos(el)*math.sin(az)],
                                    [math.cos(el)*math.cos(az)],
                                    [math.sin(el)]])
    
            rho_hat_ecef = coord.enu2ecef(rho_hat_enu, site_ecef)
            
            rho_hat_eci, dum = coord.itrf2gcrf(rho_hat_ecef, np.zeros((3,1)),
                                               UTC, EOP_data, XYs_df)
            
        
        # Store values in columns of L and R
        Lmat[:,kk] = rho_hat_eci.flatten()
        Rmat[:,kk] = site_eci.flatten()
         
    # Calculations to set up root-finding problem
    tau1 = (UTC_list[0] - UTC_list[1]).total_seconds()
    tau3 = (UTC_list[2] - UTC_list[1]).total_seconds() 
    
    a1 =  tau3/(tau3 - tau1)
    a3 = -tau1/(tau3 - tau1)
    a1u =  tau3*((tau3 - tau1)**2. - tau3**2.)/(6.*(tau3 - tau1))
    a3u = -tau1*((tau3 - tau1)**2. - tau1**2.)/(6.*(tau3 - tau1))
    
    M = np.dot(np.linalg.inv(Lmat), Rmat)
    d1 = M[1,0]*a1 - M[1,1] + M[1,2]*a3
    d2 = M[1,0]*a1u + M[1,2]*a3u
    
    # LOS and site ECI vectors
    L1_vect = Lmat[:,0].reshape(3,1)
    R1_vect = Rmat[:,0].reshape(3,1)
    L2_vect = Lmat[:,1].reshape(3,1)
    R2_vect = Rmat[:,1].reshape(3,1)
    L3_vect = Lmat[:,2].reshape(3,1)
    R3_vect = Rmat[:,2].reshape(3,1)    
    
    C = float(np.dot(L2_vect.T, R2_vect))
    
    # Solve for r2
    poly2 = np.array([1., 0., -(d1**2. + 2.*C*d1 + np.linalg.norm(R2_vect)**2.),
                      0., 0., -2.*GM*(C*d2 + d1*d2), 0., 0., -GM**2.*d2**2.])
    roots2 = np.roots(poly2)
    
    # Find positive real roots
    real_inds = list(np.where(np.isreal(roots2))[0])
    r2_list = [roots2[ind] for ind in real_inds if roots2[ind] > 0.]
        
    if len(r2_list) != 1:
        exit_flag = -1
    
    r2 = float(np.real(r2_list[0]))
    
    # Solve for position vectors
    u = GM/(r2**3.)
    c1 = a1 + a1u*u
    c2 = -1.
    c3 = a3 + a3u*u

    c_vect = -np.array([[c1], [c2], [c3]])
    crho_vect = np.dot(M, c_vect)
    rho1 = float(crho_vect[0])/c1
    rho2 = float(crho_vect[1])/c2
    rho3 = float(crho_vect[2])/c3
    
    # Note that method can be iterated to improve performance
    
    # Compute position vectors for each time
    r1_vect = rho1*L1_vect + R1_vect
    r2_vect = rho2*L2_vect + R2_vect
    r3_vect = rho3*L3_vect + R3_vect
    
    # Try Gibbs to compute v2_vect
    v2_vect, gibbs_exit = gibbs_iod(r1_vect, r2_vect, r3_vect, GM)

    # If Gibbs failed, try Herrick-Gibbs
    if not (gibbs_exit == 1):
        v2_vect, hg_exit = herrick_gibbs_iod(r1_vect, r2_vect, r3_vect, 
                                             UTC_list, GM)

    # Exit condition
    if (gibbs_exit != 1) and (hg_exit != 1):
        exit_flag = -1
    else:
        exit_flag = 1
    
    return UTC2, r2_vect, v2_vect, exit_flag


def gibbs_iod(r1_vect, r2_vect, r3_vect, GM=GME):
    '''
    This function solves Gibbs Problem, which finds a Keplerian orbit solution
    to fit three, time-sequential (t1 < t2 < t3), co-planar position vectors. 
    The method outputs the velocity vector at the middle time.
    
    Note this applies in single revolution cases only and does not include
    perturbing forces. The method can be used in the final step of Gauss 
    angles-only IOD provided the angular separation between the position 
    vectors is large enough. In particular, if the separation is less than 1 
    degree the method does not work well and Herrick-Gibbs should be used 
    instead.
    
    Parameters
    ------
    r1_vect : 3x1 numpy array
        inertial position vector at time t1 [km]
    r2_vect : 3x1 numpy array
        inertial position vector at time t2 [km]
    r3_vect : 3x1 numpy array
        inertial position vector at time t3 [km]
    GM : float, optional
        gravitational parameter (default=GME) [km^3/s^2]
        
    Returns
    ------
    v2_vect : 3x1 numpy array
        inertial velocity vector at time t2 [km/s]
    exit_flag : int
        pass/fail criteria for Halley iterations (1 = pass, -1 = fail)  
    
    References
    ------
    [1] Vallado, D., "Fundamentals of Astrodynamics and Applications," 4th ed,
        2013. (Algorithm 54)
    
    '''
    
    # Initialize output
    v2_vect = np.zeros((3,1))
    
    # Compute cross products
    z12_vect = np.cross(r1_vect, r2_vect, axis=0)
    z23_vect = np.cross(r2_vect, r3_vect, axis=0)
    z31_vect = np.cross(r3_vect, r1_vect, axis=0)
    
    # Compute vector magnitudes
    r1 = np.linalg.norm(r1_vect)
    r2 = np.linalg.norm(r2_vect)
    r3 = np.linalg.norm(r3_vect)
    z23 = np.linalg.norm(z23_vect)
    
    # Test for coplanar
    alpha_cop = math.pi/2. - math.acos(float(np.dot(z23_vect.T, r1_vect))/(z23*r1))
    alpha_12 = math.acos(float(np.dot(r1_vect.T, r2_vect))/(r1*r2))
    alpha_23 = math.acos(float(np.dot(r2_vect.T, r3_vect))/(r2*r3))

    if abs(alpha_cop) > 5.*math.pi/180.:
        exit_flag = -1
        return v2_vect, exit_flag
        
    if (alpha_12 < 5.*math.pi/180.) and (alpha_23 < 5.*math.pi/180.):
        exit_flag = -2
        return v2_vect, exit_flag
        
    print(z23_vect)
    print(alpha_cop*180/math.pi)
    print(alpha_12*180/math.pi)
    print(alpha_23*180/math.pi)
        
    # Compute vectors    
    N = r1*z23_vect + r2*z31_vect + r3*z12_vect
    D = z12_vect + z23_vect + z31_vect
    S = (r2 - r3)*r1_vect + (r3 - r1)*r2_vect + (r1 - r2)*r3_vect
    B = np.cross(D, r2_vect, axis=0)
    
    Lg = np.sqrt(GM/(np.linalg.norm(N)*np.linalg.norm(D)))
    
    v2_vect = Lg/r2*B + Lg*S
    exit_flag = 1
    
    return v2_vect, exit_flag


def herrick_gibbs_iod(r1_vect, r2_vect, r3_vect,UTC_list, GM=GME):
    '''
    This function solves Gibbs Problem for small angles to finds a Keplerian 
    orbit solution to fit three, time-sequential (t1 < t2 < t3), co-planar 
    position vectors. The method outputs the velocity vector at the middle 
    time.
    
    Note this applies in single revolution cases only and does not include
    perturbing forces. The method can be used in the final step of Gauss 
    angles-only IOD provided the angular separation between the position 
    vectors is small, and thus serves as a complement to Gibbs method. 
    In particular, if the separation is less than 1 degree the method 
    outperforms Gibbs.
    
    Parameters
    ------
    r1_vect : 3x1 numpy array
        inertial position vector at time t1 [km]
    r2_vect : 3x1 numpy array
        inertial position vector at time t2 [km]
    r3_vect : 3x1 numpy array
        inertial position vector at time t3 [km]
    UTC_list : list
        datetime objects corresponding to t1, t2, t3 [UTC]
    GM : float, optional
        gravitational parameter (default=GME) [km^3/s^2]
        
    Returns
    ------
    v2_vect : 3x1 numpy array
        inertial velocity vector at time t2 [km/s]
    exit_flag : int
        pass/fail criteria for Halley iterations (1 = pass, -1 = fail)  
    
    References
    ------
    [1] Vallado, D., "Fundamentals of Astrodynamics and Applications," 4th ed,
        2013. (Algorithm 55)
    
    '''
    
    # Initialize output
    v2_vect = np.zeros((3,1))
    
    # Compute time differences
    dt31 = (UTC_list[2] - UTC_list[0]).total_seconds()
    dt32 = (UTC_list[2] - UTC_list[1]).total_seconds()
    dt21 = (UTC_list[1] - UTC_list[0]).total_seconds()
    
    # Compute cross products
    z23_vect = np.cross(r2_vect, r3_vect, axis=0)
    
    # Compute vector magnitudes
    r1 = np.linalg.norm(r1_vect)
    r2 = np.linalg.norm(r2_vect)
    r3 = np.linalg.norm(r3_vect)
    z23 = np.linalg.norm(z23_vect)
    
    # Test for coplanar
    alpha_cop = math.pi/2. - math.acos(float(np.dot(z23_vect.T, r1_vect))/(z23*r1))
    alpha_12 = math.acos(float(np.dot(r1_vect.T, r2_vect))/(r1*r2))
    alpha_23 = math.acos(float(np.dot(r2_vect.T, r3_vect))/(r2*r3))

    if abs(alpha_cop) > 5.*math.pi/180.:
        exit_flag = -1
        return v2_vect, exit_flag
    
    if (alpha_12 > 5.*math.pi/180.) or (alpha_23 > 5.*math.pi/180.):
        exit_flag = -2
        return v2_vect, exit_flag
    
    print(z23_vect)
    print(alpha_cop*180/math.pi)
    print(alpha_12*180/math.pi)
    print(alpha_23*180/math.pi)
       
    # Compute velocity vector
    v2_vect = -dt32*((1./(dt21*dt31)) + GM/(12.*r1**3.))*r1_vect \
        + (dt32 - dt21)*((1./(dt21*dt32)) + GM/(12.*r2**3.))*r2_vect \
        + dt21*((1./(dt32*dt31)) + GM/(12.*r3**3.))*r3_vect
        
    exit_flag = 1
    
    
    
    return v2_vect, exit_flag








