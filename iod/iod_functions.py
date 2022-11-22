import numpy as np
import math
from scipy.integrate import odeint, ode
import sys
import os
import inspect
from datetime import datetime, timedelta
import time
import itertools
import random

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
    lr_star : string, optional
        flag to indicate whether to use left branch or right branch solution
        in multirev cases ('left', 'right', or 'none'). Use 'none for single
        rev cases.
        (default='none')
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
# Gooding Angles-Only IOD
###############################################################################

def gooding_angles_iod(tk_list, Yk_list, sensor_id_list, sensor_params,
                       time_format='datetime', eop_alldata=[], XYs_df=[],
                       orbit_regime='none', search_mode='middle_out',
                       periapsis_check=True, HN=1., rootfind='zeros'):
    '''    
    This function implements the Gooding angles-only IOD method, which uses
    three or more line of sight vectors (defined by RA/DEC or Az/El 
    measurements) to compute an orbit. The method is based on a grid search 
    through possible range values corresponding to measurements at the first 
    and last time. Any range pair that produces a valid solution to Lambert's
    problem is iterated until converging on an orbit that matches observations
    at intermediate times, within some tolerance. As written, the method does
    not account for perturbing forces, but can be incorporated in a higher 
    level algorithm which does.
    
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
    orbit_regime : string, optional
        flag to select solutions only from a specific orbit regime, which will
        place restrictions on the range grid search and improve computational
        performance ('LEO', 'MEO', 'HEO', 'GEO', or 'none') (default='none')
    search_mode : string, optional
        flag to indicate way to proceed through range grid which may improve
        computational performance ('bottom_up', 'middle_out', or 'random')
    periapsis_check : boolean, optional
        flag to determine whether to check the orbit does not intersect the
        central body (rp > R) (default=True)
    HN : float, optional
        control parameter to use either Halley (HN=0.5) or 
        modified Newton-Raphson (HN=1.0) to compute update to range values
        (default=1.0)
    rootfind : string, optional
        control parameter to either find zeros ('zeros') or minimum ('min') of 
        penalty function (default='zeros')
                    
    Returns
    ------
    Xo_output : list
        Cartesian state vectors at initial time [km, km/s]
    M_output : list
        integer orbit revolution numbers corresponding to solution state 
        vectors
    
    References
    ------
    [1] Gooding, R.H., "A New Procedure for the Solution of the Classical 
        Problem of Minimal Orbit Determination from Three Lines of Sight,"
        CMDA, 1997.    
    [2] Gooding, R.H., "A New Procedure for Orbit Determination Based on Three
        Lines of Sight (Angles Only)," DRA/RAE Technical Report 93004, 1993.
        
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
    M_max = compute_M_max(Lmat, Rmat, tof, GM, R, orbit_regime)
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
        
        # Compute the range pairs to use as initial guesses
        range_pairs_list, orbit_regime_fail = \
            compute_range_search_list(Lmat, Rmat, M_star, tof,
                                      orbit_regime=orbit_regime)
            
        # If no viable solutions for the given orbit regime and revolution
        # number, try next value
        if orbit_regime_fail:
            continue
        
        # Get the indices to use to retrieve range pairs
        range_ind_list = compute_range_ind_list(range_pairs_list, search_mode)
        
        if M_star == 0:
            
            
            # Try for a fast solution for single-rev cases
            
            start = time.time()
        
            # Prograde single revolution case
            lr_star = 'none'
            orbit_type = 'prograde'
            print(lr_star, orbit_type)
            
            rho0_list, rhof_list = \
                M_star_to_3rho(Lmat, Rmat, UTC_list, tof, M_star, lr_star,
                               orbit_type, range_pairs_list, range_ind_list,
                               periapsis_check=periapsis_check, HN=HN,
                               rootfind=rootfind)
                
            
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
            print(lr_star, orbit_type)
            
            rho0_list, rhof_list = \
                M_star_to_3rho(Lmat, Rmat, UTC_list, tof, M_star, lr_star,
                               orbit_type, range_pairs_list, range_ind_list,
                               periapsis_check=periapsis_check, HN=HN,
                               rootfind=rootfind)
            
            # Build outputs
            nout = len(rho0_list)
            rho0_output_list.extend(rho0_list)
            rhof_output_list.extend(rhof_list)
            M_list.extend([M_star]*nout)
            lr_list.extend([lr_star]*nout)
            type_list.extend([orbit_type]*nout)
            
            single_rev_retrograde_time = time.time() - start

            
        else:
            
            start = time.time()
            
            # Prograde multi-rev case - left branch
            lr_star = 'left'
            orbit_type = 'prograde'
            print(lr_star, orbit_type)
            
            rho0_list, rhof_list = \
                M_star_to_3rho(Lmat, Rmat, UTC_list, tof, M_star, lr_star,
                               orbit_type, range_pairs_list, range_ind_list,
                               periapsis_check=periapsis_check, HN=HN,
                               rootfind=rootfind)
            
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
            print(lr_star, orbit_type)
            
            rho0_list, rhof_list = \
                M_star_to_3rho(Lmat, Rmat, UTC_list, tof, M_star, lr_star,
                               orbit_type, range_pairs_list, range_ind_list,
                               periapsis_check=periapsis_check, HN=HN,
                               rootfind=rootfind)
            
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
            print(lr_star, orbit_type)
            
            rho0_list, rhof_list = \
                M_star_to_3rho(Lmat, Rmat, UTC_list, tof, M_star, lr_star,
                               orbit_type, range_pairs_list, range_ind_list,
                               periapsis_check=periapsis_check, HN=HN,
                               rootfind=rootfind)
            
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
            print(lr_star, orbit_type)
            
            rho0_list, rhof_list = \
                M_star_to_3rho(Lmat, Rmat, UTC_list, tof, M_star, lr_star,
                               orbit_type, range_pairs_list, range_ind_list,
                               periapsis_check=periapsis_check, HN=HN,
                               rootfind=rootfind)
            
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
                   range_pairs_list, range_ind_list, periapsis_check=True,
                   HN=1., rootfind='zeros'):
    '''
    This function iteratively adjusts range values corresponding to the first
    and last angles-only observations until finding an orbit that matches the
    angles-only observation(s) of intermediate time(s) within some tolerance.
    The function takes an input integer revolution number, for which there are
    a maximum of three valid solutions (pairs of rho0 and rhof values). The 
    function will search through a set of initial guesses at rho0/rhof values
    until finding a pair that produces a valid solution to Lambert's problem,
    which is then iterated to find final values. If a solution is found or the
    iteration fails to find a solution, the process continues with the next
    initial guess at rho0/rhof. Once three valid solutions are found, the 
    function returns to continue the process with the next valid value of
    integer revolution number.
    
    Parameters
    ------
    Lmat : 3xN numpy array
        columns correspond to line of sight unit vectors in ECI for each
        observation time
    Rmat : 3xN numpy array
        columns correspond to sensor location vectors in ECI for each 
        observation time [km]
    UTC_list : list
        datetime objects corresponding to observation times in UTC
    tof : float
        time of flight from first to last observation in seconds
    M_star : int
        integer number of orbit revolutions
    lr_star : string
        flag to indicate whether to use left branch or right branch solution
        in multirev cases ('left', 'right', or 'none'). Use 'none for single
        rev cases.
    orbit_type : string
        flag to indicate 'prograde' or 'retrograde' orbit
    range_pairs_list : list
        each entry is a list of [rho0, rhof] values to use as initial guess
    range_ind_list : list
        list of indices into range_pairs_list, used to allow different orders
        to proceed through range pair guesses (e.g. low to high, or random)
        for possible computational improvement
    periapsis_check : boolean, optional
        flag to determine whether to check the orbit does not intersect the
        central body (rp > R) (default=True)
    HN : float, optional
        control parameter to use either Halley (HN=0.5) or 
        modified Newton-Raphson (HN=1.0) to compute update to range values
        (default=1.0)
    rootfind : string, optional
        control parameter to either find zeros ('zeros') or minimum ('min') of 
        penalty function (default='zeros')
        
    Returns
    ------
    rho0_output_list : list
        range values at initial time that yield orbit solutions matching 
        intermediate observations [km]
    rhof_output_list : list
        range values at final time that yield orbit solutions matching 
        intermediate observations [km]
    
    '''
    
    # Sensor and LOS vectors
    rho0_hat = Lmat[:,0].reshape(3,1)
    q0_vect = Rmat[:,0].reshape(3,1)
    rhof_hat = Lmat[:,-1].reshape(3,1)
    qf_vect = Rmat[:,-1].reshape(3,1)
    
    # Loop over initial range guesses, if a pair rho0,rhof produces a valid 
    # orbit solution, iterate to find orbit that matches intermediate 
    # observation(s)
    rho0_output_list = []
    rhof_output_list = []
    nrange = len(range_pairs_list)
    for ii in range(nrange):
        
        # Retrieve next pair of range values
        range_ind = range_ind_list[ii]
        rho0 = float(range_pairs_list[range_ind][0])
        rhof = float(range_pairs_list[range_ind][1])
        
        print('')
        print('M_star_to_3rho')
        print('ii', ii)
        print('range_ind', range_ind)
        print('rho0', rho0)
        print('rhof', rhof)
        
        # Attempt to solve Lambert's problem for this rho0/rhof pair
        r0_vect = q0_vect + rho0*rho0_hat
        rf_vect = qf_vect + rhof*rhof_hat
        
        v0_list, vf_list, M_list, type_list, lr_list = \
            izzo_lambert(r0_vect, rf_vect, tof, M_star=M_star, lr_star=lr_star,
                         results_flag=orbit_type,
                         periapsis_check=periapsis_check)

        # If there is a Lambert solution, iterate to find range solutions to
        # fit the middle observation(s)
        if len(M_list) > 0:
        
            # Lambert solver has returned exactly one solution, iterate to find
            # range values that fit middle observation(s)
            rho0_output_list, rhof_output_list = \
                iterate_rho(rho0, rhof, tof, M_star, lr_star, orbit_type, Lmat,
                            Rmat, UTC_list, rho0_output_list, rhof_output_list,
                            periapsis_check=periapsis_check, HN=HN,
                            rootfind=rootfind)
            
            print('M_star_to_3rho')
            print(rho0_output_list)
            print(rhof_output_list)            
            
        # Per Gooding (1993) P6 and P19, a maximum of 3 solutions exist for any
        # given value of half-revolution number k. In this code, the full orbit
        # revolution number M is use, then subdivided by prograde/retrograde 
        # for single rev cases and left/right branch for multirev cases.
        # Therefore, if three solutions have been found then exit.
        if len(rho0_output_list) > 2:
            break

    return rho0_output_list, rhof_output_list


def iterate_rho(rho0_init, rhof_init, tof, M_star, lr_star, orbit_type, Lmat,
                Rmat, UTC_list, rho0_output_list, rhof_output_list,
                periapsis_check=True, HN=1., rootfind='zeros'):
    '''
    This function iteratively updates the initial and final range values until
    converging on an orbit that matches the intermediate angles-only 
    observation(s) within some tolerance.
    
    Parameters
    ------
    rho0_init : float
        range at initial time [km]
    rhof_init : float
        range at final time [km]    
    tof : float
        time of flight from the first to last observation [sec]
    M_star : int
        exact integer number of complete orbit revolutions traversed
    lr_star : string
        flag to indicate whether to use left branch or right branch solution
        in multirev cases ('left', 'right', or 'none'). Use 'none for single
        rev cases.
    orbit_type : string
        flag to indicate 'prograde' or 'retrograde' orbit
    Lmat : 3xN numpy array
        columns correspond to line of sight unit vectors in ECI for each
        observation time
    Rmat : 3xN numpy array
        columns correspond to sensor location vectors in ECI for each 
        observation time [km]    
    UTC_list : list
        datetime objects corresponding to observation times in UTC
    rho0_output_list : list
        converged solution values of initial range [km]
    rhof_output_list : list
        converged solution values of final range [km]
    periapsis_check : boolean, optional
        flag to determine whether to check the orbit does not intersect the
        central body (rp > R) (default=True)
    HN : float, optional
        control parameter to use either Halley (HN=0.5) or 
        modified Newton-Raphson (HN=1.0) to compute update to range values
        (default=1.0)
    rootfind : string, optional
        control parameter to either find zeros ('zeros') or minimum ('min') of 
        penalty function (default='zeros')
        
    Returns
    ------
    rho0_output_list : list
        converged solution values of initial range [km]
    rhof_output_list : list
        converged solution values of final range [km]

    '''
    
    # Current guess
    rho0 = float(rho0_init)
    rhof = float(rhof_init)
    
    # Compute LOS unit vectors and sensor positions
    rho0_hat = Lmat[:,0].reshape(3,1)
    q0_vect = Rmat[:,0].reshape(3,1)    
    rhof_hat = Lmat[:,-1].reshape(3,1)
    qf_vect = Rmat[:,-1].reshape(3,1)
    
    # Gooding (1997) suggests 1e-12, orekit uses 1e-14
    tol = 1e-14
    
    # Criteria for near-singular matrix, use geometric mean of NR and 
    # (Halley or mNR) solutions
    # Ref Gooding 1997 p.395
    crit_gm = 1e-3
    
    # Iterate in loop until converging on rho0,rhof pair that matches 
    # intermediate observation(s)
    conv_crit = 1.
    iters = 0
    maxiters = 100
    exit_flag = 0
    crit_min = 1.
    f_old = np.inf
    nfail = 0
    while len(rho0_output_list) < 3:
        
        # Gooding (1997) suggests 1e-5, orekit uses 1e-6
        finite_diff_step = 1e-6
        
        print('\nstart loop')
        print('iters', iters)
        print('rho0', rho0)
        print('rhof', rhof)
        
        # Check exit condition
        if iters > maxiters:
            break

#        # Check exit condition
#        if nfail > 4:
#            exit_flag = -1
#            break
#        
#        if len(rho0_output_list) == 0 and nfail > 4:
#            exit_flag = -1
#            break
        
        # Exception Handling
        # If invalid range value, exit and continue range grid search
        if rho0 < 0 or rhof < 0:

            print('rho out of range')
            print('rho0', rho0)
            print('rhof', rhof)
            
            break
        
        # Exception Handling
        # If converge on a previous solution, exit and continue range grid 
        # search
        restart_flag = False
        for ii in range(len(rho0_output_list)):
            rho0_diff = rho0 - rho0_output_list[ii]
            rhof_diff = rhof - rhof_output_list[ii]
            
            if np.sqrt(rho0_diff**2. + rhof_diff**2.) > 1.:
                continue
            else:                
                restart_flag = True

                print('converge on previous')
                print('rho0', rho0)
                print('rhof', rhof)
                print('rho0_diff', rho0_diff)
                print('rhof_diff', rhof_diff)                
                
                break
        
        if restart_flag:
            break

        # Solve Lambert problem to get LOS vector at intermediate time
        rhok_list, rhok_inds = \
            compute_intermediate_rho(rho0, rhof, tof, M_star, lr_star, 
                                     orbit_type, Lmat, Rmat, UTC_list,
                                     periapsis_check=periapsis_check)
            
        print('len rhok_list', len(rhok_list))
            
        # Exception Handling
        # If no solution to Lambert problem is found, exit and continue range
        # grid search
        if len(rhok_list) == 0:

            print('no Lambert solution')
            print('rho0', rho0)
            print('rhof', rhof)
            
            break
            
        # All exceptions passed, valid range values and Lambert solution
        # Assume a single intermediate point for now
        rhok_calc_vect = rhok_list[0]
        rk_vect = Rmat[:,1].reshape(3,1) + rhok_calc_vect
        rk = np.linalg.norm(rk_vect)
        
        # Construct basis vectors
        rhok_obs_hat = Lmat[:,1].reshape(3,1)
        u_vect = np.cross(rhok_obs_hat, rhok_calc_vect, axis=0)
        
        # Exception Handling
        # Solution can converge on rhok_calc_hat pointing 180 degrees away
        # from rhok_obs_hat, if so, exit and continue range grid search
        rhok_dot = float(np.dot(rhok_obs_hat.T, rhok_calc_vect))
        if rhok_dot/np.linalg.norm(rhok_calc_vect) < -0.99:

            print('rhok point 180 degrees away')
            print('rho0', rho0)
            print('rhof', rhof)
            print(rhok_dot)
            print(np.linalg.norm(rhok_calc_vect))
            
            break
        
        # Exit Condition
        # If rhok_obs_hat = rhok_calc_hat, the solution has converged
        # Record range values and exit to continue range grid search (if
        # fewer than 3 valid solutions have been found for this case)
        if np.linalg.norm(u_vect) == 0.:
            rho0_output_list.append(rho0)
            rhof_output_list.append(rhof)            
            break
        
        # Construct basis vectors
        p_vect = np.cross(u_vect, rhok_obs_hat, axis=0)
        p_hat = p_vect/np.linalg.norm(p_vect)
        n_vect = np.cross(rhok_obs_hat, p_hat, axis=0)
        n_hat = n_vect/np.linalg.norm(n_vect)
        
        # Compute basic f and g penalty functions
        f, g = compute_penalty(rhok_calc_vect, p_hat, n_hat)
        fc = float(f)
        
        print('f', f)
        print('g', g)
        
        
        
        # Parameter fc will be used to check step size and update if needed
        # It needs to be updated to account for previous solutions
        # Calculation of other parameters will be stored for later use in 
        # calculation of update to range values, to avoid previous solutions
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

        print('fc', fc)

        # Exception Handling
        # If previous step has produced a much larger penalty value, update
        # range values and restart loop (Gooding 1997 Eq. 8)
        if iters > 0:
            
            print('fc_old', fc_old)
            if fc > 2.*fc_old:
                
                print('step too large')
                print('fc', fc)
                print('fc_old', fc_old)
                rho0 = (fc*rho0_old + fc_old*rho0)/(fc + fc_old)
                rhof = (fc*rhof_old + fc_old*rhof)/(fc + fc_old)
                iters += 1
                continue
        
        
        
        # Use central finite difference to compute numerical derivatives of f
        # and g with respect to small changes in rho0 and rhof
        
        # Exception Handling
        # If any of the finite difference steps does not return
        # a valid Lambert solution, decrease step by factor of 10 and repeat
        # up to 3 times
        for finite_iters in range(3):
            dx = rho0 * finite_diff_step
            dy = rhof * finite_diff_step
            dx2 = dx**2.
            dy2 = dy**2.
            
            print('dx', dx)
            print('dy', dy)
            
            # Range rho0 minus delta_rho
            rho0_minus = rho0 - dx
            rhok_list, rhok_inds = \
                compute_intermediate_rho(rho0_minus, rhof, tof, M_star, lr_star,
                                         orbit_type, Lmat, Rmat, UTC_list, 
                                         periapsis_check=periapsis_check)
            if len(rhok_list) == 0:
                finite_diff_step *= 0.1
                continue
                
            fm0, gm0 = compute_penalty(rhok_list[0], p_hat, n_hat)
            
            # Range rho0 plus delta_rho
            rho0_plus = rho0 + dx
            rhok_list, rhok_inds = \
                compute_intermediate_rho(rho0_plus, rhof, tof, M_star, lr_star, 
                                         orbit_type, Lmat, Rmat, UTC_list,
                                         periapsis_check=periapsis_check)
            if len(rhok_list) == 0:
                finite_diff_step *= 0.1
                continue
                
            fp0, gp0 = compute_penalty(rhok_list[0], p_hat, n_hat)
            
            # Range rhof minus delta_rho
            rhof_minus = rhof - dy
            rhok_list, rhok_inds = \
                compute_intermediate_rho(rho0, rhof_minus, tof, M_star, lr_star,
                                         orbit_type, Lmat, Rmat, UTC_list,
                                         periapsis_check=periapsis_check)
            if len(rhok_list) == 0:
                finite_diff_step *= 0.1
                continue
                
            fmf, gmf = compute_penalty(rhok_list[0], p_hat, n_hat)
            
            # Range rhof plus delta_rho
            rhof_plus = rhof + dy
            rhok_list, rhok_inds = \
                compute_intermediate_rho(rho0, rhof_plus, tof, M_star, lr_star,
                                         orbit_type, Lmat, Rmat, UTC_list,
                                         periapsis_check=periapsis_check)
            if len(rhok_list) == 0:
                finite_diff_step *= 0.1
                continue
                
            fpf, gpf = compute_penalty(rhok_list[0], p_hat, n_hat)
            
            # Multivariate partial (rho0_plus and rhof_plus)
            rhok_list, rhok_inds = \
                compute_intermediate_rho(rho0_plus, rhof_plus, tof, M_star, lr_star,
                                         orbit_type, Lmat, Rmat, UTC_list,
                                         periapsis_check=periapsis_check)
            if len(rhok_list) == 0:
                finite_diff_step *= 0.1
                continue
            
            fp_0f, gp_0f = compute_penalty(rhok_list[0], p_hat, n_hat)
            
            # Compute derivatives
            fx = (fp0 - fm0)/(2.*dx)
            fy = (fpf - fmf)/(2.*dy)
            gx = (gp0 - gm0)/(2.*dx)
            gy = (gpf - gmf)/(2.*dy)
            
            fxx = (fp0 + fm0 - 2.*f)/dx2
            gxx = (gp0 + gm0 - 2.*g)/dx2
            fyy = (fpf + fmf - 2.*f)/dy2
            gyy = (gpf + gmf - 2.*g)/dy2
            
            fxy = (fp_0f - f)/(dx*dy) - (fx/dy + fy/dx) - 0.5*(fxx*(dx/dy) + fyy*(dy/dx))             
            gxy = (gp_0f - g)/(dx*dy) - (gx/dy + gy/dx) - 0.5*(gxx*(dx/dy) + gyy*(dy/dx))
            
            # Everything worked, no need to update finite_diff_step and repeat
            break        
        
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
            
            fx -= f*w0
            fy -= f*wf
            
            fxx -= 2.*fx*w0 + w*f*(1. + epsilon**2.*uw)
            gxx -= 2.*gx*w0
            fyy -= 2.*fy*wf + w*f*(1. + eta**2.*uw)
            gyy -= 2.*gy*wf
            
            fxy -= fy*w0 + fx*wf + w*f*epsilon*eta*uw
            gxy -= gy*w0 + gx*wf

        # Either seek zeros or minimum of the penalty function
        if rootfind == 'zeros' or iters < 2:
            
            # Compute Newton-Raphson increments (Gooding 1997 Section 3.2)
            D_NR = fx*gy - fy*gx
            
            delta_rho0_NR = -(1./D_NR) * f * gy
            delta_rhof_NR =  (1./D_NR) * f * gx
            
            print('D_NR', D_NR)
            print('delta_rho0_NR', delta_rho0_NR)
            print('delta_rhof_NR', delta_rhof_NR)
            
            # Compute Halley/mNR derivatives (Gooding 1997 Section 3.2)
            # If HN = 0.5 use Halley formula
            # If HN = 1.0 use modifed Newton-Raphson formula
            # Use of modified Newton-Raphson should be more robust in case of 
            # neighboring solutions
            fx_H = fx + HN*(fxx*delta_rho0_NR + fxy*delta_rhof_NR)
            fy_H = fy + HN*(fxy*delta_rho0_NR + fyy*delta_rhof_NR)
            gx_H = gx + HN*(gxx*delta_rho0_NR + gxy*delta_rhof_NR)
            gy_H = gy + HN*(gxy*delta_rho0_NR + gyy*delta_rhof_NR)
            
            D_H = fx_H*gy_H - fy_H*gx_H
            
            delta_rho0 = -(1./D_H) * f * gy_H
            delta_rhof =  (1./D_H) * f * gx_H
            
            # Exception Handling
            # Check for near singular derivative matrix (Gooding 1997 Eq 15-16)
            H = fx**2. + fy**2. + gx**2. + gy**2.
            dd = 2.*abs(D_NR)/(H + np.sqrt(H**2. - 4.*D_NR**2.))
            
            # If below threshold, use geometric mean of NR and (Halley or mNR)
            if dd < crit_gm:
                print('use geometric mean')
                print('dd', dd)
                print('crit_gm', crit_gm)
                print('D_NR', D_NR)
                print('test', D_NR/(fx*fy + gx*gy))
                delta_rho0 = np.sign(delta_rho0_NR) * np.sqrt(abs(delta_rho0_NR*delta_rho0))
                delta_rhof = np.sign(delta_rhof_NR) * np.sqrt(abs(delta_rhof_NR*delta_rhof))
                
            # If multiple solutions already found, use original f to check 
            # convergence criteria to be consistent for all solutions
            conv_crit = abs(f)/max(rk, rhok_dot)

        elif rootfind == 'min':
            
            # Compute derivatives of h = 0.5*(f^2 + g^2) and use Newton-Raphson
            # to find stationary point where h_x = h_y = 0
            hx = f*fx
            hy = f*fy
            hxx = f*fxx + fx**2. + gx**2.
            hyy = f*fyy + fy**2. + gy**2.
            hxy = f*fxy + fx*fy + gx*gy
            Dmin = hxx*hyy - hxy**2.
            
            print('fx', fx)
            print('fy', fy)
            print('hx', hx)
            print('hy', hy)
            print('hxx', hxx)
            print('hxy', hxy)
            print('hyy', hyy)
            print('Dmin', Dmin)
            
            test_mat = np.array([[hxx, hxy],[hxy, hyy]])
            matinv = np.linalg.inv(test_mat)
            test = -np.dot(matinv, np.array([[hx],[hy]]))
            
            print('test delta rho', test)
            
            delta_rho0 = -(hyy*hx - hxy*hy)/Dmin
            delta_rhof = -(hxx*hy - hxy*hx)/Dmin
            
            # Use h function derivatives for convergence test
            conv_crit = (abs(hx) + abs(hy))/max(rk, rhok_dot)
        
        # Store values for future comparison
        fc_old = float(fc)
        rho0_old = float(rho0)
        rhof_old = float(rhof)

        # Update range values
        rho0 += delta_rho0
        rhof += delta_rhof
        
        print('delta_rho0', delta_rho0)
        print('delta_rhof', delta_rhof)
        
        print('rho0', rho0)
        print('rhof', rhof)
        
        
        print('conv_crit', conv_crit)
        print('denom', rk, rhok_dot)        
        
        # For converged solution, store answer and exit to continue range
        # grid search
        if conv_crit < tol:
            rho0_output_list.append(rho0)
            rhof_output_list.append(rhof)

            print('rho0_output_list', rho0_output_list)
            print('rhof_output_list', rhof_output_list)
            
            break        

        print('rho0_output_list', rho0_output_list)
        print('rhof_output_list', rhof_output_list)
        
        # Increment counter
        iters += 1
        


    return rho0_output_list, rhof_output_list


def compute_M_max(Lmat, Rmat, tof, GM=GME, R=Re, orbit_regime='none'):
    '''
    The function computes the maximum number of orbit revolutions that can be
    completed in the given time of flight, with an optional restriction on
    orbit regime.
    
    Parameters
    ------
    Lmat : 3xN numpy array
        columns correspond to line of sight unit vectors in ECI for each
        observation time
    Rmat : 3xN numpy array
        columns correspond to sensor location vectors in ECI for each 
        observation time [km]
    tof : float
        time of flight from first to last observation in seconds
    GM : float, optional
        gravitational parameter of central body (default=GME) [km^3/s^2]
    R : float, optional
        radius of central body (default=Re) [km]
    orbit_regime : string, optional
        flag to select solutions only from a specific orbit regime, which will
        place restrictions on the range grid search and improve computational
        performance ('LEO', 'MEO', 'HEO', 'GEO', or 'none') (default='none')
    
    Returns
    ------
    M_max : int
        maximum number of orbit revolutions that can be completed for given
        time of flight
    
    '''
    
    # Retrieve parameters for the given orbit regime
    step, rmin, rmax, dum, dum2 = define_orbit_regime(orbit_regime)
    a_min = rmin


    # Initial line of sight, sensor location, and position vectors
    rho0_hat = Lmat[:,0].reshape(3,1)
    q0_vect = Rmat[:,0].reshape(3,1)
    rho0_min = radius2rho(rmin, rho0_hat, q0_vect)
    r0_vect = q0_vect + rho0_min*rho0_hat
    
    print(rho0_hat)
    print(q0_vect)
    print(rho0_min)
    print(r0_vect)
    
    # Final line of sight, sensor location, and position vectors
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
    if s < 2.*a_min:
        s = 2.*a_min
    
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


def compute_range_search_list(Lmat, Rmat, M_star, tof, GM=GME,
                              orbit_regime='none'):
    '''
    This function generates a list of rho0 and rhof value pairs for use as an
    initial guess in the Gooding angles-only IOD method. Minimum and maximum
    values are computed based on the sensor locations and pointing directions
    and orbit characteristics including (optionally) the orbit regime.
    
    Parameters
    ------
    Lmat : 3xN numpy array
        columns correspond to line of sight unit vectors in ECI for each
        observation time
    Rmat : 3xN numpy array
        columns correspond to sensor location vectors in ECI for each 
        observation time [km]
    M_star : int
        exact integer number of complete orbit revolutions traversed
    tof : float
        time of flight from the first to last observation [sec]
    GM : float, optional
        gravitational parameter of central body (default=GME) [km^3/s^2]
    orbit_regime : string, optional
        choose from 'LEO', 'GEO', 'HEO', 'MEO', or 'none' (default='none')
    
    Returns
    ------
    range_pairs_list : list
        list of rho0 and rhof values to use as initial guess for Gooding
        angles-only IOD [km]    
    orbit_regime_fail : boolean
        exit status flag, indicate if the given M_star does not yield a viable
        solution within the given orbit regime (fail = True)
        
    '''
    
    # Initialize output
    orbit_regime_fail = False
    
    # Retrieve parameters for the given orbit regime 
    step, rmin, rmax, dum, dum2 = define_orbit_regime(orbit_regime)
    
    # Sensor location and pointing vectors 
    rho0_hat = Lmat[:,0].reshape(3,1)
    q0_vect = Rmat[:,0].reshape(3,1)    
    rhof_hat = Lmat[:,-1].reshape(3,1)
    qf_vect = Rmat[:,-1].reshape(3,1)
    
    # Further restrict maximum range values using orbit revolution number
    if M_star > 0:
        
        # Compute orbit parameters for extreme case to get upper bound on 
        # orbit radius and range
        rp = rmin
        n_min = (M_star/tof)*2.*math.pi
        a_max = (GM/n_min**2.)**(1./3.)
        e_max = 1. - (rp/a_max)        

        rmax_M = a_max*(1. + e_max)
        
        print('M_star', M_star)
        print('a_max', a_max)
        print('e_max', e_max)
        print('rmax_M', rmax_M)
        
        # Exit condition
        # If the maximum orbit radius allowable to complete M_star revolutions
        # is less than the minimum radius of the given orbit regime, there is
        # no valid solution
        if rmax_M < rmin:
            orbit_regime_fail = True
            return [], orbit_regime_fail
        
        # If newly computed rmax is less than generic value for orbit regime,
        # update to use the new value corresponding to the revolution number
        if rmax_M < rmax:
            rmax = rmax_M
            
    # Compute minimum and maximum ranges
    rho0_min = radius2rho(rmin, rho0_hat, q0_vect)
    rhof_min = radius2rho(rmin, rhof_hat, qf_vect)  
    rho0_max = radius2rho(rmax, rho0_hat, q0_vect)
    rhof_max = radius2rho(rmax, rhof_hat, qf_vect)  
    
    print('rho0_min', rho0_min)
    print('rho0_max', rho0_max)
    print('rhof_min', rhof_min)
    print('rhof_max', rhof_max)

    # Compute range search arrays
    rho0_array = np.arange(rho0_min, rho0_max, step)
    rho0_array = np.append(rho0_array, rho0_max)
    rhof_array = np.arange(rhof_min, rhof_max, step)
    rhof_array = np.append(rhof_array, rhof_max)
    
    # Form list of range pairs    
    range_pairs = itertools.product(rho0_array, rhof_array)
    range_pairs_list = [list(pair) for pair in range_pairs]
    
    return range_pairs_list, orbit_regime_fail


def compute_range_ind_list(range_pairs_list, search_mode='middle_out'):
    '''
    This function generates a list of indices to select from the range pairs
    list in different orders, as set by the search mode input.
    
    Parameters
    ------
    range_pairs_list : list
        list of rho0 and rhof values to use as initial guess for Gooding
        angles-only IOD [km] 
    search_mode : string, optional
        determines how to index into the range_pairs_list
        'bottom_up' = start from minimum rho0, rhof and proceed upwards
        'random' = randomly indexed
        'middle_out' = start from middle and proceed in regular increments
        
    Returns
    ------
    range_ind_list : list
        indices to use to retrieve range pair values
        
    '''
    
    nrange = len(range_pairs_list)
        
    if search_mode == 'bottom_up':
        range_ind_list = list(range(nrange))
        
    elif search_mode == 'random':
        range_ind_list = list(range(nrange))
        random.shuffle(range_ind_list)

    elif search_mode == 'middle_out':
        ind = int(np.floor(nrange/2))
        increment = int(np.floor(nrange/4))
        
        while math.gcd(nrange, increment) != 1:
            increment += 1
            
        range_ind_list = [(ind+increment*ii) % nrange for ii in range(nrange)]

    return range_ind_list


def define_orbit_regime(orbit_regime):
    '''
    This function defines the basic parameters for each orbit regime.
    
    Parameters
    ------
    orbit_regime : string
        choose from 'LEO', 'GEO', 'HEO', 'MEO', or 'none' (default='none')
        
    Returns
    ------
    step : float
        step size for range grid search [km]
    rmin : float
        minimum orbit radius [km]
    rmax : float
        maximum orbit radius [km]
    rp_bounds : list
        min/max values of radius of periapsis [km]
    ra_bounds : list
        min/max values of radius of apoapsis [km]
    
    References
    ------
    [1] Holzinger, M. et al., "Uncorrelated-Track Classification,
        Characterization, and Prioritization Using Admissible Regions and 
        Bayesian Inference," JGCD 2016.
        
    '''
    
    print('Orbit Regime', orbit_regime)
    
    # These values have been modified somewhat from Ref 1. The intent is to 
    # restrict the range search grid in meaningful ways to improve 
    # computational performance, this is not an exhaustive orbit 
    # classification scheme, and does not cover all possible orbits. Any
    # orbit that does not fit in a given regime can still be solved for using
    # 'none' though this will be slow.
    
    if orbit_regime == 'LEO':
        rp_min = Re + 100.
        rp_max = Re + 2000.
        ra_min = Re + 100.
        ra_max = Re + 2000.
        rmin = rp_min
        rmax = ra_max
        step = 500.
        
    elif orbit_regime == 'MEO':
        rp_min = Re + 2000.
        rp_max = 40000.
        ra_min = Re + 2000.
        ra_max = 40000.
        rmin = rp_min
        rmax = ra_max
        step = 1000.
        
    elif orbit_regime == 'GEO':
        rp_min = 40000.
        rp_max = 45000.
        ra_min = 40000.
        ra_max = 45000.
        rmin = rp_min
        rmax = ra_max
        step = 1000.
        
    elif orbit_regime == 'HEO':
        rp_min = Re + 100.
        rp_max = 10000.
        ra_min = 35000.
        ra_max = 50000.
        rmin = rp_min
        rmax = ra_max
        step = 5000.
        
    else:
        rp_min = Re + 100.
        rp_max = 50000.
        ra_min = Re + 100.
        ra_max = 50000.
        rmin = rp_min
        rmax = ra_max
        step = 1000.
        
    rp_bounds = [rp_min, rp_max]
    ra_bounds = [ra_min, ra_max]
    
    print('rmin', rmin)
    print('rmax', rmax)
        
    return step, rmin, rmax, rp_bounds, ra_bounds


def compute_intermediate_rho(rho0, rhof, tof, M_star, lr_star, orbit_type,
                             Lmat, Rmat, UTC_list, periapsis_check=False):
    '''
    This function computes the range value(s) at intermediate observation 
    time(s) to fit the Lambert solution produced by the input rho0 and rhof
    values.
    
    Parameters
    ------
    rho0 : float
        range at initial time [km]
    rhof : float
        range at final time [km]    
    tof : float
        time of flight from the first to last observation [sec]
    M_star : int
        exact integer number of complete orbit revolutions traversed
    lr_star : string
        flag to indicate whether to use left branch or right branch solution
        in multirev cases ('left', 'right', or 'none'). Use 'none for single
        rev cases.
    orbit_type : string
        flag to indicate 'prograde' or 'retrograde' orbit
    Lmat : 3xN numpy array
        columns correspond to line of sight unit vectors in ECI for each
        observation time
    Rmat : 3xN numpy array
        columns correspond to sensor location vectors in ECI for each 
        observation time [km]    
    UTC_list : list
        datetime objects corresponding to observation times in UTC
    periapsis_check : boolean, optional
        flag to determine whether to check the orbit does not intersect the
        central body (rp > R) (default=True)
        
    Returns
    ------
    rhok_list : list
        range value(s) at intermediate observation time(s) [km]
    rhok_inds : list
        index into rhok_list to retrieve correct range value for each time
        rho[tk] = rhok_list[rhok_inds[kk]]
    
    '''
    
    # For now, assume only a single intermediate observation, but this can
    # be expanded to use multiple measurements with an appropriate cost 
    # function
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
    
    # Hyperbolic orbit can pass periapsis check but still have unreasonable
    # semi-major axis. C3 of 160 is sufficient to reach Pluto and corresponds
    # to |SMA| = 2491 km. Any |SMA| < 2500 km can safely be rejected.
    r0 = np.linalg.norm(r0_vect)
    v0 = np.linalg.norm(v0_vect)
    a = 1/(2/r0 - v0**2./GME)
    if abs(a) < 2500.:
        return [], []
    
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


def compute_penalty(rhok_vect, p_hat, n_hat):    
    '''
    This function computes the penalty function f,g values.
    
    Parameters
    ------
    rhok_vect : 3x1 numpy array
        line of sight vector at intermediate time tk in ECI [km]
    p_hat : 3x1 numpy array
        unit vector along the orbit direction
    n_hat : 3x1 numpy array
        unit vector normal to orbit plane
        
    Returns
    ------
    f : float
        penalty function value in along-track direction [km]
    g : float
        penalty function value in normal direction [km]
    
    '''
    
    # Compute basic f and g penalty functions
    f = float(np.dot(p_hat.T, rhok_vect))
    g = float(np.dot(n_hat.T, rhok_vect))
    
    return f, g


def radius2rho(r, rho_hat_eci, site_eci):
    '''
    This function computes range value from a sensor to space object
    corresponding to a given orbit radius.
    
    Parameters
    ------
    r : float
        orbit radius [km]
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


###############################################################################
# Gauss Angles-Only IOD
###############################################################################


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








