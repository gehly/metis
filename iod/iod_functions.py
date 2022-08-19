import numpy as np
import math
from scipy.integrate import odeint, ode
import sys
import os
import inspect
from datetime import datetime, timedelta

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

import dynamics.numerical_integration as numint
import utilities.astrodynamics as astro

from utilities.constants import Re, GME

###############################################################################
# Classical (Deterministic) Methods
###############################################################################


###############################################################################
# Lambert Solvers 
###############################################################################



def lambert_iod():
    
    
    return



def izzo_lambert(r1_vect, r2_vect, tof, GM=GME, R=Re, results_flag='all',
                 periapsis_check=True, maxiters=35, rtol=1e-8):
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
    
    # Compute all possible x,y values that fit T
    x_list1, y_list1, M_list1 = find_xy(lam, T, maxiters, rtol)
    
    # Loop over x,y values and compute output velocities
    v1_list = []
    v2_list = []
    M_list = []
    for ii in range(len(x_list1)):
        xi = x_list1[ii]
        yi = y_list1[ii]
        Mi = M_list1[ii]
        
        Vr1 =  gamma*((lam*yi - xi) - rho*(lam*yi + xi))/r1
        Vr2 = -gamma*((lam*yi - xi) + rho*(lam*yi + xi))/r2
        Vt1 =  gamma*sigma*(yi + lam*xi)/r1
        Vt2 =  gamma*sigma*(yi + lam*xi)/r2
        
        v1_vect = Vr1*ihat_r1 + Vt1*ihat_t1
        v2_vect = Vr2*ihat_r2 + Vt2*ihat_t2
        
        # Check for valid radius of periapsis
        rp = compute_rp(r1_vect, v1_vect, GM)
        if periapsis_check and rp < R:
            continue
        
        v1_list.append(v1_vect)
        v2_list.append(v2_vect)
        M_list.append(Mi)
        
    # If it is desired to produce all possible cases, repeat execution for
    # retrograde cases (previous results cover prograde cases)
    if results_flag == 'all':
        lam = -lam
        ihat_t1 = -ihat_t1
        ihat_t2 = -ihat_t2
        
        # Compute all possible x,y values that fit T
        x_list2, y_list2, M_list2 = find_xy(lam, T, maxiters, rtol)
        
        # Loop over x,y values and compute output velocities
        for ii in range(len(x_list2)):
            xi = x_list2[ii]
            yi = y_list2[ii]
            Mi = M_list2[ii]
            
            Vr1 =  gamma*((lam*yi - xi) - rho*(lam*yi + xi))/r1
            Vr2 = -gamma*((lam*yi - xi) + rho*(lam*yi + xi))/r2
            Vt1 =  gamma*sigma*(yi + lam*xi)/r1
            Vt2 =  gamma*sigma*(yi + lam*xi)/r2
            
            v1_vect = Vr1*ihat_r1 + Vt1*ihat_t1
            v2_vect = Vr2*ihat_r2 + Vt2*ihat_t2
            
            # Check for valid radius of periapsis
            rp = compute_rp(r1_vect, v1_vect, GM)
            if periapsis_check and rp < R:
                continue
            
            v1_list.append(v1_vect)
            v2_list.append(v2_vect)
            M_list.append(Mi)


    return v1_list, v2_list, M_list


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
    
    # Compute maximum number of complete revolutions that would fit in 
    # the simplest sense T_0M = T_00 + M*pi
    M_max = int(np.floor(T / math.pi))
    
    # Evaluate non-dimensional time of flight T
    # T(x=0) denoted T_0M
    # T_00 is T_0 for single revolution (M=0) case  
    # T_0M = T_00 + M*pi
    T_00 = math.acos(lam) + lam*np.sqrt(1. - lam**2.)
    
    # Check if input T is less than minimum T required for M_max revolutions
    # If so, reduce M_max by 1
    if T < (T_00 + M_max*math.pi) and M_max > 0:
        
        # Start Halley iterations from x=0, T=To and find T_min(M_max)
        dum, T_min, exit_flag = compute_T_min(lam, M_max, maxiters, rtol)
        
        if T < T_min and exit_flag == 1:
            M_max -= 1
            
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
        
    # Run Householder iterations for x_0 to get x,y for single rev case
    M = 0
    x, exit_flag = householder(x_0, T, lam, M, maxiters, rtol)
    if exit_flag == 1:
        y = np.sqrt(1. - lam**2.*(1. - x**2.))
        x_list.append(x)
        y_list.append(y)
        M_list.append(M)

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
        
        xr, exit_flag = householder(x0_r, T, lam, M, maxiters, rtol)
        
        if exit_flag == 1:
            yr = np.sqrt(1. - lam**2.*(1. - xr**2.))
            x_list.append(xr)
            y_list.append(yr)
            M_list.append(M)

    return x_list, y_list, M_list



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
    
    diff = 1.
    iters = 0
    exit_flag = 1
    while diff > tol:
        
        T = compute_T(x0, lam, M)
        dT, ddT, dddT = compute_T_der(x0, T, lam)
        
        # Halley step, cubic
        x = x0 - 2.*dT*ddT/(2.*ddT**2. - dT*dddT)
        
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
    
    diff = 1.
    iters = 0
    exit_flag = 1
    while diff > tol:
        
        T_x = compute_T(x0, lam, M)
        f_x = T_x - T_star
        dT, ddT, dddT = compute_T_der(x0, T_x, lam)
        
        x = x0 - f_x*((dT**2. - f_x*ddT/2.)/
                      (dT*(dT**2. - f_x*ddT) + dddT*f_x**2./6.))
    
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






def gauss_iod(tk_list, Yk_list, sensor_params):
    
    
    return


def gooding_iod():
    
    
    return





