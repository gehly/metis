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
        rp = astro.compute_rp(r1_vect, v1_vect, GM)
        if periapsis_check and rp < R:
            continue
        
        v1_list.append(v1_vect)
        v2_list.append(v2_vect)
        M_list.append(Mi)
    
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
            rp = astro.compute_rp(r1_vect, v1_vect, GM)
            if periapsis_check and rp < R:
                continue
            
            v1_list.append(v1_vect)
            v2_list.append(v2_vect)
            M_list.append(Mi)
            type_list.append('retrograde')
            
            

    return v1_list, v2_list, M_list, type_list


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






###############################################################################
# Angles-Only IOD Methods
###############################################################################

def gooding_angles_iod(tk_list, Yk_list, sensor_id_list, sensor_params,
                     time_format='datetime', eop_alldata=[], XYs_df=[]):
    '''
    
    
    '''
    
    
    
    # Choose the first and last measurements Y0 and Yf and compute LOS vectors
    
    
    # Loop over possible guesses for ranges rho0 and rhof
    
    
        # Solve Lambert's Problem (returns multiple solutions)
        
        
        # Loop over Lambert solutions
        
        
            # Compute penalty function f
            # Example f = dot(los2_obs, los2_calc)
            
        
            
            # Adjust rho0 and rhof to convergence
            # Use central finite difference to compute numerical derivatives
            # of penalty function and Newton-Raphson to step toward minimum
            
            
            
            
        # Store solution (r0_vect, v0_vect, M) 
    
    
    
    return



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








