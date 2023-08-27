import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sys
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from utilities import astrodynamics as astro
from utilities import coordinate_systems as coord
from utilities.constants import Re, GME

###############################################################################
#
# This script contains functions to analyze collision risk, including 
# calculation of Time of Closest Approach (TCA), Euclidean miss distance,
# and collision probability.
#
# References:
#
#  [1] Denenberg, E., "Satellite Closest Approach Calculation Through 
#         Chebyshev Proxy Polynomials," Acta Astronautica, 2020.
#
#  [2] 
###############################################################################



###############################################################################
# Time of Closest Approach (TCA) Functions
###############################################################################


def compute_TCA(X1, X2, trange, gvec_fcn, params, rho_min_crit=0., N=16,
                subinterval_factor=0.5):
    '''
    This function computes the Time of Closest Approach using Chebyshev Proxy
    Polynomials. Per Section 3 of Denenberg (Ref 1), the function subdivides
    the full time interval specified by trange into units no more than half the 
    orbit period of the smaller orbit, which should contain at most one local
    minimum of the relative distance between orbits. The funtion will either
    output the time and Euclidean distance at closest approach over the full
    time interval, or a list of all close approaches under a user-specified
    input rho_min_crit.
    
    Parameters
    ------
    X1 : 6x1 numpy array
        cartesian state vector of object 1 in ECI [km, km/s]
    X2 : 6x1 numpy array
        cartesian state vector of object 2 in ECI [km, km/s]
    trange : 2 element list or array [t0, tf]
        initial and final time for the full interval [sec]
    rho_min_crit : float, optional
        critical value of minimum distance (default=0.)
        if > 0, output will contain all close approaches under this distance
    N : int, optional
        order of the Chebyshev Proxy Polynomial approximation (default=16)
        default corresponds to recommended value from Denenberg Section 3
    subinterval_factor : float, optional
        factor to multiply smaller orbit period by (default=0.5)
        default corresponds to recommended value from Denenberg Section 3
    GM : float, optional
        gravitational parameter (default=GME) [km^3/s^2]
        
    Returns
    ------
    
    '''
       
    # Retrieve input parameters
    GM = params['GM']
    
    # Setup first time interval
    subinterval = compute_subinterval(X1, X2, subinterval_factor, GM)
    a = trange[0]
    b = min(trange[-1], a + subinterval)
    
    # Compute interpolation matrix for Chebyshev Proxy Polynomials of order N
    # Note that this can be reused for all polynomial approximations as it
    # only depends on the order
    interp_mat = compute_interpolation_matrix(N)
    
    # Initialize the minimum range and TCA using the endpoints of the interval
    dum, rvec, ivec, cvec = gvec_fcn(trange, X1, X2, params)
    rho0 = np.sqrt(rvec[0]**2 + ivec[0]**2 + cvec[0]**2)
    rhof = np.sqrt(rvec[-1]**2 + ivec[-1]**2 + cvec[-1]**2)
    
    if ((rho0 < rho_min_crit) and (rhof < rho_min_crit)) or (rho0 == rhof):
        T_list = [trange[0], trange[-1]]
        rho_list = [rho0, rhof]
    elif rho0 < rhof:
        T_list = [trange[0]]
        rho_list = [rho0]
    elif rhof < rho0:
        T_list = [trange[-1]]
        rho_list = [rhof]        
    
    # Loop over times in increments of subinterval until end of trange
    rho_min = min(rho_list)
    tmin = T_list[rho_list.index(rho_min)]
    while b <= trange[1]:        
    
        # Determine Chebyshev-Gauss-Lobato node locations
        tvec = compute_CGL_nodes(a, b, N)
        
        # Evaluate function at node locations
        gvec, dum1, dum2, dum3 = gvec_fcn(tvec, X1, X2, params)
        
        # Find the roots of the relative range rate g(t)
        troots = compute_gt_roots(gvec, interp_mat, a, b)
        if len(troots) == 0:
            continue
        
        # Check if roots constitute a global minimum and/or are below the
        # critical threshold
        dum, rvec, ivec, cvec = gvec_fcn(troots, X1, X2, params)
        for ii in range(len(troots)):
            rho = np.sqrt(rvec[ii]**2 + ivec[ii]**2 + cvec[ii]**2)
            
            # Store if below critical threshold
            if rho < rho_min_crit:
                rho_list.append(rho)
                T_list.append(troots[ii])
            
            # Update global minimum
            if rho < rho_min:
                rho_min = rho
                tmin = troots[ii]
            
        # Increment time interval
        if b == trange[-1]:
            break
        
        a = float(b)
        if b + subinterval <= trange[-1]:
            b += subinterval
        else:
            b = trange[-1]            
        
    # Store global minimum
    if tmin not in T_list:
        T_list.append(tmin)
        rho_list.append(rho_min)
        
    # Sort output
    if len(T_list) > 1:
        sorted_inds = np.argsort(T_list)
        T_list = [T_list[ii] for ii in sorted_inds]
        rho_list = [rho_list[ii] for ii in sorted_inds]
    
    return T_list, rho_list


def gvec_twobody_analytic(tvec, X1, X2, params):
    '''
    This function evaluates a twobody orbit propagation analytically (using 
    Newton-Raphson iteration to solve Kepler's Equation) at the times 
    specified in tvec. It then computes the relative distance and associated
    derivative between the two given orbits.
    
    Parameters
    ------
    tvec: 1D numpy array
        times for function evaluation [sec]
    X1 : 6x1 numpy array
        cartesian state vector of object 1 in ECI [km, km/s]
    X2 : 6x1 numpy array
        cartesian state vector of object 2 in ECI [km, km/s]
    params : dictionary
        physical data related to orbit propagation, such as GM, J2, ...
        
    Returns
    ------
    gvec: 1D numpy array
        evaluation of function to find roots of, in this case g(t) is the
        relative range rate between the two orbits, roots correspond to extrema
        of the relative range
    
    '''
    
    # Retrieve input data
    GM = params['GM']    
    
    # Compute function values to find roots of
    # In order to minimize rho, we seek zeros of first derivative
    # f(t) = dot(rho_vect, rho_vect)
    # g(t) = df/dt = 2*dot(drho_vect, rho_vect)
    gvec = np.zeros(tvec.shape)
    rvec = np.zeros(tvec.shape)
    ivec = np.zeros(tvec.shape)
    cvec = np.zeros(tvec.shape)
    jj = 0
    for t in tvec:
        X1_t = astro.element_conversion(X1, 1, 1, GM, t)
        X2_t = astro.element_conversion(X2, 1, 1, GM, t)
        rc_vect = X1_t[0:3].reshape(3,1)
        vc_vect = X1_t[3:6].reshape(3,1)
        rd_vect = X2_t[0:3].reshape(3,1)
        vd_vect = X2_t[3:6].reshape(3,1)
        
        rho_eci = rd_vect - rc_vect
        drho_eci = vd_vect - vc_vect
        rho_ric = coord.eci2ric(rc_vect, vc_vect, rho_eci)
        drho_ric = coord.eci2ric_vel(rc_vect, vc_vect, rho_ric, drho_eci)
        
        gvec[jj] = float(2*np.dot(rho_ric.T, drho_ric))
        rvec[jj] = float(rho_ric[0])
        ivec[jj] = float(rho_ric[1])
        cvec[jj] = float(rho_ric[2])
        jj += 1    
    
    return gvec, rvec, ivec, cvec



def compute_CGL_nodes(a, b, N):
    '''
    This function computes the location of the Chebyshev-Gauss-Lobatto nodes
    over the interval [a,b] given the order of the Chebyshev Proxy Polynomial 
    N. Per the algorithm in Denenberg, these nodes can be computed once and 
    used to approximate the derivative of the distance function, as well as the 
    relative distance components in RIC coordinates, for the same interval.
    
    Parameters
    ------
    a : float
        lower bound of interval
    b : float
        upper bound of interval
    N : int
        order of the Chebyshev Proxy Polynomial approximation
        
    Returns
    ------
    xvec : 1D (N+1) numpy array
        CGL node locations
    
    '''
    
    # Compute CGL nodes (Denenberg Eq 11)
    jvec = np.arange(0,N+1)
    xvec = ((b-a)/2.)*(np.cos(np.pi*jvec/N)) + ((b+a)/2.)
    
    return xvec


def compute_interpolation_matrix(N):
    '''
    This function computes the (N+1)x(N+1) interpolation matrix given the order
    of the Chebyshev Proxy Polynomial N. Per the algorithm in Denenberg, this 
    matrix can be computed once and reused to approximate the derivative of the
    distance function over multiple intervals, as well as to compute the 
    relative distance components in RIC coordinates.
    
    Parameters
    ------
    N : int
        order of the Chebyshev Proxy Polynomial approximation
    
    Returns
    ------
    interp_mat : (N+1)x(N+1) numpy array
        interpolation matrix
        
    '''
    
    # Compute values of pj (Denenberg Eq 13)
    pvec = np.ones(N+1,)
    pvec[0] = 2.
    pvec[N] = 2.
    
    # Compute terms of interpolation matrix (Denenberg Eq 12)
    # Set up arrays of j,k values and compute outer product matrix
    jvec = np.arange(0,N+1)
    kvec = jvec.copy()
    jk_mat = np.dot(jvec.reshape(N+1,1),kvec.reshape(1,N+1))
    
    # Compute cosine term and pj,pk matrix, then multiply component-wise
    Cmat = np.cos(np.pi/N*jk_mat)
    pjk_mat = (2./N)*(1./np.dot(pvec.reshape(N+1,1), pvec.reshape(1,N+1)))
    interp_mat = np.multiply(pjk_mat, Cmat)
    
    return interp_mat


def compute_gt_roots(gvec, interp_mat, a, b):
    
    # Order of approximation
    N = len(gvec) - 1
    
    # Compute aj coefficients (Denenberg Eq 14)
    aj_vec = np.dot(interp_mat, gvec.reshape(N+1,1))
    
    # Compute the companion matrix (Denenberg Eq 18)
    Amat = np.zeros((N,N))
    Amat[0,1] = 1.
    Amat[-1,:] = -aj_vec[0:N].flatten()/(2*aj_vec[N])
    Amat[-1,-2] += 0.5
    for jj in range(1,N-1):
        Amat[jj,jj-1] = 0.5
        Amat[jj,jj+1] = 0.5
        
    # Compute eigenvalues
    # TODO paper indicates some eigenvalues may have small imaginary component
    # but testing seems to show this is much more significant issue, needs
    # further analysis
    eig, dum = np.linalg.eig(Amat)
    eig_real = np.asarray([np.real(ee) for ee in eig if (np.isreal(ee) and ee >= -1. and ee <= 1.)])
    roots = (b+a)/2. + eig_real*(b-a)/2.

    return roots


def compute_subinterval(X1, X2, subinterval_factor=0.5, GM=GME):
    '''
    This function computes an appropriate length subinterval of the specified
    (finite) total interval on which to find the closest approach. Per the
    discussion in Denenberg Section 3, for 2 closed orbits, there will be at
    most 4 extrema (2 minima) during one revolution of the smaller orbit. Use
    of a subinterval equal to half this time yields a unique (local) minimum
    over the subinterval and has shown to work well in testing.
    
    Parameters
    ------
    X1 : 6x1 numpy array
        cartesian state vector of object 1 in ECI [km, km/s]
    X2 : 6x1 numpy array
        cartesian state vector of object 2 in ECI [km, km/s]
    subinterval_factor : float, optional
        factor to multiply smaller orbit period by (default=0.5)
    GM : float, optional
        gravitational parameter (default=GME) [km^3/s^2]
        
    Returns
    ------
    subinterval : float
        duration of appropriate subinterval [sec]
        
    '''
    
    # Compute semi-major axis
    a1 = compute_SMA(X1)
    a2 = compute_SMA(X2)
    
    # If both orbits are closed, choose the smaller to compute orbit period
    if (a1 > 0.) and (a2 > 0.):
        amin = min(a1, a2)
        period = 2.*np.pi*(amin**3./GM)
        
    # If one orbit is closed and the other is an escape trajectory, choose the
    # closed orbit to compute orbit period
    elif a1 > 0.:
        period = 2.*np.pi*(a1**3./GM)
    
    elif a2 > 0.:
        period = 2.*np.pi*(a2**3./GM)
        
    # If both orbits are escape trajectories, choose an arbitrary period 
    # corresponding to small orbit
    else:
        period = 3600.
        
    # Scale the smaller orbit period by user input
    subinterval = period*subinterval_factor
    
    return subinterval


def compute_SMA(cart, GM=GME):
    '''
    This function computes semi-major axis given a Cartesian state vector in
    inertial coordinates.
    
    Parameters
    ------
    cart : 6x1 numpy array
        cartesian state vector in ECI [km, km/s]
    GM : float, optional
        gravitational parameter (default=GME) [km^3/s^2]
        
    Returns
    ------
    a : float
        semi-major axis [km]
    
    '''
    
    # Retrieve position and velocity vectors
    r_vect = cart[0:3].flatten()
    v_vect = cart[3:6].flatten()

    # Calculate orbit parameters
    r = np.linalg.norm(r_vect)
    v2 = np.dot(v_vect, v_vect)

    # Calculate semi-major axis
    a = 1./(2./r - v2/GM)     # km    
    
    return a


if __name__ == '__main__':
    
    a = 0.
    b = 200.
    N = 16
    xvec = compute_CGL_nodes(a,b,N)
    interp_mat = compute_interpolation_matrix(N)
    
    print(xvec)
    print(interp_mat)
    