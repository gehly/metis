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
    
    # Loop over times in increments of subinterval until end of trange
    T_list = []
    rho_list = []
    while b <= trange[1]:        
    
        # Determine Chebyshev-Gauss-Lobato node locations
        tvec = compute_CGL_nodes(a, b, N)
        
        # Evaluate function at node locations
        gvec = gvec_fcn(tvec, X1, X2, params)
        
        
    
    
    
    
    return


def gvec_twobody_analytic(tvec, X1, X2, params):
    '''
    
    '''
    
    
    
    return



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
    