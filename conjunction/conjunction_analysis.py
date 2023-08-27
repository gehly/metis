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


def compute_subinterval(X1, X2, GM=GME):
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
    GM : float, optional
        gravitational parameter (default=GME) [km^3/s^2]
        
    Returns
    ------
    subinterval : float
        duration of appropriate subinterval [sec]
        
    '''
    
    # Convert X1 and X2 to orbit elements
    elem1 = astro.cart2kep(X1, GM)
    elem2 = astro.cart2kep(X2, GM)
    a1 = float(elem1[0])
    a2 = float(elem2[0])
    
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
        
    # Use 1/2 of the smaller orbit period 
    subinterval = period/2.    
    
    return subinterval


if __name__ == '__main__':
    
    a = 0.
    b = 200.
    N = 16
    xvec = compute_CGL_nodes(a,b,N)
    interp_mat = compute_interpolation_matrix(N)
    
    print(xvec)
    print(interp_mat)
    