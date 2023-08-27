import numpy as np


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





if __name__ == '__main__':
    
    a = 0.
    b = 200.
    N = 16
    xvec = compute_CGL_nodes(a,b,N)
    interp_mat = compute_interpolation_matrix(N)
    
    print(xvec)
    print(interp_mat)
    