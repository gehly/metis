import numpy as np
import math
from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys
import inspect
from scipy.integrate import dblquad

# Load tudatpy modules  
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from dynamics import dynamics_functions as dyn
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
#      Chebyshev Proxy Polynomials," Acta Astronautica, 2020.
#
#  [2] Hall, D.T., Hejduk, M.D., and Johnson, L.C., "Remedidting Non-Positive
#      Definite State Covariances for Collision Probability Estimation," 2017.
#
#  [3] https://github.com/nasa/CARA_Analysis_Tools
#
#  [4] CCSDS Recommendation for Space Data System Standards: Conjunction Data
#      Message (Recommended Standard 508.0-B-1), 2013.
#
#
###############################################################################



###############################################################################
# Time of Closest Approach (TCA) Functions
###############################################################################


def compute_TCA(X1, X2, trange, gvec_fcn, params, tudat_flag=False, 
                rho_min_crit=0., N=16, subinterval_factor=0.5):
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
    
    # Setup Tudat propagation if needed
    if tudat_flag:
        state_params = params['state_params']
        params['bodies'] = dyn.initialize_tudat(state_params)
        
        # Convert time to seconds since J2000
        if params['int_params']['time_format'] == 'datetime':
            trange = [(trange[ii] - datetime(2000, 1, 1, 12, 0, 0)).total_seconds() for ii in range(len(trange))]

    # Setup first time interval
    subinterval = compute_subinterval(X1, X2, subinterval_factor, GME)
    t0 = trange[0]
    a = trange[0]
    b = min(trange[-1], a + subinterval)
        
    # Compute interpolation matrix for Chebyshev Proxy Polynomials of order N
    # Note that this can be reused for all polynomial approximations as it
    # only depends on the order
    interp_mat = compute_interpolation_matrix(N)
    
    # Loop over times in increments of subinterval until end of trange
    T_list = []
    rho_list = []
    rho_min = np.inf
    tmin = 0.
    while b <= trange[1]:
        
        # print('')
        # print('a,b', a, b)
    
        # Determine Chebyshev-Gauss-Lobato node locations
        tvec = compute_CGL_nodes(a, b, N)
        
        # Evaluate function at node locations
        gvec, dum1, dum2, dum3 = gvec_fcn(t0, tvec, X1, X2, params)
        
        # Find the roots of the relative range rate g(t)
        troots = compute_gt_roots(gvec, interp_mat, a, b)
        
        # If this is first pass, include the interval endpoints for evaluation
        if np.isinf(rho_min):
            troots = np.concatenate((troots, np.array([trange[0], trange[-1]])))
            
                
        # Check if roots constitute a global minimum and/or are below the
        # critical threshold
        if len(troots) > 0:
            
            dum, rvec, ivec, cvec = gvec_fcn(t0, troots, X1, X2, params)
            for ii in range(len(troots)):
                rho = np.sqrt(rvec[ii]**2 + ivec[ii]**2 + cvec[ii]**2)
                
                # print('ti', troots[ii])
                # print('rho', rho)
                
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
            
    # # Evaluate the relative range at the endpoints of the interval to ensure
    # # these are not overlooked
    # dum, rvec, ivec, cvec = gvec_fcn(trange, X1, X2, params)
    # rho0 = np.sqrt(rvec[0]**2 + ivec[0]**2 + cvec[0]**2)
    # rhof = np.sqrt(rvec[-1]**2 + ivec[-1]**2 + cvec[-1]**2)
    
    # # Store the global minimum and append to lists if others are below 
    # # critical threshold
    # rho_candidates = [rho_min, rho0, rhof]
    # global_min = min(rho_candidates)
    # global_tmin = [tmin, trange[0], trange[-1]][rho_candidates.index(global_min)]
    
    
    
    # if ((rho0 < rho_min_crit) and (rhof < rho_min_crit)) or (rho0 == rhof):
    #     T_list = [trange[0], trange[-1]]
    #     rho_list = [rho0, rhof]
    # elif rho0 < rhof:
    #     T_list = [trange[0]]
    #     rho_list = [rho0]
    # elif rhof < rho0:
    #     T_list = [trange[-1]]
    #     rho_list = [rhof]   
        
    # If a global minimum has been found, store output
    if rho_min < np.inf and tmin not in T_list:
        T_list.append(tmin)
        rho_list.append(rho_min)
    
    # # Otherwise, compute and store the minimum range and TCA using the 
    # # endpoints of the interval
    # else:
           
        
    # Sort output
    if len(T_list) > 1:
        sorted_inds = np.argsort(T_list)
        T_list = [T_list[ii] for ii in sorted_inds]
        rho_list = [rho_list[ii] for ii in sorted_inds]
    
    return T_list, rho_list


def gvec_twobody_analytic(t0, tvec, X1, X2, params):
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
    gvec = np.zeros(len(tvec),)
    rvec = np.zeros(len(tvec),)
    ivec = np.zeros(len(tvec),)
    cvec = np.zeros(len(tvec),)
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


def gvec_tudat(t0, tvec, X1, X2, params):
    
    # Retrieve input data
    bodies = params['bodies']
    state_params = params['state_params']
    int_params = params['int_params']
    
    # Initial state
    Xo = np.concatenate((X1, X2), axis=0)
    
    # Compute function values to find roots of
    # In order to minimize rho, we seek zeros of first derivative
    # f(t) = dot(rho_vect, rho_vect)
    # g(t) = df/dt = 2*dot(drho_vect, rho_vect)
    gvec = np.zeros(len(tvec),)
    rvec = np.zeros(len(tvec),)
    ivec = np.zeros(len(tvec),)
    cvec = np.zeros(len(tvec),)
    jj = 0
    for ti in tvec:
        
        if ti == t0:
            X1_t = X1
            X2_t = X2
            
        else:
            tin = [t0, ti]
            tout, Xout = dyn.general_dynamics(Xo, tin, state_params, int_params, bodies)
            X1_t = Xout[-1,0:6]
            X2_t = Xout[-1,6:12]        
        
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
        period = 2.*np.pi*np.sqrt(amin**3./GM)
        
    # If one orbit is closed and the other is an escape trajectory, choose the
    # closed orbit to compute orbit period
    elif a1 > 0.:
        period = 2.*np.pi*np.sqrt(a1**3./GM)
    
    elif a2 > 0.:
        period = 2.*np.pi*np.sqrt(a2**3./GM)
        
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


###############################################################################
# 2D Probability of Collision (Pc) Functions
###############################################################################

def Pc2D_Foster(X1, P1, X2, P2, HBR, rtol=1e-8, HBR_type='circle'):
    '''
    This function computes the probability of collision (Pc) in the 2D 
    encounter plane following the method of Foster. The code has been ported
    from the MATLAB library developed by the NASA CARA team, listed in Ref 3.
    The function supports 3 types of hard body regions: circle, square, and 
    square equivalent to the area of the circle. The input covariance may be
    either 3x3 or 6x6, but only the 3x3 position covariance will be used in
    the calculation of Pc.
    
    
    Parameters
    ------
    X1 : 6x1 numpy array
        Estimated mean state vector
        Cartesian position and velocity of Object 1 in ECI [km, km/s]
    P1 : 6x6 numpy array
        Estimated covariance of Object 1 in ECI [km^2, km^2/s^2]
    X2 : 6x1 numpy array
        Estimated mean state vector
        Cartesian position and velocity of Object 2 in ECI [km, km/s]
    P2 : 6x6 numpy array
        Estimated covariance of Object 2 in ECI [km^2, km^2/s^2]
    HBR : float
        hard-body region (e.g. radius for spherical object) [km]
    rtol : float, optional
        relative tolerance for numerical quadrature (default=1e-8)
    HBR_type : string, optional
        type of hard body region ('circle', 'square', or 'squareEqArea')
        (default='circle')
    
    Returns
    ------
    Pc : float
        probability of collision
    
    '''
    
    # Retrieve and combine the position covariance
    Peci = P1[0:3,0:3] + P2[0:3,0:3]
    
    # Construct the relative encounter frame
    r1 = np.reshape(X1[0:3], (3,1))
    v1 = np.reshape(X1[3:6], (3,1))
    r2 = np.reshape(X2[0:3], (3,1))
    v2 = np.reshape(X2[3:6], (3,1))
    r = r1 - r2
    v = v1 - v2
    h = np.cross(r, v, axis=0)
    
    # print(r)
    # print(v)
    # print(h)
    
    # Unit vectors of relative encounter frame
    yhat = v/np.linalg.norm(v)
    zhat = h/np.linalg.norm(h)
    xhat = np.cross(yhat, zhat, axis=0)
    
    # Transformation matrix
    eci2xyz = np.concatenate((xhat.T, yhat.T, zhat.T))
    
    # print('')
    
    # print(xhat)
    # print(yhat)
    # print(zhat)
    # print(eci2xyz)
    
    # Transform combined covariance to relative encounter frame (xyz)
    Pxyz = np.dot(eci2xyz, np.dot(Peci, eci2xyz.T))
    
    # 2D Projected covariance on the x-z plane of the relative encounter frame
    red = np.array([[1., 0., 0.], [0., 0., 1.]])
    Pxz = np.dot(red, np.dot(Pxyz, red.T))
    
    # print('')
    # print(Pxyz)
    # print(Pxz)
    
    # Exception Handling
    # Remediate non-positive definite covariances
    Lclip = (1e-4*HBR)**2.
    Pxz_rem, Pxz_det, Pxz_inv, posdef_status, clip_status = remediate_covariance(Pxz, Lclip)
    
    
    # Calculate Double Integral
    x0 = np.linalg.norm(r)
    z0 = 0.
    
    # print('x0', x0)
    
    # Inverse of the Pxz matrix
    # cholPxz_inv = np.linalg.inv(np.linalg.cholesky(Pxz))
    # Pxz_inv = np.dot(cholPxz_inv.T, cholPxz_inv)
    # Pxz_det = np.linalg.det(Pxz)
    
    # print('')
    # print('Pxz det', Pxz_det)
    # print('Pxz inv', Pxz_inv)
    
    # Set up quadrature
    atol = 1e-13
    Integrand = lambda z, x: math.exp(-0.5*(Pxz_inv[0,0]*x**2. + Pxz_inv[0,1]*x*z + Pxz_inv[1,0]*x*z + Pxz_inv[1,1]*z**2.))

    if HBR_type == 'circle':
        lower_semicircle = lambda x: -np.sqrt(HBR**2. - (x-x0)**2.)*(abs(x-x0)<=HBR)
        upper_semicircle = lambda x:  np.sqrt(HBR**2. - (x-x0)**2.)*(abs(x-x0)<=HBR)
        Pc = (1./(2.*math.pi))*(1./np.sqrt(Pxz_det))*float(dblquad(Integrand, x0-HBR, x0+HBR, lower_semicircle, upper_semicircle, epsabs=atol, epsrel=rtol)[0])
        
    elif HBR_type == 'square':
        Pc = (1./(2.*math.pi))*(1./np.sqrt(Pxz_det))*float(dblquad(Integrand, x0-HBR, x0+HBR, z0-HBR, z0+HBR, epsabs=atol, epsrel=rtol)[0])
        
    elif HBR_type == 'squareEqArea':
        HBR_eq = HBR*np.sqrt(math.pi)/2.
        Pc = (1./(2.*math.pi))*(1./np.sqrt(Pxz_det))*float(dblquad(Integrand, x0-HBR_eq, x0+HBR_eq, z0-HBR_eq, z0+HBR_eq, epsabs=atol, epsrel=rtol)[0])
    
    else:
        print('Error: HBR type is not supported! Must be circle, square, or squareEqArea')
        print(HBR_type)
    
    return Pc



def remediate_covariance(Praw, Lclip, Lraw=[], Vraw=[]):
    '''
    This function provides a level of exception handling by detecting and 
    remediating non-positive definite covariances in the collision probability
    calculation, following the procedue in Hall et al. (Ref 2). This code has
    been ported from the MATLAB library developed by the NASA CARA team, 
    listed in Ref 3.
    
    The function employs an eigenvalue clipping method, such that eigenvalues
    below the specified Lclip value are reset to Lclip. The covariance matrix,
    determinant, and inverse are then recomputed using the original 
    eigenvectors and reset eigenvalues to ensure the output is positive (semi)
    definite. An input of Lclip = 0 will result in the output being positive
    semi-definite.
    
    Parameters
    ------
    
    
    Returns
    ------
    
    
    '''
    
    # Ensure the covariance has all real elements
    if not np.all(np.isreal(Praw)):
        print('Error: input Praw is not real!')
        print(Praw)
        return
    
    # Calculate eigenvectors and eigenvalues if not input
    if len(Lraw) == 0 and len(Vraw) == 0:
        Lraw, Vraw = np.linalg.eig(Praw)
        
    # Define the positive definite status of Praw
    posdef_status = np.sign(min(Lraw))
    
    # Clip eigenvalues if needed, and record clipping status
    Lrem = Lraw.copy()
    if min(Lraw) < Lclip:
        clip_status = True
        Lrem[Lraw < Lclip] = Lclip
    else:
        clip_status = False
        
    # Determinant of remediated covariance
    Pdet = np.prod(Lrem)
    
    # Inverse of remediated covariance
    Pinv = np.dot(Vraw, np.dot(np.diag(1./Lrem), Vraw.T))
    
    # Remediated covariance
    if clip_status:
        Prem = np.dot(Vraw, np.dot(np.diag(Lrem), Vraw.T))
    else:
        Prem = Praw.copy()
    
    
    return Prem, Pdet, Pinv, posdef_status, clip_status



###############################################################################
# Utility Functions
###############################################################################

def read_cdm_file(cdm_file):
    '''
    This function reads a CCSDS Conjunction Data Message (CDM) to retrieve
    object state and uncertainty information, collision risk metrics, and 
    metadata related to the generation of the CDM.
    
    
    '''
    
    # test = open(cdm_file).read().split()
    # print(test)
    
    
    field_name_list = []
    data_list = []
    with open(cdm_file) as file:
        for line in file:
            field_name = str(line[0:35]).strip()
            data = str(line[37:]).split()[0]
            
            field_name_list.append(field_name)
            data_list.append(data)
            

    # Retrieve pertinent data
    
    # Time of closest approach (TCA)
    ind = field_name_list.index('TCA')
    TCA_UTC = datetime.strptime(data_list[ind], '%Y-%m-%dT%H:%M:%S.%f')
    
    # Miss Distance
    retrieve_list = ['MISS_DISTANCE', 'RELATIVE_SPEED', 'RELATIVE_POSITION_R',
                     'RELATIVE_POSITION_T', 'RELATIVE_POSITION_N', 
                     'RELATIVE_VELOCITY_R', 'RELATIVE_VELOCITY_T',
                     'RELATIVE_VELOCITY_N', 'COMMENT HBR']
    
    miss_params = {}
    for item in retrieve_list:
        ind = field_name_list.index(item)
        data = float(data_list[ind])
        miss_params[item] = data
        
        
    # Object States, Covariances, and Parameters
    # ind = field_name_list.index('COMMENT HBR')
    # HBR = float(data_list[ind])
    
    for ii in range(len(field_name_list)):
        if field_name_list[ii] == 'OBJECT':
            if data_list[ii] == 'OBJECT1':
                obj1_ind = ii
            elif data_list[ii] == 'OBJECT2':
                obj2_ind = ii
                
    obj_params = {}
    for obj_id in [1,2]:
        obj_params[obj_id] = {}
        obj_params[obj_id]['mean'] = np.zeros((6,1))
        obj_params[obj_id]['covar'] = np.zeros((8,8))
        
        if obj_id == 1:
            lower = obj1_ind
            upper = obj2_ind
        else:
            lower = obj2_ind
            upper = len(field_name_list)
        
        for ii in range(lower, upper):
            field = field_name_list[ii]
            data = data_list[ii]
            if field[0:4] == 'TIME':
                data = datetime.strptime(data, '%Y-%m-%dT%H:%M:%S.%f')
            
            if field == 'X':
                obj_params[obj_id]['mean'][0] = float(data)
            elif field == 'Y':
                obj_params[obj_id]['mean'][1] = float(data)
            elif field == 'Z':
                obj_params[obj_id]['mean'][2] = float(data)
            elif field == 'X_DOT':
                obj_params[obj_id]['mean'][3] = float(data)
            elif field == 'Y_DOT':
                obj_params[obj_id]['mean'][4] = float(data)
            elif field == 'Z_DOT':
                obj_params[obj_id]['mean'][5] = float(data)
                
            elif field == 'CR_R':
                obj_params[obj_id]['covar'][0,0] = float(data)
            elif field == 'CT_R':
                obj_params[obj_id]['covar'][0,1] = float(data)
                obj_params[obj_id]['covar'][1,0] = float(data)
            elif field == 'CT_T':
                obj_params[obj_id]['covar'][1,1] = float(data)
            elif field == 'CN_R':
                obj_params[obj_id]['covar'][0,2] = float(data)
                obj_params[obj_id]['covar'][2,0] = float(data)
            elif field == 'CN_T':
                obj_params[obj_id]['covar'][1,2] = float(data)
                obj_params[obj_id]['covar'][2,1] = float(data)
            elif field == 'CN_N':
                obj_params[obj_id]['covar'][2,2] = float(data)
                
            elif field == 'CRDOT_R':
                obj_params[obj_id]['covar'][0,3] = float(data)
                obj_params[obj_id]['covar'][3,0] = float(data)
            elif field == 'CRDOT_T':
                obj_params[obj_id]['covar'][1,3] = float(data)
                obj_params[obj_id]['covar'][3,1] = float(data)
            elif field == 'CRDOT_N':
                obj_params[obj_id]['covar'][2,3] = float(data)
                obj_params[obj_id]['covar'][3,2] = float(data)
            elif field == 'CRDOT_RDOT':
                obj_params[obj_id]['covar'][3,3] = float(data)
                
            elif field == 'CTDOT_R':
                obj_params[obj_id]['covar'][0,4] = float(data)
                obj_params[obj_id]['covar'][4,0] = float(data)
            elif field == 'CTDOT_T':
                obj_params[obj_id]['covar'][1,4] = float(data)
                obj_params[obj_id]['covar'][4,1] = float(data)
            elif field == 'CTDOT_N':
                obj_params[obj_id]['covar'][2,4] = float(data)
                obj_params[obj_id]['covar'][4,2] = float(data)
            elif field == 'CTDOT_RDOT':
                obj_params[obj_id]['covar'][3,4] = float(data)
                obj_params[obj_id]['covar'][4,3] = float(data)            
            elif field == 'CTDOT_TDOT':
                obj_params[obj_id]['covar'][4,4] = float(data)
                
            elif field == 'CNDOT_R':
                obj_params[obj_id]['covar'][0,5] = float(data)
                obj_params[obj_id]['covar'][5,0] = float(data)
            elif field == 'CNDOT_T':
                obj_params[obj_id]['covar'][1,5] = float(data)
                obj_params[obj_id]['covar'][5,1] = float(data)
            elif field == 'CNDOT_N':
                obj_params[obj_id]['covar'][2,5] = float(data)
                obj_params[obj_id]['covar'][5,2] = float(data)
            elif field == 'CNDOT_RDOT':
                obj_params[obj_id]['covar'][3,5] = float(data)
                obj_params[obj_id]['covar'][5,3] = float(data) 
            elif field == 'CNDOT_TDOT':
                obj_params[obj_id]['covar'][4,5] = float(data)
                obj_params[obj_id]['covar'][5,4] = float(data)        
            elif field == 'CNDOT_NDOT':
                obj_params[obj_id]['covar'][5,5] = float(data)
                
            elif field == 'CDRG_R':
                obj_params[obj_id]['covar'][0,6] = float(data)
                obj_params[obj_id]['covar'][6,0] = float(data)
            elif field == 'CDRG_T':
                obj_params[obj_id]['covar'][1,6] = float(data)
                obj_params[obj_id]['covar'][6,1] = float(data)
            elif field == 'CDRG_N':
                obj_params[obj_id]['covar'][2,6] = float(data)
                obj_params[obj_id]['covar'][6,2] = float(data)
            elif field == 'CDRG_RDOT':
                obj_params[obj_id]['covar'][3,6] = float(data)
                obj_params[obj_id]['covar'][6,3] = float(data) 
            elif field == 'CDRG_TDOT':
                obj_params[obj_id]['covar'][4,6] = float(data)
                obj_params[obj_id]['covar'][6,4] = float(data) 
            elif field == 'CDRG_NDOT':
                obj_params[obj_id]['covar'][5,6] = float(data)
                obj_params[obj_id]['covar'][6,5] = float(data)     
            elif field == 'CDRG_DRG':
                obj_params[obj_id]['covar'][6,6] = float(data)
                
            elif field == 'CSRP_R':
                obj_params[obj_id]['covar'][0,7] = float(data)
                obj_params[obj_id]['covar'][7,0] = float(data)
            elif field == 'CSRP_T':
                obj_params[obj_id]['covar'][1,7] = float(data)
                obj_params[obj_id]['covar'][7,1] = float(data)
            elif field == 'CSRP_N':
                obj_params[obj_id]['covar'][2,7] = float(data)
                obj_params[obj_id]['covar'][7,2] = float(data)
            elif field == 'CSRP_RDOT':
                obj_params[obj_id]['covar'][3,7] = float(data)
                obj_params[obj_id]['covar'][7,3] = float(data) 
            elif field == 'CSRP_TDOT':
                obj_params[obj_id]['covar'][4,7] = float(data)
                obj_params[obj_id]['covar'][7,4] = float(data) 
            elif field == 'CSRP_NDOT':
                obj_params[obj_id]['covar'][5,7] = float(data)
                obj_params[obj_id]['covar'][7,5] = float(data)
            elif field == 'CSRP_DRG':
                obj_params[obj_id]['covar'][6,7] = float(data)
                obj_params[obj_id]['covar'][7,6] = float(data)             
            elif field == 'CSRP_SRP':
                obj_params[obj_id]['covar'][7,7] = float(data)
                
            else:
                obj_params[obj_id][field] = data

    
    
    return TCA_UTC, miss_params, obj_params



if __name__ == '__main__':
    
    
    cdm_file = os.path.join('unit_test/AlfanoTestCase01.cdm')
    
    read_cdm_file(cdm_file)
    
    
    
    
    
    
    