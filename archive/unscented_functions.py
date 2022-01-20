import numpy as np
from math import *
import os
import pickle
from scipy.integrate import odeint
import copy
import utilities.conversions as conv

###############################################################################
# This file contains functions to perform unscented transforms and unscented
# filtering (UKF)
# Functions:
#  lp_ukf
#  unscented_transform
#  unscented_polar2cart
#  unscented_radec2enu
#  unscented_cart2element
#  unscented_element2cart
###############################################################################


def lp_ukf(state_dict, meas_dict, inputs, intfcn, meas_fcn, alpha=1.,
           pnorm=2.):
    '''
    This function implements the Unscented Kalman Filter for a minimum p-norm
    distribution.

    Parameters
    ------
    state_dict : dictionary
        initial state and covariance for filter execution
    meas_dict : dictionary
        measurement data over time for the filter and parameters (noise, etc)
    inputs : dictionary
        input parameters
    intfcn : function handle
        function for dynamics model
    meas_fcn : function handle
        function for measurements
    alpha : float, optional
        sigma point distribution parameter (default=1.)
    pnorm : float, optional
        p-norm distribution parameter (default=2.)

    Returns
    ------
    filter_output : dictionary
        output state, covariance, and post-fit residuals over time
    '''

    # State information
    state_ti = sorted(state_dict.keys())[-1]
    X = state_dict[state_ti]['X']
    P = state_dict[state_ti]['P']
    Q = inputs['Q']
    L = len(X)

    # Measurement information
    meas_types = meas_dict['meas_types']
    sigma_dict = meas_dict['sigma_dict']
    p = len(meas_types)
    Rk = np.zeros((p, p))
    for ii in range(p):
        mtype = meas_types[ii]
        sig = sigma_dict[mtype]
        Rk[ii,ii] = sig**2.

    # Prior information about the distribution
    kurt = gamma(5./pnorm)*gamma(1./pnorm)/(gamma(3./pnorm)**2.)
    beta = kurt - 1.
    kappa = kurt - float(L)

    # Compute sigma point weights
    lam = alpha**2.*(L + kappa) - L
    gam = np.sqrt(L + lam)
    Wm = 1./(2.*(L + lam)) * np.ones((1, 2*L))
    Wm = list(Wm.flatten())
    Wc = copy.copy(Wm)
    Wm.insert(0, lam/(L + lam))
    Wc.insert(0, lam/(L + lam) + (1 - alpha**2 + beta))
    Wm = np.asarray(Wm)
    diagWc = np.diag(Wc)

    # Integrator tolerance
    int_tol = inputs['int_tol']

    # Initialize output
    filter_output = {}

    # Measurement times
    ti_list = sorted(meas_dict['meas'].keys())

    # Loop over times
    for ii in range(len(ti_list)):

        # Retrieve current and previous times
        ti = ti_list[ii]

        if ii == 0:
            ti_prior = state_ti
        else:
            ti_prior = ti_list[ii-1]

        delta_t = ti - ti_prior

        # Read the next observation
        Yi = meas_dict['meas'][ti]

        # Compute sigma points matrix
        sqP = np.linalg.cholesky(P)
        Xrep = np.tile(X, (1, L))
        chi = np.concatenate((X, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
        chi_v = np.reshape(chi, (L*(2*L+1), 1), order='F')

        # Integrate chi
        if ti_prior == ti:
            intout = chi_v.T
        else:
            int0 = chi_v.flatten()
            tin = [ti_prior, ti]
            intout = odeint(intfcn, int0, tin, args=(inputs,), rtol=int_tol,
                            atol=int_tol)

        # Extract values for later calculations
        chi_v = intout[-1,:]
        chi = np.reshape(chi_v, (L, 2*L+1), order='F')

        # Add process noise
        Xbar = np.dot(chi, Wm.T)
        Xbar = np.reshape(Xbar, (L, 1))
        chi_diff = chi - np.dot(Xbar, np.ones((1, (2*L+1))))
        Pbar = delta_t*Q + np.dot(chi_diff, np.dot(diagWc, chi_diff.T))

        # Re-symmetric pos def
        Pbar = 0.5 * (Pbar + Pbar.T)

        # Computed Measurements
        ybar, Pyy, Pxy = unscented_transform(Xbar, Pbar, meas_fcn, inputs,
                                             alpha, pnorm)
        Pyy += Rk

        # Measurement Update
        K = np.dot(Pxy, np.linalg.inv(Pyy))
        X = Xbar + np.dot(K, Yi-ybar)
        P = Pbar - np.dot(K, np.dot(Pyy, K.T))

        # Re-symmetric pos def
        P = 0.5 * (P + P.T)

        # Calculate post-fit residuals
        ybar_post, dum1, dum2 = unscented_transform(X, P, meas_fcn, inputs,
                                                    alpha, pnorm)
        resids = Yi - ybar_post
        
#        print 'ti', ti
#        print 'X', X
#        print 'P', P
#        print 'Yi', Yi
#        print 'ybar', ybar
#        print 'ybar_post', ybar_post
#        print 'resids', resids

        # Append data to output
        filter_output[ti] = {}
        filter_output[ti]['X'] = copy.copy(X)
        filter_output[ti]['P'] = copy.copy(P)
        filter_output[ti]['resids'] = copy.copy(resids)

    return filter_output
    
    


###############################################################################
# Unscented Tranform Functions
###############################################################################


def unscented_transform(m1, P1, transform_fcn, inputs, alpha=1., pnorm=2.):
    '''
    This function computes the unscented transform for a p-norm
    distribution and user defined transform function.

    Parameters
    ------
    m1 : nx1 numpy array
      mean state vector
    P1 : nxn numpy array
      covariance matrix
    transform_fcn : function handle
      name of transform function
    inputs : dictionary
      input parameters for transform function
    alpha : float, optional
      sigma point distribution parameter (default=1)
    pnorm : float, optional
      value of p-norm distribution (default=2)

    Returns
    ------
    m2 : mx1 numpy array
      transformed mean state vector
    P2 : mxm numpy array
      transformed covariance matrix
    Pcross : nxm numpy array
      cross correlation covariance matrix
    '''

    # Number of States
    L = int(m1.shape[0])

    # Prior information about the distribution
    kurt = gamma(5./pnorm)*gamma(1./pnorm)/(gamma(3./pnorm)**2.)
    beta = kurt - 1.
    kappa = kurt - float(L)

    # Compute sigma point weights
    lam = alpha**2.*(L + kappa) - L
    gam = np.sqrt(L + lam)
    Wm = 1./(2.*(L + lam)) * np.ones((1, 2*L))
    Wm = list(Wm.flatten())
    Wc = copy.copy(Wm)
    Wm.insert(0, lam/(L + lam))
    Wc.insert(0, lam/(L + lam) + (1 - alpha**2 + beta))
    Wm = np.asarray(Wm)
    diagWc = np.diag(Wc)

    # Compute chi - baseline sigma points
    sqP = np.linalg.cholesky(P1)
    Xrep = np.tile(m1, (1, L))
    chi = np.concatenate((m1, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
    chi_diff = chi - np.dot(m1, np.ones((1, (2*L+1))))
    
#    if m1[1] < 1.2e-5:
#        print m1
#        print 'lam',lam
#        print 'gam',gam
#        print sqP
#        print chi

    # Compute transformed sigma points
    Y = transform_fcn(chi, inputs)
    row2 = int(Y.shape[0])
    col2 = int(Y.shape[1])
    
#    print 
#    print 'unscented transform'
#    print 'Y', Y
#    print 'Wm', Wm
#    print 'sum Wm', sum(Wm)
#    print 'dot', np.dot(Y, Wm.T)
#    print 'mean az', np.mean(Y[0,:])
#    print 'mean el', np.mean(Y[1,:])

    # Compute mean and covar
    m2 = np.dot(Y, Wm.T)
    m2 = np.reshape(m2, (row2, 1))
    Y_diff = Y - np.dot(m2, np.ones((1, col2)))
    P2 = np.dot(Y_diff, np.dot(diagWc, Y_diff.T))
    Pcross = np.dot(chi_diff,  np.dot(diagWc, Y_diff.T))
    
#    print 'm2', m2
#    print 'Y_diff', Y_diff
#    print 'P2', P2

    return m2, P2, Pcross


def unscented_polar2cart(chi, inputs):
    '''
    Function for use with unscented_transform.
    Converts sigma point matrix from polar coordinates to cartesian (2D). Polar
    angle given in radians.

    Parameters
    ------
    chi : L x (2L+1) numpy array
      sigma point matrix
    inputs : dictionary
      input parameters

    Returns
    ------
    Y : m x (2L+1) numpy array
      transformed sigma point matrix
    '''

    # Size of input/output
    row = int(chi.shape[0])
    col = int(chi.shape[1])
    Y = np.zeros((row, col))

    for j in xrange(0, col):

        # Pull out column of chi
        rho = chi[0,j]
        theta = chi[1,j]

        # Transform values
        x = rho*cos(theta)
        y = rho*sin(theta)

        Y[0,j] = x
        Y[1,j] = y

    return Y


def unscented_radec2enu(chi, inputs):
    '''
    Function for use with unscented_transform.
    Converts sigma point matrix from topocentric right ascension/declination to
    ENU angles LAM/PHI. All angles given in radians.

    Parameters
    ------
    chi : L x (2L+1) numpy array
      sigma point matrix
    inputs : dictionary
      input parameters

    Returns
    ------
    Y : m x (2L+1) numpy array
      transformed sigma point matrix
    '''

    # Size of input/output
    row = int(chi.shape[0])
    col = int(chi.shape[1])
    Y = np.zeros((row, col))
    r_site = inputs['r_site']
    JD = inputs['JD']
    myIAU = inputs['myIAU']

    for j in xrange(0, col):

        # Pull out column of chi
        radec = chi[:,j]
        ra = radec[0]
        dec = radec[1]

        # Convert to ENU
        # ECI Unit Vector
        rho_hat_eci = np.array([[cos(ra)*cos(dec)],
                                [sin(ra)*cos(dec)], [sin(dec)]])

        # Rotate to ECEF
        dum = modf(float(JD))
        jdTime = [int(dum[1]), dum[0]]

        res = myIAU.ECI2ECEF(jdTime, position=rho_hat_eci)
        rho_hat_ecef = np.reshape(res[0], (3, 1))

        # Rotate to ENU
        rho_hat_enu = conv.ecef2enu(rho_hat_ecef, r_site)

        # Compute angles
        PHI = asin(rho_hat_enu[1])
        LAM = atan2(rho_hat_enu[0], rho_hat_enu[2])

        Y[0,j] = LAM
        Y[1,j] = PHI        

    return Y


def unscented_eci2enu(chi, inputs):
    '''
    Function for use with unscented_transform.
    Converts sigma point matrix from ECI to
    ENU angles LAM/PHI. All angles given in radians.

    Parameters
    ------
    chi : L x (2L+1) numpy array
      sigma point matrix
    inputs : dictionary
      input parameters

    Returns
    ------
    Y : m x (2L+1) numpy array
      transformed sigma point matrix
    '''

    # Size of input/output
    row = int(chi.shape[0])
    col = int(chi.shape[1])
    Y = np.zeros((row, col))
    r_site = inputs['r_site']
    JD = inputs['JD']
    myIAU = inputs['myIAU']

    for j in xrange(0, col):

        # Pull out column of chi
        mj = chi[:,j]
        x = mj[0]
        y = mj[1]
        z = mj[2]
        r_eci = np.array([[x],[y],[z]])
        
        # Compute stat_eci
        dum = modf(float(JD))
        jdTime = [int(dum[1]), dum[0]]
        res = myIAU.ECEF2ECI(jdTime, position=r_site)
        stat_eci = np.reshape(res[0], (3, 1))

        # Convert to ENU
        # ECI Unit Vector
        rho_hat_eci = (r_eci - stat_eci)/np.linalg.norm(r_eci - stat_eci)
                                
        # Rotate to ECEF        
        res = myIAU.ECI2ECEF(jdTime, position=rho_hat_eci)
        rho_hat_ecef = np.reshape(res[0], (3, 1))

        # Rotate to ENU
        rho_hat_enu = conv.ecef2enu(rho_hat_ecef, r_site)

        # Compute angles
        PHI = asin(rho_hat_enu[1])
        LAM = atan2(rho_hat_enu[0], rho_hat_enu[2])

        Y[0,j] = LAM
        Y[1,j] = PHI
        

    return Y


def unscented_eci2meas(chi, inputs):
    '''
    Function for use with unscented_transform.
    Converts sigma point matrix from ECI state vector
    to measurement space - as specified in inputs.

    Parameters
    ------
    chi : L x (2L+1) numpy array
      sigma point matrix
    inputs : dictionary
      input parameters including measurement types

    Returns
    ------
    Y : m x (2L+1) numpy array
      transformed sigma point matrix
    '''

    # Size of input/output
    row = int(chi.shape[0])
    col = int(chi.shape[1])
    r_site = inputs['r_site']
    meas_types = inputs['meas_types']
    JED_JD = inputs['JED_JD']
    myIAU = inputs['myIAU']
    
    # Initialize
    p = len(meas_types)
    Y = np.zeros((p,col))
    
    # Loop over sigma points
    for j in xrange(0, col):

        # Pull out column of chi
        mj = chi[:,j]
        x = mj[0]
        y = mj[1]
        z = mj[2]
        r_eci = np.array([[x],[y],[z]])
        
        # Compute stat_eci
        dum = modf(float(JED_JD))
        jdTime = [int(dum[1]), dum[0]]
        res = myIAU.ECEF2ECI(jdTime, position=r_site)
        stat_eci = np.reshape(res[0], (3, 1))

        # Convert to ENU
        # ECI Unit Vector
        rg = np.linalg.norm(r_eci - stat_eci)
        rho_hat_eci = (r_eci - stat_eci)/rg
                                
        # Rotate to ECEF        
        res = myIAU.ECI2ECEF(jdTime, position=rho_hat_eci)
        rho_hat_ecef = np.reshape(res[0], (3, 1))

        # Rotate to ENU
        rho_hat_enu = conv.ecef2enu(rho_hat_ecef, r_site)

        # Compute measurements
        i = 0
        for mtype in meas_types:
            
            if mtype == 'rg':
                Y[i,j] = rg
            
            if mtype == 'az':
                Y[i,j] = atan2(rho_hat_enu[0], rho_hat_enu[1])  # rad            
                
                # Convert az to range 0-2*pi
                if Y[i,j] < 0.:
                    Y[i,j] += 2.*pi
                    
                # Correction for rollover in az at 0
                if j == 0:
                    if Y[i,j] >= 0. and Y[i,j] < pi/2.:
                        quad = 1
                    elif Y[i,j] >= pi/2. and Y[i,j] < pi:
                        quad = 2
                    elif Y[i,j] >= pi and Y[i,j] < 1.5*pi:
                        quad = 3
                    else:
                        quad = 4

                if j > 0:       
                    if (Y[i,j] > 1.5*pi and quad == 1):
                        Y[i,j] -= 2.*pi
                    if (Y[i,j] < pi/2. and quad == 4):
                        Y[i,j] += 2.*pi
                
            if mtype == 'el':
                Y[i,j] = asin(rho_hat_enu[2])  # rad  
            
            if mtype == 'ra' :
                #Compute right ascension and append measurement
                Y[i,j] = atan2(rho_hat_eci[1], rho_hat_eci[0]) #rad

                # Correction for rollover in right ascension at +/- pi
                if j == 0:
                    if Y[i,j] >= 0 and Y[i,j] < pi/2:
                        quad = 1
                    elif Y[i,j] >= pi/2 and Y[i,j] < pi:
                        quad = 2
                    elif Y[i,j] < 0 and Y[i,j] > -pi/2:
                        quad = 4
                    else:
                        quad = 3

                if j > 0:       
                    if (Y[i,j] < 0 and quad == 2):
                        Y[i,j] = 2*pi + Y[i,j]
                    if (Y[i,j] > 0 and quad == 3):
                        Y[i,j] = -2*pi + Y[i,j]

                 

            if mtype == 'dec' :
                #Compute declination
                Y[i,j] = asin(rho_hat_eci[2])  #rad
                
        
            i += 1

    return Y
    

def unscented_ric2eci(chi, inputs):
    '''
    Function for use with unscented_transform.
    Converts sigma point matrix from RIC to ECI.

    Parameters
    ------
    chi : L x (2L+1) numpy array
      sigma point matrix
    inputs : dictionary
      input parameters

    Returns
    ------
    Y : m x (2L+1) numpy array
      transformed sigma point matrix
    '''

    # Size of input/output
    row = int(chi.shape[0])
    col = int(chi.shape[1])
    Y = np.zeros((row, col))
    r_eci = inputs['r']
    v_eci = inputs['v']

    for j in xrange(0, col):

        # Pull out column of chi
        mj = chi[:,j]
        x = mj[0]
        y = mj[1]
        z = mj[2]
        r_eci = np.array([[x],[y],[z]])
        
        

        Y[0,j] = LAM
        Y[1,j] = PHI
        

    return Y



def unscented_element2cart(chi, inputs):
    '''
    Function for use with unscented_transform.
    Converts sigma point matrix from keplerian elements to inertial cartesian
    coordinates.

    Parameters
    ------
    chi : L x (2L+1) numpy array
      sigma point matrix
    inputs : dictionary
      input parameters

    Returns
    ------
    Y : m x (2L+1) numpy array
      transformed sigma point matrix
    '''

    # Size of input/output
    L = int(chi.shape[1])
    Y = np.zeros((6, L))

    for j in xrange(0, L):

        # Pull out column of chi
        elem = chi[:,j]

        # Convert to ECI
        Xeci = conv.element_conversion(elem, 0, 1)
        Y[:,j] = Xeci.flatten()

    return Y


def unscented_cart2element(chi, inputs):
    '''
    Function for use with unscented_transform.
    Converts sigma point matrix from inertial cartesian coordinates to
    keplerian elements.

    Parameters
    ------
    chi : L x (2L+1) numpy array
      sigma point matrix
    inputs : dictionary
      input parameters

    Returns
    ------
    Y : m x (2L+1) numpy array
      transformed sigma point matrix
    '''

    # Size of input/output
    L = int(chi.shape[1])
    Y = np.zeros((6, L))

    for j in xrange(0, L):

        # Pull out column of chi
        cart = chi[:,j]

        # Convert to elements
        X = conv.element_conversion(cart, 1, 0)
        Y[:,j] = X.flatten()

    return Y


def ut_car_to_eci(chi,inputs):
    '''
    Function for use with unscented_transform.
    Converts sigma point matrix from inertial cartesian coordinates to
    keplerian elements.

    Parameters
    ------
    chi : L x (2L+1) numpy array
      sigma point matrix
    inputs : dictionary
      input parameters

    Returns
    ------
    Y : m x (2L+1) numpy array
      transformed sigma point matrix
    '''

    #Station pos/vel in ECI
    q_vect = inputs['q_vect']
    dq_vect = inputs['dq_vect']

    #Initialize output
    Y = np.zeros(chi.shape)
    L = int(chi.shape[1])

    for ind in xrange(0,L) :

        #Break out chi
        rho = float(chi[0,ind])
        drho = float(chi[1,ind])
        ra = float(chi[2,ind])
        dec = float(chi[3,ind])
        dra = float(chi[4,ind])
        ddec = float(chi[5,ind])

        #Unit vectors
        u_rho = np.array([cos(ra)*cos(dec), sin(ra)*cos(dec), sin(dec)])
        u_ra = np.array([-sin(ra)*cos(dec), cos(ra)*cos(dec), 0])
        u_dec = np.array([-cos(ra)*sin(dec), -sin(ra)*sin(dec), cos(dec)])

        #Range and Range-Rate vectors
        rho_vect = rho*u_rho
        drho_vect = drho*u_rho + rho*dra*u_ra + rho*ddec*u_dec

        #Compute pos/vel in ECI and add to output
        r_vect = q_vect + rho_vect
        v_vect = dq_vect + drho_vect
        Y[:,ind] = np.append(r_vect,v_vect)

    return Y