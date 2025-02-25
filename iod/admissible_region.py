###############################################################################
# This file contains code to implement admissible region initial orbit 
# determination (AR-IOD).
#
# References:
#
#  [1] DeMars, K.J. and Jah, M.K., "Probabilistic Initial Orbit Determination
#      Using Gaussian Mixture Models," JGCD, 2013.
#
###############################################################################


import numpy as np





###############################################################################
# Constrained Admissible Region (CAR)
###############################################################################

def optical_car_gmm(Zk, q_vect, dq_vect, params):
    '''
    This function computes a Gaussian Mixture Model (GMM) to approximate a 
    uniform distribution representing the Constrained Admissible Region (CAR)
    produced by a 4D optical measurement set, containing angles and 
    angle-rates, in particular topocentric right ascension and declination.
    
    The method is based on DeMars and Jah [1].
    

    Parameters
    ----------
    Zk : 4x1 numpy array
        measurement vector containing topocentric RA/DEC and rates [rad, rad/s]
    q_vect : 3x1 numpy array
        sensor position vector in ECI [m]
    dq_vect : 3x1 numpy array
        sensor velocity vector in ECI [m/s]
    params : dictionary
        additional parameters (CAR limits)

    Returns
    -------
    car_gmm : dictionary
        

    '''
      
    # Compute CAR boundaries
    rho_lim, drho_lim, drho_dict, rho_a_all, rho_e_all, drho_a_all, drho_e_all = \
            car_drho_limits(Zk, q_vect, dq_vect, params)
    
    
    return


def car_drho_limits(Zk, q_vect, dq_vect, params):
    '''
    This function computes a Gaussian Mixture Model (GMM) to approximate a 
    uniform distribution representing the Constrained Admissible Region (CAR)
    produced by a 4D optical measurement set, containing angles and 
    angle-rates, in particular topocentric right ascension and declination.
    
    The method is based on DeMars and Jah [1].
    

    Parameters
    ----------
    Zk : 4x1 numpy array
        measurement vector containing topocentric RA/DEC and rates [rad, rad/s]
    q_vect : 3x1 numpy array
        sensor position vector in ECI [m]
    dq_vect : 3x1 numpy array
        sensor velocity vector in ECI [m/s]
    params : dictionary
        additional parameters (CAR limits)

    Returns
    -------
    car_gmm : dictionary
        

    '''
    
    # Break out inputs
    GM = params['GM']
    Re = params['Re']
    a_max = params['a_max']
    a_min = params['a_min']
    e_max = params['e_max']
    
    # Retrieve measurement data
    Zk = Zk.flatten()
    ra = float(Zk[0])
    dec = float(Zk[1])
    dra = float(Zk[2])
    ddec = float(Zk[3])
    
    # Flatten vectors for dot and cross products
    q_vect = q_vect.flatten()
    dq_vect = dq_vect.flatten()
    
    # Unit vectors (DeMars between Eq 1-2)
    u_rho = np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)])
    u_ra = np.array([-np.sin(ra)*np.cos(dec), np.cos(ra)*np.cos(dec), 0.])
    u_dec = np.array([-np.cos(ra)*np.sin(dec), -np.sin(ra)*np.sin(dec), np.cos(dec)])
    
    # Semi-Major Axis Constraint
    # Compute coefficients (DeMars Eq 2 setup)
    w0 = np.dot(q_vect, q_vect)
    w1 = 2.*np.dot(dq_vect, u_rho)
    w2 = dra**2.*np.cos(dec)**2. + ddec**2.
    w3 = 2.*dra*np.dot(dq_vect, u_ra) + 2.*ddec*np.dot(dq_vect, u_dec)
    w4 = np.dot(dq_vect, dq_vect)
    w5 = 2.*np.dot(q_vect, u_rho)
    
    # Compute energy limits (DeMars Eq 5)
    E_max = -GM/(2.*a_max)
    if a_min == 0. :
        a_min = 1e-10
    E_min = -GM/(2.*a_min)
    
    # Eccentricity Constraint
    e_min = 0.

    # Angular Momentum Components (DeMars Eq 6 setup)
    h1 = np.cross(q_vect, u_rho)
    h2 = np.cross(u_rho, (dra*u_ra + ddec*u_dec))
    h3 = np.cross(u_rho, dq_vect) + np.cross(q_vect, (dra*u_ra + ddec*u_dec))
    h4 = np.cross(q_vect, dq_vect)
    
    # Compute coefficients
    c0 = np.dot(h1, h1)
    c1 = 2.*np.dot(h1, h2)
    c2 = 2.*np.dot(h1, h3)
    c3 = 2.*np.dot(h1, h4)
    c4 = np.dot(h2, h2)
    c5 = 2.*np.dot(h2, h3)
    c6 = 2.*np.dot(h2, h4) + np.dot(h3, h3)
    c7 = 2.*np.dot(h3, h4)
    c8 = np.dot(h4, h4)
    
    
    
    
    
    return



def car_sigma_library(a, b, sigma_in) :
    '''
    This function returns the sigma value required to approximate a uniform
    distribution with a GMM with "L" homoscedastic, evenly spaced, and
    evenly weighted components. Library based on standard uniform distribution
    (a = 0, b = 1, p = 1/(b-a)).  Will return result for minimum number of
    components required to achieve desired std or lower, up to max of 15
    components.

    Parameters
    ------
    a : float
        lower limit
    b : float
        upper limit
    sigma_in : float
        desired standard deviation

    Returns
    ------
    sigma_out : float
        actual standard deviation
    L : int
        number of components 
    
    References
    ------
    DeMars and Jah Table 1 [1]

    '''

    sig_dict = {
        1: 0.3467,
        2: 0.2903,
        3: 0.2466,
        4: 0.2001,
        5: 0.1531,
        6: 0.1225,
        7: 0.1026,
        8: 0.0884,
        9: 0.0778,
        10: 0.0696,
        11: 0.0629,
        12: 0.0575,
        13: 0.0529,
        14: 0.0490,
        15: 0.0456
        }

    for L in range(1,16) :
        sigma_out = (b-a) * sig_dict[L]
        if sigma_out < sigma_in :
            break

    return sigma_out, L


