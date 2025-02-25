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
            
    # Compute range marginal PDF quantities
    a_rho = np.min(rho_lim)
    b_rho = np.max(rho_lim)    
    sigma_rho, L_rho = car_sigma_library(a_rho, b_rho, params['sigma_rho_desired'])
    
    # Compute means and covariances for GMM components
    m_rho = []
    for i in range(L_rho):
        m_rho.append(a_rho + (b_rho-a_rho)/(L_rho+1.)*(i+1.))
    
    P_rho = [sigma_rho**2.]*L_rho
    
    
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
    rho_vect = params['rho_vect']
    a_max = params['a_max']
    a_min = params['a_min']
    e_max = params['e_max']
    e_min = 0.                      # only apply upper limit on ecc for now
        
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
    if a_min == 0.:
        a_min = 1e-10
    E_min = -GM/(2.*a_min)
    
    # Eccentricity Constraint
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

    # Loop over range values  
    rho_output = np.array([])
    drho_output = np.array([])
    rho_a_all = np.array([])
    rho_e_all = np.array([])
    drho_a_all = np.array([])
    drho_e_all = np.array([])
    drho_dict = {}
    for ii in range(len(rho_vect)):

        # Current range value
        rho = rho_vect[ii]

        # Compute F for current range (DeMars Eq 3 setup)
        F = w2*rho**2. + w3*rho + w4 - 2.*GM/np.sqrt(rho**2. + w5*rho + w0)

        # Compute values of drho for SMA limits (DeMars Eq 4)
        # Max/Min values of the radical in DeMars Eq 4
        rad_max = (w1/2.)**2. - F + 2.*E_max
        rad_min = (w1/2.)**2. - F + 2.*E_min

        drho_a = np.array([])
        if rad_max >= 0.:
            rad_max = np.sqrt(rad_max)
            drho_a = np.append(drho_a, np.array([-w1/2. + rad_max, -w1/2. - rad_max]))
        if rad_min >= 0.:
            rad_min = np.sqrt(rad_min)
            drho_a = np.append(drho_a, np.array([-w1/2. + rad_min, -w1/2. - rad_min]))

        # Eccentricity Constraints
        # Compute P and U for current range (DeMars Eq 6)
        P = c1*rho**2. + c2*rho + c3
        U = c4*rho**4. + c5*rho**3. + c6*rho**2. + c7*rho + c8

        # Compute coefficients (DeMars Eq 8)
        a0_max = F*U + GM**2.*(1.-e_max**2.)
        a0_min = F*U + GM**2.*(1.-e_min**2.)
        a1 = F*P + w1*U
        a2 = U + c0*F + w1*P
        a3 = P + c0*w1
        a4 = c0

        # Solve the quartic equation (DeMars Eq 8)
        r = np.roots(np.array([a4, a3, a2, a1, a0_max]))
        drho_ecc =  np.array([])
        for i in range(len(r)):
            if np.isreal(r[i]):
                drho_ecc = np.append(drho_ecc, float(r[i]))

        # Set up output
        # Build arrays of rho values corresponding to limits in SMA and ECC
        drho_a_all = np.append(drho_a_all, drho_a)
        drho_e_all = np.append(drho_e_all, drho_ecc)
        for ii in range(len(drho_a)):
            rho_a_all = np.append(rho_a_all, rho)
        for ii in range(len(drho_ecc)):
            rho_e_all = np.append(rho_e_all, rho)
                
        # If the eccentricity and semi-major axis limits have returned values
        # for drho, determine which form the boundaries of the CAR
        if len(drho_ecc) and len(drho_a):            

            if len(drho_ecc) == 2:
                
                if len(drho_a) == 2:
                    rho_output = np.append(rho_output, np.array([rho, rho]))
                    drho_vect = np.append(drho_ecc, drho_a)
                    drho_vect = np.sort(drho_vect)
                    drho_output = np.append(drho_output, drho_vect[1:3])
                    drho_dict[rho] = drho_vect[1:3]

                if len(drho_a) == 4:
                    drho_a = np.sort(drho_a)
                    drho_ecc = np.sort(drho_ecc)

                    # Positive side
                    drho_vect1 = np.array([])
                    if drho_a[2] < np.max(drho_ecc):
                        rho_output = np.append(rho_output, np.array([rho, rho]))
                        
                        if drho_a[3] < np.max(drho_ecc):
                            drho_vect1 = drho_a[2:4]
                            drho_output = np.append(drho_output, drho_vect1)
                        else:
                            drho_vect1 = np.array([drho_a[2], np.max(drho_ecc)])
                            drho_output = np.append(drho_output, drho_vect1)
                            
                        #drho_dict[rho] = drho_vect1

                    # Negative Side
                    drho_vect2 = np.array([])
                    if drho_a[1] > np.min(drho_ecc):
                        rho_output = np.append(rho_output, np.array([rho, rho]))

                        if drho_a[0] > np.min(drho_ecc):
                            drho_vect2 = drho_a[0:2]
                            drho_output = np.append(drho_output, drho_vect2)
                        else :
                            drho_vect2 = np.array([drho_a[1], np.min(drho_ecc)])
                            drho_output = np.append(drho_output, drho_vect2)
                            
                    drho_dict[rho] = np.append(drho_vect1, drho_vect2)                                           

            if len(drho_ecc) == 4:

                if len(drho_a) == 2:
                    rho_output = np.append(rho_output, np.array([rho, rho, rho, rho]))
                    drho_vect = np.append(drho_a, drho_ecc)
                    drho_vect = np.sort(drho_vect)
                    drho_output = np.append(drho_output, drho_vect[1:5])
                    drho_dict[rho] = drho_vect[1:5]

                if len(drho_a) == 4:
                    drho_a = np.sort(drho_a)
                    drho_ecc = np.sort(drho_ecc)

                    # Positive Side
                    drho_vect1 = np.array([])
                    if drho_a[2] < np.max(drho_ecc):
                        rho_output = np.append(rho_output, np.array([rho, rho]))

                        if drho_a[3] < np.max(drho_ecc):
                            drho_vect1 = drho_a[2:4]
                            drho_output = np.append(drho_output, drho_vect1)
                        else :
                            drho_vect1 = np.array([drho_a[2], np.max(drho_ecc)])
                            drho_output = np.append(drho_output, drho_vect1)

                        #drho_dict[rho] = drho_vect1

                    # Negative Side
                    drho_vect2 = np.array([])
                    if drho_a[1] > np.min(drho_ecc):
                        rho_output = np.append(rho_output, np.array([rho,rho]))

                        if drho_a[0] > np.min(drho_ecc):
                            drho_vect2 = drho_a[0:2]
                            drho_output = np.append(drho_output, drho_vect2)
                        else :
                            drho_vect2 = np.array([drho_a[1], np.min(drho_ecc)])
                            drho_output = np.append(drho_output, drho_vect2)

                    drho_dict[rho] = np.append(drho_vect1, drho_vect2)    
    
    
    
    return rho_output, drho_output, drho_dict, rho_a_all, rho_e_all, drho_a_all, drho_e_all


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


