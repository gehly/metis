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
import math
import matplotlib.pyplot as plt




###############################################################################
# Constrained Admissible Region (CAR)
###############################################################################

def optical_car_gmm(rho_vect, Zk, q_vect, dq_vect, params, plot_flag=False):
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
    GMM : list of GM component weights, means, covariance matrices [rho, drho]
          [w,m,P]
            w = list of weights
            m = list of means (numpy px1 arrays)
            P = list of covars (numpy pxp arrays)
        

    '''
      
    # Compute CAR boundaries
    rho_lim, drho_lim, drho_dict, rho_a_all, rho_e_all, drho_a_all, drho_e_all = \
            car_drho_limits(rho_vect, Zk, q_vect, dq_vect, params)
            
    # Compute range marginal PDF quantities
    a_rho = np.min(rho_lim)
    b_rho = np.max(rho_lim)    
    sigma_rho, L_rho = car_sigma_library(a_rho, b_rho, params['sigma_rho_desired'])
    
    # Compute means and covariances for GMM components (DeMars Eq 22)
    m_rho = []
    for i in range(L_rho):
        m_rho.append(a_rho + (b_rho-a_rho)/(L_rho+1.)*(i+1.))
    
    P_rho = [sigma_rho**2.]*L_rho
        
    # Compute weights of GMM components (DeMars Eq 23)
    # Evaluate range marginal PDF at each range value
    p_vect = []
    psum = 0.
    rho_unique = np.unique(rho_lim)
    delta_rho = rho_unique[1] - rho_unique[0]
    for rho in rho_unique:
        drho_vect = drho_dict[rho]
        a_drho = np.min(drho_vect)
        b_drho = np.max(drho_vect)
        p_vect.append((b_drho-a_drho)*delta_rho)
        #p_vect.append((b_drho-a_drho)/(b_rho-a_rho))

    #p_vect = np.asarray(p_vect)/(sum(p_vect)*delta_rho)
    norm_fact = np.trapz(p_vect, rho_unique)
    p_vect = p_vect/norm_fact

    #check = np.dot(p_vect,[delta_rho]*len(p_vect))
    #print check

    #Compute H matrix
    M = len(p_vect)
    H = np.zeros((M, L_rho))
    for i in range(M):
        for j in range(L_rho):
            rhoi = rho_unique[i]
            mj = m_rho[j]
            sigj = np.sqrt(P_rho[j])
            H[i,j] = (1/(np.sqrt(2.*np.pi)*sigj))*np.exp(-0.5*((rhoi-mj)/sigj)**2.)

    #Compute weights (least squares fit)
    w_rho = np.dot(np.linalg.inv(np.dot(H.T, H)), np.dot(H.T, p_vect))

    if abs(sum(w_rho) - 1.) > 0.1:
        print('Error: iod.car_gmm range weights not normalized!!')
        print(w_rho)
        print(sum(w_rho))

    # Compute PDF sum
    g_approx = []
    for i in range(M):
        gi = 0.
        rhoi = rho_unique[i]
        for j in range(L_rho):            
            wj = w_rho[j]
            mj = m_rho[j]
            sigj = np.sqrt(P_rho[j])
            gi += wj*(1/(np.sqrt(2.*np.pi)*sigj))*np.exp(-0.5*((rhoi-mj)/sigj)**2.)
        g_approx.append(gi)  
        
        
    # Compute range-rate marginal PDF quantities and store in GMM
    # Get drho limits for m_rho
    rho_lim2, drho_lim2, drho_dict2 = car_drho_limits(m_rho, Zk, q_vect, dq_vect, params)[0:3]
    
    w = []
    m = []
    P = []
    sig_drho_max = 0.
    xx = []
    yy = []
    zz = []    
    for i in range(L_rho):
        
        # Get values from Range PDF
        wi = w_rho[i]
        mi = m_rho[i]
        Pi = P_rho[i]

        # Get values from Range-Rate PDF
        drho_vect = drho_dict2[mi]
        for k in range(int(len(drho_vect)/2)):
            drho_k = drho_vect[2*k:2*k+2]
            a_drho = np.min(drho_k)
            b_drho = np.max(drho_k)
            sig_drho, L_drho = car_sigma_library(a_drho, b_drho, params['sigma_drho_desired'])
            if sig_drho > sig_drho_max :
                sig_drho_max = sig_drho.copy()


            #Weights, means, covar for this rho
            wj = 1./L_drho
            Pj = sig_drho**2.
            for j in range(L_drho):
                mj = a_drho + (b_drho-a_drho)/(L_drho + 1.)*(j+1.)
                w.append(wi*wj)
                m.append(np.array([[mi],[mj]]))
                P.append(np.array([[Pi,0.],[0.,Pj]]))

    

    GMM = [w,m,P]


    #Plot checks
    if plot_flag:

        print('L_rho = ',L_rho)
        print('sig_rho = ',sigma_rho)
        print('L_tot = ',len(w))
        print('sig_drho_max = ',sig_drho_max)

        mrho_RE = [mi[0]/params['Re'] for mi in m]
        mdrho = [mi[1]/1000. for mi in m]

        # Range Marginal PDF
        plt.figure()
        plt.plot(rho_unique/params['Re'],p_vect,'b--',lw=2)
        plt.plot(rho_unique/params['Re'],g_approx,'r--',lw=2)
        #plt.title('Range Marginal PDF')
        plt.legend(['PDF','GM Approx'])
        plt.xlabel('Range [ER]')
        

        # CAR with GM mean locations
        plt.figure()
        plt.plot(rho_lim/params['Re'],drho_lim/1000.,'k.')
        plt.plot(mrho_RE,mdrho,'k+')
        #plt.title('Constrained Admissible Region')
        plt.xlabel('Range [ER]')
        plt.ylabel('Range-Rate [km/s]')
        plt.legend(['CAR','GMM Means'],numpoints=1,loc='upper left')
        


        # Full AR with all limits
        plt.figure()
        plt.plot(rho_a_all/params['Re'],drho_a_all/1000.,'ro',markeredgecolor='r',markersize=2)
        plt.plot(rho_e_all/params['Re'],drho_e_all/1000.,'bo',markeredgecolor='b',markersize=2)
        plt.plot(rho_lim/params['Re'],drho_lim/1000.,'ko',markersize=2)
        plt.xlabel('Range [ER]')
        plt.ylabel('Range-Rate [km/s]')
        plt.legend(['SMA Limits','Ecc Limits','CAR'],numpoints=1)
        # plt.xlim([5.6,6.3])
        # plt.ylim([-2.,2.])

        
        plt.show()
    
    
    return GMM


def car_drho_limits(rho_vect, Zk, q_vect, dq_vect, params):
    '''
    This function computes a Gaussian Mixture Model (GMM) to approximate a 
    uniform distribution representing the Constrained Admissible Region (CAR)
    produced by a 4D optical measurement set, containing angles and 
    angle-rates, in particular topocentric right ascension and declination.
    
    The method is based on DeMars and Jah [1].
    

    Parameters
    ----------
    rho_vect : list
        range values where range-rate bounds are needed
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


def car_gmm_to_eci(GMM0, Zk, q_vect, dq_vect, params):
    '''
    This function coverts the CAR GMM in range/range-rate space to ECI
    cartesian coordinates using an unscented transform.

    Parameters
    ------
    GMM0 : list of GM component weights, means, covariance matrices [rho, drho]
            [w,m,P]
            w = list of weights
            m = list of means (numpy px1 arrays)
            P = list of covars (numpy pxp arrays)
    Zk : 4x1 numpy array
        measurement vector containing topocentric RA/DEC and rates [rad, rad/s]
    q_vect : 3x1 numpy array
        sensor position vector in ECI [m]
    dq_vect : 3x1 numpy array
        sensor velocity vector in ECI [m/s]
    params : dictionary
        additional parameters (CAR limits)

    Returns
    ------
    GMM = list of GM component weights, means, covariance matrices [ECI]
            [w,m,P]
            w = list of weights
            m = list of means (numpy nx1 arrays)
            P = list of covars (numpy nxn arrays)
    
    '''
    
    #Break out GMM
    w = GMM0[0]
    m0 = GMM0[1]
    P0 = GMM0[2]
    
    # Retrieve inputs
    sigma_dict = params['sigma_dict']
    
    # Setup for unscented transform
    params['q_vect'] = q_vect
    params['dq_vect'] = dq_vect

    #Get sigmas for meas_types
    meas_types = ['ra', 'dec', 'dra', 'ddec']
    var_vect = []
    for meas in meas_types:
        var_vect.append((sigma_dict[meas])**2.)
    
    #For each GM component use unscented transform to put in ECI
    L = len(w)
    m_list = []
    P_list = []
    for j in range(L):
        mj = m0[j]        
        mj = np.append(mj, Zk)
        mj = np.reshape(mj, (6,1))
        Pj = np.diag(P0[j])
        Pj = np.append(Pj, var_vect)
        Pj = np.diag(Pj)

        #Execute UT function
        m, P, dum = unscented_transform(mj, Pj, ut_car_to_eci, params)
        m_list.append(m)
        P_list.append(P)

    GMM = [w,m_list,P_list]
    
    
    
    return GMM


def unscented_transform(m1, P1, transform_fcn, params, alpha=1., pnorm=2.):
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
    params : dictionary
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
    kurt = math.gamma(5./pnorm)*math.gamma(1./pnorm)/(math.gamma(3./pnorm)**2.)
    beta = kurt - 1.
    kappa = kurt - float(L)

    # Compute sigma point weights
    lam = alpha**2.*(L + kappa) - L
    gam = np.sqrt(L + lam)
    Wm = 1./(2.*(L + lam)) * np.ones((1, 2*L))
    Wm = list(Wm.flatten())
    Wc = Wm.copy()
    Wm.insert(0, lam/(L + lam))
    Wc.insert(0, lam/(L + lam) + (1 - alpha**2 + beta))
    Wm = np.asarray(Wm)
    diagWc = np.diag(Wc)

    # Compute chi - baseline sigma points
    sqP = np.linalg.cholesky(P1)
    Xrep = np.tile(m1, (1, L))
    chi = np.concatenate((m1, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
    chi_diff = chi - np.dot(m1, np.ones((1, (2*L+1))))

    # Compute transformed sigma points
    Y = transform_fcn(chi, params)
    row2 = int(Y.shape[0])
    col2 = int(Y.shape[1])

    # Compute mean and covar
    m2 = np.dot(Y, Wm.T)
    m2 = np.reshape(m2, (row2, 1))
    Y_diff = Y - np.dot(m2, np.ones((1, col2)))
    P2 = np.dot(Y_diff, np.dot(diagWc, Y_diff.T))
    Pcross = np.dot(chi_diff,  np.dot(diagWc, Y_diff.T))

    return m2, P2, Pcross


def ut_car_to_eci(chi, params):
    '''
    Function for use with unscented_transform.
    Converts sigma point matrix from inertial cartesian coordinates to
    keplerian elements.

    Parameters
    ------
    chi : L x (2L+1) numpy array
      sigma point matrix
    params : dictionary
      input parameters

    Returns
    ------
    Y : m x (2L+1) numpy array
      transformed sigma point matrix
    '''

    #Station pos/vel in ECI
    q_vect = params['q_vect'].flatten()
    dq_vect = params['dq_vect'].flatten()

    #Initialize output
    Y = np.zeros(chi.shape)
    L = int(chi.shape[1])

    for ind in range(L):

        #Break out chi
        rho = float(chi[0,ind])
        drho = float(chi[1,ind])
        ra = float(chi[2,ind])
        dec = float(chi[3,ind])
        dra = float(chi[4,ind])
        ddec = float(chi[5,ind])

        #Unit vectors
        u_rho = np.array([ np.cos(ra)*np.cos(dec),  np.sin(ra)*np.cos(dec), np.sin(dec)])
        u_ra =  np.array([-np.sin(ra)*np.cos(dec),  np.cos(ra)*np.cos(dec),          0.])
        u_dec = np.array([-np.cos(ra)*np.sin(dec), -np.sin(ra)*np.sin(dec), np.cos(dec)])

        #Range and Range-Rate vectors
        rho_vect = rho*u_rho
        drho_vect = drho*u_rho + rho*dra*u_ra + rho*ddec*u_dec

        #Compute pos/vel in ECI and add to output
        r_vect = q_vect + rho_vect
        v_vect = dq_vect + drho_vect
        Y[:,ind] = np.append(r_vect, v_vect)

    return Y



