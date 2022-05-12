import numpy as np
from math import *
import os
import copy
import matplotlib.pyplot as plt
import pickle
import warnings

import filters.unscented_functions as uf
import utilities.conversions as conv

#import conversions as conv



#############################################################################
#    This file contains functions to implement Initial Orbit Determination.
#
#    1. CAR-IOD
#    
#    DeMars, K. and Jah, M., "Probabilistic Initial Orbit Determination Using
#    Gaussian Mixture Models," JGCD, 2013.
#
#
#############################################################################



#############################################################################
# Interface Functions
#############################################################################

def initiate_car(GMM0,meas_vect,stat_ecef,ti,inputs,sigma_dict) :



    #Output
    GMM = copy.deepcopy(GMM0)

    #Station Parameters
    dtheta = inputs['dtheta']
    theta = inputs['theta0'] + dtheta*ti
    q_vect = conv.ecef2eci(stat_ecef,theta)
    w_vect = np.array([0,0,dtheta])
    dq_vect = np.cross(w_vect,q_vect)

    #Vector of range values
    rho_vect = np.arange(0.,50000.,5.)

    #Set limits
    limits = {}
    limits['a_max'] = 43000.
    limits['a_min'] = 30000.
    limits['e_max'] = 0.4

    #Additional Inputs
    inputs['sig_rho_des'] = 5.   #km
    inputs['sig_drho_des'] = 0.080   #km/s

    #Get GMM representation of CAR
    GMM1 = car_gmm(rho_vect,meas_vect,q_vect,dq_vect,limits,inputs,plot_flag=0)

    #Convert to ECI
    inputs['q_vect'] = q_vect
    inputs['dq_vect'] = dq_vect
    GMM1 = car_gmm_to_eci(GMM1,meas_vect,sigma_dict,inputs)

    #Compute entropies and sigma points
    gam = inputs['gam']
    n = len(GMM0[1][0])
    entropy_list = []
    chi_v_list = []
    for j in xrange(0,len(GMM1[0])) :
        mj = GMM1[1][j]
        Pj = GMM1[2][j]

        #Compute entropy
        ej = aegis.compute_entropy(Pj)
        entropy_list.append(ej)
        
        #Get sigma points
        sqP = np.linalg.cholesky(Pj)
        Xrep = np.tile(mj,(1,n))
        chi = np.concatenate((mj, Xrep+(gam*sqP), Xrep-(gam*sqP)),axis=1)
        chi_v = np.reshape(chi,(n*(2*n+1),1),order='F')
        chi_v_list.append(chi_v)
        
    #Create new track label
    labelmax = max(GMM0[6])
    label_list = [labelmax+1]*len(GMM1[0])    

    #Add to existing state GMM
    GMM[0].extend(GMM1[0])
    GMM[1].extend(GMM1[1])
    GMM[2].extend(GMM1[2])
    GMM[3].extend(entropy_list)
    GMM[4].extend(chi_v_list)
    GMM[6].extend(label_list)

    

    return GMM
    


def initiate_car2(meas_vect,q_vect,inputs,sigma_dict,label) :


    #Station Parameters
    dtheta = inputs['dtheta']
    w_vect = np.array([0,0,dtheta])
    dq_vect = np.cross(w_vect,q_vect)

    #Vector of range values
    rho_vect = np.arange(0.,50000.,5.)

    #Set limits
    limits = {}
    limits['a_max'] = 42565.
    limits['a_min'] = 41764.
    limits['e_max'] = 0.1

    #Additional Inputs
    #inputs['sig_rho_des'] = 50.   #km
    #inputs['sig_drho_des'] = 0.5   #km/s
    inputs['sig_rho_des'] = 0.5   #km
    inputs['sig_drho_des'] = 0.02   #km/s

    #Get GMM representation of CAR
    GMM_meas = car_gmm(rho_vect,meas_vect,q_vect,dq_vect,limits,inputs,plot_flag=1)
    
    print 'GMM CAR weights'
    print GMM_meas[0]
    print 'sum w', sum(GMM_meas[0])
    print 'max w', max(GMM_meas[0])
    print 'min w', min(GMM_meas[0])
    

    #Convert to ECI
    inputs['q_vect'] = q_vect
    inputs['dq_vect'] = dq_vect
    GMM_state = car_gmm_to_eci(GMM_meas,meas_vect,sigma_dict,inputs)

    
    #Create new track labels
    label_list = [label]*len(GMM_meas[0])       

    return GMM_state, label_list


def car_wrapper(obsfile) :
    '''
    car_wrapper(obsfile)

    This function provides an interface to the CAR IOD functions. Input is
    measurements file, output is GMM in ECI cartesian coordinates.

    Inputs:
    obsfile = file path and name

    Outputs:
    GMM = list of GM component weights, means, covariance matrices [ECI]
        = [w,m,P]
            w = list of weights
            m = list of means (numpy nx1 arrays)
            P = list of covars (numpy nxn arrays)
    
    '''

    #Load measurements and other data
    pklFile = open( obsfile, 'rb' )
    data = pickle.load( pklFile )
    t_truth = data[0]
    measurements = data[1]
    X_truth = data[2]
    inputs = data[3]
    sigma_dict = data[4]
    stations = data[5]
    pklFile.close()

    t_obs = measurements['obs_time']


    #Vector of range values
    rho_vect = np.arange(0.,50000.,5.)

    #First observation
    t0 = t_obs[0]
    stat0 = measurements['station'][0]
    stat_ecef = stations[stat0]
    dtheta = inputs['dtheta']
    theta = inputs['theta0'] + dtheta*t0
    q_vect = conv.ecef2eci(stat_ecef,theta)
    w_vect = np.array([0,0,dtheta])
    dq_vect = np.cross(w_vect,q_vect)
    
    #True initial state
    ind = t_truth == t0
    Xo = X_truth[ind].flatten()

    #Compute angle rates from true Xo
    r_vect = Xo[0:3]
    v_vect = Xo[3:6]
    obs_vect = meas.compute_dra_ddec(r_vect,v_vect,q_vect,inputs)

    #Replace angles with measurements
    ra = measurements['ra'][0]
    dec = measurements['dec'][0]
    obs_vect[0] = ra
    obs_vect[1] = dec

    #Set limits
    limits = {}
    limits['a_max'] = 43000.
    limits['a_min'] = 30000.
    limits['e_max'] = 0.4

    #Additional Inputs
    inputs['sig_rho_des'] = 5.   #km
    inputs['sig_drho_des'] = 0.080   #km/s

    #Get GMM representation of CAR
    GMM = car_gmm(rho_vect,obs_vect,q_vect,dq_vect,limits,inputs,plot_flag=1)

    #Unscented transform GMM to ECI
    inputs['q_vect'] = q_vect
    inputs['dq_vect'] = dq_vect
    GMM = car_gmm_to_eci(GMM,obs_vect,sigma_dict,inputs)


    return GMM



def mult_car_wrapper(obsfile) :
    '''
    mult_car_wrapper(obsfile)

    This function provides an interface to the CAR IOD functions. Input is
    measurements file, output is GMM in ECI cartesian coordinates.

    Inputs:
    obsfile = file path and name

    Outputs:
    GMM = list of GM component weights, means, covariance matrices [ECI]
        = [w,m,P]
            w = list of weights
            m = list of means (numpy nx1 arrays)
            P = list of covars (numpy nxn arrays)
    
    '''

    #Load measurements and other data
    pklFile = open( obsfile, 'rb' )
    data = pickle.load( pklFile )
    multiobject = data[0]
    pklFile.close()

    #Initialize Output
    GMM = [[],[],[]]
    ncomp_list = []

    #Loop over number of objects
    nobj = len(multiobject)
    for obj_id in xrange(1,nobj+1) :

        t_truth = multiobject[obj_id]['t_truth']
        measurements = multiobject[obj_id]['measurements']
        X_truth = multiobject[obj_id]['X_truth']
        inputs = multiobject[obj_id]['inputs']
        sigma_dict = multiobject[obj_id]['sigma_dict']
        stations = multiobject[obj_id]['stations']

        t_obs = measurements['obs_time']


        #Vector of range values
        rho_vect = np.arange(0.,50000.,5.)

        #First observation
        t0 = t_obs[0]
        stat0 = measurements['station'][0]
        stat_ecef = stations[stat0]
        dtheta = inputs['dtheta']
        theta = inputs['theta0'] + dtheta*t0
        q_vect = conv.ecef2eci(stat_ecef,theta)
        w_vect = np.array([0,0,dtheta])
        dq_vect = np.cross(w_vect,q_vect)
        
        #True initial state
        ind = t_truth == t0
        Xo = X_truth[ind].flatten()

        #Compute angle rates from true Xo
        r_vect = Xo[0:3]
        v_vect = Xo[3:6]
        obs_vect = meas.compute_dra_ddec(r_vect,v_vect,q_vect,inputs)

        #Replace angles with measurements
        ra = measurements['ra'][0]
        dec = measurements['dec'][0]
        obs_vect[0] = ra
        obs_vect[1] = dec

        #Set limits
        limits = {}
        limits['a_max'] = 45000.
        limits['a_min'] = 30000.
        limits['e_max'] = 0.4

        #Additional Inputs
        inputs['sig_rho_des'] = 5.   #km
        inputs['sig_drho_des'] = 0.080   #km/s

        #Get GMM representation of CAR
        GMM1 = car_gmm(rho_vect,obs_vect,q_vect,dq_vect,limits,inputs,plot_flag=0)

        #Unscented transform GMM to ECI
        inputs['q_vect'] = q_vect
        inputs['dq_vect'] = dq_vect
        GMM1 = car_gmm_to_eci(GMM1,obs_vect,sigma_dict,inputs)

        GMM[0].extend(GMM1[0])
        GMM[1].extend(GMM1[1])
        GMM[2].extend(GMM1[2])

        ncomp_list.append(len(GMM1[0]))

    return GMM,ncomp_list



def car_gmm(rho_vect,obs_vect,q_vect,dq_vect,limits,inputs,plot_flag=0) :
    '''
    car_gmm(rho_vect,obs_vect,q_vect,dq_vect,limits,inputs,plot_flag=0)

    This function computes the GMM representation of the constrained
    admissible region (CAR).

    Inputs:
    rho_vect = numpy array of range values where range-rate bounds are needed
    obs_vect = 1x4 list of observed angles and rates
             = [ra dec dra ddec] in [rad, rad/s]
    q_vect = 1x3 numpy array of station position in ECI [km]
    dq_vect = 1x3 numpy array of station velocity in ECI [km/s]
    limits = dict of limits in SMA, ecc
    inputs = dict of input parameters (Re, GM, etc)

    Outputs:
    GMM = list of GM component weights, means, covariance matrices [rho,drho]
        = [w,m,P]
            w = list of weights
            m = list of means (numpy px1 arrays)
            P = list of covars (numpy pxp arrays)

    '''

    #Step 1: Compute CAR boundaries
    #Suppress warnings
    with warnings.catch_warnings() :
        warnings.simplefilter("ignore")        
        rho_lim,drho_lim,drho_dict,rho_a_all,rho_e_all,drho_a_all,drho_e_all = \
            car_drho_limits(rho_vect,obs_vect,q_vect,dq_vect,limits,inputs)

    print 'rho_vect',rho_vect
    print 'obs_vect',obs_vect
    print 'rho_lim', rho_lim
    print 'drho_lim', drho_lim
    print 'rho_a_all', rho_a_all
    print 'rho_e_all', rho_e_all
    


    #Step 2: Compute range marginal PDF quantities    
    a_rho = np.min(rho_lim)
    b_rho = np.max(rho_lim)    
    sig_rho, L_rho = car_sig_library(a_rho,b_rho,inputs['sig_rho_des'])

    #Means
    m_rho = []
    for i in xrange(0,L_rho) :
        m_rho.append(a_rho + (b_rho-a_rho)/(L_rho+1.)*(i+1.))

    #Covars
    P_rho = [sig_rho**2]*L_rho

    #Weights
    #Evaluate range marginal PDF at each range value
    p_vect = []
    psum = 0.
    rho_unique = np.unique(rho_lim)
    delta_rho = rho_unique[1] - rho_unique[0]
    for rho in rho_unique :
        drho_vect = drho_dict[rho]
        a_drho = np.min(drho_vect)
        b_drho = np.max(drho_vect)
        p_vect.append((b_drho-a_drho)*delta_rho)
        #p_vect.append((b_drho-a_drho)/(b_rho-a_rho))

    #p_vect = np.asarray(p_vect)/(sum(p_vect)*delta_rho)
    norm_fact = np.trapz(p_vect,rho_unique)
    p_vect = p_vect/norm_fact

    #check = np.dot(p_vect,[delta_rho]*len(p_vect))
    #print check

    #Compute H matrix
    M = len(p_vect)
    H = np.zeros((M,L_rho))
    for i in xrange(0,M) :
        for j in xrange(0,L_rho) :
            rhoi = rho_unique[i]
            mj = m_rho[j]
            sigj = np.sqrt(P_rho[j])
            H[i,j] = (1/(np.sqrt(2.*pi)*sigj))*exp(-0.5*((rhoi-mj)/sigj)**2.)

    #Compute weights (least squares fit)
    w_rho = np.dot( np.linalg.inv( np.dot(H.T,H) ), np.dot(H.T,p_vect))

    if abs(sum(w_rho) - 1.) > 0.1 :
        print 'Error: iod.car_gmm range weights not normalized!!'
        print w_rho
        print sum(w_rho)

    #Compute PDF sum
    g_approx = []
    for i in xrange(0,M) :
        gi = 0.
        rhoi = rho_unique[i]
        for j in xrange(0,L_rho) :            
            wj = w_rho[j]
            mj = m_rho[j]
            sigj = np.sqrt(P_rho[j])
            gi += wj*(1/(np.sqrt(2.*pi)*sigj))*exp(-0.5*((rhoi-mj)/sigj)**2.)
        g_approx.append(gi)    

    #Step 3: Compute range-rate marginal PDF quantities - store in GMM
    #Get drho limits for m_rho
    rho_lim2,drho_lim2,drho_dict2 = car_drho_limits(m_rho,obs_vect,q_vect,dq_vect,limits,inputs)[0:3]
    
    w = []
    m = []
    P = []
    sig_drho_max = 0.
    xx = []
    yy = []
    zz = []
    
    for i in xrange(0,L_rho) :
        #Get values from Range PDF
        wi = w_rho[i]
        mi = m_rho[i]
        Pi = P_rho[i]

        #Get values from Range-Rate PDF
        drho_vect = drho_dict2[mi]
        for k in xrange(0,len(drho_vect)/2) :
            drho_k = drho_vect[2*k:2*k+2]
            a_drho = np.min(drho_k)
            b_drho = np.max(drho_k)
            sig_drho,L_drho = car_sig_library(a_drho,b_drho,inputs['sig_drho_des'])
            if sig_drho > sig_drho_max :
                sig_drho_max = sig_drho.copy()


            #Weights, means, covar for this rho
            wj = 1./L_drho
            Pj = sig_drho**2.
            for j in xrange(0,L_drho) :
                mj = a_drho + (b_drho-a_drho)/(L_drho + 1.)*(j+1.)
                w.append(wi*wj)
                m.append(np.array([[mi],[mj]]))
                P.append(np.array([[Pi,0.],[0.,Pj]]))

    

    GMM = [w,m,P]


    #Plot checks
    if plot_flag :

        print 'L_rho = ',L_rho
        print 'sig_rho = ',sig_rho
        print 'L_tot = ',len(w)
        print 'sig_drho_max = ',sig_drho_max

        mrho_RE = [mi[0]/inputs['Re'] for mi in m]
        mdrho = [mi[1] for mi in m]

        #Range Marginal PDF
        plt.figure()
        plt.plot(rho_unique/inputs['Re'],p_vect,'b--',lw=2)
        plt.plot(rho_unique/inputs['Re'],g_approx,'r--',lw=2)
        #plt.title('Range Marginal PDF')
        plt.legend(['PDF','GM Approx'])
        plt.xlabel('Range [ER]')
        

        #CAR with GM mean locations
        plt.figure()
        plt.plot(rho_lim/inputs['Re'],drho_lim,'k.')
        plt.plot(mrho_RE,mdrho,'k+')
        #plt.title('Constrained Admissible Region')
        plt.xlabel('Range [ER]')
        plt.ylabel('Range-Rate [km/s]')
        plt.legend(['CAR','GMM Means'],numpoints=1)
        

        #CAR PDF
        #plt.figure()
        

        #Full AR with all limits
        plt.figure()
        plt.plot(rho_a_all/inputs['Re'],drho_a_all,'ro',markeredgecolor='r',markersize=2)
        plt.plot(rho_e_all/inputs['Re'],drho_e_all,'bo',markeredgecolor='b',markersize=2)
        plt.plot(rho_lim/inputs['Re'],drho_lim,'ko',markersize=2)
        plt.xlabel('Range [ER]')
        plt.ylabel('Range-Rate [km/s]')
        plt.legend(['SMA Limits','Ecc Limits','CAR'],numpoints=1)
        plt.xlim([5.6,6.3])
        plt.ylim([-2.,2.])
        
        plt.figure()
        plt.plot(rho_a_all/inputs['Re'],drho_a_all,'ko',markeredgecolor='k',markersize=2)
        plt.plot(rho_e_all/inputs['Re'],drho_e_all,'k_',markeredgecolor='k',markersize=4)
        plt.plot(rho_lim/inputs['Re'],drho_lim,'ko',markersize=4)
        plt.xlabel('Range [ER]')
        plt.ylabel('Range-Rate [km/s]')
        plt.legend(['SMA Limits','Ecc Limits','CAR'],numpoints=1)
        plt.xlim([5.6,6.3])
        plt.ylim([-2.,2.])
                
        
        plt.show()
                     
    return GMM



def car_gmm_to_eci(GMM0,obs_vect,sigma_dict,inputs) :
    '''
    car_gmm_to_eci(GMM0,obs_vect,sigma_dict,inputs)

    This function coverts the CAR GMM in range/range-rate space to ECI
    cartesian coordinates using an unscented transform.

    Inputs:
    GMM0 = list of GM component weights, means, covariance matrices [rho,drho]
         = [w,m,P]
            w = list of weights
            m = list of means (numpy px1 arrays)
            P = list of covars (numpy pxp arrays)
    obs_vect = 1x4 list of observed angles and rates
             = [ra dec dra ddec] in [rad, rad/s]
    sigma_dict = dict of measurement uncertainties
    inputs = dict of input parameters, includes station pos/vel in ECI

    Outputs:
    GMM = list of GM component weights, means, covariance matrices [ECI]
        = [w,m,P]
            w = list of weights
            m = list of means (numpy nx1 arrays)
            P = list of covars (numpy nxn arrays)

    '''


    #Break out GMM
    w = GMM0[0]
    m0 = GMM0[1]
    P0 = GMM0[2]
    
    # Inputs
    car_sigma_scale = inputs['car_sigma_scale']

    #Get sigmas for meas_types
    meas_types = ['ra','dec','dra','ddec']
    sig_vect = []
    for meas in meas_types :
        sig_vect.append((sigma_dict[meas]*car_sigma_scale)**2.)
    
    #For each GM component use unscented transform to put in ECI
    L = len(w)
    m_list = []
    P_list = []
    for j in xrange(0,L) :
        mj = m0[j]        
        mj = np.append(mj,obs_vect)
        mj = np.reshape(mj,(6,1))
        Pj = np.diag(P0[j])
        Pj = np.append(Pj,sig_vect)
        Pj = np.diag(Pj)

        #Execute UT function
        m, P, dum = uf.unscented_transform(mj, Pj, uf.ut_car_to_eci, inputs)
        m_list.append(m)
        P_list.append(P)

    GMM = [w,m_list,P_list]

    return GMM
             

def car_sig_library(a,b,sig_des) :
    '''
    car_sig_library(a,b,sig_des)

    This function returns the sigma value required to approximate a uniform
    distribution with a GMM with "L" homoscedastic, evenly spaced, and
    evenly weighted components. Library based on standard uniform distribution
    (a = 0, b = 1, p = 1/(b-a)).  Will return result for minimum number of
    components required to achieve desired std or lower, up to max of 15
    components.

    Inputs:
    a = lower limit (float)
    b = upper limit (float)
    sig_des = desired standard deviation (float)

    Outputs:
    sig = actual standard deviation (float)
    L = number of components (int)

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

    for L in xrange(1,16) :
        sig = (b-a) * sig_dict[L]
        if sig < sig_des :
            break

    return sig,L


def car_drho_limits(rho_vect,obs_vect,q_vect,dq_vect,limits,inputs) :
    '''
    car_drho_limits(rho_vect,obs_vect,q_vect,dq_vect,limits,inputs)

    This function computes the range and range-rate bounds of the
    constrained admissible region (CAR).

    Inputs:
    rho_vect = numpy array of range values where range-rate bounds are needed
    obs_vect = 1x4 list of observed angles and rates
             = [ra dec dra ddec] in [rad, rad/s]
    q_vect = 1x3 numpy array of station position in ECI [km]
    dq_vect = 1x3 numpy array of station velocity in ECI [km/s]
    limits = dict of limits in SMA, ecc
    inputs = dict of input parameters (Re, GM, etc)

    Outputs:
    rho_output = numpy array of range values
    drho_output = numpy array of corresponding range-rate limits
    drho_dict = dict of numpy arrays for drho limits, keys are rho values

    '''

    #Break out imputs
    GM = inputs['GM']   #km^3/s^2
    Re = inputs['Re']   #km
    a_max = limits['a_max']     #km
    a_min = limits['a_min']     #km
    e_max = limits['e_max']

    #Break out obs_vect
    ra = obs_vect[0]    #rad
    dec = obs_vect[1]   #rad
    dra = obs_vect[2]   #rad/s
    ddec = obs_vect[3]  #rad/s

    #Unit vectors
    u_rho = np.array([cos(ra)*cos(dec), sin(ra)*cos(dec), sin(dec)])
    u_ra = np.array([-sin(ra)*cos(dec), cos(ra)*cos(dec), 0])
    u_dec = np.array([-cos(ra)*sin(dec), -sin(ra)*sin(dec), cos(dec)])

    #Semi-Major Axis Constraint
    #Compute coefficients
    w0 = np.dot(q_vect,q_vect)
    w1 = 2.*np.dot(dq_vect,u_rho)
    w2 = dra**2.*cos(dec)**2. + ddec**2.
    w3 = 2.*dra*np.dot(dq_vect,u_ra) + 2.*ddec*np.dot(dq_vect,u_dec)
    w4 = np.dot(dq_vect,dq_vect)
    w5 = 2.*np.dot(q_vect,u_rho)

    #Compute energy limits
    E_max = -GM/(2.*a_max)
    if a_min == 0. :
        a_min = 1e-10
    E_min = -GM/(2.*a_min)

        

    #Eccentricity Constraint
    e_min = 0.

    #Angular Momentum Components
    h1 = np.cross(q_vect,u_rho)
    h2 = np.cross(u_rho, (dra*u_ra + ddec*u_dec))
    h3 = np.cross(u_rho,dq_vect) + np.cross(q_vect,(dra*u_ra + ddec*u_dec))
    h4 = np.cross(q_vect,dq_vect)

    #Compute coefficients
    c0 = np.dot(h1,h1)
    c1 = 2.*np.dot(h1,h2)
    c2 = 2.*np.dot(h1,h3)
    c3 = 2.*np.dot(h1,h4)
    c4 = np.dot(h2,h2)
    c5 = 2.*np.dot(h2,h3)
    c6 = 2.*np.dot(h2,h4) + np.dot(h3,h3)
    c7 = 2.*np.dot(h3,h4)
    c8 = np.dot(h4,h4)

    #Initialize Output
    rho_output = np.array([])
    drho_output = np.array([])
    rho_a_all = np.array([])
    rho_e_all = np.array([])
    drho_a_all = np.array([])
    drho_e_all = np.array([])
    drho_dict = {}

    #Loop over range values    
    for ii in xrange(0,len(rho_vect)) :

        #Current range value
        rho = rho_vect[ii]

        #Compute F for current range
        F = w2*rho**2. + w3*rho + w4 - 2.*GM/np.sqrt(rho**2. + w5*rho + w0)

        #Compute values of drho for SMA limits
        rad_max = (w1/2.)**2. - F + 2.*E_max
        rad_min = (w1/2.)**2. - F + 2.*E_min

        drho_a = np.array([])
        if rad_max >= 0. :
            rad_max = np.sqrt(rad_max)
            drho_a = np.append(drho_a,np.array([-w1/2. + rad_max, -w1/2. - rad_max]))
        if rad_min >= 0. :
            rad_min = np.sqrt(rad_min)
            drho_a = np.append(drho_a,np.array([-w1/2. + rad_min, -w1/2. - rad_min]))

        #Eccentricity Constraints
        #Compute P and U for current range
        P = c1*rho**2. + c2*rho + c3
        U = c4*rho**4. + c5*rho**3. + c6*rho**2. + c7*rho + c8

        #Compute coefficients
        a0_max = F*U + GM**2.*(1.-e_max**2.)
        a0_min = F*U + GM**2.*(1.-e_min**2.)
        a1 = F*P + w1*U
        a2 = U + c0*F + w1*P
        a3 = P + c0*w1
        a4 = c0

        #Solve the quartic equation
        r = np.roots(np.array([a4,a3,a2,a1,a0_max]))
        drho_ecc =  np.array([])
        for i in xrange(0,len(r)) :
            if np.isreal(r[i]) :
                drho_ecc = np.append(drho_ecc,float(r[i]))

        #Set up output
        #Output everything
        drho_a_all = np.append(drho_a_all,drho_a)
        drho_e_all = np.append(drho_e_all,drho_ecc)
        for ii in xrange(0,len(drho_a)) :
            rho_a_all = np.append(rho_a_all,rho)
        for ii in xrange(0,len(drho_ecc)) :
            rho_e_all = np.append(rho_e_all,rho)
                
        #Return only CAR values (ensure both arrays have values)
        if len(drho_ecc) and len(drho_a) :            

            if len(drho_ecc) == 2 :
                
                if len(drho_a) == 2 :
                    rho_output = np.append(rho_output,np.array([rho,rho]))
                    drho_vect = np.append(drho_ecc,drho_a)
                    drho_vect = np.sort(drho_vect)
                    drho_output = np.append(drho_output,drho_vect[1:3])
                    drho_dict[rho] = drho_vect[1:3]

                if len(drho_a) == 4 :
                    drho_a = np.sort(drho_a)
                    drho_ecc = np.sort(drho_ecc)

                    #Positive side
                    drho_vect1 = np.array([])
                    if drho_a[2] < np.max(drho_ecc) :
                        rho_output = np.append(rho_output,np.array([rho,rho]))
                        
                        if drho_a[3] < np.max(drho_ecc) :
                            drho_vect1 = drho_a[2:4]
                            drho_output = np.append(drho_output,drho_vect1)
                        else :
                            drho_vect1 = np.array([drho_a[2],np.max(drho_ecc)])
                            drho_output = np.append(drho_output,drho_vect1)
                            
                        #drho_dict[rho] = drho_vect1

                    #Negative Side
                    drho_vect2 = np.array([])
                    if drho_a[1] > np.min(drho_ecc) :
                        rho_output = np.append(rho_output,np.array([rho,rho]))

                        if drho_a [0] > np.min(drho_ecc) :
                            drho_vect2 = drho_a[0:2]
                            drho_output = np.append(drho_output,drho_vect2)
                        else :
                            drho_vect2 = np.array([drho_a[1],np.min(drho_ecc)])
                            drho_output = np.append(drho_output,drho_vect2)
                            
                    drho_dict[rho] = np.append(drho_vect1,drho_vect2)                                           

            if len(drho_ecc) == 4 :

                if len(drho_a) == 2 :
                    rho_output = np.append(rho_output,np.array([rho,rho,rho,rho]))
                    drho_vect = np.append(drho_a,drho_ecc)
                    drho_vect = np.sort(drho_vect)
                    drho_output = np.append(drho_output,drho_vect[1:5])
                    drho_dict[rho] = drho_vect[1:5]

                if len(drho_a) == 4 :
                    drho_a = np.sort(drho_a)
                    drho_ecc = np.sort(drho_ecc)

                    #Positive Side
                    drho_vect1 = np.array([])
                    if drho_a[2] < np.max(drho_ecc) :
                        rho_output = np.append(rho_output,np.array([rho,rho]))

                        if drho_a[3] < np.max(drho_ecc) :
                            drho_vect1 = drho_a[2:4]
                            drho_output = np.append(drho_output,drho_vect1)
                        else :
                            drho_vect1 = np.array([drho_a[2],np.max(drho_ecc)])
                            drho_output = np.append(drho_output,drho_vect1)

                        #drho_dict[rho] = drho_vect1

                    #Negative Side
                    drho_vect2 = np.array([])
                    if drho_a[1] > np.min(drho_ecc) :
                        rho_output = np.append(rho_output,np.array([rho,rho]))

                        if drho_a[0] > np.min(drho_ecc) :
                            drho_vect2 = drho_a[0:2]
                            drho_output = np.append(drho_output,drho_vect2)
                        else :
                            drho_vect2 = np.array([drho_a[1],np.min(drho_ecc)])
                            drho_output = np.append(drho_output,drho_vect2)

                    drho_dict[rho] = np.append(drho_vect1,drho_vect2)
  

    return rho_output,drho_output,drho_dict,rho_a_all,rho_e_all,drho_a_all,drho_e_all







def gauss_iod(JD_list, ra_list, dec_list, stat_ecef, inputs):
    
    
    # Break out inputs
    GM = inputs['GM']
    
    # Choose indices and reduce lists as needed
    if len(JD_list) > 3:
        inds = [0, 1, -1]
        
    JD_list = [JD_list[ii] for ii in inds]
    ra_list = [ra_list[ii] for ii in inds]
    dec_list = [dec_list[ii] for ii in inds]
    
    # Breakdown
    JD1 = float(JD_list[0])
    JD2 = float(JD_list[1])
    JD3 = float(JD_list[2])
    ra1 = float(ra_list[0])
    ra2 = float(ra_list[1])
    ra3 = float(ra_list[2])
    dec1 = float(dec_list[0])
    dec2 = float(dec_list[1])
    dec3 = float(dec_list[2])
    
    # Times
    tau1 = (JD1 - JD2)*86400.
    tau3 = (JD3 - JD2)*86400.
    
    # Coefficients
    a1 = tau3/(tau3 - tau1)
    a3 = -tau1/(tau3 - tau1)
    a1u = tau3*((tau3 - tau1)**2. - tau3**2.)/(6.*(tau3 - tau1))
    a3u = -tau1*((tau3 - tau1)**2. - tau1**2.)/(6.*(tau3 - tau1))
    
    # Unit Vectors
    L1 = np.reshape([cos(dec1)*cos(ra1), cos(dec1)*sin(ra1), sin(dec1)], (3,1))
    L2 = np.reshape([cos(dec2)*cos(ra2), cos(dec2)*sin(ra2), sin(dec2)], (3,1))
    L3 = np.reshape([cos(dec3)*cos(ra3), cos(dec3)*sin(ra3), sin(dec3)], (3,1))
    
    L = np.concatenate((L1, L2, L3), axis=1)    
    Linv = np.linalg.inv(L)
    
    # Site Coordinates
#    print stat_ecef
#    print JD1
#    print inputs
    stat_eci1 = conv.ecef2eci(stat_ecef, inputs, JD1)
    stat_eci2 = conv.ecef2eci(stat_ecef, inputs, JD2)
    stat_eci3 = conv.ecef2eci(stat_ecef, inputs, JD3)
    
    R = np.concatenate((stat_eci1, stat_eci2, stat_eci3), axis=1)

    # Calculations
    M = np.dot(Linv, R)
    d1 = M[1,0]*a1 - M[1,1] + M[1,2]*a3
    d2 = M[1,0]*a1u + M[1,2]*a3u
    
    C = float(np.dot(L2.T, stat_eci2))
    
    # Solve for r2
    poly2 = np.array([1., 0., -(d1**2. + 2.*C*d1 + np.linalg.norm(stat_eci2)**2.),
                      0., 0., -2.*GM*(C*d2 + d1*d2), 0., 0., -GM**2.*d2**2.])
    roots2 = np.roots(poly2)
    
    # Find positive real roots
    real_inds = list(np.where(np.isreal(roots2))[0])
    r2_list = []
    for ind in real_inds:
        
        if float(roots2[ind]) > 0.:
            r2 = float(roots2[ind])        
            r2_list.append(r2)
        
    if len(r2_list) != 1:
        print r2_list
        print poly2
        print roots2
        print real_inds
        mistake
        
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
    
    r1_vect = rho1*L1 + stat_eci1
    r2_vect = rho2*L2 + stat_eci2
    r3_vect = rho3*L3 + stat_eci3
    
    # Use Gibbs to compute v2_vect
    v2_vect = gibbs(r1_vect,r2_vect,r3_vect,GM)

    # Compute cartesian state at t1 assuming Keplerian dynamics
    X2 = np.concatenate((r2_vect, v2_vect), axis=0)
    X1 = conv.element_conversion(X2, 1, 1, dt=tau1)
    
    return X1



def gibbs(r1_vect, r2_vect, r3_vect, GM):
    
    # Vallado Algorithm 53

    z12 = np.cross(r1_vect, r2_vect, axis=0)
    z23 = np.cross(r2_vect, r3_vect, axis=0)
    z31 = np.cross(r3_vect, r1_vect, axis=0)
    
    # Tests
    alpha_cop = pi/2. - acos(float(np.dot(z23.T,r1_vect))/(np.linalg.norm(z23)*np.linalg.norm(r1_vect)))
    alpha_12 = acos(float(np.dot(r1_vect.T,r2_vect))/(np.linalg.norm(r1_vect)*np.linalg.norm(r2_vect)))
    alpha_23 = acos(float(np.dot(r2_vect.T,r3_vect))/(np.linalg.norm(r2_vect)*np.linalg.norm(r3_vect)))

    if abs(alpha_cop) > 5.*pi/180.:
        print 'Error: Not Coplanar! alpha = ', alpha_cop 
        mistake

    # Compute vectors
    r1 = np.linalg.norm(r1_vect)
    r2 = np.linalg.norm(r2_vect)
    r3 = np.linalg.norm(r3_vect)
    
    N = r1*z23 + r2*z31 + r3*z12
    D = z12 + z23 + z31
    S = (r2 - r3)*r1_vect + (r3 - r1)*r2_vect + (r1 - r2)*r3_vect
    B = np.cross(D, r2_vect, axis=0)
    
    Lg = np.sqrt(GM/(np.linalg.norm(N)*np.linalg.norm(D)))
    
    v2_vect = Lg/r2*B + Lg*S
        
    
    return v2_vect




