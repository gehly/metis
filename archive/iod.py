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






#def compute_extremal_dist(r0_vect, rf_vect, v0_vect, vf_vect, dtheta, M, lam, GM):
#    '''
#    
#    '''
#    
#    # Default, min/max of r0, rf
#    r0 = np.linalg.norm(r0_vect)
#    v0 = np.linalg.norm(v0_vect)
#    rf = np.linalg.norm(rf_vect)
#    r0_vect_hat = r0_vect/r0
#    rf_vect_hat = rf_vect/rf
#    minimum_distance = min(r0, rf)
#    maximum_distance = max(r0, rf)
#    
#    # Semi-major axis
#    a = 1./(2./r0 - v0**2./GM)
#
#    # Eccentricity vector 
#    h0_vect = np.array([[float(r0_vect[1]*v0_vect[2] - r0_vect[2]*v0_vect[1])],
#                        [float(r0_vect[2]*v0_vect[0] - r0_vect[0]*v0_vect[2])],
#                        [float(r0_vect[0]*v0_vect[1] - r0_vect[1]*v0_vect[0])]])
#    
#    cross1 =  np.array([[float(v0_vect[1]*h0_vect[2] - v0_vect[2]*h0_vect[1])],
#                        [float(v0_vect[2]*h0_vect[0] - v0_vect[0]*h0_vect[2])],
#                        [float(v0_vect[0]*h0_vect[1] - v0_vect[1]*h0_vect[0])]])
#    
#    e0_vect = cross1/GM - r0_vect/r0
#    e = np.linalg.norm(e0_vect)
#    e0_vect_hat = e0_vect/e
#    
#    # Apses
#    periapsis = a*(1. - e)
#    apoapsis = np.inf
#    if e < 1.:
#        apoapsis = a*(1. + e)
#        
#    # Check if the trajectory goes through periapsis
#    if M > 0:
#        
#        # Multirev case, must be elliptical and pass through both periapsis and
#        # apoapsis
#        minimum_distance = periapsis
#        maximum_distance = apoapsis
#        
#    else:
#        
#        # Compute true anomaly at t0 and tf
#        pm0 = np.sign(r0*r0*np.dot(e0_vect.T, v0_vect) - np.dot(r0_vect.T, e0_vect)*np.dot(r0_vect.T, v0_vect))
#        pmf = np.sign(rf*rf*np.dot(e0_vect.T, vf_vect) - np.dot(rf_vect.T, e0_vect)*np.dot(rf_vect.T, vf_vect))
#
#        theta0 = pm0 * math.acos(max(-1, min(1, np.dot(r0_vect_hat.T, e0_vect_hat))))
#        thetaf = pmf * math.acos(max(-1, min(1, np.dot(rf_vect_hat.T, e0_vect_hat))))
#        
#        if theta0*thetaf < 0.:
#            
#            # Initial and final positions are on opposite sides of symmetry axis
#            # Minimum and maximum distance depends on dtheta and true anomalies
#            if abs(abs(theta0) + abs(thetaf) - dtheta) < 5.*np.finfo(float).eps:
#                minimum_distance = periapsis
#            
#            # This condition can only be false for elliptic cases, and if it is
#            # false, the orbit has passed through apoapsis
#            else:
#                maximum_distance = apoapsis
#                
#        else:
#            
#            # Initial and final positions are on the same side of symmetry axis
#            # If it is a Type II transfer (longway) then the object must
#            # pass through both periapsis and apoapsis
#            if lam < 0.:
#                minimum_distance = periapsis
#                if e < 1.:
#                    maximum_distance = apoapsis
#                    
#                    
#    extremal_distances = [minimum_distance, maximum_distance]
#
#    return extremal_distances





#def battin_lambert(r1_vect, r2_vect, tof, GM, transfer_type):
#    '''
#    
#    
#    '''
#    
#    # Compute chord and unit vectors
#    r1_vect = np.reshape(r1_vect, (3,1))
#    r2_vect = np.reshape(r2_vect, (3,1))
#    c_vect = r2_vect - r1_vect
#    
#    r1 = np.linalg.norm(r1_vect)
#    r2 = np.linalg.norm(r2_vect)
#    c = np.linalg.norm(c)
#    
#    s = 0.5 * (r1 + r2 + c)
#    eps = (r2 - r1)/r1
#    
#    if transfer_type == 1:
#        t_m = 1.
#    elif transfer_type == 2:
#        t_m = -1.
#    
#    # Difference in true anomaly angle
#    cos_dtheta = np.dot(r1_vect.T, r2_vect)/(r1*r2)
#    sin_dtheta = t_m * np.sqrt(1. - cos_dtheta**2.)
#    dtheta = math.atan2(sin_dtheta, cos_dtheta)
#    if dtheta < 0.:
#        dtheta += 2.*math.pi
#    
#    
#    tan2_2w = (eps**2./4.)/(np.sqrt(r2/r1) + (r2/r1)*(2. + np.sqrt(r2/r1)))
#    r_op = np.sqrt(r1*r2)*(math.cos(dtheta/4.)**2. + tan2_2w)
#    
#    if dtheta < math.pi:
#        l = (math.sin(dtheta/4.)**2. + tan2_2w)/(math.sin(dtheta/4.)**2. + tan2_2w + math.cos(dtheta/2.))
#    else:
#        l = (math.cos(dtheta/4.)**2. + tan2_2w - math.cos(dtheta/2.))/(math.cos(dtheta/4.)**2. + tan2_2w)
#        
#    m = GM*tof**2./(8.*r_op**3.)
#    
#    # Let x = l for elliptical orbit, else x = 0
#    x = l
#    
#    
#    diff = 1.
#    tol = 1e-12
#    while diff > tol:
#        
#        eta = x/(np.sqrt(1. + x) + 1)**2.
#        zeta = compute_zeta(x, eta)
#        
#    
#    
#    
#    
#    
#    return v1_vect, v2_vect
#
#
#
#
#
#def compute_zeta(x, eta):
#    
#    for nn in range(4, 10):
#        c_n = nn**2./((2.*nn)**2. - 1.)
#        
#    
#    nn = 10
#    frac = 1.
#    while nn > 4:
#        cn = nn**2./((2.*nn)**2. - 1.)
#        denom = 1 + cn*eta
#        
#        
#        
#        
#        
#    
#    
#    return



#def multirev_lambert(r0_vect, rf_vect, tof, m, GM, transfer_type, branch):
#    '''
#    This function implements two methods to solve Lambert's Problem, a fast 
#    method developed by Dr Izzo of ESA and a robust method based on work by
#    Lancaster and Blanchard [2], and Gooding [3]. This code is written 
#    following a MATLAB version written by Rody Oldenhuis, copyright below.
#    
#    Source code from https://github.com/rodyo/FEX-Lambert
#    
#    Parameters
#    ------
#    r0_vect : 3x1 numpy array
#        position vector at t0 [km]
#    rf_vect : 3x1 numpy array
#        position vector at tf [km]
#    tof : float
#        time of flight [sec]
#    m : int
#        number of complete orbit revolutions
#    GM : float
#        graviational parameter of central body [km^3/s^2]
#        
#    Returns
#    ------
#    v0_vect : 3x1 numpy array
#        velocity vector at t0 [km/s]
#    vf_vect : 3x1 numpy array
#        velocity vector at tf [km/s]
#    extremal_distances : list
#        min and max distance from central body during orbit [km]
#    exit_flag : int
#        +1 : success
#        -1 : problem has no solution
#        -2 : both algorithms failed (should not occur)
#        
#    References
#    ------
#    [1] Izzo, D. ESA Advanced Concepts team. Code used available in MGA.M, on
#         http://www.esa.int/gsp/ACT/inf/op/globopt.htm. Last retreived Nov, 2009.
#         (broken link)
#     
#    [2] Lancaster, E.R. and Blanchard, R.C. "A unified form of Lambert's theorem."
#         NASA technical note TN D-5368,1969.
#     
#    [3] Gooding, R.H. "A procedure for the solution of Lambert's orbital boundary-value
#         problem. Celestial Mechanics and Dynamical Astronomy, 48:145-165,1990.
#    
#    
#    Copyright
#    ------
#    Copyright (c) 2018, Rody Oldenhuis
#    All rights reserved.
#    
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are met:
#    
#    1. Redistributions of source code must retain the above copyright notice, this
#       list of conditions and the following disclaimer.
#    2. Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#    
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#    
#    The views and conclusions contained in the software and documentation are those
#    of the authors and should not be interpreted as representing official policies,
#    either expressed or implied, of this project.
#    
#    '''
#    
#    # Fast Lambert
#    v0_vect, vf_vect, extremal_distances, exit_flag = \
#        fast_lambert(r0_vect, rf_vect, tof, m, GM, transfer_type, branch)
#        
#    # If not successful, run robust solver
#    if exit_flag < 0:
#        v0_vect, vf_vect, extremal_distances, exit_flag = \
#            robust_lambert(r0_vect, rf_vect, tof, m, GM, transfer_type, branch)
#    
#            
#
#
#
#
#    
#    
#    return v0_vect, vf_vect, extremal_distances, exit_flag



#
#def fast_lambert(r0_vect, rf_vect, tof, m, GM, transfer_type, branch):
#    '''
#    This function implements a computationally efficient but less robust 
#    approach to solve Lambert's Problem developed by Dario Izzo (ESA).
#    
#    Parameters
#    ------
#    r0_vect : 3x1 numpy array
#        position vector at t0 [km]
#    rf_vect : 3x1 numpy array
#        position vector at tf [km]
#    tof : float
#        time of flight [sec]
#    m : int
#        number of complete orbit revolutions
#    GM : float
#        graviational parameter of central body [km^3/s^2]
#        
#    Returns
#    ------
#    v0_vect : 3x1 numpy array
#        velocity vector at t0 [km/s]
#    vf_vect : 3x1 numpy array
#        velocity vector at tf [km/s]
#    extremal_distances : list
#        min and max distance from central body during orbit [km]
#    exit_flag : int
#        +1 : success
#        -1 : fail
#    '''
#    
#        
#    # Initialize
#    tol = 1e-14
#    max_iters = 15
#    exit_flag = -1
#    r0_vect = np.reshape(r0_vect, (3,1))
#    rf_vect = np.reshape(rf_vect, (3,1))
#    
#    # Normalize units
#    r0 = np.linalg.norm(r0_vect)
#    v0 = np.sqrt(GM/r0)
#    T = r0/v0
#    
#    r0_vect = r0_vect/r0
#    rf_vect = rf_vect/r0
#    tf = tof/T
#    logt = math.log(tf)
#    
#    # Check non-dimensional geometry
#    rf_norm = np.linalg.norm(rf_vect)
#    dtheta = math.acos(max(-1, min(1, np.dot(r0_vect.T, rf_vect)/rf_norm)))
#    
#    # Check for Type I (short) or II (long) transfer and adjust
#    type_factor = 1.
#    if transfer_type == 2:
#        dtheta = 2.*math.pi - dtheta
#        type_factor = -1.
#        
#    # Derived non-dimensional quantities
#    c = np.sqrt(1. + rf_norm**2. - 2.*rf_norm*math.cos(dtheta)) # chord
#    s = (1. + rf_norm + c)/2.                                   # semi-parameter
#    a_min = s/2.                                                # min energy ellipse SMA
#    Lambda = np.sqrt(rf_norm)*math.cos(dtheta/2.)/s             # Lambda parameter from Battin
#
#    r_cross_vect = np.array([[float(r0_vect[1]*rf_vect[2] - r0_vect[2]*rf_vect[1])],
#                             [float(r0_vect[2]*rf_vect[0] - r0_vect[0]*rf_vect[2])],
#                             [float(r0_vect[0]*rf_vect[1] - r0_vect[1]*rf_vect[0])]])
#
#    r_cross = np.linalg.norm(r_cross_vect)
#    r_cross_hat = r_cross_vect/r_cross                          # unit vector
#
#    # Setup initial values
#    if m == 0:
#        
#        # Single revolution (1 solution)
#        inn1 = -0.5233              # first initial guess
#        inn2 = 0.5233               # second initial guess
#        x1 = math.log(1. + inn1)    # transformed first initial guess
#        x2 = math.log(1. + inn2)    # transformed first second guess
#        
#    else:
#        
#        # Multirev case, select right or left branch
#        if branch == 'left':
#            inn1 = -0.5234          # first initial guess
#            inn2 = -0.2234          # second initial guess
#        
#        if branch == 'right':
#            inn1 = 0.7234           # first initial guess
#            inn2 = 0.5234           # second initial guess
#            
#        x1 = math.tan(inn1 + math.pi/2.)    # transformed first initial guess
#        x2 = math.tan(inn2 + math.pi/2.)    # transformed first second guess     
#        
#    # Initial guess
#    xx = np.array([inn1, inn2])
#    aa = a_min/(1. - np.multiply(xx, xx))
#    bbeta = np.asarray([type_factor * 2. * math.asin(np.sqrt((s-c)/2./ai)) for ai in aa])
#    aalpha = np.asarray([2.*math.acos(xi) for xi in xx])
#    
#    # Evaluate the time of flight via Lagrange expression
#    alpha_term = aalpha - np.asarray([math.sin(ai) for ai in aalpha])
#    beta_term = bbeta - np.asarray([math.sin(bi) for bi in bbeta])
#    y_term = alpha_term - beta_term + 2.*math.pi*m
#    
#    y12 = np.multiply(aa, np.multiply(np.sqrt(aa), y_term))
#    
#    # Initial estimate for y
#    if m == 0:
#        y1 = math.log(y12[0]) - logt
#        y2 = math.log(y12[1]) - logt
#    else:
#        y1 = float(y12[0]) - tf
#        y2 = float(y12[0]) - tf
#        
#    # Solve for x
#    # Newton-Raphson iteration
#    err = 1e6
#    iters = 0
#    xnew = 0.
#    while err > tol:
#        
#        # Increment iterations
#        iters += 1
#        
#        # Compute xnew
#        xnew = (x1*y2 - y1*x2)/(y2 - y1)
#        
#        if m == 0:
#            x = math.exp(xnew) - 1.
#        else:
#            x = math.atan(xnew)*2./math.pi
#        
#        a = a_min/(1. - x**2.)
#        
#        # Ellipse
#        if x < 1.:
#            beta = type_factor * 2.*math.asin(np.sqrt((s-c)/2./a))
#            alpha = 2.*math.acos(max(-1., min(1., x)))
#        
#        # Hyperbola
#        else:
#            beta = type_factor * 2.*math.asinh(np.sqrt((s-c)/(-2.*a)))
#            alpha = 2.*math.acosh(x)
#            
#            
#        # Evaluate time of flight via Lagrange expression
#        if a > 0.:
#            tof_new = a*np.sqrt(a)*((alpha - math.sin(alpha) - (beta - math.sin(beta)) + 2.*math.pi*m))
#        else:
#            tof_new = -a*np.sqrt(-a)*((math.sinh(alpha) - alpha) - (math.sinh(beta) - beta))
#            
#        # New value of y
#        if m == 0:
#            ynew = math.log(tof_new) - logt
#        else:
#            ynew = tof_new - tf
#            
#        # Save previous and current values for next iteration
#        x1 = x2
#        x2 = xnew
#        y1 = y2
#        y2 = ynew
#        err = abs(x1 - xnew)
#        
#        # Exit condition
#        if iters > max_iters:
#            exit_flag = -1
#            break
#        
#    # Convert converged value of x
#    if m == 0:
#        x = math.exp(xnew) - 1.
#    else:
#        x = math.atan(xnew)*2./math.pi
#        
#    # The solution has been evaluated in terms of log(x+1) or tan(x*pi/2), we
#    # now need the conic. As for transfer angles near to pi the Lagrange-
#    # coefficients technique goes singular (dg approaches a zero/zero that is
#    # numerically bad) we here use a different technique for those cases. When
#    # the transfer angle is exactly equal to pi, then the ih unit vector is not
#    # determined. The remaining equations, though, are still valid.
#    
#    # Solution for semi-major axis
#    a = a_min/(1. - x**2.)
#    
#    # Calculate psi
#    # Ellipse
#    if x < 1.:
#        beta = type_factor * 2.*math.asin(np.sqrt((s-c)/2./a))
#        alpha = 2.*math.acos(max(-1., min(1., x)))
#        psi = (alpha - beta)/2.
#        eta2 = 2.*a*math.sin(psi)**2./s
#        eta = np.sqrt(eta2)
#        
#    # Hyperbola
#    else:
#        beta = type_factor * 2.*math.asinh(np.sqrt((s-c)/(-2.*a)))
#        alpha = 2.*math.acosh(x)
#        psi = (alpha - beta)/2.
#        eta2 = -2.*a*math.sinh(psi)**2./s
#        eta = np.sqrt(eta2)
#        
#    # Unit of normalized unit vector
#    ih = type_factor*r_cross_hat
#    
#    # Unit vector for rf_vect
#    r0_vect_hat = r0_vect/np.linalg.norm(r0_vect)
#    rf_vect_hat = rf_vect/rf_norm
#    
#    # Cross products    
#    cross1 = np.array([[float(ih[1]*r0_vect_hat[2] - ih[2]*r0_vect_hat[1])],
#                       [float(ih[2]*r0_vect_hat[0] - ih[0]*r0_vect_hat[2])],
#                       [float(ih[0]*r0_vect_hat[1] - ih[1]*r0_vect_hat[0])]])
#    
#    cross2 = np.array([[float(ih[1]*rf_vect_hat[2] - ih[2]*rf_vect_hat[1])],
#                       [float(ih[2]*rf_vect_hat[0] - ih[0]*rf_vect_hat[2])],
#                       [float(ih[0]*rf_vect_hat[1] - ih[1]*rf_vect_hat[0])]])
#    
#    
#    
#    # Radial and tangential components for initial velocity
#    Vr1 = 1./eta/np.sqrt(a_min) * (2.*Lambda*a_min - Lambda - x*eta)
#    Vt1 = np.sqrt(rf_norm/a_min/eta2 * math.sin(dtheta/2.)**2.)
#    
#    # Radial and tangential components for final velocity
#    Vt2 = Vt1/rf_norm
#    Vr2 = (Vt1 - Vt2)/math.tan(dtheta/2.) - Vr1
#    
#    # Velocity vectors
#    v0_vect = (Vr1*r0_vect + Vt1*cross1)*v0
#    vf_vect = (Vr2*rf_vect_hat + Vt2*cross2)*v0
#    
#    # Exit flag - success
#    exit_flag = 1
#
#    # Compute min/max distance to central body
#    extremal_distances = \
#        compute_extremal_dist(r0_vect*r0, rf_vect*r0, v0_vect, vf_vect, dtheta,
#                              a*r0, m, GM, transfer_type)
#       
#    return v0_vect, vf_vect, extremal_distances, exit_flag
#
#
#def robust_lambert(r0_vect, rf_vect, tof, m, GM, transfer_type, branch):
#    '''
#    This function implements a robust method to solve Lambert's Problem based 
#    on work by Lancaster and Blanchard, and Gooding. 
#    
#    Parameters
#    ------
#    r0_vect : 3x1 numpy array
#        position vector at t0 [km]
#    rf_vect : 3x1 numpy array
#        position vector at tf [km]
#    tof : float
#        time of flight [sec]
#    m : int
#        number of complete orbit revolutions
#    GM : float
#        graviational parameter of central body [km^3/s^2]
#        
#    Returns
#    ------
#    v0_vect : 3x1 numpy array
#        velocity vector at t0 [km/s]
#    vf_vect : 3x1 numpy array
#        velocity vector at tf [km/s]
#    extremal_distances : list
#        min and max distance from central body during orbit [km]
#    exit_flag : int
#        +1 : success
#        -1 : no solution exists
#        -2 : fail (could not determine if a solution exists)
#    '''
#    
#    
#    
#    # Initialize and normalize values
#    r0_vect = np.reshape(r0_vect, (3,1))
#    rf_vect = np.reshape(rf_vect, (3,1))
#    tol = 1e-12                            # optimum for numerical noise v.s. actual precision
#    r0 = np.linalg.norm(r0_vect)              
#    rf = np.linalg.norm(rf_vect)              
#    r0_hat = r0_vect/r0                        
#    rf_hat = rf_vect/rf                         
#    cross_r0rf = np.reshape(np.cross(r0_vect.flatten(), rf_vect.flatten()), (3,1))
#    cross_mag = np.linalg.norm(cross_r0rf)
#    cross_r0rf_hat = cross_r0rf/cross_mag
#    
#    # Compute unit vectors in tangential direction
#    t0_hat = np.reshape(np.cross(cross_r0rf_hat.flatten(), r0_hat.flatten()), (3,1))
#    tf_hat = np.reshape(np.cross(cross_r0rf_hat.flatten(), rf_hat.flatten()), (3,1))
#    
#    # Compute turn angle
#    dtheta = math.acos(max(-1, min(1, np.dot(r0_hat.T, rf_hat))))
#    
#    # Check for Type I (short) or II (long) transfer and adjust
#    if transfer_type == 2:
#        dtheta = dtheta - 2.*math.pi
#    
#    # Define constants
#    c = np.sqrt(r0**2. + rf**2. - 2.*r0*rf*math.cos(dtheta))
#    s = (r0 + rf + c)/2.
#    T = np.sqrt(8.*GM/s**3.) * tof
#    q = np.sqrt(r0*rf)/s * math.cos(dtheta/2.)
#    
#    # General formulae for initial values (Gooding)
#    T0, dT0, ddT0, dddT0 = LancasterBlanchard(0., q, m)
#    Td = T0 - T
#    phr = math.fmod(2.*math.atan2(1. - q**2., 2.*q), 2.*math.pi)
#    
#    # Initial output
#    v0_vect = np.reshape([np.nan]*3, (3,1))
#    vf_vect = np.reshape([np.nan]*3, (3,1))
#    extremal_distances = [np.nan]*2
#    
#    # Single revolution case
#    if m == 0:
#        x01 = T0*Td/4./T
#        if Td > 0.:
#            x0 = x01
#        else:
#            x01 = Td/(4. - Td)
#            x02 = -np.sqrt(-Td/(T + T0/2.))
#            W = x01 + 1.7*np.sqrt(2. - phr/math.pi)
#            if W >= 0.:
#                x03 = x01
#            else:
#                x03 = x01 + (-W)**(1./16.)*(x02 - x01)
#            
#            Lambda = 1. + x03*(1. + x01)/2. - 0.03*x03**2.*np.sqrt(1. + x01)
#            x0 = Lambda*x03
#            
#        # This estimate might not give a solution
#        if x0 < -1.:
#            exit_flag = -1
#            return v0_vect, vf_vect, extremal_distances, exit_flag
#        
#    # Multi-revolution case
#    else:
#        
#        # Determine minimum Tp(x)
#        xMpi = 4./(3.*math.pi*(2.*m + 1.))
#        if phr < math.pi:
#            xM0 = xMpi*(phr/math.pi)**(1./8.)
#        elif phr > math.pi:
#            xM0 = xMpi*(2. - (2. - phr/math.pi)**(1./8.))
#        else:
#            xM0 = 0.
#            
#        # Use Halley's method
#        xM = xM0
#        Tp = np.inf
#        iters = 0
#        while abs(Tp) < tol:
#            
#            # Increment counter
#            iters += 1
#            
#            # Compute first three derivatives
#            dum, Tp, Tpp, Tppp = LancasterBlanchard(xM, q, m)
#            
#            # New value of xM
#            xMp = float(xM)
#            xM = xM - 2.*Tp*Tpp / (2.*Tpp**2. - Tp*Tppp)
#            
#            # Escape clause
#            if math.fmod(iters, 7):
#                xM = (xMp + xM)/2.
#            
#            # The method might fail
#            if iters > 25:
#                exit_flag = -2
#                return v0_vect, vf_vect, extremal_distances, exit_flag
#            
#        # xM should be elliptic (-1 < x < 1)
#        # This should be impossible to go wrong
#        if xM < -1. or xM > 1.:
#            exit_flag = -1
#            return v0_vect, vf_vect, extremal_distances, exit_flag
#        
#        # Corresponding time
#        TM, dum1, dum2, dum3 = LancasterBlanchard(xM, q, m)
#        
#        # T should lie above the minimum T
#        if TM > T:
#            exit_flag = -1
#            return v0_vect, vf_vect, extremal_distances, exit_flag
#        
#        
#        # Find two initial values for second solution (again with lambda-type patch)
#        
#        # Initial values
#        TmTM = T - TM
#        T0mTM = T0 - TM
#        dum1, Tp, Tpp, dum2 = LancasterBlanchard(xM, q, m)
#        
#        # First estimate (only if left branch)
#        if branch == 'left':
#            x = np.sqrt(TmTM/(Tpp/2. + TmTM/(1.-xM)**2.))
#            W = xM + x
#            W = 4.*W/(4. + TmTM) + (1. - W)**2.
#            x0 = x*(1. - (1. + m + (dtheta - 1./2.)) / (1. + 0.15*m)*x*(W/2. + 0.03*x*np.sqrt(W))) + xM
#            
#            # First estimate might not be able to yield possible solution
#            if x0 > 1.:
#                exit_flag = -1
#                return v0_vect, vf_vect, extremal_distances, exit_flag
#        
#        # Second estimate
#        else:
#            if Td > 0.:
#                x0 = xM - np.sqrt(TM/(Tpp/2. - TmTM*(Tpp/2./T0mTM - 1./xM**2.)))
#            else:
#                x00 = Td/(4. - Td)
#                W = x00 + 1.7*np.sqrt(2.*(1. - phr))
#                if W >= 0.:
#                    x03 = x00
#                else:
#                    x03 = x00 - np.sqrt((-W)**(1./8.))*(x00 + np.sqrt(-Td/(1.5*T0 - Td)))
#                W = 4./(4. - Td)
#                Lambda = (1. + (1. + m + 0.24*(dtheta - 1./2.)) / (1. + 0.15*m)*x03*(W/2. - 0.03*x03*np.sqrt(W)))
#                x0 = x03*Lambda
#                
#            # Estimate might not give solution
#            if x0 < -1.:
#                exit_flag = -1
#                return v0_vect, vf_vect, extremal_distances, exit_flag
#        
#    
#    
#    
#    # Find root of Lancaster and Blanchard's function
#    # (Halley's method)
#    x = x0
#    Tx = np.inf
#    iters = 0    
#    while abs(Tx) > tol:
#        
#        # Increment counter
#        iters += 1
#        
#        # Compute function value and first two derivatives
#        Tx, Tp, Tpp, dum = LancasterBlanchard(x, q, m)
#        
#        # Find the root of the difference between the function value T_x and
#        # the required time T
#        Tx = Tx - T
#        
#        # New value of x
#        xp = float(x)
#        x = x - 2.*Tx*Tp/(2.*Tp**2. - Tx*Tpp)
#        
#        # Escape clause
#        if math.fmod(iters, 7):
#            x = (xp + x)/2.
#            
#        # Halley's method might fail
#        if iters > 25:
#            exit_flag = -2
#            return v0_vect, vf_vect, extremal_distances, exit_flag
#            
#    
#    # Calculate terminal velocities
#    gamma = np.sqrt(GM*s/2.)
#    if c == 0.:
#        sigma = 1.
#        rho = 0.
#        z = abs(x)
#    else:
#        sigma = 2.*np.sqrt(r0*rf/(c**2.)) * math.sin(dtheta/2.)
#        rho = (r0 - rf)/c
#        z = np.sqrt(1. + q**2.*(x**2. - 1.))
#        
#    # Radial components
#    Vr0 = gamma*((q*z - x) - rho*(q*z + x)) / r0    
#    Vrf = -gamma*((q*z - x) + rho*(q*z + x)) / rf
#    v0_radial = Vr0*r0_hat
#    vf_radial = Vrf*rf_hat
#    
#    # Tangential components
#    Vt0 = sigma * gamma * (z + q*x) / r0
#    Vtf = sigma * gamma * (z + q*x) / rf
#    v0_tan = Vt0*t0_hat
#    vf_tan = Vtf*tf_hat
#    
#    # Cartesian velocity
#    v0_vect = v0_radial + v0_tan
#    vf_vect = vf_radial + vf_tan
#    
#    # Final outputs
#    a = s/2./(1. - x**2.)
#    extremal_distances = \
#        compute_extremal_dist(r0_vect, rf_vect, v0_vect, vf_vect, dtheta, a, m,
#                              GM, transfer_type)
#    
#    exit_flag = 1
#    
#    
#    return v0_vect, vf_vect, extremal_distances, exit_flag
#
#
#def LancasterBlanchard(x, q, m):
#    
#    # Verify input
#    if x < -1.:
#        x = abs(x) - 2.
#    elif x == -1.:
#        x += np.finfo(float).eps
#        
#    # Compute parameter E
#    E = x*x - 1.
#    
#    # Compute T(x) and derivatives
#    if x == 1:
#        
#        # Parabolic, solutions known exactly
#        T = (4./3.)*(1. - q**3.)
#        Tp = (4./5.)*(q**5. - 1.)
#        Tpp = Tp + (120./70.)*(1. - q**7.)
#        Tppp = 3.*(Tpp - Tp) + (2400./1080.)*(q**9. - 1.)
#        
#    elif abs(x-1) < 1e-2:
#        
#        # Near-parabolic, compute with series
#        sig1, dsigdx1, d2sigdx21, d3sigdx31 = compute_sigmax(-E)
#        sig2, dsigdx2, d2sigdx22, d3sigdx32 = compute_sigmax(-E*q*q)
#        
#        T = sig1 - q**3.*sig2
#        Tp = 2.*x*(q**5.*dsigdx2 - dsigdx1)
#        Tpp = Tp/x + 4.*x**2.*(d2sigdx21 - q**7.*d2sigdx22)
#        Tppp = 3.*(Tpp-Tp/x)/x + 8.*x*x*(q**9.*d3sigdx32 - d3sigdx31)
#        
#    else:
#        
#        # All other cases
#        y = np.sqrt(abs(E))
#        z = np.sqrt(1. + q**2.*E)
#        f = y*(z - q*x)
#        g = x*z - q*E
#        
#        if E < 0.:
#            d = math.atan2(f, g) + math.pi*m
#        elif E == 0.:
#            d = 0.
#        else:
#            d = math.log(max(0., (f+g)))
#            
#
#        T = 2.*(x - q*z - d/y)/E
#        Tp = (4. - 4.*q**3.*x/z - 3.*x*T)/E
#        Tpp = (-4.*q**3./z * (1. - q**2.*x**2./z**2.) - 3.*T - 3.*x*Tp)/E
#        Tppp = (4.*q**3./z**2.*((1. - q**2.*x**2./z**2.) + 2.*q**2.*x/z**2.*(z - x)) - 8.*Tp - 7.*x*Tpp)/E
#
#    return T, Tp, Tpp, Tppp
#    
#
#def compute_sigmax(y):
#    '''
#    
#    '''
#    
#    # Twenty-five factors more than enough for 16-digit precision
#    an = [4.000000000000000e-001, 2.142857142857143e-001, 4.629629629629630e-002,
#          6.628787878787879e-003, 7.211538461538461e-004, 6.365740740740740e-005,
#          4.741479925303455e-006, 3.059406328320802e-007, 1.742836409255060e-008,
#          8.892477331109578e-010, 4.110111531986532e-011, 1.736709384841458e-012,
#          6.759767240041426e-014, 2.439123386614026e-015, 8.203411614538007e-017,
#          2.583771576869575e-018, 7.652331327976716e-020, 2.138860629743989e-021,
#          5.659959451165552e-023, 1.422104833817366e-024, 3.401398483272306e-026,
#          7.762544304774155e-028, 1.693916882090479e-029, 3.541295006766860e-031,
#          7.105336187804402e-033]
#    
#
#    # powers of y
#    powers = [y**exponent for exponent in range(1, 26)]
#    
#    # Vectorize
#    powers = np.reshape(powers, (25, 1))
#    an = np.reshape(an, (25, 1))
#    deriv_factors = np.reshape(range(1,26), (25,1))
#    deriv2_factors = np.reshape(range(0,25), (25,1))
#    deriv3_factors = np.reshape(range(-1, 24), (25,1))
#
#    # sigma itself
#    sig = float(4./3. + np.dot(powers.T, an))
#
#    # dsigma / dx (derivative)
#    first_der = np.reshape(np.insert(powers, 0, 1), (26, 1))
#    dsigdx = float(np.dot(np.multiply(deriv_factors, first_der[0:25]).T, an))
#
#    # d2sigma / dx2 (second derivative)
#    second_der = np.reshape(np.insert(powers, 0, np.array([1./y, 1.])), (27, 1))
#    d2sigdx2 = float(np.dot(np.multiply(np.multiply(deriv_factors, deriv2_factors), second_der[0:25]).T, an))
#    
#    # d3sigma / dx3 (third derivative)
#    third_der = np.reshape(np.insert(powers, 0, np.array([1./y/y, 1./y, 1.])), (28, 1))
#    d3sigdx3 = float(np.dot(np.multiply(np.multiply(np.multiply(deriv_factors, deriv2_factors), deriv3_factors), third_der[0:25]).T, an))
#
#    
#    return  sig, dsigdx, d2sigdx2, d3sigdx3
    

